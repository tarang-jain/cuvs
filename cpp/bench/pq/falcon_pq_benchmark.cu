/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/preprocessing/quantize/pq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/cuda_stream_pool.hpp>

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr int64_t kExpectedRows        = 10'000'000;
constexpr int64_t kExpectedDim         = 1024;
constexpr int64_t kPqDim               = 512;
constexpr int64_t kPqLen               = 2;
constexpr int64_t kCenters             = 256;
constexpr uint32_t kMaxPointsPerCenter = 39'063;
constexpr int kMaxIter                 = 300;
constexpr double kTolerance            = 1e-4;
constexpr uint64_t kSeed               = 42;

struct options {
  std::string dataset;
  std::string mode{"serial_api"};
  std::string output_prefix{"falcon_pq"};
  int device{0};
  int concurrency{1};
  int64_t row_cap{0};
  int memory_poll_ms{50};
  bool skip_inertia{false};
};

struct mapped_fbin {
  explicit mapped_fbin(const std::string& path)
  {
    fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { throw std::runtime_error("cannot open dataset: " + path); }
    struct stat st{};
    if (fstat(fd, &st) != 0) { throw std::runtime_error("cannot stat dataset: " + path); }
    bytes   = static_cast<size_t>(st.st_size);
    mapping = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
      mapping = nullptr;
      throw std::runtime_error("cannot mmap dataset: " + path);
    }
    if (bytes < 2 * sizeof(uint32_t)) {
      throw std::runtime_error("dataset is smaller than header");
    }
    std::memcpy(&rows, mapping, sizeof(rows));
    std::memcpy(&dim, static_cast<const char*>(mapping) + sizeof(rows), sizeof(dim));
    const auto expected = 2 * sizeof(uint32_t) + static_cast<uint64_t>(rows) * dim * sizeof(float);
    if (bytes != expected) {
      throw std::runtime_error("dataset size does not match its fbin header: expected " +
                               std::to_string(expected) + ", got " + std::to_string(bytes));
    }
  }

  mapped_fbin(const mapped_fbin&)            = delete;
  mapped_fbin& operator=(const mapped_fbin&) = delete;
  ~mapped_fbin()
  {
    if (mapping != nullptr) { munmap(mapping, bytes); }
    if (fd >= 0) { close(fd); }
  }

  [[nodiscard]] const float* data() const
  {
    return reinterpret_cast<const float*>(static_cast<const char*>(mapping) + 2 * sizeof(uint32_t));
  }

  int fd{-1};
  void* mapping{nullptr};
  size_t bytes{0};
  uint32_t rows{0};
  uint32_t dim{0};
};

class nvtx_range {
 public:
  explicit nvtx_range(const std::string& name) { nvtxRangePushA(name.c_str()); }
  ~nvtx_range() { nvtxRangePop(); }
};

class memory_monitor {
 public:
  memory_monitor(int device, int interval_ms) : device_(device), interval_ms_(interval_ms)
  {
    sample();
    if (interval_ms_ > 0) {
      running_.store(true);
      thread_ = std::thread([this] {
        RAFT_CUDA_TRY(cudaSetDevice(device_));
        while (running_.load()) {
          sample();
          std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
        }
        sample();
      });
    }
  }

  ~memory_monitor() { stop(); }

  void stop()
  {
    if (thread_.joinable()) {
      running_.store(false);
      thread_.join();
    } else {
      sample();
    }
  }

  [[nodiscard]] uint64_t peak_used() const { return peak_used_.load(); }
  [[nodiscard]] uint64_t start_used() const { return start_used_; }

 private:
  void sample()
  {
    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    RAFT_CUDA_TRY(cudaMemGetInfo(&free_bytes, &total_bytes));
    const auto used = static_cast<uint64_t>(total_bytes - free_bytes);
    if (start_used_ == 0) { start_used_ = used; }
    auto old = peak_used_.load();
    while (used > old && !peak_used_.compare_exchange_weak(old, used)) {}
  }

  int device_;
  int interval_ms_;
  uint64_t start_used_{0};
  std::atomic<uint64_t> peak_used_{0};
  std::atomic<bool> running_{false};
  std::thread thread_;
};

struct subspace_result {
  double inertia{std::numeric_limits<double>::quiet_NaN()};
  int64_t iterations{-1};
  std::optional<bool> converged;
};

struct run_result {
  double training_seconds{0.0};
  uint64_t gpu_start_bytes{0};
  uint64_t gpu_peak_bytes{0};
  uint64_t host_peak_rss_bytes{0};
  std::vector<float> centroids;
  std::vector<subspace_result> subspaces;
  std::string convergence_status;
};

void usage(const char* argv0)
{
  std::cout
    << "Usage: " << argv0
    << " --dataset=PATH --mode=serial_api|parallel_prototype [options]\n"
       "  --output-prefix=PATH   Output path without extension\n"
       "  --device=N             CUDA device index (default 0)\n"
       "  --concurrency=N        Concurrent prototype codebooks (1,2,4,8,16)\n"
       "  --row-cap=N            Use the first N rows for a smoke/profile run; 0 means all\n"
       "  --memory-poll-ms=N     CUDA memory sampling period; 0 samples endpoints only\n"
       "  --skip-inertia         Skip serial post-fit inertia evaluation (profiling only)\n";
}

options parse_options(int argc, char** argv)
{
  options result;
  for (int i = 1; i < argc; ++i) {
    const std::string arg{argv[i]};
    if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    }
    if (arg == "--skip-inertia") {
      result.skip_inertia = true;
      continue;
    }
    const auto equal = arg.find('=');
    if (equal == std::string::npos) { throw std::runtime_error("expected --name=value: " + arg); }
    const auto key   = arg.substr(0, equal);
    const auto value = arg.substr(equal + 1);
    if (key == "--dataset") {
      result.dataset = value;
    } else if (key == "--mode") {
      result.mode = value;
    } else if (key == "--output-prefix") {
      result.output_prefix = value;
    } else if (key == "--device") {
      result.device = std::stoi(value);
    } else if (key == "--concurrency") {
      result.concurrency = std::stoi(value);
    } else if (key == "--row-cap") {
      result.row_cap = std::stoll(value);
    } else if (key == "--memory-poll-ms") {
      result.memory_poll_ms = std::stoi(value);
    } else {
      throw std::runtime_error("unknown option: " + key);
    }
  }
  if (result.dataset.empty()) { throw std::runtime_error("--dataset is required"); }
  if (result.mode != "serial_api" && result.mode != "parallel_prototype") {
    throw std::runtime_error("--mode must be serial_api or parallel_prototype");
  }
  if (result.concurrency != 1 && result.concurrency != 2 && result.concurrency != 4 &&
      result.concurrency != 8 && result.concurrency != 16) {
    throw std::runtime_error("--concurrency must be one of 1, 2, 4, 8, or 16");
  }
  if (result.row_cap < 0 || result.memory_poll_ms < 0) {
    throw std::runtime_error("row cap and memory poll interval must be non-negative");
  }
  return result;
}

uint64_t host_peak_rss_bytes()
{
  std::ifstream status{"/proc/self/status"};
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("VmHWM:", 0) == 0) {
      std::istringstream fields{line.substr(6)};
      uint64_t kib = 0;
      fields >> kib;
      return kib * 1024;
    }
  }
  return 0;
}

__global__ void count_nonfinite(const float* data, uint64_t size, unsigned long long* count)
{
  uint64_t i               = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride    = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  unsigned long long local = 0;
  for (; i < size; i += stride) {
    if (!isfinite(data[i])) { ++local; }
  }
  if (local != 0) { atomicAdd(count, local); }
}

void validate_device_data(const float* data, uint64_t elements, cudaStream_t stream)
{
  unsigned long long* device_count = nullptr;
  unsigned long long host_count    = 0;
  RAFT_CUDA_TRY(
    cudaMallocAsync(reinterpret_cast<void**>(&device_count), sizeof(*device_count), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(device_count, 0, sizeof(*device_count), stream));
  count_nonfinite<<<4096, 256, 0, stream>>>(data, elements, device_count);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(&host_count, device_count, sizeof(host_count), cudaMemcpyDeviceToHost, stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  RAFT_CUDA_TRY(cudaFreeAsync(device_count, stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  if (host_count != 0) {
    throw std::runtime_error("dataset contains " + std::to_string(host_count) +
                             " non-finite values");
  }
}

cuvs::cluster::kmeans::params kmeans_params()
{
  cuvs::cluster::kmeans::params params;
  params.n_clusters          = kCenters;
  params.init                = cuvs::cluster::kmeans::params::InitMethod::KMeansPlusPlus;
  params.max_iter            = kMaxIter;
  params.tol                 = kTolerance;
  params.rng_state           = raft::random::RngState{kSeed};
  params.n_init              = 1;
  params.oversampling_factor = 2.0;
  params.metric              = cuvs::distance::DistanceType::L2Expanded;
  return params;
}

cuvs::preprocessing::quantize::pq::params pq_params()
{
  cuvs::preprocessing::quantize::pq::params params;
  params.pq_bits                      = 8;
  params.pq_dim                       = kPqDim;
  params.use_subspaces                = true;
  params.use_vq                       = false;
  params.vq_n_centers                 = 0;
  params.kmeans_params                = kmeans_params();
  params.max_train_points_per_pq_code = kMaxPointsPerCenter;
  return params;
}

void copy_subspace(const raft::resources& resource,
                   const float* dataset,
                   int64_t rows,
                   int64_t subspace,
                   float* output)
{
  raft::copy_matrix(output,
                    kPqLen,
                    dataset + subspace * kPqLen,
                    kExpectedDim,
                    kPqLen,
                    rows,
                    raft::resource::get_cuda_stream(resource));
}

std::vector<subspace_result> evaluate_inertia(const raft::resources& resource,
                                              const float* dataset,
                                              int64_t rows,
                                              const float* centers)
{
  std::vector<subspace_result> result(kPqDim);
  auto subspace_data = raft::make_device_matrix<float, int64_t>(resource, rows, kPqLen);
  for (int64_t subspace = 0; subspace < kPqDim; ++subspace) {
    copy_subspace(resource, dataset, rows, subspace, subspace_data.data_handle());
    const auto center_view = raft::make_device_matrix_view<const float, int64_t>(
      centers + subspace * kCenters * kPqLen, kCenters, kPqLen);
    float inertia = 0.0f;
    cuvs::cluster::kmeans::cluster_cost(resource,
                                        raft::make_const_mdspan(subspace_data.view()),
                                        center_view,
                                        raft::make_host_scalar_view(&inertia));
    raft::resource::sync_stream(resource);
    result[subspace].inertia    = inertia;
    result[subspace].iterations = -1;
    result[subspace].converged  = std::nullopt;
  }
  return result;
}

run_result run_serial_api(const raft::resources& resource,
                          raft::device_matrix_view<const float, int64_t> dataset,
                          int device,
                          int memory_poll_ms,
                          bool skip_inertia)
{
  run_result result;
  memory_monitor monitor{device, memory_poll_ms};
  const auto begin = std::chrono::steady_clock::now();
  cudaProfilerStart();
  {
    nvtx_range range{"falcon_pq/serial_api/train"};
    auto quantizer = cuvs::preprocessing::quantize::pq::build(resource, pq_params(), dataset);
    raft::resource::sync_stream(resource);
    cudaProfilerStop();
    result.training_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    monitor.stop();
    result.gpu_start_bytes = monitor.start_used();
    result.gpu_peak_bytes  = monitor.peak_used();

    const auto& codebook = quantizer.vpq_codebooks.pq_code_book;
    if (codebook.extent(0) != kPqDim * kCenters || codebook.extent(1) != kPqLen) {
      throw std::runtime_error("serial API returned an unexpected codebook shape");
    }
    result.centroids.resize(kPqDim * kCenters * kPqLen);
    RAFT_CUDA_TRY(cudaMemcpy(result.centroids.data(),
                             codebook.data_handle(),
                             result.centroids.size() * sizeof(float),
                             cudaMemcpyDeviceToHost));
    if (!skip_inertia) {
      nvtx_range eval_range{"falcon_pq/serial_api/inertia_validation"};
      result.subspaces = evaluate_inertia(
        resource, dataset.data_handle(), dataset.extent(0), codebook.data_handle());
    }
  }
  result.convergence_status =
    skip_inertia ? "not_evaluated_for_profile" : "not_exposed_by_pq_build";
  result.host_peak_rss_bytes = host_peak_rss_bytes();
  return result;
}

run_result run_parallel_prototype(const raft::resources& base_resource,
                                  raft::device_matrix_view<const float, int64_t> dataset,
                                  int device,
                                  int concurrency,
                                  int memory_poll_ms)
{
  run_result result;
  result.subspaces.resize(kPqDim);
  auto centers = raft::make_device_matrix<float, int64_t>(base_resource, kPqDim * kCenters, kPqLen);
  rmm::cuda_stream_pool streams(static_cast<size_t>(concurrency));
  std::vector<raft::resources> resources;
  resources.reserve(concurrency);
  for (int i = 0; i < concurrency; ++i) {
    resources.push_back(base_resource);
    raft::resource::set_cuda_stream(resources.back(), streams.get_stream(i));
  }

  memory_monitor monitor{device, memory_poll_ms};
  const auto begin = std::chrono::steady_clock::now();
  cudaProfilerStart();
  {
    nvtx_range range{"falcon_pq/parallel_prototype/train"};
    for (int64_t wave = 0; wave < kPqDim; wave += concurrency) {
      const auto wave_size = std::min<int64_t>(concurrency, kPqDim - wave);
      std::vector<std::future<void>> futures;
      futures.reserve(wave_size);
      for (int64_t slot = 0; slot < wave_size; ++slot) {
        const int64_t subspace = wave + slot;
        futures.emplace_back(std::async(std::launch::async, [&, slot, subspace] {
          RAFT_CUDA_TRY(cudaSetDevice(device));
          const auto name = "falcon_pq/parallel_prototype/subspace_" + std::to_string(subspace);
          nvtx_range subspace_range{name};
          auto& resource = resources[slot];
          auto subspace_data =
            raft::make_device_matrix<float, int64_t>(resource, dataset.extent(0), kPqLen);
          copy_subspace(resource,
                        dataset.data_handle(),
                        dataset.extent(0),
                        subspace,
                        subspace_data.data_handle());
          const auto center_view = raft::make_device_matrix_view<float, int64_t>(
            centers.data_handle() + subspace * kCenters * kPqLen, kCenters, kPqLen);
          float inertia     = 0.0f;
          int64_t n_iter    = 0;
          const auto params = kmeans_params();
          cuvs::cluster::kmeans::fit(resource,
                                     params,
                                     raft::make_const_mdspan(subspace_data.view()),
                                     std::nullopt,
                                     center_view,
                                     raft::make_host_scalar_view(&inertia),
                                     raft::make_host_scalar_view(&n_iter));
          raft::resource::sync_stream(resource);
          result.subspaces[subspace].inertia    = inertia;
          result.subspaces[subspace].iterations = n_iter;
          result.subspaces[subspace].converged  = n_iter < kMaxIter;
        }));
      }
      for (auto& future : futures) {
        future.get();
      }
    }
    for (auto& resource : resources) {
      raft::resource::sync_stream(resource);
    }
  }
  cudaProfilerStop();
  result.training_seconds =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  monitor.stop();
  result.gpu_start_bytes = monitor.start_used();
  result.gpu_peak_bytes  = monitor.peak_used();
  result.centroids.resize(kPqDim * kCenters * kPqLen);
  RAFT_CUDA_TRY(cudaMemcpy(result.centroids.data(),
                           centers.data_handle(),
                           result.centroids.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));
  const bool all_converged =
    std::all_of(result.subspaces.begin(), result.subspaces.end(), [](const auto& subspace) {
      return subspace.converged.value_or(false);
    });
  result.convergence_status =
    all_converged ? "all_converged_before_max_iter" : "one_or_more_reached_max_iter";
  result.host_peak_rss_bytes = host_peak_rss_bytes();
  return result;
}

std::string json_escape(const std::string& value)
{
  std::ostringstream out;
  for (const auto c : value) {
    switch (c) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << c;
    }
  }
  return out.str();
}

std::string gpu_uuid(const cudaUUID_t& uuid)
{
  std::ostringstream out;
  out << "GPU-" << std::hex << std::setfill('0');
  for (int i = 0; i < 16; ++i) {
    out << std::setw(2) << static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[i]));
    if (i == 3 || i == 5 || i == 7 || i == 9) { out << '-'; }
  }
  return out.str();
}

void validate_result(const run_result& result, bool inertia_required)
{
  if (result.centroids.size() != kPqDim * kCenters * kPqLen) {
    throw std::runtime_error("codebook shape validation failed");
  }
  if (!std::all_of(result.centroids.begin(), result.centroids.end(), [](float value) {
        return std::isfinite(value);
      })) {
    throw std::runtime_error("codebook contains non-finite centroids");
  }
  if (inertia_required) {
    if (result.subspaces.size() != kPqDim) {
      throw std::runtime_error("missing per-subspace results");
    }
    for (const auto& subspace : result.subspaces) {
      if (!std::isfinite(subspace.inertia) || subspace.inertia < 0.0) {
        throw std::runtime_error("invalid per-subspace inertia");
      }
    }
  }
}

void write_outputs(const options& opts,
                   const mapped_fbin& source,
                   int64_t rows_used,
                   const cudaDeviceProp& device_prop,
                   int driver_version,
                   int runtime_version,
                   const run_result& result)
{
  const std::filesystem::path prefix{opts.output_prefix};
  if (!prefix.parent_path().empty()) { std::filesystem::create_directories(prefix.parent_path()); }
  const auto total_inertia = std::accumulate(
    result.subspaces.begin(), result.subspaces.end(), 0.0, [](double sum, const auto& item) {
      return sum + (std::isfinite(item.inertia) ? item.inertia : 0.0);
    });

  std::ofstream json{opts.output_prefix + ".json"};
  if (!json) { throw std::runtime_error("cannot create JSON result"); }
  json << std::setprecision(17) << "{\n"
       << "  \"mode\": \"" << opts.mode << "\",\n"
       << "  \"concurrency\": " << opts.concurrency << ",\n"
       << "  \"dataset_path\": \"" << json_escape(opts.dataset) << "\",\n"
       << "  \"dataset_header_rows\": " << source.rows << ",\n"
       << "  \"dataset_header_dim\": " << source.dim << ",\n"
       << "  \"dataset_file_bytes\": " << source.bytes << ",\n"
       << "  \"rows_used\": " << rows_used << ",\n"
       << "  \"complete_row_usage\": " << (rows_used == source.rows ? "true" : "false") << ",\n"
       << "  \"codebook_shape\": [" << kPqDim << ", " << kCenters << ", " << kPqLen << "],\n"
       << "  \"dataset_finite\": true,\n"
       << "  \"centroids_finite\": true,\n"
       << "  \"output_validated\": true,\n"
       << "  \"pq_bits\": 8,\n"
       << "  \"max_train_points_per_pq_code\": " << kMaxPointsPerCenter << ",\n"
       << "  \"train_point_limit\": " << kMaxPointsPerCenter * kCenters << ",\n"
       << "  \"kmeans_init\": \"scalable_kmeans_parallel\",\n"
       << "  \"kmeans_oversampling_factor\": 2.0,\n"
       << "  \"seed\": " << kSeed << ",\n"
       << "  \"n_init\": 1,\n"
       << "  \"max_iter\": " << kMaxIter << ",\n"
       << "  \"tolerance\": " << kTolerance << ",\n"
       << "  \"training_seconds\": " << result.training_seconds << ",\n"
       << "  \"total_inertia\": " << total_inertia << ",\n"
       << "  \"convergence_status\": \"" << result.convergence_status << "\",\n"
       << "  \"gpu_start_bytes\": " << result.gpu_start_bytes << ",\n"
       << "  \"gpu_peak_bytes\": " << result.gpu_peak_bytes << ",\n"
       << "  \"gpu_training_peak_delta_bytes\": "
       << (result.gpu_peak_bytes - std::min(result.gpu_peak_bytes, result.gpu_start_bytes)) << ",\n"
       << "  \"host_peak_rss_bytes\": " << result.host_peak_rss_bytes << ",\n"
       << "  \"cuda_device\": " << opts.device << ",\n"
       << "  \"gpu_name\": \"" << json_escape(device_prop.name) << "\",\n"
       << "  \"gpu_uuid\": \"" << gpu_uuid(device_prop.uuid) << "\",\n"
       << "  \"driver_version\": " << driver_version << ",\n"
       << "  \"cuda_runtime_version\": " << runtime_version << ",\n"
       << "  \"subspaces\": [\n";
  for (size_t i = 0; i < result.subspaces.size(); ++i) {
    const auto& item = result.subspaces[i];
    json << "    {\"index\": " << i << ", \"inertia\": " << item.inertia << ", \"iterations\": ";
    if (item.iterations < 0) {
      json << "null";
    } else {
      json << item.iterations;
    }
    json << ", \"converged\": ";
    if (!item.converged.has_value()) {
      json << "null";
    } else {
      json << (*item.converged ? "true" : "false");
    }
    json << "}" << (i + 1 == result.subspaces.size() ? "\n" : ",\n");
  }
  json << "  ]\n}\n";
  json.close();

  std::ofstream csv{opts.output_prefix + ".subspaces.csv"};
  if (!csv) { throw std::runtime_error("cannot create CSV result"); }
  csv << "subspace,inertia,iterations,converged\n" << std::setprecision(17);
  for (size_t i = 0; i < result.subspaces.size(); ++i) {
    const auto& item = result.subspaces[i];
    csv << i << ',' << item.inertia << ',';
    if (item.iterations >= 0) { csv << item.iterations; }
    csv << ',';
    if (item.converged.has_value()) { csv << (*item.converged ? "true" : "false"); }
    csv << '\n';
  }
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    const auto opts = parse_options(argc, argv);
    RAFT_CUDA_TRY(cudaSetDevice(opts.device));
    mapped_fbin source{opts.dataset};
    if (source.rows != kExpectedRows || source.dim != kExpectedDim) {
      throw std::runtime_error("expected Falcon header 10000000 x 1024, got " +
                               std::to_string(source.rows) + " x " + std::to_string(source.dim));
    }
    const int64_t rows_used =
      opts.row_cap == 0 ? source.rows : std::min<int64_t>(opts.row_cap, source.rows);
    if (rows_used < kCenters) { throw std::runtime_error("row cap must be at least 256"); }
    if (static_cast<uint64_t>(kMaxPointsPerCenter) * kCenters < static_cast<uint64_t>(rows_used)) {
      throw std::runtime_error(
        "configured training-point limit does not include all selected rows");
    }

    raft::resources resource;
    const auto stream = raft::resource::get_cuda_stream(resource);
    auto dataset      = raft::make_device_matrix<float, int64_t>(resource, rows_used, kExpectedDim);
    constexpr uint64_t chunk_bytes = uint64_t{256} << 20;
    const uint64_t data_bytes = static_cast<uint64_t>(rows_used) * kExpectedDim * sizeof(float);
    for (uint64_t offset = 0; offset < data_bytes; offset += chunk_bytes) {
      const auto bytes = std::min<uint64_t>(chunk_bytes, data_bytes - offset);
      RAFT_CUDA_TRY(cudaMemcpyAsync(reinterpret_cast<char*>(dataset.data_handle()) + offset,
                                    reinterpret_cast<const char*>(source.data()) + offset,
                                    bytes,
                                    cudaMemcpyHostToDevice,
                                    stream));
    }
    raft::resource::sync_stream(resource);
    validate_device_data(
      dataset.data_handle(), static_cast<uint64_t>(rows_used) * kExpectedDim, stream);

    cudaDeviceProp device_prop{};
    RAFT_CUDA_TRY(cudaGetDeviceProperties(&device_prop, opts.device));
    int driver_version  = 0;
    int runtime_version = 0;
    RAFT_CUDA_TRY(cudaDriverGetVersion(&driver_version));
    RAFT_CUDA_TRY(cudaRuntimeGetVersion(&runtime_version));

    const auto dataset_view = raft::make_device_matrix_view<const float, int64_t>(
      dataset.data_handle(), rows_used, kExpectedDim);
    run_result result;
    if (opts.mode == "serial_api") {
      result =
        run_serial_api(resource, dataset_view, opts.device, opts.memory_poll_ms, opts.skip_inertia);
    } else {
      result = run_parallel_prototype(
        resource, dataset_view, opts.device, opts.concurrency, opts.memory_poll_ms);
    }
    validate_result(result, !opts.skip_inertia);
    write_outputs(opts, source, rows_used, device_prop, driver_version, runtime_version, result);
    std::cout << std::setprecision(6) << "mode=" << opts.mode << " concurrency=" << opts.concurrency
              << " rows=" << rows_used << " seconds=" << result.training_seconds
              << " convergence=" << result.convergence_status << " output=" << opts.output_prefix
              << ".json\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FALCON_PQ_BENCH failed: " << error.what() << '\n';
    return 1;
  }
}
