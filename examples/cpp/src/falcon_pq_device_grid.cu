/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

#include "../../../cpp/src/cluster/detail/kmeans.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

struct options {
  std::string base = "/datasets/falcon_1024_10M/base_falcon_1024_10M.fbin";
  std::string queries = "/datasets/falcon_1024_10M/query_falcon_1024_10K.fbin";
  std::string groundtruth = "/datasets/falcon_1024_10M/falcon_1024_10M_gt100_10K.fbin";
  std::string method = "scalable";
  std::string step8 = "kmeans++";
  int pq_len = 8;
  int64_t max_train_per_code = 1024;
  int64_t train_rows = 0;
  int64_t init_size = 1024;
  int64_t row_cap = 0;
  int64_t chunk_rows = 32768;
  int max_iter = 300;
  double tol = 1e-4;
  uint64_t seed = 42;
  int topk = 100;
};

double elapsed_ms(clock_type::time_point start)
{
  return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

void fit_kmeans(raft::resources const& resources,
                const cuvs::cluster::kmeans::params& params,
                raft::device_matrix_view<const float, int64_t> data,
                raft::device_matrix_view<float, int64_t> centers,
                float& inertia,
                int64_t& iterations)
{
  std::optional<raft::device_vector_view<const float, int64_t>> weights = std::nullopt;
  cuvs::cluster::kmeans::detail::kmeans_fit<float, int64_t>(
    resources,
    params,
    data,
    weights,
    centers,
    raft::make_host_scalar_view(&inertia),
    raft::make_host_scalar_view(&iterations),
    std::nullopt);
}

options parse_args(int argc, char** argv)
{
  options out;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (++i >= argc) { throw std::runtime_error("missing value after " + arg); }
      return argv[i];
    };
    if (arg == "--base") out.base = next();
    else if (arg == "--queries") out.queries = next();
    else if (arg == "--groundtruth") out.groundtruth = next();
    else if (arg == "--method") out.method = next();
    else if (arg == "--step8") out.step8 = next();
    else if (arg == "--pq-len") out.pq_len = std::stoi(next());
    else if (arg == "--max-train-per-code") out.max_train_per_code = std::stoll(next());
    else if (arg == "--train-rows") out.train_rows = std::stoll(next());
    else if (arg == "--init-size") out.init_size = std::stoll(next());
    else if (arg == "--row-cap") out.row_cap = std::stoll(next());
    else if (arg == "--chunk-rows") out.chunk_rows = std::stoll(next());
    else if (arg == "--max-iter") out.max_iter = std::stoi(next());
    else if (arg == "--tol") out.tol = std::stod(next());
    else if (arg == "--seed") out.seed = std::stoull(next());
    else if (arg == "--topk") out.topk = std::stoi(next());
    else if (arg == "--help") {
      std::cout
        << "Falcon device PQ benchmark (one grid cell)\n"
        << "  --method classic|scalable|random\n"
        << "  --step8 random|kmeans++\n"
        << "  --pq-len 1|2|4|8\n"
        << "  --max-train-per-code N, or --train-rows N\n"
        << "  --init-size N (ignored for random)\n"
        << "  --row-cap N (smoke tests only)\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (out.method != "classic" && out.method != "scalable" && out.method != "random") {
    throw std::runtime_error("--method must be classic, scalable, or random");
  }
  if (out.step8 != "random" && out.step8 != "kmeans++") {
    throw std::runtime_error("--step8 must be random or kmeans++");
  }
  if (out.pq_len != 1 && out.pq_len != 2 && out.pq_len != 4 && out.pq_len != 8) {
    throw std::runtime_error("--pq-len must be 1, 2, 4, or 8");
  }
  if (out.topk != 100) { throw std::runtime_error("Falcon evaluation requires --topk 100"); }
  return out;
}

template <typename T>
std::tuple<int64_t, int64_t, std::vector<T>> load_bin(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { throw std::runtime_error("failed to open " + path); }
  int32_t rows = 0;
  int32_t cols = 0;
  in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
  if (!in || rows <= 0 || cols <= 0) { throw std::runtime_error("invalid fbin header: " + path); }
  std::vector<T> data(static_cast<size_t>(rows) * cols);
  in.read(reinterpret_cast<char*>(data.data()),
          static_cast<std::streamsize>(data.size() * sizeof(T)));
  if (!in) { throw std::runtime_error("short read: " + path); }
  return {rows, cols, std::move(data)};
}

struct memory_observer {
  size_t total = 0;
  size_t peak_used = 0;

  memory_observer() { sample(); }

  void sample()
  {
    size_t free = 0;
    size_t current_total = 0;
    if (cudaMemGetInfo(&free, &current_total) == cudaSuccess) {
      total = current_total;
      peak_used = std::max(peak_used, current_total - free);
    }
  }
};

struct device_chunk {
  int64_t offset;
  int64_t rows;
  std::unique_ptr<rmm::device_uvector<float>> data;
};

struct chunked_dataset {
  int64_t rows = 0;
  int64_t dim = 0;
  int64_t chunk_rows = 0;
  std::vector<device_chunk> chunks;
};

chunked_dataset load_chunked_dataset(const std::string& path,
                                     int64_t requested_chunk_rows,
                                     int64_t row_cap,
                                     cudaStream_t stream,
                                     memory_observer& memory)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { throw std::runtime_error("failed to open " + path); }
  int32_t file_rows = 0;
  int32_t dim = 0;
  in.read(reinterpret_cast<char*>(&file_rows), sizeof(file_rows));
  in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  if (!in || file_rows <= 0 || dim <= 0) { throw std::runtime_error("invalid base header"); }

  chunked_dataset result;
  result.rows = row_cap > 0 ? std::min<int64_t>(row_cap, file_rows) : file_rows;
  result.dim = dim;
  result.chunk_rows = std::min(requested_chunk_rows, result.rows);

  size_t free = 0;
  size_t total = 0;
  cudaMemGetInfo(&free, &total);
  const uint64_t base_bytes = static_cast<uint64_t>(result.rows) * result.dim * sizeof(float);
  const uint64_t reserve = 768ull * 1024ull * 1024ull;
  if (base_bytes + reserve > free) {
    throw std::runtime_error("insufficient device memory for full base plus training reserve");
  }

  std::vector<float> host(static_cast<size_t>(result.chunk_rows) * result.dim);
  for (int64_t offset = 0; offset < result.rows; offset += result.chunk_rows) {
    const int64_t rows = std::min(result.chunk_rows, result.rows - offset);
    const size_t elems = static_cast<size_t>(rows) * result.dim;
    in.read(reinterpret_cast<char*>(host.data()),
            static_cast<std::streamsize>(elems * sizeof(float)));
    if (!in) { throw std::runtime_error("short read while loading base"); }
    auto data = std::make_unique<rmm::device_uvector<float>>(elems, stream);
    cudaMemcpyAsync(data->data(), host.data(), elems * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    result.chunks.push_back({offset, rows, std::move(data)});
    memory.sample();
  }
  return result;
}

__global__ void gather_subspace_kernel(float* output,
                                       float* const* chunks,
                                       int64_t output_rows,
                                       int64_t total_rows,
                                       int64_t full_dim,
                                       int64_t chunk_rows,
                                       int pq_len,
                                       int subspace,
                                       int64_t permutation_multiplier,
                                       int64_t permutation_offset)
{
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t count = output_rows * pq_len;
  if (idx >= count) return;
  const int64_t output_row = idx / pq_len;
  const int feature = static_cast<int>(idx % pq_len);
  const int64_t source_row =
    (output_row * permutation_multiplier + permutation_offset) % total_rows;
  const int64_t chunk = source_row / chunk_rows;
  const int64_t local_row = source_row - chunk * chunk_rows;
  output[idx] = chunks[chunk][local_row * full_dim + subspace * pq_len + feature];
}

__global__ void fill_indices_kernel(int64_t* indices, int64_t rows, int64_t offset)
{
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < rows) indices[i] = offset + i;
}

int64_t permutation_multiplier(int64_t rows)
{
  int64_t value = 104729;
  while (std::gcd(value, rows) != 1) value += 2;
  return value;
}

rmm::device_uvector<float> gather_subspace(const chunked_dataset& dataset,
                                           rmm::device_uvector<float*>& device_pointers,
                                           int64_t rows,
                                           int pq_len,
                                           int subspace,
                                           uint64_t seed,
                                           cudaStream_t stream)
{
  rmm::device_uvector<float> output(static_cast<size_t>(rows) * pq_len, stream);
  const int threads = 256;
  const int64_t count = rows * pq_len;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  gather_subspace_kernel<<<blocks, threads, 0, stream>>>(output.data(),
                                                         device_pointers.data(),
                                                         rows,
                                                         dataset.rows,
                                                         dataset.dim,
                                                         dataset.chunk_rows,
                                                         pq_len,
                                                         subspace,
                                                         permutation_multiplier(dataset.rows),
                                                         static_cast<int64_t>(seed % dataset.rows));
  auto status = cudaPeekAtLastError();
  if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
  return output;
}

double recall_at_k(const std::vector<int32_t>& expected,
                   int64_t expected_k,
                   const std::vector<int64_t>& actual,
                   int64_t queries,
                   int64_t k)
{
  uint64_t matches = 0;
  for (int64_t q = 0; q < queries; ++q) {
    for (int64_t a = 0; a < k; ++a) {
      const auto value = actual[static_cast<size_t>(q * k + a)];
      for (int64_t e = 0; e < k; ++e) {
        if (value == expected[static_cast<size_t>(q * expected_k + e)]) {
          ++matches;
          break;
        }
      }
    }
  }
  return static_cast<double>(matches) / static_cast<double>(queries * k);
}

void print_json(const options& opt,
                int64_t base_rows,
                int64_t train_rows,
                int pq_dim,
                double load_ms,
                double init_ms,
                double lloyd_ms,
                double inertia,
                const std::vector<int64_t>& iterations,
                bool converged,
                double index_build_ms,
                double encode_ms,
                double search_ms,
                double recall,
                size_t peak_bytes)
{
  std::cout << std::setprecision(10) << "RESULT_JSON {"
            << "\"status\":\"" << (converged ? "ok" : "non_converged") << "\","
            << "\"method\":\"" << opt.method << "\","
            << "\"step8\":\"" << (opt.method == "scalable" ? opt.step8 : "na") << "\","
            << "\"pq_len\":" << opt.pq_len << ","
            << "\"pq_dim\":" << pq_dim << ","
            << "\"base_rows\":" << base_rows << ","
            << "\"max_train_per_code\":" << opt.max_train_per_code << ","
            << "\"train_rows\":" << train_rows << ","
            << "\"init_size\":" << (opt.method == "random" ? 0 : opt.init_size) << ","
            << "\"n_init\":1,"
            << "\"tol\":" << opt.tol << ","
            << "\"max_iter\":" << opt.max_iter << ","
            << "\"load_ms\":" << load_ms << ","
            << "\"init_ms\":" << init_ms << ","
            << "\"lloyd_ms\":" << lloyd_ms << ","
            << "\"train_ms\":" << init_ms + lloyd_ms << ","
            << "\"inertia\":" << inertia << ","
            << "\"index_build_ms\":" << index_build_ms << ","
            << "\"encode_ms\":" << encode_ms << ","
            << "\"search_ms\":" << search_ms << ","
            << "\"recall_at_100\":" << recall << ","
            << "\"peak_device_bytes_observed\":" << peak_bytes << ","
            << "\"subspace_iterations\":[";
  for (size_t i = 0; i < iterations.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << iterations[i];
  }
  std::cout << "]}" << std::endl;
}

int run(const options& opt)
{
  if (setenv("CUVS_KMEANS_PP_SYNC_STEPS", "1", 1) != 0) {
    throw std::runtime_error("failed to enable profiling synchronization");
  }
  if (setenv("CUVS_KMEANS_PP_STEP8", opt.step8.c_str(), 1) != 0) {
    throw std::runtime_error("failed to select scalable Step 8 mode");
  }

  raft::device_resources resources;
  auto stream = raft::resource::get_cuda_stream(resources);
  memory_observer memory;

  const auto load_start = clock_type::now();
  auto dataset = load_chunked_dataset(opt.base, opt.chunk_rows, opt.row_cap, stream, memory);
  const double load_ms = elapsed_ms(load_start);
  if (dataset.dim % opt.pq_len != 0) { throw std::runtime_error("pq_len must divide dimension"); }
  const int pq_dim = static_cast<int>(dataset.dim / opt.pq_len);
  const int64_t requested_train_rows =
    opt.train_rows > 0 ? opt.train_rows : opt.max_train_per_code * int64_t{256};
  const int64_t train_rows = std::min(dataset.rows, requested_train_rows);
  if (train_rows < 256) { throw std::runtime_error("training requires at least 256 rows"); }
  if (opt.method != "random" && (opt.init_size < 256 || opt.init_size > train_rows)) {
    throw std::runtime_error("init_size must be in [256, train_rows]");
  }

  std::vector<float*> host_pointers;
  host_pointers.reserve(dataset.chunks.size());
  for (auto const& chunk : dataset.chunks) host_pointers.push_back(chunk.data->data());
  rmm::device_uvector<float*> device_pointers(host_pointers.size(), stream);
  cudaMemcpyAsync(device_pointers.data(),
                  host_pointers.data(),
                  host_pointers.size() * sizeof(float*),
                  cudaMemcpyHostToDevice,
                  stream);

  std::vector<float> pq_centers(static_cast<size_t>(pq_dim) * opt.pq_len * 256);
  std::vector<int64_t> iterations;
  iterations.reserve(pq_dim);
  double init_ms = 0.0;
  double lloyd_ms = 0.0;
  double inertia_sum = 0.0;
  bool converged = true;

  for (int subspace = 0; subspace < pq_dim; ++subspace) {
    auto train = gather_subspace(
      dataset, device_pointers, train_rows, opt.pq_len, subspace, opt.seed, stream);
    auto train_view = raft::make_device_matrix_view<const float, int64_t>(
      train.data(), train_rows, static_cast<int64_t>(opt.pq_len));
    auto centers = raft::make_device_matrix<float, int64_t>(resources, 256, opt.pq_len);

    cuvs::cluster::kmeans::params init_params;
    init_params.metric = cuvs::distance::DistanceType::L2Expanded;
    init_params.n_clusters = 256;
    init_params.n_init = 1;
    init_params.max_iter = 0;
    init_params.tol = opt.tol;
    init_params.rng_state.seed = opt.seed + static_cast<uint64_t>(subspace);
    if (opt.method == "random") {
      init_params.init = cuvs::cluster::kmeans::params::InitMethod::Random;
    } else {
      init_params.init = cuvs::cluster::kmeans::params::InitMethod::KMeansPlusPlus;
      init_params.oversampling_factor = opt.method == "classic" ? 0.0 : 2.0;
    }

    const int64_t init_rows = opt.method == "random" ? train_rows : opt.init_size;
    auto init_view = raft::make_device_matrix_view<const float, int64_t>(
      train.data(), init_rows, static_cast<int64_t>(opt.pq_len));
    float init_inertia = 0.0f;
    int64_t init_iterations = 0;
    auto init_start = clock_type::now();
    fit_kmeans(resources,
               init_params,
               init_view,
               centers.view(),
               init_inertia,
               init_iterations);
    raft::resource::sync_stream(resources);
    init_ms += elapsed_ms(init_start);

    cuvs::cluster::kmeans::params final_params;
    final_params.metric = cuvs::distance::DistanceType::L2Expanded;
    final_params.n_clusters = 256;
    final_params.init = cuvs::cluster::kmeans::params::InitMethod::Array;
    final_params.n_init = 1;
    final_params.max_iter = opt.max_iter;
    final_params.tol = opt.tol;
    final_params.rng_state.seed = opt.seed + static_cast<uint64_t>(subspace);
    float inertia = 0.0f;
    int64_t n_iter = 0;
    auto lloyd_start = clock_type::now();
    fit_kmeans(resources, final_params, train_view, centers.view(), inertia, n_iter);
    raft::resource::sync_stream(resources);
    lloyd_ms += elapsed_ms(lloyd_start);
    inertia_sum += inertia;
    iterations.push_back(n_iter);
    converged = converged && n_iter < opt.max_iter;

    std::vector<float> centers_host(static_cast<size_t>(256) * opt.pq_len);
    cudaMemcpyAsync(centers_host.data(),
                    centers.data_handle(),
                    centers_host.size() * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);
    for (int d = 0; d < opt.pq_len; ++d) {
      for (int c = 0; c < 256; ++c) {
        pq_centers[(static_cast<size_t>(subspace) * opt.pq_len + d) * 256 + c] =
          centers_host[static_cast<size_t>(c) * opt.pq_len + d];
      }
    }
    memory.sample();
  }

  device_pointers.resize(0, stream);
  raft::resource::sync_stream(resources);

  auto [query_rows, query_dim, query_host] = load_bin<float>(opt.queries);
  auto [gt_rows, gt_k, groundtruth] = load_bin<int32_t>(opt.groundtruth);
  if (query_dim != dataset.dim || gt_rows != query_rows || gt_k < opt.topk) {
    throw std::runtime_error("query or ground-truth shape mismatch");
  }
  auto queries = raft::make_device_matrix<float, int64_t>(resources, query_rows, query_dim);
  cudaMemcpyAsync(queries.data_handle(),
                  query_host.data(),
                  query_host.size() * sizeof(float),
                  cudaMemcpyHostToDevice,
                  stream);
  raft::resource::sync_stream(resources);

  cuvs::neighbors::ivf_pq::index_params index_params;
  index_params.metric = cuvs::distance::DistanceType::L2Expanded;
  index_params.n_lists = 1;
  index_params.pq_bits = 8;
  index_params.pq_dim = pq_dim;
  index_params.codebook_kind = cuvs::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;
  index_params.force_random_rotation = false;
  index_params.add_data_on_build = false;
  index_params.conservative_memory_allocation = true;

  auto pq_view = raft::make_mdspan<const float, uint32_t, raft::row_major, true, false>(
    pq_centers.data(),
    raft::make_extents<uint32_t>(static_cast<uint32_t>(pq_dim),
                                 static_cast<uint32_t>(opt.pq_len),
                                 uint32_t{256}));
  std::vector<float> zero_center(static_cast<size_t>(dataset.dim), 0.0f);
  auto center_view = raft::make_host_matrix_view<const float, uint32_t>(
    zero_center.data(), 1, static_cast<uint32_t>(dataset.dim));
  using index_type = cuvs::neighbors::ivf_pq::index<int64_t>;
  double index_build_ms = 0.0;
  auto make_index_segment = [&]() {
    auto index_start = clock_type::now();
    auto segment = cuvs::neighbors::ivf_pq::build(resources,
                                                   index_params,
                                                   static_cast<uint32_t>(dataset.dim),
                                                   pq_view,
                                                   center_view,
                                                   std::nullopt,
                                                   std::nullopt);
    raft::resource::sync_stream(resources);
    index_build_ms += elapsed_ms(index_start);
    return std::make_unique<index_type>(std::move(segment));
  };

  // A physical cuVS list uses uint32_t mdspan extents, so keep each backing
  // segment below 4 GiB. Together the segments are one logical IVF list: they
  // share the same sole coarse center, PQ codebook, and global source ids.
  constexpr uint64_t max_segment_bytes = uint64_t{3} << 30;
  const int64_t segment_rows = std::max<int64_t>(
    dataset.chunk_rows,
    static_cast<int64_t>((max_segment_bytes / static_cast<uint64_t>(pq_dim) /
                          static_cast<uint64_t>(dataset.chunk_rows)) *
                         static_cast<uint64_t>(dataset.chunk_rows)));
  std::vector<std::unique_ptr<index_type>> index_segments;
  index_segments.push_back(make_index_segment());
  int64_t rows_in_segment = 0;

  double encode_ms = 0.0;
  for (auto& chunk : dataset.chunks) {
    if (rows_in_segment > 0 && rows_in_segment + chunk.rows > segment_rows) {
      index_segments.push_back(make_index_segment());
      rows_in_segment = 0;
    }
    auto chunk_view = raft::make_device_matrix_view<const float, int64_t>(
      chunk.data->data(), chunk.rows, dataset.dim);
    rmm::device_uvector<int64_t> indices(chunk.rows, stream);
    const int threads = 256;
    const int blocks = static_cast<int>((chunk.rows + threads - 1) / threads);
    fill_indices_kernel<<<blocks, threads, 0, stream>>>(indices.data(), chunk.rows, chunk.offset);
    auto index_view = raft::make_device_vector_view<const int64_t, int64_t>(
      indices.data(), chunk.rows);
    auto encode_start = clock_type::now();
    cuvs::neighbors::ivf_pq::extend(
      resources, chunk_view, std::make_optional(index_view), index_segments.back().get());
    raft::resource::sync_stream(resources);
    encode_ms += elapsed_ms(encode_start);
    rows_in_segment += chunk.rows;
    // Destroy the allocation after encoding. device_uvector::resize(0) retains
    // capacity, which prevents the progressively growing IVF-PQ list from
    // reclaiming the consumed float chunks.
    chunk.data.reset();
    raft::resource::sync_stream(resources);
    memory.sample();
  }

  cuvs::neighbors::ivf_pq::search_params search_params;
  search_params.n_probes = 1;
  search_params.lut_dtype = CUDA_R_32F;
  search_params.internal_distance_dtype = CUDA_R_32F;
  search_params.coarse_search_dtype = CUDA_R_32F;
  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(resources, query_rows, opt.topk);
  auto distances = raft::make_device_matrix<float, int64_t>(resources, query_rows, opt.topk);
  const size_t result_size = static_cast<size_t>(query_rows) * opt.topk;
  std::vector<int64_t> segment_neighbors(index_segments.size() * result_size);
  std::vector<float> segment_distances(index_segments.size() * result_size);
  auto search_start = clock_type::now();
  for (size_t segment = 0; segment < index_segments.size(); ++segment) {
    cuvs::neighbors::ivf_pq::search(resources,
                                    search_params,
                                    *index_segments[segment],
                                    raft::make_const_mdspan(queries.view()),
                                    neighbors.view(),
                                    distances.view());
    raft::resource::sync_stream(resources);
    cudaMemcpyAsync(segment_neighbors.data() + segment * result_size,
                    neighbors.data_handle(),
                    result_size * sizeof(int64_t),
                    cudaMemcpyDeviceToHost,
                    stream);
    cudaMemcpyAsync(segment_distances.data() + segment * result_size,
                    distances.data_handle(),
                    result_size * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);
  }
  const double search_ms = elapsed_ms(search_start);

  std::vector<int64_t> neighbors_host(result_size);
  std::vector<std::pair<float, int64_t>> candidates(index_segments.size() * opt.topk);
  for (int64_t query = 0; query < query_rows; ++query) {
    size_t candidate = 0;
    for (size_t segment = 0; segment < index_segments.size(); ++segment) {
      const size_t base = segment * result_size + static_cast<size_t>(query) * opt.topk;
      for (int rank = 0; rank < opt.topk; ++rank) {
        candidates[candidate++] = {segment_distances[base + rank],
                                   segment_neighbors[base + rank]};
      }
    }
    std::partial_sort(candidates.begin(),
                      candidates.begin() + opt.topk,
                      candidates.end(),
                      [](auto const& lhs, auto const& rhs) { return lhs.first < rhs.first; });
    for (int rank = 0; rank < opt.topk; ++rank) {
      neighbors_host[static_cast<size_t>(query) * opt.topk + rank] = candidates[rank].second;
    }
  }
  const double recall = recall_at_k(groundtruth, gt_k, neighbors_host, query_rows, opt.topk);
  memory.sample();

  print_json(opt,
             dataset.rows,
             train_rows,
             pq_dim,
             load_ms,
             init_ms,
             lloyd_ms,
             inertia_sum,
             iterations,
             converged,
             index_build_ms,
             encode_ms,
             search_ms,
             recall,
             memory.peak_used);
  return 0;
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    return run(parse_args(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "FATAL: " << error.what() << std::endl;
    return 1;
  }
}
