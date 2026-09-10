/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
#define CUDA_TRY(call)                                                                    \
  do {                                                                                    \
    auto status = (call);                                                                 \
    if (status != cudaSuccess) {                                                          \
      throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(status));   \
    }                                                                                     \
  } while (false)

using steady_clock = std::chrono::steady_clock;
struct options {
  std::filesystem::path dataset;
  uint64_t rows = 5000000, batch_rows = 1000000;
  uint32_t mesoclusters = 32, iterations = 20, cpu_threads = 100;
};

uint64_t number(std::string_view flag, char const* value)
{
  size_t used = 0;
  auto result = std::stoull(value, &used);
  if (used != std::strlen(value) || result == 0) {
    throw std::invalid_argument(std::string(flag) + " must be positive");
  }
  return result;
}

options parse(int argc, char** argv)
{
  options o;
  for (int i = 1; i < argc; ++i) {
    if (i + 1 == argc) throw std::invalid_argument(std::string(argv[i]) + " needs a value");
    std::string_view flag{argv[i]};
    auto* value = argv[++i];
    if (flag == "--dataset") o.dataset = value;
    else if (flag == "--rows") o.rows = number(flag, value);
    else if (flag == "--batch-rows") o.batch_rows = number(flag, value);
    else if (flag == "--mesoclusters") o.mesoclusters = number(flag, value);
    else if (flag == "--iterations") o.iterations = number(flag, value);
    else if (flag == "--cpu-threads") o.cpu_threads = number(flag, value);
    else throw std::invalid_argument("unknown option " + std::string(flag));
  }
  if (o.dataset.empty()) throw std::invalid_argument("--dataset is required");
  o.batch_rows = std::min(o.batch_rows, o.rows);
  return o;
}

class pinned_bytes {
 public:
  explicit pinned_bytes(size_t bytes) : bytes_{bytes} { CUDA_TRY(cudaHostAlloc(&ptr_, bytes, 0)); }
  ~pinned_bytes() { if (ptr_) cudaFreeHost(ptr_); }
  pinned_bytes(pinned_bytes const&) = delete;
  pinned_bytes& operator=(pinned_bytes const&) = delete;
  pinned_bytes(pinned_bytes&& x) noexcept : ptr_{x.ptr_}, bytes_{x.bytes_} { x.ptr_ = nullptr; }
  void* data() { return ptr_; }
  size_t size() const { return bytes_; }
 private:
  void* ptr_{};
  size_t bytes_{};
};

template <class T>
class device_array {
 public:
  explicit device_array(size_t n) { CUDA_TRY(cudaMalloc(&ptr_, n * sizeof(T))); }
  ~device_array() { if (ptr_) cudaFree(ptr_); }
  device_array(device_array const&) = delete;
  device_array& operator=(device_array const&) = delete;
  T* data() { return ptr_; }
 private:
  T* ptr_{};
};

struct fbin {
  uint64_t rows;
  uint32_t dim;
  pinned_bytes values;
  fbin(uint64_t n, uint32_t d) : rows{n}, dim{d}, values{size_t(n) * d * sizeof(float)} {}
};

fbin load(std::filesystem::path const& path, uint64_t rows)
{
  std::ifstream in(path, std::ios::binary);
  uint32_t h[2]{};
  if (!in.read(reinterpret_cast<char*>(h), sizeof(h)) || h[0] == 0 || h[1] == 0) {
    throw std::runtime_error("invalid fbin");
  }
  uintmax_t expected = 8 + uintmax_t(h[0]) * h[1] * sizeof(float);
  if (std::filesystem::file_size(path) != expected || rows > h[0]) {
    throw std::runtime_error("fbin size/header mismatch");
  }
  fbin x(rows, h[1]);
  if (!in.read(static_cast<char*>(x.values.data()), x.values.size())) {
    throw std::runtime_error("short fbin read");
  }
  return x;
}

__global__ void bucket(float const* in,
                       float* out,
                       uint32_t const* labels,
                       uint64_t global_offset,
                       uint64_t rows,
                       uint32_t dim,
                       uint64_t const* offsets,
                       unsigned long long* counts)
{
  uint64_t row = blockIdx.x;
  if (row >= rows) return;
  __shared__ unsigned long long dst;
  if (threadIdx.x == 0) {
    auto label = labels[global_offset + row];
    dst = offsets[label] + atomicAdd(counts + label, 1ULL);
  }
  __syncthreads();
  for (uint32_t col = threadIdx.x; col < dim; col += blockDim.x) {
    out[dst * dim + col] = in[row * dim + col];
  }
}

double elapsed(steady_clock::time_point start)
{
  return std::chrono::duration<double>(steady_clock::now() - start).count();
}
struct result { double seconds, gibps; };

result sequential(float const* host,
                  uint64_t rows,
                  uint32_t dim,
                  uint64_t batch_rows,
                  uint32_t mesos,
                  uint32_t iterations,
                  std::vector<uint32_t> const& labels,
                  std::vector<std::vector<uint64_t>> const& batch_offsets,
                  cudaStream_t stream)
{
  device_array<float> input(size_t(batch_rows) * dim), output(size_t(batch_rows) * dim);
  device_array<uint32_t> d_labels(rows);
  device_array<uint64_t> d_offsets(mesos);
  device_array<unsigned long long> d_counts(mesos);
  CUDA_TRY(cudaMemcpyAsync(d_labels.data(), labels.data(), rows * sizeof(uint32_t),
                           cudaMemcpyHostToDevice, stream));
  CUDA_TRY(cudaStreamSynchronize(stream));
  auto start = steady_clock::now();
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    size_t batch_id = 0;
    for (uint64_t off = 0; off < rows; off += batch_rows, ++batch_id) {
      uint64_t n = std::min(batch_rows, rows - off);
      CUDA_TRY(cudaMemcpyAsync(input.data(), host + off * dim, n * dim * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
      CUDA_TRY(cudaMemcpyAsync(d_offsets.data(), batch_offsets[batch_id].data(),
                               mesos * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
      CUDA_TRY(cudaMemsetAsync(d_counts.data(), 0, mesos * sizeof(unsigned long long), stream));
      bucket<<<unsigned(n), 256, 0, stream>>>(input.data(), output.data(), d_labels.data(), off, n,
                                              dim, d_offsets.data(), d_counts.data());
      CUDA_TRY(cudaGetLastError());
    }
  }
  CUDA_TRY(cudaStreamSynchronize(stream));
  double s = elapsed(start);
  double gib = double(rows) * dim * sizeof(float) * iterations / double(uint64_t{1} << 30);
  return {s, gib / s};
}

result gather(float const* host,
              uint64_t rows,
              uint32_t dim,
              uint64_t batch_rows,
              uint32_t iterations,
              std::vector<uint64_t> const& ids,
              std::vector<uint64_t> const& offsets,
              uint32_t threads,
              cudaStream_t stream)
{
  pinned_bytes staging(size_t(batch_rows) * dim * sizeof(float));
  device_array<float> device_batch(size_t(batch_rows) * dim);
  auto* packed = static_cast<float*>(staging.data());
  omp_set_dynamic(0);
  omp_set_num_threads(threads);
  auto start = steady_clock::now();
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    for (size_t meso = 0; meso + 1 < offsets.size(); ++meso) {
      for (uint64_t off = offsets[meso]; off < offsets[meso + 1]; off += batch_rows) {
        uint64_t n = std::min(batch_rows, offsets[meso + 1] - off);
#pragma omp parallel for schedule(static)
        for (int64_t local = 0; local < int64_t(n); ++local) {
          uint64_t source = ids[off + uint64_t(local)];
          std::memcpy(packed + uint64_t(local) * dim, host + source * dim, size_t(dim) * sizeof(float));
        }
        CUDA_TRY(cudaMemcpyAsync(device_batch.data(), packed, n * dim * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));
        CUDA_TRY(cudaStreamSynchronize(stream));
      }
    }
  }
  double s = elapsed(start);
  double gib = double(rows) * dim * sizeof(float) * iterations / double(uint64_t{1} << 30);
  return {s, gib / s};
}
}  // namespace

int main(int argc, char** argv)
{
  try {
    auto o = parse(argc, argv);
    CUDA_TRY(cudaSetDevice(0));
    cudaStream_t stream;
    CUDA_TRY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto load_start = steady_clock::now();
    auto x = load(o.dataset, o.rows);
    double load_seconds = elapsed(load_start);
    auto* host = static_cast<float const*>(x.values.data());

    std::vector<uint32_t> labels(o.rows);
    std::vector<uint64_t> counts(o.mesoclusters);
    for (uint64_t row = 0; row < o.rows; ++row) {
      uint64_t mixed = row * 11400714819323198485ull;
      labels[row] = uint32_t((mixed >> 32) % o.mesoclusters);
      ++counts[labels[row]];
    }
    std::vector<uint64_t> offsets(o.mesoclusters + 1);
    for (uint32_t i = 0; i < o.mesoclusters; ++i) offsets[i + 1] = offsets[i] + counts[i];
    auto cursor = offsets;
    std::vector<uint64_t> ids(o.rows);
    for (uint64_t row = 0; row < o.rows; ++row) ids[cursor[labels[row]]++] = row;

    std::vector<std::vector<uint64_t>> batch_offsets;
    for (uint64_t off = 0; off < o.rows; off += o.batch_rows) {
      uint64_t n = std::min(o.batch_rows, o.rows - off);
      std::vector<uint64_t> local(o.mesoclusters), prefix(o.mesoclusters);
      for (uint64_t row = 0; row < n; ++row) ++local[labels[off + row]];
      for (uint32_t i = 1; i < o.mesoclusters; ++i) prefix[i] = prefix[i - 1] + local[i - 1];
      batch_offsets.push_back(std::move(prefix));
    }

    std::cout << std::fixed << std::setprecision(3)
              << "rows=" << o.rows << "\ndimensions=" << x.dim
              << "\nbatch_rows=" << o.batch_rows << "\nmesoclusters=" << o.mesoclusters
              << "\niterations=" << o.iterations << "\ncpu_threads=" << o.cpu_threads
              << "\nhost_allocation=pinned(cudaHostAlloc)"
              << "\ndataset_load_seconds=" << load_seconds << '\n';
    auto a = sequential(host, o.rows, x.dim, o.batch_rows, o.mesoclusters, o.iterations,
                        labels, batch_offsets, stream);
    std::cout << "sequential_gpu_bucket_seconds=" << a.seconds
              << "\nsequential_useful_gib_per_second=" << a.gibps << '\n';
    auto b = gather(host, o.rows, x.dim, o.batch_rows, o.iterations, ids, offsets,
                    o.cpu_threads, stream);
    std::cout << "mesocluster_host_gather_seconds=" << b.seconds
              << "\nmesocluster_gather_useful_gib_per_second=" << b.gibps << '\n';
    CUDA_TRY(cudaStreamDestroy(stream));
  } catch (std::exception const& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
