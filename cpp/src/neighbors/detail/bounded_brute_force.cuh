/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, Sirius Contributors.
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "./bounded_gemm.cuh"
#include "./int8_brute_force.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <raft/core/resource/device_properties.hpp>

#include <type_traits>

namespace cuvs::neighbors::detail::bounded_int8 {

// Direct distances from the original FP32 vectors, one warp per surviving pair.
// Square-root metrics use FP64 until after sqrt so a representable distance cannot overflow
// merely because its square exceeds FP32. Squared-distance metrics retain FP32 arithmetic.
template <bool SquareRoot>
RAFT_KERNEL refine_kernel(const float* dataset,
                          const float* queries,
                          int64_t dim,
                          const int32_t* rows,
                          const int64_t* ids,
                          int64_t count,
                          float* distances)
{
  int const lane = threadIdx.x % 32;
  for (int64_t p = int64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32; p < count;
       p += int64_t(gridDim.x) * (blockDim.x / 32)) {
    using accumulator_type = std::conditional_t<SquareRoot, double, float>;
    accumulator_type sum   = 0;
    for (int64_t j = lane; j < dim; j += 32) {
      auto const delta = accumulator_type(queries[int64_t(rows[p]) * dim + j]) -
                         accumulator_type(dataset[ids[p] * dim + j]);
      if constexpr (SquareRoot) {
        sum = fma(delta, delta, sum);
      } else {
        sum = fmaf(delta, delta, sum);
      }
    }
    for (int s = 16; s > 0; s /= 2) {
      sum += __shfl_xor_sync(0xffffffffu, sum, s);
    }
    if (lane == 0) {
      if constexpr (SquareRoot) {
        distances[p] = static_cast<float>(sqrt(sum));
      } else {
        distances[p] = sum;
      }
    }
  }
}

// Any k distinct original rows give an upper bound on the true kth distance. Evaluate that
// bound with directed FP64 operations, rather than assuming an FP32 reduction rounded upward.
// Both the initial SQ-selected seed and subsequent running top-k use this same calculation.
RAFT_KERNEL limits_kernel(const float* dataset,
                          const float* queries,
                          const int64_t* neighbors,
                          const double* query_error,
                          double dataset_error,
                          double scale,
                          int64_t k,
                          int64_t dim,
                          int64_t* limits)
{
  __shared__ double maxima[8];
  int const lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int64_t const row = blockIdx.x;
  double largest    = 0.0;
  for (int64_t rank = warp; rank < k; rank += 8) {
    int64_t const id = neighbors[row * k + rank];
    double sum       = 0.0;
    for (int64_t j = lane; j < dim; j += 32) {
      double const q     = queries[row * dim + j];
      double const x     = dataset[id * dim + j];
      double const delta = fmax(fabs(__dsub_rd(q, x)), fabs(__dsub_ru(q, x)));
      sum                = __dadd_ru(sum, __dmul_ru(delta, delta));
    }
    for (int s = 16; s > 0; s /= 2) {
      sum = __dadd_ru(sum, __shfl_xor_sync(0xffffffffu, sum, s));
    }
    largest = fmax(largest, sum);
  }
  if (lane == 0) { maxima[warp] = largest; }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 0; w < 8; ++w) {
      largest = fmax(largest, maxima[w]);
    }
    double reach                   = __dadd_ru(__dsqrt_ru(largest), query_error[row]);
    reach                          = __ddiv_ru(__dadd_ru(reach, dataset_error), scale);
    double const threshold         = __dmul_ru(reach, reach);
    int64_t const maximum_distance = dim * int64_t{255 * 255};
    limits[row]                    = !isfinite(threshold) || threshold >= double(maximum_distance)
                                       ? maximum_distance
                                       : __double2ll_rd(threshold);
  }
}

// Sirius groups candidates by query and sorts their refined distances. Pack the query and
// nonnegative FP32 distance into one key, with the global dataset ID as the sorting payload.
RAFT_KERNEL merge_keys_kernel(const float* old_distances,
                              const int64_t* old_ids,
                              int64_t k,
                              int64_t old_count,
                              const int32_t* rows,
                              const float* distances,
                              const int64_t* ids,
                              int64_t count,
                              uint64_t* keys,
                              int64_t* values)
{
  for (int64_t p = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; p < old_count + count;
       p += int64_t(gridDim.x) * blockDim.x) {
    auto const row      = p < old_count ? p / k : rows[p - old_count];
    auto const distance = p < old_count ? old_distances[p] : distances[p - old_count];
    auto const id       = p < old_count ? old_ids[p] : ids[p - old_count];
    // Empty accumulator slots sort after valid infinite distances as well.
    keys[p]   = (uint64_t(row) << 32) | (id < 0 ? 0xffffffffu : __float_as_uint(distance));
    values[p] = id;
  }
}

RAFT_KERNEL row_starts_kernel(const uint64_t* keys, int64_t n, int64_t* starts)
{
  for (int64_t p = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; p < n;
       p += int64_t(gridDim.x) * blockDim.x) {
    auto const row = keys[p] >> 32;
    if (p == 0 || row != (keys[p - 1] >> 32)) { starts[row] = p; }
  }
}

RAFT_KERNEL take_topk_kernel(const uint64_t* keys,
                             const int64_t* values,
                             const int64_t* starts,
                             int64_t m,
                             int64_t k,
                             float* distances,
                             int64_t* ids)
{
  for (int64_t p = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; p < m * k;
       p += int64_t(gridDim.x) * blockDim.x) {
    auto const source = starts[p / k] + p % k;
    distances[p]      = __uint_as_float(static_cast<unsigned>(keys[source]));
    ids[p]            = values[source];
  }
}

inline void search(raft::resources const& res,
                   brute_force::device_sq_index<float> const& index,
                   raft::device_matrix_view<const float, int64_t> original,
                   raft::device_matrix_view<const float, int64_t> queries,
                   raft::device_matrix_view<int64_t, int64_t> neighbors,
                   raft::device_matrix_view<float, int64_t> distances)
{
  auto const n = int64_t(index.size()), m = queries.extent(0), d = int64_t(index.dim());
  auto const k           = neighbors.extent(1);
  auto const metric      = index.metric();
  bool const square_root = metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                           metric == cuvs::distance::DistanceType::L2SqrtUnexpanded;
  auto refine = square_root ? refine_kernel<true> : refine_kernel<false>;
  RAFT_EXPECTS(metric == cuvs::distance::DistanceType::L2Expanded ||
                 metric == cuvs::distance::DistanceType::L2Unexpanded ||
                 metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                 metric == cuvs::distance::DistanceType::L2SqrtUnexpanded,
               "Bounded SQ search supports only L2 metrics");
  RAFT_EXPECTS(d > 0 && d % 16 == 0, "Bounded SQ search requires a dimension divisible by 16");
  RAFT_EXPECTS(original.extent(0) == n && original.extent(1) == d && queries.extent(1) == d,
               "Original dataset and query dimensions must match the SQ index");
  RAFT_EXPECTS(neighbors.extent(0) == m && distances.extent(0) == m && distances.extent(1) == k,
               "Output shapes must match the number of queries and k");
  RAFT_EXPECTS(k > 0 && k <= n, "k must satisfy 1 <= k <= index size");
  auto const dataset = index.dataset();
  RAFT_EXPECTS(reinterpret_cast<uintptr_t>(dataset.data_const_view().data_handle()) % 16 == 0,
               "Bounded SQ search requires 16-byte-aligned codes");
  auto const manifest = dataset.const_manifest_view();
  RAFT_EXPECTS(manifest.max_error >= 0.0, "SQ maximum reconstruction error must be nonnegative");
  auto const affine = get_sq_affine_parameters<float>(manifest, d);
  if (m == 0) { return; }
  auto const stream = raft::resource::get_cuda_stream(res);

  // Limit workspace independently of the full query/corpus product. All candidates in a tile
  // fit, even when the bound is infinite; there is no fixed survivor cap or lossy truncation.
  int64_t const corpus_tile = std::min(n, std::max<int64_t>(65536, k));
  int64_t const query_tile  = std::min<int64_t>(
    m, std::max<int64_t>(1, std::min<int64_t>(128, (int64_t{1} << 23) / (corpus_tile + k))));
  int64_t const capacity       = query_tile * corpus_tile;
  int64_t const merge_capacity = capacity + query_tile * k;
  int64_t const seed_rows      = std::min(n, std::max<int64_t>(1024, k));
  int64_t const seed_stride    = raft::round_up_safe(seed_rows, int64_t{4});
  RAFT_EXPECTS(seed_stride <= std::numeric_limits<int>::max(),
               "Bounded SQ search seed is too large for cuBLAS");

  auto query_codes         = raft::make_device_matrix<int8_t, int64_t>(res, query_tile, d);
  auto query_norms         = raft::make_device_vector<double, int64_t>(res, query_tile);
  auto query_errors        = raft::make_device_vector<double, int64_t>(res, query_tile);
  auto limits              = raft::make_device_vector<int64_t, int64_t>(res, query_tile);
  auto count               = raft::make_device_vector<unsigned long long, int64_t>(res, 1);
  auto candidate_rows      = raft::make_device_vector<int32_t, int64_t>(res, capacity);
  auto candidate_ids       = raft::make_device_vector<int64_t, int64_t>(res, capacity);
  auto candidate_distances = raft::make_device_vector<float, int64_t>(res, capacity);
  auto seed_scores    = raft::make_device_matrix<int32_t, int64_t>(res, query_tile, seed_stride);
  auto seed_distances = raft::make_device_matrix<float, int64_t>(res, query_tile, seed_rows);
  auto seed_codes =
    raft::make_device_matrix<int8_t, int64_t>(res, seed_stride != seed_rows ? seed_stride : 0, d);
  auto keys          = raft::make_device_vector<uint64_t, int64_t>(res, merge_capacity);
  auto sorted_keys   = raft::make_device_vector<uint64_t, int64_t>(res, merge_capacity);
  auto values        = raft::make_device_vector<int64_t, int64_t>(res, merge_capacity);
  auto sorted_values = raft::make_device_vector<int64_t, int64_t>(res, merge_capacity);
  auto starts        = raft::make_device_vector<int64_t, int64_t>(res, query_tile);
  int row_bits       = 1;
  while ((int64_t{1} << row_bits) < query_tile) {
    ++row_bits;
  }
  size_t sort_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(nullptr,
                                                sort_bytes,
                                                keys.data_handle(),
                                                sorted_keys.data_handle(),
                                                values.data_handle(),
                                                sorted_values.data_handle(),
                                                merge_capacity,
                                                0,
                                                32 + row_bits,
                                                stream.get()));
  auto workspace = raft::make_device_vector<char, int64_t>(res, sort_bytes);
  if (raft::resource::get_dry_run_flag(res)) { return; }

  auto const* seed_ptr = dataset.data_const_view().data_handle();
  if (seed_stride != seed_rows) {
    RAFT_CUDA_TRY(cudaMemsetAsync(seed_codes.data_handle(), 0, seed_stride * d, stream.get()));
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      seed_codes.data_handle(), seed_ptr, seed_rows * d, cudaMemcpyDeviceToDevice, stream.get()));
    seed_ptr = seed_codes.data_handle();
  }
  auto const sm_count = raft::resource::get_device_properties(res).multiProcessorCount;
  for (int64_t qo = 0; qo < m; qo += query_tile) {
    int64_t const mq = std::min(query_tile, m - qo);
    auto const query_view =
      raft::make_device_matrix_view<const float, int64_t>(queries.data_handle() + qo * d, mq, d);
    auto const code_view =
      raft::make_device_matrix_view<int8_t, int64_t>(query_codes.data_handle(), mq, d);
    auto const norm_view =
      raft::make_device_vector_view<double, int64_t>(query_norms.data_handle(), mq);
    auto* output_ids       = neighbors.data_handle() + qo * k;
    auto* output_distances = distances.data_handle() + qo * k;
    cuvs::preprocessing::quantize::scalar::transform(res, manifest, query_view, code_view);
    raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
      res, raft::make_const_mdspan(code_view), norm_view);
    cuvs::preprocessing::quantize::scalar::detail::quantization_errors<float>(
      res,
      manifest,
      query_view,
      raft::make_const_mdspan(code_view),
      query_errors.data_handle(),
      nullptr);

    // Seed only establishes a conservative radius; its approximate winners are not final results.
    // Rank in code units: the shared positive scale cannot change the order, and applying it
    // could overflow the FP32 selection keys for large, finite original coordinates.
    int8_gemm(res,
              seed_ptr,
              query_codes.data_handle(),
              seed_scores.data_handle(),
              seed_stride,
              mq,
              d,
              seed_stride);
    int8_postprocess_kernel<<<dim3(int8_grid_size(seed_rows), mq),
                              kInt8BlockSize,
                              0,
                              stream.get()>>>(seed_scores.data_handle(),
                                              seed_distances.data_handle(),
                                              seed_rows,
                                              seed_stride,
                                              query_norms.data_handle(),
                                              index.norms().data_handle(),
                                              nullptr,
                                              nullptr,
                                              1.0,
                                              cuvs::distance::DistanceType::L2Expanded);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    cuvs::selection::select_k(
      res,
      raft::make_device_matrix_view<const float, int64_t>(
        seed_distances.data_handle(), mq, seed_rows),
      std::nullopt,
      raft::make_device_matrix_view<float, int64_t>(output_distances, mq, k),
      raft::make_device_matrix_view<int64_t, int64_t>(output_ids, mq, k),
      true,
      true);
    auto refresh_limits = [&] {
      limits_kernel<<<mq, 256, 0, stream.get()>>>(original.data_handle(),
                                                  query_view.data_handle(),
                                                  output_ids,
                                                  query_errors.data_handle(),
                                                  manifest.max_error,
                                                  affine.scale,
                                                  k,
                                                  d,
                                                  limits.data_handle());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    };
    refresh_limits();
    // The seed rows are scanned again below. Reset the accumulator to avoid duplicate IDs.
    raft::linalg::map_offset(
      res,
      raft::make_device_vector_view<float, int64_t>(output_distances, mq * k),
      [] __device__(int64_t) { return std::numeric_limits<float>::infinity(); });
    RAFT_CUDA_TRY(cudaMemsetAsync(output_ids, 0xff, mq * k * sizeof(int64_t), stream.get()));

    for (int64_t xo = 0; xo < n; xo += corpus_tile) {
      int64_t const nx = std::min(corpus_tile, n - xo);
      RAFT_CUDA_TRY(
        cudaMemsetAsync(count.data_handle(), 0, sizeof(unsigned long long), stream.get()));
      auto const ny =
        static_cast<unsigned>(std::min<int64_t>(raft::ceildiv(nx, int64_t{kTileN}), 2 * sm_count));
      fused_filter_kernel<<<dim3(raft::ceildiv(mq, int64_t{kTileM}), ny), 256, 0, stream.get()>>>(
        dataset.data_const_view().data_handle() + xo * d,
        index.norms().data_handle() + xo,
        nx,
        xo,
        query_codes.data_handle(),
        query_norms.data_handle(),
        mq,
        static_cast<int>(d),
        limits.data_handle(),
        candidate_rows.data_handle(),
        candidate_ids.data_handle(),
        count.data_handle());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      unsigned long long survivors = 0;
      raft::copy(res,
                 raft::make_host_scalar_view(&survivors),
                 raft::make_device_scalar_view<const unsigned long long>(count.data_handle()));
      // CUB needs the ragged candidate count on the host. No distances or candidate vectors
      // are transferred; refinement and merging stay on the resource's CUDA stream.
      raft::resource::sync_stream(res);
      RAFT_EXPECTS(survivors <= static_cast<unsigned long long>(mq * nx),
                   "Invalid bounded candidate count");
      if (survivors == 0) { continue; }
      refine<<<int8_grid_size(raft::ceildiv(int64_t(survivors), int64_t{8}) * 256),
               256,
               0,
               stream.get()>>>(original.data_handle(),
                               query_view.data_handle(),
                               d,
                               candidate_rows.data_handle(),
                               candidate_ids.data_handle(),
                               survivors,
                               candidate_distances.data_handle());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      auto const total = mq * k + int64_t(survivors);
      merge_keys_kernel<<<int8_grid_size(total), 256, 0, stream.get()>>>(
        output_distances,
        output_ids,
        k,
        mq * k,
        candidate_rows.data_handle(),
        candidate_distances.data_handle(),
        candidate_ids.data_handle(),
        survivors,
        keys.data_handle(),
        values.data_handle());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      auto bytes = sort_bytes;
      RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(workspace.data_handle(),
                                                    bytes,
                                                    keys.data_handle(),
                                                    sorted_keys.data_handle(),
                                                    values.data_handle(),
                                                    sorted_values.data_handle(),
                                                    total,
                                                    0,
                                                    32 + row_bits,
                                                    stream.get()));
      row_starts_kernel<<<int8_grid_size(total), 256, 0, stream.get()>>>(
        sorted_keys.data_handle(), total, starts.data_handle());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      take_topk_kernel<<<int8_grid_size(mq * k), 256, 0, stream.get()>>>(
        sorted_keys.data_handle(),
        sorted_values.data_handle(),
        starts.data_handle(),
        mq,
        k,
        output_distances,
        output_ids);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      if (xo + nx < n) { refresh_limits(); }
    }
  }
}

}  // namespace cuvs::neighbors::detail::bounded_int8
