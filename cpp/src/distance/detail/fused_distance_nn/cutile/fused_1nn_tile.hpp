/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <cuda_runtime.h>

#include <cuvs/detail/jit_lto/cutile_compat.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

namespace cuvs {
namespace distance {
namespace detail {

template <typename DataT>
inline constexpr bool is_fused_1nn_cutile_data_v =
  std::is_same_v<DataT, float> || std::is_same_v<DataT, half> || std::is_same_v<DataT, int8_t> ||
  std::is_same_v<DataT, uint8_t>;

// Norm buffers use int64 for byte inputs and FP32 otherwise; FP16 norms accumulate in FP32.
// For FP32 inputs, accumulate TF32-rounded squares in FP32 to match cuTile MMA operands.
template <typename DataT>
using fused_1nn_cutile_norm_t = std::conditional_t<std::is_integral_v<DataT>, int64_t, float>;

// Distance outputs use FP32; integer reductions stay int64 until the final store.
template <typename DataT>
using fused_1nn_cutile_distance_t = float;

template <typename DataT>
inline constexpr int64_t fused_1nn_cutile_max_batch_m = [] {
  constexpr int64_t max_i32         = std::numeric_limits<int>::max();
  constexpr int64_t batch_alignment = 16 / raft::gcd<int64_t>(16, sizeof(DataT));
  return max_i32 - max_i32 % batch_alignment;
}();

template <typename DataT, typename IdxT>
constexpr size_t fused_1nn_cutile_index_workspace_rows(IdxT m)
{
  const auto rows = static_cast<int64_t>(m);
  if (rows <= 0) { return 0; }
  return static_cast<size_t>(
    rows < fused_1nn_cutile_max_batch_m<DataT> ? rows : fused_1nn_cutile_max_batch_m<DataT>);
}

/**
 * Return whether the input problem has a compatible cuTile launcher.
 *
 * This output-independent probe lets callers select native result storage before allocating it.
 */
template <typename DataT, typename IdxT>
  requires is_fused_1nn_cutile_data_v<DataT>
bool is_fused_1nn_tile_available(
  const DataT* x, const DataT* y, IdxT m, IdxT n, IdxT k, cuvs::distance::DistanceType metric);

/**
 * Launch fused 1-NN with cuTile.
 *
 * All launch arguments are validated. An int64 output index requires an int32 workspace sized to
 * fused_1nn_cutile_index_workspace_rows<DataT>(m). This function throws instead of falling back
 * when the explicitly requested cuTile backend is unavailable.
 */
template <typename DataT, typename IdxT>
  requires is_fused_1nn_cutile_data_v<DataT>
void launch_fused_1nn_tile(raft::resources const& handle,
                           IdxT* nearest_idx,
                           fused_1nn_cutile_distance_t<DataT>* nearest_dist,
                           const DataT* x,
                           const DataT* y,
                           const fused_1nn_cutile_norm_t<DataT>* xn,
                           const fused_1nn_cutile_norm_t<DataT>* yn,
                           IdxT m,
                           IdxT n,
                           IdxT k,
                           cuvs::distance::DistanceType metric,
                           bool is_sqrt,
                           void* index_workspace);
template <typename DataT>
bool try_fused_1nn_mixed_tile(raft::resources const& handle,
                              int* indices,
                              float* distances,
                              const DataT* input,
                              const float* centers,
                              const float* input_norms,
                              const float* center_norms,
                              int rows,
                              int candidates,
                              int features,
                              bool take_sqrt);

}  // namespace detail
}  // namespace distance
}  // namespace cuvs
