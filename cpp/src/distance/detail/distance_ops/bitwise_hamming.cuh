/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/error.hpp>

#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <type_traits>

namespace cuvs::distance::detail::ops {

/**
 * @brief the Bitwise Hamming distance matrix calculation
 *  It computes the following equation:
 *
 *    c_ij = sum_k popcount(x_ik XOR y_kj)
 *
 * where x and y are binary data packed as uint8_t, or as uint32_t words holding four packed
 * bytes each (k then counts words).
 */
template <typename DataType, typename AccType, typename IdxType>
struct bitwise_hamming_distance_op {
  using DataT = DataType;
  using AccT  = AccType;
  using IdxT  = IdxType;

  IdxT k;

  bitwise_hamming_distance_op(IdxT k_) : k(k_)
  {
    static_assert(std::is_same_v<DataT, uint8_t> || std::is_same_v<DataT, uint32_t>,
                  "BitwiseHamming only supports packed uint8_t or uint32_t data");
    static_assert(std::is_same_v<AccT, uint32_t>, "BitwiseHamming requires a uint32_t accumulator");
    RAFT_EXPECTS(
      k >= 0 && static_cast<uint64_t>(k) <= std::numeric_limits<AccT>::max() / (8 * sizeof(DataT)),
      "BitwiseHamming dimension exceeds the uint32_t accumulator range");
  }

  static constexpr bool use_norms            = false;
  static constexpr bool expensive_inner_loop = false;

  template <typename Policy>
  static constexpr size_t shared_mem_size()
  {
    return Policy::SmemSize;
  }

  __device__ __forceinline__ void core(AccT& acc, DataT& x, DataT& y) const
  {
    if constexpr (std::is_same_v<DataT, uint32_t>) {
      acc += static_cast<AccT>(__popc(x ^ y));
    } else {
      // Ensure proper masking and casting to avoid undefined behavior
      uint32_t xor_val    = static_cast<uint32_t>(static_cast<uint8_t>(x ^ y));
      uint32_t masked_val = xor_val & 0xffu;
      int popcount        = __popc(masked_val);
      acc += static_cast<AccT>(popcount);
    }
  }

  template <typename Policy>
  __device__ __forceinline__ void epilog(AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                                         AccT* regxn,
                                         AccT* regyn,
                                         IdxT gridStrideX,
                                         IdxT gridStrideY) const
  {
  }
};

}  // namespace cuvs::distance::detail::ops
