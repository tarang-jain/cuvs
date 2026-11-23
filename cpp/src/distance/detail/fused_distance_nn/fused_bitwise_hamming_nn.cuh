/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../distance_ops/bitwise_hamming.cuh"  // ops::bitwise_hamming_distance_op
#include "../pairwise_distance_base.cuh"        // PairwiseDistances
#include "helper_structs.cuh"
#include "simt_kernel.cuh"

#include <raft/linalg/contractions.cuh>  // raft::linalg::KernelPolicy

#include <cstdint>

namespace cuvs {
namespace distance {
namespace detail {

template <typename DataT,
          typename OutT,
          typename IdxT,
          typename Policy,
          typename ReduceOpT,
          typename KVPReduceOpT>
void launchFusedBitwiseHammingNN(OutT* min,
                                 const DataT* x,
                                 const DataT* y,
                                 IdxT m,
                                 IdxT n,
                                 IdxT k,
                                 int* workspace,
                                 ReduceOpT redOp,
                                 KVPReduceOpT pairRedOp,
                                 cudaStream_t stream)
{
  typedef Policy P;

  dim3 blk(P::Nthreads);
  constexpr auto maxVal  = std::numeric_limits<uint32_t>::max();
  using distance_op_type = ops::bitwise_hamming_distance_op<DataT, uint32_t, IdxT>;
  distance_op_type distance_op{k};
  auto kernel = fusedDistanceNNkernel<DataT,
                                      OutT,
                                      IdxT,
                                      P,
                                      ReduceOpT,
                                      KVPReduceOpT,
                                      distance_op_type,
                                      raft::identity_op>;

  constexpr size_t shmemSize = P::SmemSize;

  dim3 grid = launchConfigGenerator<P>(m, n, shmemSize, kernel);

  kernel<<<grid, blk, shmemSize, stream>>>(min,
                                           x,
                                           y,
                                           nullptr,
                                           nullptr,
                                           m,
                                           n,
                                           k,
                                           maxVal,
                                           workspace,
                                           redOp,
                                           pairRedOp,
                                           distance_op,
                                           raft::identity_op{});

  RAFT_CUDA_TRY(cudaGetLastError());
}

/**
 * Hamming distance over packed rows read as uint32_t words, one popcount per word.
 * The tiles match the uint8_t Policy4x4 / Policy4x4Skinny shapes, with K-blocks of 64 and 32 bytes.
 */
template <int Veclen, typename OutT, typename IdxT, typename ReduceOpT, typename KVPReduceOpT>
void fusedBitwiseHammingNNWords(OutT* min,
                                const uint8_t* x,
                                const uint8_t* y,
                                IdxT m,
                                IdxT n,
                                IdxT k,
                                int* workspace,
                                ReduceOpT redOp,
                                KVPReduceOpT pairRedOp,
                                cudaStream_t stream)
{
  using PolicyT       = raft::linalg::KernelPolicy<uint32_t, Veclen, 16, 4, 4, 16, 16>;
  using SkinnyPolicyT = raft::linalg::KernelPolicy<uint32_t, Veclen, 8, 4, 4, 8, 8>;

  const auto* xw = reinterpret_cast<const uint32_t*>(x);
  const auto* yw = reinterpret_cast<const uint32_t*>(y);
  const IdxT kw  = k / IdxT{4};
  if (kw < IdxT{PolicyT::Kblk}) {
    launchFusedBitwiseHammingNN<uint32_t, OutT, IdxT, SkinnyPolicyT>(
      min, xw, yw, m, n, kw, workspace, redOp, pairRedOp, stream);
  } else {
    launchFusedBitwiseHammingNN<uint32_t, OutT, IdxT, PolicyT>(
      min, xw, yw, m, n, kw, workspace, redOp, pairRedOp, stream);
  }
}

/**
 * @brief Fused BitwiseHamming distance and 1-nearest-neighbor
 *
 * This implementation is only meaningful for uint8_t data type.
 * The if constexpr in fusedDistanceNNImpl ensures it's only called for uint8_t.
 * Rows that are a multiple of 4 bytes and 4-byte aligned are processed as uint32_t words;
 * other rows fall back to the byte-wise kernel with `Policy`.
 */
template <typename DataT,
          typename OutT,
          typename IdxT,
          typename Policy,
          typename ReduceOpT,
          typename KVPReduceOpT>
void fusedBitwiseHammingNN(OutT* min,
                           const DataT* x,
                           const DataT* y,
                           const DataT* xn,
                           const DataT* yn,
                           IdxT m,
                           IdxT n,
                           IdxT k,
                           int* workspace,
                           ReduceOpT redOp,
                           KVPReduceOpT pairRedOp,
                           bool sqrt,
                           cudaStream_t stream)
{
  static_assert(std::is_same_v<DataT, uint8_t>, "BitwiseHamming only supports uint8_t");

  const auto bytes = static_cast<std::size_t>(k);
  const auto px    = reinterpret_cast<std::uintptr_t>(x);
  const auto py    = reinterpret_cast<std::uintptr_t>(y);
  auto aligned     = [&](std::size_t a) { return bytes % a == 0 && px % a == 0 && py % a == 0; };

  if (aligned(16)) {
    fusedBitwiseHammingNNWords<4>(min, x, y, m, n, k, workspace, redOp, pairRedOp, stream);
  } else if (aligned(8)) {
    fusedBitwiseHammingNNWords<2>(min, x, y, m, n, k, workspace, redOp, pairRedOp, stream);
  } else if (aligned(4)) {
    fusedBitwiseHammingNNWords<1>(min, x, y, m, n, k, workspace, redOp, pairRedOp, stream);
  } else {
    launchFusedBitwiseHammingNN<DataT, OutT, IdxT, Policy>(
      min, x, y, m, n, k, workspace, redOp, pairRedOp, stream);
  }
}

}  // namespace detail
}  // namespace distance
}  // namespace cuvs
