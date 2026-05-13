/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../distance_ops/l2_exp.cuh"     // ops::l2_exp_distance_op
#include "../pairwise_distance_base.cuh"  // PairwiseDistances
#include "cutlass_base.cuh"
#include "helper_structs.cuh"
#include "simt_kernel.cuh"
#include <raft/core/kvp.hpp>             // raft::KeyValuePair
#include <raft/core/operators.hpp>       // raft::identity_op
#include <raft/linalg/contractions.cuh>  // Policy
#include <raft/util/arch.cuh>            // raft::util::arch::SM_*
#include <raft/util/cuda_utils.cuh>      // raft::ceildiv, raft::shfl

#include <cstddef>  // size_t
#include <cstdint>
#include <limits>  // std::numeric_limits
#include <type_traits>

namespace cuvs {
namespace distance {

namespace detail {

template <typename InputT, typename MathT, typename OutT, typename IdxT, typename ReduceOpT>
RAFT_KERNEL mappedL2NNFallbackKernel(OutT* min,
                                     const InputT* x,
                                     const MathT* y,
                                     const MathT* xn,
                                     const MathT* yn,
                                     IdxT m,
                                     IdxT n,
                                     IdxT k,
                                     bool sqrt,
                                     ReduceOpT redOp)
{
  IdxT row =
    static_cast<IdxT>(blockIdx.x) * static_cast<IdxT>(blockDim.x) + static_cast<IdxT>(threadIdx.x);
  if (row >= m) { return; }

  mapped_tile_load_op<InputT, MathT> map_op;
  raft::KeyValuePair<IdxT, MathT> best;
  best.key   = 0;
  best.value = std::numeric_limits<MathT>::max();

  for (IdxT col = 0; col < n; ++col) {
    MathT dot = MathT{0};
    for (IdxT feature = 0; feature < k; ++feature) {
      dot += map_op(x[row * k + feature]) * y[col * k + feature];
    }

    MathT dist = xn[row] + yn[col] - MathT{2} * dot;
    dist       = dist * static_cast<MathT>(dist > MathT{0});
    if (sqrt) { dist = raft::sqrt(dist); }

    if (dist < best.value) {
      best.key   = col;
      best.value = dist;
    }
  }

  redOp(row, min + row, best);
}

template <typename DataT,
          typename OutT,
          typename IdxT,
          typename Policy,
          typename ReduceOpT,
          typename KVPReduceOpT>
void fusedL2NNImpl(OutT* min,
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
                   bool initOutBuffer,
                   cudaStream_t stream)
{
  // The kernel policy is determined by fusedL2NN.
  typedef Policy P;

  dim3 blk(P::Nthreads);
  auto nblks            = raft::ceildiv<int>(m, P::Nthreads);
  constexpr auto maxVal = std::numeric_limits<DataT>::max();
  typedef raft::KeyValuePair<IdxT, DataT> KVPair;

  if (initOutBuffer) {
    initKernel<DataT, OutT, IdxT, ReduceOpT>
      <<<nblks, P::Nthreads, 0, stream>>>(min, m, maxVal, redOp);
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  namespace arch = raft::util::arch;
  using AccT     = DataT;
  ops::l2_exp_distance_op<DataT, AccT, IdxT> distance_op{sqrt};

  raft::identity_op fin_op{};

  auto kernel = fusedDistanceNNkernel<DataT,
                                      OutT,
                                      IdxT,
                                      P,
                                      ReduceOpT,
                                      KVPReduceOpT,
                                      decltype(distance_op),
                                      decltype(fin_op)>;

  // Get pointer to fp32 SIMT kernel to determine the best compute architecture
  // out of all for which the kernel was compiled for that matches closely
  // to the current device. Other methods to determine the architecture (that do not
  // require a pointer) can be error prone. See:
  // https://github.com/NVIDIA/cub/issues/545
  void* kernel_ptr   = reinterpret_cast<void*>(kernel);
  auto runtime_arch  = arch::kernel_virtual_arch(kernel_ptr);
  auto cutlass_range = arch::SM_range(arch::SM_80(), arch::SM_future());

  if (cutlass_range.contains(runtime_arch)) {
    // If device is SM_80 or later, use CUTLASS-based kernel.
    using L2Op                  = cuvs::distance::detail::ops::l2_exp_cutlass_op<DataT, DataT>;
    using kvp_cg_min_reduce_op_ = kvp_cg_min_reduce_op<DataT, IdxT, OutT>;
    kvp_cg_min_reduce_op_ cg_reduce_op;
    L2Op L2_dist_op(sqrt);

    IdxT lda, ldb, ldd;
    lda = k, ldb = k, ldd = n;

    cutlassFusedDistanceNN<DataT,
                           DataT,
                           OutT,
                           IdxT,
                           P::Veclen,
                           kvp_cg_min_reduce_op_,
                           L2Op,
                           ReduceOpT,
                           KVPReduceOpT>(x,
                                         y,
                                         xn,
                                         yn,
                                         m,
                                         n,
                                         k,
                                         lda,
                                         ldb,
                                         ldd,
                                         min,
                                         workspace,
                                         cg_reduce_op,
                                         L2_dist_op,
                                         redOp,
                                         pairRedOp,
                                         stream);
  } else {
    // If device less than SM_80, use fp32 SIMT kernel.
    constexpr size_t shmemSize = P::SmemSize + ((P::Mblk + P::Nblk) * sizeof(DataT));
    dim3 grid                  = launchConfigGenerator<P>(m, n, shmemSize, kernel);

    kernel<<<grid, blk, shmemSize, stream>>>(
      min, x, y, xn, yn, m, n, k, maxVal, workspace, redOp, pairRedOp, distance_op, fin_op);
    RAFT_CUDA_TRY(cudaGetLastError());
  }
}

template <typename InputT,
          typename MathT,
          typename OutT,
          typename IdxT,
          typename Policy,
          typename ReduceOpT,
          typename KVPReduceOpT>
void fusedL2NNMappedImpl(OutT* min,
                         const InputT* x,
                         const MathT* y,
                         const MathT* xn,
                         const MathT* yn,
                         IdxT m,
                         IdxT n,
                         IdxT k,
                         int* workspace,
                         ReduceOpT redOp,
                         KVPReduceOpT pairRedOp,
                         bool sqrt,
                         bool initOutBuffer,
                         cudaStream_t stream)
{
  static_assert(std::is_same_v<MathT, float>, "mapped fused L2 currently supports float math only");
  static_assert(std::is_same_v<InputT, int8_t> || std::is_same_v<InputT, uint8_t>,
                "mapped fused L2 currently supports int8_t/uint8_t input only");

  typedef Policy P;

  dim3 blk(P::Nthreads);
  auto nblks            = raft::ceildiv<int>(m, P::Nthreads);
  constexpr auto maxVal = std::numeric_limits<MathT>::max();

  if (initOutBuffer) {
    initKernel<MathT, OutT, IdxT, ReduceOpT>
      <<<nblks, P::Nthreads, 0, stream>>>(min, m, maxVal, redOp);
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  namespace arch = raft::util::arch;
  ops::l2_exp_cutlass_op<MathT, MathT> L2_dist_op(sqrt);

  raft::identity_op fin_op{};
  auto simt_kernel   = fusedDistanceNNkernel<MathT,
                                             OutT,
                                             IdxT,
                                             P,
                                             ReduceOpT,
                                             KVPReduceOpT,
                                             decltype(L2_dist_op),
                                             decltype(fin_op)>;
  void* kernel_ptr   = reinterpret_cast<void*>(simt_kernel);
  auto runtime_arch  = arch::kernel_virtual_arch(kernel_ptr);
  auto cutlass_range = arch::SM_range(arch::SM_80(), arch::SM_future());

  if (cutlass_range.contains(runtime_arch)) {
    using kvp_cg_min_reduce_op_ = kvp_cg_min_reduce_op<MathT, IdxT, OutT>;
    kvp_cg_min_reduce_op_ cg_reduce_op;

    IdxT lda, ldb, ldd;
    lda = k, ldb = k, ldd = n;

    cutlassFusedDistanceNNMapped<InputT,
                                 MathT,
                                 MathT,
                                 OutT,
                                 IdxT,
                                 P::Veclen,
                                 kvp_cg_min_reduce_op_,
                                 decltype(L2_dist_op),
                                 ReduceOpT,
                                 KVPReduceOpT>(x,
                                               y,
                                               xn,
                                               yn,
                                               m,
                                               n,
                                               k,
                                               lda,
                                               ldb,
                                               ldd,
                                               min,
                                               workspace,
                                               cg_reduce_op,
                                               L2_dist_op,
                                               redOp,
                                               pairRedOp,
                                               stream);
  } else {
    constexpr int block_size = 128;
    auto grid_size           = raft::ceildiv<int>(static_cast<int>(m), block_size);
    mappedL2NNFallbackKernel<InputT, MathT, OutT, IdxT, ReduceOpT>
      <<<grid_size, block_size, 0, stream>>>(min, x, y, xn, yn, m, n, k, sqrt, redOp);
    RAFT_CUDA_TRY(cudaGetLastError());
  }
}

}  // namespace detail
}  // namespace distance
}  // namespace cuvs
