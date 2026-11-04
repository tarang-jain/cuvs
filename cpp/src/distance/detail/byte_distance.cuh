#pragma once

#include <cuvs/distance/distance.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cuda_utils.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cuvs::distance::detail {

template <typename DataT>
inline constexpr bool is_byte_distance_v =
  std::is_same_v<DataT, int8_t> || std::is_same_v<DataT, uint8_t>;

template <typename DataT, typename CenterT, typename IndexT>
__device__ auto byte_warp_distance(const DataT* input,
                                   const CenterT* center,
                                   IndexT features,
                                   bool inner_product)
{
  using AccT =
    std::conditional_t<std::is_integral_v<DataT> && std::is_integral_v<CenterT>, int64_t, double>;
  AccT distance{};
  for (IndexT feature = threadIdx.x % 32; feature < features; feature += 32) {
    auto left       = static_cast<AccT>(input[feature]);
    auto right      = static_cast<AccT>(center[feature]);
    auto difference = left - right;
    distance += inner_product ? left * right : difference * difference;
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    distance += __shfl_down_sync(0xffffffff, distance, offset);
  }
  return distance;
}

template <typename DataT, typename CenterT, typename IndexT>
__global__ void byte_pairwise_kernel(const DataT* input,
                                     const CenterT* centers,
                                     float* distances,
                                     IndexT rows,
                                     IndexT candidates,
                                     IndexT features,
                                     bool take_sqrt,
                                     bool inner_product)
{
  const auto stride = int64_t{gridDim.x} * (blockDim.x / 32);
  const auto pairs  = int64_t{rows} * candidates;
  for (auto pair = int64_t{blockIdx.x} * (blockDim.x / 32) + threadIdx.x / 32; pair < pairs;
       pair += stride) {
    auto distance = byte_warp_distance(input + (pair / candidates) * features,
                                       centers + (pair % candidates) * features,
                                       features,
                                       inner_product);
    if (threadIdx.x % 32 == 0) {
      distances[pair] = take_sqrt ? sqrt(static_cast<double>(distance)) : distance;
    }
  }
}

template <typename DataT, typename CenterT, typename IndexT>
void byte_pairwise_distance(raft::resources const& handle,
                            raft::device_matrix_view<const DataT, IndexT> input,
                            raft::device_matrix_view<const CenterT, IndexT> centers,
                            raft::device_matrix_view<float, IndexT> distances,
                            DistanceType metric)
{
  static_assert(is_byte_distance_v<DataT> || is_byte_distance_v<CenterT>);
  RAFT_EXPECTS(input.extent(0) >= 0 && centers.extent(0) >= 0 && input.extent(1) > 0,
               "Byte pairwise distance requires nonnegative rows and positive features");
  RAFT_EXPECTS(input.extent(1) == centers.extent(1) && distances.extent(0) == input.extent(0) &&
                 distances.extent(1) == centers.extent(0),
               "Byte pairwise distance shape mismatch");
  RAFT_EXPECTS(metric == DistanceType::L2Expanded || metric == DistanceType::L2Unexpanded ||
                 metric == DistanceType::L2SqrtExpanded ||
                 metric == DistanceType::L2SqrtUnexpanded || metric == DistanceType::InnerProduct,
               "Byte pairwise distance supports L2 and inner product");
  if (input.extent(0) == 0 || centers.extent(0) == 0) { return; }
  RAFT_EXPECTS(input.data_handle() && centers.data_handle() && distances.data_handle(),
               "Byte pairwise distance requires non-null buffers");
  RAFT_EXPECTS(int64_t{input.extent(0)} <= std::numeric_limits<int64_t>::max() / centers.extent(0),
               "Byte pairwise distance output size overflow");
  auto pairs  = int64_t{input.extent(0)} * centers.extent(0);
  auto blocks = static_cast<unsigned>(std::min<int64_t>(raft::ceildiv(pairs, int64_t{8}), 65535));
  byte_pairwise_kernel<<<blocks, 256, 0, raft::resource::get_cuda_stream(handle).get()>>>(
    input.data_handle(),
    centers.data_handle(),
    distances.data_handle(),
    input.extent(0),
    centers.extent(0),
    input.extent(1),
    metric == DistanceType::L2SqrtExpanded || metric == DistanceType::L2SqrtUnexpanded,
    metric == DistanceType::InnerProduct);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename DataT, typename CenterT, typename IndexT>
__global__ void byte_top_1_kernel(const DataT* input,
                                  const CenterT* centers,
                                  IndexT* indices,
                                  float* distances,
                                  IndexT rows,
                                  IndexT candidates,
                                  IndexT features,
                                  bool take_sqrt,
                                  bool inner_product)
{
  using AccT =
    std::conditional_t<std::is_integral_v<DataT> && std::is_integral_v<CenterT>, int64_t, double>;
  const auto stride = int64_t{gridDim.x} * (blockDim.x / 32);
  for (auto row = int64_t{blockIdx.x} * (blockDim.x / 32) + threadIdx.x / 32; row < rows;
       row += stride) {
    AccT best = std::numeric_limits<AccT>::max();
    IndexT best_index{};
    for (IndexT candidate = 0; candidate < candidates; ++candidate) {
      auto distance = byte_warp_distance(
        input + row * features, centers + int64_t{candidate} * features, features, inner_product);
      auto score = inner_product ? -distance : distance;
      if (score < best) {
        best       = score;
        best_index = candidate;
      }
    }
    if (threadIdx.x % 32 == 0) {
      if (indices != nullptr) { indices[row] = best_index; }
      auto distance  = inner_product ? -best : best;
      distances[row] = take_sqrt ? sqrt(static_cast<double>(distance)) : distance;
    }
  }
}

template <typename DataT, typename CenterT, typename IndexT>
void byte_top_1(raft::resources const& handle,
                const DataT* input,
                const CenterT* centers,
                IndexT* indices,
                float* distances,
                IndexT rows,
                IndexT candidates,
                IndexT features,
                bool take_sqrt,
                bool inner_product = false)
{
  auto blocks =
    static_cast<unsigned>(std::min<int64_t>(raft::ceildiv(int64_t{rows}, int64_t{8}), 65535));
  if (blocks == 0) { return; }
  byte_top_1_kernel<<<blocks, 256, 0, raft::resource::get_cuda_stream(handle).get()>>>(
    input, centers, indices, distances, rows, candidates, features, take_sqrt, inner_product);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename DataT, typename NormT, typename IndexT>
__global__ void byte_norm_kernel(const DataT* input, NormT* norms, IndexT rows, IndexT features)
{
  const auto stride = int64_t{gridDim.x} * (blockDim.x / 32);
  for (auto row = int64_t{blockIdx.x} * (blockDim.x / 32) + threadIdx.x / 32; row < rows;
       row += stride) {
    int64_t sum{};
    for (IndexT feature = threadIdx.x % 32; feature < features; feature += 32) {
      int value = input[row * features + feature];
      sum += value * value;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (threadIdx.x % 32 == 0) { norms[row] = static_cast<NormT>(sum); }
  }
}

template <typename DataT, typename NormT, typename IndexT>
void byte_norm(raft::resources const& handle,
               raft::device_matrix_view<const DataT, IndexT> input,
               raft::device_vector_view<NormT, IndexT> norms)
{
  static_assert(is_byte_distance_v<DataT>);
  RAFT_EXPECTS(norms.extent(0) == input.extent(0), "Byte norm shape mismatch");
  auto blocks = static_cast<unsigned>(
    std::min<int64_t>(raft::ceildiv(int64_t{input.extent(0)}, int64_t{8}), 65535));
  if (blocks == 0) { return; }
  byte_norm_kernel<<<blocks, 256, 0, raft::resource::get_cuda_stream(handle).get()>>>(
    input.data_handle(), norms.data_handle(), input.extent(0), input.extent(1));
  RAFT_CUDA_TRY(cudaGetLastError());
}

}  // namespace cuvs::distance::detail
