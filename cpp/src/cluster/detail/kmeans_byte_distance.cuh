#pragma once

#include "../../distance/detail/byte_distance.cuh"
#include "../../distance/top_1_nn.cuh"

#include <optional>
#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/unary_op.cuh>
#include <rmm/device_uvector.hpp>

namespace cuvs::cluster::kmeans::detail {

template <typename InputT, typename IndexT>
__global__ void byte_assigned_distance_kernel(const InputT* input,
                                              const float* centers,
                                              const IndexT* indices,
                                              float* distances,
                                              IndexT rows,
                                              IndexT features,
                                              bool take_sqrt)
{
  const auto stride = int64_t{gridDim.x} * (blockDim.x / 32);
  for (auto row = int64_t{blockIdx.x} * (blockDim.x / 32) + threadIdx.x / 32; row < rows;
       row += stride) {
    auto distance = cuvs::distance::detail::byte_warp_distance(
      input + row * features, centers + int64_t{indices[row]} * features, features, false);
    if (threadIdx.x % 32 == 0) { distances[row] = take_sqrt ? sqrt(distance) : distance; }
  }
}

template <typename InputT, typename IndexT>
void byte_float_assignment(raft::resources const& handle,
                           raft::device_matrix_view<const InputT, IndexT> input,
                           raft::device_matrix_view<const float, IndexT> centers,
                           IndexT* indices,
                           float* distances,
                           const float* input_norms,
                           const float* center_norms,
                           rmm::device_uvector<char>& workspace,
                           bool take_sqrt)
{
  const auto rows       = input.extent(0);
  const auto candidates = centers.extent(0);
  const auto features   = input.extent(1);
  bool launched         = false;
#if CUVS_CUTILE_ENABLED
  constexpr auto max_i32 = std::numeric_limits<int>::max();
  if (rows <= max_i32 && candidates <= max_i32 && features <= max_i32) {
    auto stream = raft::resource::get_cuda_stream(handle);
    if (workspace.size() < sizeof(int) * rows) { workspace.resize(sizeof(int) * rows, stream); }
    auto* native_indices = reinterpret_cast<int*>(workspace.data());
    launched             = cuvs::distance::detail::try_fused_1nn_mixed_tile(handle,
                                                                native_indices,
                                                                distances,
                                                                input.data_handle(),
                                                                centers.data_handle(),
                                                                input_norms,
                                                                center_norms,
                                                                static_cast<int>(rows),
                                                                static_cast<int>(candidates),
                                                                static_cast<int>(features),
                                                                take_sqrt);
    if (launched) {
      raft::linalg::unaryOp(indices, native_indices, rows, raft::cast_op<IndexT>{}, stream.get());
      auto blocks =
        static_cast<unsigned>(std::min<int64_t>(raft::ceildiv(int64_t{rows}, int64_t{8}), 65535));
      byte_assigned_distance_kernel<<<blocks, 256, 0, stream.get()>>>(
        input.data_handle(), centers.data_handle(), indices, distances, rows, features, take_sqrt);
      RAFT_CUDA_TRY(cudaGetLastError());
    }
  }
#endif
  if (!launched) {
    cuvs::distance::detail::byte_top_1(handle,
                                       input.data_handle(),
                                       centers.data_handle(),
                                       indices,
                                       distances,
                                       rows,
                                       candidates,
                                       features,
                                       take_sqrt);
  }
}

template <typename DataT, typename IndexT, typename InputT>
  requires(cuvs::distance::detail::is_byte_distance_v<InputT> && std::is_same_v<DataT, float>)
cuvs::distance::Top1nnResultView<float, IndexT> minClusterAndDistanceCompute(
  raft::resources const& handle,
  raft::device_matrix_view<const InputT, IndexT> input,
  raft::device_matrix_view<const DataT, IndexT> centers,
  rmm::device_uvector<char>& output_storage,
  raft::device_vector_view<const DataT, IndexT> input_norms,
  rmm::device_uvector<DataT>& center_norms,
  cuvs::distance::DistanceType metric,
  int batch_samples,
  int batch_centroids,
  rmm::device_uvector<char>& workspace,
  std::optional<raft::device_vector_view<const DataT, IndexT>> = std::nullopt)
{
  RAFT_EXPECTS(input.extent(1) == centers.extent(1) && centers.extent(0) > 0,
               "Byte k-means centroid shape mismatch");
  RAFT_EXPECTS(input.extent(0) >= 0 && input.extent(1) > 0,
               "Byte k-means requires nonnegative rows and positive features");
  RAFT_EXPECTS(input.extent(0) == 0 || (input.data_handle() && centers.data_handle()),
               "Byte k-means requires non-null input and centroids");
  RAFT_EXPECTS(input_norms.extent(0) >= input.extent(0), "Missing byte input norms");
  RAFT_EXPECTS(metric == cuvs::distance::DistanceType::L2Expanded ||
                 metric == cuvs::distance::DistanceType::L2SqrtExpanded,
               "Byte k-means supports L2Expanded and L2SqrtExpanded");
  auto stream = raft::resource::get_cuda_stream(handle);
  auto rows   = input.extent(0);
  cuvs::distance::detail::Top1nnPlan<IndexT> plan;
  plan.available        = true;
  plan.m                = rows;
  plan.output_layout    = cuvs::distance::detail::Top1nnOutputLayout::Separate;
  plan.output_alignment = 16;
  plan.distance_offset  = (sizeof(IndexT) * rows + 15) / 16 * 16;
  plan.output_bytes     = plan.distance_offset + sizeof(float) * rows;
  if (output_storage.size() < plan.output_bytes) {
    output_storage.resize(plan.output_bytes, stream);
  }
  auto result = cuvs::distance::bind_top_1_nn_result_view<float>(
    plan, rows, output_storage.data(), output_storage.size());
  if (rows == 0) { return result; }
  auto* indices   = reinterpret_cast<IndexT*>(output_storage.data());
  auto* distances = reinterpret_cast<float*>(output_storage.data() + plan.distance_offset);
  center_norms.resize(centers.extent(0), stream);
  raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
    handle, centers, raft::make_device_vector_view(center_norms.data(), centers.extent(0)));
  const IndexT batch_rows = batch_samples > 0 ? std::min<IndexT>(batch_samples, rows) : rows;
  for (IndexT offset = 0; offset < rows; offset += batch_rows) {
    const auto size = std::min(batch_rows, rows - offset);
    byte_float_assignment(handle,
                          raft::make_device_matrix_view(
                            input.data_handle() + offset * input.extent(1), size, input.extent(1)),
                          centers,
                          indices + offset,
                          distances + offset,
                          input_norms.data_handle() + offset,
                          center_norms.data(),
                          workspace,
                          metric == cuvs::distance::DistanceType::L2SqrtExpanded);
  }
  return result;
}

template <typename DataT, typename IndexT, typename InputT>
  requires(cuvs::distance::detail::is_byte_distance_v<InputT> && std::is_same_v<DataT, float>)
void minClusterDistanceCompute(
  raft::resources const& handle,
  raft::device_matrix_view<const InputT, IndexT> input,
  raft::device_matrix_view<DataT, IndexT> centers,
  raft::device_vector_view<DataT, IndexT> distances,
  raft::device_vector_view<const DataT, IndexT> input_norms,
  rmm::device_uvector<DataT>& center_norms,
  cuvs::distance::DistanceType metric,
  int batch_samples,
  int batch_centroids,
  rmm::device_uvector<char>& workspace,
  std::optional<raft::device_vector_view<const DataT, IndexT>> tf32_norms = std::nullopt,
  bool accurate_distances                                                 = false)
{
  rmm::device_uvector<char> output(0, raft::resource::get_cuda_stream(handle));
  auto result = minClusterAndDistanceCompute<DataT, IndexT>(handle,
                                                            input,
                                                            raft::make_const_mdspan(centers),
                                                            output,
                                                            input_norms,
                                                            center_norms,
                                                            metric,
                                                            batch_samples,
                                                            batch_centroids,
                                                            workspace,
                                                            tf32_norms);
  result.visit_native(
    [&](auto*, auto* values) {
      raft::copy(
        distances.data_handle(), values, input.extent(0), raft::resource::get_cuda_stream(handle));
    },
    [](auto*) {});
}

template <typename InputT, typename IndexT>
void byte_seed_assignment(raft::resources const& handle,
                          raft::device_matrix_view<const InputT, IndexT> input,
                          raft::device_matrix_view<const InputT, IndexT> centers,
                          raft::device_vector_view<const int64_t, IndexT> input_norms,
                          IndexT* indices,
                          float* distances,
                          rmm::device_uvector<char>& workspace)
{
  auto center_norms = raft::make_device_vector<int64_t>(handle, centers.extent(0));
  cuvs::distance::detail::byte_norm(handle, centers, center_norms.view());
  cuvs::distance::detail::Top1nnTuning tuning;
  auto plan = cuvs::distance::probe_top_1_nn(handle,
                                             input.data_handle(),
                                             centers.data_handle(),
                                             input.extent(0),
                                             centers.extent(0),
                                             input.extent(1),
                                             tuning,
                                             cuvs::distance::DistanceType::L2Expanded,
                                             cuvs::distance::detail::Top1nnBackend::Auto,
                                             indices != nullptr);
  RAFT_EXPECTS(plan.available, "Native byte top-1 is unavailable");
  auto stream = raft::resource::get_cuda_stream(handle);
  if (workspace.size() < plan.workspace_bytes) { workspace.resize(plan.workspace_bytes, stream); }
  cuvs::distance::top_1_nn(handle,
                           cuvs::distance::Top1nnOutput<IndexT, float>{indices, distances},
                           input.data_handle(),
                           centers.data_handle(),
                           input_norms.data_handle(),
                           center_norms.data_handle(),
                           input.extent(0),
                           centers.extent(0),
                           input.extent(1),
                           tuning,
                           workspace.data(),
                           workspace.size(),
                           false,
                           true,
                           true,
                           cuvs::distance::DistanceType::L2Expanded,
                           0.0f,
                           plan);
}

}  // namespace cuvs::cluster::kmeans::detail
