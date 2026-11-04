#include "detail/kmeans.cuh"
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/multi_gpu.hpp>

namespace cuvs::cluster::kmeans {

void fit(raft::resources const& handle,
         const params& params,
         raft::device_matrix_view<const uint8_t, int64_t> X,
         std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter)
{
  RAFT_EXPECTS(!raft::resource::is_multi_gpu(handle) && !raft::resource::comms_initialized(handle),
               "Native byte Lloyd k-means currently supports a single GPU");
  detail::kmeans_fit<float, int64_t>(handle, params, X, sample_weight, centroids, inertia, n_iter);
}

void fit(raft::resources const& handle,
         const params& params,
         raft::host_matrix_view<const uint8_t, int64_t> X,
         std::optional<raft::host_vector_view<const float, int64_t>> sample_weight,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter)
{
  RAFT_EXPECTS(!raft::resource::is_multi_gpu(handle) && !raft::resource::comms_initialized(handle),
               "Native byte Lloyd k-means currently supports a single GPU");
  detail::kmeans_fit<float, int64_t>(handle, params, X, sample_weight, centroids, inertia, n_iter);
}

}  // namespace cuvs::cluster::kmeans
