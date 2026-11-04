#include "detail/kmeans.cuh"

namespace cuvs::cluster::kmeans {

void transform(raft::resources const& handle,
               const params& params,
               raft::device_matrix_view<const int8_t, int64_t> X,
               raft::device_matrix_view<const float, int64_t> centroids,
               raft::device_matrix_view<float, int64_t> X_new)
{
  detail::kmeans_transform<float, int64_t>(handle, params, X, centroids, X_new);
}

}  // namespace cuvs::cluster::kmeans
