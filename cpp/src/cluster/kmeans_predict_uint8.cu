#include "detail/kmeans.cuh"

namespace cuvs::cluster::kmeans {

void predict(raft::resources const& handle,
             const params& params,
             raft::device_matrix_view<const uint8_t, int64_t> X,
             std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<int64_t, int64_t> labels,
             bool normalize_weight,
             raft::host_scalar_view<float> inertia)
{
  detail::kmeans_predict<float, int64_t>(
    handle, params, X, sample_weight, centroids, labels, normalize_weight, inertia);
}

}  // namespace cuvs::cluster::kmeans
