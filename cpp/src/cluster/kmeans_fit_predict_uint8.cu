#include "kmeans_impl_fit_predict.cuh"

namespace cuvs::cluster::kmeans {

void fit_predict(raft::resources const& handle,
                 const params& params,
                 raft::device_matrix_view<const uint8_t, int64_t> X,
                 std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
                 std::optional<raft::device_matrix_view<float, int64_t>> centroids,
                 raft::device_vector_view<int64_t, int64_t> labels,
                 raft::host_scalar_view<float> inertia,
                 raft::host_scalar_view<int64_t> n_iter)
{
  cuvs::cluster::kmeans::fit_predict<float, int64_t>(
    handle, params, X, sample_weight, centroids, labels, inertia, n_iter);
}

}  // namespace cuvs::cluster::kmeans
