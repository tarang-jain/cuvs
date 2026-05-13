/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include "kmeans_impl.cuh"
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans {

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::host_matrix_view<const int8_t, int64_t> X,
         std::optional<raft::host_vector_view<const float, int64_t>> sample_weight,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter)
{
  cuvs::cluster::kmeans::fit<int8_t, float, int64_t>(
    handle, params, X, sample_weight, centroids, inertia, n_iter);
}

}  // namespace cuvs::cluster::kmeans
