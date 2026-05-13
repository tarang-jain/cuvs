/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>

namespace cuvs::cluster::kmeans {

namespace {

template <typename ByteT>
void predict_byte_to_float(raft::resources const& handle,
                           const cuvs::cluster::kmeans::params& params,
                           raft::device_matrix_view<const ByteT, int64_t> X,
                           raft::device_matrix_view<const float, int64_t> centroids,
                           raft::device_vector_view<int64_t, int64_t> labels,
                           raft::host_scalar_view<float> inertia)
{
  auto n_samples  = X.extent(0);
  auto n_features = X.extent(1);

  // Convert byte data to float
  auto X_float = raft::make_device_matrix<float, int64_t>(handle, n_samples, n_features);
  raft::linalg::map(
    handle,
    raft::make_device_vector_view<const ByteT, int64_t>(X.data_handle(), n_samples * n_features),
    raft::make_device_vector_view<float, int64_t>(X_float.data_handle(), n_samples * n_features),
    raft::cast_op<float>{});

  std::optional<raft::device_vector_view<const float, int64_t>> no_weights = std::nullopt;

  cuvs::cluster::kmeans::predict(handle,
                                 params,
                                 raft::make_const_mdspan(X_float.view()),
                                 no_weights,
                                 centroids,
                                 labels,
                                 true,
                                 inertia);
}

}  // namespace

void predict(raft::resources const& handle,
             const cuvs::cluster::kmeans::params& params,
             raft::device_matrix_view<const int8_t, int64_t> X,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<int64_t, int64_t> labels,
             raft::host_scalar_view<float> inertia)
{
  predict_byte_to_float(handle, params, X, centroids, labels, inertia);
}

void predict(raft::resources const& handle,
             const cuvs::cluster::kmeans::params& params,
             raft::device_matrix_view<const uint8_t, int64_t> X,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<int64_t, int64_t> labels,
             raft::host_scalar_view<float> inertia)
{
  predict_byte_to_float(handle, params, X, centroids, labels, inertia);
}

}  // namespace cuvs::cluster::kmeans
