/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../kmeans.cuh"
#include "kmeans_common.cuh"
#include <cuvs/cluster/kmeans.hpp>

#include "../../core/nvtx.hpp"
#include "../../distance/distance.cuh"
#include "../../neighbors/detail/ann_utils.cuh"

#include <cuvs/distance/distance.hpp>
#include <raft/core/copy.cuh>
#include <raft/core/device_setter.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_id.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/matrix_vector.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/normalize.cuh>
#include <raft/linalg/reduce_cols_by_key.cuh>
#include <raft/linalg/reduce_rows_by_key.cuh>
#include <raft/matrix/argmin.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/matrix/init.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/device_atomics.cuh>
#include <raft/util/integer_utils.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/gather.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace cuvs::cluster::kmeans::detail {

/**
 * @brief Predict labels for the dataset; floating-point types only.
 *
 * NB: no minibatch splitting is done here, it may require large amount of temporary memory (n_rows
 * * n_cluster * sizeof(MathT)).
 *
 * @tparam MathT  type of the centroids and mapped data
 * @tparam IdxT   index type
 * @tparam LabelT label type
 *
 * @param[in] handle The raft handle.
 * @param[in] params Structure containing the hyper-parameters
 * @param[in] centers Pointer to the row-major matrix of cluster centers [n_clusters, dim]
 * @param[in] n_clusters Number of clusters/centers
 * @param[in] dim Dimensionality of the data
 * @param[in] dataset Pointer to the data [n_rows, dim]
 * @param[in] dataset_norm Pointer to the precomputed norm (for L2 metrics only) [n_rows]
 * @param[in] n_rows Number samples in the `dataset`
 * @param[out] labels Output predictions [n_rows]
 * @param[inout] mr (optional) Memory resource to use for temporary allocations
 */
template <typename MathT, typename IdxT, typename LabelT>
inline std::enable_if_t<std::is_floating_point_v<MathT>> predict_core(
  const raft::resources& handle,
  const cuvs::cluster::kmeans::balanced_params& params,
  const MathT* centers,
  IdxT n_clusters,
  IdxT dim,
  const MathT* dataset,
  const MathT* dataset_norm,
  IdxT n_rows,
  LabelT* labels,
  rmm::device_async_resource_ref mr)
{
  auto stream = raft::resource::get_cuda_stream(handle);
  switch (params.metric) {
    case cuvs::distance::DistanceType::L2Expanded:
    case cuvs::distance::DistanceType::L2SqrtExpanded:
    case cuvs::distance::DistanceType::CosineExpanded: {
      rmm::device_uvector<MathT> L2NormBuf_OR_DistBuf(0, stream, mr);
      rmm::device_uvector<char> workspace(0, stream, mr);

      auto X_view = raft::make_device_matrix_view<const MathT, IdxT>(dataset, n_rows, dim);
      auto centroids_view =
        raft::make_device_matrix_view<const MathT, IdxT>(centers, n_clusters, dim);
      auto X_norm_view = raft::make_device_vector_view<const MathT, IdxT>(dataset_norm, n_rows);

      auto minClusterAndDistance = raft::make_device_mdarray<raft::KeyValuePair<IdxT, MathT>, IdxT>(
        handle, mr, raft::make_extents<IdxT>(n_rows));

      cuvs::cluster::kmeans::detail::minClusterAndDistanceCompute<MathT, IdxT>(
        handle,
        X_view,
        centroids_view,
        minClusterAndDistance.view(),
        X_norm_view,
        L2NormBuf_OR_DistBuf,
        params.metric,
        0,  // batch_samples (unused for fused reduction)
        0,  // batch_centroids (unused for fused reduction)
        workspace);

      // Copy keys to output labels
      raft::linalg::map(handle,
                        raft::make_const_mdspan(minClusterAndDistance.view()),
                        raft::make_device_vector_view<LabelT, IdxT>(labels, n_rows),
                        raft::compose_op<raft::cast_op<LabelT>, raft::key_op>());
      break;
    }
    case cuvs::distance::DistanceType::InnerProduct: {
      // TODO: pass buffer
      rmm::device_uvector<MathT> distances(n_rows * n_clusters, stream, mr);

      MathT alpha = -1.0;
      MathT beta  = 0.0;

      raft::linalg::gemm(handle,
                         true,
                         false,
                         n_clusters,
                         n_rows,
                         dim,
                         &alpha,
                         centers,
                         dim,
                         dataset,
                         dim,
                         &beta,
                         distances.data(),
                         n_clusters,
                         stream.get());

      auto distances_const_view = raft::make_device_matrix_view<const MathT, IdxT, raft::row_major>(
        distances.data(), n_rows, n_clusters);
      auto labels_view = raft::make_device_vector_view<LabelT, IdxT>(labels, n_rows);
      raft::matrix::argmin(handle, distances_const_view, labels_view);
      break;
    }
    default: {
      RAFT_FAIL("The chosen distance metric is not supported (%d)", int(params.metric));
    }
  }
}

/**
 * @brief Suggest a minibatch size for kmeans prediction.
 *
 * This function is used as a heuristic to split the work over a large dataset
 * to reduce the size of temporary memory allocations.
 *
 * @tparam MathT type of the centroids and mapped data
 * @tparam IdxT  index type
 *
 * @param[in] n_clusters number of clusters in kmeans clustering
 * @param[in] n_rows Number of samples in the dataset
 * @param[in] dim Number of features in the dataset
 * @param[in] metric Distance metric
 * @param[in] needs_conversion Whether the data needs to be converted to MathT
 * @return A suggested minibatch size and the expected memory cost per-row (in bytes)
 */
template <typename MathT, typename IdxT>
auto calc_minibatch_size(const raft::resources& handle,
                         IdxT n_clusters,
                         IdxT n_rows,
                         IdxT dim,
                         cuvs::distance::DistanceType metric,
                         bool needs_conversion) -> std::tuple<IdxT, size_t>
{
  n_clusters = std::max<IdxT>(1, n_clusters);

  // Estimate memory needs per row (i.e element of the batch).
  size_t mem_per_row = 0;
  switch (metric) {
    case distance::DistanceType::L2Expanded:
    case distance::DistanceType::L2SqrtExpanded: {
      if (use_fused<MathT, IdxT, IdxT>(handle, n_rows, n_clusters, dim)) {
        // fusedL2NN needs a mutex and a key-value pair for each row.
        mem_per_row += sizeof(int);
        mem_per_row += sizeof(raft::KeyValuePair<IdxT, MathT>);
      } else {
        // unfused path needs a full GEMM output (distance matrix row).
        mem_per_row += sizeof(MathT) * n_clusters;
      }
    } break;
    // Other metrics require storing a distance matrix.
    default: {
      mem_per_row += sizeof(MathT) * n_clusters;
    }
  }

  // If we need to convert to MathT, space required for the converted batch.
  if (!needs_conversion) { mem_per_row += sizeof(MathT) * dim; }

  // Heuristic: calculate the minibatch size in order to use at most 80% or 512MB workspace memory.
  // We go below 1GB here as the allocation is mostly done in a single chunk which
  // is problematic if e.g. a pool allocator manages its own chunks <= 1GB.
  const auto free_ws_size = raft::resource::get_workspace_free_bytes(handle);
  const auto available_ws_size =
    std::min<size_t>((free_ws_size * size_t{8}) / size_t{10}, size_t{1} << 29);

  IdxT minibatch_size = std::max<IdxT>(IdxT{1}, static_cast<IdxT>(available_ws_size / mem_per_row));

  minibatch_size = raft::round_down_safe<IdxT>(minibatch_size, IdxT{64});
  minibatch_size = std::min<IdxT>(minibatch_size, n_rows);
  return std::make_tuple(minibatch_size, mem_per_row);
}

/**
 * @brief Given the data and labels, calculate cluster centers and sizes in one sweep.
 *
 * @note all pointers must be accessible on the device.
 *
 * @tparam T          element type
 * @tparam MathT      type of the centroids and mapped data
 * @tparam IdxT       index type
 * @tparam LabelT     label type
 * @tparam CounterT   counter type supported by CUDA's native atomicAdd
 * @tparam MappingOpT type of the mapping operation
 *
 * @param[in] handle The raft handle.
 * @param[inout] centers Pointer to the output [n_clusters, dim]
 * @param[inout] cluster_sizes Number of rows in each cluster [n_clusters]
 * @param[in] n_clusters Number of clusters/centers
 * @param[in] dim Dimensionality of the data
 * @param[in] dataset Pointer to the data [n_rows, dim]
 * @param[in] n_rows Number of samples in the `dataset`
 * @param[in] labels Output predictions [n_rows]
 * @param[in] reset_counters Whether to clear the output arrays before calculating.
 *    When set to `false`, this function may be used to update existing centers and sizes using
 *    the weighted average principle.
 * @param[in] mapping_op Mapping operation from T to MathT
 * @param[inout] mr (optional) Memory resource to use for temporary allocations on the device
 */
template <typename T,
          typename MathT,
          typename IdxT,
          typename LabelT,
          typename CounterT,
          typename MappingOpT>
void calc_centers_and_sizes(const raft::resources& handle,
                            MathT* centers,
                            CounterT* cluster_sizes,
                            IdxT n_clusters,
                            IdxT dim,
                            const T* dataset,
                            IdxT n_rows,
                            const LabelT* labels,
                            bool reset_counters,
                            MappingOpT mapping_op,
                            rmm::device_async_resource_ref mr)
{
  auto stream = raft::resource::get_cuda_stream(handle);

  auto centersView      = raft::make_device_matrix_view<MathT>(centers, n_clusters, dim);
  auto clusterSizesView = raft::make_device_vector_view<const CounterT>(cluster_sizes, n_clusters);

  if (!reset_counters) {
    raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(
      handle, raft::make_const_mdspan(centersView), clusterSizesView, centersView, raft::mul_op{});
  }

  rmm::device_uvector<char> workspace(0, stream, mr);

  // If we reset the counters, we can compute directly the new sizes in cluster_sizes.
  // If we don't reset, we compute in a temporary buffer and add in a separate step.
  rmm::device_uvector<CounterT> temp_cluster_sizes(0, stream, mr);
  CounterT* temp_sizes = cluster_sizes;
  if (!reset_counters) {
    temp_cluster_sizes.resize(n_clusters, stream);
    temp_sizes = temp_cluster_sizes.data();
  }

  // Apply mapping only when the data and math types are different.
  if constexpr (std::is_same_v<T, MathT>) {
    raft::linalg::reduce_rows_by_key(dataset,
                                     dim,
                                     labels,
                                     nullptr,
                                     n_rows,
                                     dim,
                                     n_clusters,
                                     centers,
                                     stream.get(),
                                     reset_counters);
  } else {
    // todo(lsugy): use iterator from KV output of fusedL2NN
    thrust::transform_iterator<MappingOpT, const T*> mapping_itr(dataset, mapping_op);
    raft::linalg::reduce_rows_by_key(mapping_itr,
                                     dim,
                                     labels,
                                     nullptr,
                                     n_rows,
                                     dim,
                                     n_clusters,
                                     centers,
                                     stream.get(),
                                     reset_counters);
  }

  // Compute weight of each cluster
  cuvs::cluster::kmeans::detail::countLabels(
    handle, labels, temp_sizes, n_rows, n_clusters, workspace);

  // Add previous sizes if necessary
  if (!reset_counters) {
    raft::linalg::add(
      handle,
      raft::make_device_vector_view<const CounterT, IdxT>(cluster_sizes, n_clusters),
      raft::make_device_vector_view<const CounterT, IdxT>(temp_sizes, n_clusters),
      raft::make_device_vector_view<CounterT, IdxT>(cluster_sizes, n_clusters));
  }

  raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(handle,
                                                             raft::make_const_mdspan(centersView),
                                                             clusterSizesView,
                                                             centersView,
                                                             raft::div_checkzero_op{});
}

/** Computes the L2 norm of the dataset, converting to MathT if necessary */
template <typename T, typename MathT, typename IdxT, typename MappingOpT, typename FinOpT>
void compute_norm(const raft::resources& handle,
                  MathT* dataset_norm,
                  const T* dataset,
                  IdxT dim,
                  IdxT n_rows,
                  MappingOpT mapping_op,
                  FinOpT norm_fin_op,
                  std::optional<rmm::device_async_resource_ref> mr = std::nullopt)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("compute_norm");
  auto stream = raft::resource::get_cuda_stream(handle);
  rmm::device_uvector<MathT> mapped_dataset(
    0, stream, mr.value_or(raft::resource::get_workspace_resource_ref(handle)));

  const MathT* dataset_ptr = nullptr;

  if (std::is_same_v<MathT, T>) {
    dataset_ptr = reinterpret_cast<const MathT*>(dataset);
  } else {
    mapped_dataset.resize(n_rows * dim, stream);

    raft::linalg::map(
      handle,
      raft::make_device_vector_view<const T, IdxT>(dataset, n_rows * dim),
      raft::make_device_vector_view<MathT, IdxT>(mapped_dataset.data(), n_rows * dim),
      mapping_op);

    dataset_ptr = static_cast<const MathT*>(mapped_dataset.data());
  }

  raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
    handle,
    raft::make_device_matrix_view<const MathT, IdxT, raft::row_major>(dataset_ptr, n_rows, dim),
    raft::make_device_vector_view<MathT, IdxT>(dataset_norm, n_rows),
    norm_fin_op);
}

/**
 * @brief Predict labels for the dataset.
 *
 * @tparam T element type
 * @tparam MathT type of the centroids and mapped data
 * @tparam IdxT index type
 * @tparam LabelT label type
 * @tparam MappingOpT type of the mapping operation
 *
 * @param[in] handle The raft handle
 * @param[in] params Structure containing the hyper-parameters
 * @param[in] centers Pointer to the row-major matrix of cluster centers [n_clusters, dim]
 * @param[in] n_clusters Number of clusters/centers
 * @param[in] dim Dimensionality of the data
 * @param[in] dataset Pointer to the data [n_rows, dim]
 * @param[in] n_rows Number samples in the `dataset`
 * @param[out] labels Output predictions [n_rows]
 * @param[in] mapping_op Mapping operation from T to MathT
 * @param[inout] mr (optional) memory resource to use for temporary allocations
 * @param[in] dataset_norm (optional) Pre-computed norms of each row in the dataset [n_rows]
 */
template <typename T, typename MathT, typename IdxT, typename LabelT, typename MappingOpT>
void predict(const raft::resources& handle,
             const cuvs::cluster::kmeans::balanced_params& params,
             const MathT* centers,
             IdxT n_clusters,
             IdxT dim,
             const T* dataset,
             IdxT n_rows,
             LabelT* labels,
             MappingOpT mapping_op,
             std::optional<rmm::device_async_resource_ref> mr = std::nullopt,
             const MathT* dataset_norm                        = nullptr)
{
  auto stream = raft::resource::get_cuda_stream(handle);
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "predict(%zu, %u)", static_cast<size_t>(n_rows), n_clusters);
  auto mem_res = mr.value_or(raft::resource::get_workspace_resource_ref(handle));
  auto [max_minibatch_size, _mem_per_row] = calc_minibatch_size<MathT>(
    handle, n_clusters, n_rows, dim, params.metric, std::is_same_v<T, MathT>);
  rmm::device_uvector<MathT> cur_dataset(
    std::is_same_v<T, MathT> ? 0 : max_minibatch_size * dim, stream, mem_res);
  bool need_compute_norm =
    dataset_norm == nullptr && (params.metric == cuvs::distance::DistanceType::L2Expanded ||
                                params.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                                params.metric == cuvs::distance::DistanceType::CosineExpanded);
  rmm::device_uvector<MathT> cur_dataset_norm(
    need_compute_norm ? max_minibatch_size : 0, stream, mem_res);
  const MathT* dataset_norm_ptr = nullptr;
  auto cur_dataset_ptr          = cur_dataset.data();
  for (IdxT offset = 0; offset < n_rows; offset += max_minibatch_size) {
    IdxT minibatch_size = std::min<IdxT>(max_minibatch_size, n_rows - offset);

    if constexpr (std::is_same_v<T, MathT>) {
      cur_dataset_ptr = const_cast<MathT*>(dataset + offset * dim);
    } else {
      raft::linalg::map(
        handle,
        raft::make_device_vector_view<const T, IdxT>(dataset + offset * dim, minibatch_size * dim),
        raft::make_device_vector_view<MathT, IdxT>(cur_dataset_ptr, minibatch_size * dim),
        mapping_op);
    }

    // Compute the norm now if it hasn't been pre-computed.
    if (need_compute_norm) {
      if (params.metric == cuvs::distance::DistanceType::CosineExpanded)
        compute_norm(handle,
                     cur_dataset_norm.data(),
                     cur_dataset_ptr,
                     dim,
                     minibatch_size,
                     mapping_op,
                     raft::sqrt_op{},
                     mr);
      else
        compute_norm(handle,
                     cur_dataset_norm.data(),
                     cur_dataset_ptr,
                     dim,
                     minibatch_size,
                     mapping_op,
                     raft::identity_op{},
                     mr);
      dataset_norm_ptr = cur_dataset_norm.data();
    } else if (dataset_norm != nullptr) {
      dataset_norm_ptr = dataset_norm + offset;
    }

    predict_core(handle,
                 params,
                 centers,
                 n_clusters,
                 dim,
                 cur_dataset_ptr,
                 dataset_norm_ptr,
                 minibatch_size,
                 labels + offset,
                 mem_res);
  }
}

template <uint32_t BlockDimY,
          typename T,
          typename MathT,
          typename IdxT,
          typename LabelT,
          typename CounterT,
          typename MappingOpT>
__launch_bounds__((raft::WarpSize * BlockDimY)) RAFT_KERNEL
  adjust_centers_random_donor_kernel(MathT* centers,  // [n_clusters, dim]
                                     IdxT n_clusters,
                                     IdxT dim,
                                     const T* dataset,  // [n_rows, dim]
                                     IdxT n_rows,
                                     const LabelT* labels,           // [n_rows]
                                     const CounterT* cluster_sizes,  // [n_clusters]
                                     MathT lower_threshold,
                                     IdxT average,
                                     MathT centroid_offset,
                                     IdxT seed,
                                     IdxT* search_count,
                                     IdxT* update_count,
                                     MappingOpT mapping_op)
{
  IdxT receiver_cluster = threadIdx.y + BlockDimY * static_cast<IdxT>(blockIdx.x);
  if (receiver_cluster >= n_clusters) return;
  auto receiver_size = static_cast<IdxT>(cluster_sizes[receiver_cluster]);
  if (static_cast<MathT>(receiver_size) >= lower_threshold) return;

  IdxT i = n_rows;
  IdxT j = raft::laneId();
  if (j == 0) {
    IdxT attempt = 0;
    do {
      auto old = atomicAdd(search_count, IdxT{1});
      auto candidate =
        static_cast<IdxT>((static_cast<int64_t>(seed) * static_cast<int64_t>(old + 1)) %
                          static_cast<int64_t>(n_rows));
      if (static_cast<IdxT>(cluster_sizes[labels[candidate]]) >= average) { i = candidate; }
      ++attempt;
    } while (i >= n_rows && attempt < n_rows);
  }
  i = raft::shfl(i, 0);
  if (i >= n_rows) return;

  auto donor_cluster = static_cast<IdxT>(labels[i]);
  if (j == 0) { atomicAdd(update_count, IdxT{1}); }

  for (; j < dim; j += raft::WarpSize) {
    auto donor_center = centers[j + dim * donor_cluster];
    auto donor_point  = mapping_op(dataset[j + dim * i]);
    auto val          = donor_center + centroid_offset * (donor_point - donor_center);
    centers[j + dim * receiver_cluster] = val;
  }
}

template <uint32_t BlockDimY,
          typename T,
          typename MathT,
          typename IdxT,
          typename LabelT,
          typename MappingOpT>
__launch_bounds__((raft::WarpSize * BlockDimY)) RAFT_KERNEL
  adjust_centers_kernel(MathT* centers,  // [n_clusters, dim]
                        IdxT n_pairs,
                        IdxT dim,
                        const T* dataset,  // [n_rows, dim]
                        IdxT n_rows,
                        const LabelT* labels,  // [n_rows]
                        const IdxT* receiver_clusters,
                        const IdxT* donor_clusters,
                        MathT centroid_offset,
                        IdxT seed,
                        IdxT* update_count,
                        MappingOpT mapping_op)
{
  IdxT pair_id = threadIdx.y + BlockDimY * static_cast<IdxT>(blockIdx.x);
  if (pair_id >= n_pairs) return;

  auto receiver_cluster = receiver_clusters[pair_id];
  auto donor_cluster    = donor_clusters[pair_id];
  IdxT i                = n_rows;
  IdxT j                = raft::laneId();
  for (IdxT attempt = 0; attempt < n_rows; attempt += raft::WarpSize) {
    auto candidate =
      static_cast<IdxT>((static_cast<int64_t>(seed) * static_cast<int64_t>(attempt + j + 1) +
                         static_cast<int64_t>(pair_id)) %
                        static_cast<int64_t>(n_rows));
    auto found = static_cast<IdxT>(labels[candidate]) == donor_cluster;
    auto mask  = __ballot_sync(raft::warp_full_mask(), found);
    if (mask != 0) {
      auto source_lane = __ffs(mask) - 1;
      i                = raft::shfl(found ? candidate : n_rows, source_lane);
      if (j == source_lane) { atomicAdd(update_count, IdxT{1}); }
      break;
    }
  }
  if (i >= n_rows) return;

  // Reinitialize the small cluster close to the large cluster centroid, with a small offset towards
  // a random donor point so it can split the large partition in the next prediction step.
  for (; j < dim; j += raft::WarpSize) {
    auto donor_center = centers[j + dim * donor_cluster];
    auto donor_point  = mapping_op(dataset[j + dim * i]);
    auto val          = donor_center + centroid_offset * (donor_point - donor_center);
    centers[j + dim * receiver_cluster] = val;
  }
}

/**
 * @brief Adjust centers for clusters that have small number of entries.
 *
 * With SizeSorted donor selection, cluster sizes are sorted, then the smallest clusters are paired
 * with the largest clusters. For each pair where the small cluster is underfull or the large
 * cluster is overfull, the small cluster center is moved towards a data point from the large
 * cluster.
 *
 * With Random donor selection, underfull clusters are reinitialized from random data points whose
 * current cluster size is at least the average cluster size. This matches the historical
 * rebalancing behavior used by IVF-PQ, but the upper balance threshold does not control donor
 * selection in this mode.
 *
 * NB: if this function returns `true`, you should update the labels.
 *
 * NB: all pointers must be on the device side.
 *
 * @tparam T element type
 * @tparam MathT type of the centroids and mapped data
 * @tparam IdxT index type
 * @tparam LabelT label type
 * @tparam CounterT counter type supported by CUDA's native atomicAdd
 * @tparam MappingOpT type of the mapping operation
 *
 * @param[in] handle The raft handle
 * @param[inout] centers cluster centers [n_clusters, dim]
 * @param[in] n_clusters number of rows in `centers`
 * @param[in] dim number of columns in `centers` and `dataset`
 * @param[in] dataset a host pointer to the row-major data matrix [n_rows, dim]
 * @param[in] n_rows number of rows in `dataset`
 * @param[in] labels a host pointer to the cluster indices [n_rows]
 * @param[in] cluster_sizes number of rows in each cluster [n_clusters]
 * @param[in] balance_lower_tolerance defines the underfull cluster criterion:
 *                   min_cluster_size < average_size * balance_lower_tolerance
 *                   0 < balance_lower_tolerance < 1
 * @param[in] balance_upper_tolerance defines the overfull donor cluster criterion:
 *                   max_cluster_size > average_size * balance_upper_tolerance
 *                   balance_upper_tolerance > 1
 * @param[in] centroid_offset offset from the donor cluster centroid towards a donor point
 * @param[in] mapping_op Mapping operation from T to MathT
 * @param[inout] device_memory  memory resource to use for temporary allocations
 *
 * @return whether any of the centers has been updated (and thus, `labels` need to be recalculated).
 */
template <typename T,
          typename MathT,
          typename IdxT,
          typename LabelT,
          typename CounterT,
          typename MappingOpT>
auto adjust_centers(const raft::resources& handle,
                    MathT* centers,
                    IdxT n_clusters,
                    IdxT dim,
                    const T* dataset,
                    IdxT n_rows,
                    const LabelT* labels,
                    const CounterT* cluster_sizes,
                    MathT balance_lower_tolerance,
                    MathT balance_upper_tolerance,
                    MathT centroid_offset,
                    cuvs::cluster::kmeans::balanced_donor_selection donor_selection,
                    MappingOpT mapping_op,
                    rmm::device_async_resource_ref device_memory) -> bool
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "adjust_centers(%zu, %u)", static_cast<size_t>(n_rows), n_clusters);
  if (n_clusters == 0) { return false; }
  auto stream = raft::resource::get_cuda_stream(handle);
  constexpr static std::array kPrimes{29,   71,   113,  173,  229,  281,  349,  409,  463,  541,
                                      601,  659,  733,  809,  863,  941,  1013, 1069, 1151, 1223,
                                      1291, 1373, 1451, 1511, 1583, 1657, 1733, 1811, 1889, 1987,
                                      2053, 2129, 2213, 2287, 2357, 2423, 2531, 2617, 2687, 2741};
  static IdxT i_primes = 0;

  auto average         = static_cast<MathT>(n_rows) / static_cast<MathT>(n_clusters);
  auto lower_threshold = average * balance_lower_tolerance;
  auto upper_threshold = average * balance_upper_tolerance;
  std::vector<CounterT> host_cluster_sizes(n_clusters);
  raft::update_host(host_cluster_sizes.data(), cluster_sizes, n_clusters, stream);
  raft::resource::sync_stream(handle, stream);

  std::vector<std::pair<CounterT, IdxT>> sorted_clusters;
  sorted_clusters.reserve(n_clusters);
  for (IdxT cluster = 0; cluster < n_clusters; ++cluster) {
    sorted_clusters.emplace_back(host_cluster_sizes[cluster], cluster);
  }
  std::sort(sorted_clusters.begin(), sorted_clusters.end());

  std::vector<IdxT> host_receiver_clusters;
  std::vector<IdxT> host_donor_clusters;
  host_receiver_clusters.reserve(n_clusters / 2);
  host_donor_clusters.reserve(n_clusters / 2);
  for (IdxT pair_id = 0; pair_id < n_clusters / 2; ++pair_id) {
    auto const& [small_size, small_cluster] = sorted_clusters[pair_id];
    auto const& [large_size, large_cluster] = sorted_clusters[n_clusters - 1 - pair_id];
    if (small_cluster == large_cluster) { break; }
    if (large_size == 0) { break; }
    if (static_cast<MathT>(small_size) >= lower_threshold &&
        static_cast<MathT>(large_size) <= upper_threshold) {
      break;
    }
    host_receiver_clusters.push_back(small_cluster);
    host_donor_clusters.push_back(large_cluster);
  }
  auto n_pairs = static_cast<IdxT>(host_receiver_clusters.size());
  if (n_pairs == 0) { return false; }

  IdxT ofst;
  do {
    i_primes = (i_primes + 1) % kPrimes.size();
    ofst     = kPrimes[i_primes];
  } while (n_rows % ofst == 0);

  rmm::device_uvector<IdxT> receiver_clusters(n_pairs, stream, device_memory);
  rmm::device_uvector<IdxT> donor_clusters(n_pairs, stream, device_memory);
  constexpr uint32_t kBlockDimY = 4;
  const dim3 block_dim(raft::WarpSize, kBlockDimY, 1);
  rmm::device_scalar<IdxT> update_count(stream, device_memory);
  update_count.set_value_to_zero_async(stream);

  if (donor_selection == cuvs::cluster::kmeans::balanced_donor_selection::Random) {
    rmm::device_scalar<IdxT> search_count(stream, device_memory);
    search_count.set_value_to_zero_async(stream);
    const dim3 grid_dim(raft::ceildiv(n_clusters, static_cast<IdxT>(kBlockDimY)), 1, 1);
    adjust_centers_random_donor_kernel<kBlockDimY>
      <<<grid_dim, block_dim, 0, stream.get()>>>(centers,
                                                 n_clusters,
                                                 dim,
                                                 dataset,
                                                 n_rows,
                                                 labels,
                                                 cluster_sizes,
                                                 lower_threshold,
                                                 static_cast<IdxT>(n_rows / n_clusters),
                                                 centroid_offset,
                                                 ofst,
                                                 search_count.data(),
                                                 update_count.data(),
                                                 mapping_op);
    return update_count.value(stream) > 0;  // NB: rmm scalar performs the sync
  }

  raft::update_device(receiver_clusters.data(), host_receiver_clusters.data(), n_pairs, stream);
  raft::update_device(donor_clusters.data(), host_donor_clusters.data(), n_pairs, stream);
  const dim3 grid_dim(raft::ceildiv(n_pairs, static_cast<IdxT>(kBlockDimY)), 1, 1);
  adjust_centers_kernel<kBlockDimY>
    <<<grid_dim, block_dim, 0, stream.get()>>>(centers,
                                               n_pairs,
                                               dim,
                                               dataset,
                                               n_rows,
                                               labels,
                                               receiver_clusters.data(),
                                               donor_clusters.data(),
                                               centroid_offset,
                                               ofst,
                                               update_count.data(),
                                               mapping_op);
  auto n_updates = update_count.value(stream);  // NB: rmm scalar performs the sync
  RAFT_EXPECTS(n_updates == n_pairs, "Balanced k-means failed to update all adjusted centers");
  return n_updates > 0;
}

/**
 * @brief Run the control loop shared by in-core and streamed balanced k-means.
 *
 * The data-residency-specific callbacks own donor selection and batch processing. Iteration
 * accounting, conditional center normalization, and balancing pullback are kept here so the
 * device and host paths cannot diverge algorithmically.
 */
template <typename MathT, typename IdxT, typename AdjustCentersOp, typename ProcessBatchesOp>
void balancing_em_iters(const raft::resources& handle,
                        const cuvs::cluster::kmeans::balanced_params& params,
                        uint32_t n_iters,
                        IdxT n_clusters,
                        IdxT dim,
                        MathT* cluster_centers,
                        uint32_t balancing_pullback,
                        MathT balance_lower_tolerance,
                        MathT balance_upper_tolerance,
                        AdjustCentersOp&& adjust_centers_op,
                        ProcessBatchesOp&& process_batches_op)
{
  RAFT_EXPECTS(balance_lower_tolerance > MathT{0} && balance_lower_tolerance < MathT{1},
               "Balanced k-means lower balance tolerance must be in the range (0, 1)");
  RAFT_EXPECTS(balance_upper_tolerance > MathT{1},
               "Balanced k-means upper balance tolerance must be greater than 1");
  RAFT_EXPECTS(params.centroid_offset > 0.0f && params.centroid_offset <= 1.0f,
               "Balanced k-means centroid offset must be in the range (0, 1]");

  uint32_t balancing_counter = balancing_pullback;
  for (uint32_t iter = 0; iter < n_iters; ++iter) {
    if (iter > 0 && adjust_centers_op()) {
      if (balancing_counter++ >= balancing_pullback) {
        balancing_counter -= balancing_pullback;
        ++n_iters;
      }
    }

    switch (params.metric) {
      case cuvs::distance::DistanceType::InnerProduct:
      case cuvs::distance::DistanceType::CosineExpanded:
      case cuvs::distance::DistanceType::CorrelationExpanded: {
        auto clusters_in_view = raft::make_device_matrix_view<const MathT, IdxT, raft::row_major>(
          cluster_centers, n_clusters, dim);
        auto clusters_out_view = raft::make_device_matrix_view<MathT, IdxT, raft::row_major>(
          cluster_centers, n_clusters, dim);
        raft::linalg::row_normalize<raft::linalg::L2Norm>(
          handle, clusters_in_view, clusters_out_view);
        break;
      }
      default: break;
    }

    process_batches_op();
  }
}

/**
 * @brief Expectation-maximization-balancing combined in an iterative process.
 *
 * Note, the `cluster_centers` is assumed to be already initialized here.
 * Thus, this function can be used for fine-tuning existing clusters;
 * to train from scratch, use `build_clusters` function below.
 *
 * @tparam T      element type
 * @tparam MathT  type of the centroids and mapped data
 * @tparam IdxT   index type
 * @tparam LabelT label type
 * @tparam CounterT counter type supported by CUDA's native atomicAdd
 * @tparam MappingOpT type of the mapping operation
 *
 * @param[in] handle The raft handle
 * @param[in] params Structure containing the hyper-parameters
 * @param[in] n_iters Requested number of iterations (can differ from params.n_iter!)
 * @param[in] dim Dimensionality of the dataset
 * @param[in] dataset Pointer to a managed row-major array [n_rows, dim]
 * @param[in] dataset_norm Pointer to the precomputed norm (for L2 metrics only) [n_rows]
 * @param[in] n_rows Number of rows in the dataset
 * @param[in] n_cluster Requested number of clusters
 * @param[inout] cluster_centers Pointer to a managed row-major array [n_clusters, dim]
 * @param[out] cluster_labels Pointer to a managed row-major array [n_rows]
 * @param[out] cluster_sizes Pointer to a managed row-major array [n_clusters]
 * @param[in] balancing_pullback
 *   if the cluster centers are rebalanced on this number of iterations,
 *   one extra iteration is performed (this could happen several times) (default should be `2`).
 *   In other words, the first and then every `ballancing_pullback`-th rebalancing operation adds
 *   one more iteration to the main cycle.
 * @param[in] balance_lower_tolerance
 *   Small clusters are rebalanced when their paired small cluster is smaller than
 *   `avg_size * balance_lower_tolerance`.
 * @param[in] balance_upper_tolerance
 *   If the paired large cluster is larger than `avg_size * balance_upper_tolerance`, the small
 *   cluster is rebalanced towards it.
 * @param[in] mapping_op Mapping operation from T to MathT
 * @param[inout] device_memory
 *   A memory resource for device allocations (makes sense to provide a memory pool here)
 */
template <typename T,
          typename MathT,
          typename IdxT,
          typename LabelT,
          typename CounterT,
          typename MappingOpT>
void balancing_em_iters(const raft::resources& handle,
                        const cuvs::cluster::kmeans::balanced_params& params,
                        uint32_t n_iters,
                        IdxT dim,
                        const T* dataset,
                        const MathT* dataset_norm,
                        IdxT n_rows,
                        IdxT n_clusters,
                        MathT* cluster_centers,
                        LabelT* cluster_labels,
                        CounterT* cluster_sizes,
                        uint32_t balancing_pullback,
                        MathT balance_lower_tolerance,
                        MathT balance_upper_tolerance,
                        MappingOpT mapping_op,
                        rmm::device_async_resource_ref device_memory)
{
  auto adjust_centers_op = [&] {
    return adjust_centers(handle,
                          cluster_centers,
                          n_clusters,
                          dim,
                          dataset,
                          n_rows,
                          cluster_labels,
                          cluster_sizes,
                          balance_lower_tolerance,
                          balance_upper_tolerance,
                          static_cast<MathT>(params.centroid_offset),
                          params.donor_selection,
                          mapping_op,
                          device_memory);
  };
  auto process_batches_op = [&] {
    predict(handle,
            params,
            cluster_centers,
            n_clusters,
            dim,
            dataset,
            n_rows,
            cluster_labels,
            mapping_op,
            device_memory,
            dataset_norm);
    calc_centers_and_sizes(handle,
                           cluster_centers,
                           cluster_sizes,
                           n_clusters,
                           dim,
                           dataset,
                           n_rows,
                           cluster_labels,
                           true,
                           mapping_op,
                           device_memory);
  };

  balancing_em_iters(handle,
                     params,
                     n_iters,
                     n_clusters,
                     dim,
                     cluster_centers,
                     balancing_pullback,
                     balance_lower_tolerance,
                     balance_upper_tolerance,
                     adjust_centers_op,
                     process_batches_op);
}

/** Randomly initialize cluster centers and then call `balancing_em_iters`. */
template <typename T,
          typename MathT,
          typename IdxT,
          typename LabelT,
          typename CounterT,
          typename MappingOpT>
void build_clusters(const raft::resources& handle,
                    const cuvs::cluster::kmeans::balanced_params& params,
                    IdxT dim,
                    const T* dataset,
                    IdxT n_rows,
                    IdxT n_clusters,
                    MathT* cluster_centers,
                    LabelT* cluster_labels,
                    CounterT* cluster_sizes,
                    MappingOpT mapping_op,
                    rmm::device_async_resource_ref device_memory,
                    const MathT* dataset_norm = nullptr)
{
  auto stream = raft::resource::get_cuda_stream(handle);
  // "randomly" initialize labels
  auto labels_view = raft::make_device_vector_view<LabelT, IdxT>(cluster_labels, n_rows);
  raft::linalg::map_offset(
    handle,
    labels_view,
    raft::compose_op(raft::cast_op<LabelT>(), raft::mod_const_op<IdxT>(n_clusters)));

  // update centers to match the initialized labels.
  calc_centers_and_sizes(handle,
                         cluster_centers,
                         cluster_sizes,
                         n_clusters,
                         dim,
                         dataset,
                         n_rows,
                         cluster_labels,
                         true,
                         mapping_op,
                         device_memory);

  // run EM
  balancing_em_iters(handle,
                     params,
                     params.n_iters,
                     dim,
                     dataset,
                     dataset_norm,
                     n_rows,
                     n_clusters,
                     cluster_centers,
                     cluster_labels,
                     cluster_sizes,
                     2,
                     static_cast<MathT>(params.balance_lower_tolerance),
                     static_cast<MathT>(params.balance_upper_tolerance),
                     mapping_op,
                     device_memory);
}

/** Calculate how many fine clusters should belong to each mesocluster. */
template <typename IdxT, typename CounterT>
inline auto arrange_fine_clusters(IdxT n_clusters,
                                  IdxT n_mesoclusters,
                                  IdxT n_rows,
                                  const CounterT* mesocluster_sizes)
{
  std::vector<IdxT> fine_clusters_nums(n_mesoclusters);
  std::vector<IdxT> fine_clusters_csum(n_mesoclusters + 1);
  fine_clusters_csum[0] = 0;

  IdxT n_lists_rem       = n_clusters;
  IdxT n_nonempty_ms_rem = 0;
  for (IdxT i = 0; i < n_mesoclusters; i++) {
    n_nonempty_ms_rem += mesocluster_sizes[i] > CounterT{0} ? 1 : 0;
  }
  IdxT n_rows_rem               = n_rows;
  CounterT mesocluster_size_sum = 0;
  CounterT mesocluster_size_max = 0;
  IdxT fine_clusters_nums_max   = 0;
  for (IdxT i = 0; i < n_mesoclusters; i++) {
    if (i < n_mesoclusters - 1) {
      // Although the algorithm is meant to produce balanced clusters, when something
      // goes wrong, we may get empty clusters (e.g. during development/debugging).
      // The code below ensures a proportional arrangement of fine cluster numbers
      // per mesocluster, even if some clusters are empty.
      if (mesocluster_sizes[i] == 0) {
        fine_clusters_nums[i] = 0;
      } else {
        n_nonempty_ms_rem--;
        auto s = static_cast<IdxT>(
          static_cast<double>(n_lists_rem * mesocluster_sizes[i]) / n_rows_rem + .5);
        s                     = std::min<IdxT>(s, n_lists_rem - n_nonempty_ms_rem);
        fine_clusters_nums[i] = std::max(s, IdxT{1});
      }
    } else {
      fine_clusters_nums[i] = n_lists_rem;
    }
    n_lists_rem -= fine_clusters_nums[i];
    n_rows_rem -= mesocluster_sizes[i];
    mesocluster_size_max = max(mesocluster_size_max, mesocluster_sizes[i]);
    mesocluster_size_sum += mesocluster_sizes[i];
    fine_clusters_nums_max    = max(fine_clusters_nums_max, fine_clusters_nums[i]);
    fine_clusters_csum[i + 1] = fine_clusters_csum[i] + fine_clusters_nums[i];
  }

  RAFT_EXPECTS(static_cast<IdxT>(mesocluster_size_sum) == n_rows,
               "mesocluster sizes do not add up (%zu) to the total trainset size (%zu)",
               static_cast<size_t>(mesocluster_size_sum),
               static_cast<size_t>(n_rows));
  RAFT_EXPECTS(fine_clusters_csum[n_mesoclusters] == n_clusters,
               "fine cluster numbers do not add up (%zu) to the total number of clusters (%zu)",
               static_cast<size_t>(fine_clusters_csum[n_mesoclusters]),
               static_cast<size_t>(n_clusters));

  return std::make_tuple(static_cast<IdxT>(mesocluster_size_max),
                         fine_clusters_nums_max,
                         std::move(fine_clusters_nums),
                         std::move(fine_clusters_csum));
}

/**
 *  Given the (coarse) mesoclusters and the distribution of fine clusters within them,
 *  build the fine clusters.
 *
 *  Processing one mesocluster at a time:
 *   1. Copy mesocluster data into a separate buffer
 *   2. Predict fine cluster
 *   3. Refince the fine cluster centers
 *
 *  As a result, the fine clusters are what is returned by `build_hierarchical`;
 *  this function returns the total number of fine clusters, which can be checked to be
 *  the same as the requested number of clusters.
 *
 *  Note: this function uses at most `fine_clusters_nums_max` points per mesocluster for training;
 *  if one of the clusters is larger than that (as given by `mesocluster_sizes`), the extra data
 *  is ignored.
 */
template <typename T,
          typename MathT,
          typename IdxT,
          typename LabelT,
          typename CounterT,
          typename MappingOpT>
auto build_fine_clusters(const raft::resources& handle,
                         const cuvs::cluster::kmeans::balanced_params& params,
                         IdxT dim,
                         const T* dataset_mptr,
                         const MathT* dataset_norm_mptr,
                         const LabelT* labels_mptr,
                         IdxT n_rows,
                         const IdxT* fine_clusters_nums,
                         const IdxT* fine_clusters_csum,
                         const CounterT* mesocluster_sizes,
                         IdxT n_mesoclusters,
                         IdxT mesocluster_size_max,
                         IdxT fine_clusters_nums_max,
                         MathT* cluster_centers,
                         MappingOpT mapping_op,
                         rmm::device_async_resource_ref managed_memory,
                         rmm::device_async_resource_ref device_memory) -> IdxT
{
  auto stream = raft::resource::get_cuda_stream(handle);
  rmm::device_uvector<IdxT> mc_trainset_ids_buf(mesocluster_size_max, stream, managed_memory);
  // for small cluster counts the maximum mesocluster size is proportional to the number of rows, so
  // we use large workspace
  auto large_ws = raft::resource::get_large_workspace_resource_ref(handle);
  rmm::device_uvector<MathT> mc_trainset_buf(mesocluster_size_max * dim, stream, large_ws);
  rmm::device_uvector<MathT> mc_trainset_norm_buf(mesocluster_size_max, stream, device_memory);
  auto mc_trainset_ids  = mc_trainset_ids_buf.data();
  auto mc_trainset      = mc_trainset_buf.data();
  auto mc_trainset_norm = mc_trainset_norm_buf.data();

  // label (cluster ID) of each vector
  rmm::device_uvector<LabelT> mc_trainset_labels(mesocluster_size_max, stream, device_memory);

  rmm::device_uvector<MathT> mc_trainset_ccenters(
    fine_clusters_nums_max * dim, stream, device_memory);
  // number of vectors in each cluster
  rmm::device_uvector<CounterT> mc_trainset_csizes_tmp(
    fine_clusters_nums_max, stream, device_memory);

  // Training clusters in each meso-cluster
  IdxT n_clusters_done = 0;
  for (IdxT i = 0; i < n_mesoclusters; i++) {
    IdxT k = 0;
    for (IdxT j = 0; j < n_rows && k < mesocluster_size_max; j++) {
      if (labels_mptr[j] == LabelT(i)) { mc_trainset_ids[k++] = j; }
    }
    if (k != static_cast<IdxT>(mesocluster_sizes[i]))
      RAFT_LOG_DEBUG("Incorrect mesocluster size at %d. %zu vs %zu",
                     static_cast<int>(i),
                     static_cast<size_t>(k),
                     static_cast<size_t>(mesocluster_sizes[i]));
    if (k == 0) {
      RAFT_LOG_DEBUG("Empty cluster %d", i);
      RAFT_EXPECTS(fine_clusters_nums[i] == 0,
                   "Number of fine clusters must be zero for the empty mesocluster (got %d)",
                   static_cast<int>(fine_clusters_nums[i]));
      continue;
    } else {
      RAFT_EXPECTS(fine_clusters_nums[i] > 0,
                   "Number of fine clusters must be non-zero for a non-empty mesocluster");
    }

    thrust::transform_iterator<MappingOpT, const T*> mapping_itr(dataset_mptr, mapping_op);
    raft::matrix::gather(mapping_itr, dim, n_rows, mc_trainset_ids, k, mc_trainset, stream.get());
    if (params.metric == cuvs::distance::DistanceType::L2Expanded ||
        params.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
        params.metric == cuvs::distance::DistanceType::CosineExpanded) {
      thrust::gather(raft::resource::get_thrust_policy(handle),
                     mc_trainset_ids,
                     mc_trainset_ids + k,
                     dataset_norm_mptr,
                     mc_trainset_norm);
    }

    build_clusters(handle,
                   params,
                   dim,
                   mc_trainset,
                   k,
                   fine_clusters_nums[i],
                   mc_trainset_ccenters.data(),
                   mc_trainset_labels.data(),
                   mc_trainset_csizes_tmp.data(),
                   mapping_op,
                   device_memory,
                   mc_trainset_norm);

    raft::copy(handle,
               raft::make_device_vector_view(cluster_centers + (dim * fine_clusters_csum[i]),
                                             fine_clusters_nums[i] * dim),
               raft::make_device_vector_view<const MathT>(mc_trainset_ccenters.data(),
                                                          fine_clusters_nums[i] * dim));
    raft::resource::sync_stream(handle, stream);
    n_clusters_done += fine_clusters_nums[i];
  }
  return n_clusters_done;
}

}  // namespace  cuvs::cluster::kmeans::detail

namespace cuvs::cluster::kmeans::detail {
namespace host_balanced {

using index_type   = int64_t;
using label_type   = uint32_t;
using counter_type = unsigned long long;
using kvp_type     = raft::KeyValuePair<index_type, float>;

// RAFT does not expose an event abstraction. These events are the narrow CUDA-runtime exception
// needed to hand double-buffer slots between independent nonblocking streams without synchronizing
// either stream and losing H2D/compute overlap.
class cuda_event {
 public:
  cuda_event() { RAFT_CUDA_TRY(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming)); }
  ~cuda_event() noexcept
  {
    if (event_ != nullptr) { RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(event_)); }
  }
  cuda_event(cuda_event const&)            = delete;
  cuda_event& operator=(cuda_event const&) = delete;
  cuda_event(cuda_event&&)                 = delete;
  cuda_event& operator=(cuda_event&&)      = delete;
  [[nodiscard]] auto get() const noexcept -> cudaEvent_t { return event_; }

 private:
  cudaEvent_t event_{};
};

struct prediction_scratch {
  prediction_scratch(const raft::resources& handle, index_type batch_rows)
    : labels(batch_rows,
             raft::resource::get_cuda_stream(handle),
             raft::resource::get_workspace_resource_ref(handle)),
      nearest(batch_rows,
              raft::resource::get_cuda_stream(handle),
              raft::resource::get_workspace_resource_ref(handle)),
      norms(batch_rows,
            raft::resource::get_cuda_stream(handle),
            raft::resource::get_workspace_resource_ref(handle)),
      l2_norm_or_distance(0,
                          raft::resource::get_cuda_stream(handle),
                          raft::resource::get_workspace_resource_ref(handle)),
      workspace(0,
                raft::resource::get_cuda_stream(handle),
                raft::resource::get_workspace_resource_ref(handle))
  {
  }

  rmm::device_uvector<label_type> labels;
  rmm::device_uvector<kvp_type> nearest;
  rmm::device_uvector<float> norms;
  rmm::device_uvector<float> l2_norm_or_distance;
  rmm::device_uvector<char> workspace;
};

inline void predict_batch(const raft::resources& handle,
                          const cuvs::cluster::kmeans::balanced_params& params,
                          const float* centers,
                          index_type n_clusters,
                          index_type dim,
                          const float* data,
                          index_type n_rows,
                          const float* norms,
                          prediction_scratch& scratch)
{
  auto X = raft::make_device_matrix_view<const float, index_type>(data, n_rows, dim);
  auto C = raft::make_device_matrix_view<const float, index_type>(centers, n_clusters, dim);
  auto N = raft::make_device_vector_view<const float, index_type>(norms, n_rows);
  auto K = raft::make_device_vector_view<kvp_type, index_type>(scratch.nearest.data(), n_rows);
  cuvs::cluster::kmeans::detail::minClusterAndDistanceCompute<float, index_type>(
    handle, X, C, K, N, scratch.l2_norm_or_distance, params.metric, 0, 0, scratch.workspace);
  raft::linalg::map(
    handle,
    raft::make_const_mdspan(K),
    raft::make_device_vector_view<label_type, index_type>(scratch.labels.data(), n_rows),
    raft::compose_op<raft::cast_op<label_type>, raft::key_op>());
}

inline auto batch_rows(const cuvs::cluster::kmeans::balanced_params& params, index_type n_rows)
  -> index_type
{
  if (params.device_buffer_samples <= 0 || params.device_buffer_samples > n_rows) { return n_rows; }
  return static_cast<index_type>(params.device_buffer_samples);
}

inline void compute_norms(const raft::resources& handle,
                          raft::host_matrix_view<const float, index_type> X,
                          index_type rows_per_batch,
                          float* host_norms)
{
  auto stream = raft::resource::get_cuda_stream(handle);
  auto mr     = raft::resource::get_workspace_resource_ref(handle);
  rmm::cuda_stream copy_stream(rmm::cuda_stream::flags::non_blocking);
  auto batches = cuvs::spatial::knn::detail::utils::batch_load_iterator(
    handle, X, rows_per_batch, copy_stream.view(), mr, true);
  rmm::device_uvector<float> device_norms(rows_per_batch, stream, mr);

  batches.prefetch_next_batch();
  for (auto const& batch : batches) {
    auto n_rows = static_cast<index_type>(batch.size());
    auto data =
      raft::make_device_matrix_view<const float, index_type>(batch.data(), n_rows, X.extent(1));
    auto norms = raft::make_device_vector_view<float, index_type>(device_norms.data(), n_rows);
    raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(handle, data, norms);
    raft::update_host(host_norms + batch.offset(), device_norms.data(), batch.size(), stream);
    batches.prefetch_next_batch();
  }
  raft::resource::sync_stream(handle, stream);
  raft::resource::sync_stream(handle, copy_stream.view());
}

struct adjustment_scratch {
  adjustment_scratch(const raft::resources& handle, index_type n_clusters, index_type dim)
    : host_points(raft::make_pinned_vector<float, index_type>(handle, (n_clusters / 2) * dim)),
      sorted(static_cast<size_t>(n_clusters)),
      receivers(static_cast<size_t>(n_clusters / 2)),
      donors(static_cast<size_t>(n_clusters / 2)),
      donor_rows(static_cast<size_t>(n_clusters / 2)),
      donor_by_receiver(static_cast<size_t>(n_clusters), index_type{-1}),
      point_by_receiver(static_cast<size_t>(n_clusters), index_type{-1}),
      device_points((n_clusters / 2) * dim,
                    raft::resource::get_cuda_stream(handle),
                    raft::resource::get_workspace_resource_ref(handle)),
      device_donor_by_receiver(n_clusters,
                               raft::resource::get_cuda_stream(handle),
                               raft::resource::get_workspace_resource_ref(handle)),
      device_point_by_receiver(n_clusters,
                               raft::resource::get_cuda_stream(handle),
                               raft::resource::get_workspace_resource_ref(handle))
  {
  }

  raft::pinned_vector<float, index_type> host_points;
  std::vector<std::pair<counter_type, index_type>> sorted;
  std::vector<index_type> receivers;
  std::vector<index_type> donors;
  std::vector<index_type> donor_rows;
  std::vector<index_type> donor_by_receiver;
  std::vector<index_type> point_by_receiver;
  rmm::device_uvector<float> device_points;
  rmm::device_uvector<index_type> device_donor_by_receiver;
  rmm::device_uvector<index_type> device_point_by_receiver;
};

inline auto adjust_centers(const raft::resources& handle,
                           const cuvs::cluster::kmeans::balanced_params& params,
                           raft::host_matrix_view<const float, index_type> X,
                           const label_type* host_labels,
                           const std::vector<counter_type>& sizes,
                           float lower_tolerance,
                           float upper_tolerance,
                           float* centers,
                           adjustment_scratch& scratch) -> bool
{
  auto n_rows     = X.extent(0);
  auto dim        = X.extent(1);
  auto n_clusters = static_cast<index_type>(sizes.size());
  auto average    = static_cast<float>(n_rows) / static_cast<float>(n_clusters);
  auto lower      = average * lower_tolerance;
  auto upper      = average * upper_tolerance;

  for (index_type cluster = 0; cluster < n_clusters; ++cluster) {
    scratch.sorted[cluster] = {sizes[cluster], cluster};
  }
  std::sort(scratch.sorted.begin(), scratch.sorted.end());

  index_type n_pairs = 0;
  for (; n_pairs < n_clusters / 2; ++n_pairs) {
    auto const [small_size, receiver] = scratch.sorted[n_pairs];
    auto const [large_size, donor]    = scratch.sorted[n_clusters - 1 - n_pairs];
    if (receiver == donor || large_size == 0) break;
    if (static_cast<float>(small_size) >= lower && static_cast<float>(large_size) <= upper) break;
    scratch.receivers[n_pairs] = receiver;
    scratch.donors[n_pairs]    = donor;
  }
  if (n_pairs == 0) return false;

  constexpr std::array<index_type, 40> primes{
    29,   71,   113,  173,  229,  281,  349,  409,  463,  541,  601,  659,  733,  809,
    863,  941,  1013, 1069, 1151, 1223, 1291, 1373, 1451, 1511, 1583, 1657, 1733, 1811,
    1889, 1987, 2053, 2129, 2213, 2287, 2357, 2423, 2531, 2617, 2687, 2741};
  static thread_local size_t prime_index = 0;
  index_type seed;
  do {
    prime_index = (prime_index + 1) % primes.size();
    seed        = primes[prime_index];
  } while (n_rows % seed == 0);

  std::fill_n(scratch.donor_rows.begin(), n_pairs, n_rows);
#pragma omp parallel for schedule(static)
  for (index_type pair = 0; pair < n_pairs; ++pair) {
    for (index_type attempt = 0; attempt < n_rows; ++attempt) {
      auto candidate =
        static_cast<index_type>((static_cast<__int128>(seed) * (attempt + 1) + pair) % n_rows);
      if (host_labels[candidate] == static_cast<label_type>(scratch.donors[pair])) {
        scratch.donor_rows[pair] = candidate;
        break;
      }
    }
  }
  for (index_type pair = 0; pair < n_pairs; ++pair) {
    RAFT_EXPECTS(scratch.donor_rows[pair] < n_rows, "Failed to find a point in donor cluster");
    std::memcpy(scratch.host_points.data_handle() + pair * dim,
                X.data_handle() + scratch.donor_rows[pair] * dim,
                static_cast<size_t>(dim) * sizeof(float));
  }

  auto stream = raft::resource::get_cuda_stream(handle);
  raft::update_device(
    scratch.device_points.data(), scratch.host_points.data_handle(), n_pairs * dim, stream);
  std::fill(scratch.donor_by_receiver.begin(), scratch.donor_by_receiver.end(), index_type{-1});
  std::fill(scratch.point_by_receiver.begin(), scratch.point_by_receiver.end(), index_type{-1});
  for (index_type pair = 0; pair < n_pairs; ++pair) {
    auto receiver                       = scratch.receivers[pair];
    scratch.donor_by_receiver[receiver] = scratch.donors[pair];
    scratch.point_by_receiver[receiver] = pair;
  }
  raft::update_device(
    scratch.device_donor_by_receiver.data(), scratch.donor_by_receiver.data(), n_clusters, stream);
  raft::update_device(
    scratch.device_point_by_receiver.data(), scratch.point_by_receiver.data(), n_clusters, stream);

  auto centers_view = raft::make_device_vector_view<float, index_type>(centers, n_clusters * dim);
  raft::linalg::map_offset(
    handle,
    centers_view,
    [centers,
     donor_points      = scratch.device_points.data(),
     donor_by_receiver = scratch.device_donor_by_receiver.data(),
     point_by_receiver = scratch.device_point_by_receiver.data(),
     dim,
     centroid_offset = params.centroid_offset] __device__(auto i, float center) {
      auto offset   = static_cast<index_type>(i);
      auto receiver = offset / dim;
      auto donor    = donor_by_receiver[receiver];
      if (donor < 0) return center;
      auto col          = offset % dim;
      auto point        = point_by_receiver[receiver];
      auto donor_center = centers[donor * dim + col];
      auto donor_point  = donor_points[point * dim + col];
      return donor_center + centroid_offset * (donor_point - donor_center);
    },
    raft::make_const_mdspan(centers_view));
  return true;
}

inline void process_batch(const raft::resources& handle,
                          const cuvs::cluster::kmeans::balanced_params& params,
                          const float* centers,
                          index_type n_clusters,
                          index_type dim,
                          const float* data,
                          index_type n_rows,
                          index_type offset,
                          const float* host_norms,
                          label_type* host_labels,
                          bool predict,
                          prediction_scratch& prediction,
                          const counter_type* count_weights,
                          float* sums,
                          counter_type* counts)
{
  auto stream = raft::resource::get_cuda_stream(handle);
  if (predict) {
    raft::update_device(prediction.norms.data(), host_norms + offset, n_rows, stream);
    predict_batch(
      handle, params, centers, n_clusters, dim, data, n_rows, prediction.norms.data(), prediction);
  } else {
    raft::update_device(prediction.labels.data(), host_labels + offset, n_rows, stream);
  }

  raft::linalg::reduce_rows_by_key(data,
                                   dim,
                                   prediction.labels.data(),
                                   static_cast<char*>(nullptr),
                                   n_rows,
                                   dim,
                                   n_clusters,
                                   sums,
                                   stream.get(),
                                   false);
  raft::linalg::reduce_cols_by_key(
    handle,
    raft::make_device_matrix_view<const counter_type, index_type>(
      count_weights, index_type{1}, n_rows),
    raft::make_device_vector_view<const label_type, index_type>(prediction.labels.data(), n_rows),
    raft::make_device_matrix_view<counter_type, index_type>(counts, index_type{1}, n_clusters),
    n_clusters,
    false);
  raft::update_host(host_labels + offset, prediction.labels.data(), n_rows, stream);
}

inline void balancing_em_iters(const raft::resources& handle,
                               const cuvs::cluster::kmeans::balanced_params& params,
                               raft::host_matrix_view<const float, index_type> X,
                               const float* host_norms,
                               index_type n_clusters,
                               float* centers,
                               label_type* host_labels,
                               std::vector<counter_type>& host_sizes,
                               uint32_t n_iters,
                               uint32_t balancing_pullback,
                               float lower_tolerance,
                               float upper_tolerance,
                               bool initialize_centers)
{
  auto n_rows         = X.extent(0);
  auto dim            = X.extent(1);
  auto rows_per_batch = batch_rows(params, n_rows);
  auto stream         = raft::resource::get_cuda_stream(handle);
  auto mr             = raft::resource::get_workspace_resource_ref(handle);
  auto center_elems   = n_clusters * dim;

  prediction_scratch prediction(handle, rows_per_batch);
  adjustment_scratch adjustment(handle, n_clusters, dim);
  rmm::device_uvector<float> next_centers(center_elems, stream, mr);
  rmm::device_uvector<float> sums(center_elems, stream, mr);
  rmm::device_uvector<counter_type> counts(n_clusters, stream, mr);
  rmm::device_uvector<counter_type> count_weights(rows_per_batch, stream, mr);
  auto pinned_sizes = raft::make_pinned_vector<counter_type, index_type>(handle, n_clusters);
  raft::matrix::fill(
    handle,
    raft::make_device_vector_view<counter_type, index_type>(count_weights.data(), rows_per_batch),
    counter_type{1});

  rmm::cuda_stream copy_stream(rmm::cuda_stream::flags::non_blocking);
  auto batches = cuvs::spatial::knn::detail::utils::batch_load_iterator(
    handle, X, rows_per_batch, copy_stream.view(), mr, true);

  auto process_batches = [&](bool predict) {
    raft::matrix::fill(handle,
                       raft::make_device_vector_view<float, index_type>(sums.data(), center_elems),
                       float{0});
    raft::matrix::fill(
      handle,
      raft::make_device_vector_view<counter_type, index_type>(counts.data(), n_clusters),
      counter_type{0});

    batches.reset();
    batches.prefetch_next_batch();
    for (auto const& batch : batches) {
      process_batch(handle,
                    params,
                    centers,
                    n_clusters,
                    dim,
                    batch.data(),
                    static_cast<index_type>(batch.size()),
                    static_cast<index_type>(batch.offset()),
                    host_norms,
                    host_labels,
                    predict,
                    prediction,
                    count_weights.data(),
                    sums.data(),
                    counts.data());
      batches.prefetch_next_batch();
    }

    auto next_centers_view =
      raft::make_device_vector_view<float, index_type>(next_centers.data(), center_elems);
    auto counts_ptr = counts.data();
    raft::linalg::map_offset(
      handle,
      next_centers_view,
      [counts_ptr, centers, dim] __device__(auto i, float sum) {
        auto offset = static_cast<index_type>(i);
        auto count  = counts_ptr[offset / dim];
        return count == 0 ? centers[offset] : sum / static_cast<float>(count);
      },
      raft::make_device_vector_view<const float, index_type>(sums.data(), center_elems));
    raft::update_host(pinned_sizes.data_handle(), counts.data(), counts.size(), stream);
    raft::resource::sync_stream(handle, stream);
    raft::resource::sync_stream(handle, copy_stream.view());
    std::copy_n(pinned_sizes.data_handle(), n_clusters, host_sizes.begin());
  };

  if (initialize_centers) {
#pragma omp parallel for schedule(static)
    for (index_type row = 0; row < n_rows; ++row) {
      host_labels[row] = static_cast<label_type>(row % n_clusters);
    }
    raft::matrix::fill(
      handle, raft::make_device_vector_view<float, index_type>(centers, center_elems), float{0});
    process_batches(false);
    raft::copy(centers, next_centers.data(), center_elems, stream);
  }

  auto adjust_centers_op = [&] {
    return adjust_centers(handle,
                          params,
                          X,
                          host_labels,
                          host_sizes,
                          lower_tolerance,
                          upper_tolerance,
                          centers,
                          adjustment);
  };
  auto process_batches_op = [&] {
    process_batches(true);
    raft::copy(centers, next_centers.data(), center_elems, stream);
  };
  cuvs::cluster::kmeans::detail::balancing_em_iters(handle,
                                                    params,
                                                    n_iters,
                                                    n_clusters,
                                                    dim,
                                                    centers,
                                                    balancing_pullback,
                                                    lower_tolerance,
                                                    upper_tolerance,
                                                    adjust_centers_op,
                                                    process_batches_op);
  raft::resource::sync_stream(handle, stream);
}

inline void build_clusters(const raft::resources& handle,
                           const cuvs::cluster::kmeans::balanced_params& params,
                           raft::host_matrix_view<const float, index_type> X,
                           const float* host_norms,
                           index_type n_clusters,
                           float* centers,
                           label_type* host_labels,
                           std::vector<counter_type>& host_sizes)
{
  balancing_em_iters(handle,
                     params,
                     X,
                     host_norms,
                     n_clusters,
                     centers,
                     host_labels,
                     host_sizes,
                     params.n_iters,
                     2,
                     params.balance_lower_tolerance,
                     params.balance_upper_tolerance,
                     true);
}

inline auto build_fine_clusters(const raft::resources& handle,
                                const cuvs::cluster::kmeans::balanced_params& params,
                                raft::host_matrix_view<const float, index_type> X,
                                const float* host_norms,
                                const label_type* mesocluster_labels,
                                const index_type* fine_cluster_counts,
                                const index_type* fine_cluster_offsets,
                                const counter_type* mesocluster_sizes,
                                index_type n_mesoclusters,
                                index_type mesocluster_size_max,
                                index_type fine_clusters_max,
                                float* centers) -> index_type
{
  auto n_rows = X.extent(0);
  auto dim    = X.extent(1);
  std::vector<index_type> capped_sizes(n_mesoclusters);
  std::vector<index_type> offsets(static_cast<size_t>(n_mesoclusters) + 1);
  for (index_type meso = 0; meso < n_mesoclusters; ++meso) {
    capped_sizes[meso] = std::min<index_type>(mesocluster_sizes[meso], mesocluster_size_max);
    offsets[meso + 1]  = offsets[meso] + capped_sizes[meso];
  }

  std::vector<index_type> row_ids(offsets.back());
  auto cursors = offsets;
  for (index_type row = 0; row < n_rows; ++row) {
    auto meso = static_cast<index_type>(mesocluster_labels[row]);
    RAFT_EXPECTS(meso < n_mesoclusters, "Mesocluster label is out of range");
    if (cursors[meso] < offsets[meso + 1]) { row_ids[cursors[meso]++] = row; }
  }

  std::vector<index_type> work;
  index_type max_rows = 0;
  for (index_type meso = 0; meso < n_mesoclusters; ++meso) {
    if (capped_sizes[meso] == 0) {
      RAFT_EXPECTS(fine_cluster_counts[meso] == 0,
                   "An empty mesocluster was assigned fine clusters");
      continue;
    }
    RAFT_EXPECTS(fine_cluster_counts[meso] > 0,
                 "A non-empty mesocluster was not assigned fine clusters");
    max_rows = std::max(max_rows, capped_sizes[meso]);
    work.push_back(meso);
  }
  if (work.empty()) return 0;

  auto stream        = raft::resource::get_cuda_stream(handle);
  auto mr            = raft::resource::get_workspace_resource_ref(handle);
  auto large_mr      = raft::resource::get_large_workspace_resource_ref(handle);
  auto slots         = work.size() > 1 ? 2 : 1;
  auto slot_elements = static_cast<size_t>(max_rows) * static_cast<size_t>(dim);
  RAFT_EXPECTS(slot_elements <= static_cast<size_t>(std::numeric_limits<index_type>::max()) /
                                  static_cast<size_t>(slots),
               "Host mesocluster staging allocation exceeds the supported index range");
  auto total_elements = static_cast<index_type>(slots * slot_elements);
  auto total_rows     = static_cast<index_type>(slots) * max_rows;
  auto host_data      = raft::make_pinned_vector<float, index_type>(handle, total_elements);
  auto host_norm      = raft::make_pinned_vector<float, index_type>(handle, total_rows);
  rmm::device_uvector<float> device_data(total_elements, stream, large_mr);
  rmm::device_uvector<float> device_norm(total_rows, stream, mr);
  rmm::device_uvector<label_type> local_labels(max_rows, stream, mr);
  rmm::device_uvector<counter_type> local_sizes(fine_clusters_max, stream, mr);
  rmm::cuda_stream copy_stream(rmm::cuda_stream::flags::non_blocking);
  cuda_event ready[2];
  cuda_event consumed[2];
  for (int slot = 0; slot < slots; ++slot) {
    RAFT_CUDA_TRY(cudaEventRecord(consumed[slot].get(), stream.get()));
  }

  auto device = raft::resource::get_device_id(handle);
  auto submit = [&](size_t sequence) {
    return std::async(std::launch::async, [&, sequence, device] {
      auto device_scope = raft::device_setter{device};
      auto meso         = work[sequence];
      auto slot         = static_cast<int>(sequence % slots);
      auto rows         = capped_sizes[meso];
      if (sequence >= static_cast<size_t>(slots)) {
        RAFT_CUDA_TRY(cudaEventSynchronize(consumed[slot].get()));
      }
      auto* stage      = host_data.data_handle() + slot * max_rows * dim;
      auto* norm_stage = host_norm.data_handle() + slot * max_rows;
#pragma omp parallel for schedule(static)
      for (index_type local = 0; local < rows; ++local) {
        auto source = row_ids[offsets[meso] + local];
        std::memcpy(stage + local * dim,
                    X.data_handle() + source * dim,
                    static_cast<size_t>(dim) * sizeof(float));
        norm_stage[local] = host_norms[source];
      }
      auto* device_stage = device_data.data() + slot * max_rows * dim;
      auto* device_norms = device_norm.data() + slot * max_rows;
      RAFT_CUDA_TRY(cudaStreamWaitEvent(copy_stream.view().get(), consumed[slot].get(), 0));
      raft::update_device(device_stage, stage, rows * dim, copy_stream.view());
      raft::update_device(device_norms, norm_stage, rows, copy_stream.view());
      RAFT_CUDA_TRY(cudaEventRecord(ready[slot].get(), copy_stream.view().get()));
    });
  };

  index_type clusters_done = 0;
  auto pending             = submit(0);
  for (size_t sequence = 0; sequence < work.size(); ++sequence) {
    auto meso = work[sequence];
    auto slot = static_cast<int>(sequence % slots);
    auto rows = capped_sizes[meso];
    pending.get();
    RAFT_CUDA_TRY(cudaStreamWaitEvent(stream.get(), ready[slot].get(), 0));
    if (sequence + 1 < work.size()) { pending = submit(sequence + 1); }

    auto* device_stage = device_data.data() + slot * max_rows * dim;
    auto* device_norms = device_norm.data() + slot * max_rows;
    cuvs::cluster::kmeans::detail::build_clusters(handle,
                                                  params,
                                                  dim,
                                                  device_stage,
                                                  rows,
                                                  fine_cluster_counts[meso],
                                                  centers + fine_cluster_offsets[meso] * dim,
                                                  local_labels.data(),
                                                  local_sizes.data(),
                                                  raft::identity_op{},
                                                  mr,
                                                  device_norms);
    clusters_done += fine_cluster_counts[meso];
    RAFT_CUDA_TRY(cudaEventRecord(consumed[slot].get(), stream.get()));
  }
  raft::resource::sync_stream(handle, stream);
  raft::resource::sync_stream(handle, copy_stream.view());
  return clusters_done;
}

inline auto compute_inertia(const raft::resources& handle,
                            raft::host_matrix_view<const float, index_type> X,
                            index_type rows_per_batch,
                            const float* centers,
                            index_type n_clusters) -> float
{
  auto stream = raft::resource::get_cuda_stream(handle);
  auto mr     = raft::resource::get_workspace_resource_ref(handle);
  rmm::cuda_stream copy_stream(rmm::cuda_stream::flags::non_blocking);
  auto batches = cuvs::spatial::knn::detail::utils::batch_load_iterator(
    handle, X, rows_per_batch, copy_stream.view(), mr, true);
  auto C = raft::make_device_matrix_view<const float, index_type>(centers, n_clusters, X.extent(1));
  float result = 0;
  batches.prefetch_next_batch();
  for (auto const& batch : batches) {
    float batch_cost = 0;
    auto data        = raft::make_device_matrix_view<const float, index_type>(
      batch.data(), static_cast<index_type>(batch.size()), X.extent(1));
    cuvs::cluster::kmeans::cluster_cost(handle, data, C, raft::make_host_scalar_view(&batch_cost));
    result += batch_cost;
    batches.prefetch_next_batch();
  }
  raft::resource::sync_stream(handle, stream);
  raft::resource::sync_stream(handle, copy_stream.view());
  return result;
}

}  // namespace host_balanced

}  // namespace cuvs::cluster::kmeans::detail

namespace cuvs::cluster::kmeans::detail {

/**
 * @brief Hierarchical balanced k-means for host- or device-resident input.
 *
 * Device input follows the existing in-core path. Host input streams the coarse and global
 * refinement passes, then gathers each mesocluster from host memory into a double-buffered,
 * device-resident fine-training buffer.
 *
 * @tparam T Input element type.
 * @tparam MathT Arithmetic and centroid type.
 * @tparam IdxT Index type.
 * @tparam Accessor Input accessor policy; determines host or device processing at compile time.
 * @tparam MappingOpT Mapping operation from T to MathT.
 *
 * @param[in] handle The raft resources.
 * @param[in] params Balanced k-means parameters.
 * @param[in] X Row-major input data [n_rows, dim].
 * @param[out] cluster_centers Output cluster centers [n_clusters, dim].
 * @param[in] n_clusters Number of requested clusters.
 * @param[in] mapping_op Mapping operation from T to MathT.
 * @param[out] inertia Optional sum of squared distances to the final centers.
 */
template <typename T, typename MathT, typename IdxT, typename Accessor, typename MappingOpT>
void build_hierarchical(
  const raft::resources& handle,
  const cuvs::cluster::kmeans::balanced_params& params,
  raft::mdspan<const T, raft::matrix_extent<IdxT>, raft::row_major, Accessor> X,
  MathT* cluster_centers,
  IdxT n_clusters,
  MappingOpT mapping_op,
  MathT* inertia = nullptr)
{
  constexpr bool data_on_device = raft::is_device_mdspan_v<decltype(X)>;
  if constexpr (!data_on_device) {
    static_assert(std::is_same_v<T, float>);
    static_assert(std::is_same_v<MathT, float>);
    static_assert(std::is_same_v<IdxT, int64_t>);
    RAFT_EXPECTS(params.metric == cuvs::distance::DistanceType::L2Expanded ||
                   params.metric == cuvs::distance::DistanceType::L2SqrtExpanded,
                 "Host balanced k-means currently supports L2Expanded and L2SqrtExpanded");
    RAFT_EXPECTS(
      params.donor_selection == cuvs::cluster::kmeans::balanced_donor_selection::SizeSorted,
      "Host balanced k-means currently supports SizeSorted donor selection");
  }

  RAFT_EXPECTS(params.balance_lower_tolerance > 0.0f && params.balance_lower_tolerance < 1.0f,
               "Balanced k-means lower balance tolerance must be in the range (0, 1)");
  RAFT_EXPECTS(params.balance_upper_tolerance > 1.0f,
               "Balanced k-means upper balance tolerance must be greater than 1");
  RAFT_EXPECTS(params.centroid_offset > 0.0f && params.centroid_offset <= 1.0f,
               "Balanced k-means centroid offset must be in the range (0, 1]");

  auto stream    = raft::resource::get_cuda_stream(handle);
  auto n_rows    = X.extent(0);
  auto dim       = X.extent(1);
  auto dataset   = X.data_handle();
  using LabelT   = uint32_t;
  using CounterT = std::conditional_t<sizeof(IdxT) == 8, unsigned long long int, unsigned int>;

  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "build_hierarchical(%zu, %u)", static_cast<size_t>(n_rows), n_clusters);

  auto n_mesoclusters = std::min(n_clusters, static_cast<IdxT>(std::sqrt(n_clusters) + 0.5));
  RAFT_LOG_DEBUG("build_hierarchical: n_mesoclusters: %u", n_mesoclusters);

  rmm::mr::managed_memory_resource managed_memory;
  auto device_memory = raft::resource::get_workspace_resource_ref(handle);
  auto needs_norm    = params.metric == cuvs::distance::DistanceType::L2Expanded ||
                    params.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                    params.metric == cuvs::distance::DistanceType::CosineExpanded;
  auto rows_per_batch =
    data_on_device ? n_rows : host_balanced::batch_rows(params, static_cast<int64_t>(n_rows));

  rmm::device_uvector<MathT> device_norms(
    data_on_device && needs_norm ? n_rows : 0, stream, device_memory);
  auto host_norms =
    raft::make_pinned_vector<MathT, IdxT>(handle, !data_on_device && needs_norm ? n_rows : 0);
  const MathT* dataset_norm = nullptr;
  if constexpr (data_on_device) {
    if (needs_norm) {
      auto [max_minibatch_size, _] = calc_minibatch_size<MathT>(
        handle, n_clusters, n_rows, dim, params.metric, std::is_same_v<T, MathT>);
      for (IdxT offset = 0; offset < n_rows; offset += max_minibatch_size) {
        auto minibatch_size = std::min<IdxT>(max_minibatch_size, n_rows - offset);
        if (params.metric == cuvs::distance::DistanceType::CosineExpanded) {
          compute_norm(handle,
                       device_norms.data() + offset,
                       dataset + dim * offset,
                       dim,
                       minibatch_size,
                       mapping_op,
                       raft::sqrt_op{},
                       device_memory);
        } else {
          compute_norm(handle,
                       device_norms.data() + offset,
                       dataset + dim * offset,
                       dim,
                       minibatch_size,
                       mapping_op,
                       raft::identity_op{},
                       device_memory);
        }
      }
      dataset_norm = device_norms.data();
    }
  } else {
    host_balanced::compute_norms(handle, X, rows_per_batch, host_norms.data_handle());
    dataset_norm = host_norms.data_handle();
  }

  rmm::device_uvector<LabelT> device_mesocluster_labels(
    data_on_device ? n_rows : 0, stream, managed_memory);
  auto host_mesocluster_labels =
    raft::make_pinned_vector<LabelT, IdxT>(handle, data_on_device ? 0 : n_rows);
  auto* mesocluster_labels =
    data_on_device ? device_mesocluster_labels.data() : host_mesocluster_labels.data_handle();
  std::vector<CounterT> mesocluster_sizes(n_mesoclusters);
  {
    rmm::device_uvector<MathT> mesocluster_centers(n_mesoclusters * dim, stream, device_memory);
    if constexpr (data_on_device) {
      rmm::device_uvector<CounterT> device_mesocluster_sizes(n_mesoclusters, stream, device_memory);
      build_clusters(handle,
                     params,
                     dim,
                     dataset,
                     n_rows,
                     n_mesoclusters,
                     mesocluster_centers.data(),
                     mesocluster_labels,
                     device_mesocluster_sizes.data(),
                     mapping_op,
                     device_memory,
                     dataset_norm);
      raft::update_host(
        mesocluster_sizes.data(), device_mesocluster_sizes.data(), n_mesoclusters, stream);
      raft::resource::sync_stream(handle, stream);
    } else {
      host_balanced::build_clusters(handle,
                                    params,
                                    X,
                                    dataset_norm,
                                    n_mesoclusters,
                                    mesocluster_centers.data(),
                                    mesocluster_labels,
                                    mesocluster_sizes);
    }
  }

  auto [mesocluster_size_max, fine_clusters_max, fine_cluster_counts, fine_cluster_offsets] =
    arrange_fine_clusters(n_clusters, n_mesoclusters, n_rows, mesocluster_sizes.data());

  auto balanced_max = static_cast<IdxT>(raft::div_rounding_up_safe<size_t>(
    2lu * static_cast<size_t>(n_rows), std::max<size_t>(static_cast<size_t>(n_mesoclusters), 1lu)));
  if (mesocluster_size_max > balanced_max) {
    RAFT_LOG_DEBUG(
      "build_hierarchical: built unbalanced mesoclusters (max_mesocluster_size == %u > %u). "
      "At most %u points will be used for training within each mesocluster. "
      "Consider increasing the number of training iterations n_iters.",
      mesocluster_size_max,
      balanced_max,
      balanced_max);
    RAFT_LOG_TRACE_VEC(mesocluster_sizes.data(), n_mesoclusters);
    RAFT_LOG_TRACE_VEC(fine_cluster_counts.data(), n_mesoclusters);
    mesocluster_size_max = balanced_max;
  }

  IdxT clusters_done;
  if constexpr (data_on_device) {
    clusters_done = build_fine_clusters(handle,
                                        params,
                                        dim,
                                        dataset,
                                        dataset_norm,
                                        mesocluster_labels,
                                        n_rows,
                                        fine_cluster_counts.data(),
                                        fine_cluster_offsets.data(),
                                        mesocluster_sizes.data(),
                                        n_mesoclusters,
                                        mesocluster_size_max,
                                        fine_clusters_max,
                                        cluster_centers,
                                        mapping_op,
                                        managed_memory,
                                        device_memory);
  } else {
    clusters_done = host_balanced::build_fine_clusters(handle,
                                                       params,
                                                       X,
                                                       dataset_norm,
                                                       mesocluster_labels,
                                                       fine_cluster_counts.data(),
                                                       fine_cluster_offsets.data(),
                                                       mesocluster_sizes.data(),
                                                       n_mesoclusters,
                                                       mesocluster_size_max,
                                                       fine_clusters_max,
                                                       cluster_centers);
  }
  RAFT_EXPECTS(clusters_done == n_clusters, "Didn't process all clusters.");

  auto global_iters               = std::max<uint32_t>(params.n_iters / 10, 2);
  constexpr float relaxing_factor = 1.0f;
  auto balance_lower_tolerance =
    static_cast<MathT>(params.balance_lower_tolerance * relaxing_factor);
  auto balance_upper_tolerance =
    static_cast<MathT>(params.balance_upper_tolerance / relaxing_factor);
  RAFT_LOG_DEBUG("n_iters: %u, tolerance: %f, %f\n",
                 global_iters,
                 balance_lower_tolerance,
                 balance_upper_tolerance);

  if constexpr (data_on_device) {
    rmm::device_uvector<CounterT> cluster_sizes(n_clusters, stream, device_memory);
    rmm::device_uvector<LabelT> labels(n_rows, stream, device_memory);
    balancing_em_iters(handle,
                       params,
                       global_iters,
                       dim,
                       dataset,
                       dataset_norm,
                       n_rows,
                       n_clusters,
                       cluster_centers,
                       labels.data(),
                       cluster_sizes.data(),
                       5,
                       balance_lower_tolerance,
                       balance_upper_tolerance,
                       mapping_op,
                       device_memory);
  } else {
    std::vector<CounterT> cluster_sizes(n_clusters);
    host_balanced::balancing_em_iters(handle,
                                      params,
                                      X,
                                      dataset_norm,
                                      n_clusters,
                                      cluster_centers,
                                      mesocluster_labels,
                                      cluster_sizes,
                                      global_iters,
                                      5,
                                      balance_lower_tolerance,
                                      balance_upper_tolerance,
                                      false);
  }

  if (inertia != nullptr) {
    if constexpr (data_on_device) {
      if constexpr (std::is_same_v<T, MathT>) {
        auto data_view = raft::make_device_matrix_view<const MathT, IdxT>(
          reinterpret_cast<const MathT*>(dataset), n_rows, dim);
        auto centroids_view =
          raft::make_device_matrix_view<const MathT, IdxT>(cluster_centers, n_clusters, dim);
        cuvs::cluster::kmeans::cluster_cost(
          handle, data_view, centroids_view, raft::make_host_scalar_view<MathT>(inertia));
      } else {
        RAFT_LOG_WARN("Inertia is not computed for non float/double types");
      }
    } else {
      *inertia =
        host_balanced::compute_inertia(handle, X, rows_per_batch, cluster_centers, n_clusters);
    }
  }
}

}  // namespace cuvs::cluster::kmeans::detail
