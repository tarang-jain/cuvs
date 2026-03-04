/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "kmeans.cuh"
#include "kmeans_batched.cuh"
#include "kmeans_common.cuh"

#include "../../core/omp_wrapper.hpp"
#include "../../neighbors/detail/ann_utils.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/matrix/init.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

namespace cuvs::cluster::kmeans::detail {

/**
 * @brief Multi-GPU full-batch K-Means fit using SNMG (single-node multi-GPU).
 *
 * Algorithm per iteration:
 *   1. Broadcast centroids from root device to all GPUs (via host staging).
 *   2. Partition the host data across GPUs (contiguous row splits).
 *   3. Each GPU processes its partition in sub-batches:
 *      - Load sub-batch from host to device via batch_load_iterator.
 *      - Compute nearest centroid for each point.
 *      - Accumulate weighted centroid sums and cluster counts.
 *   4. Gather partial centroid sums and counts to host, reduce.
 *   5. Copy combined sums to root device, finalize centroids.
 *   6. Check convergence on root device.
 *
 * @tparam T      Data type (float, double)
 * @tparam IdxT   Index type (int, int64_t)
 *
 * @param[in]     clique        SNMG resources (raft::device_resources_snmg)
 * @param[in]     params        K-means parameters
 * @param[in]     X             Input data on HOST [n_samples x n_features]
 * @param[in]     batch_size    Rows loaded per GPU per sub-batch
 * @param[in]     sample_weight Optional per-sample weights (on host)
 * @param[inout]  centroids     Cluster centers on root device [n_clusters x n_features]
 * @param[out]    inertia       Sum of squared distances (if final_inertia_check)
 * @param[out]    n_iter        Number of iterations completed
 */
template <typename T, typename IdxT>
void fit_mg(raft::resources const& clique,
            const cuvs::cluster::kmeans::params& params,
            raft::host_matrix_view<const T, IdxT> X,
            IdxT batch_size,
            std::optional<raft::host_vector_view<const T, IdxT>> sample_weight,
            raft::device_matrix_view<T, IdxT> centroids,
            raft::host_scalar_view<T> inertia,
            raft::host_scalar_view<IdxT> n_iter)
{
  int num_ranks   = raft::resource::get_num_ranks(clique);
  auto n_samples  = X.extent(0);
  auto n_features = X.extent(1);
  auto n_clusters = params.n_clusters;
  auto metric     = params.metric;

  RAFT_EXPECTS(batch_size > 0, "batch_size must be positive");
  RAFT_EXPECTS(n_clusters > 0, "n_clusters must be positive");
  RAFT_EXPECTS(num_ranks > 0, "num_ranks must be positive");
  RAFT_EXPECTS(static_cast<IdxT>(centroids.extent(0)) == n_clusters,
               "centroids.extent(0) must equal n_clusters");
  RAFT_EXPECTS(centroids.extent(1) == n_features, "centroids.extent(1) must equal n_features");

  raft::default_logger().set_level(params.verbosity);

  RAFT_LOG_DEBUG(
    "KMeans MG fit: n_samples=%zu, n_features=%zu, n_clusters=%d, "
    "batch_size=%zu, num_gpus=%d",
    static_cast<size_t>(n_samples),
    static_cast<size_t>(n_features),
    n_clusters,
    static_cast<size_t>(batch_size),
    num_ranks);

  // ---------- Root device: initialization ----------
  const raft::resources& root_res = raft::resource::set_current_device_to_root_rank(clique);
  cudaStream_t root_stream        = raft::resource::get_cuda_stream(root_res);

  if (params.init != cuvs::cluster::kmeans::params::InitMethod::Array) {
    rmm::device_uvector<char> workspace(0, root_stream);
    init_centroids_from_host_sample(root_res, params, X, centroids, workspace);
  }

  // Root-device buffers (pre-allocated, reused across iterations)
  auto new_centroids    = raft::make_device_matrix<T, IdxT>(root_res, n_clusters, n_features);
  auto d_centroid_sums  = raft::make_device_matrix<T, IdxT>(root_res, n_clusters, n_features);
  auto d_cluster_counts = raft::make_device_vector<T, IdxT>(root_res, n_clusters);

  // ---------- Host buffers ----------
  size_t sums_size = static_cast<size_t>(n_clusters) * static_cast<size_t>(n_features);

  auto h_centroids      = raft::make_host_matrix<T, IdxT>(n_clusters, n_features);
  auto h_centroid_sums  = raft::make_host_matrix<T, IdxT>(n_clusters, n_features);
  auto h_cluster_counts = raft::make_host_vector<T, IdxT>(n_clusters);

  // Per-rank host partial sums
  std::vector<std::vector<T>> rank_partial_sums(num_ranks);
  std::vector<std::vector<T>> rank_partial_counts(num_ranks);
  for (int r = 0; r < num_ranks; r++) {
    rank_partial_sums[r].resize(sums_size);
    rank_partial_counts[r].resize(n_clusters);
  }

  // Per-rank cost accumulators (for inertia_check)
  std::vector<T> rank_costs(num_ranks, T{0});

  // Data partitioning: contiguous row splits
  IdxT rows_per_gpu = raft::ceildiv(n_samples, static_cast<IdxT>(num_ranks));

  T priorClusteringCost = 0;

  // ---------- OMP setup ----------
  int saved_nested     = cuvs::core::omp::get_nested();
  int omp_threads      = cuvs::core::omp::get_max_threads();
  int threads_per_rank = std::max(1, omp_threads / num_ranks);
  cuvs::core::omp::set_nested(1);

  // ========== Main iteration loop ==========
  for (n_iter[0] = 1; n_iter[0] <= params.max_iter; ++n_iter[0]) {
    RAFT_LOG_DEBUG("KMeans MG: Iteration %d", n_iter[0]);

    // 1. Copy current centroids from root device to host
    raft::resource::set_current_device_to_root_rank(clique);
    raft::copy(h_centroids.data_handle(), centroids.data_handle(), centroids.size(), root_stream);
    raft::resource::sync_stream(root_res, root_stream);

    // Reset per-rank cost accumulators
    std::fill(rank_costs.begin(), rank_costs.end(), T{0});

    // 2. Parallel: each GPU processes its data partition
    cuvs::core::omp::check_threads(num_ranks);
#pragma omp parallel for num_threads(num_ranks)
    for (int rank = 0; rank < num_ranks; rank++) {
      cuvs::core::omp::set_num_threads(threads_per_rank);

      const raft::resources& dev_res = raft::resource::set_current_device_to_rank(clique, rank);
      cudaStream_t stream            = raft::resource::get_cuda_stream(dev_res);

      // Data range for this GPU
      IdxT start   = rank * rows_per_gpu;
      IdxT end     = std::min(start + rows_per_gpu, n_samples);
      IdxT local_n = end - start;

      if (local_n <= 0) {
        std::fill(rank_partial_sums[rank].begin(), rank_partial_sums[rank].end(), T{0});
        std::fill(rank_partial_counts[rank].begin(), rank_partial_counts[rank].end(), T{0});
        raft::resource::sync_stream(dev_res, stream);
        cuvs::core::omp::set_num_threads(omp_threads);
        continue;
      }

      // Per-GPU device buffers
      auto gpu_centroids      = raft::make_device_matrix<T, IdxT>(dev_res, n_clusters, n_features);
      auto gpu_centroid_sums  = raft::make_device_matrix<T, IdxT>(dev_res, n_clusters, n_features);
      auto gpu_cluster_counts = raft::make_device_vector<T, IdxT>(dev_res, n_clusters);

      IdxT effective_batch = std::min(batch_size, local_n);
      auto batch_weights   = raft::make_device_vector<T, IdxT>(dev_res, effective_batch);
      auto minClusterAndDistance =
        raft::make_device_vector<raft::KeyValuePair<IdxT, T>, IdxT>(dev_res, effective_batch);
      auto L2NormBatch = raft::make_device_vector<T, IdxT>(dev_res, effective_batch);
      rmm::device_uvector<T> L2NormBuf_OR_DistBuf(0, stream);
      rmm::device_uvector<char> workspace(0, stream);
      rmm::device_scalar<T> clusterCostD(stream);

      // Copy centroids from host to this GPU
      raft::copy(gpu_centroids.data_handle(), h_centroids.data_handle(), centroids.size(), stream);

      // Zero accumulators
      raft::matrix::fill(dev_res, gpu_centroid_sums.view(), T{0});
      raft::matrix::fill(dev_res, gpu_cluster_counts.view(), T{0});

      // Process partition in sub-batches
      using namespace cuvs::spatial::knn::detail::utils;
      const T* local_data = X.data_handle() + start * n_features;
      batch_load_iterator<T> data_batches(local_data, local_n, n_features, batch_size, stream);

      auto gpu_centroids_const = raft::make_device_matrix_view<const T, IdxT>(
        gpu_centroids.data_handle(), n_clusters, n_features);

      T local_cost = 0;

      for (const auto& data_batch : data_batches) {
        IdxT current_batch_size = static_cast<IdxT>(data_batch.size());

        auto batch_data_view = raft::make_device_matrix_view<const T, IdxT>(
          data_batch.data(), current_batch_size, n_features);

        // Set weights
        auto batch_weights_fill_view =
          raft::make_device_vector_view<T, IdxT>(batch_weights.data_handle(), current_batch_size);
        if (sample_weight) {
          IdxT global_offset = start + static_cast<IdxT>(data_batch.offset());
          raft::copy(batch_weights.data_handle(),
                     sample_weight->data_handle() + global_offset,
                     current_batch_size,
                     stream);
        } else {
          raft::matrix::fill(dev_res, batch_weights_fill_view, T{1});
        }

        auto batch_weights_view = raft::make_device_vector_view<const T, IdxT>(
          batch_weights.data_handle(), current_batch_size);

        // L2 norms
        if (metric == cuvs::distance::DistanceType::L2Expanded ||
            metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
          raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
            L2NormBatch.data_handle(), data_batch.data(), n_features, current_batch_size, stream);
        }

        auto L2NormBatch_const = raft::make_device_vector_view<const T, IdxT>(
          L2NormBatch.data_handle(), current_batch_size);
        auto minClusterAndDistance_view =
          raft::make_device_vector_view<raft::KeyValuePair<IdxT, T>, IdxT>(
            minClusterAndDistance.data_handle(), current_batch_size);

        // Compute nearest centroid
        cuvs::cluster::kmeans::detail::minClusterAndDistanceCompute<T, IdxT>(
          dev_res,
          batch_data_view,
          gpu_centroids_const,
          minClusterAndDistance_view,
          L2NormBatch_const,
          L2NormBuf_OR_DistBuf,
          metric,
          params.batch_samples,
          params.batch_centroids,
          workspace);

        // Accumulate centroid sums and counts
        auto minClusterAndDistance_const = raft::make_const_mdspan(minClusterAndDistance_view);
        accumulate_batch_centroids<T, IdxT>(dev_res,
                                            batch_data_view,
                                            minClusterAndDistance_const,
                                            batch_weights_view,
                                            gpu_centroid_sums.view(),
                                            gpu_cluster_counts.view());

        // Optionally accumulate batch cost
        if (params.inertia_check) {
          cuvs::cluster::kmeans::detail::computeClusterCost(
            dev_res,
            minClusterAndDistance_view,
            workspace,
            raft::make_device_scalar_view(clusterCostD.data()),
            raft::value_op{},
            raft::add_op{});
          auto clusterCost_host = raft::make_host_scalar<T>(0);
          raft::copy(clusterCost_host.data_handle(), clusterCostD.data(), 1, stream);
          raft::resource::sync_stream(dev_res, stream);
          local_cost += clusterCost_host.data_handle()[0];
        }
      }  // end sub-batch loop

      rank_costs[rank] = local_cost;

      // Copy partial sums to host
      raft::copy(
        rank_partial_sums[rank].data(), gpu_centroid_sums.data_handle(), sums_size, stream);
      raft::copy(
        rank_partial_counts[rank].data(), gpu_cluster_counts.data_handle(), n_clusters, stream);
      raft::resource::sync_stream(dev_res, stream);

      cuvs::core::omp::set_num_threads(omp_threads);
    }  // end OMP parallel

    // 3. Reduce partial sums on host
    std::fill(h_centroid_sums.data_handle(), h_centroid_sums.data_handle() + sums_size, T{0});
    std::fill(h_cluster_counts.data_handle(), h_cluster_counts.data_handle() + n_clusters, T{0});
    for (int r = 0; r < num_ranks; r++) {
      for (size_t i = 0; i < sums_size; i++) {
        h_centroid_sums.data_handle()[i] += rank_partial_sums[r][i];
      }
      for (size_t i = 0; i < static_cast<size_t>(n_clusters); i++) {
        h_cluster_counts.data_handle()[i] += rank_partial_counts[r][i];
      }
    }

    // 4. Copy combined sums to root device and finalize centroids
    raft::resource::set_current_device_to_root_rank(clique);
    raft::copy(
      d_centroid_sums.data_handle(), h_centroid_sums.data_handle(), sums_size, root_stream);
    raft::copy(
      d_cluster_counts.data_handle(), h_cluster_counts.data_handle(), n_clusters, root_stream);

    auto centroids_const =
      raft::make_device_matrix_view<const T, IdxT>(centroids.data_handle(), n_clusters, n_features);
    auto d_centroid_sums_const = raft::make_device_matrix_view<const T, IdxT>(
      d_centroid_sums.data_handle(), n_clusters, n_features);
    auto d_cluster_counts_const =
      raft::make_device_vector_view<const T, IdxT>(d_cluster_counts.data_handle(), n_clusters);

    finalize_centroids<T, IdxT>(root_res,
                                d_centroid_sums_const,
                                d_cluster_counts_const,
                                centroids_const,
                                new_centroids.view());

    // 5. Convergence check
    auto sqrdNorm = raft::make_device_scalar<T>(root_res, T{0});
    raft::linalg::mapThenSumReduce(sqrdNorm.data_handle(),
                                   centroids.size(),
                                   raft::sqdiff_op{},
                                   root_stream,
                                   new_centroids.data_handle(),
                                   centroids.data_handle());

    raft::copy(centroids.data_handle(), new_centroids.data_handle(), centroids.size(), root_stream);

    T sqrdNormError = 0;
    raft::copy(&sqrdNormError, sqrdNorm.data_handle(), 1, root_stream);

    bool done = false;
    if (params.inertia_check && n_iter[0] > 1) {
      T total_cost = std::accumulate(rank_costs.begin(), rank_costs.end(), T{0});
      T delta      = total_cost / priorClusteringCost;
      if (delta > 1 - params.tol) done = true;
      priorClusteringCost = total_cost;
    }

    raft::resource::sync_stream(root_res, root_stream);
    if (sqrdNormError < params.tol) done = true;

    if (done) {
      RAFT_LOG_DEBUG("KMeans MG: Converged after %d iterations", n_iter[0]);
      break;
    }
  }  // end iteration loop

  // ========== Final inertia computation (multi-GPU) ==========
  if (params.final_inertia_check) {
    inertia[0] = 0;
    std::vector<T> rank_final_costs(num_ranks, T{0});

    // Copy final centroids to host
    raft::resource::set_current_device_to_root_rank(clique);
    raft::copy(h_centroids.data_handle(), centroids.data_handle(), centroids.size(), root_stream);
    raft::resource::sync_stream(root_res, root_stream);

    cuvs::core::omp::check_threads(num_ranks);
#pragma omp parallel for num_threads(num_ranks)
    for (int rank = 0; rank < num_ranks; rank++) {
      cuvs::core::omp::set_num_threads(threads_per_rank);

      const raft::resources& dev_res = raft::resource::set_current_device_to_rank(clique, rank);
      cudaStream_t stream            = raft::resource::get_cuda_stream(dev_res);

      IdxT start   = rank * rows_per_gpu;
      IdxT end     = std::min(start + rows_per_gpu, n_samples);
      IdxT local_n = end - start;

      if (local_n <= 0) {
        cuvs::core::omp::set_num_threads(omp_threads);
        continue;
      }

      auto gpu_centroids = raft::make_device_matrix<T, IdxT>(dev_res, n_clusters, n_features);
      raft::copy(gpu_centroids.data_handle(), h_centroids.data_handle(), centroids.size(), stream);

      auto gpu_centroids_const = raft::make_device_matrix_view<const T, IdxT>(
        gpu_centroids.data_handle(), n_clusters, n_features);

      IdxT effective_batch = std::min(batch_size, local_n);
      auto minClusterAndDistance =
        raft::make_device_vector<raft::KeyValuePair<IdxT, T>, IdxT>(dev_res, effective_batch);
      auto L2NormBatch = raft::make_device_vector<T, IdxT>(dev_res, effective_batch);
      rmm::device_uvector<T> L2NormBuf_OR_DistBuf(0, stream);
      rmm::device_uvector<char> workspace(0, stream);
      rmm::device_scalar<T> clusterCostD(stream);

      T local_cost = 0;

      using namespace cuvs::spatial::knn::detail::utils;
      const T* local_data = X.data_handle() + start * n_features;
      batch_load_iterator<T> data_batches(local_data, local_n, n_features, batch_size, stream);

      for (const auto& data_batch : data_batches) {
        IdxT current_batch_size = static_cast<IdxT>(data_batch.size());

        auto batch_data_view = raft::make_device_matrix_view<const T, IdxT>(
          data_batch.data(), current_batch_size, n_features);

        if (metric == cuvs::distance::DistanceType::L2Expanded ||
            metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
          raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
            L2NormBatch.data_handle(), data_batch.data(), n_features, current_batch_size, stream);
        }

        auto L2NormBatch_const = raft::make_device_vector_view<const T, IdxT>(
          L2NormBatch.data_handle(), current_batch_size);
        auto minClusterAndDistance_view =
          raft::make_device_vector_view<raft::KeyValuePair<IdxT, T>, IdxT>(
            minClusterAndDistance.data_handle(), current_batch_size);

        cuvs::cluster::kmeans::detail::minClusterAndDistanceCompute<T, IdxT>(
          dev_res,
          batch_data_view,
          gpu_centroids_const,
          minClusterAndDistance_view,
          L2NormBatch_const,
          L2NormBuf_OR_DistBuf,
          metric,
          params.batch_samples,
          params.batch_centroids,
          workspace);

        cuvs::cluster::kmeans::detail::computeClusterCost(
          dev_res,
          minClusterAndDistance_view,
          workspace,
          raft::make_device_scalar_view(clusterCostD.data()),
          raft::value_op{},
          raft::add_op{});

        auto cost_host = raft::make_host_scalar<T>(0);
        raft::copy(cost_host.data_handle(), clusterCostD.data(), 1, stream);
        raft::resource::sync_stream(dev_res, stream);
        local_cost += cost_host.data_handle()[0];
      }

      rank_final_costs[rank] = local_cost;
      cuvs::core::omp::set_num_threads(omp_threads);
    }

    inertia[0] = std::accumulate(rank_final_costs.begin(), rank_final_costs.end(), T{0});
    RAFT_LOG_DEBUG("KMeans MG: Final inertia=%f", static_cast<double>(inertia[0]));
  } else {
    inertia[0] = 0;
    RAFT_LOG_DEBUG("KMeans MG: Completed (inertia computation skipped)");
  }

  // Restore OMP settings
  cuvs::core::omp::set_nested(saved_nested);
}

/**
 * @brief Multi-GPU prediction for host data.
 *
 * Distributes prediction across GPUs, each processing a partition of the data.
 *
 * @param[in]     clique          SNMG resources
 * @param[in]     params          K-means parameters
 * @param[in]     X               Input data on HOST [n_samples x n_features]
 * @param[in]     batch_size      Rows loaded per GPU per sub-batch
 * @param[in]     sample_weight   Optional per-sample weights (on host)
 * @param[in]     centroids       Cluster centers on root device [n_clusters x n_features]
 * @param[out]    labels          Predicted labels on HOST [n_samples]
 * @param[in]     normalize_weight  Whether to normalize weights
 * @param[out]    inertia         Sum of squared distances
 */
template <typename T, typename IdxT>
void predict_mg(raft::resources const& clique,
                const cuvs::cluster::kmeans::params& params,
                raft::host_matrix_view<const T, IdxT> X,
                IdxT batch_size,
                std::optional<raft::host_vector_view<const T, IdxT>> sample_weight,
                raft::device_matrix_view<const T, IdxT> centroids,
                raft::host_vector_view<IdxT, IdxT> labels,
                bool normalize_weight,
                raft::host_scalar_view<T> inertia)
{
  int num_ranks   = raft::resource::get_num_ranks(clique);
  auto n_samples  = X.extent(0);
  auto n_features = X.extent(1);
  auto n_clusters = params.n_clusters;

  RAFT_EXPECTS(batch_size > 0, "batch_size must be positive");
  RAFT_EXPECTS(labels.extent(0) == n_samples, "labels.extent(0) must equal n_samples");

  // Copy centroids to host for distribution
  const raft::resources& root_res = raft::resource::set_current_device_to_root_rank(clique);
  cudaStream_t root_stream        = raft::resource::get_cuda_stream(root_res);

  auto h_centroids = raft::make_host_matrix<T, IdxT>(n_clusters, n_features);
  raft::copy(h_centroids.data_handle(), centroids.data_handle(), centroids.size(), root_stream);
  raft::resource::sync_stream(root_res, root_stream);

  IdxT rows_per_gpu = raft::ceildiv(n_samples, static_cast<IdxT>(num_ranks));

  std::vector<T> rank_inertias(num_ranks, T{0});

  int saved_nested     = cuvs::core::omp::get_nested();
  int omp_threads      = cuvs::core::omp::get_max_threads();
  int threads_per_rank = std::max(1, omp_threads / num_ranks);
  cuvs::core::omp::set_nested(1);

  cuvs::core::omp::check_threads(num_ranks);
#pragma omp parallel for num_threads(num_ranks)
  for (int rank = 0; rank < num_ranks; rank++) {
    cuvs::core::omp::set_num_threads(threads_per_rank);

    const raft::resources& dev_res = raft::resource::set_current_device_to_rank(clique, rank);
    cudaStream_t stream            = raft::resource::get_cuda_stream(dev_res);

    IdxT start   = rank * rows_per_gpu;
    IdxT end     = std::min(start + rows_per_gpu, n_samples);
    IdxT local_n = end - start;

    if (local_n <= 0) {
      cuvs::core::omp::set_num_threads(omp_threads);
      continue;
    }

    // Per-GPU: copy centroids from host
    auto gpu_centroids = raft::make_device_matrix<T, IdxT>(dev_res, n_clusters, n_features);
    raft::copy(gpu_centroids.data_handle(), h_centroids.data_handle(), centroids.size(), stream);

    IdxT effective_batch = std::min(batch_size, local_n);
    auto batch_data      = raft::make_device_matrix<T, IdxT>(dev_res, effective_batch, n_features);
    auto batch_weights   = raft::make_device_vector<T, IdxT>(dev_res, effective_batch);
    auto batch_labels    = raft::make_device_vector<IdxT, IdxT>(dev_res, effective_batch);

    T local_inertia = 0;

    for (IdxT batch_start = 0; batch_start < local_n; batch_start += batch_size) {
      IdxT current_batch_size = std::min(batch_size, local_n - batch_start);

      raft::copy(batch_data.data_handle(),
                 X.data_handle() + (start + batch_start) * n_features,
                 current_batch_size * n_features,
                 stream);

      if (sample_weight) {
        raft::copy(batch_weights.data_handle(),
                   sample_weight->data_handle() + start + batch_start,
                   current_batch_size,
                   stream);
      }

      auto batch_data_view = raft::make_device_matrix_view<const T, IdxT>(
        batch_data.data_handle(), current_batch_size, n_features);

      auto gpu_centroids_const = raft::make_device_matrix_view<const T, IdxT>(
        gpu_centroids.data_handle(), n_clusters, n_features);

      T batch_inertia = 0;
      cuvs::cluster::kmeans::detail::kmeans_predict<T, IdxT>(
        dev_res,
        params,
        batch_data_view,
        batch_weights.view(),
        gpu_centroids_const,
        batch_labels.view(),
        normalize_weight,
        raft::make_host_scalar_view(&batch_inertia));

      // Copy labels back to host
      raft::copy(labels.data_handle() + start + batch_start,
                 batch_labels.data_handle(),
                 current_batch_size,
                 stream);

      local_inertia += batch_inertia;
    }

    raft::resource::sync_stream(dev_res, stream);
    rank_inertias[rank] = local_inertia;
    cuvs::core::omp::set_num_threads(omp_threads);
  }

  cuvs::core::omp::set_nested(saved_nested);
  inertia[0] = std::accumulate(rank_inertias.begin(), rank_inertias.end(), T{0});
}

/**
 * @brief Multi-GPU fit + predict.
 */
template <typename T, typename IdxT>
void fit_predict_mg(raft::resources const& clique,
                    const cuvs::cluster::kmeans::params& params,
                    raft::host_matrix_view<const T, IdxT> X,
                    IdxT batch_size,
                    std::optional<raft::host_vector_view<const T, IdxT>> sample_weight,
                    raft::device_matrix_view<T, IdxT> centroids,
                    raft::host_vector_view<IdxT, IdxT> labels,
                    raft::host_scalar_view<T> inertia,
                    raft::host_scalar_view<IdxT> n_iter)
{
  T fit_inertia = 0;
  fit_mg<T, IdxT>(clique,
                  params,
                  X,
                  batch_size,
                  sample_weight,
                  centroids,
                  raft::make_host_scalar_view(&fit_inertia),
                  n_iter);

  auto centroids_const = raft::make_device_matrix_view<const T, IdxT>(
    centroids.data_handle(), centroids.extent(0), centroids.extent(1));

  predict_mg<T, IdxT>(
    clique, params, X, batch_size, sample_weight, centroids_const, labels, false, inertia);
}

}  // namespace cuvs::cluster::kmeans::detail
