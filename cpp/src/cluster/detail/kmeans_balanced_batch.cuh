/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../neighbors/detail/ann_utils.cuh"
#include "../kmeans.cuh"
#include "kmeans_common.cuh"

#include <raft/core/copy.cuh>
#include <raft/core/device_setter.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_id.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce_cols_by_key.cuh>
#include <raft/linalg/reduce_rows_by_key.cuh>
#include <raft/matrix/init.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>

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
#include <utility>
#include <vector>

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

inline void streamed_balanced_em(const raft::resources& handle,
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

  auto reduce_dataset = [&](bool predict) {
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
      auto current_rows = static_cast<index_type>(batch.size());
      auto offset       = static_cast<index_type>(batch.offset());
      if (predict) {
        raft::update_device(prediction.norms.data(), host_norms + offset, batch.size(), stream);
        predict_batch(handle,
                      params,
                      centers,
                      n_clusters,
                      dim,
                      batch.data(),
                      current_rows,
                      prediction.norms.data(),
                      prediction);
      } else {
        raft::update_device(prediction.labels.data(), host_labels + offset, batch.size(), stream);
      }

      raft::linalg::reduce_rows_by_key(batch.data(),
                                       dim,
                                       prediction.labels.data(),
                                       static_cast<char*>(nullptr),
                                       current_rows,
                                       dim,
                                       n_clusters,
                                       sums.data(),
                                       stream.get(),
                                       false);
      raft::linalg::reduce_cols_by_key(
        handle,
        raft::make_device_matrix_view<const counter_type, index_type>(
          count_weights.data(), index_type{1}, current_rows),
        raft::make_device_vector_view<const label_type, index_type>(prediction.labels.data(),
                                                                    current_rows),
        raft::make_device_matrix_view<counter_type, index_type>(
          counts.data(), index_type{1}, n_clusters),
        n_clusters,
        false);
      raft::update_host(host_labels + offset, prediction.labels.data(), batch.size(), stream);
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
    reduce_dataset(false);
    raft::copy(centers, next_centers.data(), center_elems, stream);
  }

  uint32_t balancing_counter = balancing_pullback;
  for (uint32_t iter = 0; iter < n_iters; ++iter) {
    if (iter > 0 && adjust_centers(handle,
                                   params,
                                   X,
                                   host_labels,
                                   host_sizes,
                                   lower_tolerance,
                                   upper_tolerance,
                                   centers,
                                   adjustment)) {
      if (balancing_counter++ >= balancing_pullback) {
        balancing_counter -= balancing_pullback;
        ++n_iters;
      }
    }
    reduce_dataset(true);
    raft::copy(centers, next_centers.data(), center_elems, stream);
  }
  raft::resource::sync_stream(handle, stream);
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
