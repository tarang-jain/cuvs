#pragma once

#include "kmeans_common.cuh"

#include <raft/linalg/reduce.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng.cuh>

namespace cuvs::cluster::kmeans::detail {

template <typename DataT, typename IndexT, typename Accessor, typename InputT>
void kmeans_fit(
  raft::resources const& handle,
  const cuvs::cluster::kmeans::params& params,
  raft::mdspan<const InputT, raft::matrix_extent<IndexT>, raft::row_major, Accessor> input,
  std::optional<raft::mdspan<const DataT,
                             raft::vector_extent<IndexT>,
                             raft::layout_right,
                             raft::host_device_accessor<cuda::std::default_accessor<const DataT>,
                                                        Accessor::mem_type>>> weights,
  raft::device_matrix_view<DataT, IndexT> centers,
  raft::host_scalar_view<DataT> inertia,
  raft::host_scalar_view<IndexT> iterations,
  std::optional<std::reference_wrapper<rmm::device_uvector<char>>> workspace = std::nullopt);

template <typename InputT, typename IndexT>
void byte_kmeans_plus_plus(raft::resources const& handle,
                           const params& params,
                           raft::device_matrix_view<const InputT, IndexT> input,
                           raft::device_matrix_view<InputT, IndexT> centers,
                           rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream      = raft::resource::get_cuda_stream(handle).get();
  const auto rows          = input.extent(0);
  const auto features      = input.extent(1);
  const IndexT trials      = 2 + static_cast<IndexT>(std::ceil(std::log(centers.extent(0))));
  auto indices             = raft::make_device_vector<IndexT>(handle, trials);
  auto candidates          = raft::make_device_matrix<InputT>(handle, trials, features);
  auto norms               = raft::make_device_vector<int64_t>(handle, rows);
  auto distances           = raft::make_device_vector<float>(handle, rows);
  auto candidate_distances = raft::make_device_matrix<float>(handle, trials, rows);
  auto costs               = raft::make_device_vector<float>(handle, trials);
  std::vector<float> host_costs(trials);
  cuvs::distance::detail::byte_norm(handle, input, norms.view());
  raft::random::RngState rng(params.rng_state.seed, params.rng_state.type);
  std::mt19937 generator(params.rng_state.seed);
  std::uniform_int_distribution<IndexT> choose(0, rows - 1);
  raft::copy(
    centers.data_handle(), input.data_handle() + choose(generator) * features, features, stream);
  byte_seed_assignment(
    handle,
    input,
    raft::make_device_matrix_view<const InputT, IndexT>(centers.data_handle(), 1, features),
    raft::make_const_mdspan(norms.view()),
    static_cast<IndexT*>(nullptr),
    distances.data_handle(),
    workspace);
  auto cost = raft::make_device_scalar<float>(handle, 0.0f);
  computeClusterCost(
    handle, distances.view(), workspace, cost.view(), raft::identity_op{}, raft::add_op{});
  float current_cost;
  raft::copy(&current_cost, cost.data_handle(), 1, stream);
  raft::resource::sync_stream(handle);
  for (IndexT picked = 1; picked < centers.extent(0); ++picked) {
    if (current_cost == 0.0f) {
      raft::matrix::sample_rows(
        handle,
        rng,
        input,
        raft::make_device_matrix_view(
          centers.data_handle() + picked * features, centers.extent(0) - picked, features));
      break;
    }
    raft::random::discrete(handle, rng, indices.view(), raft::make_const_mdspan(distances.view()));
    raft::matrix::gather(handle, input, raft::make_const_mdspan(indices.view()), candidates.view());
    cuvs::distance::detail::byte_pairwise_distance(handle,
                                                   raft::make_const_mdspan(candidates.view()),
                                                   input,
                                                   candidate_distances.view(),
                                                   cuvs::distance::DistanceType::L2Expanded);
    raft::linalg::matrix_vector_op<raft::Apply::ALONG_ROWS>(
      handle,
      raft::make_const_mdspan(candidate_distances.view()),
      raft::make_const_mdspan(distances.view()),
      candidate_distances.view(),
      raft::min_op{});
    raft::linalg::reduce<raft::Apply::ALONG_ROWS>(
      handle, raft::make_const_mdspan(candidate_distances.view()), costs.view(), 0.0f);
    raft::copy(host_costs.data(), costs.data_handle(), trials, stream);
    raft::resource::sync_stream(handle);
    const auto best = std::min_element(host_costs.begin(), host_costs.end()) - host_costs.begin();
    current_cost    = host_costs[best];
    raft::copy(
      distances.data_handle(), candidate_distances.data_handle() + best * rows, rows, stream);
    raft::copy(centers.data_handle() + picked * features,
               candidates.data_handle() + best * features,
               features,
               stream);
  }
}

template <typename InputT, typename IndexT>
void init_byte_centroids(raft::resources const& handle,
                         const params& params,
                         raft::device_matrix_view<const InputT, IndexT> input,
                         raft::device_matrix_view<float, IndexT> centers,
                         rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream   = raft::resource::get_cuda_stream(handle).get();
  const auto rows       = input.extent(0);
  const auto features   = input.extent(1);
  const IndexT clusters = params.n_clusters;
  auto seeds            = raft::make_device_matrix<InputT, IndexT>(handle, clusters, features);
  auto copy_seeds       = [&] {
    raft::linalg::unaryOp(
      centers.data_handle(), seeds.data_handle(), seeds.size(), raft::cast_op<float>{}, stream);
  };
  if (params.oversampling_factor == 0) {
    byte_kmeans_plus_plus(handle, params, input, seeds.view(), workspace);
    copy_seeds();
    return;
  }
  raft::random::RngState rng(params.rng_state.seed, params.rng_state.type);
  std::mt19937 generator(params.rng_state.seed);
  std::uniform_int_distribution<IndexT> choose(0, rows - 1);
  const auto first = choose(generator);
  rmm::device_uvector<InputT> potential(features, stream);
  raft::copy(potential.data(), input.data_handle() + first * features, features, stream);
  auto selected = raft::make_device_vector<uint8_t>(handle, rows);
  raft::matrix::fill(handle, selected.view(), uint8_t{0});
  raft::linalg::map(handle,
                    raft::make_device_vector_view(selected.data_handle() + first, IndexT{1}),
                    raft::const_op<uint8_t>{1});
  auto norms = raft::make_device_vector<int64_t>(handle, rows);
  cuvs::distance::detail::byte_norm(handle, input, norms.view());
  auto distances      = raft::make_device_vector<float>(handle, rows);
  auto next_distances = raft::make_device_vector<float>(handle, rows);
  auto random         = raft::make_device_vector<float>(handle, rows);
  auto cost           = raft::make_device_scalar<float>(handle, 0.0f);
  byte_seed_assignment(
    handle,
    input,
    raft::make_device_matrix_view<const InputT, IndexT>(potential.data(), 1, features),
    raft::make_const_mdspan(norms.view()),
    static_cast<IndexT*>(nullptr),
    distances.data_handle(),
    workspace);
  auto read_cost = [&] {
    computeClusterCost(
      handle, distances.view(), workspace, cost.view(), raft::identity_op{}, raft::add_op{});
    float value;
    raft::copy(&value, cost.data_handle(), 1, stream);
    raft::resource::sync_stream(handle);
    return value;
  };
  float current_cost = read_cost();
  const auto rounds =
    current_cost > 0 ? std::clamp(static_cast<int>(std::ceil(std::log(current_cost))), 0, 8) : 0;
  for (int round = 0; round < rounds && current_cost > 0; ++round) {
    raft::random::uniform(handle, rng, random.data_handle(), rows, 0.0f, 1.0f);
    SamplingOp<float, IndexT> select(current_cost,
                                     params.oversampling_factor,
                                     clusters,
                                     random.data_handle(),
                                     selected.data_handle());
    rmm::device_uvector<InputT> candidates(0, stream);
    sampleCentroids<float, IndexT>(
      handle, input, distances.view(), selected.view(), select, candidates, workspace);
    if (candidates.size() == 0) { continue; }
    const auto previous_size = potential.size();
    potential.resize(previous_size + candidates.size(), stream);
    raft::copy(potential.data() + previous_size, candidates.data(), candidates.size(), stream);
    if (round + 1 < rounds) {
      byte_seed_assignment(
        handle,
        input,
        raft::make_device_matrix_view<const InputT, IndexT>(
          candidates.data(), static_cast<IndexT>(candidates.size() / features), features),
        raft::make_const_mdspan(norms.view()),
        static_cast<IndexT*>(nullptr),
        next_distances.data_handle(),
        workspace);
      raft::linalg::map(handle,
                        distances.view(),
                        raft::min_op{},
                        raft::make_const_mdspan(distances.view()),
                        raft::make_const_mdspan(next_distances.view()));
      current_cost = read_cost();
    }
  }
  const IndexT count = potential.size() / features;
  auto potential_view =
    raft::make_device_matrix_view<const InputT, IndexT>(potential.data(), count, features);
  if (count > clusters) {
    auto labels = raft::make_device_vector<IndexT>(handle, rows);
    byte_seed_assignment(handle,
                         input,
                         potential_view,
                         raft::make_const_mdspan(norms.view()),
                         labels.data_handle(),
                         distances.data_handle(),
                         workspace);
    auto weights = raft::make_device_vector<float>(handle, count);
    raft::matrix::fill(handle, random.view(), 1.0f);
    raft::linalg::reduce_cols_by_key(random.data_handle(),
                                     labels.data_handle(),
                                     weights.data_handle(),
                                     IndexT{1},
                                     rows,
                                     count,
                                     stream);
    byte_kmeans_plus_plus(handle, params, potential_view, seeds.view(), workspace);
    copy_seeds();
    auto recluster   = params;
    recluster.init   = params::Array;
    recluster.n_init = 1;
    float inertia;
    IndexT iterations;
    kmeans_fit<float, IndexT>(handle,
                              recluster,
                              potential_view,
                              std::make_optional(raft::make_const_mdspan(weights.view())),
                              centers,
                              raft::make_host_scalar_view(&inertia),
                              raft::make_host_scalar_view(&iterations),
                              std::ref(workspace));
  } else {
    raft::matrix::sample_rows(handle, rng, input, seeds.view());
    raft::copy(seeds.data_handle(), potential.data(), potential.size(), stream);
    copy_seeds();
  }
}

}  // namespace cuvs::cluster::kmeans::detail
