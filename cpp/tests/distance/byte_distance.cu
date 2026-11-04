#include "../../src/distance/detail/byte_distance.cuh"
#include "../../src/distance/top_1_nn.cuh"

#include <cuvs/distance/distance.hpp>
#include <gtest/gtest.h>
#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

#include <limits>
#include <vector>

namespace cuvs::distance {

template <typename InputT>
class ByteDistanceTest : public ::testing::Test {};

using ByteDistanceTypes = ::testing::Types<int8_t, uint8_t>;
TYPED_TEST_SUITE(ByteDistanceTest, ByteDistanceTypes);

TYPED_TEST(ByteDistanceTest, ExactOrderingTailsAndOverflow)
{
  using InputT = TypeParam;
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  for (int64_t features : {int64_t{17}, int64_t{1024}, int64_t{40000}}) {
    constexpr int64_t rows       = 3;
    constexpr int64_t candidates = 131;
    std::vector<InputT> input(rows * features, std::numeric_limits<InputT>::min());
    std::vector<InputT> centers(candidates * features, std::numeric_limits<InputT>::max());
    centers[129 * features] = std::numeric_limits<InputT>::max() - 1;
    centers[130 * features] = std::numeric_limits<InputT>::max() - 1;
    if (features == 1024) {
      for (int64_t candidate = 0; candidate < candidates; ++candidate) {
        centers[candidate * features] = std::numeric_limits<InputT>::min() + (candidate < 129);
      }
    }
    auto data         = raft::make_device_matrix<InputT, int64_t>(handle, rows, features);
    auto center_data  = raft::make_device_matrix<InputT, int64_t>(handle, candidates, features);
    auto pairwise     = raft::make_device_matrix<float, int64_t>(handle, rows, candidates);
    auto input_norms  = raft::make_device_vector<int64_t, int64_t>(handle, rows);
    auto center_norms = raft::make_device_vector<int64_t, int64_t>(handle, candidates);
    auto indices      = raft::make_device_vector<int64_t, int64_t>(handle, rows);
    auto distances    = raft::make_device_vector<float, int64_t>(handle, rows);
    raft::copy(data.data_handle(), input.data(), input.size(), stream);
    raft::copy(center_data.data_handle(), centers.data(), centers.size(), stream);
    detail::byte_norm(handle, raft::make_const_mdspan(data.view()), input_norms.view());
    detail::byte_norm(handle, raft::make_const_mdspan(center_data.view()), center_norms.view());
    pairwise_distance(handle,
                      raft::make_const_mdspan(data.view()),
                      raft::make_const_mdspan(center_data.view()),
                      pairwise.view(),
                      DistanceType::L2Expanded);
    std::vector<float> actual_pairwise(rows * candidates);
    raft::copy(actual_pairwise.data(), pairwise.data_handle(), actual_pairwise.size(), stream);
    raft::resource::sync_stream(handle);
    const int64_t full_distance = features * int64_t{255} * 255 - (features == 1024 ? 65024 : 0);
    const int64_t improvement   = features == 1024 ? 1 : 509;
    for (int64_t row = 0; row < rows; ++row) {
      for (int64_t candidate = 0; candidate < candidates; ++candidate) {
        const auto expected = candidate >= 129 ? full_distance - improvement : full_distance;
        EXPECT_EQ(actual_pairwise[row * candidates + candidate], static_cast<float>(expected));
      }
    }
    for (auto backend : {detail::Top1nnBackend::Auto,
                         detail::Top1nnBackend::Unfused,
                         detail::Top1nnBackend::Cutile}) {
      if (backend == detail::Top1nnBackend::Cutile &&
          !detail::is_top_1_nn_backend_available(backend,
                                                 data.data_handle(),
                                                 center_data.data_handle(),
                                                 rows,
                                                 candidates,
                                                 features,
                                                 DistanceType::L2Expanded)) {
        continue;
      }
      detail::Top1nnTuning tuning;
      auto plan = probe_top_1_nn(handle,
                                 data.data_handle(),
                                 center_data.data_handle(),
                                 rows,
                                 candidates,
                                 features,
                                 tuning,
                                 DistanceType::L2Expanded,
                                 backend,
                                 true,
                                 true);
      ASSERT_TRUE(plan.available);
      rmm::device_uvector<char> workspace(plan.workspace_bytes, stream);
      top_1_nn(handle,
               Top1nnOutput<int64_t, float>{indices.data_handle(), distances.data_handle()},
               data.data_handle(),
               center_data.data_handle(),
               input_norms.data_handle(),
               center_norms.data_handle(),
               rows,
               candidates,
               features,
               tuning,
               workspace.data(),
               workspace.size(),
               false,
               true,
               true,
               DistanceType::L2Expanded,
               0.0f,
               plan);
      std::vector<int64_t> actual_indices(rows);
      std::vector<float> actual_distances(rows);
      raft::copy(actual_indices.data(), indices.data_handle(), rows, stream);
      raft::copy(actual_distances.data(), distances.data_handle(), rows, stream);
      raft::resource::sync_stream(handle);
      for (int64_t row = 0; row < rows; ++row) {
        EXPECT_EQ(actual_indices[row], 129);
        EXPECT_EQ(actual_distances[row], static_cast<float>(full_distance - improvement));
      }
    }
  }
}

}  // namespace cuvs::distance
