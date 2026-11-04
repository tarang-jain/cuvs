#include <cuvs/cluster/kmeans.hpp>
#include <gtest/gtest.h>
#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resources.hpp>
#include <rmm/cuda_stream_pool.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace cuvs {

template <typename InputT>
class ByteKmeansTest : public ::testing::Test {};

using ByteKmeansTypes = ::testing::Types<int8_t, uint8_t>;
TYPED_TEST_SUITE(ByteKmeansTest, ByteKmeansTypes);

TYPED_TEST(ByteKmeansTest, WeightedFractionalMeansAndStreaming)
{
  using InputT               = TypeParam;
  constexpr int64_t rows     = 7;
  constexpr int64_t features = 17;
  constexpr int64_t clusters = 3;
  const int origin           = std::is_signed_v<InputT> ? -110 : 170;
  std::vector<InputT> input(rows * features);
  std::vector<float> weights{1, 2, 3, 1, 4, 2, 1};
  std::vector<float> initial(clusters * features, 1000.0f);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t feature = 0; feature < features; ++feature) {
      input[row * features + feature] = origin + (row < 3 ? row : 40 + row) + feature % 3;
    }
  }
  for (int64_t feature = 0; feature < features; ++feature) {
    initial[feature]            = input[feature];
    initial[features + feature] = input[3 * features + feature];
  }
  std::vector<float> expected = initial;
  for (int64_t cluster = 0; cluster < 2; ++cluster) {
    const int64_t begin = cluster == 0 ? 0 : 3;
    const int64_t end   = cluster == 0 ? 3 : rows;
    for (int64_t feature = 0; feature < features; ++feature) {
      double sum   = 0;
      double count = 0;
      for (int64_t row = begin; row < end; ++row) {
        sum += static_cast<double>(input[row * features + feature]) * weights[row];
        count += weights[row];
      }
      expected[cluster * features + feature] = sum / count;
    }
  }
  raft::resources handle;
  raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
  auto stream         = raft::resource::get_cuda_stream(handle);
  auto data           = raft::make_device_matrix<InputT, int64_t>(handle, rows, features);
  auto device_weights = raft::make_device_vector<float, int64_t>(handle, rows);
  auto centers        = raft::make_device_matrix<float, int64_t>(handle, clusters, features);
  auto labels         = raft::make_device_vector<int64_t, int64_t>(handle, rows);
  auto transformed    = raft::make_device_matrix<float, int64_t>(handle, rows, clusters);
  raft::copy(data.data_handle(), input.data(), input.size(), stream);
  raft::copy(device_weights.data_handle(), weights.data(), weights.size(), stream);
  cluster::kmeans::params params;
  params.n_clusters    = clusters;
  params.init          = cluster::kmeans::params::Array;
  params.max_iter      = 1;
  params.batch_samples = 3;
  for (bool host : {false, true}) {
    params.device_buffer_samples = host ? 2 : 0;
    raft::copy(centers.data_handle(), initial.data(), initial.size(), stream);
    float inertia;
    int64_t iterations;
    if (host) {
      cluster::kmeans::fit(
        handle,
        params,
        raft::make_host_matrix_view<const InputT, int64_t>(input.data(), rows, features),
        std::make_optional(raft::make_host_vector_view<const float, int64_t>(weights.data(), rows)),
        centers.view(),
        raft::make_host_scalar_view(&inertia),
        raft::make_host_scalar_view(&iterations));
    } else {
      cluster::kmeans::fit(handle,
                           params,
                           raft::make_const_mdspan(data.view()),
                           std::make_optional(raft::make_const_mdspan(device_weights.view())),
                           centers.view(),
                           raft::make_host_scalar_view(&inertia),
                           raft::make_host_scalar_view(&iterations));
    }
    std::vector<float> actual(centers.size());
    raft::copy(actual.data(), centers.data_handle(), actual.size(), stream);
    raft::resource::sync_stream(handle);
    EXPECT_EQ(iterations, 1);
    for (std::size_t element = 0; element < actual.size(); ++element) {
      EXPECT_NEAR(actual[element], expected[element], 2e-5);
    }
    float predicted_inertia;
    cluster::kmeans::predict(handle,
                             params,
                             raft::make_const_mdspan(data.view()),
                             std::make_optional(raft::make_const_mdspan(device_weights.view())),
                             raft::make_const_mdspan(centers.view()),
                             labels.view(),
                             true,
                             raft::make_host_scalar_view(&predicted_inertia));
    cluster::kmeans::transform(handle,
                               params,
                               raft::make_const_mdspan(data.view()),
                               raft::make_const_mdspan(centers.view()),
                               transformed.view());
    std::vector<int64_t> actual_labels(rows);
    std::vector<float> actual_distances(rows * clusters);
    raft::copy(actual_labels.data(), labels.data_handle(), rows, stream);
    raft::copy(actual_distances.data(), transformed.data_handle(), actual_distances.size(), stream);
    raft::resource::sync_stream(handle);
    double reference_inertia = 0;
    for (int64_t row = 0; row < rows; ++row) {
      double best     = std::numeric_limits<double>::max();
      int64_t nearest = -1;
      for (int64_t center = 0; center < clusters; ++center) {
        double distance = 0;
        for (int64_t feature = 0; feature < features; ++feature) {
          double difference = input[row * features + feature] - actual[center * features + feature];
          distance += difference * difference;
        }
        EXPECT_NEAR(
          actual_distances[row * clusters + center], distance, std::max(1e-5, distance * 1e-6));
        if (distance < best) {
          best    = distance;
          nearest = center;
        }
      }
      EXPECT_EQ(actual_labels[row], nearest);
      reference_inertia += best * weights[row] * rows / 14.0;
    }
    EXPECT_NEAR(inertia, reference_inertia, 1e-3);
    EXPECT_NEAR(predicted_inertia, reference_inertia, 1e-3);
    float cost;
    cluster::kmeans::cluster_cost(
      handle,
      raft::make_const_mdspan(data.view()),
      raft::make_const_mdspan(centers.view()),
      raft::make_host_scalar_view(&cost),
      std::make_optional(raft::make_const_mdspan(device_weights.view())));
    EXPECT_NEAR(cost, reference_inertia * 14.0 / rows, 1e-3);
  }
}

TYPED_TEST(ByteKmeansTest, SeedingAndIdenticalSamples)
{
  using InputT               = TypeParam;
  constexpr int64_t rows     = 33;
  constexpr int64_t features = 16;
  constexpr int64_t clusters = 3;
  raft::resources handle;
  auto stream  = raft::resource::get_cuda_stream(handle);
  auto data    = raft::make_device_matrix<InputT, int64_t>(handle, rows, features);
  auto centers = raft::make_device_matrix<float, int64_t>(handle, clusters, features);
  auto labels  = raft::make_device_vector<int64_t, int64_t>(handle, rows);
  std::vector<InputT> input(rows * features);
  for (bool identical : {false, true}) {
    for (int64_t row = 0; row < rows; ++row) {
      for (int64_t feature = 0; feature < features; ++feature) {
        input[row * features + feature] =
          identical ? std::numeric_limits<InputT>::max()
                    : std::numeric_limits<InputT>::min() + (row % clusters) * 70;
      }
    }
    raft::copy(data.data_handle(), input.data(), input.size(), stream);
    for (bool host : {false, true}) {
      for (int seeding = 0; seeding < 3; ++seeding) {
        cluster::kmeans::params params;
        params.n_clusters            = clusters;
        params.n_init                = 2;
        params.max_iter              = 10;
        params.device_buffer_samples = host ? 7 : 0;
        params.init_size             = rows;
        params.oversampling_factor   = seeding == 0 ? 0 : 2;
        params.init =
          seeding == 2 ? cluster::kmeans::params::Random : cluster::kmeans::params::KMeansPlusPlus;
        float inertia;
        int64_t iterations;
        if (host) {
          cluster::kmeans::fit(
            handle,
            params,
            raft::make_host_matrix_view<const InputT, int64_t>(input.data(), rows, features),
            std::nullopt,
            centers.view(),
            raft::make_host_scalar_view(&inertia),
            raft::make_host_scalar_view(&iterations));
        } else {
          cluster::kmeans::fit_predict(handle,
                                       params,
                                       raft::make_const_mdspan(data.view()),
                                       std::nullopt,
                                       std::make_optional(centers.view()),
                                       labels.view(),
                                       raft::make_host_scalar_view(&inertia),
                                       raft::make_host_scalar_view(&iterations));
        }
        EXPECT_TRUE(std::isfinite(inertia));
        EXPECT_GE(iterations, 1);
        EXPECT_LE(iterations, params.max_iter);
        if (identical || seeding != 2) { EXPECT_NEAR(inertia, 0.0f, 1e-4); }
      }
    }
  }
}

TYPED_TEST(ByteKmeansTest, InvalidShapesAndInitialization)
{
  using InputT = TypeParam;
  raft::resources handle;
  auto data = raft::make_device_matrix<InputT, int64_t>(handle, 4, 2);
  std::vector<InputT> host_data(8, InputT{0});
  auto host_input = raft::make_host_matrix_view<const InputT, int64_t>(host_data.data(), 4, 2);
  raft::copy(data.data_handle(),
             host_data.data(),
             host_data.size(),
             raft::resource::get_cuda_stream(handle));
  auto centers     = raft::make_device_matrix<float, int64_t>(handle, 2, 2);
  auto labels      = raft::make_device_vector<int64_t, int64_t>(handle, 3);
  auto transformed = raft::make_device_matrix<float, int64_t>(handle, 3, 2);
  cluster::kmeans::params params;
  params.n_clusters = 2;
  params.init_size  = 1;
  float inertia;
  int64_t iterations;
  EXPECT_THROW(cluster::kmeans::fit(handle,
                                    params,
                                    host_input,
                                    std::nullopt,
                                    centers.view(),
                                    raft::make_host_scalar_view(&inertia),
                                    raft::make_host_scalar_view(&iterations)),
               raft::logic_error);
  EXPECT_NO_THROW(cluster::kmeans::fit(handle,
                                       params,
                                       raft::make_const_mdspan(data.view()),
                                       std::nullopt,
                                       centers.view(),
                                       raft::make_host_scalar_view(&inertia),
                                       raft::make_host_scalar_view(&iterations)));
  for (auto initialization : {cluster::kmeans::params::Array, cluster::kmeans::params::Random}) {
    params.init = initialization;
    EXPECT_NO_THROW(cluster::kmeans::fit(handle,
                                         params,
                                         host_input,
                                         std::nullopt,
                                         centers.view(),
                                         raft::make_host_scalar_view(&inertia),
                                         raft::make_host_scalar_view(&iterations)));
  }
  params.init_size = 0;
  EXPECT_THROW(cluster::kmeans::predict(handle,
                                        params,
                                        raft::make_const_mdspan(data.view()),
                                        std::nullopt,
                                        raft::make_const_mdspan(centers.view()),
                                        labels.view(),
                                        false,
                                        raft::make_host_scalar_view(&inertia)),
               raft::logic_error);
  EXPECT_THROW(cluster::kmeans::transform(handle,
                                          params,
                                          raft::make_const_mdspan(data.view()),
                                          raft::make_const_mdspan(centers.view()),
                                          transformed.view()),
               raft::logic_error);
  params.metric = distance::DistanceType::CosineExpanded;
  EXPECT_THROW(cluster::kmeans::fit(handle,
                                    params,
                                    raft::make_const_mdspan(data.view()),
                                    std::nullopt,
                                    centers.view(),
                                    raft::make_host_scalar_view(&inertia),
                                    raft::make_host_scalar_view(&iterations)),
               raft::logic_error);
}

}  // namespace cuvs
