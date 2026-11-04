/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../src/neighbors/detail/bounded_gemm.cuh"
#include <cuvs/neighbors/brute_force.hpp>
#include <gtest/gtest.h>
#include <raft/core/copy.cuh>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

namespace cuvs::neighbors::brute_force {
namespace {

void check_search(int64_t n,
                  int64_t m,
                  int64_t d,
                  int64_t k,
                  preprocessing::quantize::scalar::range_mode range,
                  float quantile   = 1.f,
                  bool square_root = false)
{
  raft::resources res;
  auto const stream = raft::resource::get_cuda_stream(res);
  std::vector<float> x(n * d), q(m * d);
  // Deterministic, nontrivial distances. Queries target corpus rows beyond the seed sample.
  for (int64_t i = 0; i < n * d; ++i) {
    x[i] = std::sin(float((i * 17 + i / d * 13) % 100003)) + float(i / d % 19) * .07f;
  }
  for (int64_t i = 0; i < m; ++i) {
    auto const row = (n - 1 - (i * 131) % n);
    for (int64_t j = 0; j < d; ++j) {
      q[i * d + j] = x[row * d + j] + .013f;
    }
  }
  if (m > 1 && quantile < 1.f) {
    // Both dataset and query outliers must be covered by their measured coding errors.
    x[(n - 1) * d] = 100.f;
    q[d]           = 110.f;
  }
  auto data      = raft::make_device_matrix<float, int64_t>(res, n, d);
  auto queries   = raft::make_device_matrix<float, int64_t>(res, m, d);
  auto ids       = raft::make_device_matrix<int64_t, int64_t>(res, m, k);
  auto distances = raft::make_device_matrix<float, int64_t>(res, m, k);
  raft::update_device(data.data_handle(), x.data(), x.size(), stream);
  raft::update_device(queries.data_handle(), q.data(), q.size(), stream);
  auto encoded =
    preprocessing::quantize::scalar::build(res,
                                           preprocessing::quantize::scalar::params{quantile, range},
                                           raft::make_const_mdspan(data.view()));
  index_params params;
  params.metric =
    square_root ? distance::DistanceType::L2SqrtUnexpanded : distance::DistanceType::L2Expanded;
  auto index = build(res, params, std::as_const(encoded).view());
  search(res,
         search_params{},
         index,
         raft::make_const_mdspan(data.view()),
         raft::make_const_mdspan(queries.view()),
         ids.view(),
         distances.view());
  std::vector<int64_t> actual(m * k);
  std::vector<float> actual_distances(m * k);
  raft::update_host(actual.data(), ids.data_handle(), actual.size(), stream);
  raft::update_host(
    actual_distances.data(), distances.data_handle(), actual_distances.size(), stream);
  raft::resource::sync_stream(res);
  for (int64_t i = 0; i < m; ++i) {
    std::vector<std::pair<double, int64_t>> expected(n);
    for (int64_t r = 0; r < n; ++r) {
      double sum = 0;
      for (int64_t j = 0; j < d; ++j) {
        double delta = double(q[i * d + j]) - x[r * d + j];
        sum += delta * delta;
      }
      expected[r] = {square_root ? std::sqrt(sum) : sum, r};
    }
    std::partial_sort(expected.begin(), expected.begin() + k, expected.end());
    std::set<int64_t> unique;
    for (int64_t rank = 0; rank < k; ++rank) {
      auto const id = actual[i * k + rank];
      ASSERT_GE(id, 0);
      ASSERT_LT(id, n);
      EXPECT_TRUE(unique.insert(id).second);
      EXPECT_NEAR(actual_distances[i * k + rank],
                  expected[rank].first,
                  5e-6 * std::max(1.0, expected[rank].first));
      // Equal-distance ties may choose different IDs; every returned point must meet the kth
      // radius.
      double sum = 0;
      for (int64_t j = 0; j < d; ++j) {
        double delta = double(q[i * d + j]) - x[id * d + j];
        sum += delta * delta;
      }
      EXPECT_LE(square_root ? std::sqrt(sum) : sum,
                expected[k - 1].first + 5e-6 * std::max(1.0, expected[k - 1].first));
      if (rank) { EXPECT_LE(actual_distances[i * k + rank - 1], actual_distances[i * k + rank]); }
    }
  }
}

TEST(BoundedSQSearch, PartialTilesAndAllNeighbors)
{
  check_search(17, 3, 48, 17, preprocessing::quantize::scalar::range_mode::global, 1.f, true);
}
TEST(BoundedSQSearch, Above256AndMultipleQueryTiles)
{
  check_search(257, 131, 272, 7, preprocessing::quantize::scalar::range_mode::per_dimension);
}
TEST(BoundedSQSearch, Clipped1024DimensionalVectors)
{
  check_search(1031, 5, 1024, 10, preprocessing::quantize::scalar::range_mode::per_dimension, .5f);
}
TEST(BoundedSQSearch, RunningBoundAcrossCorpusTiles)
{
  check_search(65553, 2, 16, 5, preprocessing::quantize::scalar::range_mode::global);
}

TEST(BoundedSQSearch, UnknownErrorRetainsEveryPairAndHandlesTies)
{
  raft::resources res;
  auto const stream   = raft::resource::get_cuda_stream(res);
  constexpr int64_t n = 137, m = 3, d = 16, k = n;
  std::vector<float> x(n * d, 0.f), q(m * d, 0.f);
  auto data      = raft::make_device_matrix<float, int64_t>(res, n, d);
  auto queries   = raft::make_device_matrix<float, int64_t>(res, m, d);
  auto codes     = raft::make_device_matrix<int8_t, int64_t>(res, n, d);
  auto ids       = raft::make_device_matrix<int64_t, int64_t>(res, m, k);
  auto distances = raft::make_device_matrix<float, int64_t>(res, m, k);
  raft::update_device(data.data_handle(), x.data(), x.size(), stream);
  raft::update_device(queries.data_handle(), q.data(), q.size(), stream);
  preprocessing::quantize::scalar::quantizer<float> quantizer{-1.f, 1.f};
  preprocessing::quantize::scalar::transform(
    res, quantizer, raft::make_const_mdspan(data.view()), codes.view());
  device_sq_dataset<float, int64_t, float> encoded{std::move(codes), std::move(quantizer)};
  EXPECT_TRUE(std::isinf(encoded.const_manifest_view().max_error));
  auto index = build(res, index_params{}, std::as_const(encoded).view());
  search(res,
         search_params{},
         index,
         raft::make_const_mdspan(data.view()),
         raft::make_const_mdspan(queries.view()),
         ids.view(),
         distances.view());
  std::vector<int64_t> actual(m * k);
  std::vector<float> result(m * k);
  raft::update_host(actual.data(), ids.data_handle(), actual.size(), stream);
  raft::update_host(result.data(), distances.data_handle(), result.size(), stream);
  raft::resource::sync_stream(res);
  for (int64_t i = 0; i < m; ++i) {
    std::set<int64_t> unique(actual.begin() + i * k, actual.begin() + (i + 1) * k);
    EXPECT_EQ(unique.size(), n);
    EXPECT_EQ(*unique.begin(), 0);
    EXPECT_EQ(*unique.rbegin(), n - 1);
  }
  for (auto value : result) {
    EXPECT_EQ(value, 0.f);
  }
}

TEST(BoundedSQSearch, LargeFiniteCoordinates)
{
  raft::resources res;
  auto const stream = raft::resource::get_cuda_stream(res);
  std::vector<float> x(3 * 16), q(16, -3e30f);
  for (int i = 0; i < 3 * 16; ++i) {
    x[i] = (i / 16 + 1) * 3e30f;
  }
  auto data      = raft::make_device_matrix<float, int64_t>(res, 3, 16);
  auto queries   = raft::make_device_matrix<float, int64_t>(res, 1, 16);
  auto ids       = raft::make_device_matrix<int64_t, int64_t>(res, 1, 3);
  auto distances = raft::make_device_matrix<float, int64_t>(res, 1, 3);
  raft::update_device(data.data_handle(), x.data(), x.size(), stream);
  raft::update_device(queries.data_handle(), q.data(), q.size(), stream);
  auto encoded = preprocessing::quantize::scalar::build(
    res, preprocessing::quantize::scalar::params{1.f}, raft::make_const_mdspan(data.view()));
  for (auto metric : {distance::DistanceType::L2Expanded,
                      distance::DistanceType::L2Unexpanded,
                      distance::DistanceType::L2SqrtExpanded,
                      distance::DistanceType::L2SqrtUnexpanded}) {
    index_params params;
    params.metric          = metric;
    bool const square_root = metric == distance::DistanceType::L2SqrtExpanded ||
                             metric == distance::DistanceType::L2SqrtUnexpanded;
    auto index = build(res, params, std::as_const(encoded).view());
    search(res,
           search_params{},
           index,
           raft::make_const_mdspan(data.view()),
           raft::make_const_mdspan(queries.view()),
           ids.view(),
           distances.view());
    std::vector<int64_t> actual(3);
    std::vector<float> result(3);
    raft::update_host(actual.data(), ids.data_handle(), 3, stream);
    raft::update_host(result.data(), distances.data_handle(), 3, stream);
    raft::resource::sync_stream(res);
    if (square_root) {
      EXPECT_EQ(actual, (std::vector<int64_t>{0, 1, 2}));
      for (int r = 0; r < 3; ++r) {
        // The Euclidean distance fits in FP32 even though its square does not.
        double const expected = 4.0 * (double(x[r * 16]) - double(q[0]));
        EXPECT_NEAR(result[r], expected, 1e-6 * expected);
      }
    } else {
      std::sort(actual.begin(), actual.end());
      EXPECT_EQ(actual, (std::vector<int64_t>{0, 1, 2}));
      for (float distance : result) {
        EXPECT_EQ(distance, std::numeric_limits<float>::infinity());
      }
    }
  }
}

TEST(BoundedSQSearch, ShapeAndMetricValidation)
{
  raft::resources res;
  auto data   = raft::make_device_matrix<float, int64_t>(res, 2, 16);
  auto codes  = raft::make_device_matrix<int8_t, int64_t>(res, 2, 16);
  auto stream = raft::resource::get_cuda_stream(res);
  RAFT_CUDA_TRY(cudaMemsetAsync(data.data_handle(), 0, 32 * sizeof(float), stream.get()));
  RAFT_CUDA_TRY(cudaMemsetAsync(codes.data_handle(), 0, 32, stream.get()));
  device_sq_dataset<float, int64_t, float> encoded{
    std::move(codes), preprocessing::quantize::scalar::quantizer<float>{0.f, 1.f}};
  auto ids       = raft::make_device_matrix<int64_t, int64_t>(res, 1, 1);
  auto distances = raft::make_device_matrix<float, int64_t>(res, 1, 1);
  auto index     = build(res, index_params{}, std::as_const(encoded).view());
  auto q         = raft::make_device_matrix_view<const float, int64_t>(data.data_handle(), 1, 16);
  EXPECT_THROW(
    search(res,
           search_params{},
           index,
           raft::make_device_matrix_view<const float, int64_t>(data.data_handle(), 1, 16),
           q,
           ids.view(),
           distances.view()),
    raft::logic_error);
  index_params params;
  params.metric = distance::DistanceType::InnerProduct;
  auto ip       = build(res, params, std::as_const(encoded).view());
  EXPECT_THROW(search(res,
                      search_params{},
                      ip,
                      raft::make_const_mdspan(data.view()),
                      q,
                      ids.view(),
                      distances.view()),
               raft::logic_error);
  // An empty query batch must not launch a zero-sized grid.
  search(res,
         search_params{},
         index,
         raft::make_const_mdspan(data.view()),
         raft::make_device_matrix_view<const float, int64_t>(nullptr, 0, 16),
         raft::make_device_matrix_view<int64_t, int64_t>(nullptr, 0, 1),
         raft::make_device_matrix_view<float, int64_t>(nullptr, 0, 1));
}

TEST(BoundedSQGemm, ExactIntegerPredicateBeyondFloatPrecision)
{
  raft::resources res;
  auto const stream   = raft::resource::get_cuda_stream(res);
  constexpr int64_t n = 129, m = 131, d = 1024;
  std::vector<int8_t> x(n * d), q(m * d);
  std::vector<double> xn(n), qn(m);
  std::vector<int64_t> limit(m);
  for (int64_t r = 0; r < n; ++r) {
    for (int64_t j = 0; j < d; ++j) {
      x[r * d + j] = r % 3 == 0 ? 127 : r % 3 == 1 ? -128 : 0;
      if (j == 0 && r % 3 == 0) { x[r * d + j] = 126; }
      xn[r] += int(x[r * d + j]) * int(x[r * d + j]);
    }
  }
  for (int64_t r = 0; r < m; ++r) {
    for (int64_t j = 0; j < d; ++j) {
      q[r * d + j] = -128;
      qn[r] += 128 * 128;
    }
    // Odd distance above 2^24: casting to float before comparing would change the verdict.
    limit[r] = d * 255 * 255 - 509 - r % 2;
  }
  auto xd    = raft::make_device_matrix<int8_t, int64_t>(res, n, d);
  auto qd    = raft::make_device_matrix<int8_t, int64_t>(res, m, d);
  auto xnd   = raft::make_device_vector<double, int64_t>(res, n);
  auto qnd   = raft::make_device_vector<double, int64_t>(res, m);
  auto ld    = raft::make_device_vector<int64_t, int64_t>(res, m);
  auto rows  = raft::make_device_vector<int32_t, int64_t>(res, m * n);
  auto ids   = raft::make_device_vector<int64_t, int64_t>(res, m * n);
  auto count = raft::make_device_vector<unsigned long long, int64_t>(res, 1);
  raft::update_device(xd.data_handle(), x.data(), x.size(), stream);
  raft::update_device(qd.data_handle(), q.data(), q.size(), stream);
  raft::update_device(xnd.data_handle(), xn.data(), xn.size(), stream);
  raft::update_device(qnd.data_handle(), qn.data(), qn.size(), stream);
  raft::update_device(ld.data_handle(), limit.data(), limit.size(), stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(count.data_handle(), 0, sizeof(unsigned long long), stream.get()));
  // One corpus tile in the grid makes each block walk two tiles, exercising pipeline reuse.
  cuvs::neighbors::detail::bounded_int8::fused_filter_kernel<<<dim3(2, 1), 256, 0, stream.get()>>>(
    xd.data_handle(),
    xnd.data_handle(),
    n,
    0,
    qd.data_handle(),
    qnd.data_handle(),
    m,
    d,
    ld.data_handle(),
    rows.data_handle(),
    ids.data_handle(),
    count.data_handle());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  unsigned long long total = 0;
  raft::update_host(&total, count.data_handle(), 1, stream);
  raft::resource::sync_stream(res);
  ASSERT_LE(total, m * n);
  std::vector<int32_t> host_rows(total);
  std::vector<int64_t> host_ids(total);
  raft::update_host(host_rows.data(), rows.data_handle(), total, stream);
  raft::update_host(host_ids.data(), ids.data_handle(), total, stream);
  raft::resource::sync_stream(res);
  std::vector<bool> seen(m * n);
  for (size_t i = 0; i < total; ++i) {
    ASSERT_GE(host_rows[i], 0);
    ASSERT_LT(host_rows[i], m);
    ASSERT_GE(host_ids[i], 0);
    ASSERT_LT(host_ids[i], n);
    auto const p = host_rows[i] * n + host_ids[i];
    EXPECT_FALSE(seen[p]);
    seen[p] = true;
  }
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t r = 0; r < n; ++r) {
      int64_t exact = 0;
      for (int64_t j = 0; j < d; ++j) {
        int delta = int(q[i * d + j]) - int(x[r * d + j]);
        exact += delta * delta;
      }
      EXPECT_EQ(seen[i * n + r], exact <= limit[i]) << "query " << i << ", row " << r;
    }
  }
}

}  // namespace
}  // namespace cuvs::neighbors::brute_force
