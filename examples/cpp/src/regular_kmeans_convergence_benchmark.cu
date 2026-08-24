/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/cluster/kmeans.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng_state.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct case_spec {
  std::string name;
  cuvs::cluster::kmeans::params::InitMethod init;
  double oversampling;
  int64_t init_size;
  std::string step8{"kmeans++"};
  int n_init{1};
};

struct fit_result {
  double seconds;
  float inertia;
  int64_t n_iter;
};

struct options {
  std::string dataset{"/datasets/blobs-5M-1536-k1024/base.fbin"};
  int64_t rows{5'000'000};
  int n_clusters{1024};
  int max_iter{300};
  double tol{1e-6};
  std::unordered_set<std::string> cases;
};

void cuda_check(cudaError_t status, const char* operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

options parse_options(int argc, char** argv)
{
  options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next             = [&]() -> std::string {
      if (++i >= argc) { throw std::invalid_argument("missing value after " + arg); }
      return argv[i];
    };
    if (arg == "--dataset") {
      opts.dataset = next();
    } else if (arg == "--rows") {
      opts.rows = std::stoll(next());
    } else if (arg == "--n-clusters") {
      opts.n_clusters = std::stoi(next());
    } else if (arg == "--max-iter") {
      opts.max_iter = std::stoi(next());
    } else if (arg == "--tol") {
      opts.tol = std::stod(next());
    } else if (arg == "--cases") {
      while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        opts.cases.emplace(argv[++i]);
      }
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  return opts;
}

auto load_fbin(const raft::resources& res, const options& opts)
  -> raft::device_matrix<float, int64_t>
{
  std::ifstream input(opts.dataset, std::ios::binary);
  if (!input) { throw std::runtime_error("cannot open " + opts.dataset); }

  int32_t file_rows = 0;
  int32_t dim       = 0;
  input.read(reinterpret_cast<char*>(&file_rows), sizeof(file_rows));
  input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  if (!input || opts.rows <= 0 || opts.rows > file_rows || dim <= 0) {
    throw std::runtime_error("invalid fbin dimensions or requested row count");
  }

  const int64_t rows = opts.rows;
  std::cout << "loading " << rows << " x " << dim << " ("
            << (static_cast<double>(rows) * dim * sizeof(float) / 1e9)
            << " GB) to an int64-indexed device matrix\n";
  auto data = raft::make_device_matrix<float, int64_t>(res, rows, dim);

  constexpr int64_t chunk_rows = 125'000;
  std::vector<float> host(static_cast<size_t>(chunk_rows) * dim);
  for (int64_t row = 0; row < rows; row += chunk_rows) {
    const int64_t count_rows = std::min(chunk_rows, rows - row);
    const size_t count       = static_cast<size_t>(count_rows) * dim;
    input.read(reinterpret_cast<char*>(host.data()), count * sizeof(float));
    if (!input) { throw std::runtime_error("short read from " + opts.dataset); }
    cuda_check(cudaMemcpy(data.data_handle() + row * static_cast<int64_t>(dim),
                          host.data(),
                          count * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy H2D");
    if ((row + count_rows) % 1'000'000 == 0 || row + count_rows == rows) {
      std::cout << "  H2D " << row + count_rows << '/' << rows << '\n';
    }
  }
  return data;
}

fit_result run_fit(const raft::resources& res,
                   raft::device_matrix_view<const float, int64_t> data,
                   const case_spec& spec,
                   int n_clusters,
                   int max_iter,
                   double tol)
{
  setenv("CUVS_KMEANS_PP_STEP8", spec.step8.c_str(), 1);
  cuvs::cluster::kmeans::params params;
  params.n_clusters         = n_clusters;
  params.init               = spec.init;
  params.max_iter           = max_iter;
  params.tol                = tol;
  params.n_init             = spec.n_init;
  params.oversampling_factor = spec.oversampling;
  params.init_size          = spec.init_size;
  params.rng_state          = raft::random::RngState{0};

  auto centroids =
    raft::make_device_matrix<float, int64_t>(res, n_clusters, data.extent(1));
  float inertia  = 0.0f;
  int64_t n_iter = 0;
  raft::resource::sync_stream(res);
  const auto begin = std::chrono::steady_clock::now();
  cuvs::cluster::kmeans::fit(res,
                             params,
                             data,
                             std::nullopt,
                             centroids.view(),
                             raft::make_host_scalar_view(&inertia),
                             raft::make_host_scalar_view(&n_iter));
  raft::resource::sync_stream(res);
  const double seconds =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  return {seconds, inertia, n_iter};
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    const auto opts = parse_options(argc, argv);
    raft::resources res;
    const auto data = load_fbin(res, opts);
    const int64_t k = opts.n_clusters;

    const std::vector<case_spec> cases{
      {"random", cuvs::cluster::kmeans::params::Random, 2.0, 0},
      {"scalable_full_step8_kmeans++",
       cuvs::cluster::kmeans::params::KMeansPlusPlus,
       2.0,
       opts.rows},
      {"scalable_3k_step8_kmeans++",
       cuvs::cluster::kmeans::params::KMeansPlusPlus,
       2.0,
       3 * k},
      {"scalable_5k_step8_kmeans++",
       cuvs::cluster::kmeans::params::KMeansPlusPlus,
       2.0,
       5 * k},
      {"scalable_7k_step8_kmeans++",
       cuvs::cluster::kmeans::params::KMeansPlusPlus,
       2.0,
       7 * k},
      {"scalable_8k_step8_kmeans++",
       cuvs::cluster::kmeans::params::KMeansPlusPlus,
       2.0,
       8 * k},
      {"scalable_10k_step8_kmeans++",
       cuvs::cluster::kmeans::params::KMeansPlusPlus,
       2.0,
       10 * k},
      {"scalable_12k_step8_kmeans++",
       cuvs::cluster::kmeans::params::KMeansPlusPlus,
       2.0,
       12 * k},
      {"scalable_16k_step8_kmeans++",
       cuvs::cluster::kmeans::params::KMeansPlusPlus,
       2.0,
       16 * k},
      {"scalable_32k_step8_kmeans++",
       cuvs::cluster::kmeans::params::KMeansPlusPlus,
       2.0,
       32 * k},
      {"scalable_64k_step8_kmeans++",
       cuvs::cluster::kmeans::params::KMeansPlusPlus,
       2.0,
       64 * k},
      {"scalable_12k_step8_random",
       cuvs::cluster::kmeans::params::KMeansPlusPlus,
       2.0,
       12 * k,
       "random"},
      {"classic_12k_kmeans++",
       cuvs::cluster::kmeans::params::KMeansPlusPlus,
       0.0,
       12 * k},
      {"classic_32k_kmeans++",
       cuvs::cluster::kmeans::params::KMeansPlusPlus,
       0.0,
       32 * k},
    };

    const auto data_view = raft::make_device_matrix_view<const float, int64_t>(
      data.data_handle(), data.extent(0), data.extent(1));
    for (const auto& spec : cases) {
      if (!opts.cases.empty() && !opts.cases.contains(spec.name)) { continue; }
      const auto init = run_fit(res, data_view, spec, opts.n_clusters, 0, opts.tol);
      const auto full =
        run_fit(res, data_view, spec, opts.n_clusters, opts.max_iter, opts.tol);
      std::cout << "RESULT case=" << spec.name << " init_s=" << init.seconds
                << " total_s=" << full.seconds << " n_iter=" << full.n_iter
                << " init_inertia=" << init.inertia << " inertia=" << full.inertia << '\n';
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
