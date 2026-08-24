/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * IVF-PQ search on SIFT-1M with a single inverted list (n_lists=1, n_probes=1).
 *
 * PQ codebooks are trained with the preprocessing PQ API (classic k-means by
 * default: KMeansPlusPlus init, oversampling_factor=2, 12 iterations — the same
 * settings as pq_kmeanspp_recall_example), then copied into an IVF-PQ index
 * with a zero centroid and identity rotation so ADC is over the full base set.
 *
 * Step-8 of scalable k-means++ is selected with CUVS_KMEANS_PP_STEP8:
 *   random    (default)  initRandom on the oversampled coreset
 *   kmeans++             kmeansPlusPlus on the oversampled coreset
 */

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/preprocessing/quantize/pq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct args {
  std::string base        = "/datasets/sift-128-euclidean/base.fbin";
  std::string queries     = "/datasets/sift-128-euclidean/query.fbin";
  std::string groundtruth = "/datasets/sift-128-euclidean/groundtruth.neighbors.ibin";
  uint32_t pq_bits        = 8;
  uint32_t pq_dim         = 64;
  int max_iter            = 12;
  double oversampling     = 2.0;
  uint32_t train_per_code = 1024;
  int64_t k_neighbors     = 10;
  uint32_t seed           = 42;
};

void usage(const char* prog)
{
  std::cerr << "Usage: " << prog << " [options]\n"
            << "  --base PATH              base.fbin [" << args{}.base << "]\n"
            << "  --queries PATH           query.fbin [" << args{}.queries << "]\n"
            << "  --groundtruth PATH       neighbors.ibin [" << args{}.groundtruth << "]\n"
            << "  --pq-bits N              [8]\n"
            << "  --pq-dim N               [64]\n"
            << "  --max-iter N             classic k-means iters [12]\n"
            << "  --oversampling F         KMeans|| oversampling_factor [2.0]\n"
            << "  --train-per-code N       max_train_points_per_pq_code [1024]\n"
            << "  --k N                    recall@k [10]\n"
            << "  --seed N                 [42]\n"
            << "Env: CUVS_KMEANS_PP_STEP8=random|kmeans++\n";
}

args parse_args(int argc, char** argv)
{
  args out;
  for (int i = 1; i < argc; ++i) {
    auto need = [&](const char* flag) -> const char* {
      if (i + 1 >= argc) { throw std::invalid_argument(std::string("missing value for ") + flag); }
      return argv[++i];
    };
    std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else if (a == "--base") {
      out.base = need(a.c_str());
    } else if (a == "--queries") {
      out.queries = need(a.c_str());
    } else if (a == "--groundtruth") {
      out.groundtruth = need(a.c_str());
    } else if (a == "--pq-bits") {
      out.pq_bits = static_cast<uint32_t>(std::stoul(need(a.c_str())));
    } else if (a == "--pq-dim") {
      out.pq_dim = static_cast<uint32_t>(std::stoul(need(a.c_str())));
    } else if (a == "--max-iter") {
      out.max_iter = std::stoi(need(a.c_str()));
    } else if (a == "--oversampling") {
      out.oversampling = std::stod(need(a.c_str()));
    } else if (a == "--train-per-code") {
      out.train_per_code = static_cast<uint32_t>(std::stoul(need(a.c_str())));
    } else if (a == "--k") {
      out.k_neighbors = std::stoll(need(a.c_str()));
    } else if (a == "--seed") {
      out.seed = static_cast<uint32_t>(std::stoul(need(a.c_str())));
    } else {
      throw std::invalid_argument("unknown option: " + a);
    }
  }
  return out;
}

template <typename T>
std::tuple<int64_t, int64_t, std::vector<T>> load_bin(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { throw std::runtime_error("failed to open " + path); }
  int32_t n = 0, d = 0;
  in.read(reinterpret_cast<char*>(&n), sizeof(n));
  in.read(reinterpret_cast<char*>(&d), sizeof(d));
  if (!in || n <= 0 || d <= 0) { throw std::runtime_error("bad header in " + path); }
  std::vector<T> data(static_cast<size_t>(n) * static_cast<size_t>(d));
  in.read(reinterpret_cast<char*>(data.data()),
          static_cast<std::streamsize>(data.size() * sizeof(T)));
  if (!in) { throw std::runtime_error("short read of " + path); }
  return {n, d, std::move(data)};
}

double calc_recall(const std::vector<int64_t>& expected,
                   const std::vector<int64_t>& actual,
                   int64_t n_queries,
                   int64_t k)
{
  size_t match       = 0;
  const size_t total = static_cast<size_t>(n_queries) * static_cast<size_t>(k);
  for (int64_t i = 0; i < n_queries; ++i) {
    for (int64_t a = 0; a < k; ++a) {
      const int64_t act = actual[static_cast<size_t>(i * k + a)];
      for (int64_t e = 0; e < k; ++e) {
        if (act == expected[static_cast<size_t>(i * k + e)]) {
          ++match;
          break;
        }
      }
    }
  }
  return static_cast<double>(match) / static_cast<double>(total);
}

const char* step8_env()
{
  const char* v = std::getenv("CUVS_KMEANS_PP_STEP8");
  return (v == nullptr || std::strlen(v) == 0) ? "random" : v;
}

// PQ codebook is [pq_dim * book_size, pq_len]; IVF-PQ PER_SUBSPACE is [pq_dim, pq_len, book_size].
std::vector<float> transpose_pq_codebook(const std::vector<float>& src,
                                         uint32_t pq_dim,
                                         uint32_t pq_len,
                                         uint32_t book_size)
{
  std::vector<float> dst(src.size());
  for (uint32_t s = 0; s < pq_dim; ++s) {
    for (uint32_t c = 0; c < book_size; ++c) {
      for (uint32_t d = 0; d < pq_len; ++d) {
        dst[(static_cast<size_t>(s) * pq_len + d) * book_size + c] =
          src[(static_cast<size_t>(s) * book_size + c) * pq_len + d];
      }
    }
  }
  return dst;
}

}  // namespace

int main(int argc, char** argv)
{
  args opt;
  try {
    opt = parse_args(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }

  const char* step8_mode = step8_env();
  std::cout << "SIFT IVF-PQ single-list / recall (PQ API + manual index)\n"
            << "  base=" << opt.base << "\n"
            << "  queries=" << opt.queries << "\n"
            << "  groundtruth=" << opt.groundtruth << "\n"
            << "  n_lists=1 n_probes=1 pq_bits=" << opt.pq_bits << " pq_dim=" << opt.pq_dim
            << " max_iter=" << opt.max_iter << " oversampling=" << opt.oversampling << "\n"
            << "  train_per_code=" << opt.train_per_code << " k=" << opt.k_neighbors
            << " seed=" << opt.seed << "\n"
            << "  CUVS_KMEANS_PP_STEP8=" << step8_mode << "\n";

  auto [n_base, dim, base_host]      = load_bin<float>(opt.base);
  auto [n_queries, qdim, query_host] = load_bin<float>(opt.queries);
  auto [n_gt, gt_k, gt_host32]       = load_bin<int32_t>(opt.groundtruth);
  if (qdim != dim) { throw std::runtime_error("query dim != base dim"); }
  if (n_gt != n_queries) { throw std::runtime_error("groundtruth rows != n_queries"); }
  if (gt_k < opt.k_neighbors) { throw std::runtime_error("groundtruth k smaller than --k"); }
  if (dim % static_cast<int64_t>(opt.pq_dim) != 0) {
    throw std::runtime_error("dim must be divisible by pq_dim for identity rotation");
  }

  std::cout << "  loaded base " << n_base << " x " << dim << ", queries " << n_queries << " x "
            << qdim << ", gt k=" << gt_k << "\n";

  raft::device_resources res;
  rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(),
                                        4ull * 1024ull * 1024ull * 1024ull);
  rmm::mr::set_current_device_resource(pool_mr);
  raft::resource::set_cuda_stream_pool(res, std::make_shared<rmm::cuda_stream_pool>(1));
  auto stream = raft::resource::get_cuda_stream(res);

  auto dataset = raft::make_device_matrix<float, int64_t>(res, n_base, dim);
  auto queries = raft::make_device_matrix<float, int64_t>(res, n_queries, dim);
  raft::copy(dataset.data_handle(), base_host.data(), base_host.size(), stream);
  raft::copy(queries.data_handle(), query_host.data(), query_host.size(), stream);
  raft::resource::sync_stream(res);

  cuvs::cluster::kmeans::params km;
  km.metric              = cuvs::distance::DistanceType::L2Expanded;
  km.n_clusters          = 1 << opt.pq_bits;
  km.init                = cuvs::cluster::kmeans::params::InitMethod::KMeansPlusPlus;
  km.max_iter            = opt.max_iter;
  km.n_init              = 1;
  km.oversampling_factor = opt.oversampling;
  km.rng_state.seed      = opt.seed;

  using kmeans_params_variant = cuvs::preprocessing::quantize::pq::kmeans_params_variant;
  auto pq_params =
    cuvs::preprocessing::quantize::pq::params(opt.pq_bits,
                                              opt.pq_dim,
                                              /*use_subspaces=*/true,
                                              /*use_vq=*/false,
                                              /*vq_n_centers=*/0,
                                              kmeans_params_variant{km},
                                              opt.train_per_code,
                                              /*max_train_points_per_vq_cluster=*/1024);

  raft::resource::sync_stream(res);
  auto t0        = std::chrono::steady_clock::now();
  auto quantizer = cuvs::preprocessing::quantize::pq::build(
    res, pq_params, raft::make_const_mdspan(dataset.view()));
  raft::resource::sync_stream(res);
  auto train_ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  const uint32_t pq_len    = static_cast<uint32_t>(dim) / opt.pq_dim;
  const uint32_t book_size = 1u << opt.pq_bits;
  auto book                = quantizer.vpq_codebooks.pq_code_book.view();
  if (book.extent(0) != opt.pq_dim * book_size || book.extent(1) != pq_len) {
    throw std::runtime_error("unexpected PQ codebook shape");
  }
  std::vector<float> book_host(static_cast<size_t>(book.extent(0)) * book.extent(1));
  raft::copy(book_host.data(), book.data_handle(), book_host.size(), stream);
  raft::resource::sync_stream(res);
  auto pq_ivf_host = transpose_pq_codebook(book_host, opt.pq_dim, pq_len, book_size);

  auto pq_centers = raft::make_mdspan<const float, uint32_t, raft::row_major, true, false>(
    pq_ivf_host.data(), raft::make_extents<uint32_t>(opt.pq_dim, pq_len, book_size));
  std::vector<float> zero_center(static_cast<size_t>(dim), 0.f);
  auto centers = raft::make_host_matrix_view<const float, uint32_t>(zero_center.data(), 1, dim);

  cuvs::neighbors::ivf_pq::index_params index_params;
  index_params.metric                         = cuvs::distance::DistanceType::L2Expanded;
  index_params.n_lists                        = 1;
  index_params.pq_bits                        = opt.pq_bits;
  index_params.pq_dim                         = opt.pq_dim;
  index_params.codebook_kind                  = cuvs::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;
  index_params.force_random_rotation          = false;
  index_params.add_data_on_build              = false;
  index_params.conservative_memory_allocation = true;

  t0         = std::chrono::steady_clock::now();
  auto index = cuvs::neighbors::ivf_pq::build(
    res, index_params, static_cast<uint32_t>(dim), pq_centers, centers, std::nullopt, std::nullopt);
  std::optional<raft::device_vector_view<const int64_t, int64_t>> no_idx = std::nullopt;
  cuvs::neighbors::ivf_pq::extend(res, raft::make_const_mdspan(dataset.view()), no_idx, &index);
  raft::resource::sync_stream(res);
  auto assemble_ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  std::cout << "  index n_lists=" << index.n_lists() << " size=" << index.size()
            << " pq_dim=" << index.pq_dim() << "\n";

  cuvs::neighbors::ivf_pq::search_params search_params;
  search_params.n_probes                = 1;
  search_params.lut_dtype               = CUDA_R_32F;
  search_params.internal_distance_dtype = CUDA_R_32F;
  search_params.coarse_search_dtype     = CUDA_R_32F;

  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, n_queries, opt.k_neighbors);
  auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, opt.k_neighbors);
  t0             = std::chrono::steady_clock::now();
  cuvs::neighbors::ivf_pq::search(res,
                                  search_params,
                                  index,
                                  raft::make_const_mdspan(queries.view()),
                                  neighbors.view(),
                                  distances.view());
  raft::resource::sync_stream(res);
  auto search_ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  std::vector<int64_t> actual(static_cast<size_t>(n_queries * opt.k_neighbors));
  raft::copy(actual.data(), neighbors.data_handle(), actual.size(), stream);
  raft::resource::sync_stream(res);

  std::vector<int64_t> expected(static_cast<size_t>(n_queries * opt.k_neighbors));
  for (int64_t i = 0; i < n_queries; ++i) {
    for (int64_t j = 0; j < opt.k_neighbors; ++j) {
      expected[static_cast<size_t>(i * opt.k_neighbors + j)] =
        static_cast<int64_t>(gt_host32[static_cast<size_t>(i * gt_k + j)]);
    }
  }

  const double recall = calc_recall(expected, actual, n_queries, opt.k_neighbors);

  std::cout << "RESULT" << " n_lists=1 n_probes=1 pq_dim=" << opt.pq_dim << " train_ms=" << train_ms
            << " assemble_ms=" << assemble_ms << " search_ms=" << search_ms << " recall@"
            << opt.k_neighbors << "=" << recall << "\n";
  return 0;
}
