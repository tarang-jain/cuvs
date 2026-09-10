/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/cluster/kmeans.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <nvtx3/nvToolsExt.h>

#include <chrono>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

class default_nvtx_range {
 public:
  explicit default_nvtx_range(char const* name)
  {
    auto const message = nvtxDomainRegisterStringA(nullptr, name);
    nvtxEventAttributes_t attributes{};
    attributes.version            = NVTX_VERSION;
    attributes.size               = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attributes.messageType        = NVTX_MESSAGE_TYPE_REGISTERED;
    attributes.message.registered = message;
    nvtxDomainRangePushEx(nullptr, &attributes);
  }

  ~default_nvtx_range() { nvtxDomainRangePop(nullptr); }

  default_nvtx_range(default_nvtx_range const&)                    = delete;
  default_nvtx_range(default_nvtx_range&&)                         = delete;
  auto operator=(default_nvtx_range const&) -> default_nvtx_range& = delete;
  auto operator=(default_nvtx_range&&) -> default_nvtx_range&      = delete;
};

struct options {
  std::filesystem::path dataset;
  std::uint64_t max_rows{0};
  std::uint32_t partitions{0};
  std::uint32_t iterations{0};
  std::string metric;
};

[[noreturn]] void usage_error(std::string const& message)
{
  throw std::invalid_argument(
    message +
    "\nUsage: BALANCED_KMEANS_PROFILE --dataset PATH --max-rows ROWS "
    "--partitions COUNT --iterations COUNT --metric "
    "{l2,l2-sqrt,cosine,inner-product}");
}

template <typename T>
T parse_positive_integer(std::string_view option, std::string_view text)
{
  std::uint64_t value = 0;
  auto const* begin   = text.data();
  auto const* end     = text.data() + text.size();
  auto [ptr, error]   = std::from_chars(begin, end, value);
  if (error != std::errc{} || ptr != end || value == 0 ||
      value > static_cast<std::uint64_t>(std::numeric_limits<T>::max())) {
    usage_error(std::string(option) + " must be a positive integer in range");
  }
  return static_cast<T>(value);
}

options parse_options(int argc, char** argv)
{
  options result;
  for (int i = 1; i < argc; ++i) {
    std::string_view option{argv[i]};
    if (i + 1 >= argc) { usage_error(std::string(option) + " requires a value"); }
    std::string_view value{argv[++i]};
    if (option == "--dataset") {
      result.dataset = value;
    } else if (option == "--max-rows") {
      result.max_rows = parse_positive_integer<std::uint64_t>(option, value);
    } else if (option == "--partitions") {
      result.partitions = parse_positive_integer<std::uint32_t>(option, value);
    } else if (option == "--iterations") {
      result.iterations = parse_positive_integer<std::uint32_t>(option, value);
    } else if (option == "--metric") {
      result.metric = value;
    } else {
      usage_error("unknown option: " + std::string(option));
    }
  }
  if (result.dataset.empty()) { usage_error("--dataset is required"); }
  if (result.max_rows == 0) { usage_error("--max-rows is required"); }
  if (result.partitions == 0) { usage_error("--partitions is required"); }
  if (result.iterations == 0) { usage_error("--iterations is required"); }
  if (result.metric.empty()) { usage_error("--metric is required"); }
  return result;
}

auto parse_metric(std::string const& metric) -> cuvs::distance::DistanceType
{
  if (metric == "l2") { return cuvs::distance::DistanceType::L2Expanded; }
  if (metric == "l2-sqrt") { return cuvs::distance::DistanceType::L2SqrtExpanded; }
  if (metric == "cosine") { return cuvs::distance::DistanceType::CosineExpanded; }
  if (metric == "inner-product") { return cuvs::distance::DistanceType::InnerProduct; }
  usage_error("unsupported --metric value: " + metric);
}

struct fbin_data {
  std::uint64_t rows;
  std::uint32_t dimensions;
  std::vector<float> values;
};

fbin_data load_fbin(std::filesystem::path const& path, std::uint64_t max_rows)
{
  std::ifstream input(path, std::ios::binary);
  if (!input) { throw std::runtime_error("cannot open dataset: " + path.string()); }

  std::uint32_t header[2]{};
  input.read(reinterpret_cast<char*>(header), sizeof(header));
  if (!input) { throw std::runtime_error("cannot read fbin header: " + path.string()); }
  auto const rows       = header[0];
  auto const dimensions = header[1];
  if (rows == 0 || dimensions == 0) { throw std::runtime_error("fbin dimensions must be nonzero"); }

  constexpr auto header_bytes = std::uintmax_t{sizeof(header)};
  auto const value_count = static_cast<std::uintmax_t>(rows) * dimensions;
  if (value_count > (std::numeric_limits<std::uintmax_t>::max() - header_bytes) / sizeof(float)) {
    throw std::runtime_error("fbin size calculation overflow");
  }
  auto const expected_bytes = header_bytes + value_count * sizeof(float);
  auto const actual_bytes   = std::filesystem::file_size(path);
  if (actual_bytes != expected_bytes) {
    throw std::runtime_error("fbin size mismatch: expected " + std::to_string(expected_bytes) +
                             " bytes, found " + std::to_string(actual_bytes));
  }
  if (max_rows > rows) { throw std::invalid_argument("--max-rows exceeds the fbin row count"); }
  if (max_rows > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("row count exceeds the supported int64 range");
  }
  auto const selected_values = static_cast<std::uintmax_t>(max_rows) * dimensions;
  if (selected_values > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      selected_values > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max()) /
                          sizeof(float)) {
    throw std::invalid_argument("selected dataset exceeds the supported host range");
  }

  std::vector<float> values(static_cast<std::size_t>(selected_values));
  auto const read_bytes = static_cast<std::streamsize>(selected_values * sizeof(float));
  input.read(reinterpret_cast<char*>(values.data()), read_bytes);
  if (input.gcount() != read_bytes) {
    throw std::runtime_error("short read while loading selected fbin rows");
  }
  return {max_rows, dimensions, std::move(values)};
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    auto const args   = parse_options(argc, argv);
    auto const metric = parse_metric(args.metric);
    auto host_data    = load_fbin(args.dataset, args.max_rows);
    if (args.partitions > host_data.rows) {
      throw std::invalid_argument("--partitions cannot exceed --max-rows");
    }

    auto const rows       = static_cast<std::int64_t>(host_data.rows);
    auto const dimensions = static_cast<std::int64_t>(host_data.dimensions);
    raft::device_resources resources;
    auto dataset = raft::make_device_matrix<float, std::int64_t>(resources, rows, dimensions);
    auto centers = raft::make_device_matrix<float, std::int64_t>(
      resources, static_cast<std::int64_t>(args.partitions), dimensions);
    auto stream = raft::resource::get_cuda_stream(resources);

    raft::copy(dataset.data_handle(), host_data.values.data(), dataset.size(), stream);
    raft::resource::sync_stream(resources, stream);
    host_data.values.clear();
    host_data.values.shrink_to_fit();

    cuvs::cluster::kmeans::balanced_params params;
    params.metric  = metric;
    params.n_iters = args.iterations;

    std::cout << "dataset=" << args.dataset << '\n'
              << "rows=" << rows << '\n'
              << "dimensions=" << dimensions << '\n'
              << "partitions=" << args.partitions << '\n'
              << "iterations=" << args.iterations << '\n'
              << "metric=" << args.metric << '\n';

    auto const start = std::chrono::steady_clock::now();
    {
      default_nvtx_range profile_range("balanced_kmeans_profile");
      cuvs::cluster::kmeans::fit(
        resources, params, raft::make_const_mdspan(dataset.view()), centers.view());
      raft::resource::sync_stream(resources, stream);
    }
    auto const elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
    std::cout << "training_elapsed_seconds=" << elapsed.count() << '\n';
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
