/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdint>
#include <dlpack/dlpack.h>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>

#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/ivf_sq.h>
#include <cuvs/neighbors/ivf_sq.hpp>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

namespace cuvs::neighbors::ivf_sq {
void convert_c_index_params(cuvsIvfSqIndexParams params,
                            cuvs::neighbors::ivf_sq::index_params* out)
{
  out->metric                        = static_cast<cuvs::distance::DistanceType>((int)params.metric);
  out->metric_arg                    = params.metric_arg;
  out->add_data_on_build             = params.add_data_on_build;
  out->n_lists                       = params.n_lists;
  out->kmeans_n_iters                = params.kmeans_n_iters;
  out->max_train_points_per_cluster  = params.max_train_points_per_cluster;
  out->conservative_memory_allocation = params.conservative_memory_allocation;
}
void convert_c_search_params(cuvsIvfSqSearchParams params,
                             cuvs::neighbors::ivf_sq::search_params* out)
{
  out->n_probes = params.n_probes;
}
}  // namespace cuvs::neighbors::ivf_sq

namespace {

using index_type = cuvs::neighbors::ivf_sq::index<uint8_t>;

void _reset_index(cuvsIvfSqIndex_t index)
{
  RAFT_EXPECTS(index != nullptr, "index cannot be null");
  auto index_ptr = reinterpret_cast<index_type*>(index->addr);
  index->addr    = 0;
  index->dtype   = DLDataType{};
  delete index_ptr;
}

template <typename T>
void* _build(cuvsResources_t res, cuvsIvfSqIndexParams params, DLManagedTensor* dataset_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  auto build_params = cuvs::neighbors::ivf_sq::index_params();
  cuvs::neighbors::ivf_sq::convert_c_index_params(params, &build_params);

  auto dataset = dataset_tensor->dl_tensor;

  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<const T, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    return new index_type(cuvs::neighbors::ivf_sq::build(*res_ptr, build_params, mds));
  } else {
    using mdspan_type = raft::host_matrix_view<const T, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    return new index_type(cuvs::neighbors::ivf_sq::build(*res_ptr, build_params, mds));
  }
}

template <typename T>
void _search(cuvsResources_t res,
             cuvsIvfSqSearchParams params,
             cuvsIvfSqIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor,
             cuvsFilter* filter)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_sq::index<uint8_t>*>(index.addr);

  auto search_params = cuvs::neighbors::ivf_sq::search_params();
  cuvs::neighbors::ivf_sq::convert_c_search_params(params, &search_params);

  using queries_mdspan_type   = raft::device_matrix_view<const T, int64_t, raft::row_major>;
  using neighbors_mdspan_type = raft::device_matrix_view<int64_t, int64_t, raft::row_major>;
  using distances_mdspan_type = raft::device_matrix_view<float, int64_t, raft::row_major>;
  auto queries_mds            = cuvs::core::from_dlpack<queries_mdspan_type>(queries_tensor);
  auto neighbors_mds          = cuvs::core::from_dlpack<neighbors_mdspan_type>(neighbors_tensor);
  auto distances_mds          = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);

  if (filter == nullptr || filter->type == NO_FILTER) {
    cuvs::neighbors::ivf_sq::search(
      *res_ptr, search_params, *index_ptr, queries_mds, neighbors_mds, distances_mds);
  } else if (filter->type == BITSET) {
    using filter_mdspan_type    = raft::device_vector_view<std::uint32_t, int64_t, raft::row_major>;
    auto removed_indices_tensor = reinterpret_cast<DLManagedTensor*>(filter->addr);
    auto removed_indices = cuvs::core::from_dlpack<filter_mdspan_type>(removed_indices_tensor);
    cuvs::core::bitset_view<std::uint32_t, int64_t> removed_indices_bitset(removed_indices,
                                                                           index_ptr->size());
    auto bitset_filter_obj = cuvs::neighbors::filtering::bitset_filter(removed_indices_bitset);
    cuvs::neighbors::ivf_sq::search(*res_ptr,
                                    search_params,
                                    *index_ptr,
                                    queries_mds,
                                    neighbors_mds,
                                    distances_mds,
                                    bitset_filter_obj);
  } else {
    RAFT_FAIL("Unsupported filter type: BITMAP");
  }
}

void _serialize(cuvsResources_t res, const char* filename, cuvsIvfSqIndex index)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_sq::index<uint8_t>*>(index.addr);
  cuvs::neighbors::ivf_sq::serialize(*res_ptr, std::string(filename), *index_ptr);
}

void* _deserialize(cuvsResources_t res, const char* filename)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto index   = new cuvs::neighbors::ivf_sq::index<uint8_t>(*res_ptr);
  cuvs::neighbors::ivf_sq::deserialize(*res_ptr, std::string(filename), index);
  return index;
}

template <typename T>
void _extend(cuvsResources_t res,
             DLManagedTensor* new_vectors,
             DLManagedTensor* new_indices,
             cuvsIvfSqIndex index)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_sq::index<uint8_t>*>(index.addr);

  bool on_device = cuvs::core::is_dlpack_device_compatible(new_vectors->dl_tensor);
  if (new_indices != nullptr &&
      on_device != cuvs::core::is_dlpack_device_compatible(new_indices->dl_tensor)) {
    RAFT_FAIL("extend inputs must both either be on device memory or host memory");
  }

  if (on_device) {
    using vectors_mdspan_type = raft::device_matrix_view<const T, int64_t, raft::row_major>;
    using indices_mdspan_type = raft::device_vector_view<const int64_t, int64_t>;
    auto vectors_mds          = cuvs::core::from_dlpack<vectors_mdspan_type>(new_vectors);
    std::optional<indices_mdspan_type> indices_mds;
    if (new_indices != nullptr) {
      indices_mds.emplace(cuvs::core::from_dlpack<indices_mdspan_type>(new_indices));
    }
    cuvs::neighbors::ivf_sq::extend(*res_ptr, vectors_mds, indices_mds, index_ptr);
  } else {
    using vectors_mdspan_type = raft::host_matrix_view<const T, int64_t, raft::row_major>;
    using indices_mdspan_type = raft::host_vector_view<const int64_t, int64_t>;
    auto vectors_mds          = cuvs::core::from_dlpack<vectors_mdspan_type>(new_vectors);
    std::optional<indices_mdspan_type> indices_mds;
    if (new_indices != nullptr) {
      indices_mds.emplace(cuvs::core::from_dlpack<indices_mdspan_type>(new_indices));
    }
    cuvs::neighbors::ivf_sq::extend(*res_ptr, vectors_mds, indices_mds, index_ptr);
  }
}

void _get_centers(cuvsIvfSqIndex index, DLManagedTensor* centers)
{
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_sq::index<uint8_t>*>(index.addr);
  cuvs::core::to_dlpack(index_ptr->centers(), centers);
}
}  // namespace

extern "C" cuvsError_t cuvsIvfSqIndexCreate(cuvsIvfSqIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] { *index = new cuvsIvfSqIndex{}; });
}

extern "C" cuvsError_t cuvsIvfSqIndexDestroy(cuvsIvfSqIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    _reset_index(index_c_ptr);
    delete index_c_ptr;
  });
}

extern "C" cuvsError_t cuvsIvfSqBuild(cuvsResources_t res,
                                      cuvsIvfSqIndexParams_t params,
                                      DLManagedTensor* dataset_tensor,
                                      cuvsIvfSqIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = dataset_tensor->dl_tensor;

    if (dataset.dtype.code != kDLFloat ||
        (dataset.dtype.bits != 32 && dataset.dtype.bits != 16)) {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }

    _reset_index(index);

    index->dtype.code = dataset.dtype.code;
    index->dtype.bits = dataset.dtype.bits;

    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      index->addr = reinterpret_cast<uintptr_t>(_build<float>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      index->addr = reinterpret_cast<uintptr_t>(_build<half>(res, *params, dataset_tensor));
    }
  });
}

static cuvsError_t _cuvsIvfSqSearchImpl(cuvsResources_t res,
                                        cuvsIvfSqSearchParams_t params,
                                        cuvsIvfSqIndex_t index_c_ptr,
                                        DLManagedTensor* queries_tensor,
                                        DLManagedTensor* neighbors_tensor,
                                        DLManagedTensor* distances_tensor,
                                        cuvsFilter* filter)
{
  return cuvs::core::translate_exceptions([=] {
    auto queries   = queries_tensor->dl_tensor;
    auto neighbors = neighbors_tensor->dl_tensor;
    auto distances = distances_tensor->dl_tensor;

    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(queries),
                 "queries should have device compatible memory");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(neighbors),
                 "neighbors should have device compatible memory");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(distances),
                 "distances should have device compatible memory");

    RAFT_EXPECTS(neighbors.dtype.code == kDLInt && neighbors.dtype.bits == 64,
                 "neighbors should be of type int64_t");
    RAFT_EXPECTS(distances.dtype.code == kDLFloat && distances.dtype.bits == 32,
                 "distances should be of type float32");

    auto index = *index_c_ptr;
    if (queries.dtype.code == kDLFloat && queries.dtype.bits == 32) {
      _search<float>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else if (queries.dtype.code == kDLFloat && queries.dtype.bits == 16) {
      _search<half>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else {
      RAFT_FAIL("Unsupported queries DLtensor dtype: %d and bits: %d",
                queries.dtype.code,
                queries.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsIvfSqSearch(cuvsResources_t res,
                                       cuvsIvfSqSearchParams_t params,
                                       cuvsIvfSqIndex_t index_c_ptr,
                                       DLManagedTensor* queries_tensor,
                                       DLManagedTensor* neighbors_tensor,
                                       DLManagedTensor* distances_tensor,
                                       cuvsFilter filter)
{
  return _cuvsIvfSqSearchImpl(
    res, params, index_c_ptr, queries_tensor, neighbors_tensor, distances_tensor, &filter);
}

extern "C" cuvsError_t cuvsIvfSqIndexParamsCreate(cuvsIvfSqIndexParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params = new cuvsIvfSqIndexParams{.metric                         = L2Expanded,
                                       .metric_arg                     = 2.0f,
                                       .add_data_on_build              = true,
                                       .n_lists                        = 1024,
                                       .kmeans_n_iters                 = 20,
                                       .max_train_points_per_cluster    = 256,
                                       .conservative_memory_allocation = false};
  });
}

extern "C" cuvsError_t cuvsIvfSqIndexParamsDestroy(cuvsIvfSqIndexParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsIvfSqSearchParamsCreate(cuvsIvfSqSearchParams_t* params)
{
  return cuvs::core::translate_exceptions(
    [=] { *params = new cuvsIvfSqSearchParams{.n_probes = 20}; });
}

extern "C" cuvsError_t cuvsIvfSqSearchParamsDestroy(cuvsIvfSqSearchParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsIvfSqDeserialize(cuvsResources_t res,
                                            const char* filename,
                                            cuvsIvfSqIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    _reset_index(index);
    index->addr = reinterpret_cast<uintptr_t>(_deserialize(res, filename));
  });
}

extern "C" cuvsError_t cuvsIvfSqSerialize(cuvsResources_t res,
                                          const char* filename,
                                          cuvsIvfSqIndex_t index)
{
  return cuvs::core::translate_exceptions([=] { _serialize(res, filename, *index); });
}

extern "C" cuvsError_t cuvsIvfSqExtend(cuvsResources_t res,
                                       DLManagedTensor* new_vectors,
                                       DLManagedTensor* new_indices,
                                       cuvsIvfSqIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto vectors = new_vectors->dl_tensor;

    if (index->dtype.code == 0 && index->dtype.bits == 0) {
      index->dtype.code = vectors.dtype.code;
      index->dtype.bits = vectors.dtype.bits;
    }

    if (vectors.dtype.code == kDLFloat && vectors.dtype.bits == 32) {
      _extend<float>(res, new_vectors, new_indices, *index);
    } else if (vectors.dtype.code == kDLFloat && vectors.dtype.bits == 16) {
      _extend<half>(res, new_vectors, new_indices, *index);
    } else {
      RAFT_FAIL(
        "Unsupported vectors DLtensor dtype: %d and bits: %d", vectors.dtype.code, vectors.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsIvfSqIndexGetNLists(cuvsIvfSqIndex_t index, int64_t* n_lists)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(index != nullptr, "index cannot be null");
    RAFT_EXPECTS(index->addr != 0, "index must be built before getting n_lists");
    auto index_ptr =
      reinterpret_cast<cuvs::neighbors::ivf_sq::index<uint8_t>*>(index->addr);
    *n_lists = index_ptr->n_lists();
  });
}

extern "C" cuvsError_t cuvsIvfSqIndexGetDim(cuvsIvfSqIndex_t index, int64_t* dim)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(index != nullptr, "index cannot be null");
    RAFT_EXPECTS(index->addr != 0, "index must be built before getting dim");
    auto index_ptr =
      reinterpret_cast<cuvs::neighbors::ivf_sq::index<uint8_t>*>(index->addr);
    *dim = index_ptr->dim();
  });
}

extern "C" cuvsError_t cuvsIvfSqIndexGetSize(cuvsIvfSqIndex_t index, int64_t* size)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(index != nullptr, "index cannot be null");
    RAFT_EXPECTS(index->addr != 0, "index must be built before getting size");
    auto index_ptr =
      reinterpret_cast<cuvs::neighbors::ivf_sq::index<uint8_t>*>(index->addr);
    *size = index_ptr->size();
  });
}

extern "C" cuvsError_t cuvsIvfSqIndexGetCenters(cuvsIvfSqIndex_t index, DLManagedTensor* centers)
{
  return cuvs::core::translate_exceptions([=] { _get_centers(*index, centers); });
}

extern "C" cuvsError_t cuvsIvfSqIndexGetListSizes(cuvsIvfSqIndex_t index,
                                                  DLManagedTensor* list_sizes)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(index != nullptr && index->addr != 0, "IVF-SQ index must be built");
    auto* index_ptr = reinterpret_cast<index_type*>(index->addr);
    cuvs::core::to_dlpack(index_ptr->list_sizes(), list_sizes);
  });
}

extern "C" cuvsError_t cuvsIvfSqIndexGetListIndices(cuvsIvfSqIndex_t index,
                                                    uint32_t label,
                                                    DLManagedTensor* out_indices)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(index != nullptr && index->addr != 0, "IVF-SQ index must be built");
    auto* index_ptr = reinterpret_cast<index_type*>(index->addr);
    RAFT_EXPECTS(label < index_ptr->n_lists(), "list label is out of range");
    RAFT_EXPECTS(index_ptr->lists()[label] != nullptr, "list has not been allocated");
    cuvs::core::to_dlpack(index_ptr->lists()[label]->indices.view(), out_indices);
  });
}
extern "C" cuvsError_t cuvsIvfSqIndexUnpackContiguousListData(cuvsResources_t res,
                                                              cuvsIvfSqIndex_t index,
                                                              DLManagedTensor* out_codes,
                                                              uint32_t label,
                                                              uint32_t offset)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(index != nullptr && index->addr != 0, "IVF-SQ index must be built");
    auto* index_ptr = reinterpret_cast<index_type*>(index->addr);
    RAFT_EXPECTS(label < index_ptr->n_lists(), "list label is out of range");
    RAFT_EXPECTS(index_ptr->lists()[label] != nullptr, "list has not been allocated");
    using output_type = raft::device_matrix_view<uint8_t, uint32_t, raft::row_major>;
    auto output       = cuvs::core::from_dlpack<output_type>(out_codes);
    auto logical_dim  = index_ptr->dim();
    RAFT_EXPECTS(output.extent(1) == logical_dim, "output dimensionality does not match index");

    auto* res_ptr   = reinterpret_cast<raft::resources*>(res);
    auto list_data  = raft::make_const_mdspan(index_ptr->lists()[label]->data.view());
    auto padded_dim = list_data.extent(1);
    constexpr uint32_t vec_len =
      cuvs::neighbors::ivf_sq::list_spec<uint32_t, uint8_t, int64_t>::kVecLen;
    if (logical_dim == padded_dim) {
      cuvs::neighbors::ivf_flat::helpers::codepacker::unpack(
        *res_ptr, list_data, vec_len, offset, output);
      return;
    }

    auto padded =
      raft::make_device_matrix<uint8_t, uint32_t>(*res_ptr, output.extent(0), padded_dim);
    cuvs::neighbors::ivf_flat::helpers::codepacker::unpack(
      *res_ptr, list_data, vec_len, offset, padded.view());
    auto stream = raft::resource::get_cuda_stream(*res_ptr);
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(output.data_handle(),
                                    logical_dim * sizeof(uint8_t),
                                    padded.data_handle(),
                                    padded_dim * sizeof(uint8_t),
                                    logical_dim * sizeof(uint8_t),
                                    output.extent(0),
                                    cudaMemcpyDeviceToDevice,
                                    stream));
  });
}

extern "C" cuvsError_t cuvsIvfSqIndexGetVMin(cuvsIvfSqIndex_t index, DLManagedTensor* vmin)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(index != nullptr && index->addr != 0, "IVF-SQ index must be built");
    auto* index_ptr = reinterpret_cast<index_type*>(index->addr);
    cuvs::core::to_dlpack(index_ptr->sq_vmin(), vmin);
  });
}

extern "C" cuvsError_t cuvsIvfSqIndexGetDelta(cuvsIvfSqIndex_t index, DLManagedTensor* delta)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(index != nullptr && index->addr != 0, "IVF-SQ index must be built");
    auto* index_ptr = reinterpret_cast<index_type*>(index->addr);
    cuvs::core::to_dlpack(index_ptr->sq_delta(), delta);
  });
}

namespace {

void ivf_sq_recompute_internal_state(raft::resources const& res, index_type* index)
{
  auto n_lists = index->n_lists();
  auto stream  = raft::resource::get_cuda_stream(res);
  std::vector<uint32_t> sizes(n_lists);
  std::vector<uint8_t*> data_ptrs(n_lists);
  std::vector<int64_t*> inds_ptrs(n_lists);

  raft::update_host(sizes.data(), index->list_sizes().data_handle(), n_lists, stream);
  for (uint32_t label = 0; label < n_lists; ++label) {
    auto const& list = index->lists()[label];
    data_ptrs[label] = list ? list->data.data_handle() : nullptr;
    inds_ptrs[label] = list ? list->indices.data_handle() : nullptr;
  }
  raft::update_device(index->data_ptrs().data_handle(), data_ptrs.data(), n_lists, stream);
  raft::update_device(index->inds_ptrs().data_handle(), inds_ptrs.data(), n_lists, stream);
  raft::resource::sync_stream(res);

  std::sort(sizes.begin(), sizes.end(), std::greater<uint32_t>{});
  auto accumulated = index->accum_sorted_sizes();
  accumulated(0)   = 0;
  for (uint32_t label = 0; label < n_lists; ++label) {
    accumulated(label + 1) = accumulated(label) + sizes[label];
  }
}

void ivf_sq_copy_trained_state(raft::resources const& res,
                               index_type* destination,
                               index_type const* source)
{
  raft::copy(res, destination->centers(), source->centers());
  raft::copy(res, destination->sq_vmin(), source->sq_vmin());
  raft::copy(res, destination->sq_delta(), source->sq_delta());
  if (source->center_norms().has_value()) {
    destination->allocate_center_norms(res);
    raft::copy(res, destination->center_norms().value(), source->center_norms().value());
  }
}

void ivf_sq_extend_list(cuvsResources_t res,
                        cuvsIvfSqIndex_t index,
                        DLManagedTensor* new_codes,
                        DLManagedTensor* new_indices,
                        uint32_t label)
{
  auto* res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto* index_ptr = reinterpret_cast<index_type*>(index->addr);
  RAFT_EXPECTS(index_ptr != nullptr, "IVF-SQ index must be built");
  RAFT_EXPECTS(label < index_ptr->n_lists(), "list label is out of range");

  using codes_type   = raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major>;
  using indices_type = raft::device_vector_view<const int64_t, uint32_t>;
  auto codes         = cuvs::core::from_dlpack<codes_type>(new_codes);
  auto indices       = cuvs::core::from_dlpack<indices_type>(new_indices);
  RAFT_EXPECTS(codes.extent(0) == indices.extent(0), "codes and indices length mismatch");
  RAFT_EXPECTS(codes.extent(1) == index_ptr->dim(), "code dimensionality mismatch");
  if (codes.extent(0) == 0) { return; }

  auto stream = raft::resource::get_cuda_stream(*res_ptr);
  uint32_t old_size{};
  raft::update_host(&old_size, index_ptr->list_sizes().data_handle() + label, 1, stream);
  raft::resource::sync_stream(*res_ptr);
  RAFT_EXPECTS(codes.extent(0) <= std::numeric_limits<uint32_t>::max() - old_size,
               "list size exceeds uint32 range");
  auto new_size = old_size + codes.extent(0);

  auto spec = cuvs::neighbors::ivf_sq::list_spec<uint32_t, uint8_t, int64_t>{
    index_ptr->dim(), index_ptr->conservative_memory_allocation()};
  cuvs::neighbors::ivf::resize_list(*res_ptr, index_ptr->lists()[label], spec, new_size, old_size);
  auto list_data   = index_ptr->lists()[label]->data.view();
  auto logical_dim = codes.extent(1);
  auto padded_dim  = list_data.extent(1);
  constexpr uint32_t vec_len =
    cuvs::neighbors::ivf_sq::list_spec<uint32_t, uint8_t, int64_t>::kVecLen;
  if (logical_dim == padded_dim) {
    cuvs::neighbors::ivf_flat::helpers::codepacker::pack(
      *res_ptr, codes, vec_len, old_size, list_data);
  } else {
    auto padded =
      raft::make_device_matrix<uint8_t, uint32_t>(*res_ptr, codes.extent(0), padded_dim);
    RAFT_CUDA_TRY(
      cudaMemsetAsync(padded.data_handle(), 0, padded.size() * sizeof(uint8_t), stream));
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(padded.data_handle(),
                                    padded_dim * sizeof(uint8_t),
                                    codes.data_handle(),
                                    logical_dim * sizeof(uint8_t),
                                    logical_dim * sizeof(uint8_t),
                                    codes.extent(0),
                                    cudaMemcpyDeviceToDevice,
                                    stream));
    cuvs::neighbors::ivf_flat::helpers::codepacker::pack(
      *res_ptr, raft::make_const_mdspan(padded.view()), vec_len, old_size, list_data);
  }
  raft::copy(index_ptr->lists()[label]->indices.data_handle() + old_size,
             indices.data_handle(),
             indices.extent(0),
             stream);
  raft::copy(index_ptr->list_sizes().data_handle() + label, &new_size, 1, stream);
  ivf_sq_recompute_internal_state(*res_ptr, index_ptr);
}

}  // namespace

extern "C" cuvsError_t cuvsIvfSqIndexReset(cuvsResources_t res, cuvsIvfSqIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(index != nullptr && index->addr != 0, "IVF-SQ index must be built");
    auto* res_ptr    = reinterpret_cast<raft::resources*>(res);
    auto* index_ptr  = reinterpret_cast<index_type*>(index->addr);
    auto replacement = std::make_unique<index_type>(*res_ptr,
                                                    index_ptr->metric(),
                                                    index_ptr->n_lists(),
                                                    index_ptr->dim(),
                                                    index_ptr->conservative_memory_allocation());
    ivf_sq_copy_trained_state(*res_ptr, replacement.get(), index_ptr);
    delete index_ptr;
    index->addr = reinterpret_cast<uintptr_t>(replacement.release());
  });
}

extern "C" cuvsError_t cuvsIvfSqBuildFromCenters(cuvsResources_t res,
                                                 cuvsIvfSqIndexParams_t params,
                                                 DLDataType index_dtype,
                                                 DLManagedTensor* centers,
                                                 DLManagedTensor* center_norms,
                                                 DLManagedTensor* vmin,
                                                 DLManagedTensor* delta,
                                                 cuvsIvfSqIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(index != nullptr && index->addr == 0, "output index handle must be empty");
    RAFT_EXPECTS(index_dtype.code == kDLFloat && (index_dtype.bits == 32 || index_dtype.bits == 16),
                 "IVF-SQ index dtype must be float32 or float16");
    auto* res_ptr     = reinterpret_cast<raft::resources*>(res);
    using matrix_type = raft::device_matrix_view<const float, uint32_t, raft::row_major>;
    using vector_type = raft::device_vector_view<const float, uint32_t>;
    auto centers_view = cuvs::core::from_dlpack<matrix_type>(centers);
    auto vmin_view    = cuvs::core::from_dlpack<vector_type>(vmin);
    auto delta_view   = cuvs::core::from_dlpack<vector_type>(delta);
    RAFT_EXPECTS(centers_view.extent(0) == params->n_lists, "centers row count must equal n_lists");
    RAFT_EXPECTS(vmin_view.extent(0) == centers_view.extent(1), "vmin dimensionality mismatch");
    RAFT_EXPECTS(delta_view.extent(0) == centers_view.extent(1), "delta dimensionality mismatch");

    auto build_params = cuvs::neighbors::ivf_sq::index_params{};
    cuvs::neighbors::ivf_sq::convert_c_index_params(*params, &build_params);
    auto replacement = std::make_unique<index_type>(*res_ptr, build_params, centers_view.extent(1));
    raft::copy(*res_ptr, replacement->centers(), centers_view);
    raft::copy(*res_ptr, replacement->sq_vmin(), vmin_view);
    raft::copy(*res_ptr, replacement->sq_delta(), delta_view);
    if (center_norms != nullptr) {
      auto norms_view = cuvs::core::from_dlpack<vector_type>(center_norms);
      RAFT_EXPECTS(norms_view.extent(0) == params->n_lists,
                   "center_norms length must equal n_lists");
      replacement->allocate_center_norms(*res_ptr);
      RAFT_EXPECTS(replacement->center_norms().has_value(),
                   "center norms are not supported for the configured metric");
      raft::copy(*res_ptr, replacement->center_norms().value(), norms_view);
    }
    index->dtype = index_dtype;
    index->addr  = reinterpret_cast<uintptr_t>(replacement.release());
  });
}

extern "C" cuvsError_t cuvsIvfSqIndexExtendList(cuvsResources_t res,
                                                cuvsIvfSqIndex_t index,
                                                DLManagedTensor* new_codes,
                                                DLManagedTensor* new_indices,
                                                uint32_t label)
{
  return cuvs::core::translate_exceptions(
    [=] { ivf_sq_extend_list(res, index, new_codes, new_indices, label); });
}
