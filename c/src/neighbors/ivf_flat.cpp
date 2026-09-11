
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <dlpack/dlpack.h>
#include <raft/core/copy.hpp>

#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/numpy_serializer.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/ivf_flat.h>
#include <cuvs/neighbors/ivf_flat.hpp>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

#include <fstream>
#include <limits>

namespace cuvs::neighbors::ivf_flat {
void convert_c_index_params(cuvsIvfFlatIndexParams params,
                            cuvs::neighbors::ivf_flat::index_params* out)
{
  out->metric                   = static_cast<cuvs::distance::DistanceType>((int)params.metric),
  out->metric_arg               = params.metric_arg;
  out->add_data_on_build        = params.add_data_on_build;
  out->n_lists                  = params.n_lists;
  out->kmeans_n_iters           = params.kmeans_n_iters;
  out->kmeans_trainset_fraction = params.kmeans_trainset_fraction;
  out->adaptive_centers         = params.adaptive_centers;
  out->conservative_memory_allocation = params.conservative_memory_allocation;
}
void convert_c_search_params(cuvsIvfFlatSearchParams params,
                             cuvs::neighbors::ivf_flat::search_params* out)
{
  out->n_probes = params.n_probes;
}
}  // namespace cuvs::neighbors::ivf_flat

namespace {

template <typename T, typename IdxT>
void* _build(cuvsResources_t res, cuvsIvfFlatIndexParams params, DLManagedTensor* dataset_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  auto build_params = cuvs::neighbors::ivf_flat::index_params();
  cuvs::neighbors::ivf_flat::convert_c_index_params(params, &build_params);

  auto dataset = dataset_tensor->dl_tensor;
  auto dim     = dataset.shape[1];

  auto index = new cuvs::neighbors::ivf_flat::index<T, IdxT>(*res_ptr, build_params, dim);

  using mdspan_type = raft::device_matrix_view<T const, IdxT, raft::row_major>;
  auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);

  cuvs::neighbors::ivf_flat::build(*res_ptr, build_params, mds, *index);

  return index;
}

template <typename T, typename IdxT>
void _search(cuvsResources_t res,
             cuvsIvfFlatSearchParams params,
             cuvsIvfFlatIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor,
             cuvsFilter filter)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_flat::index<T, IdxT>*>(index.addr);

  auto search_params = cuvs::neighbors::ivf_flat::search_params();
  convert_c_search_params(params, &search_params);

  using queries_mdspan_type   = raft::device_matrix_view<T const, IdxT, raft::row_major>;
  using neighbors_mdspan_type = raft::device_matrix_view<IdxT, IdxT, raft::row_major>;
  using distances_mdspan_type = raft::device_matrix_view<float, IdxT, raft::row_major>;
  auto queries_mds            = cuvs::core::from_dlpack<queries_mdspan_type>(queries_tensor);
  auto neighbors_mds          = cuvs::core::from_dlpack<neighbors_mdspan_type>(neighbors_tensor);
  auto distances_mds          = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);

  if (filter.type == NO_FILTER) {
    cuvs::neighbors::ivf_flat::search(
      *res_ptr, search_params, *index_ptr, queries_mds, neighbors_mds, distances_mds);
  } else if (filter.type == BITSET) {
    using filter_mdspan_type    = raft::device_vector_view<std::uint32_t, int64_t, raft::row_major>;
    auto removed_indices_tensor = reinterpret_cast<DLManagedTensor*>(filter.addr);
    auto removed_indices = cuvs::core::from_dlpack<filter_mdspan_type>(removed_indices_tensor);
    cuvs::core::bitset_view<std::uint32_t, int64_t> removed_indices_bitset(removed_indices,
                                                                           index_ptr->size());
    auto bitset_filter_obj = cuvs::neighbors::filtering::bitset_filter(removed_indices_bitset);
    cuvs::neighbors::ivf_flat::search(*res_ptr,
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

template <typename T, typename IdxT>
void _serialize(cuvsResources_t res, const char* filename, cuvsIvfFlatIndex index)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_flat::index<T, IdxT>*>(index.addr);
  cuvs::neighbors::ivf_flat::serialize(*res_ptr, std::string(filename), *index_ptr);
}

template <typename T, typename IdxT>
void* _deserialize(cuvsResources_t res, const char* filename)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto index   = new cuvs::neighbors::ivf_flat::index<T, IdxT>(*res_ptr);
  cuvs::neighbors::ivf_flat::deserialize(*res_ptr, std::string(filename), index);
  return index;
}

template <typename T, typename IdxT>
void _extend(cuvsResources_t res,
             DLManagedTensor* new_vectors,
             DLManagedTensor* new_indices,
             cuvsIvfFlatIndex index)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_flat::index<T, IdxT>*>(index.addr);
  using vectors_mdspan_type = raft::device_matrix_view<T const, IdxT, raft::row_major>;
  using indices_mdspan_type = raft::device_vector_view<IdxT, IdxT>;

  auto vectors_mds = cuvs::core::from_dlpack<vectors_mdspan_type>(new_vectors);
  auto indices_mds = cuvs::core::from_dlpack<indices_mdspan_type>(new_indices);

  cuvs::neighbors::ivf_flat::extend(*res_ptr, vectors_mds, indices_mds, index_ptr);
}

template <typename T, typename IdxT>
void get_centers(cuvsIvfFlatIndex index, DLManagedTensor* centers)
{
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_flat::index<T, IdxT>*>(index.addr);
  cuvs::core::to_dlpack(index_ptr->centers(), centers);
}
}  // namespace

extern "C" cuvsError_t cuvsIvfFlatIndexCreate(cuvsIvfFlatIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] { *index = new cuvsIvfFlatIndex{}; });
}

extern "C" cuvsError_t cuvsIvfFlatIndexDestroy(cuvsIvfFlatIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    auto index = *index_c_ptr;

    if (index.dtype.code == kDLFloat && index.dtype.bits == 32) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<float, int64_t>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLFloat && index.dtype.bits == 16) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<half, int64_t>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLInt) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<int8_t, int64_t>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLUInt) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>*>(index.addr);
      delete index_ptr;
    }
    delete index_c_ptr;
  });
}

extern "C" cuvsError_t cuvsIvfFlatBuild(cuvsResources_t res,
                                        cuvsIvfFlatIndexParams_t params,
                                        DLManagedTensor* dataset_tensor,
                                        cuvsIvfFlatIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = dataset_tensor->dl_tensor;

    index->dtype = dataset.dtype;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      index->addr =
        reinterpret_cast<uintptr_t>(_build<float, int64_t>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      index->addr =
        reinterpret_cast<uintptr_t>(_build<half, int64_t>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      index->addr =
        reinterpret_cast<uintptr_t>(_build<int8_t, int64_t>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      index->addr =
        reinterpret_cast<uintptr_t>(_build<uint8_t, int64_t>(res, *params, dataset_tensor));
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsIvfFlatSearch(cuvsResources_t res,
                                         cuvsIvfFlatSearchParams_t params,
                                         cuvsIvfFlatIndex_t index_c_ptr,
                                         DLManagedTensor* queries_tensor,
                                         DLManagedTensor* neighbors_tensor,
                                         DLManagedTensor* distances_tensor,
                                         cuvsFilter filter)

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
    RAFT_EXPECTS(queries.dtype.code == index.dtype.code, "type mismatch between index and queries");

    if (queries.dtype.code == kDLFloat && queries.dtype.bits == 32) {
      _search<float, int64_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else if (queries.dtype.code == kDLFloat && queries.dtype.bits == 16) {
      _search<half, int64_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else if (queries.dtype.code == kDLInt && queries.dtype.bits == 8) {
      _search<int8_t, int64_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else if (queries.dtype.code == kDLUInt && queries.dtype.bits == 8) {
      _search<uint8_t, int64_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else {
      RAFT_FAIL("Unsupported queries DLtensor dtype: %d and bits: %d",
                queries.dtype.code,
                queries.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsIvfFlatIndexParamsCreate(cuvsIvfFlatIndexParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params = new cuvsIvfFlatIndexParams{.metric                         = L2Expanded,
                                         .metric_arg                     = 2.0f,
                                         .add_data_on_build              = true,
                                         .n_lists                        = 1024,
                                         .kmeans_n_iters                 = 20,
                                         .kmeans_trainset_fraction       = 0.5,
                                         .adaptive_centers               = false,
                                         .conservative_memory_allocation = false};
  });
}

extern "C" cuvsError_t cuvsIvfFlatIndexParamsDestroy(cuvsIvfFlatIndexParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsIvfFlatSearchParamsCreate(cuvsIvfFlatSearchParams_t* params)
{
  return cuvs::core::translate_exceptions(
    [=] { *params = new cuvsIvfFlatSearchParams{.n_probes = 20}; });
}

extern "C" cuvsError_t cuvsIvfFlatSearchParamsDestroy(cuvsIvfFlatSearchParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsIvfFlatDeserialize(cuvsResources_t res,
                                              const char* filename,
                                              cuvsIvfFlatIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    // read the numpy dtype from the beginning of the file
    std::ifstream is(filename, std::ios::in | std::ios::binary);
    if (!is) { RAFT_FAIL("Cannot open file %s", filename); }
    char dtype_string[4]{};
    if (!is.read(dtype_string, sizeof(dtype_string))) {
      RAFT_FAIL("Invalid or truncated index header in file %s", filename);
    }
    auto dtype =
      raft::numpy_serializer::parse_descr(std::string(dtype_string, sizeof(dtype_string)));
    is.close();

    index->dtype.bits = dtype.itemsize * 8;
    if (dtype.kind == 'f' && dtype.itemsize == 4) {
      index->addr       = reinterpret_cast<uintptr_t>(_deserialize<float, int64_t>(res, filename));
      index->dtype.code = kDLFloat;
    } else if (dtype.kind == 'e' && dtype.itemsize == 2) {
      index->addr       = reinterpret_cast<uintptr_t>(_deserialize<half, int64_t>(res, filename));
      index->dtype.code = kDLFloat;
      index->dtype.bits = 16;
    } else if (dtype.kind == 'i' && dtype.itemsize == 1) {
      index->addr       = reinterpret_cast<uintptr_t>(_deserialize<int8_t, int64_t>(res, filename));
      index->dtype.code = kDLInt;
    } else if (dtype.kind == 'u' && dtype.itemsize == 1) {
      index->addr = reinterpret_cast<uintptr_t>(_deserialize<uint8_t, int64_t>(res, filename));
      index->dtype.code = kDLUInt;
    } else {
      RAFT_FAIL(
        "Unsupported dtype in file %s itemsize %i kind %i", filename, dtype.itemsize, dtype.kind);
    }
  });
}

extern "C" cuvsError_t cuvsIvfFlatSerialize(cuvsResources_t res,
                                            const char* filename,
                                            cuvsIvfFlatIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      _serialize<float, int64_t>(res, filename, *index);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      _serialize<half, int64_t>(res, filename, *index);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      _serialize<int8_t, int64_t>(res, filename, *index);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      _serialize<uint8_t, int64_t>(res, filename, *index);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsIvfFlatExtend(cuvsResources_t res,
                                         DLManagedTensor* new_vectors,
                                         DLManagedTensor* new_indices,
                                         cuvsIvfFlatIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      _extend<float, int64_t>(res, new_vectors, new_indices, *index);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      _extend<half, int64_t>(res, new_vectors, new_indices, *index);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      _extend<int8_t, int64_t>(res, new_vectors, new_indices, *index);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      _extend<uint8_t, int64_t>(res, new_vectors, new_indices, *index);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsIvfFlatIndexGetNLists(cuvsIvfFlatIndex_t index, int64_t* n_lists)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<float, int64_t>*>(index->addr);
      *n_lists = index_ptr->n_lists();
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<half, int64_t>*>(index->addr);
      *n_lists = index_ptr->n_lists();
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<int8_t, int64_t>*>(index->addr);
      *n_lists = index_ptr->n_lists();
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>*>(index->addr);
      *n_lists = index_ptr->n_lists();
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsIvfFlatIndexGetDim(cuvsIvfFlatIndex_t index, int64_t* dim)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<float, int64_t>*>(index->addr);
      *dim = index_ptr->dim();
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<half, int64_t>*>(index->addr);
      *dim = index_ptr->dim();
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<int8_t, int64_t>*>(index->addr);
      *dim = index_ptr->dim();
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>*>(index->addr);
      *dim = index_ptr->dim();
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsIvfFlatIndexGetCenters(cuvsIvfFlatIndex_t index,
                                                  DLManagedTensor* centers)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      get_centers<float, int64_t>(*index, centers);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      get_centers<half, int64_t>(*index, centers);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      get_centers<int8_t, int64_t>(*index, centers);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      get_centers<uint8_t, int64_t>(*index, centers);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

namespace {

template <typename T>
auto ivf_flat_index(cuvsIvfFlatIndex_t index)
{
  RAFT_EXPECTS(index != nullptr && index->addr != 0, "IVF-Flat index must be built");
  return reinterpret_cast<cuvs::neighbors::ivf_flat::index<T, int64_t>*>(index->addr);
}

template <typename T>
void ivf_flat_get_list_indices(cuvsIvfFlatIndex_t index,
                               uint32_t label,
                               DLManagedTensor* out_indices)
{
  auto* index_ptr = ivf_flat_index<T>(index);
  RAFT_EXPECTS(label < index_ptr->n_lists(), "list label is out of range");
  RAFT_EXPECTS(index_ptr->lists()[label] != nullptr, "list has not been allocated");
  cuvs::core::to_dlpack(index_ptr->lists()[label]->indices.view(), out_indices);
}

template <typename T>
void ivf_flat_unpack(cuvsResources_t res,
                     cuvsIvfFlatIndex_t index,
                     DLManagedTensor* out_vectors,
                     uint32_t label,
                     uint32_t offset)
{
  auto* index_ptr = ivf_flat_index<T>(index);
  RAFT_EXPECTS(label < index_ptr->n_lists(), "list label is out of range");
  RAFT_EXPECTS(index_ptr->lists()[label] != nullptr, "list has not been allocated");
  using output_type = raft::device_matrix_view<T, uint32_t, raft::row_major>;
  auto output       = cuvs::core::from_dlpack<output_type>(out_vectors);
  RAFT_EXPECTS(output.extent(1) == index_ptr->dim(), "output dimensionality does not match index");
  cuvs::neighbors::ivf_flat::helpers::codepacker::unpack(
    *reinterpret_cast<raft::resources*>(res),
    raft::make_const_mdspan(index_ptr->lists()[label]->data.view()),
    index_ptr->veclen(),
    offset,
    output);
}

template <typename Fn>
void dispatch_ivf_flat(cuvsIvfFlatIndex_t index, Fn&& fn)
{
  if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
    fn(ivf_flat_index<float>(index));
  } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
    fn(ivf_flat_index<half>(index));
  } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
    fn(ivf_flat_index<int8_t>(index));
  } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
    fn(ivf_flat_index<uint8_t>(index));
  } else {
    RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
  }
}

}  // namespace

extern "C" cuvsError_t cuvsIvfFlatIndexGetSize(cuvsIvfFlatIndex_t index, int64_t* size)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(size != nullptr, "size output cannot be null");
    dispatch_ivf_flat(index, [=](auto* index_ptr) { *size = index_ptr->size(); });
  });
}

extern "C" cuvsError_t cuvsIvfFlatIndexGetListSizes(cuvsIvfFlatIndex_t index,
                                                    DLManagedTensor* list_sizes)
{
  return cuvs::core::translate_exceptions([=] {
    dispatch_ivf_flat(
      index, [=](auto* index_ptr) { cuvs::core::to_dlpack(index_ptr->list_sizes(), list_sizes); });
  });
}

extern "C" cuvsError_t cuvsIvfFlatIndexGetListIndices(cuvsIvfFlatIndex_t index,
                                                      uint32_t label,
                                                      DLManagedTensor* out_indices)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      ivf_flat_get_list_indices<float>(index, label, out_indices);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      ivf_flat_get_list_indices<half>(index, label, out_indices);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      ivf_flat_get_list_indices<int8_t>(index, label, out_indices);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      ivf_flat_get_list_indices<uint8_t>(index, label, out_indices);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsIvfFlatIndexUnpackContiguousListData(cuvsResources_t res,
                                                                cuvsIvfFlatIndex_t index,
                                                                DLManagedTensor* out_vectors,
                                                                uint32_t label,
                                                                uint32_t offset)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      ivf_flat_unpack<float>(res, index, out_vectors, label, offset);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      ivf_flat_unpack<half>(res, index, out_vectors, label, offset);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      ivf_flat_unpack<int8_t>(res, index, out_vectors, label, offset);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      ivf_flat_unpack<uint8_t>(res, index, out_vectors, label, offset);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

namespace {

template <typename T>
void ivf_flat_build_from_centers(cuvsResources_t res,
                                 cuvsIvfFlatIndexParams_t params,
                                 DLManagedTensor* centers,
                                 DLManagedTensor* center_norms,
                                 cuvsIvfFlatIndex_t index)
{
  auto* res_ptr      = reinterpret_cast<raft::resources*>(res);
  using centers_type = raft::device_matrix_view<const float, uint32_t, raft::row_major>;
  auto centers_view  = cuvs::core::from_dlpack<centers_type>(centers);
  RAFT_EXPECTS(centers_view.extent(0) == params->n_lists, "centers row count must equal n_lists");

  auto build_params = cuvs::neighbors::ivf_flat::index_params{};
  cuvs::neighbors::ivf_flat::convert_c_index_params(*params, &build_params);
  auto* index_ptr = new cuvs::neighbors::ivf_flat::index<T, int64_t>(
    *res_ptr, build_params, centers_view.extent(1));
  cuvs::neighbors::ivf_flat::helpers::reset_index(*res_ptr, index_ptr);
  raft::copy(*res_ptr, index_ptr->centers(), centers_view);

  if (center_norms != nullptr) {
    using norms_type = raft::device_vector_view<const float, uint32_t>;
    auto norms_view  = cuvs::core::from_dlpack<norms_type>(center_norms);
    RAFT_EXPECTS(norms_view.extent(0) == params->n_lists, "center_norms length must equal n_lists");
    index_ptr->allocate_center_norms(*res_ptr);
    RAFT_EXPECTS(index_ptr->center_norms().has_value(),
                 "center norms are not supported for the configured metric");
    raft::copy(*res_ptr, index_ptr->center_norms().value(), norms_view);
  }
  index->addr = reinterpret_cast<uintptr_t>(index_ptr);
}

template <typename T>
void ivf_flat_extend_list(cuvsResources_t res,
                          cuvsIvfFlatIndex_t index,
                          DLManagedTensor* new_vectors,
                          DLManagedTensor* new_indices,
                          uint32_t label)
{
  auto* res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto* index_ptr = ivf_flat_index<T>(index);
  RAFT_EXPECTS(label < index_ptr->n_lists(), "list label is out of range");

  using vectors_type = raft::device_matrix_view<const T, uint32_t, raft::row_major>;
  using indices_type = raft::device_vector_view<const int64_t, uint32_t>;
  auto vectors       = cuvs::core::from_dlpack<vectors_type>(new_vectors);
  auto indices       = cuvs::core::from_dlpack<indices_type>(new_indices);
  RAFT_EXPECTS(vectors.extent(0) == indices.extent(0), "vectors and indices length mismatch");
  RAFT_EXPECTS(vectors.extent(1) == index_ptr->dim(), "vector dimensionality mismatch");
  if (vectors.extent(0) == 0) { return; }

  auto stream = raft::resource::get_cuda_stream(*res_ptr);
  uint32_t old_size{};
  raft::update_host(&old_size, index_ptr->list_sizes().data_handle() + label, 1, stream);
  raft::resource::sync_stream(*res_ptr);
  RAFT_EXPECTS(vectors.extent(0) <= std::numeric_limits<uint32_t>::max() - old_size,
               "list size exceeds uint32 range");
  auto new_size = old_size + vectors.extent(0);

  auto spec = cuvs::neighbors::ivf_flat::list_spec<uint32_t, T, int64_t>{
    index_ptr->dim(), index_ptr->conservative_memory_allocation()};
  cuvs::neighbors::ivf::resize_list(*res_ptr, index_ptr->lists()[label], spec, new_size, old_size);
  cuvs::neighbors::ivf_flat::helpers::codepacker::pack(
    *res_ptr, vectors, index_ptr->veclen(), old_size, index_ptr->lists()[label]->data.view());
  raft::copy(index_ptr->lists()[label]->indices.data_handle() + old_size,
             indices.data_handle(),
             indices.extent(0),
             stream);
  raft::copy(index_ptr->list_sizes().data_handle() + label, &new_size, 1, stream);
  cuvs::neighbors::ivf_flat::helpers::recompute_internal_state(*res_ptr, index_ptr);
}

}  // namespace

extern "C" cuvsError_t cuvsIvfFlatIndexReset(cuvsResources_t res, cuvsIvfFlatIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    dispatch_ivf_flat(index, [=](auto* index_ptr) {
      cuvs::neighbors::ivf_flat::helpers::reset_index(*reinterpret_cast<raft::resources*>(res),
                                                      index_ptr);
    });
  });
}

extern "C" cuvsError_t cuvsIvfFlatBuildFromCenters(cuvsResources_t res,
                                                   cuvsIvfFlatIndexParams_t params,
                                                   DLDataType index_dtype,
                                                   DLManagedTensor* centers,
                                                   DLManagedTensor* center_norms,
                                                   cuvsIvfFlatIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(index != nullptr && index->addr == 0, "output index handle must be empty");
    index->dtype = index_dtype;
    if (index_dtype.code == kDLFloat && index_dtype.bits == 32) {
      ivf_flat_build_from_centers<float>(res, params, centers, center_norms, index);
    } else if (index_dtype.code == kDLFloat && index_dtype.bits == 16) {
      ivf_flat_build_from_centers<half>(res, params, centers, center_norms, index);
    } else if (index_dtype.code == kDLInt && index_dtype.bits == 8) {
      ivf_flat_build_from_centers<int8_t>(res, params, centers, center_norms, index);
    } else if (index_dtype.code == kDLUInt && index_dtype.bits == 8) {
      ivf_flat_build_from_centers<uint8_t>(res, params, centers, center_norms, index);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index_dtype.code, index_dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsIvfFlatIndexExtendList(cuvsResources_t res,
                                                  cuvsIvfFlatIndex_t index,
                                                  DLManagedTensor* new_vectors,
                                                  DLManagedTensor* new_indices,
                                                  uint32_t label)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      ivf_flat_extend_list<float>(res, index, new_vectors, new_indices, label);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      ivf_flat_extend_list<half>(res, index, new_vectors, new_indices, label);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      ivf_flat_extend_list<int8_t>(res, index, new_vectors, new_indices, label);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      ivf_flat_extend_list<uint8_t>(res, index, new_vectors, new_indices, label);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}
