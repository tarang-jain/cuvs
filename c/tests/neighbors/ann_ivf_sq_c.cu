/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cuda.h>
#include <vector>

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>

#include "neighbors/ann_utils.cuh"
#include <cuvs/neighbors/ivf_sq.h>

extern "C" void run_ivf_sq(cuvsResources_t res,
                           int64_t n_rows,
                           int64_t n_queries,
                           int64_t n_dim,
                           uint32_t n_neighbors,
                           float* index_data,
                           float* query_data,
                           float* distances_data,
                           int64_t* neighbors_data,
                           cuvsDistanceType metric,
                           size_t n_probes,
                           size_t n_lists);

template <typename T>
void generate_random_data(raft::handle_t const& handle, T* devPtr, size_t size)
{
  raft::random::RngState r(1234ULL);
  raft::random::uniform(handle, r, devPtr, size, T(0.1), T(2.0));
};

template <typename T, typename IdxT>
void recall_eval(raft::handle_t const& handle,
                 T* query_data,
                 T* index_data,
                 IdxT* neighbors,
                 T* distances,
                 size_t n_queries,
                 size_t n_rows,
                 size_t n_dim,
                 size_t n_neighbors,
                 cuvsDistanceType metric,
                 size_t n_probes,
                 size_t n_lists)
{
  auto distances_ref = raft::make_device_matrix<T, IdxT>(handle, n_queries, n_neighbors);
  auto neighbors_ref = raft::make_device_matrix<IdxT, IdxT>(handle, n_queries, n_neighbors);
  cuvs::neighbors::naive_knn<T, T, IdxT>(
    handle,
    distances_ref.data_handle(),
    neighbors_ref.data_handle(),
    query_data,
    index_data,
    n_queries,
    n_rows,
    n_dim,
    n_neighbors,
    static_cast<cuvs::distance::DistanceType>((uint16_t)metric));

  size_t size = n_queries * n_neighbors;
  std::vector<IdxT> neighbors_h(size);
  std::vector<T> distances_h(size);
  std::vector<IdxT> neighbors_ref_h(size);
  std::vector<T> distances_ref_h(size);

  auto stream = raft::resource::get_cuda_stream(handle);
  raft::copy(neighbors_h.data(), neighbors, size, stream);
  raft::copy(distances_h.data(), distances, size, stream);
  raft::copy(neighbors_ref_h.data(), neighbors_ref.data_handle(), size, stream);
  raft::copy(distances_ref_h.data(), distances_ref.data_handle(), size, stream);
  raft::resource::sync_stream(handle);

  double min_recall = static_cast<double>(n_probes) / static_cast<double>(n_lists);
  ASSERT_TRUE(cuvs::neighbors::eval_neighbours(neighbors_ref_h,
                                               neighbors_h,
                                               distances_ref_h,
                                               distances_h,
                                               n_queries,
                                               n_neighbors,
                                               0.001,
                                               min_recall));
};

TEST(IvfSqC, BuildSearch)
{
  int64_t n_rows       = 8096;
  int64_t n_queries    = 128;
  int64_t n_dim        = 32;
  uint32_t n_neighbors = 8;

  raft::handle_t handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  cuvsDistanceType metric = L2Expanded;
  size_t n_probes         = 20;
  size_t n_lists          = 1024;

  rmm::device_uvector<float> index_data(n_rows * n_dim, stream);
  rmm::device_uvector<float> query_data(n_queries * n_dim, stream);
  rmm::device_uvector<int64_t> neighbors_data(n_queries * n_neighbors, stream);
  rmm::device_uvector<float> distances_data(n_queries * n_neighbors, stream);

  generate_random_data(handle, index_data.data(), n_rows * n_dim);
  generate_random_data(handle, query_data.data(), n_queries * n_dim);

  cuvsResources_t res;
  cuvsResourcesCreate(&res);
  cuvsStreamSet(res, stream.get());

  run_ivf_sq(res,
             n_rows,
             n_queries,
             n_dim,
             n_neighbors,
             index_data.data(),
             query_data.data(),
             distances_data.data(),
             neighbors_data.data(),
             metric,
             n_probes,
             n_lists);

  recall_eval(handle,
              query_data.data(),
              index_data.data(),
              neighbors_data.data(),
              distances_data.data(),
              n_queries,
              n_rows,
              n_dim,
              n_neighbors,
              metric,
              n_probes,
              n_lists);

  cuvsResourcesDestroy(res);
}

TEST(IvfSqC, ExtendAndUnpackOddDimension)
{
  constexpr uint32_t n_rows = 37;
  constexpr uint32_t n_dim  = 19;

  raft::handle_t handle;
  auto stream  = raft::resource::get_cuda_stream(handle);
  auto centers = raft::make_device_matrix<float, uint32_t>(handle, 1, n_dim);
  auto vmin    = raft::make_device_vector<float, uint32_t>(handle, n_dim);
  auto delta   = raft::make_device_vector<float, uint32_t>(handle, n_dim);
  auto codes   = raft::make_device_matrix<uint8_t, uint32_t>(handle, n_rows, n_dim);
  auto ids     = raft::make_device_vector<int64_t, uint32_t>(handle, n_rows);
  auto output  = raft::make_device_matrix<uint8_t, uint32_t>(handle, n_rows, n_dim);

  std::vector<float> centers_h(n_dim, 0.0f);
  std::vector<float> vmin_h(n_dim, 0.0f);
  std::vector<float> delta_h(n_dim, 1.0f);
  std::vector<uint8_t> codes_h(n_rows * n_dim);
  std::vector<int64_t> ids_h(n_rows);
  for (uint32_t row = 0; row < n_rows; ++row) {
    ids_h[row] = static_cast<int64_t>(1000 + row);
    for (uint32_t col = 0; col < n_dim; ++col) {
      codes_h[row * n_dim + col] = static_cast<uint8_t>((row * 31 + col * 7) & 0xff);
    }
  }
  raft::copy(centers.data_handle(), centers_h.data(), centers_h.size(), stream);
  raft::copy(vmin.data_handle(), vmin_h.data(), vmin_h.size(), stream);
  raft::copy(delta.data_handle(), delta_h.data(), delta_h.size(), stream);
  raft::copy(codes.data_handle(), codes_h.data(), codes_h.size(), stream);
  raft::copy(ids.data_handle(), ids_h.data(), ids_h.size(), stream);

  int device_id = 0;
  ASSERT_EQ(cudaGetDevice(&device_id), cudaSuccess);
  auto make_tensor = [device_id](void* data, DLDataType dtype, int ndim, int64_t* shape) {
    DLManagedTensor tensor{};
    tensor.dl_tensor.data        = data;
    tensor.dl_tensor.device      = DLDevice{kDLCUDA, device_id};
    tensor.dl_tensor.ndim        = ndim;
    tensor.dl_tensor.dtype       = dtype;
    tensor.dl_tensor.shape       = shape;
    tensor.dl_tensor.strides     = nullptr;
    tensor.dl_tensor.byte_offset = 0;
    return tensor;
  };

  int64_t centers_shape[2] = {1, n_dim};
  int64_t range_shape[1]   = {n_dim};
  int64_t codes_shape[2]   = {n_rows, n_dim};
  int64_t ids_shape[1]     = {n_rows};
  auto centers_tensor =
    make_tensor(centers.data_handle(), DLDataType{kDLFloat, 32, 1}, 2, centers_shape);
  auto vmin_tensor  = make_tensor(vmin.data_handle(), DLDataType{kDLFloat, 32, 1}, 1, range_shape);
  auto delta_tensor = make_tensor(delta.data_handle(), DLDataType{kDLFloat, 32, 1}, 1, range_shape);
  auto codes_tensor = make_tensor(codes.data_handle(), DLDataType{kDLUInt, 8, 1}, 2, codes_shape);
  auto ids_tensor   = make_tensor(ids.data_handle(), DLDataType{kDLInt, 64, 1}, 1, ids_shape);
  auto output_tensor = make_tensor(output.data_handle(), DLDataType{kDLUInt, 8, 1}, 2, codes_shape);

  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);
  ASSERT_EQ(cuvsStreamSet(res, stream.get()), CUVS_SUCCESS);

  cuvsIvfSqIndexParams_t params;
  ASSERT_EQ(cuvsIvfSqIndexParamsCreate(&params), CUVS_SUCCESS);
  params->n_lists                        = 1;
  params->add_data_on_build              = false;
  params->conservative_memory_allocation = true;

  cuvsIvfSqIndex_t index;
  ASSERT_EQ(cuvsIvfSqIndexCreate(&index), CUVS_SUCCESS);
  ASSERT_EQ(cuvsIvfSqBuildFromCenters(res,
                                      params,
                                      DLDataType{kDLFloat, 32, 1},
                                      &centers_tensor,
                                      nullptr,
                                      &vmin_tensor,
                                      &delta_tensor,
                                      index),
            CUVS_SUCCESS);
  ASSERT_EQ(cuvsIvfSqIndexExtendList(res, index, &codes_tensor, &ids_tensor, 0), CUVS_SUCCESS);
  ASSERT_EQ(cuvsIvfSqIndexUnpackContiguousListData(res, index, &output_tensor, 0, 0), CUVS_SUCCESS);
  ASSERT_EQ(cuvsStreamSync(res), CUVS_SUCCESS);

  std::vector<uint8_t> output_h(codes_h.size());
  raft::copy(output_h.data(), output.data_handle(), output_h.size(), stream);
  raft::resource::sync_stream(handle);
  EXPECT_EQ(output_h, codes_h);

  DLManagedTensor list_indices{};
  ASSERT_EQ(cuvsIvfSqIndexGetListIndices(index, 0, &list_indices), CUVS_SUCCESS);
  std::vector<int64_t> output_ids_h(n_rows);
  raft::copy(output_ids_h.data(),
             static_cast<int64_t*>(list_indices.dl_tensor.data),
             output_ids_h.size(),
             stream);
  raft::resource::sync_stream(handle);
  EXPECT_EQ(output_ids_h, ids_h);
  if (list_indices.deleter != nullptr) { list_indices.deleter(&list_indices); }

  int64_t size = 0;
  ASSERT_EQ(cuvsIvfSqIndexGetSize(index, &size), CUVS_SUCCESS);
  EXPECT_EQ(size, n_rows);
  ASSERT_EQ(cuvsIvfSqIndexReset(res, index), CUVS_SUCCESS);
  ASSERT_EQ(cuvsIvfSqIndexGetSize(index, &size), CUVS_SUCCESS);
  EXPECT_EQ(size, 0);

  EXPECT_EQ(cuvsIvfSqIndexDestroy(index), CUVS_SUCCESS);
  EXPECT_EQ(cuvsIvfSqIndexParamsDestroy(params), CUVS_SUCCESS);
  EXPECT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
}
