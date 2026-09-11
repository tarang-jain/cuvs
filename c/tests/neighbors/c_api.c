/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/all_neighbors.h>
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/ivf_flat.h>
#include <cuvs/neighbors/ivf_pq.h>
#include <cuvs/neighbors/ivf_sq.h>
#include <cuvs/neighbors/tiered_index.h>

#include <dlpack/dlpack.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

void test_compile_cagra()
{
  // simple smoke test to make sure that we can compile the cagra.h API
  // using a c compiler. This isn't aiming to be a full test, just checking
  // that the exposed C-API is valid C code and doesn't contain C++ features
  assert(!"test_compile_cagra is not meant to be run");

  cuvsCagraIndex_t index;
  cuvsCagraIndexCreate(&index);
  cuvsCagraIndexDestroy(index);
}

void test_compile_tiered_index()
{
  // Smoke test to ensure that the tiered_index.h API compiles correctly
  // using a c compiler. Not a full test.
  assert(!"test_compile_tiered_index is not meant to be run");

  cuvsTieredIndex_t tiered_index;
  cuvsTieredIndexCreate(&tiered_index);
  cuvsTieredIndexDestroy(tiered_index);

  cuvsTieredIndexParams_t index_params;
  cuvsResources_t resources;
  cuvsFilter prefilter;
  DLManagedTensor dataset, neighbors, distances;
  cuvsTieredIndexParamsCreate(&index_params);
  cuvsTieredIndexParamsDestroy(index_params);
  cuvsTieredIndexBuild(resources, index_params, &dataset, tiered_index);
  cuvsTieredIndexSearch(resources, NULL, tiered_index, &dataset, &neighbors, &distances, prefilter);
  cuvsTieredIndexExtend(resources, &dataset, tiered_index);
}

void test_compile_ivf_sq()
{
  assert(!"test_compile_ivf_sq is not meant to be run");

  cuvsIvfSqIndex_t index;
  cuvsIvfSqIndexCreate(&index);
  cuvsIvfSqIndexDestroy(index);
}

void test_compile_all_neighbors()
{
  // Smoke test to ensure that the all_neighbors.h API compiles correctly
  // using a c compiler. Not a full test.
  assert(!"test_compile_all_neighbors is not meant to be run");

  cuvsAllNeighborsIndexParams_t params;
  cuvsResources_t resources;
  DLManagedTensor dataset, indices, distances, core_distances;
  cuvsAllNeighborsIndexParamsCreate(&params);
  cuvsAllNeighborsIndexParamsDestroy(params);
  cuvsAllNeighborsBuild(resources, params, &dataset, &indices, &distances, &core_distances, 1.0f);
}

void test_compile_faiss_extension_apis()
{
  assert(!"test_compile_faiss_extension_apis is not meant to be run");

  cuvsResources_t resources = 0;
  DLManagedTensor tensor;
  DLDataType dtype  = {kDLFloat, 32, 1};
  cuvsFilter filter = {0, NO_FILTER};

  cuvsBruteForceIndex_t brute_force;
  cuvsBruteForceBuildWithNorms(resources, &tensor, &tensor, L2Expanded, 0.0f, brute_force);

  cuvsCagraIndexParams_t cagra_params;
  cuvsCagraIndexParamsCreate(&cagra_params);
  cagra_params->guarantee_connectivity = true;
  cuvsCagraOptimizeGraph(resources, &tensor, &tensor, true);

  cuvsAllNeighborsIndexParams_t all_neighbors_params;
  cuvsAllNeighborsIndexParamsCreate(&all_neighbors_params);
  all_neighbors_params->ivf_pq_search_params   = NULL;
  all_neighbors_params->ivf_pq_refinement_rate = 1.0f;
  all_neighbors_params->ivf_pq_sizing_rows     = 0;

  cuvsIvfFlatIndex_t flat;
  int64_t size;
  cuvsIvfFlatIndexReset(resources, flat);
  cuvsIvfFlatIndexGetSize(flat, &size);
  cuvsIvfFlatIndexGetListSizes(flat, &tensor);
  cuvsIvfFlatIndexGetListIndices(flat, 0, &tensor);
  cuvsIvfFlatIndexUnpackListData(resources, flat, &tensor, 0, 0);
  cuvsIvfFlatBuildFromCenters(resources, NULL, dtype, &tensor, &tensor, flat);
  cuvsIvfFlatIndexExtendList(resources, flat, &tensor, &tensor, 0);

  cuvsIvfPqIndex_t pq;
  cuvsIvfPqIndexReset(resources, pq);
  cuvsIvfPqSearchWithFilter(resources, NULL, pq, &tensor, &tensor, &tensor, filter);
  cuvsIvfPqIndexExtendList(resources, pq, &tensor, &tensor, 0);

  cuvsIvfSqIndex_t sq;
  cuvsIvfSqIndexReset(resources, sq);
  cuvsIvfSqIndexGetListSizes(sq, &tensor);
  cuvsIvfSqIndexGetListIndices(sq, 0, &tensor);
  cuvsIvfSqIndexUnpackListData(resources, sq, &tensor, 0, 0);
  cuvsIvfSqIndexGetVMin(sq, &tensor);
  cuvsIvfSqIndexGetDelta(sq, &tensor);
  cuvsIvfSqBuildFromCenters(resources, NULL, dtype, &tensor, &tensor, &tensor, &tensor, sq);
  cuvsIvfSqIndexExtendList(resources, sq, &tensor, &tensor, 0);
}

int main()
{
  if (getenv("CUVS_RUN_COMPILE_ONLY_API_CALLS") != NULL) { test_compile_faiss_extension_apis(); }
  // These are smoke tests that check that the C-APIs compile with a C compiler.
  // These are not meant to be run.
  test_compile_cagra();
  test_compile_ivf_sq();
  test_compile_tiered_index();
  test_compile_all_neighbors();

  return 0;
}
