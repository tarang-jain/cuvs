/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>

#include <dlpack/dlpack.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generic dataset layout kind for C API dataset handles.
 */
typedef enum {
  CUVS_DATASET_LAYOUT_STANDARD = 0,
  CUVS_DATASET_LAYOUT_PADDED   = 1,
  CUVS_DATASET_LAYOUT_PQ       = 2
} cuvsDatasetLayout_t;

/**
 * @brief Memory space holding a C API dataset handle's data.
 */
typedef enum {
  CUVS_DATASET_MEM_TYPE_HOST   = 0,
  CUVS_DATASET_MEM_TYPE_DEVICE = 1
} cuvsDatasetMemType_t;

/**
 * @brief Dataset handle representing owning storage or a non-owning view.
 *
 * `addr` points to C++ dataset storage or view metadata managed by the C API. `mem_type`
 * identifies the memory space, `layout` identifies the data layout, and `is_owning` indicates
 * whether the handle owns its backing data.
 */
typedef struct {
  uintptr_t addr;
  void (*destroy_addr)(void*);
  DLDataType dtype;
  cuvsDatasetMemType_t mem_type;
  cuvsDatasetLayout_t layout;
  bool is_owning;
} cuvsDataset;
typedef cuvsDataset* cuvsDataset_t;

/** Parameters for creating a PQ-compressed dataset. */
struct cuvsPqParams {
  /**
   * The bit length of each encoded PQ element. Possible values are [4, 5, 6, 7, 8].
   * Smaller values reduce index size and can improve search performance at the cost of recall.
   */
  uint32_t pq_bits;
  /**
   * The dimensionality of the vector after PQ compression. When zero, a heuristic selects it.
   * The source dimension must currently be a multiple of `pq_dim`.
   */
  uint32_t pq_dim;
  /** VQ codebook size. When zero, a heuristic selects the number of coarse cluster centers. */
  uint32_t vq_n_centers;
  /** Number of k-means iterations for both VQ and PQ training. */
  uint32_t kmeans_n_iters;
  /** Fraction of the input used for VQ k-means training. Zero selects a heuristic value. */
  double vq_kmeans_trainset_fraction;
  /** Fraction of the input used for PQ k-means training. Zero selects a heuristic value. */
  double pq_kmeans_trainset_fraction;
};
typedef struct cuvsPqParams* cuvsPqParams_t;

/** @brief Allocate PQ parameters and populate them with defaults. */
CUVS_EXPORT cuvsError_t cuvsPqParamsCreate(cuvsPqParams_t* params);

/** @brief Destroy PQ parameters. */
CUVS_EXPORT cuvsError_t cuvsPqParamsDestroy(cuvsPqParams_t params);

/**
 * @brief Create an empty owning dataset handle.
 *
 * The dataset storage, memory type, layout, and dtype are populated by the operation that fills
 * this handle.
 */
CUVS_EXPORT cuvsError_t cuvsDatasetCreate(cuvsDataset_t* dataset);

/**
 * @brief Create an owning padded dataset in the requested memory space.
 *
 * The source tensor may reside in host- or device-accessible memory. Its contents are copied into
 * newly allocated padded storage in `target_mem_type`.
 *
 * @param[in] res cuVS resources
 * @param[in] dataset source tensor
 * @param[in] target_mem_type memory space in which to allocate the padded dataset
 * @param[out] padded_dataset newly allocated owning padded dataset
 */
CUVS_EXPORT cuvsError_t cuvsDatasetMakePadded(cuvsResources_t res,
                                              DLManagedTensor* dataset,
                                              cuvsDatasetMemType_t target_mem_type,
                                              cuvsDataset_t* padded_dataset);

/**
 * @brief Compress a dense dataset into a device PQ dataset.
 */
CUVS_EXPORT cuvsError_t cuvsDatasetMakePq(cuvsResources_t res,
                                          cuvsPqParams_t params,
                                          cuvsDataset_t dataset,
                                          cuvsDataset_t* pq_dataset);

/**
 * @brief Create a non-owning padded dataset view from a host- or device-resident tensor.
 *
 * Memory residency is inferred from the tensor.
 */
CUVS_EXPORT cuvsError_t cuvsDatasetMakePaddedView(cuvsResources_t res,
                                                  DLManagedTensor* dataset,
                                                  cuvsDataset_t* padded_dataset);

/**
 * @brief Create a non-owning standard dataset view from a host- or device-resident tensor.
 *
 * Memory residency is inferred from the tensor.
 */
CUVS_EXPORT cuvsError_t cuvsDatasetMakeStandardView(cuvsResources_t res,
                                                    DLManagedTensor* dataset,
                                                    cuvsDataset_t* standard_dataset);

/** @brief Destroy a dataset handle created by a `cuvsDatasetMake*` function. */
CUVS_EXPORT cuvsError_t cuvsDatasetDestroy(cuvsDataset_t dataset);

/** @brief Get the memory residency of a dataset handle. */
CUVS_EXPORT cuvsError_t cuvsDatasetGetMemType(cuvsDataset_t dataset,
                                              cuvsDatasetMemType_t* mem_type);

/** @brief Get the layout of a dataset handle. */
CUVS_EXPORT cuvsError_t cuvsDatasetGetLayout(cuvsDataset_t dataset, cuvsDatasetLayout_t* layout);

/** @brief Get whether a dataset handle owns its backing storage. */
CUVS_EXPORT cuvsError_t cuvsDatasetGetIsOwning(cuvsDataset_t dataset, bool* is_owning);

/** @brief Get the element dtype of a dataset handle. */
CUVS_EXPORT cuvsError_t cuvsDatasetGetDtype(cuvsDataset_t dataset, DLDataType* dtype);

#ifdef __cplusplus
}
#endif
