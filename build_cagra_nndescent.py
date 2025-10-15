#!/usr/bin/env python3
"""
Script to build a CAGRA graph using NN Descent algorithm with RMM pool memory resource.
"""

import numpy as np
import cupy as cp
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
from cuvs.neighbors import cagra

# Set up RMM pool memory resource
# Create a CUDA memory resource as upstream
cuda_mr = rmm.mr.CudaMemoryResource()

# Create a pool memory resource with the CUDA MR as upstream
# Using 2GB initial pool size and 10GB maximum pool size
# pool_mr = rmm.mr.PoolMemoryResource(
#     cuda_mr,
#     initial_pool_size=2 * 1024**3,  # 2 GB
#     # maximum_pool_size=10 * 1024**3  # 10 GB
# )

# Set the pool as the current memory resource
rmm.mr.set_current_device_resource(cuda_mr)

# Configure CuPy to use RMM for memory allocation
cp.cuda.set_allocator(rmm_cupy_allocator)

print("RMM pool memory resource configured successfully")
# print(f"Pool initial size: 2 GB")
# print(f"Pool maximum size: 10 GB")

# Create random int8_t dataset
# 30M rows x 1024 columns
n_rows = 30_000_000
n_cols = 1024

print(f"\nCreating random int8 dataset: {n_rows} x {n_cols}")
print(f"Estimated memory: {n_rows * n_cols / (1024**3):.2f} GB")

# Generate random int8 data on GPU
shape = (30_000_000, 1024)
n_bytes = shape[0] * shape[1]

arr = cp.frombuffer(cp.random.bytes(n_bytes), dtype=cp.int8).reshape(shape)

print("Dataset created successfully")

# Configure CAGRA index parameters with NN Descent
print("\nConfiguring CAGRA index parameters...")
print(f"Graph degree: 64")
print(f"Intermediate graph degree: 128")
print(f"Graph build algorithm: NN_DESCENT")

# Create index parameters for NN Descent
index_params = cagra.IndexParams(
    metric="sqeuclidean",  # Metric for int8 data
    build_algo="nn_descent",
    graph_degree=64,
    intermediate_graph_degree=128
)

print("\nBuilding CAGRA index with NN Descent...")
# Build the index
index = cagra.build(index_params, arr)

print("CAGRA index built successfully!")
print(f"\nIndex details:")
print(f"  Dataset shape: {arr.shape}")
print(f"  Dataset dtype: {arr.dtype}")
print(f"  Graph degree: {index_params.graph_degree}")
print(f"  Metric: {index_params.metric}")

# Optional: Print memory statistics
print(f"\nMemory statistics:")
mempool = cp.get_default_memory_pool()
print(f"  Used bytes: {mempool.used_bytes() / (1024**3):.2f} GB")
print(f"  Total bytes: {mempool.total_bytes() / (1024**3):.2f} GB")

print("\nScript completed successfully!")



