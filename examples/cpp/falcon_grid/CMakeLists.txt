# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(falcon_pq_device_grid LANGUAGES CXX CUDA)

find_package(cuvs REQUIRED)
find_library(CUVS_STATIC_LIBRARY cuvs_static REQUIRED)

add_executable(FALCON_PQ_DEVICE_GRID ../src/falcon_pq_device_grid.cu)
target_compile_features(FALCON_PQ_DEVICE_GRID PRIVATE cxx_std_20 cuda_std_20)
target_include_directories(
  FALCON_PQ_DEVICE_GRID BEFORE PRIVATE ../../../cpp/include ../../../cpp/src
)
target_link_libraries(FALCON_PQ_DEVICE_GRID PRIVATE ${CUVS_STATIC_LIBRARY} cuvs::cuvs)
