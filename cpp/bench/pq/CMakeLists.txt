# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

add_executable(FALCON_PQ_BENCH falcon_pq_benchmark.cu)
target_link_libraries(
  FALCON_PQ_BENCH
  PRIVATE cuvs::cuvs raft::raft rmm::rmm CUDA::cudart CUDA::nvtx3 Threads::Threads
          ${CUVS_CTK_MATH_DEPENDENCIES} $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
          $<TARGET_NAME_IF_EXISTS:conda_env>
)
set_target_properties(
  FALCON_PQ_BENCH
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CUVS_BINARY_DIR}/bench/pq"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  FALCON_PQ_BENCH
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUVS_CXX_FLAGS}>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${CUVS_CUDA_FLAGS}>"
)
