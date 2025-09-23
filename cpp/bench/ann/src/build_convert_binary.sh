#!/bin/bash

# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build script for the binary quantization conversion utility

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Binary Quantization Converter...${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../../.." && pwd )"

# Create build directory
BUILD_DIR="${PROJECT_ROOT}/cpp/build"
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Creating build directory: ${BUILD_DIR}${NC}"
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# Configure with CMake if not already configured
if [ ! -f "CMakeCache.txt" ]; then
    echo -e "${YELLOW}Configuring with CMake...${NC}"
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_CUDA_ARCHITECTURES="75;80;86;90" \
             -DBUILD_TESTS=OFF \
             -DBUILD_CUVS_BENCH_ANN=ON
fi

# Build the converter
echo -e "${GREEN}Compiling convert_binary_quantize...${NC}"

# Check if we can build the target directly
if cmake --build . --target convert_binary_quantize 2>/dev/null; then
    echo -e "${GREEN}Successfully built convert_binary_quantize${NC}"
else
    # If the target doesn't exist, compile manually
    echo -e "${YELLOW}Target not found in CMake, compiling manually...${NC}"
    
    # Find necessary include paths and libraries
    CUDA_PATH=${CUDA_PATH:-/usr/local/cuda}
    
    # Compile command
    nvcc -std=c++17 \
         -I"${PROJECT_ROOT}/cpp/include" \
         -I"${PROJECT_ROOT}/_deps/raft-src/cpp/include" \
         -I"${CUDA_PATH}/include" \
         -I"${BUILD_DIR}/_deps/cccl-src/thrust" \
         -I"${BUILD_DIR}/_deps/cccl-src/libcudacxx/include" \
         -I"${BUILD_DIR}/_deps/cccl-src/cub" \
         -I"${BUILD_DIR}/_deps/dlpack-src/include" \
         -I"${BUILD_DIR}/_deps/fmt-src/include" \
         -I"${BUILD_DIR}/_deps/spdlog-src/include" \
         -arch=sm_75 \
         -O3 \
         -o convert_binary_quantize \
         "${SCRIPT_DIR}/convert_binary_quantize.cpp" \
         -L"${BUILD_DIR}/src" -lcuvs \
         -L"${CUDA_PATH}/lib64" -lcudart -lcublas -lcusparse \
         -L"${BUILD_DIR}/_deps/raft-build/src" -lraft
fi

# Check if build was successful
if [ -f "${BUILD_DIR}/convert_binary_quantize" ] || [ -f "${BUILD_DIR}/cpp/bench/ann/convert_binary_quantize" ]; then
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo -e "${GREEN}Binary location: ${BUILD_DIR}/convert_binary_quantize${NC}"
    echo ""
    echo "Usage:"
    echo "  ./convert_binary_quantize <input.fbin> <output.u8bin> [threshold_type]"
    echo ""
    echo "Threshold types:"
    echo "  0 = zero (default)"
    echo "  1 = mean"
    echo "  2 = sampling_median"
else
    echo -e "${RED}✗ Build failed!${NC}"
    exit 1
fi



