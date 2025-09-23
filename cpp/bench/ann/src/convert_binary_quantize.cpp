/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuvs/preprocessing/quantize/binary.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/util/integer_utils.hpp>

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Utility to read binary file header (nrows, ncols)
template <typename T>
void read_bin(const std::string& filename, std::vector<T>& data, size_t& nrows, size_t& ncols)
{
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  // Read dimensions
  uint32_t n, d;
  file.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
  nrows = n;
  ncols = d;

  // Read data
  size_t total_elements = nrows * ncols;
  data.resize(total_elements);
  file.read(reinterpret_cast<char*>(data.data()), total_elements * sizeof(T));
  file.close();

  std::cout << "Read dataset: " << nrows << " rows, " << ncols << " columns" << std::endl;
}

// Utility to write binary file
template <typename T>
void write_bin(const std::string& filename, const T* data, size_t nrows, size_t ncols)
{
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot create file: " + filename);
  }

  // Write dimensions
  uint32_t n = static_cast<uint32_t>(nrows);
  uint32_t d = static_cast<uint32_t>(ncols);
  file.write(reinterpret_cast<char*>(&n), sizeof(uint32_t));
  file.write(reinterpret_cast<char*>(&d), sizeof(uint32_t));

  // Write data
  size_t total_elements = nrows * ncols;
  file.write(reinterpret_cast<const char*>(data), total_elements * sizeof(T));
  file.close();

  std::cout << "Wrote dataset: " << nrows << " rows, " << ncols << " columns to " << filename
            << std::endl;
}

int main(int argc, char* argv[])
{
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.fbin> <output.u8bin> [threshold_type]"
              << std::endl;
    std::cerr << "threshold_type: 0=zero (default), 1=mean, 2=sampling_median" << std::endl;
    return 1;
  }

  std::string input_file  = argv[1];
  std::string output_file = argv[2];

  // Parse optional threshold type
  cuvs::preprocessing::quantize::binary::bit_threshold threshold =
    cuvs::preprocessing::quantize::binary::bit_threshold::zero;
  if (argc > 3) {
    int threshold_type = std::stoi(argv[3]);
    switch (threshold_type) {
      case 0:
        threshold = cuvs::preprocessing::quantize::binary::bit_threshold::zero;
        std::cout << "Using zero threshold" << std::endl;
        break;
      case 1:
        threshold = cuvs::preprocessing::quantize::binary::bit_threshold::mean;
        std::cout << "Using mean threshold" << std::endl;
        break;
      case 2:
        threshold = cuvs::preprocessing::quantize::binary::bit_threshold::sampling_median;
        std::cout << "Using sampling_median threshold" << std::endl;
        break;
      default:
        std::cerr << "Invalid threshold type. Using zero." << std::endl;
        threshold = cuvs::preprocessing::quantize::binary::bit_threshold::zero;
    }
  }

  try {
    // Initialize CUDA
    cudaSetDevice(0);

    // Create RAFT resources
    raft::device_resources handle;

    // Read input float dataset
    std::vector<float> host_data;
    size_t nrows, ncols;
    read_bin<float>(input_file, host_data, nrows, ncols);

    // Calculate quantized dimensions (pack 8 bits per byte)
    int64_t quantized_dim = raft::div_rounding_up_safe<int64_t>(ncols, 8);
    std::cout << "Original dim: " << ncols << ", Quantized dim: " << quantized_dim << std::endl;

    // Copy data to device
    auto device_data = raft::make_device_matrix<float, int64_t>(handle, nrows, ncols);
    raft::copy(device_data.data_handle(),
               host_data.data(),
               nrows * ncols,
               handle.get_stream());
    handle.sync_stream();

    // Create output buffer for quantized data
    auto quantized_device = raft::make_device_matrix<uint8_t, int64_t>(handle, nrows, quantized_dim);

    // Set up binary quantization parameters
    cuvs::preprocessing::quantize::binary::params params;
    params.threshold     = threshold;
    params.sampling_ratio = 0.1f;  // default sampling ratio for sampling_median

    // Train the quantizer
    std::cout << "Training binary quantizer..." << std::endl;
    auto quantizer = cuvs::preprocessing::quantize::binary::train(
      handle, params, raft::make_const_mdspan(device_data.view()));

    // Apply quantization transform
    std::cout << "Applying binary quantization transform..." << std::endl;
    cuvs::preprocessing::quantize::binary::transform(
      handle, quantizer, raft::make_const_mdspan(device_data.view()), quantized_device.view());

    // Copy quantized data back to host
    std::vector<uint8_t> host_quantized(nrows * quantized_dim);
    raft::copy(host_quantized.data(),
               quantized_device.data_handle(),
               nrows * quantized_dim,
               handle.get_stream());
    handle.sync_stream();

    // Write output file
    write_bin<uint8_t>(output_file, host_quantized.data(), nrows, quantized_dim);

    std::cout << "Binary quantization complete!" << std::endl;
    std::cout << "Output saved to: " << output_file << std::endl;

    // Print some statistics
    size_t original_size  = nrows * ncols * sizeof(float);
    size_t quantized_size = nrows * quantized_dim * sizeof(uint8_t);
    float compression_ratio = static_cast<float>(original_size) / static_cast<float>(quantized_size);
    std::cout << "Original size: " << original_size << " bytes" << std::endl;
    std::cout << "Quantized size: " << quantized_size << " bytes" << std::endl;
    std::cout << "Compression ratio: " << compression_ratio << "x" << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}



