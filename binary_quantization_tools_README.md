# Binary Quantization Tools for faiss_gpu_binary_cagra

This directory contains tools for converting float datasets to binary quantized format for use with the `faiss_gpu_binary_cagra` index wrapper.

## Overview

Binary quantization converts floating-point vectors to binary vectors by comparing each element against a threshold. This results in 32x compression (from float32 to 1 bit per dimension), with each dimension packed into bits.

## Tools Provided

### 1. C++ Converter (`cpp/bench/ann/src/convert_binary_quantize.cpp`)

High-performance converter using cuVS binary quantization API directly.

#### Building
```bash
cd cpp/bench/ann/src
chmod +x build_convert_binary.sh
./build_convert_binary.sh
```

#### Usage
```bash
./convert_binary_quantize <input.fbin> <output.u8bin> [threshold_type]

# Example:
./convert_binary_quantize sift-128-euclidean.fbin sift-128-binary.u8bin 1
```

Threshold types:
- `0` = zero (default) - compare against 0
- `1` = mean - compare against per-dimension mean
- `2` = sampling_median - compare against sampled median

### 2. Python Converter (`python/convert_binary_quantize.py`)

Flexible Python script with both cuVS and NumPy implementations.

#### Requirements
```bash
pip install numpy
# Optional but recommended:
pip install cupy-cuda11x  # or appropriate CUDA version
pip install cuvs
```

#### Usage
```bash
python convert_binary_quantize.py <input.fbin> <output.u8bin> [--threshold {zero,mean,sampling_median}]

# Examples:
# Using cuVS (if available)
python convert_binary_quantize.py sift-128-euclidean.fbin sift-128-binary.u8bin --threshold mean

# Force NumPy implementation
python convert_binary_quantize.py sift-128-euclidean.fbin sift-128-binary.u8bin --use-simple
```

## File Formats

### Input: .fbin
Binary format with:
- Header: 8 bytes (2 x uint32)
  - nrows (uint32)
  - ncols (uint32)
- Data: nrows × ncols × sizeof(float32)

### Output: .u8bin
Binary quantized format with:
- Header: 8 bytes (2 x uint32)
  - nrows (uint32)
  - quantized_dim (uint32) = ⌈ncols / 8⌉
- Data: nrows × quantized_dim × sizeof(uint8)

Each uint8 packs 8 binary dimensions, with bit `i % 8` of byte `i // 8` representing dimension `i`.

## Using with faiss_gpu_binary_cagra

After converting your dataset, use it with the binary CAGRA index:

```python
import faiss
import numpy as np

# Load binary quantized data
def load_u8bin(filename):
    with open(filename, 'rb') as f:
        nrows = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        ncols = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(nrows, ncols)
    return data

# Load data
data = load_u8bin("sift-128-binary.u8bin")
queries = load_u8bin("sift-queries-binary.u8bin")

# Create binary CAGRA index
n, d_packed = data.shape
d_original = d_packed * 8  # Original dimension before packing

# Configure index
res = faiss.StandardGpuResources()
config = faiss.GpuIndexBinaryCagraConfig()
config.graph_degree = 32
config.intermediate_graph_degree = 64
config.build_algo = faiss.graph_build_algo.NN_DESCENT
config.nn_descent_niter = 20

# Build index
index = faiss.GpuIndexBinaryCagra(res, d_original, faiss.METRIC_Hamming, config)
index.train(n, data)
index.add(n, data)

# Search
k = 10
distances, neighbors = index.search(queries, k)
```

### Benchmark Configuration

For use with cuVS benchmarks, add to your configuration:

```yaml
binary_quantized_sift:
  base_file: sift-128-binary.u8bin
  query_file: sift-queries-binary.u8bin
  distance: BitwiseHamming
  dims: 128  # Original dimension (will be packed to 16 bytes)
  
faiss_gpu_binary_cagra:
  build:
    graph_degree: 32
    intermediate_graph_degree: 64
    cagra_build_algo: "NN_DESCENT"
    nn_descent_niter: 20
  search:
    - {p: {num_parents: 1, seed_ratio: 20, num_seeds: 64}}
    - {p: {num_parents: 1, seed_ratio: 40, num_seeds: 128}}
```

## Performance Notes

1. **Compression**: 32x reduction in memory usage (float32 → 1 bit)
2. **Speed**: BitwiseHamming distance is extremely fast on modern GPUs
3. **Accuracy**: Some loss compared to full precision, but often acceptable for approximate search
4. **Threshold Selection**:
   - `zero`: Best for zero-centered data
   - `mean`: Good general-purpose choice
   - `sampling_median`: Robust to outliers

## Troubleshooting

### Build Issues
- Ensure CUDA toolkit is installed and `nvcc` is in PATH
- Check that cuVS is built with binary quantization support
- Verify include paths in build script match your installation

### Runtime Issues
- Ensure input file is valid .fbin format
- Check CUDA device is available
- Verify sufficient GPU memory for dataset

### Accuracy Issues
- Try different threshold types
- Consider if binary quantization is appropriate for your data distribution
- Test with a small subset first

## Examples

Convert various standard benchmark datasets:

```bash
# SIFT dataset (128 dimensions → 16 bytes per vector)
./convert_binary_quantize sift-128-euclidean.fbin sift-128-binary.u8bin 1

# GIST dataset (960 dimensions → 120 bytes per vector)  
./convert_binary_quantize gist-960-euclidean.fbin gist-960-binary.u8bin 1

# Deep1B dataset (96 dimensions → 12 bytes per vector)
./convert_binary_quantize deep-96-angular.fbin deep-96-binary.u8bin 2
```

## References

- [cuVS Binary Quantization Documentation](https://docs.rapids.ai/api/cuvs/stable/cpp_api/preprocessing_binary.html)
- [FAISS Binary Indexes](https://github.com/facebookresearch/faiss/wiki/Binary-indexes)
- [CAGRA: GPU Graph-Based Approximate Nearest Neighbor Search](https://arxiv.org/abs/2308.15136)



