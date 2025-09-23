#!/usr/bin/env python3
"""
Copyright (c) 2025, NVIDIA CORPORATION.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import numpy as np
import struct
import os
try:
    import cupy as cp
    from cuvs.preprocessing import binary_quantize
    HAS_CUVS = True
except ImportError:
    HAS_CUVS = False
    print("Warning: cuVS not available. Will use simple NumPy-based quantization.")


def read_fbin(filename):
    """Read float binary file in .fbin format."""
    with open(filename, 'rb') as f:
        # Read header (nrows, ncols as uint32)
        nrows = struct.unpack('I', f.read(4))[0]
        ncols = struct.unpack('I', f.read(4))[0]
        
        # Read float data
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(nrows, ncols)
        
    print(f"Read dataset: {nrows} rows, {ncols} columns")
    return data


def write_u8bin(filename, data):
    """Write uint8 binary file in .u8bin format."""
    nrows, ncols = data.shape
    with open(filename, 'wb') as f:
        # Write header (nrows, ncols as uint32)
        f.write(struct.pack('I', nrows))
        f.write(struct.pack('I', ncols))
        
        # Write uint8 data
        f.write(data.astype(np.uint8).tobytes())
        
    print(f"Wrote dataset: {nrows} rows, {ncols} columns to {filename}")


def binary_quantize_simple(data, threshold='zero'):
    """
    Simple binary quantization without cuVS.
    Pack 8 bits into each uint8_t.
    """
    nrows, ncols = data.shape
    quantized_dim = (ncols + 7) // 8  # Ceiling division
    
    # Calculate threshold
    if threshold == 'mean':
        thresh = np.mean(data, axis=0, keepdims=True)
    elif threshold == 'zero':
        thresh = np.zeros((1, ncols), dtype=np.float32)
    else:  # median
        thresh = np.median(data, axis=0, keepdims=True)
    
    # Binarize: 1 if data > threshold, else 0
    binary = (data > thresh).astype(np.uint8)
    
    # Pack bits into bytes
    quantized = np.zeros((nrows, quantized_dim), dtype=np.uint8)
    for i in range(ncols):
        byte_idx = i // 8
        bit_idx = i % 8
        quantized[:, byte_idx] |= (binary[:, i] << bit_idx)
    
    return quantized


def binary_quantize_cuvs(data, threshold='zero'):
    """
    Binary quantization using cuVS API.
    Pack 8 bits into each uint8_t.
    """
    import cupy as cp
    from cuvs.preprocessing import binary_quantize
    
    nrows, ncols = data.shape
    quantized_dim = (ncols + 7) // 8
    
    # Convert to CuPy array
    data_gpu = cp.asarray(data, dtype=cp.float32)
    
    # Create output array
    quantized_gpu = cp.zeros((nrows, quantized_dim), dtype=cp.uint8)
    
    # Map threshold types
    threshold_map = {
        'zero': 0,
        'mean': 1,
        'sampling_median': 2
    }
    
    # Perform binary quantization
    binary_quantize(
        data_gpu,
        quantized_gpu,
        threshold=threshold_map.get(threshold, 0),
        sampling_ratio=0.1
    )
    
    # Copy back to host
    quantized = cp.asnumpy(quantized_gpu)
    
    return quantized


def main():
    parser = argparse.ArgumentParser(
        description='Convert float dataset to binary quantized uint8 format'
    )
    parser.add_argument('input_file', help='Input .fbin file')
    parser.add_argument('output_file', help='Output .u8bin file')
    parser.add_argument(
        '--threshold',
        choices=['zero', 'mean', 'sampling_median'],
        default='zero',
        help='Threshold type for binary quantization (default: zero)'
    )
    parser.add_argument(
        '--use-simple',
        action='store_true',
        help='Force use of simple NumPy implementation instead of cuVS'
    )
    
    args = parser.parse_args()
    
    # Read input data
    data = read_fbin(args.input_file)
    nrows, ncols = data.shape
    
    # Calculate quantized dimensions
    quantized_dim = (ncols + 7) // 8
    print(f"Original dim: {ncols}, Quantized dim: {quantized_dim}")
    
    # Perform binary quantization
    if HAS_CUVS and not args.use_simple:
        print(f"Using cuVS binary quantization with {args.threshold} threshold...")
        quantized = binary_quantize_cuvs(data, args.threshold)
    else:
        print(f"Using simple NumPy binary quantization with {args.threshold} threshold...")
        quantized = binary_quantize_simple(data, args.threshold)
    
    # Write output
    write_u8bin(args.output_file, quantized)
    
    # Print statistics
    original_size = nrows * ncols * 4  # float32 = 4 bytes
    quantized_size = nrows * quantized_dim
    compression_ratio = original_size / quantized_size
    
    print(f"\nBinary quantization complete!")
    print(f"Original size: {original_size:,} bytes")
    print(f"Quantized size: {quantized_size:,} bytes")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    # Verify output file
    if os.path.exists(args.output_file):
        file_size = os.path.getsize(args.output_file)
        expected_size = 8 + quantized_size  # 8 bytes for header + data
        if file_size == expected_size:
            print(f"✓ Output file size verified: {file_size:,} bytes")
        else:
            print(f"⚠ Warning: Output file size {file_size:,} != expected {expected_size:,}")


if __name__ == '__main__':
    main()



