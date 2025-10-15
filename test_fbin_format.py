#!/usr/bin/env python3
"""
Test script to verify the fbin file format is correct.
Creates small test files and reads them back to verify.
"""

import numpy as np
import os

def write_bin(fname, data):
    """Write data in binary format with shape header."""
    with open(fname, "wb") as f:
        np.asarray(data.shape, dtype=np.uint32).tofile(f)
        data.tofile(f)

def read_bin(fname, dtype=np.float32):
    """Read data from binary format with shape header."""
    with open(fname, "rb") as f:
        # Read shape (2 uint32 values)
        shape = np.fromfile(f, dtype=np.uint32, count=2)
        # Read data
        data = np.fromfile(f, dtype=dtype)
        data = data.reshape(shape)
    return data

def test_format():
    """Test the fbin format with small arrays."""
    print("Testing fbin format...")
    
    # Create test data
    test_base = np.random.randint(-127, 128, (1000, 1024), dtype=np.int8)
    test_query = np.random.randint(-127, 128, (10, 1024), dtype=np.int8)
    
    # Write test files
    write_bin("test_base.fbin", test_base)
    write_bin("test_query.fbin", test_query)
    
    # Read back and verify
    read_base = read_bin("test_base.fbin", dtype=np.int8)
    read_query = read_bin("test_query.fbin", dtype=np.int8)
    
    # Check shapes and values
    assert read_base.shape == test_base.shape, f"Base shape mismatch: {read_base.shape} != {test_base.shape}"
    assert read_query.shape == test_query.shape, f"Query shape mismatch: {read_query.shape} != {test_query.shape}"
    assert np.array_equal(read_base, test_base), "Base data mismatch"
    assert np.array_equal(read_query, test_query), "Query data mismatch"
    
    print(f"✓ Test base shape: {read_base.shape}, dtype: {read_base.dtype}")
    print(f"✓ Test query shape: {read_query.shape}, dtype: {read_query.dtype}")
    print("✓ Format verification successful!")
    
    # Clean up test files
    os.remove("test_base.fbin")
    os.remove("test_query.fbin")
    
    # If real files exist, check their format
    if os.path.exists("base.fbin"):
        print("\nChecking actual base.fbin...")
        with open("base.fbin", "rb") as f:
            shape = np.fromfile(f, dtype=np.uint32, count=2)
            print(f"  Shape from header: {shape}")
            print(f"  Expected size: {shape[0] * shape[1]} bytes")
            
            # Check file size matches expected
            f.seek(0, 2)  # Go to end
            file_size = f.tell()
            expected_size = 8 + shape[0] * shape[1]  # 8 bytes for header + data
            print(f"  Actual file size: {file_size} bytes")
            if file_size == expected_size:
                print("  ✓ File size matches expected")
            else:
                print(f"  ⚠ File size mismatch: expected {expected_size}, got {file_size}")
    
    if os.path.exists("query.fbin"):
        print("\nChecking actual query.fbin...")
        with open("query.fbin", "rb") as f:
            shape = np.fromfile(f, dtype=np.uint32, count=2)
            print(f"  Shape from header: {shape}")
            print(f"  Expected size: {shape[0] * shape[1]} bytes")
            
            # Check file size matches expected
            f.seek(0, 2)  # Go to end
            file_size = f.tell()
            expected_size = 8 + shape[0] * shape[1]  # 8 bytes for header + data
            print(f"  Actual file size: {file_size} bytes")
            if file_size == expected_size:
                print("  ✓ File size matches expected")
            else:
                print(f"  ⚠ File size mismatch: expected {expected_size}, got {file_size}")

if __name__ == "__main__":
    test_format()




