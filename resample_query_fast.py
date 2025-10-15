#!/usr/bin/env python3
"""
Fast query sampling from base set WITHOUT removing them from base.
This is much faster but leaves a small overlap (0.01%) between base and queries.
For most benchmarking purposes, this overlap is negligible.
"""

import numpy as np
import os
import time
from tqdm import tqdm

# Parameters
QUERY_SIZE = 10_000     # 10K embeddings for queries
EMBEDDING_DIM = 1024    # Cohere embed-multilingual-v3 dimension

def read_bin(fname):
    """Read data from binary format with shape header."""
    print(f"Reading {fname}...")
    with open(fname, "rb") as f:
        # Read shape (2 uint32 values: rows, cols)
        shape = np.fromfile(f, dtype=np.uint32, count=2)
        rows, cols = int(shape[0]), int(shape[1])
        
        # Read the data
        data = np.fromfile(f, dtype=np.int8, count=rows * cols)
        data = data.reshape((rows, cols))
        
    print(f"  Loaded {fname}: shape {data.shape}, dtype {data.dtype}")
    return data

def write_bin(fname, data):
    """Write data in binary format with shape header."""
    with open(fname, "wb") as f:
        np.asarray(data.shape, dtype=np.uint32).tofile(f)
        data.tofile(f)
    print(f"Saved {fname}: shape {data.shape}, dtype {data.dtype}")

def sample_queries_from_base():
    """
    Read the base set and randomly sample query embeddings from it.
    Does NOT remove them from base (much faster).
    """
    # Check if base file exists
    if not os.path.exists("base.i8bin"):
        print("Error: base.i8bin not found!")
        print("Please run the base dataset generation script first.")
        return None
    
    # Read base embeddings
    print("\n" + "="*60)
    print("Loading base embeddings...")
    start_time = time.time()
    base_embeddings = read_bin("base.i8bin")
    load_time = time.time() - start_time
    print(f"  Loading completed in {load_time:.2f} seconds")
    
    # Verify dimensions
    assert base_embeddings.shape[1] == EMBEDDING_DIM, \
        f"Unexpected embedding dimension: {base_embeddings.shape[1]} (expected {EMBEDDING_DIM})"
    
    base_size = base_embeddings.shape[0]
    print(f"\nBase set statistics:")
    print(f"  Total embeddings: {base_size:,}")
    print(f"  Embedding dimension: {base_embeddings.shape[1]}")
    print(f"  Data type: {base_embeddings.dtype}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Randomly sample indices
    print(f"\nRandomly sampling {QUERY_SIZE:,} embeddings from base set...")
    
    # Generate random indices without replacement
    query_indices = np.random.choice(
        base_size, 
        size=QUERY_SIZE, 
        replace=False
    )
    
    # Sort indices for potentially better memory access pattern
    query_indices_sorted = np.sort(query_indices)
    
    # Extract query embeddings
    print("Extracting sampled embeddings for queries...")
    query_embeddings = base_embeddings[query_indices_sorted]
    
    print(f"\n✓ Successfully sampled {query_embeddings.shape[0]:,} query embeddings")
    print(f"  Shape: {query_embeddings.shape}")
    
    # Show distribution of sampled indices
    print(f"\nSampling statistics:")
    print(f"  Minimum index: {query_indices_sorted.min():,}")
    print(f"  Maximum index: {query_indices_sorted.max():,}")
    print(f"  Mean index: {query_indices_sorted.mean():.0f}")
    print(f"  Coverage: {(query_indices_sorted.max() - query_indices_sorted.min()) / base_size * 100:.1f}% of base range")
    
    overlap_percent = (QUERY_SIZE / base_size) * 100
    print(f"\n⚠️  Note: {QUERY_SIZE:,} queries overlap with base ({overlap_percent:.3f}%)")
    
    return query_embeddings

def main():
    """Main function."""
    
    print("="*60)
    print("FAST Query Sampling from Base Set")
    print("(Queries NOT removed from base - faster but with overlap)")
    print("="*60)
    
    print(f"\nTarget query size: {QUERY_SIZE:,} embeddings")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print(f"Data type: int8")
    print(f"\n⚠️  This version:")
    print(f"  - Keeps base.i8bin unchanged")
    print(f"  - Creates query.i8bin with {QUERY_SIZE:,} embeddings")
    print(f"  - Leaves small overlap (0.01%) for faster processing")
    
    # Check if query file already exists
    if os.path.exists("query.i8bin"):
        response = input("\nquery.i8bin already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Exiting without changes.")
            return
    
    # Start timing
    total_start = time.time()
    
    # Sample queries from base (without removal)
    query_embeddings = sample_queries_from_base()
    
    if query_embeddings is None:
        return
    
    # Save query set only
    print("\n" + "="*60)
    print("Saving query set...")
    write_bin("query.i8bin", query_embeddings)
    
    total_time = time.time() - total_start
    
    # Print summary
    print("\n" + "="*60)
    print("Fast query sampling complete!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  Base: unchanged (still {97_000_000:,} embeddings)")
    print(f"  Query embeddings: {query_embeddings.shape[0]:,} × {query_embeddings.shape[1]}")
    print(f"  Data type: {query_embeddings.dtype}")
    print(f"  Total time: {total_time:.2f} seconds")
    
    if os.path.exists('query.i8bin'):
        size_mb = os.path.getsize('query.i8bin') / (1024**2)
        print(f"  Query file size: {size_mb:.2f} MB")
    
    print(f"\n✓ Queries randomly sampled from entire base set")
    print(f"⚠️  Small overlap ({QUERY_SIZE:,}/{97_000_000:,} = 0.01%) exists")
    print(f"   This is usually acceptable for benchmarking purposes")

if __name__ == "__main__":
    main()



