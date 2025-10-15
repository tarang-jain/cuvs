#!/usr/bin/env python3
"""
Resample query dataset by randomly selecting embeddings from the base set.
This creates a query set that is representative of all languages in the base set,
and removes those queries from the base to ensure no overlap.
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

def resample_queries_from_base():
    """
    Read the base set, randomly sample query embeddings from it,
    and remove those queries from the base to ensure no overlap.
    """
    # Check if base file exists
    if not os.path.exists("base.i8bin"):
        print("Error: base.i8bin not found!")
        print("Please run the base dataset generation script first.")
        return None, None
    
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
    print(f"\nOriginal base set statistics:")
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
    
    # Use numpy's optimized delete function for efficient removal
    print("Removing sampled embeddings from base set (using np.delete)...")
    
    # np.delete is optimized for removing elements at specified indices
    # It handles the memory copying efficiently in C
    base_embeddings_updated = np.delete(base_embeddings, query_indices, axis=0)
    
    print(f"\n✓ Successfully sampled {query_embeddings.shape[0]:,} query embeddings")
    print(f"  Query shape: {query_embeddings.shape}")
    print(f"  Updated base shape: {base_embeddings_updated.shape}")
    
    # Show distribution of sampled indices
    print(f"\nSampling statistics:")
    print(f"  Minimum index: {query_indices_sorted.min():,}")
    print(f"  Maximum index: {query_indices_sorted.max():,}")
    print(f"  Mean index: {query_indices_sorted.mean():.0f}")
    print(f"  Coverage: {(query_indices_sorted.max() - query_indices_sorted.min()) / base_size * 100:.1f}% of original base range")
    
    return base_embeddings_updated, query_embeddings

def main():
    """Main function."""
    
    print("="*60)
    print("Query Resampling from Base Set")
    print("Randomly selecting queries and removing them from base")
    print("="*60)
    
    print(f"\nTarget query size: {QUERY_SIZE:,} embeddings")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print(f"Data type: int8")
    print(f"\nThis will create:")
    print(f"  - base.i8bin with 96,990,000 embeddings (queries removed)")
    print(f"  - query.i8bin with 10,000 embeddings (no overlap with base)")
    
    # Check if files already exist
    files_exist = []
    if os.path.exists("base.i8bin"):
        files_exist.append("base.i8bin")
    if os.path.exists("query.i8bin"):
        files_exist.append("query.i8bin")
    
    if files_exist:
        print(f"\nWarning: The following files will be overwritten:")
        for f in files_exist:
            print(f"  - {f}")
        response = input("\nContinue and overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Exiting without changes.")
            return
    
    # Start timing
    total_start = time.time()
    
    # Resample queries from base
    base_embeddings_updated, query_embeddings = resample_queries_from_base()
    
    if base_embeddings_updated is None or query_embeddings is None:
        return
    
    # Save both files
    print("\n" + "="*60)
    print("Saving updated datasets...")
    
    # Save updated base (with queries removed)
    print("\nSaving updated base set (queries removed)...")
    write_bin("base.i8bin", base_embeddings_updated)
    
    # Save query set
    print("\nSaving query set...")
    write_bin("query.i8bin", query_embeddings)
    
    total_time = time.time() - total_start
    
    # Print summary
    print("\n" + "="*60)
    print("Dataset resampling complete!")
    print("="*60)
    print(f"\nFinal dataset sizes:")
    print(f"  Base embeddings: {base_embeddings_updated.shape[0]:,} × {base_embeddings_updated.shape[1]}")
    print(f"  Query embeddings: {query_embeddings.shape[0]:,} × {query_embeddings.shape[1]}")
    print(f"  Data type: {base_embeddings_updated.dtype}")
    print(f"  Total time: {total_time:.2f} seconds")
    
    if os.path.exists('base.i8bin'):
        size_gb = os.path.getsize('base.i8bin') / (1024**3)
        print(f"\nFile sizes:")
        print(f"  Base file: {size_gb:.2f} GB")
    if os.path.exists('query.i8bin'):
        size_mb = os.path.getsize('query.i8bin') / (1024**2)
        print(f"  Query file: {size_mb:.2f} MB")
    
    print("\n✓ Queries are randomly sampled from the entire base set")
    print("✓ No overlap between base and query sets (queries removed from base)")
    print("✓ Better benchmarking with disjoint sets")

if __name__ == "__main__":
    main()



