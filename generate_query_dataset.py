#!/usr/bin/env python3
"""
Generate query dataset for Cohere Wikipedia multilingual embeddings.
Takes 10,000 Spanish embeddings that are disjoint from the base set.
The base set used the first ~3.2M Spanish embeddings, so we skip those.
"""

import numpy as np
from datasets import load_dataset
import os
from tqdm import tqdm
import time

# Parameters
QUERY_SIZE = 10_000     # 10K embeddings for queries
EMBEDDING_DIM = 1024    # Cohere embed-multilingual-v3 dimension

# Spanish was partially used in the base set
# Calculation: 97M total - (en:41.5M + de:20.8M + fr:17.8M + ru:13.7M) = 3.2M from Spanish
SPANISH_USED_IN_BASE = 3_200_000  # First 3.2M Spanish embeddings were used in base set

def write_bin(fname, data):
    """Write data in binary format with shape header."""
    with open(fname, "wb") as f:
        np.asarray(data.shape, dtype=np.uint32).tofile(f)
        data.tofile(f)
    print(f"Saved {fname}: shape {data.shape}, dtype {data.dtype}")

def generate_query_embeddings():
    """
    Generate query embeddings from Spanish training split.
    Skip the first 3.2M embeddings that were used in the base set.
    """
    print(f"\nGenerating query embeddings from Spanish training split")
    print(f"  Language: Spanish (es)")
    print(f"  Skipping first {SPANISH_USED_IN_BASE:,} embeddings (used in base set)")
    print(f"  Target: {QUERY_SIZE:,} query embeddings")
    print()
    
    # Load the Spanish training split
    print("Loading Spanish training split...")
    try:
        dataset = load_dataset(
            "Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary",
            "es",
            split="train",
            streaming=True
        )
        
        query_embeddings = []
        collected = 0
        skipped = 0
        
        # First skip the embeddings that were used in the base set
        print(f"Skipping {SPANISH_USED_IN_BASE:,} embeddings...")
        skip_pbar = tqdm(total=SPANISH_USED_IN_BASE, desc="Skipping")
        for doc in dataset:
            skipped += 1
            skip_pbar.update(1)
            if skipped >= SPANISH_USED_IN_BASE:
                break
        skip_pbar.close()
        
        # Now collect the query embeddings
        print(f"\nCollecting {QUERY_SIZE:,} query embeddings...")
        with tqdm(total=QUERY_SIZE, desc="Collecting queries") as pbar:
            for doc in dataset:
                # Get the pre-computed int8 embedding
                emb_int8 = np.array(doc['emb_int8'], dtype=np.int8)
                query_embeddings.append(emb_int8)
                collected += 1
                pbar.update(1)
                
                # Stop when we have enough
                if collected >= QUERY_SIZE:
                    break
        
        print(f"\n✓ Successfully collected {collected:,} query embeddings")
        print(f"  (After skipping {skipped:,} embeddings used in base set)")
        
        # Convert to numpy array
        print("Converting to numpy array...")
        query_embeddings = np.array(query_embeddings, dtype=np.int8)
        
        # Verify shape
        assert query_embeddings.shape == (QUERY_SIZE, EMBEDDING_DIM), \
            f"Unexpected shape: {query_embeddings.shape}"
        
        return query_embeddings
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        raise

def main():
    """Main function."""
    
    print("="*60)
    print("Cohere Wikipedia Query Embeddings Generator")
    print("Using pre-computed INT8 embeddings from Spanish training split")
    print("="*60)
    
    print(f"\nDataset: Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary")
    print(f"Language: Spanish (es)")
    print(f"Split: train (skipping first {SPANISH_USED_IN_BASE:,} used in base set)")
    print(f"Target size: {QUERY_SIZE:,} embeddings")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print(f"Data type: int8 (pre-computed by Cohere)")
    
    # Check if file already exists
    if os.path.exists("query.fbin"):
        response = input("\nquery.fbin already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Exiting without changes.")
            return
    
    # Start timing
    start_time = time.time()
    
    # Generate query embeddings
    query_embeddings = generate_query_embeddings()
    
    elapsed_time = time.time() - start_time
    print(f"\nCollection completed in {elapsed_time:.2f} seconds")
    
    # Save to file
    print("\nSaving to disk...")
    write_bin("query.fbin", query_embeddings)
    
    # Print summary
    print("\n" + "="*60)
    print("Query dataset generation complete!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  Query embeddings: {query_embeddings.shape[0]:,} × {query_embeddings.shape[1]}")
    print(f"  Data type: {query_embeddings.dtype}")
    print(f"  Total time: {elapsed_time:.2f} seconds")
    print(f"  Throughput: {QUERY_SIZE / elapsed_time:.0f} embeddings/second")
    
    if os.path.exists('query.fbin'):
        size_mb = os.path.getsize('query.fbin') / (1024**2)
        print(f"  Query file size: {size_mb:.2f} MB")
        print(f"\nOutput file: query.fbin")

if __name__ == "__main__":
    main()
