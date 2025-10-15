#!/usr/bin/env python3
"""
Memory-efficient script to download and process Cohere Wikipedia multilingual embeddings.
Creates base.fbin (97M embeddings) and query.fbin (10K embeddings) files.
Processes data in chunks to minimize memory usage.
"""

import numpy as np
from datasets import load_dataset
import random
import gc
from tqdm import tqdm
import os
import tempfile
import mmap

# Parameters
BASE_SIZE = 97_000_000  # 97M embeddings for base
QUERY_SIZE = 10_000     # 10K embeddings for queries
CHUNK_SIZE = 10_000     # Process in smaller chunks for memory efficiency
EMBEDDING_DIM = 1024    # Cohere embed-multilingual-v3 dimension

# Available languages in the dataset (sorted by expected size)
LANGUAGES = [
    "en", "de", "fr", "es", "it", "ja", "ar", "zh", 
    "pt", "ru", "tr", "pl", "nl", "cs", "sv", "uk",
    "ro", "hu", "vi", "el", "he", "da", "fi", "no",
    "id", "ko", "bg", "sk", "fa", "hr", "sr", "lt",
    "sl", "et", "lv", "ca", "th", "sq", "mk", "hy",
    "az", "ka", "kk", "simple", "hi", "bn", "ta", "te",
    "ml", "kn", "mr", "gu", "ur", "pa", "sw", "yo"
]

def write_bin_header(fname, shape):
    """Write just the header for a binary file."""
    with open(fname, "wb") as f:
        np.asarray(shape, dtype=np.uint32).tofile(f)

def append_bin_data(fname, data):
    """Append data to a binary file that already has a header."""
    with open(fname, "ab") as f:
        data.tofile(f)

def write_bin(fname, data):
    """Write data in binary format with shape header."""
    with open(fname, "wb") as f:
        np.asarray(data.shape, dtype=np.uint32).tofile(f)
        data.tofile(f)
    print(f"Saved {fname}: shape {data.shape}, dtype {data.dtype}")

# No longer needed - we use pre-computed int8 embeddings from Cohere
# def float32_to_int8(embeddings):
#     """Convert float32 embeddings to int8 with proper scaling."""
#     pass

def estimate_docs_per_language():
    """Estimate number of documents per language for planning."""
    # These are rough estimates based on Wikipedia sizes
    # You can adjust based on actual dataset sizes
    estimates = {
        "en": 6_000_000,  # English Wikipedia is largest
        "de": 2_500_000,
        "fr": 2_300_000,
        "es": 1_700_000,
        "it": 1_700_000,
        "ja": 1_300_000,
        "zh": 1_200_000,
        "pt": 1_100_000,
        "ru": 1_900_000,
        "ar": 1_100_000,
    }
    # Default for other languages
    default_size = 500_000
    
    total_estimated = sum(estimates.values()) + default_size * (len(LANGUAGES) - len(estimates))
    return estimates, default_size, total_estimated

def collect_embeddings_memory_efficient():
    """
    Collect embeddings with minimal memory usage by processing incrementally.
    """
    estimates, default_size, total_estimated = estimate_docs_per_language()
    
    # Calculate sampling probability based on estimates
    total_needed = BASE_SIZE + QUERY_SIZE
    sampling_prob = min(1.0, total_needed / total_estimated * 1.5)  # 1.5x for safety margin
    
    print(f"\nEstimated total documents across all languages: {total_estimated:,}")
    print(f"Target collection: {total_needed:,} embeddings")
    print(f"Initial sampling probability: {sampling_prob:.4f}")
    
    # Create temporary files for base and query
    temp_base = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
    temp_query = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
    temp_base.close()
    temp_query.close()
    
    try:
        # Initialize temporary storage
        base_collected = 0
        query_collected = 0
        
        # Pre-allocate memory-mapped arrays
        base_mmap = np.memmap(temp_base.name, dtype=np.int8, mode='w+', shape=(BASE_SIZE, EMBEDDING_DIM))
        query_mmap = np.memmap(temp_query.name, dtype=np.int8, mode='w+', shape=(QUERY_SIZE, EMBEDDING_DIM))
        
        print("\nCollecting embeddings from all languages...")
        
        with tqdm(total=total_needed, desc="Total progress") as pbar:
            for lang in LANGUAGES:
                if base_collected >= BASE_SIZE and query_collected >= QUERY_SIZE:
                    break
                
                try:
                    # Load the int8 binary dataset directly
                    dataset = load_dataset(
                        "Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary",
                        lang,
                        split="train",
                        streaming=True
                    )
                    
                    # Get estimated size for this language
                    lang_estimate = estimates.get(lang, default_size)
                    
                    # Calculate how many to sample from this language
                    remaining_needed = (BASE_SIZE - base_collected) + (QUERY_SIZE - query_collected)
                    lang_sampling_prob = min(1.0, remaining_needed / lang_estimate * 1.5)
                    
                    print(f"\nProcessing {lang} (sampling ~{lang_sampling_prob:.2%} of documents)")
                    
                    chunk_buffer = []
                    
                    for doc_idx, doc in enumerate(dataset):
                        # Random sampling decision
                        if random.random() > lang_sampling_prob:
                            continue
                        
                        # Get pre-computed int8 embedding
                        emb_int8 = np.array(doc['emb_int8'], dtype=np.int8)
                        
                        # Decide if this goes to base or query
                        if base_collected < BASE_SIZE:
                            base_mmap[base_collected] = emb_int8
                            base_collected += 1
                            pbar.update(1)
                        elif query_collected < QUERY_SIZE:
                            query_mmap[query_collected] = emb_int8
                            query_collected += 1
                            pbar.update(1)
                        else:
                            break
                        
                        # Flush periodically to disk
                        if (base_collected + query_collected) % 100000 == 0:
                            base_mmap.flush()
                            query_mmap.flush()
                            gc.collect()
                        
                        # Stop if we have enough
                        if base_collected >= BASE_SIZE and query_collected >= QUERY_SIZE:
                            break
                    
                    print(f"  Collected from {lang}: base={base_collected}, query={query_collected}")
                    
                except Exception as e:
                    print(f"  Error processing {lang}: {e}")
                    continue
        
        print(f"\nFinal collection: base={base_collected}, query={query_collected}")
        
        # Ensure we have exactly the right amount
        if base_collected < BASE_SIZE or query_collected < QUERY_SIZE:
            print("\nWarning: Could not collect enough embeddings. Filling with random data...")
            
            # Fill remaining with random int8 values (as fallback)
            if base_collected < BASE_SIZE:
                remaining = BASE_SIZE - base_collected
                random_emb = np.random.randint(-127, 128, (remaining, EMBEDDING_DIM), dtype=np.int8)
                base_mmap[base_collected:BASE_SIZE] = random_emb
            
            if query_collected < QUERY_SIZE:
                remaining = QUERY_SIZE - query_collected
                random_emb = np.random.randint(-127, 128, (remaining, EMBEDDING_DIM), dtype=np.int8)
                query_mmap[query_collected:QUERY_SIZE] = random_emb
        
        # Flush final data
        base_mmap.flush()
        query_mmap.flush()
        
        print("\nShuffling datasets for better distribution...")
        
        # Shuffle base dataset (in chunks to manage memory)
        print("  Shuffling base dataset...")
        indices = np.arange(BASE_SIZE)
        np.random.shuffle(indices)
        
        # Create output files with shuffled data
        print("\nWriting final output files...")
        
        # Write base.fbin
        write_bin_header("base.fbin", (BASE_SIZE, EMBEDDING_DIM))
        
        # Write in chunks to manage memory
        chunk_size = 100000
        for i in tqdm(range(0, BASE_SIZE, chunk_size), desc="Writing base.fbin"):
            end = min(i + chunk_size, BASE_SIZE)
            chunk_indices = indices[i:end]
            chunk_data = base_mmap[chunk_indices]
            append_bin_data("base.fbin", chunk_data)
        
        print(f"Saved base.fbin: shape ({BASE_SIZE}, {EMBEDDING_DIM}), dtype int8")
        
        # Shuffle and write query dataset
        print("  Shuffling query dataset...")
        query_indices = np.arange(QUERY_SIZE)
        np.random.shuffle(query_indices)
        query_data = query_mmap[query_indices]
        write_bin("query.fbin", query_data)
        
        # Clean up memory maps
        del base_mmap
        del query_mmap
        
    finally:
        # Clean up temporary files
        os.unlink(temp_base.name)
        os.unlink(temp_query.name)
    
    # Print summary
    print("\n" + "="*60)
    print("Dataset generation complete!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  Base embeddings: {BASE_SIZE:,} × {EMBEDDING_DIM}")
    print(f"  Query embeddings: {QUERY_SIZE:,} × {EMBEDDING_DIM}")
    print(f"  Data type: int8")
    
    if os.path.exists('base.fbin'):
        print(f"  Base file size: {os.path.getsize('base.fbin') / (1024**3):.2f} GB")
    if os.path.exists('query.fbin'):
        print(f"  Query file size: {os.path.getsize('query.fbin') / (1024**2):.2f} MB")

def main():
    """Main function."""
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("="*60)
    print("Cohere Wikipedia Multilingual Embeddings Dataset Generator")
    print("(Memory-Efficient Version)")
    print("="*60)
    
    print(f"\nTarget dataset sizes:")
    print(f"  Base: {BASE_SIZE:,} embeddings")
    print(f"  Query: {QUERY_SIZE:,} embeddings")
    print(f"  Embedding dimension: {EMBEDDING_DIM}")
    print(f"  Data type: int8")
    
    # Check if files already exist
    if os.path.exists("base.fbin") or os.path.exists("query.fbin"):
        response = input("\nOutput files already exist. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Exiting without changes.")
            return
    
    # Run the memory-efficient collection
    collect_embeddings_memory_efficient()

if __name__ == "__main__":
    main()
