#!/usr/bin/env python3
"""
Script to download and process Cohere Wikipedia multilingual embeddings.
Creates base.fbin (97M embeddings) and query.fbin (10K embeddings) files.
"""

import numpy as np
from datasets import load_dataset
import random
import gc
from tqdm import tqdm
import os

# Parameters
BASE_SIZE = 97_000_000  # 97M embeddings for base
QUERY_SIZE = 10_000     # 10K embeddings for queries
BATCH_SIZE = 100_000    # Process in batches to manage memory
EMBEDDING_DIM = 1024    # Cohere embed-multilingual-v3 dimension

# Available languages in the dataset
LANGUAGES = [
    "en", "de", "fr", "es", "it", "ja", "ar", "zh", 
    "pt", "ru", "tr", "pl", "nl", "cs", "sv", "uk",
    "ro", "hu", "vi", "el", "he", "da", "fi", "no",
    "id", "ko", "bg", "sk", "fa", "hr", "sr", "lt",
    "sl", "et", "lv", "ca", "th", "sq", "mk", "hy",
    "az", "ka", "kk", "simple", "hi", "bn", "ta", "te",
    "ml", "kn", "mr", "gu", "ur", "pa", "sw", "yo"
]

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

def count_total_embeddings():
    """Count total available embeddings across all languages."""
    total = 0
    print("Counting available embeddings across languages...")
    for lang in tqdm(LANGUAGES, desc="Languages"):
        try:
            # Load dataset info without downloading data
            dataset = load_dataset(
                f"Cohere/wikipedia-2023-11-embed-multilingual-v3",
                lang,
                split="train",
                streaming=True
            )
            # Get approximate count (this is fast for streaming datasets)
            # For exact count, we'd need to iterate through all
            # Using a reasonable estimate based on dataset info
            total += 1_000_000  # Approximate per language
        except Exception as e:
            print(f"Warning: Could not load language {lang}: {e}")
    return total

def collect_embeddings_uniform(total_needed, languages):
    """
    Collect embeddings uniformly across all available languages.
    Returns array of shape (total_needed, 1024) in int8 format.
    """
    embeddings_collected = []
    indices_collected = []
    
    # Calculate how many embeddings to take per language (approximately)
    embeddings_per_lang = total_needed // len(languages)
    extra_needed = total_needed % len(languages)
    
    print(f"\nCollecting {total_needed} embeddings uniformly across {len(languages)} languages")
    print(f"Approximately {embeddings_per_lang} embeddings per language")
    
    total_collected = 0
    
    for lang_idx, lang in enumerate(tqdm(languages, desc="Processing languages")):
        try:
            # Determine how many to collect from this language
            lang_quota = embeddings_per_lang
            if lang_idx < extra_needed:
                lang_quota += 1
            
            if total_collected >= total_needed:
                break
                
            # Load the int8 binary dataset directly
            dataset = load_dataset(
                "Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary",
                lang,
                split="train",
                streaming=True
            )
            
            lang_embeddings = []
            lang_collected = 0
            
            # Collect embeddings from this language
            for idx, doc in enumerate(dataset):
                if lang_collected >= lang_quota:
                    break
                    
                # Random sampling: skip some documents randomly
                if random.random() > 0.5 and lang_collected < lang_quota - 1000:
                    continue
                    
                # Use pre-computed int8 embeddings
                emb_int8 = np.array(doc['emb_int8'], dtype=np.int8)
                lang_embeddings.append(emb_int8)
                indices_collected.append((lang, idx))
                lang_collected += 1
                
                # Process in batches to manage memory
                if len(lang_embeddings) >= BATCH_SIZE:
                    batch = np.array(lang_embeddings, dtype=np.int8)
                    embeddings_collected.append(batch)
                    lang_embeddings = []
                    gc.collect()
            
            # Process remaining embeddings
            if lang_embeddings:
                batch = np.array(lang_embeddings, dtype=np.int8)
                embeddings_collected.append(batch)
            
            total_collected += lang_collected
            print(f"  {lang}: collected {lang_collected} embeddings")
            
            gc.collect()
            
        except Exception as e:
            print(f"  Warning: Error processing language {lang}: {e}")
            continue
    
    # Concatenate all collected embeddings
    print(f"\nTotal embeddings collected: {total_collected}")
    
    if embeddings_collected:
        all_embeddings = np.vstack(embeddings_collected)
        
        # Shuffle to ensure good mixing
        print("Shuffling embeddings...")
        indices = np.arange(len(all_embeddings))
        np.random.shuffle(indices)
        all_embeddings = all_embeddings[indices]
        
        return all_embeddings[:total_needed], indices_collected
    else:
        raise ValueError("No embeddings were collected")

def main():
    """Main function to generate base and query datasets."""
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("="*60)
    print("Cohere Wikipedia Multilingual Embeddings Dataset Generator")
    print("="*60)
    
    # Collect total embeddings needed (base + query)
    total_needed = BASE_SIZE + QUERY_SIZE
    
    print(f"\nTarget dataset sizes:")
    print(f"  Base: {BASE_SIZE:,} embeddings")
    print(f"  Query: {QUERY_SIZE:,} embeddings")
    print(f"  Total: {total_needed:,} embeddings")
    
    # Check if files already exist
    if os.path.exists("base.fbin") or os.path.exists("query.fbin"):
        response = input("\nOutput files already exist. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Exiting without changes.")
            return
    
    # Collect embeddings
    print(f"\nUsing {len(LANGUAGES)} languages for uniform sampling")
    all_embeddings, indices = collect_embeddings_uniform(total_needed, LANGUAGES)
    
    # Split into base and query sets
    print("\nSplitting into base and query sets...")
    base_embeddings = all_embeddings[:BASE_SIZE]
    query_embeddings = all_embeddings[BASE_SIZE:BASE_SIZE + QUERY_SIZE]
    
    # Verify shapes
    print(f"\nFinal shapes:")
    print(f"  Base: {base_embeddings.shape}")
    print(f"  Query: {query_embeddings.shape}")
    
    # Save to files
    print("\nSaving to disk...")
    write_bin("base.fbin", base_embeddings)
    write_bin("query.fbin", query_embeddings)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Dataset generation complete!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  Base embeddings: {base_embeddings.shape[0]:,} × {base_embeddings.shape[1]}")
    print(f"  Query embeddings: {query_embeddings.shape[0]:,} × {query_embeddings.shape[1]}")
    print(f"  Data type: {base_embeddings.dtype}")
    print(f"  Base file size: {os.path.getsize('base.fbin') / (1024**3):.2f} GB")
    print(f"  Query file size: {os.path.getsize('query.fbin') / (1024**2):.2f} MB")

if __name__ == "__main__":
    main()
