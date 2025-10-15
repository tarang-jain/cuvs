#!/usr/bin/env python3
"""
Sequential dataset generation for Cohere Wikipedia multilingual embeddings.
Takes entire languages in order of size until reaching 97M embeddings.
Simple, efficient, and minimizes API calls.
"""

import numpy as np
from datasets import load_dataset
import os
from tqdm import tqdm
import time

# Parameters
BASE_SIZE = 97_000_000  # 97M embeddings for base
QUERY_SIZE = 10_000     # 10K embeddings for queries
EMBEDDING_DIM = 1024    # Cohere embed-multilingual-v3 dimension

# Exact language counts from the dataset card (sorted by size)
LANGUAGE_COUNTS = [
    ("en", 41_500_000),
    ("de", 20_800_000),
    ("fr", 17_800_000),
    ("ru", 13_700_000),
    ("es", 12_900_000),
    ("it", 10_500_000),
    ("ceb", 9_820_000),
    ("ja", 6_630_000),
    ("nl", 6_100_000),
    ("pl", 5_970_000),
    ("pt", 5_640_000),
    ("sv", 4_910_000),
    ("ca", 4_160_000),
    ("ar", 3_690_000),
    ("cs", 3_120_000),
    ("he", 2_950_000),
    ("hu", 2_920_000),
    ("fi", 2_430_000),
    ("id", 2_360_000),
    ("no", 2_210_000),
    ("sr", 2_150_000),
    ("fa", 2_070_000),
    ("uk", 1_990_000),
    ("zh", 1_920_000),
    ("ro", 1_770_000),
    ("tr", 1_660_000),
    ("el", 1_600_000),
    ("ko", 1_510_000),
    ("bg", 1_460_000),
    ("hy", 1_390_000),
    ("eu", 1_330_000),
    ("vi", 1_290_000),
    ("war", 1_260_000),
    ("da", 1_220_000),
    ("eo", 1_220_000),
    ("sh", 1_140_000),
    ("tt", 1_130_000),
    ("arz", 1_090_000),
    ("gl", 1_060_000),
    ("et", 1_050_000),
    ("ce", 1_010_000),
    ("ast", 1_010_000),
    ("sl", 985_000),
    ("hr", 911_000),
    ("sk", 874_000),
    ("ms", 870_000),
    ("be", 857_000),
    ("uz", 840_000),
    ("th", 840_000),
    ("az", 818_000),
    ("mk", 785_000),
    ("lt", 770_000),
    ("bn", 768_000),
    ("cy", 762_000),
    ("ta", 686_000),
    ("simple", 646_000),
    ("te", 635_000),
    ("kk", 627_000),
    ("ka", 595_000),
    ("hi", 542_000),
    ("nn", 531_000),
    ("lv", 485_000),
    ("af", 462_000),
    ("ba", 435_000),
    ("tl", 420_000),
    ("bs", 397_000),
    ("sq", 389_000),
    ("ml", 385_000),
    ("min", 373_000),
    ("la", 341_000),
    ("pnb", 336_000),
    ("be-x-old", 315_000),
    ("kn", 309_000),
    ("azb", 294_000),
    ("oc", 283_000),
    ("zh-min-nan", 281_000),
    ("fy", 248_000),
    ("my", 241_000),
    ("tt", 217_000),
    ("lb", 217_000),
    ("als", 206_000),
    ("mr", 203_000),
    ("br", 200_000),
    ("pa", 188_000),
    ("is", 177_000),
    ("mg", 172_000),
    ("sw", 172_000),
    ("ha", 168_000),
    ("nds", 166_000),
    ("ur", 162_000),
    ("an", 143_000),
    ("jv", 142_000),
    ("ps", 138_000),
    ("zh-yue", 136_000),
    ("ig", 132_000),
    ("tg", 128_000),
    ("new", 129_000),
    ("ga", 125_000),
    ("lld", 125_000),
    ("su", 124_000),
    ("cv", 123_000),
    ("ckb", 121_000),
    ("si", 119_000),
    ("lmo", 104_000),
    ("io", 102_000),
    ("gu", 99_500),
    ("kv", 13_534),  # Continuing with smaller languages as needed
    # Add more if needed to reach exactly 97M
]

def write_bin(fname, data):
    """Write data in binary format with shape header."""
    with open(fname, "wb") as f:
        np.asarray(data.shape, dtype=np.uint32).tofile(f)
        data.tofile(f)
    print(f"Saved {fname}: shape {data.shape}, dtype {data.dtype}")

def collect_sequential_embeddings():
    """
    Collect embeddings sequentially by taking entire languages in order.
    Much simpler and fewer API calls than random sampling.
    """
    total_needed = BASE_SIZE + QUERY_SIZE
    embeddings_collected = []
    total_collected = 0
    languages_used = []
    
    print(f"\nSequential collection strategy:")
    print(f"  Target: {total_needed:,} embeddings")
    print(f"  Method: Take entire languages in descending order of size")
    print()
    
    # Calculate which languages we need
    cumulative = 0
    languages_to_fetch = []
    for lang, count in LANGUAGE_COUNTS:
        if cumulative >= total_needed:
            break
        
        if cumulative + count <= total_needed:
            # Take all embeddings from this language
            languages_to_fetch.append((lang, count, count))
            cumulative += count
        else:
            # Take only what we need from this language
            needed = total_needed - cumulative
            languages_to_fetch.append((lang, count, needed))
            cumulative += needed
            break
    
    # Display plan
    print("Collection plan:")
    for lang, total_in_lang, to_collect in languages_to_fetch:
        if to_collect == total_in_lang:
            print(f"  {lang}: ALL {total_in_lang:,} embeddings")
        else:
            print(f"  {lang}: {to_collect:,} / {total_in_lang:,} embeddings")
    print(f"\nTotal to collect: {cumulative:,} embeddings")
    print()
    
    # Collect embeddings
    for lang, total_in_lang, to_collect in languages_to_fetch:
        print(f"\nProcessing {lang} ({to_collect:,} embeddings)...")
        
        try:
            # Load dataset for this language
            dataset = load_dataset(
                "Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary",
                lang,
                split="train",
                streaming=True
            )
            
            lang_embeddings = []
            lang_collected = 0
            
            # Use tqdm for progress bar
            with tqdm(total=to_collect, desc=f"  {lang}") as pbar:
                for doc in dataset:
                    # Get the pre-computed int8 embedding
                    emb_int8 = np.array(doc['emb_int8'], dtype=np.int8)
                    lang_embeddings.append(emb_int8)
                    lang_collected += 1
                    pbar.update(1)
                    
                    # Stop when we have enough from this language
                    if lang_collected >= to_collect:
                        break
                    
                    # Periodically add to main collection to manage memory
                    if len(lang_embeddings) >= 10000:
                        embeddings_collected.extend(lang_embeddings)
                        lang_embeddings = []
            
            # Add remaining embeddings
            if lang_embeddings:
                embeddings_collected.extend(lang_embeddings)
            
            total_collected += lang_collected
            languages_used.append((lang, lang_collected))
            print(f"  ✓ Collected {lang_collected:,} embeddings from {lang}")
            
            # Small delay to be nice to the API
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  ✗ Error processing {lang}: {e}")
            continue
        
        # Check if we have enough
        if total_collected >= total_needed:
            print(f"\n✓ Successfully collected {total_collected:,} embeddings")
            break
    
    # Convert to numpy array
    print("\nConverting to numpy array...")
    all_embeddings = np.array(embeddings_collected[:total_needed], dtype=np.int8)
    
    # Shuffle to mix languages (optional, but good for training)
    print("Shuffling embeddings...")
    indices = np.arange(len(all_embeddings))
    np.random.shuffle(indices)
    all_embeddings = all_embeddings[indices]
    
    # Split into base and query
    base_embeddings = all_embeddings[:BASE_SIZE]
    query_embeddings = all_embeddings[BASE_SIZE:BASE_SIZE + QUERY_SIZE]
    
    # Print summary of languages used
    print("\n" + "="*60)
    print("Languages used in final dataset:")
    for lang, count in languages_used:
        percentage = (count / total_needed) * 100
        print(f"  {lang}: {count:,} embeddings ({percentage:.1f}%)")
    print("="*60)
    
    return base_embeddings, query_embeddings

def main():
    """Main function."""
    
    # Set random seed for reproducibility (for shuffling)
    np.random.seed(42)
    
    print("="*60)
    print("Cohere Wikipedia Embeddings - Sequential Collection")
    print("Using pre-computed INT8 embeddings")
    print("="*60)
    
    print(f"\nDataset: Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary")
    print(f"\nTarget dataset sizes:")
    print(f"  Base: {BASE_SIZE:,} embeddings")
    print(f"  Query: {QUERY_SIZE:,} embeddings")
    print(f"  Total needed: {BASE_SIZE + QUERY_SIZE:,} embeddings")
    print(f"  Embedding dimension: {EMBEDDING_DIM}")
    print(f"  Data type: int8 (pre-computed by Cohere)")
    
    # Check if files already exist
    if os.path.exists("base.fbin") or os.path.exists("query.fbin"):
        response = input("\nOutput files already exist. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Exiting without changes.")
            return
    
    # Start timing
    start_time = time.time()
    
    # Collect embeddings
    base_embeddings, query_embeddings = collect_sequential_embeddings()
    
    elapsed_time = time.time() - start_time
    print(f"\nCollection completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
    
    # Save to files
    print("\nSaving to disk...")
    write_bin("base.fbin", base_embeddings)
    write_bin("query.fbin", query_embeddings)
    
    # Print summary
    print("\n" + "="*60)
    print("Dataset generation complete!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  Base embeddings: {base_embeddings.shape[0]:,} × {base_embeddings.shape[1]}")
    print(f"  Query embeddings: {query_embeddings.shape[0]:,} × {query_embeddings.shape[1]}")
    print(f"  Data type: {base_embeddings.dtype}")
    print(f"  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"  Throughput: {(BASE_SIZE + QUERY_SIZE) / elapsed_time:.0f} embeddings/second")
    
    if os.path.exists('base.fbin'):
        print(f"  Base file size: {os.path.getsize('base.fbin') / (1024**3):.2f} GB")
    if os.path.exists('query.fbin'):
        print(f"  Query file size: {os.path.getsize('query.fbin') / (1024**2):.2f} MB")

if __name__ == "__main__":
    main()
