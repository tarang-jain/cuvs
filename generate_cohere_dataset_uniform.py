#!/usr/bin/env python3
"""
Uniform random sampling script for Cohere Wikipedia multilingual embeddings.
Creates base.fbin (97M embeddings) and query.fbin (10K embeddings) files.
Samples uniformly from the entire dataset (~247M embeddings) without replacement.
Uses the pre-computed int8 embeddings from Cohere.
"""

import numpy as np
from datasets import load_dataset
import random
import gc
from tqdm import tqdm
import os
import tempfile

# Parameters
BASE_SIZE = 97_000_000  # 97M embeddings for base
QUERY_SIZE = 10_000     # 10K embeddings for queries
EMBEDDING_DIM = 1024    # Cohere embed-multilingual-v3 dimension
TOTAL_EMBEDDINGS = 247_154_006  # Exact total from dataset card

# Available languages in the dataset (from dataset card)
LANGUAGES = [
    "en", "de", "fr", "es", "it", "ja", "ar", "zh", 
    "pt", "ru", "tr", "pl", "nl", "cs", "sv", "uk",
    "ro", "hu", "vi", "el", "he", "da", "fi", "no",
    "id", "ko", "bg", "sk", "fa", "hr", "sr", "lt",
    "sl", "et", "lv", "ca", "th", "sq", "mk", "hy",
    "az", "ka", "kk", "simple", "hi", "bn", "ta", "te",
    "ml", "kn", "mr", "gu", "ur", "pa", "sw", "yo",
    "ab", "ace", "ady", "af", "als", "alt", "am", "ami", 
    "an", "ang", "anp", "arc", "ary", "arz", "as", "ast",
    "atj", "av", "avk", "awa", "ay", "azb", "ba", "ban",
    "bar", "bat-smg", "bcl", "be", "be-x-old", "bh", "bi",
    "bjn", "blk", "bm", "bo", "bpy", "br", "bs", "bug",
    "bxr", "cbk-zam", "cdo", "ce", "ceb", "ch", "chr", "chy",
    "ckb", "co", "cr", "crh", "csb", "cu", "cv", "cy",
    "dag", "din", "diq", "dsb", "dty", "dv", "dz", "ee",
    "eml", "eo", "ext", "fat", "ff", "fiu-vro", "fj", "fo",
    "fon", "frp", "frr", "fur", "fy", "ga", "gag", "gan",
    "gcr", "gd", "gl", "glk", "gn", "gom", "gor", "got",
    "gpe", "guc", "gur", "guw", "gv", "ha", "hak", "haw",
    "hif", "hsb", "ht", "hyw", "ia", "ie", "ig", "ik",
    "ilo", "inh", "io", "is", "iu", "jam", "jbo", "jv",
    "kaa", "kab", "kbd", "kbp", "kcg", "kg", "ki", "kl",
    "km", "koi", "krc", "ks", "ksh", "ku", "kv", "kw",
    "ky", "la", "lad", "lb", "lbe", "lez", "lfn", "lg",
    "li", "lij", "lld", "lmo", "ln", "lo", "ltg", "mad",
    "mai", "map-bms", "mdf", "mg", "mhr", "mi", "min", "mni",
    "mnw", "mrj", "ms", "mt", "mwl", "my", "myv", "mzn",
    "nah", "nap", "nds", "nds-nl", "ne", "new", "nia", "nn",
    "nov", "nqo", "nrm", "nso", "nv", "ny", "oc", "olo",
    "om", "or", "os", "pag", "pam", "pap", "pcd", "pcm",
    "pdc", "pfl", "pi", "pih", "pms", "pnb", "pnt", "ps",
    "pwn", "qu", "rm", "rmy", "rn", "roa-rup", "roa-tara", "rue",
    "rw", "sa", "sah", "sat", "sc", "scn", "sco", "sd",
    "se", "sg", "sh", "shi", "shn", "si", "skr", "sm",
    "smn", "sn", "so", "srn", "ss", "st", "stq", "su",
    "szl", "szy", "tay", "tcy", "tet", "tg", "ti", "tk"
]

def write_bin(fname, data):
    """Write data in binary format with shape header."""
    with open(fname, "wb") as f:
        np.asarray(data.shape, dtype=np.uint32).tofile(f)
        data.tofile(f)
    print(f"Saved {fname}: shape {data.shape}, dtype {data.dtype}")

def write_bin_header(fname, shape):
    """Write just the header for a binary file."""
    with open(fname, "wb") as f:
        np.asarray(shape, dtype=np.uint32).tofile(f)

def append_bin_data(fname, data):
    """Append data to a binary file that already has a header."""
    with open(fname, "ab") as f:
        data.tofile(f)

def uniform_random_sampling_simple():
    """
    Simple uniform random sampling using rejection sampling.
    Each embedding has equal probability of being selected.
    Uses pre-computed int8 embeddings from Cohere.
    """
    total_needed = BASE_SIZE + QUERY_SIZE
    
    # Calculate sampling probability
    sampling_prob = total_needed / TOTAL_EMBEDDINGS
    
    # Add a safety margin to ensure we get enough samples
    sampling_prob = min(1.0, sampling_prob * 1.2)
    
    print(f"\nSampling configuration:")
    print(f"  Total embeddings in dataset: {TOTAL_EMBEDDINGS:,}")
    print(f"  Target collection: {total_needed:,}")
    print(f"  Sampling probability: {sampling_prob:.6f}")
    
    # Temporary storage
    collected_embeddings = []
    collected_count = 0
    
    print("\nCollecting embeddings uniformly from all languages...")
    print("Using pre-computed int8 embeddings from Cohere...")
    
    with tqdm(total=total_needed, desc="Collecting") as pbar:
        for lang_idx, lang in enumerate(LANGUAGES):
            if collected_count >= total_needed:
                break
            
            try:
                # Load the int8 binary dataset directly
                dataset = load_dataset(
                    "Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary",
                    lang,
                    split="train",
                    streaming=True
                )
                
                lang_collected = 0
                
                for doc in dataset:
                    # Uniform random sampling decision
                    if random.random() < sampling_prob:
                        # Use the pre-computed int8 embeddings
                        emb_int8 = np.array(doc['emb_int8'], dtype=np.int8)
                        collected_embeddings.append(emb_int8)
                        collected_count += 1
                        lang_collected += 1
                        pbar.update(1)
                        
                        # Stop if we have enough
                        if collected_count >= total_needed:
                            break
                
                if lang_collected > 0:
                    print(f"\n  {lang}: collected {lang_collected:,} embeddings")
                
            except Exception as e:
                print(f"\n  Error processing {lang}: {e}")
                continue
            
            # Periodically clear memory
            if collected_count % 100000 == 0:
                gc.collect()
    
    print(f"\nTotal collected: {collected_count:,}")
    
    # Convert to numpy array
    all_embeddings = np.array(collected_embeddings, dtype=np.int8)
    
    # Shuffle for good measure
    print("\nShuffling collected embeddings...")
    indices = np.arange(len(all_embeddings))
    np.random.shuffle(indices)
    all_embeddings = all_embeddings[indices]
    
    # Ensure we have exactly the right amount
    if len(all_embeddings) < total_needed:
        print(f"\nWarning: Only collected {len(all_embeddings):,} embeddings.")
        print("Adjusting datasets accordingly...")
        
        # Adjust sizes proportionally
        base_ratio = BASE_SIZE / total_needed
        actual_base_size = int(len(all_embeddings) * base_ratio)
        actual_query_size = len(all_embeddings) - actual_base_size
        
        return all_embeddings[:actual_base_size], all_embeddings[actual_base_size:]
    else:
        # We have enough, use exact sizes
        return all_embeddings[:BASE_SIZE], all_embeddings[BASE_SIZE:BASE_SIZE + QUERY_SIZE]

def uniform_random_sampling_memory_efficient():
    """
    Memory-efficient uniform random sampling using reservoir sampling technique.
    Processes embeddings one at a time without loading all into memory.
    Uses pre-computed int8 embeddings from Cohere.
    """
    total_needed = BASE_SIZE + QUERY_SIZE
    
    print(f"\nMemory-efficient uniform sampling:")
    print(f"  Target: {total_needed:,} embeddings")
    print(f"  Total embeddings in dataset: {TOTAL_EMBEDDINGS:,}")
    print("  Using pre-computed int8 embeddings...")
    
    # Create memory-mapped arrays for reservoir
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
    temp_file.close()
    
    try:
        # Pre-allocate reservoir
        reservoir = np.memmap(
            temp_file.name, 
            dtype=np.int8, 
            mode='w+', 
            shape=(total_needed, EMBEDDING_DIM)
        )
        
        embeddings_seen = 0
        reservoir_filled = 0
        
        print("\nProcessing all languages with reservoir sampling...")
        
        with tqdm(desc="Processing embeddings") as pbar:
            for lang in LANGUAGES:
                try:
                    # Load the int8 binary dataset directly
                    dataset = load_dataset(
                        "Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary",
                        lang,
                        split="train",
                        streaming=True
                    )
                    
                    lang_processed = 0
                    
                    for doc in dataset:
                        embeddings_seen += 1
                        
                        # Reservoir sampling algorithm
                        if reservoir_filled < total_needed:
                            # Fill the reservoir with pre-computed int8 embeddings
                            emb_int8 = np.array(doc['emb_int8'], dtype=np.int8)
                            reservoir[reservoir_filled] = emb_int8
                            reservoir_filled += 1
                        else:
                            # Randomly replace elements in reservoir
                            # Probability of selection = reservoir_size / embeddings_seen
                            replace_idx = random.randint(0, embeddings_seen - 1)
                            if replace_idx < total_needed:
                                emb_int8 = np.array(doc['emb_int8'], dtype=np.int8)
                                reservoir[replace_idx] = emb_int8
                        
                        lang_processed += 1
                        pbar.update(1)
                        
                        # Flush periodically
                        if embeddings_seen % 100000 == 0:
                            reservoir.flush()
                            pbar.set_description(f"Processed {embeddings_seen:,} embeddings")
                            gc.collect()
                    
                    if lang_processed > 0:
                        print(f"\n  {lang}: processed {lang_processed:,} embeddings")
                    
                except Exception as e:
                    print(f"\n  Error processing {lang}: {e}")
                    continue
        
        print(f"\nTotal embeddings seen: {embeddings_seen:,}")
        print(f"Reservoir filled: {reservoir_filled:,}")
        
        # Shuffle the reservoir
        print("\nShuffling reservoir...")
        indices = np.arange(min(reservoir_filled, total_needed))
        np.random.shuffle(indices)
        
        # Split into base and query
        base_indices = indices[:BASE_SIZE]
        query_indices = indices[BASE_SIZE:BASE_SIZE + QUERY_SIZE]
        
        # Create output arrays
        base_embeddings = np.array(reservoir[base_indices])
        query_embeddings = np.array(reservoir[query_indices])
        
        # Clean up
        del reservoir
        
        return base_embeddings, query_embeddings
        
    finally:
        # Clean up temp file
        os.unlink(temp_file.name)

def main():
    """Main function."""
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("="*60)
    print("Cohere Wikipedia Embeddings - Uniform Random Sampling")
    print("Using pre-computed INT8 embeddings")
    print("="*60)
    
    print(f"\nDataset: Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary")
    print(f"Total embeddings available: {TOTAL_EMBEDDINGS:,}")
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
    
    # Choose sampling method
    print("\nSelect sampling method:")
    print("1. Simple (loads into memory, faster but needs ~100GB RAM)")
    print("2. Memory-efficient (reservoir sampling, slower but needs ~10GB RAM)")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        base_embeddings, query_embeddings = uniform_random_sampling_simple()
    else:
        base_embeddings, query_embeddings = uniform_random_sampling_memory_efficient()
    
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
    
    if os.path.exists('base.fbin'):
        print(f"  Base file size: {os.path.getsize('base.fbin') / (1024**3):.2f} GB")
    if os.path.exists('query.fbin'):
        print(f"  Query file size: {os.path.getsize('query.fbin') / (1024**2):.2f} MB")

if __name__ == "__main__":
    main()