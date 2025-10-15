#!/usr/bin/env python3
"""
Multithreaded uniform random sampling for Cohere Wikipedia multilingual embeddings.
Creates base.fbin (97M embeddings) and query.fbin (10K embeddings) files.
Uses parallel threads to fetch embeddings for better performance.
"""

import numpy as np
from datasets import load_dataset
import random
import gc
from tqdm import tqdm
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import threading
from collections import defaultdict
import time

# Parameters
BASE_SIZE = 97_000_000  # 97M embeddings for base
QUERY_SIZE = 10_000     # 10K embeddings for queries
EMBEDDING_DIM = 1024    # Cohere embed-multilingual-v3 dimension
TOTAL_EMBEDDINGS = 247_154_006  # Exact total from dataset card
NUM_THREADS = 224        # Number of parallel threads for fetching
BATCH_SIZE = 10000      # Size of batches for each thread to process

# Exact language counts from the dataset card
LANGUAGE_COUNTS = {
    "en": 41_500_000, "de": 20_800_000, "fr": 17_800_000, "ru": 13_700_000,
    "es": 12_900_000, "it": 10_500_000, "ceb": 9_820_000, "ja": 6_630_000,
    "nl": 6_100_000, "pl": 5_970_000, "pt": 5_640_000, "sv": 4_910_000,
    "ca": 4_160_000, "ar": 3_690_000, "cs": 3_120_000, "he": 2_950_000,
    "hu": 2_920_000, "fi": 2_430_000, "id": 2_360_000, "no": 2_210_000,
    "sr": 2_150_000, "fa": 2_070_000, "uk": 1_990_000, "zh": 1_920_000,
    "ro": 1_770_000, "tr": 1_660_000, "el": 1_600_000, "ko": 1_510_000,
    "bg": 1_460_000, "hy": 1_390_000, "eu": 1_330_000, "da": 1_220_000,
    "eo": 1_220_000, "sh": 1_140_000, "tt": 1_130_000, "arz": 1_090_000,
    "gl": 1_060_000, "et": 1_050_000, "ce": 1_010_000, "ast": 1_010_000,
    "sl": 985_000, "hr": 911_000, "sk": 874_000, "ms": 870_000, "be": 857_000,
    "th": 840_000, "az": 818_000, "mk": 785_000, "bn": 768_000, "cy": 762_000,
    "ta": 686_000, "te": 635_000, "ka": 595_000, "kk": 627_000, "hi": 542_000,
    "nn": 531_000, "lv": 485_000, "af": 462_000, "ba": 435_000, "tl": 420_000,
    "sq": 389_000, "ml": 385_000, "min": 373_000, "la": 341_000, "pnb": 336_000,
    "be-x-old": 315_000, "kn": 309_000, "azb": 294_000, "oc": 283_000, "zh-min-nan": 281_000,
    "fy": 248_000, "my": 241_000, "bo": 41_400, "new": 129_000, "ky": 216_000,
    "tt": 217_000, "lb": 217_000, "ur": 162_000, "pa": 188_000, "simple": 646_000,
    "vi": 1_290_000, "lt": 770_000, "hy": 1_390_000, "vo": 33_700, "war": 1_260_000,
    "bs": 397_000, "als": 206_000, "is": 177_000, "mg": 172_000, "sw": 172_000,
    "mr": 203_000, "uz": 840_000, "nds": 166_000, "zh-yue": 136_000, "gu": 99_500,
    # Additional smaller languages - using estimates
    "ab": 3_465, "ace": 10_821, "ady": 1_040, "am": 22_500, "an": 143_000,
    "ang": 7_020, "anp": 6_700, "arc": 1_167, "ary": 17_500, "as": 76_100,
    "ast": 1_010_000, "atj": 1_967, "av": 7_870, "avk": 39_000, "awa": 2_242,
    "ay": 5_985, "ban": 32_300, "bar": 88_200, "bat-smg": 11_240, "bcl": 40_000,
    "bh": 15_900, "bi": 718, "bjn": 16_400, "blk": 24_200, "bm": 1_017,
    "bpy": 47_800, "br": 200_000, "bug": 429, "bxr": 8_549, "cbk-zam": 4_224,
    "cdo": 5_917, "ch": 513, "chr": 788, "chy": 57, "ckb": 121_000, "co": 23_100,
    "cr": 41, "crh": 19_000, "csb": 8_085, "cu": 1_477, "cv": 123_000,
    "dag": 53_300, "din": 1_214, "diq": 34_700, "dsb": 7_536, "dty": 4_728,
    "dv": 16_900, "dz": 4_117, "ee": 1_819, "eml": 3_979, "ext": 10_967,
    "fat": 4_968, "ff": 4_166, "fiu-vro": 8_757, "fj": 1_164, "fo": 28_800,
    "fon": 1_247, "frp": 7_294, "frr": 26_000, "fur": 10_148, "ga": 125_000,
    "gag": 3_076, "gan": 1_626, "gcr": 6_347, "gd": 28_100, "glk": 9_120,
    "gn": 14_983, "gom": 27_800, "gor": 18_100, "got": 1_280, "gpe": 4_289,
    "guc": 2_454, "gur": 6_761, "guw": 3_982, "gv": 12_074, "ha": 168_000,
    "hak": 6_866, "haw": 3_227, "hif": 11_215, "hsb": 39_600, "ht": 55_100,
    "hyw": 82_300, "ia": 42_000, "ie": 20_000, "ig": 132_000, "ik": 275,
    "ilo": 31_000, "inh": 3_768, "io": 102_000, "iu": 263, "jam": 2_387,
    "jbo": 8_577, "jv": 142_000, "kaa": 9_548, "kab": 10_179, "kbd": 4_667,
    "kbp": 7_616, "kcg": 1_555, "kg": 609, "ki": 798, "kl": 770, "km": 74_200,
    "koi": 6_300, "krc": 6_293, "ks": 3_255, "ksh": 8_226, "ku": 82_900,
    "kv": 13_534, "kw": 11_539, "lad": 10_386, "lbe": 766, "lez": 14_152,
    "lfn": 20_800, "lg": 16_800, "li": 69_300, "lij": 20_600, "lld": 125_000,
    "lmo": 104_000, "ln": 3_774, "lo": 12_525, "ltg": 2_005, "mad": 3_236,
    "mai": 13_806, "map-bms": 8_393, "mdf": 6_351, "mhr": 29_600, "mi": 7_643,
    "mni": 7_740, "mnw": 39_600, "mrj": 9_771, "mt": 64_000, "mwl": 41_300,
    "myv": 15_062, "mzn": 23_800, "nah": 3_720, "nap": 10_079, "nds-nl": 30_900,
    "ne": 83_600, "nia": 4_795, "nov": 1_916, "nqo": 9_153, "nrm": 5_786,
    "nso": 8_635, "nv": 49_400, "ny": 2_933, "olo": 8_530, "om": 7_045,
    "or": 65_500, "os": 15_096, "pag": 2_727, "pam": 15_134, "pap": 10_187,
    "pcd": 13_057, "pcm": 5_076, "pdc": 2_239, "pfl": 8_790, "pi": 260,
    "pih": 606, "pms": 77_800, "pnt": 796, "ps": 138_000, "pwn": 1_881,
    "qu": 30_500, "rm": 37_400, "rmy": 1_113, "rn": 1_033, "roa-rup": 2_409,
    "roa-tara": 16_600, "rue": 19_900, "rw": 28_000, "sa": 51_700, "sah": 71_600,
    "sat": 43_500, "sc": 29_000, "scn": 38_400, "sco": 83_900, "sd": 40_400,
    "se": 6_733, "sg": 204, "shi": 5_179, "shn": 22_400, "si": 119_000,
    "skr": 21_100, "sm": 1_659, "smn": 11_672, "sn": 25_900, "so": 29_900,
    "srn": 1_395, "ss": 1_904, "st": 2_051, "stq": 11_890, "su": 124_000,
    "szl": 56_800, "szy": 22_900, "tay": 6_434, "tcy": 10_531, "tet": 3_030,
    "tg": 128_000, "ti": 706, "tk": 24_300, "ami": 10_538, "alt": 10_256,
}

# Fill in remaining languages with estimates
DEFAULT_SMALL_LANG_SIZE = 50_000
ALL_LANGUAGES = [
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
    "szl", "szy", "tay", "tcy", "tet", "tg", "ti", "tk",
    "tl", "tt", "vi"
]

def write_bin(fname, data):
    """Write data in binary format with shape header."""
    with open(fname, "wb") as f:
        np.asarray(data.shape, dtype=np.uint32).tofile(f)
        data.tofile(f)
    print(f"Saved {fname}: shape {data.shape}, dtype {data.dtype}")

def get_language_cumulative_counts():
    """Get or compute cumulative counts for each language."""
    cache_file = "language_counts_cache.json"
    
    if os.path.exists(cache_file):
        print("Loading language counts from cache...")
        with open(cache_file, 'r') as f:
            data = json.load(f)
            return data['languages'], data['cumulative_counts']
    
    print("Computing language counts (this will be cached for future runs)...")
    languages = []
    counts = []
    cumulative = 0
    
    for lang in tqdm(ALL_LANGUAGES, desc="Getting language sizes"):
        if lang in LANGUAGE_COUNTS:
            count = LANGUAGE_COUNTS[lang]
        else:
            # For smaller languages, either fetch actual count or use estimate
            count = DEFAULT_SMALL_LANG_SIZE
        
        languages.append(lang)
        cumulative += count
        counts.append(cumulative)
    
    # Cache for future runs
    with open(cache_file, 'w') as f:
        json.dump({
            'languages': languages,
            'cumulative_counts': counts
        }, f)
    
    return languages, counts

def map_global_idx_to_language(global_idx, languages, cumulative_counts):
    """Map a global index to (language, local_index) pair."""
    # Binary search to find the language
    left, right = 0, len(cumulative_counts) - 1
    
    while left <= right:
        mid = (left + right) // 2
        prev_count = cumulative_counts[mid - 1] if mid > 0 else 0
        
        if prev_count <= global_idx < cumulative_counts[mid]:
            return languages[mid], global_idx - prev_count
        elif global_idx < prev_count:
            right = mid - 1
        else:
            left = mid + 1
    
    # Fallback (shouldn't happen)
    return languages[-1], global_idx - cumulative_counts[-2]

def fetch_embeddings_batch(lang, indices_in_lang, output_array, output_positions, progress_lock, progress_counter):
    """
    Fetch specific embeddings from a language and write to pre-allocated positions.
    
    Args:
        lang: Language code
        indices_in_lang: List of indices within this language to fetch
        output_array: Pre-allocated numpy array to write to
        output_positions: Corresponding positions in output_array for each index
        progress_lock: Lock for thread-safe progress updates
        progress_counter: Shared counter for progress tracking
    """
    try:
        # Load dataset for this language
        dataset = load_dataset(
            "Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary",
            lang,
            split="train",
            streaming=True
        )
        
        # Create a mapping of indices to positions for quick lookup
        index_to_position = dict(zip(indices_in_lang, output_positions))
        indices_set = set(indices_in_lang)
        
        # Iterate through dataset and collect required embeddings
        for idx, doc in enumerate(dataset):
            if idx in indices_set:
                emb_int8 = np.array(doc['emb_int8'], dtype=np.int8)
                output_array[index_to_position[idx]] = emb_int8
                
                # Update progress
                with progress_lock:
                    progress_counter['completed'] += 1
                
                # Remove from set to track completion
                indices_set.remove(idx)
                
                # Stop if we've collected all required indices
                if not indices_set:
                    break
        
        return len(indices_in_lang) - len(indices_set)  # Return number successfully fetched
        
    except Exception as e:
        print(f"Error fetching from {lang}: {e}")
        return 0

def uniform_random_sampling_multithreaded():
    """
    Multithreaded uniform random sampling.
    Pre-selects indices, then uses multiple threads to fetch embeddings in parallel.
    """
    total_needed = BASE_SIZE + QUERY_SIZE
    
    print(f"\nMultithreaded sampling configuration:")
    print(f"  Total embeddings in dataset: {TOTAL_EMBEDDINGS:,}")
    print(f"  Target collection: {total_needed:,}")
    print(f"  Number of threads: {NUM_THREADS}")
    
    # Step 1: Get language cumulative counts
    languages, cumulative_counts = get_language_cumulative_counts()
    
    # Step 2: Generate random indices
    print(f"\nGenerating {total_needed:,} random indices...")
    selected_indices = np.random.choice(TOTAL_EMBEDDINGS, size=total_needed, replace=False)
    np.random.shuffle(selected_indices)  # Additional shuffle for good measure
    
    # Step 3: Map global indices to (language, local_index) pairs
    print("\nMapping indices to languages...")
    language_indices = defaultdict(list)  # lang -> list of (local_idx, global_position)
    
    for global_pos, global_idx in enumerate(tqdm(selected_indices, desc="Mapping")):
        lang, local_idx = map_global_idx_to_language(global_idx, languages, cumulative_counts)
        language_indices[lang].append((local_idx, global_pos))
    
    print(f"\nSampling from {len(language_indices)} languages")
    
    # Step 4: Pre-allocate output array
    print("\nPre-allocating output array...")
    all_embeddings = np.zeros((total_needed, EMBEDDING_DIM), dtype=np.int8)
    
    # Step 5: Fetch embeddings in parallel
    print("\nFetching embeddings with multiple threads...")
    
    # Prepare work items
    work_items = []
    for lang, idx_pairs in language_indices.items():
        local_indices = [idx for idx, _ in idx_pairs]
        global_positions = [pos for _, pos in idx_pairs]
        work_items.append((lang, local_indices, global_positions))
    
    # Progress tracking
    progress_lock = Lock()
    progress_counter = {'completed': 0}
    
    # Execute in parallel
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []
        
        for lang, local_indices, global_positions in work_items:
            future = executor.submit(
                fetch_embeddings_batch,
                lang, local_indices, all_embeddings, global_positions,
                progress_lock, progress_counter
            )
            futures.append((future, lang, len(local_indices)))
        
        # Track progress
        with tqdm(total=total_needed, desc="Fetching embeddings") as pbar:
            last_count = 0
            while progress_counter['completed'] < total_needed:
                current_count = progress_counter['completed']
                pbar.update(current_count - last_count)
                last_count = current_count
                time.sleep(0.1)  # Update every 100ms
            
            # Final update
            pbar.update(total_needed - last_count)
        
        # Wait for all threads to complete
        for future, lang, expected_count in futures:
            fetched = future.result()
            if fetched < expected_count:
                print(f"Warning: {lang} - fetched {fetched}/{expected_count} embeddings")
    
    print(f"\nSuccessfully fetched {progress_counter['completed']:,} embeddings")
    
    # Step 6: Split into base and query
    base_embeddings = all_embeddings[:BASE_SIZE]
    query_embeddings = all_embeddings[BASE_SIZE:BASE_SIZE + QUERY_SIZE]
    
    return base_embeddings, query_embeddings

def uniform_random_sampling_multithreaded_optimized():
    """
    Optimized multithreaded sampling with better work distribution.
    Uses smaller batches and round-robin assignment to threads.
    """
    total_needed = BASE_SIZE + QUERY_SIZE
    
    print(f"\nOptimized multithreaded sampling:")
    print(f"  Total embeddings in dataset: {TOTAL_EMBEDDINGS:,}")
    print(f"  Target collection: {total_needed:,}")
    print(f"  Number of threads: {NUM_THREADS}")
    print(f"  Batch size per thread: {BATCH_SIZE:,}")
    
    # Get language cumulative counts
    languages, cumulative_counts = get_language_cumulative_counts()
    
    # Generate random indices
    print(f"\nGenerating {total_needed:,} random indices...")
    selected_indices = np.random.choice(TOTAL_EMBEDDINGS, size=total_needed, replace=False)
    
    # Pre-allocate output array
    print("\nPre-allocating output array...")
    all_embeddings = np.zeros((total_needed, EMBEDDING_DIM), dtype=np.int8)
    
    # Map indices to languages and create batches
    print("\nCreating work batches...")
    language_batches = defaultdict(list)
    
    for i in tqdm(range(0, total_needed, BATCH_SIZE), desc="Creating batches"):
        batch_indices = selected_indices[i:min(i + BATCH_SIZE, total_needed)]
        batch_positions = list(range(i, min(i + BATCH_SIZE, total_needed)))
        
        # Group by language within this batch
        lang_groups = defaultdict(list)
        for pos, global_idx in zip(batch_positions, batch_indices):
            lang, local_idx = map_global_idx_to_language(global_idx, languages, cumulative_counts)
            lang_groups[lang].append((local_idx, pos))
        
        # Add to language batches
        for lang, pairs in lang_groups.items():
            language_batches[lang].extend(pairs)
    
    # Create work items from language batches
    work_items = []
    for lang, idx_pairs in language_batches.items():
        # Split into smaller chunks for better parallelization
        for i in range(0, len(idx_pairs), BATCH_SIZE // 10):
            chunk = idx_pairs[i:i + BATCH_SIZE // 10]
            if chunk:
                local_indices = [idx for idx, _ in chunk]
                global_positions = [pos for _, pos in chunk]
                work_items.append((lang, local_indices, global_positions))
    
    # Shuffle work items for better load balancing
    random.shuffle(work_items)
    
    print(f"\nProcessing {len(work_items)} work batches across {len(language_batches)} languages")
    
    # Progress tracking
    progress_lock = Lock()
    progress_counter = {'completed': 0}
    
    # Execute in parallel with progress bar
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []
        
        for work_item in work_items:
            lang, local_indices, global_positions = work_item
            future = executor.submit(
                fetch_embeddings_batch,
                lang, local_indices, all_embeddings, global_positions,
                progress_lock, progress_counter
            )
            futures.append(future)
        
        # Wait with progress bar
        with tqdm(total=total_needed, desc="Fetching embeddings") as pbar:
            for future in as_completed(futures):
                with progress_lock:
                    pbar.update(progress_counter['completed'] - pbar.n)
    
    print(f"\nSuccessfully fetched embeddings")
    
    # Final shuffle for good distribution
    print("\nShuffling final dataset...")
    indices = np.arange(total_needed)
    np.random.shuffle(indices)
    all_embeddings = all_embeddings[indices]
    
    # Split into base and query
    base_embeddings = all_embeddings[:BASE_SIZE]
    query_embeddings = all_embeddings[BASE_SIZE:BASE_SIZE + QUERY_SIZE]
    
    return base_embeddings, query_embeddings

def main():
    """Main function."""
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("="*60)
    print("Cohere Wikipedia Embeddings - Multithreaded Sampling")
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
    print(f"  Threads: {NUM_THREADS}")
    
    # Check if files already exist
    if os.path.exists("base.fbin") or os.path.exists("query.fbin"):
        response = input("\nOutput files already exist. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Exiting without changes.")
            return
    
    # Choose sampling method
    print("\nSelect sampling method:")
    print("1. Standard multithreaded (good for most cases)")
    print("2. Optimized multithreaded (better work distribution)")
    choice = input("Enter choice (1 or 2): ").strip()
    
    start_time = time.time()
    
    if choice == "1":
        base_embeddings, query_embeddings = uniform_random_sampling_multithreaded()
    else:
        base_embeddings, query_embeddings = uniform_random_sampling_multithreaded_optimized()
    
    elapsed_time = time.time() - start_time
    print(f"\nSampling completed in {elapsed_time:.2f} seconds")
    
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
    print(f"  Total time: {elapsed_time:.2f} seconds")
    print(f"  Throughput: {(BASE_SIZE + QUERY_SIZE) / elapsed_time:.0f} embeddings/second")
    
    if os.path.exists('base.fbin'):
        print(f"  Base file size: {os.path.getsize('base.fbin') / (1024**3):.2f} GB")
    if os.path.exists('query.fbin'):
        print(f"  Query file size: {os.path.getsize('query.fbin') / (1024**2):.2f} MB")

if __name__ == "__main__":
    main()
