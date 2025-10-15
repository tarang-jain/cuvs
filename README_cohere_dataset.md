# Cohere Wikipedia Multilingual Embeddings Dataset Generator

This repository contains scripts to generate large-scale embedding datasets from the Cohere Wikipedia multilingual embeddings dataset.

## Dataset Information

The scripts download and process embeddings from the [Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary) dataset, which contains pre-computed int8 embeddings for Wikipedia articles in 50+ languages using Cohere's `embed-multilingual-v3.0` model.

### Original Dataset
- **Total embeddings**: 247,154,006 across all languages (exact count)
- **Dimension**: 1024
- **Type**: int8 (pre-quantized by Cohere from normalized float32)
- **Languages**: 50+ languages including English, German, French, Spanish, Chinese, Japanese, etc.
- **Dataset**: [Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary)

### Processed Output
- **Base dataset**: 97,000,000 embeddings (base.fbin)
- **Query dataset**: 10,000 embeddings (query.fbin)  
- **Type**: int8 (directly from Cohere's pre-quantized dataset)
- **Format**: Binary with shape header (compatible with FAISS/cuVS benchmarks)

## Scripts

### 1. `generate_cohere_dataset_sequential.py` (SIMPLEST - Sequential Collection)

Takes entire languages in order of size until reaching 97M embeddings. Minimizes API calls.

**Features:**
- Takes complete languages (English, German, French, etc.) in descending order
- No random sampling - just sequential reading
- Minimal API calls (only ~3-4 languages needed)
- Most efficient for API usage

**Usage:**
```bash
python generate_cohere_dataset_sequential.py
```

**Expected composition:**
- English: 41.5M embeddings (42.8%)
- German: 20.8M embeddings (21.4%)
- French: 17.8M embeddings (18.3%)
- Russian: 13.7M embeddings (14.1%)
- Spanish: 3.21M embeddings (3.3%) - partial

### 2. `generate_cohere_dataset_multithreaded.py` (Multithreaded Random Sampling)

High-performance multithreaded implementation for uniform random sampling.

**Features:**
- Pre-generates all random indices upfront
- Uses multiple threads to fetch embeddings in parallel
- Writes to pre-allocated numpy arrays (no thread contention)
- Caches language counts for faster subsequent runs
- 5-10x faster than single-threaded versions
- True uniform random sampling across all 247M embeddings

**Usage:**
```bash
python generate_cohere_dataset_multithreaded.py
# Choose option 1 for standard multithreading
# Choose option 2 for optimized work distribution
```

### 2. `generate_cohere_dataset_uniform.py` (Single-threaded Random Sampling)

Performs uniform random sampling from the entire pool of 247,154,006 embeddings without any per-language quotas.

**Features:**
- True uniform random sampling across all 247M embeddings
- Each embedding has equal probability of selection
- Two modes: Simple (fast, needs RAM) or Reservoir sampling (memory-efficient)
- No language bias - natural distribution based on dataset composition
- Uses Cohere's pre-computed int8 embeddings (no conversion needed)

**Usage:**
```bash
python generate_cohere_dataset_uniform.py
# Choose option 1 for fast processing (needs ~100GB RAM)
# Choose option 2 for memory-efficient processing (needs ~10GB RAM)
```

### 3. `generate_cohere_dataset_efficient.py` (Per-Language Sampling)

Memory-efficient version with controlled per-language sampling quotas.

**Features:**
- Processes embeddings in chunks to minimize memory usage
- Uses memory-mapped numpy arrays for intermediate storage
- Attempts to sample evenly across languages
- Handles 97M+ embeddings without loading all into memory

**Usage:**
```bash
python generate_cohere_dataset_efficient.py
```

### 4. `generate_cohere_dataset_corrected.py` (Per-Language Standard)

Standard version with per-language quotas. Requires more RAM but is simpler.

**Features:**
- Loads embeddings from all languages with quotas
- Attempts uniform sampling across languages
- Batch processing for memory management

**Usage:**
```bash
python generate_cohere_dataset_corrected.py
```

### 5. `test_fbin_format.py`

Test script to verify the binary file format is correct.

**Usage:**
```bash
python test_fbin_format.py
```

## Requirements

```bash
pip install datasets numpy tqdm
```

## Performance Comparison

| Script | Speed | RAM Usage | API Calls | Method |
|--------|-------|-----------|-----------|---------|
| **sequential** | ~30-60 min | 10 GB | ~5 | Take complete languages in order |
| multithreaded | ~10-20 min* | 95 GB | ~1000+ | Random sampling with parallel fetching |
| uniform (simple) | ~2-4 hours | 95 GB | ~300 | Sequential random sampling |
| uniform (reservoir) | ~3-6 hours | 10 GB | ~300 | Reservoir sampling |
| efficient | ~3-5 hours | 10-20 GB | ~300 | Memory-mapped with quotas |

*May hit rate limits without authentication

**Recommendation**: Use `generate_cohere_dataset_sequential.py` for simplicity and minimal API usage.

## File Format

The `.fbin` files use a simple binary format:
- First 8 bytes: Shape (2 x uint32) - [num_vectors, dimension]
- Remaining bytes: Data (int8 values)

Example reading code:
```python
def read_bin(fname, dtype=np.int8):
    with open(fname, "rb") as f:
        shape = np.fromfile(f, dtype=np.uint32, count=2)
        data = np.fromfile(f, dtype=dtype)
        data = data.reshape(shape)
    return data
```

## Expected Output Sizes

- **base.fbin**: ~94 GB (97M vectors × 1024 dimensions × 1 byte)
- **query.fbin**: ~10 MB (10K vectors × 1024 dimensions × 1 byte)

## Processing Time

Depending on your internet connection and system:
- Download speed: Limited by HuggingFace dataset streaming
- Processing: ~2-6 hours for 97M embeddings
- Disk I/O: Make sure you have fast storage (SSD recommended)

## Sampling Strategies

### Uniform Random Sampling (`generate_cohere_dataset_uniform.py`)
- **Method**: Each of the 247,154,006 embeddings has equal probability of being selected
- **Result**: Natural language distribution (more English, German, French embeddings)
- **Best for**: When you want the sample to reflect the actual Wikipedia content distribution
- **Note**: Uses pre-computed int8 embeddings from Cohere, no conversion needed

### Per-Language Quota Sampling (other scripts)
- **Method**: Attempts to sample evenly from each language
- **Result**: More balanced language representation
- **Best for**: When you want linguistic diversity regardless of Wikipedia size per language

## Notes

1. **Disjoint Sets**: The base and query datasets are completely disjoint (no overlapping embeddings)

2. **Int8 Format**: Uses Cohere's pre-quantized int8 embeddings from the `emb_int8` field (no conversion needed)

3. **Sampling Without Replacement**: Each embedding appears at most once in the output

4. **Reproducibility**: Scripts use fixed random seeds (42) for reproducible sampling

5. **Memory Requirements**:
   - Uniform sampling (simple mode): ~100 GB RAM
   - Uniform sampling (reservoir mode): ~10 GB RAM  
   - Efficient version: ~10-20 GB RAM
   - Standard version: ~100+ GB RAM

## Troubleshooting

### Out of Memory
Use the `generate_cohere_dataset_efficient.py` script which processes data incrementally.

### Slow Download
The dataset is streamed from HuggingFace. Download speed depends on your connection to HuggingFace servers.

### Incomplete Collection
If the script cannot collect enough embeddings (some languages may be unavailable), it will fill the remaining slots with random int8 values and warn you.

## Citation

If you use this dataset, please cite the original Cohere dataset:
```
@dataset{cohere_wikipedia_2023,
  title={Wikipedia 2023-11 Embed Multilingual V3},
  author={Cohere},
  year={2023},
  publisher={HuggingFace}
}
```
