#!/usr/bin/env python3
"""
Quick verification script to show the difference between float32 and int8 datasets.
"""

from datasets import load_dataset
import numpy as np

def check_datasets():
    """Compare the float32 and int8 datasets."""
    
    print("="*60)
    print("Cohere Dataset Comparison")
    print("="*60)
    
    # Load a small sample from both datasets
    lang = "simple"  # Use a small language for quick testing
    
    print(f"\nLoading sample from '{lang}' language...")
    
    # Load float32 dataset
    print("\n1. Float32 dataset (original):")
    print("   Dataset: Cohere/wikipedia-2023-11-embed-multilingual-v3")
    try:
        dataset_float32 = load_dataset(
            "Cohere/wikipedia-2023-11-embed-multilingual-v3",
            lang,
            split="train",
            streaming=True
        )
        
        # Get first document
        for doc in dataset_float32:
            emb_float = np.array(doc['emb'], dtype=np.float32)
            print(f"   Field name: 'emb'")
            print(f"   Shape: {emb_float.shape}")
            print(f"   Dtype: {emb_float.dtype}")
            print(f"   Range: [{emb_float.min():.4f}, {emb_float.max():.4f}]")
            print(f"   First 5 values: {emb_float[:5]}")
            break
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n2. Int8 dataset (pre-quantized):")
    print("   Dataset: Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary")
    try:
        dataset_int8 = load_dataset(
            "Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary",
            lang,
            split="train",
            streaming=True
        )
        
        # Get first document
        for doc in dataset_int8:
            emb_int8 = np.array(doc['emb_int8'], dtype=np.int8)
            print(f"   Field name: 'emb_int8'")
            print(f"   Shape: {emb_int8.shape}")
            print(f"   Dtype: {emb_int8.dtype}")
            print(f"   Range: [{emb_int8.min()}, {emb_int8.max()}]")
            print(f"   First 5 values: {emb_int8[:5]}")
            
            # Show what manual conversion would look like
            if 'emb_float' in locals():
                manual_int8 = (emb_float * 127.0).astype(np.int8)
                print(f"\n   Manual conversion from float32:")
                print(f"   First 5 values: {manual_int8[:5]}")
                print(f"   Match with pre-quantized: {np.array_equal(manual_int8[:5], emb_int8[:5])}")
            break
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "="*60)
    print("Key Differences:")
    print("="*60)
    print("1. Float32 dataset: uses field 'emb' with float32 values")
    print("2. Int8 dataset: uses field 'emb_int8' with pre-quantized int8 values")
    print("3. No conversion needed when using int8-binary dataset")
    print("4. Int8 dataset is ~4x smaller in storage")
    print("\nRecommendation: Use the int8-binary dataset directly!")

if __name__ == "__main__":
    check_datasets()




