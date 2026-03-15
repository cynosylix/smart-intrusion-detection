"""
Test script to verify embeddings are working correctly

This script helps debug why faces show as "Unknown"
"""

import numpy as np
import os
from pathlib import Path

def test_embeddings(embeddings_folder='embeddings'):
    """Test if embeddings are loaded correctly"""
    print("=" * 60)
    print("Testing Embeddings")
    print("=" * 60)
    
    if not os.path.exists(embeddings_folder):
        print(f"❌ Embeddings folder '{embeddings_folder}' does not exist!")
        return
    
    npy_files = list(Path(embeddings_folder).glob("*.npy"))
    
    if len(npy_files) == 0:
        print(f"❌ No .npy files found in '{embeddings_folder}'")
        print(f"   Please upload face images first using:")
        print(f"   python face_upload_embedding.py <image> <name>")
        return
    
    print(f"\nFound {len(npy_files)} embedding file(s):\n")
    
    embeddings_data = {}
    
    for npy_file in npy_files:
        try:
            person_name = npy_file.stem
            embedding = np.load(str(npy_file))
            
            embeddings_data[person_name] = embedding
            
            print(f"✓ {person_name}.npy")
            print(f"  Shape: {embedding.shape}")
            print(f"  Dtype: {embedding.dtype}")
            print(f"  Min: {embedding.min():.4f}, Max: {embedding.max():.4f}")
            print(f"  Mean: {embedding.mean():.4f}")
            print()
            
        except Exception as e:
            print(f"❌ Error loading {npy_file}: {e}\n")
    
    # Test distance calculation
    if len(embeddings_data) >= 2:
        print("=" * 60)
        print("Testing Distance Calculation")
        print("=" * 60)
        
        names = list(embeddings_data.keys())
        emb1 = embeddings_data[names[0]]
        emb2 = embeddings_data[names[1]]
        
        distance = np.linalg.norm(emb1 - emb2)
        print(f"\nDistance between '{names[0]}' and '{names[1]}': {distance:.4f}")
        print(f"\nRecommended threshold range: 0.6 - 1.2")
        print(f"  If distance < 0.6: Very similar (same person)")
        print(f"  If distance 0.6-1.0: Likely same person")
        print(f"  If distance > 1.2: Different person")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✓ Loaded {len(embeddings_data)} embedding(s) successfully")
    print(f"\nIf all faces show as 'Unknown', try:")
    print(f"  1. Increase threshold (press '+' in live recognition)")
    print(f"  2. Check if embeddings were created correctly")
    print(f"  3. Make sure you're using the same person in camera")

if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else 'embeddings'
    test_embeddings(folder)
