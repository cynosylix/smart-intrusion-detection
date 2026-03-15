"""
Test face matching - verify embeddings work correctly
"""
import cv2
import numpy as np
import os
from pathlib import Path

try:
    from deepface import DeepFace
except ImportError:
    print("Error: DeepFace not installed")
    exit(1)

def test_matching():
    """Test if matching works"""
    print("=" * 60)
    print("Testing Face Matching")
    print("=" * 60)
    
    # Load stored embeddings
    embeddings_folder = 'embeddings'
    if not os.path.exists(embeddings_folder):
        print(f"Error: {embeddings_folder} folder not found")
        return
    
    npy_files = list(Path(embeddings_folder).glob("*.npy"))
    if len(npy_files) == 0:
        print("No embeddings found")
        return
    
    print(f"\nFound {len(npy_files)} embedding(s):")
    stored_embeddings = {}
    for npy_file in npy_files:
        name = npy_file.stem
        emb = np.load(str(npy_file))
        stored_embeddings[name] = emb
        print(f"  - {name}: shape {emb.shape}")
    
    # Test with a sample image
    test_image = input("\nEnter path to test image (or press Enter to use camera): ").strip()
    
    if not test_image:
        # Use camera
        print("\nOpening camera...")
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Failed to capture from camera")
            return
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print("Captured frame from camera")
    else:
        if not os.path.exists(test_image):
            print(f"Image not found: {test_image}")
            return
        
        bgr = cv2.imread(test_image)
        if bgr is None:
            print(f"Failed to read image: {test_image}")
            return
        
        rgb_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        print(f"Loaded image: {test_image}")
    
    # Detect face
    print("\nDetecting face...")
    gray = cv2.cvtColor(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    
    if len(faces) == 0:
        print("No face detected!")
        return
    
    x, y, w, h = faces[0]
    padding = 10
    x1, y1 = max(0, x-padding), max(0, y-padding)
    x2, y2 = min(rgb_frame.shape[1], x+w+padding), min(rgb_frame.shape[0], y+h+padding)
    face_rgb = rgb_frame[y1:y2, x1:x2]
    
    print(f"Face detected: {x2-x1}x{y2-y1} pixels")
    
    # Extract embedding
    print("\nExtracting embedding...")
    try:
        result = DeepFace.represent(
            face_rgb,
            model_name='Facenet512',
            enforce_detection=False,
            detector_backend='skip'
        )
        
        if isinstance(result, list) and len(result) > 0:
            query_emb = np.array(result[0]['embedding'])
            print(f"Embedding extracted: shape {query_emb.shape}")
        else:
            print("Failed to extract embedding")
            return
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Match
    print("\n" + "=" * 60)
    print("Matching Results:")
    print("=" * 60)
    
    best_match = None
    best_dist = float('inf')
    
    for name, stored_emb in stored_embeddings.items():
        if query_emb.shape != stored_emb.shape:
            print(f"{name}: Shape mismatch! Query {query_emb.shape} vs Stored {stored_emb.shape}")
            continue
        
        dist = np.linalg.norm(query_emb - stored_emb)
        print(f"{name}: distance = {dist:.4f}")
        
        if dist < best_dist:
            best_dist = dist
            best_match = name
    
    print("\n" + "=" * 60)
    print("Result:")
    print("=" * 60)
    print(f"Best match: {best_match}")
    print(f"Distance: {best_dist:.4f}")
    print(f"\nThreshold recommendations:")
    print(f"  If distance < 0.6: Very likely match")
    print(f"  If distance 0.6-1.0: Likely match")
    print(f"  If distance > 1.2: Unlikely match")
    print(f"\nTry threshold: {best_dist + 0.2:.2f}")

if __name__ == "__main__":
    test_matching()
