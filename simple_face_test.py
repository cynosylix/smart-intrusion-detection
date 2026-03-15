"""
Simple test to verify face recognition is working
Run this to see what's happening step by step
"""
import cv2
import numpy as np
import os

try:
    from deepface import DeepFace
except ImportError:
    print("Install deepface: pip install deepface")
    exit(1)

# Load stored embedding
emb_file = 'embeddings/sj.npy'
if not os.path.exists(emb_file):
    print(f"Error: {emb_file} not found")
    print("First create embedding: python face_upload_embedding.py image.jpg sj")
    exit(1)

stored_emb = np.load(emb_file)
print(f"Loaded stored embedding: {stored_emb.shape}")

# Open camera
print("\nOpening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

print("Press SPACE to capture and test, 'q' to quit\n")

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    
    # Draw detection
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.putText(frame, "Press SPACE to test recognition", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Face Recognition Test", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
    if key == ord(' ') and len(faces) > 0:  # SPACE key
        x, y, w, h = faces[0]
        padding = 10
        x1, y1 = max(0, x-padding), max(0, y-padding)
        x2, y2 = min(frame.shape[1], x+w+padding), min(frame.shape[0], y+h+padding)
        
        face_bgr = frame[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        
        print("\n" + "="*60)
        print("Testing recognition...")
        print(f"Face size: {x2-x1}x{y2-y1}")
        
        try:
            # Extract embedding
            print("Extracting embedding...")
            result = DeepFace.represent(
                face_rgb,
                model_name='Facenet512',
                enforce_detection=False,
                detector_backend='skip'
            )
            
            if isinstance(result, list) and len(result) > 0:
                query_emb = np.array(result[0]['embedding'])
                print(f"Query embedding shape: {query_emb.shape}")
                
                # Calculate distance
                distance = np.linalg.norm(query_emb - stored_emb)
                print(f"\nDistance to 'sj': {distance:.4f}")
                
                # Test different thresholds
                print("\nThreshold test:")
                for thresh in [0.5, 0.6, 0.8, 1.0, 1.2, 1.5]:
                    match = "MATCH" if distance <= thresh else "NO MATCH"
                    print(f"  Threshold {thresh:.1f}: {match}")
                
                if distance <= 1.0:
                    print(f"\n✓ MATCH! Person recognized as 'sj'")
                    print(f"  Recommended threshold: {distance + 0.2:.2f}")
                else:
                    print(f"\n✗ NO MATCH (distance too high)")
                    print(f"  Try increasing threshold above {distance:.2f}")
            else:
                print("Failed to extract embedding")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        print("="*60 + "\n")

cap.release()
cv2.destroyAllWindows()
