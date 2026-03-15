"""
Live Camera Face Capture and Embedding Extraction using MediaPipe

This script:
1. Opens the webcam
2. Continuously reads frames
3. Detects a face using MediaPipe Face Detection
4. When a face is detected:
   - Crops the face
   - Converts it to an embedding using DeepFace Facenet512
   - Saves the embedding as .npy in embeddings/ folder

Usage:
    python face_upload_embedding.py <person_name> [camera_index]

Example:
    python face_upload_embedding.py "john"
    python face_upload_embedding.py "john" 0
"""

import cv2
import numpy as np
import os
import sys

try:
    from deepface import DeepFace
except ImportError:
    print("❌ Error: DeepFace is not installed!")
    print("   Please install it using: pip install deepface")
    sys.exit(1)

try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe import ImageFormat
except ImportError:
    print("❌ Error: MediaPipe is not installed!")
    print("   Please install it using: pip install mediapipe")
    sys.exit(1)


def detect_face_from_frame(frame_bgr, face_detector):
    """
    Detect face in a BGR frame using MediaPipe Face Detection.
    
    Args:
        frame_bgr: BGR frame from camera
        face_detector: MediaPipe FaceDetector object
        
    Returns:
        face_box: (x1, y1, x2, y2) OR None
        face_rgb: cropped face (RGB) OR None
        annotated_frame: frame with rectangles drawn (for display)
    """
    annotated = frame_bgr.copy()
    h, w = frame_bgr.shape[:2]
    
    # Convert BGR to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert numpy array to MediaPipe Image
    mp_image = vision.Image(image_format=ImageFormat.SRGB, data=frame_rgb)
    
    # Detect faces using MediaPipe
    detection_result = face_detector.detect(mp_image)
    
    if not detection_result.detections or len(detection_result.detections) == 0:
        return None, None, annotated
    
    # Get the first (largest) face detection
    detection = detection_result.detections[0]
    bbox = detection.bounding_box
    
    # Get bounding box coordinates
    x = bbox.origin_x
    y = bbox.origin_y
    face_w = bbox.width
    face_h = bbox.height
    
    # Add padding
    padding = 20
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + face_w + padding)
    y2 = min(h, y + face_h + padding)
    
    # Draw bounding box for visualization
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw confidence score
    confidence = detection.categories[0].score if detection.categories else 0.0
    cv2.putText(annotated, f"Face: {confidence:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Crop face region
    face_bgr_crop = frame_bgr[y1:y2, x1:x2]
    if face_bgr_crop.size == 0:
        return None, None, annotated
    
    # Convert BGR to RGB for DeepFace
    face_rgb = cv2.cvtColor(face_bgr_crop, cv2.COLOR_BGR2RGB)
    
    print(f"✓ Face detected at: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"  Face size: {x2 - x1}x{y2 - y1} pixels")
    print(f"  Confidence: {confidence:.2f}")
    
    return (x1, y1, x2, y2), face_rgb, annotated


def extract_embedding(face_image):
    """
    Extract face embedding using DeepFace Facenet512 model
    
    Args:
        face_image: RGB numpy array of face image
        
    Returns:
        embedding: 512-dimensional numpy array or None
    """
    try:
        print("Extracting embedding using DeepFace Facenet512...")
        
        # Ensure reasonable size for DeepFace
        min_size = 80
        h, w = face_image.shape[:2]
        if h < min_size or w < min_size:
            scale = max(min_size / h, min_size / w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            face_image = cv2.resize(face_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # DeepFace expects RGB image
        embedding = DeepFace.represent(
            face_image,
            model_name='Facenet512',
            enforce_detection=False,  # Don't re-detect, we already have face
            detector_backend='skip'   # Skip detection, use provided face
        )
        
        # DeepFace returns a list with dict containing 'embedding'
        if isinstance(embedding, list) and len(embedding) > 0:
            embedding_vector = np.array(embedding[0]['embedding'])
            print(f"✓ Embedding extracted: {len(embedding_vector)} dimensions")
            return embedding_vector
        
        print("❌ Failed to extract embedding")
        return None
            
    except Exception as e:
        print(f"❌ Error extracting embedding: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_embedding(person_name, embedding, embeddings_folder='embeddings'):
    """
    Save embedding to .npy file
    
    Args:
        person_name: Name of the person
        embedding: Numpy array of embedding
        embeddings_folder: Folder to save embeddings
        
    Returns:
        file_path: Path to saved .npy file
    """
    # Create embeddings folder if it doesn't exist
    os.makedirs(embeddings_folder, exist_ok=True)
    
    # Sanitize person name for filename
    safe_name = "".join(c for c in person_name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name.replace(' ', '_').lower()
    
    if not safe_name:
        safe_name = "unknown"
    
    # Save embedding
    file_path = os.path.join(embeddings_folder, f"{safe_name}.npy")
    np.save(file_path, embedding)
    
    print(f"✓ Embedding saved to: {file_path}")
    return file_path


def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage:")
        print("  python face_upload_embedding.py <person_name> [camera_index]")
        print("\nExample:")
        print('  python face_upload_embedding.py "john"')
        print('  python face_upload_embedding.py "john" 0')
        sys.exit(1)
    
    person_name = sys.argv[1]
    camera_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    print("=" * 60)
    print("Live Camera Face Capture and Embedding Extraction")
    print("Using MediaPipe for Face Detection")
    print("=" * 60)
    print(f"Person: {person_name}")
    print(f"Camera: {camera_index}")
    print("=" * 60)
    
    # Initialize MediaPipe Face Detection (IMAGE mode for synchronous detection)
    base_options = python.BaseOptions(model_asset_path=None)  # Use default model
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
        min_suppression_threshold=0.3
    )
    
    # Create face detector
    face_detector = vision.FaceDetector.create_from_options(options)
    
    # Open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera (index {camera_index})")
        print("   Try another index: python face_upload_embedding.py <name> 1")
        sys.exit(1)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nCamera opened. Showing live feed...")
    print("As soon as a face is clearly detected, it will capture that frame,")
    print("extract the embedding, save it, and close.")
    print("Press 'q' to quit without saving.")
    print("Press SPACE to manually capture when face is detected.")
    print("=" * 60)
    
    captured_embedding = None
    face_detected_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from camera")
                break
            
            # Detect face using MediaPipe
            face_box, face_rgb, annotated = detect_face_from_frame(frame, face_detector)
            
            # Show live preview
            cv2.putText(annotated, "Press SPACE to capture | 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if face_rgb is not None:
                face_detected_count += 1
                cv2.putText(annotated, f"Face detected! ({face_detected_count})", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Face Capture - Press SPACE to capture | 'q' to quit", annotated)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q") or key == 27:  # 'q' or ESC
                print("\nCancelled by user.")
                break
            
            if key == ord(" ") and face_rgb is not None:  # SPACE key - capture face
                print("\n[Step 2] Extracting embedding...")
                embedding = extract_embedding(face_rgb)
                
                if embedding is not None:
                    print("\n[Step 3] Saving embedding...")
                    save_embedding(person_name, embedding)
                    captured_embedding = embedding
                    print("\n✓ Face captured and embedding saved!")
                    print("Closing camera...")
                    break
                else:
                    print("❌ Failed: Could not extract embedding from detected face")
                    print("Try again - make sure face is clearly visible")
            
            # Auto-capture after face is detected for 10 consecutive frames
            if face_rgb is not None and face_detected_count >= 10:
                print("\nAuto-capturing after stable face detection...")
                print("[Step 2] Extracting embedding...")
                embedding = extract_embedding(face_rgb)
                
                if embedding is not None:
                    print("\n[Step 3] Saving embedding...")
                    save_embedding(person_name, embedding)
                    captured_embedding = embedding
                    print("\n✓ Face captured and embedding saved!")
                    break
                else:
                    print("❌ Failed: Could not extract embedding")
                    face_detected_count = 0  # Reset counter
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        # Cleanup
        face_detector.close()
        cap.release()
        cv2.destroyAllWindows()
    
    if captured_embedding is not None:
        print("\n" + "=" * 60)
        print("✓ SUCCESS!")
        print(f"  Person: {person_name}")
        print(f"  Embedding dimensions: {len(captured_embedding)}")
        print("=" * 60)
        print("\n✓ Process completed successfully!")
    else:
        print("\n❌ Process did not complete successfully (no embedding saved).")
        print("   Make sure:")
        print("   - Your face is clearly visible")
        print("   - There's good lighting")
        print("   - Press SPACE when face is detected")


if __name__ == "__main__":
    main()
