"""
Live Face Recognition using face_recognition library

This script:
1. Loads known face encodings from embeddings folder or image files
2. Opens webcam and captures live video
3. Detects faces in each frame
4. Compares detected faces with known faces
5. Displays "Known Person: <name>" or "Unknown" on the video feed
"""

import cv2
import face_recognition
import numpy as np
import os
from pathlib import Path

# Note: face_recognition library uses 128-dimensional encodings
# DeepFace uses 512-dimensional embeddings - these are NOT compatible
# So we only load from image files, not .npy files

# Option 2: Load from image files (original method)
def load_encodings_from_images(image_files, names):
    """Load face encodings from image files"""
    known_face_encodings = []
    known_face_names = []
    
    for img_file, name in zip(image_files, names):
        if os.path.exists(img_file):
            try:
                img = face_recognition.load_image_file(img_file)
                encodings = face_recognition.face_encodings(img)
                if len(encodings) > 0:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                    print(f"✓ Loaded encoding for: {name}")
                else:
                    print(f"❌ No face found in: {img_file}")
            except Exception as e:
                print(f"❌ Error loading {img_file}: {e}")
        else:
            print(f"❌ Image file not found: {img_file}")
    
    return known_face_encodings, known_face_names

# Load from image files (face_recognition library format)
print("=" * 60)
print("Loading known faces from image files...")
print("=" * 60)

# Replace these with your actual image file paths
known_face_images = [
    "mmmm.jpeg",  # Replace with your image file
    "ss.jpg"    # Replace with your friend's image file
]
known_face_names = ["shinat", "amal"]  # Labels for recognized faces

known_face_encodings, known_face_names = load_encodings_from_images(
    known_face_images, known_face_names
)

if len(known_face_encodings) == 0:
    print("\n❌ Error: No known faces loaded!")
    print("   Please update the image file paths in this script:")
    print("   - Make sure image files exist (Mammu.jpeg, Moha.jpeg)")
    print("   - Or update the 'known_face_images' list with your image paths")
    print("   - Make sure each image contains a clear face")
    exit(1)

print(f"\n✓ Loaded {len(known_face_encodings)} known face(s)")
print("=" * 60)

# Open webcam
print("\nOpening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open camera")
    exit(1)

print("Camera opened successfully!")
print("Press 'q' to quit")
print("=" * 60 + "\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to RGB (face_recognition uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect face locations and encode them
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    # Process each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        color = (0, 0, 255)  # Red for unknown
        
        # Find the best match
        if True in matches:
            # Calculate face distances
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            # Use a tolerance threshold (default is 0.6)
            if face_distances[best_match_index] < 0.6:
                name = known_face_names[best_match_index]
                color = (0, 255, 0)  # Green for known
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown
        
        # Draw rectangle around face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw label
        label = f"Known: {name}" if name != "Unknown" else "Unknown"
        cv2.putText(frame, label, (left, top - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Show status
    status_text = f"Known faces: {len(known_face_names)} | Detected: {len(face_locations)}"
    cv2.putText(frame, status_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Face Recognition Live", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✓ Face recognition stopped")
