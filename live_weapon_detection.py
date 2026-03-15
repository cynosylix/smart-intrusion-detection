"""
Live Weapon Detection using YOLO
Real-time weapon detection (Knife and Pistol) from webcam feed
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import sys

# Try to find the model file
model_paths = [
    "realtime-weapon-detection/best.pt",
    "Weapon-Detection-YOLOv5/yolov5s.pt",
    "best.pt",
    "yolov5s.pt"
]

model_path = None
for path in model_paths:
    if Path(path).exists():
        model_path = path
        print(f"✓ Found model: {model_path}")
        break

if not model_path:
    print("✗ Error: Could not find model file!")
    print("Looking for:")
    for path in model_paths:
        print(f"  - {path}")
    sys.exit(1)

# Load model
weapon_classes = None
try:
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    print("✓ Model loaded successfully!")
    
    # Show model classes and find Knife/Pistol
    if hasattr(model, 'names'):
        print(f"✓ Model classes: {model.names}")
        
        # Find Knife and Pistol class IDs
        weapon_class_ids = []
        for cls_id, cls_name in model.names.items():
            cls_lower = cls_name.lower()
            if cls_lower in ['knife', 'pistol']:
                weapon_class_ids.append(cls_id)
                print(f"  ✓ {cls_name} (ID: {cls_id})")
        
        if weapon_class_ids:
            weapon_classes = weapon_class_ids
            print(f"✓ Filtering to detect only: {[model.names[cid] for cid in weapon_class_ids]}")
        else:
            print("⚠ Warning: No Knife or Pistol classes found. Will detect all classes.")
    else:
        print("⚠ Warning: Could not determine model classes")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

# Open webcam
print("\nOpening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("✗ Error: Could not open camera!")
    print("Try changing camera index (modify VideoCapture(0) to VideoCapture(1))")
    sys.exit(1)

print("✓ Camera opened successfully!")
print("\n" + "=" * 60)
print("Weapon Detection System Started")
print("=" * 60)
print("Press 'q' or ESC to quit")
print("=" * 60 + "\n")

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("✗ Error: Failed to read frame from camera")
            break
        
        # Run detection with confidence threshold
        # Filter to only detect Knife and Pistol classes if available
        results = model(frame, conf=0.5, verbose=False, classes=weapon_classes)
        
        # Plot results
        annotated = results[0].plot()
        
        # Add status text
        cv2.putText(annotated, "Weapon Detection Active", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Weapon Detection - Press 'q' to quit", annotated)
        
        # Check for quit (ESC or 'q')
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            print("\nStopping detection...")
            break

except KeyboardInterrupt:
    print("\n\nInterrupted by user")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Goodbye!")