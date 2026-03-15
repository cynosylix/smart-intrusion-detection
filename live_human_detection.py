"""
Live Human Detection using YOLO
Real-time human/person detection from webcam feed
"""

from ultralytics import YOLO
import cv2
import time
from pathlib import Path
import sys

# Use YOLOv8 pretrained model (includes 'person' class)
# This will download automatically if not present
model_name = "yolov8n.pt"  # Nano model for speed, can use yolov8s.pt, yolov8m.pt for better accuracy

print("=" * 60)
print("Initializing Human Detection System...")
print("=" * 60)

# Load YOLOv8 model (pretrained on COCO dataset)
try:
    print(f"Loading YOLOv8 model: {model_name}")
    print("(This will download automatically on first run)")
    model = YOLO(model_name)
    print("✓ Model loaded successfully!")
    
    # Find 'person' class ID (usually 0 in COCO dataset)
    person_class_id = None
    if hasattr(model, 'names'):
        print(f"✓ Model classes available: {len(model.names)} classes")
        
        # Find person class
        for cls_id, cls_name in model.names.items():
            if cls_name.lower() == 'person':
                person_class_id = cls_id
                print(f"✓ Found 'person' class (ID: {person_class_id})")
                break
        
        if person_class_id is None:
            print("⚠ Warning: 'person' class not found. Will detect all classes.")
            print("Available classes:", list(model.names.values())[:10], "...")
    else:
        print("⚠ Warning: Could not determine model classes")
        # Default to class 0 (person in COCO)
        person_class_id = 0
        print("Using default person class ID: 0")
        
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

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✓ Camera opened successfully!")

# Get camera info
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"✓ Camera resolution: {actual_width}x{actual_height}")

# Statistics
detection_count = 0
fps_counter = 0
fps_start_time = time.time()
current_fps = 0

print("\n" + "=" * 60)
print("Human Detection System Started")
print("=" * 60)
print("Detecting: Person/Human")
print("Confidence Threshold: 0.5")
print("\nControls:")
print("  Press 'q' or ESC to quit")
print("  Press 's' to save current frame")
print("  Press 'r' to reset detection counter")
print("=" * 60 + "\n")

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("✗ Error: Failed to read frame from camera")
            break
        
        # Run detection - filter to only detect person class
        results = model(frame, conf=0.5, verbose=False, classes=[person_class_id] if person_class_id is not None else None)
        
        # Count detections
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        if num_detections > 0:
            detection_count += num_detections
        
        # Plot results
        annotated = results[0].plot()
        
        # Calculate FPS
        fps_counter += 1
        if fps_counter % 30 == 0:
            elapsed = time.time() - fps_start_time
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start_time = time.time()
        
        # Add status overlay
        h, w = annotated.shape[:2]
        
        # Status bar
        status_bar_height = 60
        status_color = (0, 255, 0) if num_detections > 0 else (0, 0, 0)
        cv2.rectangle(annotated, (0, 0), (w, status_bar_height), status_color, -1)
        cv2.rectangle(annotated, (0, 0), (w, status_bar_height), (255, 255, 255), 2)
        
        # Status text
        if num_detections > 0:
            status_text = f"⚠️ HUMAN DETECTED: {num_detections} person(s)"
            text_color = (255, 255, 255)
        else:
            status_text = "Human Detection Active - No Humans Detected"
            text_color = (0, 255, 0)
        
        cv2.putText(annotated, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Detection count and FPS
        info_text = f"Total Detections: {detection_count} | FPS: {current_fps:.1f}"
        cv2.putText(annotated, info_text, (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw warning border if human detected
        if num_detections > 0:
            cv2.rectangle(annotated, (0, 0), (w, h), (0, 255, 0), 5)
        
        cv2.imshow("Human Detection - Press 'q' to quit", annotated)
        
        # Check for quit (ESC or 'q')
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            print("\nStopping detection...")
            break
        elif key == ord('s'):
            # Save current frame
            try:
                output_dir = Path("detections")
                output_dir.mkdir(exist_ok=True)
                filename = output_dir / f"human_detection_{int(time.time())}.jpg"
                cv2.imwrite(str(filename), annotated)
                print(f"✓ Frame saved: {filename}")
            except Exception as e:
                print(f"✗ Error saving frame: {e}")
        elif key == ord('r'):
            # Reset detection counter
            detection_count = 0
            print("✓ Detection counter reset")

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
    
    # Session summary
    session_duration = time.time() - fps_start_time
    print("\n" + "=" * 60)
    print("Session Summary:")
    print("=" * 60)
    print(f"  Total Human Detections: {detection_count}")
    print(f"  Average FPS: {current_fps:.1f}")
    if session_duration > 0:
        detection_rate = detection_count / session_duration
        print(f"  Detection Rate: {detection_rate:.2f} detections/second")
    print("=" * 60)
    print("Camera released. Goodbye!")
