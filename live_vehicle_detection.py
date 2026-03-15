"""
Live Vehicle Detection using YOLO
Real-time vehicle detection from webcam feed
Detects: car, truck, bus, motorcycle, bicycle, train, boat, airplane, etc.
"""

from ultralytics import YOLO
import cv2
import time
from pathlib import Path
import sys

# Use YOLOv8 pretrained model (includes vehicle classes)
# This will download automatically if not present
model_name = "yolov8n.pt"  # Nano model for speed, can use yolov8s.pt, yolov8m.pt for better accuracy

# Vehicle classes in COCO dataset
VEHICLE_CLASSES = {
    'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench'
}

# Focus on actual vehicles
FOCUS_VEHICLES = {
    'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'
}

print("=" * 60)
print("Initializing Vehicle Detection System...")
print("=" * 60)

# Load YOLOv8 model (pretrained on COCO dataset)
try:
    print(f"Loading YOLOv8 model: {model_name}")
    print("(This will download automatically on first run)")
    model = YOLO(model_name)
    print("✓ Model loaded successfully!")
    
    # Find vehicle class IDs
    vehicle_class_ids = []
    if hasattr(model, 'names'):
        print(f"✓ Model classes available: {len(model.names)} classes")
        
        # Find vehicle classes
        for cls_id, cls_name in model.names.items():
            cls_lower = cls_name.lower()
            if cls_lower in FOCUS_VEHICLES:
                vehicle_class_ids.append(cls_id)
                print(f"  ✓ {cls_name} (ID: {cls_id})")
        
        if not vehicle_class_ids:
            # Fallback: use all vehicle-related classes
            print("⚠ No focus vehicles found, searching all vehicle classes...")
            for cls_id, cls_name in model.names.items():
                cls_lower = cls_name.lower()
                if cls_lower in VEHICLE_CLASSES:
                    vehicle_class_ids.append(cls_id)
                    if len(vehicle_class_ids) <= 10:  # Show first 10
                        print(f"  ✓ {cls_name} (ID: {cls_id})")
        
        if vehicle_class_ids:
            print(f"\n✓ Filtering to detect {len(vehicle_class_ids)} vehicle class(es)")
            detected_vehicles = [model.names[cid] for cid in vehicle_class_ids[:10]]
            print(f"  Examples: {', '.join(detected_vehicles)}")
        else:
            print("⚠ Warning: No vehicle classes found. Will detect all classes.")
    else:
        print("⚠ Warning: Could not determine model classes")
        # Default vehicle classes in COCO (approximate IDs)
        vehicle_class_ids = [2, 3, 5, 6, 7, 8]  # Common vehicle IDs
        
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
detection_count = {}
fps_counter = 0
fps_start_time = time.time()
current_fps = 0

print("\n" + "=" * 60)
print("Vehicle Detection System Started")
print("=" * 60)
print(f"Detecting: {len(vehicle_class_ids)} vehicle class(es)")
print("Confidence Threshold: 0.5")
print("\nControls:")
print("  Press 'q' or ESC to quit")
print("  Press 's' to save current frame")
print("  Press 'r' to reset detection counter")
print("  Press 'i' to show detection info")
print("=" * 60 + "\n")

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("✗ Error: Failed to read frame from camera")
            break
        
        # Run detection - filter to only detect vehicle classes
        results = model(frame, conf=0.5, verbose=False, 
                       classes=vehicle_class_ids if vehicle_class_ids else None)
        
        # Count detections by type
        current_detections = {}
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = model.names.get(cls_id, 'Unknown')
                conf = float(box.conf[0].cpu().numpy())
                
                if cls_name not in current_detections:
                    current_detections[cls_name] = 0
                current_detections[cls_name] += 1
                
                # Update total count
                if cls_name not in detection_count:
                    detection_count[cls_name] = 0
                detection_count[cls_name] += 1
        
        num_detections = sum(current_detections.values())
        
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
        status_bar_height = 80
        status_color = (255, 165, 0) if num_detections > 0 else (0, 0, 0)  # Orange for vehicles
        cv2.rectangle(annotated, (0, 0), (w, status_bar_height), status_color, -1)
        cv2.rectangle(annotated, (0, 0), (w, status_bar_height), (255, 255, 255), 2)
        
        # Status text
        if num_detections > 0:
            vehicles_detected = ', '.join([f"{name}({count})" for name, count in 
                                         sorted(current_detections.items(), 
                                               key=lambda x: x[1], reverse=True)[:3]])
            status_text = f"🚗 VEHICLES DETECTED: {vehicles_detected}"
            if len(current_detections) > 3:
                status_text += f" +{len(current_detections)-3} more"
            text_color = (255, 255, 255)
        else:
            status_text = "Vehicle Detection Active - No Vehicles Detected"
            text_color = (0, 255, 0)
        
        cv2.putText(annotated, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Detection count and FPS
        total_detections = sum(detection_count.values())
        info_text = f"Total Detections: {total_detections} | FPS: {current_fps:.1f}"
        cv2.putText(annotated, info_text, (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show top detected vehicles
        if detection_count:
            top_vehicles = sorted(detection_count.items(), key=lambda x: x[1], reverse=True)[:3]
            top_text = " | ".join([f"{name}: {count}" for name, count in top_vehicles])
            cv2.putText(annotated, top_text, (10, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Draw warning border if vehicles detected
        if num_detections > 0:
            cv2.rectangle(annotated, (0, 0), (w, h), (255, 165, 0), 5)  # Orange border
        
        cv2.imshow("Vehicle Detection - Press 'q' to quit", annotated)
        
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
                filename = output_dir / f"vehicle_detection_{int(time.time())}.jpg"
                cv2.imwrite(str(filename), annotated)
                print(f"✓ Frame saved: {filename}")
            except Exception as e:
                print(f"✗ Error saving frame: {e}")
        elif key == ord('r'):
            # Reset detection counter
            detection_count = {}
            print("✓ Detection counter reset")
        elif key == ord('i'):
            # Show detection info
            print("\n" + "-" * 60)
            print("Detection Statistics:")
            print("-" * 60)
            if detection_count:
                sorted_vehicles = sorted(detection_count.items(), key=lambda x: x[1], reverse=True)
                for i, (vehicle, count) in enumerate(sorted_vehicles, 1):
                    print(f"{i}. {vehicle}: {count} detections")
            else:
                print("No detections yet")
            print("-" * 60 + "\n")

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
    print(f"  Total Vehicle Detections: {sum(detection_count.values())}")
    print(f"  Average FPS: {current_fps:.1f}")
    
    if detection_count:
        print("\n  Detections by Vehicle Type:")
        sorted_vehicles = sorted(detection_count.items(), key=lambda x: x[1], reverse=True)
        for vehicle, count in sorted_vehicles:
            print(f"    - {vehicle}: {count}")
    
    if session_duration > 0:
        total_detections = sum(detection_count.values())
        detection_rate = total_detections / session_duration
        print(f"\n  Detection Rate: {detection_rate:.2f} detections/second")
    
    print("=" * 60)
    print("Camera released. Goodbye!")
