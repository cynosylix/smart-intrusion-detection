"""
Live Number Plate Detection and Recognition
Real-time license plate detection and OCR text reading from webcam feed
"""

import cv2
import sys
import numpy as np

# Patch cv2.imshow if not available (for headless OpenCV)
if not hasattr(cv2, 'imshow'):
    def dummy_imshow(*args, **kwargs):
        pass
    cv2.imshow = dummy_imshow
    
    def dummy_destroyAllWindows(*args, **kwargs):
        pass
    cv2.destroyAllWindows = dummy_destroyAllWindows
    
    def dummy_waitKey(*args, **kwargs):
        return -1
    cv2.waitKey = dummy_waitKey

from ultralytics import YOLO
import time
from pathlib import Path

# Try to import OCR libraries
try:
    import easyocr
    OCR_ENGINE = "easyocr"
    print("✓ Using EasyOCR for text recognition")
except ImportError:
    try:
        import pytesseract
        OCR_ENGINE = "tesseract"
        print("✓ Using Tesseract OCR for text recognition")
    except ImportError:
        OCR_ENGINE = None
        print("⚠ Warning: No OCR library found. Install EasyOCR or pytesseract")
        print("  Install EasyOCR: pip install easyocr")
        print("  Install Tesseract: pip install pytesseract (requires Tesseract binary)")

# Use YOLOv8 pretrained model or license plate detection model
# For license plate detection, we can use a custom model or COCO classes
model_name = "yolov8n.pt"  # Will try to detect license plates

# License plate related classes in COCO (if available)
# Note: Standard COCO doesn't have license plate class, so we'll detect vehicles first
# and then try to find plates within vehicle regions, or use a custom approach

print("=" * 60)
print("Initializing Number Plate Detection System...")
print("=" * 60)

# Load YOLOv8 model
try:
    print(f"Loading YOLOv8 model: {model_name}")
    print("(This will download automatically on first run)")
    model = YOLO(model_name)
    print("✓ Model loaded successfully!")
    
    # Initialize OCR if available
    reader = None
    if OCR_ENGINE == "easyocr":
        print("Initializing EasyOCR...")
        reader = easyocr.Reader(['en'], gpu=False)  # English only, CPU mode
        print("✓ EasyOCR initialized!")
    elif OCR_ENGINE == "tesseract":
        print("✓ Tesseract OCR ready")
    
    if not OCR_ENGINE:
        print("⚠ Warning: OCR not available. Only plate detection will work.")
        
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
plate_texts = []
fps_counter = 0
fps_start_time = time.time()
current_fps = 0

def preprocess_plate(plate_roi):
    """
    Preprocess license plate ROI for better OCR
    """
    # Convert to grayscale
    if len(plate_roi.shape) == 3:
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_roi
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    # Resize for better OCR (if too small)
    h, w = denoised.shape
    if h < 30 or w < 100:
        scale = max(30/h, 100/w)
        new_h, new_w = int(h * scale), int(w * scale)
        denoised = cv2.resize(denoised, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    return denoised

def read_plate_text(plate_roi, reader):
    """
    Read text from license plate using OCR
    """
    if reader is None:
        return None
    
    try:
        # Preprocess plate
        processed = preprocess_plate(plate_roi)
        
        if OCR_ENGINE == "easyocr":
            # EasyOCR
            results = reader.readtext(processed)
            if results:
                # Get the text with highest confidence
                text = ""
                confidence = 0
                for (bbox, detected_text, conf) in results:
                    if conf > confidence:
                        text = detected_text
                        confidence = conf
                # Clean text (remove spaces, special chars)
                text = ''.join(c for c in text if c.isalnum() or c in ['-', ' '])
                return text.strip() if confidence > 0.5 else None
        
        elif OCR_ENGINE == "tesseract":
            # Tesseract OCR
            custom_config = r'--oem 3 --psm 7'  # Single line text
            text = pytesseract.image_to_string(processed, config=custom_config)
            text = ''.join(c for c in text if c.isalnum() or c in ['-', ' '])
            return text.strip() if text.strip() else None
            
    except Exception as e:
        print(f"⚠ OCR Error: {e}")
        return None
    
    return None

def detect_plates_in_frame(frame, model):
    """
    Detect potential license plates in frame
    Since COCO doesn't have license plate class, we'll:
    1. Detect vehicles (cars, trucks, buses)
    2. Look for rectangular regions that might be plates
    3. Or use edge detection to find plate-like regions
    """
    # Method 1: Detect vehicles and look for plates in vehicle regions
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck in COCO
    results = model(frame, conf=0.3, verbose=False, classes=vehicle_classes)
    
    plate_regions = []
    
    # Method 2: Use edge detection to find rectangular regions
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours that look like license plates
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h > 0 else 0
        area = cv2.contourArea(contour)
        
        # License plates typically have aspect ratio between 2:1 and 5:1
        # and minimum area
        if 2.0 <= aspect_ratio <= 5.0 and area > 500:
            # Check if it's in lower portion of frame (where plates usually are)
            if y > frame.shape[0] * 0.3:  # Lower 70% of frame
                plate_regions.append((x, y, w, h))
    
    # Also check vehicle bounding boxes for plate regions
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Extract vehicle region
                vehicle_roi = frame[y1:y2, x1:x2]
                if vehicle_roi.size > 0:
                    # Look for plate in lower portion of vehicle
                    vehicle_h = y2 - y1
                    plate_y1 = y1 + int(vehicle_h * 0.6)  # Lower 40% of vehicle
                    plate_y2 = y2
                    plate_roi = frame[plate_y1:plate_y2, x1:x2]
                    
                    if plate_roi.size > 0:
                        # Check if this region looks like a plate
                        plate_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                        plate_edges = cv2.Canny(plate_gray, 50, 150)
                        plate_contours, _ = cv2.findContours(plate_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for cnt in plate_contours:
                            px, py, pw, ph = cv2.boundingRect(cnt)
                            p_aspect = pw / float(ph) if ph > 0 else 0
                            p_area = cv2.contourArea(cnt)
                            
                            if 2.0 <= p_aspect <= 5.0 and p_area > 100:
                                plate_regions.append((x1 + px, plate_y1 + py, pw, ph))
    
    return plate_regions

print("\n" + "=" * 60)
print("Number Plate Detection System Started")
print("=" * 60)
print("Detection Method: Edge detection + Vehicle-based detection")
if OCR_ENGINE:
    print(f"OCR Engine: {OCR_ENGINE.upper()}")
else:
    print("OCR Engine: Not available (detection only)")

# Check if display is available
try:
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imshow("test", test_img)
    cv2.destroyWindow("test")
    display_available = True
    print("\nControls:")
    print("  Press 'q' or ESC to quit")
    print("  Press 's' to save current frame")
    print("  Press 'r' to reset detection counter")
    print("  Press 't' to show detected plate texts")
except:
    display_available = False
    print("\n⚠ Display not available (headless mode)")
    print("  Frames will be auto-saved when plates are detected")
    print("  Press Ctrl+C to stop")

print("=" * 60 + "\n")

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("✗ Error: Failed to read frame from camera")
            break
        
        # Detect license plates
        plate_regions = detect_plates_in_frame(frame, model)
        
        # Process each detected plate
        detected_plates = []
        for (x, y, w, h) in plate_regions:
            # Extract plate ROI
            plate_roi = frame[y:y+h, x:x+w]
            
            if plate_roi.size > 0:
                # Read text from plate
                plate_text = read_plate_text(plate_roi, reader) if OCR_ENGINE else None
                
                detected_plates.append({
                    'bbox': (x, y, w, h),
                    'text': plate_text
                })
                
                if plate_text:
                    detection_count += 1
                    if plate_text not in plate_texts:
                        plate_texts.append(plate_text)
        
        # Draw detections
        annotated = frame.copy()
        
        for plate_info in detected_plates:
            x, y, w, h = plate_info['bbox']
            text = plate_info['text']
            
            # Draw bounding box
            color = (0, 255, 0) if text else (0, 255, 255)  # Green if text read, yellow if not
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            
            # Draw text label
            if text:
                label = f"Plate: {text}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x, y-label_size[1]-5), 
                            (x+label_size[0], y), color, -1)
                cv2.putText(annotated, label, (x, y-5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            else:
                label = "License Plate"
                cv2.putText(annotated, label, (x, y-5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
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
        status_color = (0, 255, 0) if detected_plates else (0, 0, 0)
        cv2.rectangle(annotated, (0, 0), (w, status_bar_height), status_color, -1)
        cv2.rectangle(annotated, (0, 0), (w, status_bar_height), (255, 255, 255), 2)
        
        # Status text
        if detected_plates:
            plates_with_text = [p['text'] for p in detected_plates if p['text']]
            if plates_with_text:
                status_text = f"🚗 PLATES DETECTED: {len(detected_plates)} | READ: {', '.join(plates_with_text[:2])}"
            else:
                status_text = f"🚗 PLATES DETECTED: {len(detected_plates)} (Reading text...)"
            text_color = (255, 255, 255)
        else:
            status_text = "Number Plate Detection Active - No Plates Detected"
            text_color = (0, 255, 0)
        
        cv2.putText(annotated, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Detection count and FPS
        info_text = f"Total Plates Read: {detection_count} | Unique: {len(plate_texts)} | FPS: {current_fps:.1f}"
        cv2.putText(annotated, info_text, (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show recent plate texts
        if plate_texts:
            recent_texts = plate_texts[-3:]
            texts_display = " | ".join(recent_texts)
            cv2.putText(annotated, f"Recent: {texts_display}", (10, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Try to display frame, fallback to saving if GUI not available
        try:
            cv2.imshow("Number Plate Detection - Press 'q' to quit", annotated)
            display_available = True
        except cv2.error as e:
            if "not implemented" in str(e).lower() or "imshow" in str(e).lower():
                display_available = False
                # Auto-save frames when display not available
                if num_detections > 0 and fps_counter % 30 == 0:  # Save every 30 frames when plates detected
                    try:
                        output_dir = Path("detections")
                        output_dir.mkdir(exist_ok=True)
                        filename = output_dir / f"plate_detection_{int(time.time())}.jpg"
                        cv2.imwrite(str(filename), annotated)
                        print(f"✓ Frame saved (no display): {filename}")
                    except:
                        pass
            else:
                raise
        
        if display_available:
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
                    filename = output_dir / f"plate_detection_{int(time.time())}.jpg"
                    cv2.imwrite(str(filename), annotated)
                    print(f"✓ Frame saved: {filename}")
                except Exception as e:
                    print(f"✗ Error saving frame: {e}")
            elif key == ord('r'):
                # Reset detection counter
                detection_count = 0
                plate_texts = []
                print("✓ Detection counter reset")
            elif key == ord('t'):
                # Show detected plate texts
                print("\n" + "-" * 60)
                print("Detected License Plates:")
                print("-" * 60)
                if plate_texts:
                    for i, text in enumerate(plate_texts, 1):
                        print(f"{i}. {text}")
                else:
                    print("No plates read yet")
                print("-" * 60 + "\n")
        else:
            # No display available - print status and check for keyboard interrupt
            if detected_plates:
                plates_with_text = [p['text'] for p in detected_plates if p['text']]
                if plates_with_text:
                    print(f"Plates detected: {', '.join(plates_with_text)}")
            
            # Small delay to prevent CPU overload
            time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\nInterrupted by user")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Cleanup
    cap.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass  # Ignore if windows not available
    
    # Session summary
    session_duration = time.time() - fps_start_time
    print("\n" + "=" * 60)
    print("Session Summary:")
    print("=" * 60)
    print(f"  Total Plates Detected: {detection_count}")
    print(f"  Unique Plate Texts: {len(plate_texts)}")
    print(f"  Average FPS: {current_fps:.1f}")
    
    if plate_texts:
        print("\n  Detected License Plates:")
        for i, text in enumerate(plate_texts, 1):
            print(f"    {i}. {text}")
    
    if session_duration > 0:
        detection_rate = detection_count / session_duration
        print(f"\n  Detection Rate: {detection_rate:.2f} plates/second")
    
    print("=" * 60)
    print("Camera released. Goodbye!")
