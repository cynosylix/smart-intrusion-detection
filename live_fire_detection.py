"""
Live Fire Detection using YOLOv8
Real-time fire and smoke detection from webcam feed
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
from pathlib import Path

class LiveFireDetector:
    """
    Real-time fire and smoke detection using YOLOv8
    """
    def __init__(self, model_path, camera_index=0, conf_threshold=0.25):
        """
        Initialize the live fire detector
        
        Args:
            model_path: Path to YOLOv8 model weights (.pt file)
            camera_index: Camera device index (default: 0)
            conf_threshold: Confidence threshold for detection (default: 0.25)
        """
        print("=" * 60)
        print("Initializing Fire Detection System...")
        print("=" * 60)
        
        # Load YOLOv8 model
        print(f"Loading model from: {model_path}")
        try:
            self.model = YOLO(model_path)
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
        
        self.camera_index = camera_index
        self.conf_threshold = conf_threshold
        self.cap = None
        
        # Detection classes (Fire and Smoke)
        self.classes = {0: 'Fire', 1: 'Smoke'}
        
        # Colors for bounding boxes
        self.colors = {
            'Fire': (0, 0, 255),      # Red for fire
            'Smoke': (128, 128, 128)  # Gray for smoke
        }
        
        # Statistics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.detection_count = {'Fire': 0, 'Smoke': 0}
        
    def draw_detections(self, frame, results):
        """
        Draw detection boxes and labels on frame
        
        Args:
            frame: Input frame (BGR)
            results: YOLOv8 detection results
            
        Returns:
            Annotated frame
        """
        h, w = frame.shape[:2]
        fire_detected = False
        smoke_detected = False
        
        # Process detections
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class and confidence
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                # Filter by confidence threshold
                if conf < self.conf_threshold:
                    continue
                
                # Get class name
                class_name = self.classes.get(cls_id, 'Unknown')
                color = self.colors.get(class_name, (255, 255, 255))
                
                # Update detection count
                if class_name in self.detection_count:
                    self.detection_count[class_name] += 1
                    if class_name == 'Fire':
                        fire_detected = True
                    elif class_name == 'Smoke':
                        smoke_detected = True
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                label = f"{class_name} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = max(y1, label_size[1] + 10)
                
                cv2.rectangle(frame, (x1, label_y - label_size[1] - 5), 
                            (x1 + label_size[0], label_y + 5), color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, label_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw warning overlay if fire detected
        if fire_detected:
            # Red border
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
            
            # Warning text
            warning_text = "⚠️ FIRE DETECTED! ⚠️"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(warning_text, font, 1.2, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h - 30
            
            # Text background
            cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10),
                         (text_x + text_size[0] + 10, text_y + 10), (0, 0, 255), -1)
            
            # Warning text
            cv2.putText(frame, warning_text, (text_x, text_y), font, 1.2, 
                       (255, 255, 255), 3)
        
        # Draw status bar at top
        status_bar_height = 60
        cv2.rectangle(frame, (0, 0), (w, status_bar_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, status_bar_height), (255, 255, 255), 2)
        
        # Status text
        status_text = "Fire Detection Active"
        if fire_detected:
            status_text = "⚠️ FIRE ALERT!"
        elif smoke_detected:
            status_text = "Smoke Detected"
        
        cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # Detection counts
        count_text = f"Fire: {self.detection_count['Fire']} | Smoke: {self.detection_count['Smoke']}"
        cv2.putText(frame, count_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (200, 200, 200), 1)
        
        return frame
    
    def draw_fps(self, frame):
        """
        Draw FPS counter on frame
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with FPS counter
        """
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            elapsed = time.time() - self.fps_start_time
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = time.time()
        
        fps_text = f"FPS: {self.current_fps:.1f}"
        h, w = frame.shape[:2]
        cv2.putText(frame, fps_text, (w - 120, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        """
        Start live fire detection from webcam
        """
        # Open webcam
        print(f"\nOpening camera {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"✗ Error: Could not open camera {self.camera_index}")
            print("Try changing camera_index (e.g., --camera 1)")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✓ Camera opened successfully!")
        print("\n" + "=" * 60)
        print("Fire Detection System Started")
        print("=" * 60)
        print("Controls:")
        print("  Press 'q' to quit")
        print("  Press 's' to save current frame")
        print("  Press 'r' to reset detection counter")
        print("=" * 60 + "\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("✗ Error: Failed to read frame from camera")
                    break
                
                # Run inference
                results = self.model(frame, conf=self.conf_threshold, verbose=False)
                
                # Draw detections
                frame = self.draw_detections(frame, results)
                
                # Draw FPS
                frame = self.draw_fps(frame)
                
                # Display frame
                cv2.imshow('Live Fire Detection System', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nStopping detection...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f"fire_detection_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"✓ Frame saved: {filename}")
                elif key == ord('r'):
                    # Reset detection counter
                    self.detection_count = {'Fire': 0, 'Smoke': 0}
                    print("✓ Detection counter reset")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "=" * 60)
            print("Session Summary:")
            print(f"  Total Fire Detections: {self.detection_count['Fire']}")
            print(f"  Total Smoke Detections: {self.detection_count['Smoke']}")
            print("=" * 60)
            print("Camera released. Goodbye!")


def main():
    """
    Main function for command-line usage
    """
    parser = argparse.ArgumentParser(
        description='Live Fire Detection using YOLOv8'
    )
    
    # Default model path
    default_model = Path("fire-and-smoke-detection-yolov8/weights/best.pt")
    
    parser.add_argument('--model', type=str, 
                       default=str(default_model),
                       help=f'Path to YOLOv8 model weights (default: {default_model})')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index (default: 0)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    
    args = parser.parse_args()
    
    # Check if model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"✗ Error: Model file not found: {model_path}")
        print(f"Please provide a valid path to the model file.")
        return
    
    # Initialize and run detector
    detector = LiveFireDetector(
        model_path=str(model_path),
        camera_index=args.camera,
        conf_threshold=args.conf
    )
    detector.run()


if __name__ == '__main__':
    main()
