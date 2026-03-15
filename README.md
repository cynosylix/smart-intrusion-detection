# Smart Intrusion Detection System

Real-time detection systems for home safety including fire/smoke detection, weapon detection, human detection, animal detection, vehicle detection, and number plate detection with OCR using YOLO models with webcam support.

## Features

### Fire Detection System
- 🔥 **Real-time Detection**: Live fire and smoke detection from webcam feed
- ⚡ **High Performance**: Optimized YOLOv8 model for fast inference
- 🎯 **Accurate**: Pre-trained model fine-tuned on fire and smoke dataset

### Weapon Detection System
- 🔫 **Real-time Detection**: Live weapon detection (Knife and Pistol) from webcam feed
- ⚡ **High Performance**: Optimized YOLOv5 model for fast inference
- 🎯 **Accurate**: Pre-trained model with 91% mAP@0.5 accuracy
- ⚠️ **Alert System**: Visual and audio alerts when weapons are detected

### Human Detection System
- 👤 **Real-time Detection**: Live human/person detection from webcam feed
- ⚡ **High Performance**: YOLOv8 pretrained model (downloads automatically)
- 🎯 **Accurate**: COCO pretrained model with high accuracy
- 📊 **Statistics**: Detection counter and FPS monitoring

### Animal Detection System
- 🐾 **Real-time Detection**: Live animal detection from webcam feed
- ⚡ **High Performance**: YOLOv8 pretrained model (downloads automatically)
- 🎯 **Accurate**: Detects multiple animal types (dog, cat, bird, horse, etc.)
- 📊 **Statistics**: Detection counter by animal type and FPS monitoring

### Vehicle Detection System
- 🚗 **Real-time Detection**: Live vehicle detection from webcam feed
- ⚡ **High Performance**: YOLOv8 pretrained model (downloads automatically)
- 🎯 **Accurate**: Detects multiple vehicle types (car, truck, bus, motorcycle, bicycle, etc.)
- 📊 **Statistics**: Detection counter by vehicle type and FPS monitoring

### Number Plate Detection & Recognition System
- 🚗 **Real-time Detection**: Live license plate detection from webcam feed
- 📝 **OCR Recognition**: Automatic text reading from detected plates
- ⚡ **High Performance**: YOLOv8 + EasyOCR/Tesseract OCR
- 🎯 **Accurate**: Edge detection + vehicle-based plate detection
- 📊 **Statistics**: Tracks detected plate texts and detection counts

### Common Features
- 📊 **Visual Feedback**: Color-coded bounding boxes and warning alerts
- 📈 **Statistics**: Detection counter and FPS display
- 💾 **Frame Saving**: Save detected frames with a single keypress
- 📝 **Alert History**: Track detection history

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Fire Detection
Run the live fire detection:
```bash
python live_fire_detection.py
```
The script will automatically load the model from `fire-and-smoke-detection-yolov8/weights/best.pt`

### Weapon Detection
Run the live weapon detection:
```bash
python live_weapon_detection.py
```
The script will automatically load the model from `realtime-weapon-detection/best.pt`

### Human Detection
Run the live human detection:
```bash
python live_human_detection.py
```
The script uses YOLOv8 pretrained model (downloads automatically on first run)

### Animal Detection
Run the live animal detection:
```bash
python live_animal_detection.py
```
The script uses YOLOv8 pretrained model (downloads automatically on first run)

### Vehicle Detection
Run the live vehicle detection:
```bash
python live_vehicle_detection.py
```
The script uses YOLOv8 pretrained model (downloads automatically on first run)

### Number Plate Detection & Recognition
Run the live number plate detection:
```bash
python live_number_plate_detection.py
```
The script uses YOLOv8 for detection and EasyOCR/Tesseract for text recognition

## Usage

### Fire Detection

#### Basic Usage
```bash
python live_fire_detection.py
```

#### Advanced Options
```bash
python live_fire_detection.py --camera 1 --conf 0.3 --model path/to/model.pt
```

**Arguments:**
- `--model`: Path to YOLOv8 model weights (default: `fire-and-smoke-detection-yolov8/weights/best.pt`)
- `--camera`: Camera device index (default: 0)
- `--conf`: Confidence threshold for detection (default: 0.25)

**Controls:**
- **Press 'q'**: Quit the application
- **Press 's'**: Save current frame as image
- **Press 'r'**: Reset detection counter

### Weapon Detection

#### Basic Usage
```bash
python live_weapon_detection.py
```

#### Advanced Options
```bash
python live_weapon_detection.py --camera 1 --conf 0.3 --model path/to/model.pt
```

**Arguments:**
- `--model`: Path to YOLOv5 model weights (default: `Weapon-Detection-YOLOv5/yolov5s.pt`)
- `--camera`: Camera device index (default: 0)
- `--conf`: Confidence threshold for detection (default: 0.25)

**Controls:**
- **Press 'q'**: Quit the application
- **Press 's'**: Save current frame as image
- **Press 'r'**: Reset detection counter and alert history
- **Press 'a'**: Show alert history (last 10 alerts)

### Human Detection

#### Basic Usage
```bash
python live_human_detection.py
```

**Features:**
- Uses YOLOv8 pretrained model (no model file needed)
- Detects humans/persons in real-time
- Visual alerts when humans are detected
- Detection counter and FPS display

**Controls:**
- **Press 'q'**: Quit the application
- **Press 's'**: Save current frame as image
- **Press 'r'**: Reset detection counter

### Animal Detection

#### Basic Usage
```bash
python live_animal_detection.py
```

**Features:**
- Uses YOLOv8 pretrained model (no model file needed)
- Detects multiple animal types: dog, cat, bird, horse, sheep, cow, elephant, bear, zebra, giraffe
- Visual alerts when animals are detected
- Detection counter by animal type
- FPS display

**Controls:**
- **Press 'q'**: Quit the application
- **Press 's'**: Save current frame as image
- **Press 'r'**: Reset detection counter
- **Press 'i'**: Show detection statistics

### Vehicle Detection

#### Basic Usage
```bash
python live_vehicle_detection.py
```

**Features:**
- Uses YOLOv8 pretrained model (no model file needed)
- Detects multiple vehicle types: car, truck, bus, motorcycle, bicycle, train, boat, airplane
- Visual alerts when vehicles are detected
- Detection counter by vehicle type
- FPS display

**Controls:**
- **Press 'q'**: Quit the application
- **Press 's'**: Save current frame as image
- **Press 'r'**: Reset detection counter
- **Press 'i'**: Show detection statistics

### Number Plate Detection & Recognition

#### Basic Usage
```bash
python live_number_plate_detection.py
```

**Features:**
- Uses YOLOv8 for vehicle detection
- Edge detection for license plate localization
- EasyOCR or Tesseract for text recognition
- Real-time plate detection and text reading
- Tracks unique plate texts

**Controls:**
- **Press 'q'**: Quit the application
- **Press 's'**: Save current frame as image
- **Press 'r'**: Reset detection counter
- **Press 't'**: Show detected plate texts

**Installation:**
For OCR functionality, install EasyOCR (recommended):
```bash
pip install easyocr
```

Or install Tesseract OCR:
```bash
pip install pytesseract
# Also install Tesseract binary from: https://github.com/tesseract-ocr/tesseract

## Model Information

### Fire Detection Model
- **Architecture**: YOLOv8n (Nano) - optimized for speed and efficiency
- **Classes**: Fire, Smoke
- **Input Size**: 640x640 pixels
- **Trained On**: Custom fire and smoke detection dataset

### Weapon Detection Model
- **Architecture**: YOLOv5/YOLOv8 - balanced speed and accuracy
- **Classes**: Knife, Pistol
- **Input Size**: 640x640 pixels
- **Accuracy**: 91% mAP@0.5 (92% Precision, 84% Recall)
- **Trained On**: OD-WeaponDetection dataset

### Human Detection Model
- **Architecture**: YOLOv8n (Nano) - optimized for speed
- **Classes**: Person (from COCO dataset)
- **Input Size**: 640x640 pixels
- **Model**: Pretrained on COCO dataset (80 classes, person class ID: 0)
- **Download**: Automatically downloads on first run

### Animal Detection Model
- **Architecture**: YOLOv8n (Nano) - optimized for speed
- **Classes**: Multiple animals (dog, cat, bird, horse, sheep, cow, elephant, bear, zebra, giraffe)
- **Input Size**: 640x640 pixels
- **Model**: Pretrained on COCO dataset (80 classes)
- **Download**: Automatically downloads on first run

### Vehicle Detection Model
- **Architecture**: YOLOv8n (Nano) - optimized for speed
- **Classes**: Multiple vehicles (car, truck, bus, motorcycle, bicycle, train, boat, airplane)
- **Input Size**: 640x640 pixels
- **Model**: Pretrained on COCO dataset (80 classes)
- **Download**: Automatically downloads on first run

### Number Plate Detection Model
- **Detection**: YOLOv8n for vehicle detection + Edge detection for plate localization
- **OCR Engine**: EasyOCR (recommended) or Tesseract OCR
- **Method**: Vehicle-based detection + contour analysis
- **Text Recognition**: Automatic OCR from detected plate regions
- **Preprocessing**: Grayscale conversion, thresholding, denoising for better OCR accuracy

## Visual Features

### Fire Detection
- **Red bounding boxes**: Fire detections
- **Gray bounding boxes**: Smoke detections
- **Red border**: Warning alert when fire is detected
- **Status bar**: Shows detection status and counts
- **FPS counter**: Real-time performance indicator

### Weapon Detection
- **Orange bounding boxes**: Knife detections
- **Blue bounding boxes**: Pistol detections
- **Red border**: Warning alert when weapons are detected
- **Status bar**: Shows detection status, counts, and recent alerts
- **FPS counter**: Real-time performance indicator
- **Alert history**: Track recent detections

### Human Detection
- **Green bounding boxes**: Human/person detections
- **Green border**: Visual alert when humans are detected
- **Status bar**: Shows detection status and counts
- **FPS counter**: Real-time performance indicator
- **Detection counter**: Total humans detected in session

### Animal Detection
- **Cyan/Yellow bounding boxes**: Animal detections (varies by animal type)
- **Cyan border**: Visual alert when animals are detected
- **Status bar**: Shows detection status, animal types, and counts
- **FPS counter**: Real-time performance indicator
- **Detection counter**: Total detections by animal type
- **Statistics**: Press 'i' to view detailed detection statistics

### Vehicle Detection
- **Orange bounding boxes**: Vehicle detections (varies by vehicle type)
- **Orange border**: Visual alert when vehicles are detected
- **Status bar**: Shows detection status, vehicle types, and counts
- **FPS counter**: Real-time performance indicator
- **Detection counter**: Total detections by vehicle type
- **Statistics**: Press 'i' to view detailed detection statistics

### Number Plate Detection
- **Green bounding boxes**: License plates with text successfully read
- **Yellow bounding boxes**: License plates detected (text reading in progress/failed)
- **Green border**: Visual alert when plates are detected
- **Text labels**: Shows detected plate numbers above bounding boxes
- **Status bar**: Shows detection status, plate texts, and counts
- **FPS counter**: Real-time performance indicator
- **Plate tracking**: Tracks unique plate texts detected
- **Statistics**: Press 't' to view all detected plate texts

## Performance

- Optimized for real-time inference
- Supports GPU acceleration (if available)
- Frame skipping for better performance on slower systems

## Troubleshooting

### Camera not opening
- Try different camera indices: `--camera 1` or `--camera 2`
- Check if camera is being used by another application

### Model not found
- Ensure the model file exists at: `fire-and-smoke-detection-yolov8/weights/best.pt`
- Or specify custom path with `--model` argument

### Low FPS
- Lower the confidence threshold: `--conf 0.3`
- Close other applications using the camera
- Use GPU if available (CUDA)

## Requirements

- Python 3.7+
- Ultralytics YOLOv8
- OpenCV
- PyTorch
- Webcam/Camera

## Notes

### Fire Detection
- The model detects both fire and smoke
- Confidence threshold can be adjusted for sensitivity
- Detection counter tracks total detections in the session
- Saved frames include all annotations and detections

### Weapon Detection
- The model detects knives and pistols
- Alert history tracks recent detections (last 5 seconds)
- Visual alerts are displayed when weapons are detected
- Confidence threshold can be adjusted for sensitivity
- Detection counter tracks total detections per weapon type
