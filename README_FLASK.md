# Flask Web Application - Smart Intrusion Detection System

A unified web interface that combines all detection systems in one frame.

## Features

- 🌐 **Web Interface**: Access all detections through a browser
- 📹 **Live Video Stream**: Real-time video feed with all detections
- 📊 **Real-time Statistics**: Live updates of detection counts and status
- 🎨 **Modern UI**: Beautiful, responsive design
- 🔔 **Alert System**: Visual alerts when threats are detected
- 📱 **Responsive**: Works on desktop and mobile devices

## Installation

1. Install Flask:
```bash
pip install flask
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

Or access from another device on your network:
```
http://YOUR_IP_ADDRESS:5000
```

## Features

### Detection Systems Integrated

1. **Fire Detection** 🔥
   - Detects fire and smoke
   - Red alert when fire detected

2. **Weapon Detection** 🔫
   - Detects knives and pistols
   - Shows weapon types detected

3. **Human Detection** 👤
   - Detects people/persons
   - Counts human detections

4. **Animal Detection** 🐾
   - Detects multiple animal types
   - Shows animal types detected

5. **Vehicle Detection** 🚗
   - Detects cars, trucks, buses, etc.
   - Shows vehicle types detected

6. **Number Plate Detection** 🚗
   - Detects and reads license plates
   - Shows detected plate texts

### Web Interface Controls

- **Start Detection**: Begin real-time detection
- **Stop Detection**: Pause detection
- **Reset Stats**: Clear all detection counters

### Statistics Panel

- Real-time detection counts
- Status indicators (green = safe, red = threat)
- Detection types and details
- Auto-updates every second

## API Endpoints

- `GET /` - Main web interface
- `GET /video_feed` - Video stream endpoint
- `GET /stats` - Get detection statistics (JSON)
- `GET /start` - Start detection
- `GET /stop` - Stop detection
- `GET /reset` - Reset statistics

## Configuration

The application automatically:
- Loads available models
- Initializes OCR if EasyOCR is installed
- Connects to default camera (index 0)

## Troubleshooting

### Camera not working
- Check if camera is available: `cv2.VideoCapture(0)`
- Try different camera index in `app.py`

### Models not loading
- Ensure model files exist in correct paths
- Check console output for model loading status

### Performance issues
- Reduce frame resolution in `generate_frames()`
- Lower confidence thresholds
- Use smaller YOLO models (yolov8n instead of yolov8m)

## Notes

- The application runs all detections on each frame
- Detection statistics update in real-time
- Video stream uses MJPEG format for browser compatibility
- All detections are drawn on the same video frame
