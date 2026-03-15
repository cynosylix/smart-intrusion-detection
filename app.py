"""
Flask Web Application for Smart Intrusion Detection System
Combines all detection systems (Fire, Weapon, Human, Animal, Vehicle, Number Plate) in one interface
Includes user authentication (Login/Registration)
"""

from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash, session, send_file
from werkzeug.utils import secure_filename
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path
import threading
import queue
import sys
from threading import Lock
import sqlite3
from functools import wraps
import os
import pickle
import json
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from PIL import Image

# Face recognition library
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠ Warning: face_recognition library not available. Install with: pip install face-recognition")

# DeepFace face recognition module (Facenet512 embeddings) - for upload only
from face_module import (
    FaceConfig,
    ensure_dirs,
    detect_face_haar,
    crop_rgb,
    embedding_facenet512,
    save_embedding,
    load_all_embeddings,
    match_embedding,
)


app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'  # Change this in production!

# Email notification configuration
EMAIL_CONFIG = {
    'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'smtp_port': int(os.getenv('SMTP_PORT', 587)),
    'smtp_username': os.getenv('SMTP_USERNAME', 'smartintrusion@gmail.com'),
    'smtp_password': os.getenv('SMTP_PASSWORD', 'tjrc aeqz wifd gzns'),
    'from_email': os.getenv('FROM_EMAIL', 'smartintrusion@gmail.com'),
    'to_email': os.getenv('TO_EMAIL', 'smartintrusion@gmail.com')  # Default recipient, can be overridden per user
}


# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Database setup
def init_db():
    """Initialize SQLite database for users and detection history"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS detection_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  detection_type TEXT NOT NULL,
                  frame_path TEXT NOT NULL,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  details TEXT,
                  user_id INTEGER,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_faces
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  image_path TEXT NOT NULL,
                  face_embedding BLOB NOT NULL,
                  face_location TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

init_db()

# Create detection history folder
DETECTION_HISTORY_FOLDER = 'detection_history'
os.makedirs(DETECTION_HISTORY_FOLDER, exist_ok=True)

# Create uploaded videos folder
UPLOAD_VIDEO_FOLDER = 'uploaded_videos'
os.makedirs(UPLOAD_VIDEO_FOLDER, exist_ok=True)

# Create processed videos folder
PROCESSED_VIDEO_FOLDER = 'processed_videos'
os.makedirs(PROCESSED_VIDEO_FOLDER, exist_ok=True)

# Create user faces folder
USER_FACES_FOLDER = 'user_faces'
os.makedirs(USER_FACES_FOLDER, exist_ok=True)

# Create embeddings folder (DeepFace)
EMBEDDINGS_FOLDER = 'embeddings'
os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)

# Face recognition config (DeepFace Facenet512)
FACE_CFG = FaceConfig(embeddings_dir=EMBEDDINGS_FOLDER)

# Allowed video extensions
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'}

# Allowed image extensions for face upload
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_video_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def allowed_image_file(filename):
    """Check if image file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def load_stored_face_embeddings():
    """
    Load face encodings from image files using face_recognition library
    Returns: (known_face_encodings, known_face_names)
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return [], []
    
    known_face_encodings = []
    known_face_names = []
    
    # Image files and their corresponding names (same as live_face_recognition.py)
    known_face_images = [
        "mmmm.jpeg",
        "ss.jpg"
    ]
    known_face_names_list = ["shinat", "amal"]
    
    for img_file, name in zip(known_face_images, known_face_names_list):
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
            print(f"⚠ Image file not found: {img_file}")
    
    return known_face_encodings, known_face_names


# LBPH Face Recognition setup
_lbph_recognizer = None
_lbph_label_map = {}
_lbph_face_cascade = None
_lbph_trained = False
_lbph_cache_time = 0

def load_and_train_lbph_recognizer():
    """
    Load face datasets from face_dataset/ folder and train LBPH recognizer
    Returns: (recognizer, label_map, face_cascade) or (None, None, None) if no data
    """
    global _lbph_recognizer, _lbph_label_map, _lbph_face_cascade, _lbph_trained
    
    try:
        dataset_path = "./face_dataset/"
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path, exist_ok=True)
            return None, None, None
        
        # Load Haar Cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
        if not os.path.exists(cascade_path):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print("⚠ Could not load Haar Cascade classifier")
            return None, None, None
        
        # Check if dataset files exist
        dataset_files = [f for f in os.listdir(dataset_path) if f.endswith("_faces.npy")]
        if len(dataset_files) == 0:
            print("⚠ No face datasets found in face_dataset/ folder")
            return None, None, None
        
        # Load datasets
        faces = []
        labels = []
        label_map = {}
        label_id = 0
        
        for file in dataset_files:
            name = file.replace("_faces.npy", "")
            label_map[label_id] = name
            
            data = np.load(os.path.join(dataset_path, file))
            for img in data:
                faces.append(img)
                labels.append(label_id)
            
            label_id += 1
        
        if len(faces) == 0:
            print("⚠ No face images found in datasets")
            return None, None, None
        
        faces = np.array(faces)
        labels = np.array(labels)
        
        # Create and train LBPH recognizer
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8)
        except AttributeError:
            print("❌ cv2.face module not available. Please install opencv-contrib-python:")
            print("   pip install opencv-contrib-python")
            return None, None, None
        
        recognizer.train(faces, labels)
        
        print(f"✓ LBPH recognizer trained with {len(faces)} faces from {len(label_map)} persons")
        print(f"  Persons: {list(label_map.values())}")
        
        _lbph_recognizer = recognizer
        _lbph_label_map = label_map
        _lbph_face_cascade = face_cascade
        _lbph_trained = True
        _lbph_cache_time = time.time()
        
        return recognizer, label_map, face_cascade
        
    except Exception as e:
        print(f"❌ Error loading/training LBPH recognizer: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def recognize_person_in_frame(frame_bgr, person_box, known_encodings=None, known_names=None):
    """
    Given a YOLO person box (x1,y1,x2,y2) on a BGR frame:
    - crop person ROI
    - detect face using Haar Cascade
    - recognize using LBPH recognizer
    Returns: (name_or_None, confidence)
    """
    global _lbph_recognizer, _lbph_label_map, _lbph_face_cascade, _lbph_trained, _lbph_cache_time
    
    try:
        # Load/train recognizer if not already done (or refresh every 60 seconds)
        if (_lbph_recognizer is None or not _lbph_trained or 
            (time.time() - _lbph_cache_time > 60)):
            recognizer, label_map, face_cascade = load_and_train_lbph_recognizer()
            if recognizer is None:
                return None, float("inf")
    
        x1, y1, x2, y2 = map(int, person_box)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_bgr.shape[1], x2)
        y2 = min(frame_bgr.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None, float("inf")

        # Crop person ROI
        roi_bgr = frame_bgr[y1:y2, x1:x2]
        if roi_bgr.size == 0:
            return None, float("inf")

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using Haar Cascade
        faces_detected = _lbph_face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces_detected) == 0:
            return None, float("inf")
        
        # Use the first detected face
        (fx, fy, fw, fh) = faces_detected[0]
        face = gray[fy:fy+fh, fx:fx+fw]
        face = cv2.resize(face, (200, 200))
        
        # Recognize using LBPH
        label, confidence = _lbph_recognizer.predict(face)
        
        THRESHOLD = 65  # Lower = stricter
        
        if confidence > THRESHOLD:
            return None, confidence  # Unknown person
        else:
            name = _lbph_label_map.get(label, "Unknown")
            return name, confidence
        
    except Exception as e:
        print(f"recognize_person_in_frame error: {e}")
        import traceback
        traceback.print_exc()
        return None, float("inf")

def send_detection_email(detection_type, filepath, details=None, user_email=None):
    """Send email notification for detection"""
    global email_notifications_enabled, EMAIL_CONFIG
    
    if not email_notifications_enabled:
        return False
    
    try:
        # Get recipient email
        recipient_email = user_email if user_email else EMAIL_CONFIG.get('to_email')
        if not recipient_email or not EMAIL_CONFIG.get('smtp_username') or not EMAIL_CONFIG.get('smtp_password'):
            print("⚠ Email not configured or recipient not set")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG.get('from_email', EMAIL_CONFIG.get('smtp_username'))
        msg['To'] = recipient_email
        msg['Subject'] = f"🚨 Detection Alert: {detection_type.upper()} Detected"
        
        # Build email body
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: #f44336;">🚨 Detection Alert</h2>
            <p><strong>Detection Type:</strong> {detection_type.upper()}</p>
            <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        if details:
            if 'types' in details:
                body += f"<p><strong>Detected Types:</strong> {', '.join(details['types'])}</p>"
            elif 'texts' in details:
                body += f"<p><strong>Detected Plates:</strong> {', '.join(details['texts'])}</p>"
            elif 'count' in details:
                body += f"<p><strong>Count:</strong> {details['count']}</p>"
        
        body += """
            <p><strong>Note:</strong> Please check the detection history for the captured frame.</p>
            <hr>
            <p style="color: #666; font-size: 0.9em;">This is an automated alert from Smart Intrusion Detection System.</p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Attach image if file exists
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    img_data = f.read()
                image = MIMEImage(img_data)
                image.add_header('Content-Disposition', 'attachment', filename=os.path.basename(filepath))
                msg.attach(image)
            except Exception as e:
                print(f"⚠ Could not attach image to email: {e}")
        
        # Send email
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['smtp_username'], EMAIL_CONFIG['smtp_password'])
        server.send_message(msg)
        server.quit()
        
        print(f"✓ Email notification sent to {recipient_email}")
        return True
    except Exception as e:
        print(f"✗ Error sending email: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_detection_frame(frame, detection_type, details=None, user_id=None, user_email=None):
    """Save detected frame with timestamp to history and send email if enabled"""
    try:
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        filename = f"{detection_type}_{timestamp_str}.jpg"
        filepath = os.path.join(DETECTION_HISTORY_FOLDER, filename)
        
        # Save the frame
        cv2.imwrite(filepath, frame)
        
        # Save to database
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # Prepare details JSON
        details_json = None
        if details:
            details_json = json.dumps(details)
        
        c.execute('''INSERT INTO detection_history 
                     (detection_type, frame_path, timestamp, details, user_id)
                     VALUES (?, ?, ?, ?, ?)''',
                  (detection_type, filepath, timestamp, details_json, user_id))
        conn.commit()
        conn.close()
        
        print(f"✓ Saved detection frame: {filepath}")
        
        # Send email notification if enabled
        if email_notifications_enabled:
            send_detection_email(detection_type, filepath, details, user_email)
        
        return filepath
    except Exception as e:
        print(f"✗ Error saving detection frame: {e}")
        import traceback
        traceback.print_exc()
        return None

class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT id, username, email FROM users WHERE id = ?', (user_id,))
    user_data = c.fetchone()
    conn.close()
    if user_data:
        return User(user_data[0], user_data[1], user_data[2])
    return None

# Patch cv2 if needed
if not hasattr(cv2, 'imshow'):
    def dummy_imshow(*args, **kwargs):
        pass
    cv2.imshow = dummy_imshow
    cv2.destroyAllWindows = lambda: None
    cv2.waitKey = lambda *args: -1

# Global variables
camera = None
detection_active = False
detection_thread = None
frame_queue = queue.Queue(maxsize=2)
stats_lock = Lock()
detection_stats = {
    'fire': {'count': 0, 'detected': False},
    'weapon': {'count': 0, 'detected': False, 'types': []},
    'human': {'count': 0, 'detected': False},
    'animal': {'count': 0, 'detected': False, 'types': []},
    'vehicle': {'count': 0, 'detected': False, 'types': []},
    'plate': {'count': 0, 'detected': False, 'texts': []}
}
# Track previous detection state to save only new detections
previous_detection_state = {
    'fire': False,
    'weapon': False,
    'human': False,
    'animal': False,
    'vehicle': False,
    'plate': False
}

# Detection enable/disable flags
detection_enabled = {
    'fire': False,
    'weapon': False,
    'human': False,
    'animal': False,
    'vehicle': False,
    'plate': False
}

# Email notification toggle (default OFF)
email_notifications_enabled = False

# Live image collection state
live_collection_active = False
live_collection_camera = None
live_collection_person_name = None
live_collection_face_images = []
live_collection_count = 0
live_collection_lock = Lock()
live_collection_max_samples = 120

# Load models
models = {}
print("Loading detection models...")

# Fire detection model
try:
    fire_model_path = Path("fire-and-smoke-detection-yolov8/weights/best.pt")
    if fire_model_path.exists():
        models['fire'] = YOLO(str(fire_model_path))
        print("✓ Fire detection model loaded")
except Exception as e:
    print(f"⚠ Fire model not available: {e}")

# Weapon detection model
try:
    weapon_model_path = Path("realtime-weapon-detection/best.pt")
    if weapon_model_path.exists():
        models['weapon'] = YOLO(str(weapon_model_path))
        print("✓ Weapon detection model loaded")
except Exception as e:
    print(f"⚠ Weapon model not available: {e}")

# Human/Animal/Vehicle/Plate detection (use YOLOv8 pretrained)
try:
    models['general'] = YOLO("yolov8n.pt")
    print("✓ General detection model loaded (Human, Animal, Vehicle)")
except Exception as e:
    print(f"⚠ General model not available: {e}")

# OCR reader for number plates (same as live_number_plate_detection.py)
ocr_reader = None
ocr_engine = None
try:
    import easyocr
    ocr_engine = "easyocr"
    print("Initializing EasyOCR...")
    ocr_reader = easyocr.Reader(['en'], gpu=False)  # English only, CPU mode
    print("✓ EasyOCR initialized!")
except ImportError:
    try:
        import pytesseract
        ocr_engine = "tesseract"
        print("✓ Tesseract OCR ready")
    except ImportError:
        ocr_engine = None
        print("⚠ Warning: No OCR library found. Install EasyOCR or pytesseract")
        print("  Install EasyOCR: pip install easyocr")
        print("  Install Tesseract: pip install pytesseract (requires Tesseract binary)")

if not ocr_engine:
    print("⚠ Warning: OCR not available. Only plate detection will work (no text reading).")

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

def read_plate_text(plate_roi, reader, engine):
    """
    Read text from license plate using OCR
    """
    if reader is None and engine != "tesseract":
        return None
    
    try:
        # Preprocess plate
        processed = preprocess_plate(plate_roi)
        
        if engine == "easyocr" and reader:
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
        
        elif engine == "tesseract":
            # Tesseract OCR
            import pytesseract
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
    Uses edge detection and vehicle-based detection
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

def detect_all(frame):
    """
    Run all detection models on a single frame
    """
    global detection_stats
    
    results = {}
    annotated_frame = frame.copy()
    
    # Local stats for this frame
    frame_stats = {
        'fire': {'detected': False, 'count': 0},
        'weapon': {'detected': False, 'types': []},
        'human': {'detected': False, 'count': 0},
        'animal': {'detected': False, 'types': []},
        'vehicle': {'detected': False, 'types': []},
        'plate': {'detected': False, 'texts': []}
    }
    
    # Fire detection
    if detection_enabled['fire'] and 'fire' in models:
        try:
            fire_results = models['fire'](frame, conf=0.25, verbose=False)
            for result in fire_results:
                if result.boxes is not None and len(result.boxes) > 0:
                    frame_stats['fire']['detected'] = True
                    frame_stats['fire']['count'] = len(result.boxes)
            results['fire'] = fire_results
        except:
            pass
    
    # Weapon detection
    if detection_enabled['weapon'] and 'weapon' in models:
        try:
            weapon_results = models['weapon'](frame, conf=0.5, verbose=False)
            weapons = []
            for result in weapon_results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = models['weapon'].names.get(cls_id, 'Unknown')
                        if cls_name.lower() in ['knife', 'pistol']:
                            if cls_name not in weapons:
                                weapons.append(cls_name)
            if weapons:
                frame_stats['weapon']['detected'] = True
                frame_stats['weapon']['types'] = weapons
            results['weapon'] = weapon_results
        except:
            pass
    
    # Human detection with face recognition
    if detection_enabled['human'] and 'general' in models:
        try:
            person_class_id = None
            for cls_id, cls_name in models['general'].names.items():
                if cls_name.lower() == 'person':
                    person_class_id = cls_id
                    break
            
            if person_class_id is not None:
                human_results = models['general'](frame, conf=0.5, verbose=False, classes=[person_class_id])
                if human_results[0].boxes is not None and len(human_results[0].boxes) > 0:
                    frame_stats['human']['detected'] = True
                    frame_stats['human']['count'] = len(human_results[0].boxes)
                    
                    # Face recognition using LBPH: Check if detected persons are known or unknown
                    try:
                        # Check each detected person
                        for box in human_results[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # Recognize person using LBPH face recognition
                            name, confidence = recognize_person_in_frame(frame, (x1, y1, x2, y2))

                            # Draw label on the right side of bounding box
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            if name:
                                print(f"[LBPH FACE RECOGNITION] Known Person: {name} (confidence={confidence:.2f})")
                                label = f"Known Person: {name}"
                                color = (0, 255, 0)  # Green for known
                            else:
                                label = "Unknown Person"
                                color = (0, 0, 255)  # Red for unknown

                            # Draw rectangle
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                            
                            # Calculate label size and position (above the box)
                            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            label_x = x1  # Position above the box
                            label_y = y1 - 10  # Above the box
                            
                            # Make sure label doesn't go off screen
                            if label_y < label_size[1]:
                                label_y = y2 + label_size[1] + 10  # Put below if no space above
                            
                            # Draw label background rectangle
                            cv2.rectangle(
                                annotated_frame,
                                (label_x - 5, label_y - label_size[1] - 5),
                                (label_x + label_size[0] + 5, label_y + baseline + 5),
                                color,
                                -1,
                            )
                            
                            # Draw label text
                            cv2.putText(
                                annotated_frame,
                                label,
                                (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 255),
                                2,
                            )
                    except Exception as e:
                        print(f"LBPH Face recognition error: {e}")
                        import traceback
                        traceback.print_exc()
                        pass
                
                results['human'] = human_results
        except:
            pass
        
    # Animal detection (using same logic as live_animal_detection.py)
    if detection_enabled['animal'] and 'general' in models:
        try:
            # Focus animals list (same as live_animal_detection.py)
            FOCUS_ANIMALS = {
                'dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe'
            }
            
            # Find animal class IDs
            animal_class_ids = []
            for cls_id, cls_name in models['general'].names.items():
                cls_lower = cls_name.lower()
                if cls_lower in FOCUS_ANIMALS:
                    animal_class_ids.append(cls_id)
                
            if animal_class_ids:
                # Run detection with animal classes filter (conf=0.5 as in script)
                animal_results = models['general'](frame, conf=0.5, verbose=False, classes=animal_class_ids)
                animals = []
                
                if animal_results[0].boxes is not None and len(animal_results[0].boxes) > 0:
                    for box in animal_results[0].boxes:
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = models['general'].names.get(cls_id, 'Unknown')
                        
                        # Double check it's an animal (safety check)
                        if cls_name.lower() in FOCUS_ANIMALS:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            conf = float(box.conf[0].cpu().numpy())
                            
                            # Draw bounding box (orange color for animals, matching the script style)
                            color = (255, 165, 0)  # Orange
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw label
                            label = f"{cls_name} {conf:.2f}"
                            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            label_y = max(y1, label_size[1] + 10)
                            
                            # Label background
                            cv2.rectangle(annotated_frame,
                                        (x1, label_y - label_size[1] - 5),
                                        (x1 + label_size[0] + 5, label_y + baseline + 5),
                                        color, -1)
                            
                            # Label text
                            cv2.putText(annotated_frame, label, (x1, label_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            if cls_name not in animals:
                                animals.append(cls_name)
                    
                    if animals:
                        frame_stats['animal']['detected'] = True
                        frame_stats['animal']['types'] = animals
                
                results['animal'] = animal_results
        except Exception as e:
            print(f"Animal detection error: {e}")
            import traceback
            traceback.print_exc()
            pass
        
        # Vehicle detection
    if detection_enabled['vehicle']:
        try:
            vehicle_classes = []
            for cls_id, cls_name in models['general'].names.items():
                if cls_name.lower() in ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train', 'boat', 'airplane']:
                        vehicle_classes.append(cls_id)
                
            if vehicle_classes:
                vehicle_results = models['general'](frame, conf=0.5, verbose=False, classes=vehicle_classes)
                vehicles = []
                if vehicle_results[0].boxes is not None:
                    for box in vehicle_results[0].boxes:
                            cls_id = int(box.cls[0].cpu().numpy())
                            cls_name = models['general'].names.get(cls_id, 'Unknown')
                            if cls_name not in vehicles:
                                vehicles.append(cls_name)
                    if vehicles:
                        frame_stats['vehicle']['detected'] = True
                        frame_stats['vehicle']['types'] = vehicles
                    results['vehicle'] = vehicle_results
        except:
            pass
    
    # Number plate detection (using EXACT logic from live_number_plate_detection.py)
    # When plate toggle is enabled, this runs the same detection as live_number_plate_detection.py
    if detection_enabled['plate'] and 'general' in models:
        try:
            # Detect license plates using edge detection and vehicle-based detection
            # This is the EXACT same logic as live_number_plate_detection.py - detect_plates_in_frame()
            plate_regions = detect_plates_in_frame(frame, models['general'])
            
            plates_detected = []
            
            # Process each detected plate region (exact same as script)
            for (x, y, w, h) in plate_regions:
                # Extract plate ROI
                plate_roi = frame[y:y+h, x:x+w]
                
                if plate_roi.size > 0:
                    # Read text from plate using OCR (same as script)
                    plate_text = None
                    if ocr_engine:
                        plate_text = read_plate_text(plate_roi, ocr_reader, ocr_engine)
                    
                    # Draw bounding box (same colors as script: green if text, yellow if not)
                    color = (0, 255, 0) if plate_text else (0, 255, 255)
                    cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw text label (exact same as script)
                    if plate_text:
                        label = f"Plate: {plate_text}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_frame, (x, y-label_size[1]-5), 
                                    (x+label_size[0], y), color, -1)
                        cv2.putText(annotated_frame, label, (x, y-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        
                        if plate_text not in plates_detected:
                            plates_detected.append(plate_text)
                    else:
                        # No text read - show "License Plate" label (same as script)
                        label = "License Plate"
                        cv2.putText(annotated_frame, label, (x, y-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Update stats (same as script logic)
            if plate_regions:
                frame_stats['plate']['detected'] = True
                frame_stats['plate']['count'] = len(plate_regions)
                if plates_detected:
                    frame_stats['plate']['texts'] = plates_detected
        except Exception as e:
            print(f"Plate detection error: {e}")
            import traceback
            traceback.print_exc()
            pass
    
    # Update global stats with thread lock and save frames for new detections
    global previous_detection_state
    with stats_lock:
        # Check for new detections and save frames
        detection_types = ['fire', 'weapon', 'human', 'animal', 'vehicle', 'plate']
        for det_type in detection_types:
            if frame_stats[det_type]['detected'] and not previous_detection_state[det_type]:
                # New detection - save frame
                details = {}
                if det_type == 'weapon' and frame_stats[det_type]['types']:
                    details['types'] = frame_stats[det_type]['types']
                elif det_type == 'animal' and frame_stats[det_type]['types']:
                    details['types'] = frame_stats[det_type]['types']
                elif det_type == 'vehicle' and frame_stats[det_type]['types']:
                    details['types'] = frame_stats[det_type]['types']
                elif det_type == 'plate' and frame_stats[det_type]['texts']:
                    details['texts'] = frame_stats[det_type]['texts']
                elif det_type in ['fire', 'human']:
                    details['count'] = frame_stats[det_type]['count']
                
                # Get user_id and email if available
                user_id = None
                user_email = None
                try:
                    if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
                        user_id = current_user.id
                        # Get user email from database
                        conn = sqlite3.connect('users.db')
                        c = conn.cursor()
                        c.execute('SELECT email FROM users WHERE id = ?', (user_id,))
                        user_data = c.fetchone()
                        conn.close()
                        if user_data:
                            user_email = user_data[0]
                except:
                    pass
                
                save_detection_frame(annotated_frame, det_type, details if details else None, user_id, user_email)
            
            # Update previous state
            previous_detection_state[det_type] = frame_stats[det_type]['detected']
        
        detection_stats['fire']['detected'] = frame_stats['fire']['detected']
        detection_stats['weapon']['detected'] = frame_stats['weapon']['detected']
        detection_stats['human']['detected'] = frame_stats['human']['detected']
        detection_stats['animal']['detected'] = frame_stats['animal']['detected']
        detection_stats['vehicle']['detected'] = frame_stats['vehicle']['detected']
        detection_stats['plate']['detected'] = frame_stats['plate']['detected']
        
        if frame_stats['fire']['detected']:
            detection_stats['fire']['count'] += frame_stats['fire']['count']
        if frame_stats['human']['detected']:
            detection_stats['human']['count'] += frame_stats['human']['count']
        if frame_stats['weapon']['detected']:
            detection_stats['weapon']['count'] += len(frame_stats['weapon']['types'])
            detection_stats['weapon']['types'] = list(set(detection_stats['weapon']['types'] + frame_stats['weapon']['types']))[-5:]
        if frame_stats['animal']['detected']:
            detection_stats['animal']['count'] += len(frame_stats['animal']['types'])
            detection_stats['animal']['types'] = list(set(detection_stats['animal']['types'] + frame_stats['animal']['types']))[-5:]
        if frame_stats['vehicle']['detected']:
            detection_stats['vehicle']['count'] += len(frame_stats['vehicle']['types'])
            detection_stats['vehicle']['types'] = list(set(detection_stats['vehicle']['types'] + frame_stats['vehicle']['types']))[-5:]
        if frame_stats['plate']['detected']:
            detection_stats['plate']['count'] += len(frame_stats['plate']['texts'])
            detection_stats['plate']['texts'] = list(set(detection_stats['plate']['texts'] + frame_stats['plate']['texts']))[:10]
    
    # Draw all detections on frame
    for detection_type, result in results.items():
        if result:
            try:
                annotated_frame = result[0].plot(img=annotated_frame)
            except:
                pass
    
    return annotated_frame

def generate_fire_frames():
    """
    Generate video frames with fire detection only
    """
    global camera, detection_active
    
    fire_camera = cv2.VideoCapture(0)
    fire_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    fire_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while detection_active:
        ret, frame = fire_camera.read()
        if not ret:
            break
        
        annotated_frame = frame.copy()
        fire_detected = False
        fire_count = 0
        
        # Fire detection only
        if 'fire' in models:
            try:
                fire_results = models['fire'](frame, conf=0.25, verbose=False)
                if fire_results and len(fire_results) > 0:
                    annotated_frame = fire_results[0].plot(img=annotated_frame)
                    if fire_results[0].boxes is not None and len(fire_results[0].boxes) > 0:
                        fire_detected = True
                        fire_count = len(fire_results[0].boxes)
            except:
                pass
        
        # Update stats
        with stats_lock:
            detection_stats['fire']['detected'] = fire_detected
            if fire_detected:
                detection_stats['fire']['count'] += fire_count
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    fire_camera.release()

def generate_weapon_frames():
    """
    Generate video frames with weapon detection only (Knife and Pistol)
    Only detects and displays Knife and Pistol weapons, filters out all other detections
    """
    global camera, detection_active
    
    weapon_camera = cv2.VideoCapture(0)
    weapon_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    weapon_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Find Knife and Pistol class IDs
    weapon_class_ids = None
    if 'weapon' in models:
        try:
            weapon_class_ids = []
            for cls_id, cls_name in models['weapon'].names.items():
                if cls_name.lower() in ['knife', 'pistol']:
                    weapon_class_ids.append(cls_id)
        except:
            pass
    
    while detection_active:
        ret, frame = weapon_camera.read()
        if not ret:
            break
        
        annotated_frame = frame.copy()
        weapon_detected = False
        weapon_types = []
        
        # Weapon detection only - ONLY detect Knife and Pistol
        if 'weapon' in models:
            try:
                # Use classes parameter to only detect Knife and Pistol
                weapon_results = models['weapon'](frame, conf=0.5, verbose=False, classes=weapon_class_ids if weapon_class_ids else None)
                
                if weapon_results and len(weapon_results) > 0:
                    result = weapon_results[0]
                    
                    # Filter and draw only Knife and Pistol detections
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            cls_id = int(box.cls[0].cpu().numpy())
                            cls_name = models['weapon'].names.get(cls_id, 'Unknown')
                            
                            # Only process Knife and Pistol
                            if cls_name.lower() in ['knife', 'pistol']:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                conf = float(box.conf[0].cpu().numpy())
                                
                                # Draw bounding box (orange for knife, red for pistol)
                                color = (0, 165, 255) if cls_name.lower() == 'knife' else (0, 0, 255)
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                                
                                # Draw label background
                                label = f"{cls_name} {conf:.2f}"
                                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                label_y = max(y1, label_size[1] + 10)
                                
                                cv2.rectangle(annotated_frame, 
                                             (x1, label_y - label_size[1] - 10), 
                                             (x1 + label_size[0], label_y + baseline), 
                                             color, -1)
                                
                                # Draw label text
                                cv2.putText(annotated_frame, label, (x1, label_y - 5), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                
                                if cls_name not in weapon_types:
                                    weapon_types.append(cls_name)
                                    weapon_detected = True
            except Exception as e:
                # If class filtering fails, filter manually
                try:
                    weapon_results = models['weapon'](frame, conf=0.5, verbose=False)
                    if weapon_results and len(weapon_results) > 0:
                        result = weapon_results[0]
                        if result.boxes is not None:
                            for box in result.boxes:
                                cls_id = int(box.cls[0].cpu().numpy())
                                cls_name = models['weapon'].names.get(cls_id, 'Unknown')
                                
                                # Only draw Knife and Pistol
                                if cls_name.lower() in ['knife', 'pistol']:
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                    conf = float(box.conf[0].cpu().numpy())
                                    
                                    color = (0, 165, 255) if cls_name.lower() == 'knife' else (0, 0, 255)
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                                    
                                    label = f"{cls_name} {conf:.2f}"
                                    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                    label_y = max(y1, label_size[1] + 10)
                                    
                                    cv2.rectangle(annotated_frame, 
                                                 (x1, label_y - label_size[1] - 10), 
                                                 (x1 + label_size[0], label_y + baseline), 
                                                 color, -1)
                                    cv2.putText(annotated_frame, label, (x1, label_y - 5), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                    
                                    if cls_name not in weapon_types:
                                        weapon_types.append(cls_name)
                                        weapon_detected = True
                except:
                    pass
        
        # Update stats
        with stats_lock:
            detection_stats['weapon']['detected'] = weapon_detected
            if weapon_detected:
                detection_stats['weapon']['count'] += len(weapon_types)
                detection_stats['weapon']['types'] = list(set(detection_stats['weapon']['types'] + weapon_types))[-5:]
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    weapon_camera.release()

def generate_human_frames():
    """
    Generate video frames with human detection only (Person class)
    Only detects and displays humans/persons, filters out all other detections
    """
    global camera, detection_active
    
    human_camera = cv2.VideoCapture(0)
    human_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    human_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Find Person class ID
    person_class_id = None
    if 'general' in models:
        try:
            for cls_id, cls_name in models['general'].names.items():
                if cls_name.lower() == 'person':
                    person_class_id = cls_id
                    break
        except:
            pass
    
    while detection_active:
        ret, frame = human_camera.read()
        if not ret:
            break
        
        annotated_frame = frame.copy()
        human_detected = False
        human_count = 0
        
        # Human detection only - ONLY detect Person class
        if 'general' in models and person_class_id is not None:
            try:
                # Only detect Person class
                human_results = models['general'](frame, conf=0.5, verbose=False, classes=[person_class_id])
                
                if human_results and len(human_results) > 0:
                    result = human_results[0]
                    
                    # Draw only Person detections - verify each box is Person
                    if result.boxes is not None and len(result.boxes) > 0:
                        # Verify each detection is actually Person before drawing
                        for box in result.boxes:
                            cls_id = int(box.cls[0].cpu().numpy())
                            cls_name = models['general'].names.get(cls_id, 'Unknown')
                            
                            # STRICT CHECK: Only process if it's Person class
                            if cls_name.lower() == 'person':
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                conf = float(box.conf[0].cpu().numpy())
                                
                                # Draw bounding box (green for person)
                                color = (0, 255, 0)
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                                
                                # Draw label background
                                label = f"Person {conf:.2f}"
                                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                label_y = max(y1, label_size[1] + 10)
                                
                                cv2.rectangle(annotated_frame, 
                                             (x1, label_y - label_size[1] - 10), 
                                             (x1 + label_size[0], label_y + baseline), 
                                             color, -1)
                                
                                # Draw label text
                                cv2.putText(annotated_frame, label, (x1, label_y - 5), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                
                                human_count += 1
                                human_detected = True
            except Exception as e:
                # Fallback: if class filtering fails, filter manually - ONLY Person
                try:
                    # Run detection without class filter first
                    all_results = models['general'](frame, conf=0.5, verbose=False)
                    if all_results and len(all_results) > 0:
                        result = all_results[0]
                        if result.boxes is not None:
                            # Filter to ONLY Person class
                            for box in result.boxes:
                                cls_id = int(box.cls[0].cpu().numpy())
                                cls_name = models['general'].names.get(cls_id, 'Unknown')
                                
                                # STRICT FILTER: Only draw if it's exactly "person"
                                if cls_name.lower() == 'person':
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                    conf = float(box.conf[0].cpu().numpy())
                                    
                                    # Draw bounding box (green for person)
                                    color = (0, 255, 0)
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                                    
                                    # Draw label background
                                    label = f"Person {conf:.2f}"
                                    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                    label_y = max(y1, label_size[1] + 10)
                                    
                                    cv2.rectangle(annotated_frame, 
                                                 (x1, label_y - label_size[1] - 10), 
                                                 (x1 + label_size[0], label_y + baseline), 
                                                 color, -1)
                                    
                                    # Draw label text
                                    cv2.putText(annotated_frame, label, (x1, label_y - 5), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                    
                                    human_count += 1
                                    human_detected = True
                except:
                    pass
        
        # Update stats
        with stats_lock:
            detection_stats['human']['detected'] = human_detected
            if human_detected:
                detection_stats['human']['count'] += human_count
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    human_camera.release()

def generate_animal_frames():
    """
    Generate video frames with animal detection only
    Only detects and displays animals, filters out all other detections
    """
    global camera, detection_active
    
    animal_camera = cv2.VideoCapture(0)
    animal_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    animal_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Define animal classes to detect - STRICTLY ONLY these animals
    # This list ensures NO humans, vehicles, or other objects are detected
    animal_classes_list = ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
    
    # Find animal class IDs - ONLY animal classes will be detected
    animal_class_ids = None
    if 'general' in models:
        try:
            animal_class_ids = []
            for cls_id, cls_name in models['general'].names.items():
                # STRICT CHECK: Only include if it's in our animal list
                if cls_name.lower() in animal_classes_list:
                    animal_class_ids.append(cls_id)
            # If no animal classes found, set to None to prevent detection
            if len(animal_class_ids) == 0:
                animal_class_ids = None
        except:
            pass
    
    while detection_active:
        ret, frame = animal_camera.read()
        if not ret:
            break
        
        annotated_frame = frame.copy()
        animal_detected = False
        animal_types = []
        
        # Animal detection only - STRICTLY ONLY detect animal classes
        # This ensures NO humans, vehicles, or other objects are detected
        # Only proceed if we have animal class IDs to detect
        if 'general' in models and animal_class_ids and len(animal_class_ids) > 0:
            try:
                # Use classes parameter to ONLY detect animal classes - no other classes will be detected
                # The model will ONLY return detections for the specified animal classes
                animal_results = models['general'](frame, conf=0.5, verbose=False, classes=animal_class_ids)
                
                if animal_results and len(animal_results) > 0:
                    result = animal_results[0]
                    
                    # Draw only Animal detections - triple verify each box is an animal
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            cls_id = int(box.cls[0].cpu().numpy())
                            cls_name = models['general'].names.get(cls_id, 'Unknown')
                            
                            # STRICT CHECK: Only process if it's an animal class
                            # This ensures no humans, vehicles, or other objects slip through
                            # Double verification: check class name is in our animal list
                            if cls_name.lower() in animal_classes_list:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                conf = float(box.conf[0].cpu().numpy())
                                
                                # Draw bounding box (orange for animals)
                                color = (255, 165, 0)  # Orange color for animals
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                                
                                # Draw label background
                                label = f"{cls_name} {conf:.2f}"
                                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                label_y = max(y1, label_size[1] + 10)
                                
                                cv2.rectangle(annotated_frame, 
                                             (x1, label_y - label_size[1] - 10), 
                                             (x1 + label_size[0], label_y + baseline), 
                                             color, -1)
                                
                                # Draw label text
                                cv2.putText(annotated_frame, label, (x1, label_y - 5), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                
                                if cls_name not in animal_types:
                                    animal_types.append(cls_name)
                                    animal_detected = True
            except Exception as e:
                # Fallback: if class filtering fails, filter manually - STRICTLY ONLY Animals
                # This ensures that even if model-level filtering fails, we still only show animals
                try:
                    all_results = models['general'](frame, conf=0.5, verbose=False)
                    if all_results and len(all_results) > 0:
                        result = all_results[0]
                        if result.boxes is not None:
                            # Filter to STRICTLY ONLY animal classes - ignore everything else
                            for box in result.boxes:
                                cls_id = int(box.cls[0].cpu().numpy())
                                cls_name = models['general'].names.get(cls_id, 'Unknown')
                                
                                # STRICT FILTER: Only draw if it's an animal
                                # This ensures NO humans, vehicles, or other objects are displayed
                                if cls_name.lower() in animal_classes_list:
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                    conf = float(box.conf[0].cpu().numpy())
                                    
                                    color = (255, 165, 0)  # Orange color
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                                    
                                    label = f"{cls_name} {conf:.2f}"
                                    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                    label_y = max(y1, label_size[1] + 10)
                                    
                                    cv2.rectangle(annotated_frame, 
                                                 (x1, label_y - label_size[1] - 10), 
                                                 (x1 + label_size[0], label_y + baseline), 
                                                 color, -1)
                                    cv2.putText(annotated_frame, label, (x1, label_y - 5), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                    
                                    if cls_name not in animal_types:
                                        animal_types.append(cls_name)
                                        animal_detected = True
                except:
                    pass
        
        # Update stats
        with stats_lock:
            detection_stats['animal']['detected'] = animal_detected
            if animal_detected:
                detection_stats['animal']['count'] += len(animal_types)
                detection_stats['animal']['types'] = list(set(detection_stats['animal']['types'] + animal_types))[-5:]
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    animal_camera.release()

def generate_vehicle_frames():
    """
    Generate video frames with vehicle detection only
    Only detects and displays vehicles, filters out all other detections
    """
    global camera, detection_active
    
    vehicle_camera = cv2.VideoCapture(0)
    vehicle_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vehicle_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Define vehicle classes to detect - STRICTLY ONLY these vehicles
    # This list ensures NO humans, animals, or other objects are detected
    vehicle_classes_list = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train', 'boat', 'airplane']
    
    # Find vehicle class IDs - ONLY vehicle classes will be detected
    vehicle_class_ids = None
    if 'general' in models:
        try:
            vehicle_class_ids = []
            for cls_id, cls_name in models['general'].names.items():
                # STRICT CHECK: Only include if it's in our vehicle list
                if cls_name.lower() in vehicle_classes_list:
                    vehicle_class_ids.append(cls_id)
            # If no vehicle classes found, set to None to prevent detection
            if len(vehicle_class_ids) == 0:
                vehicle_class_ids = None
        except:
            pass
    
    while detection_active:
        ret, frame = vehicle_camera.read()
        if not ret:
            break
        
        annotated_frame = frame.copy()
        vehicle_detected = False
        vehicle_types = []
        
        # Vehicle detection only - STRICTLY ONLY detect vehicle classes
        # This ensures NO humans, animals, or other objects are detected
        # Only proceed if we have vehicle class IDs to detect
        if 'general' in models and vehicle_class_ids and len(vehicle_class_ids) > 0:
            try:
                # Use classes parameter to ONLY detect vehicle classes - no other classes will be detected
                # The model will ONLY return detections for the specified vehicle classes
                vehicle_results = models['general'](frame, conf=0.5, verbose=False, classes=vehicle_class_ids)
                
                if vehicle_results and len(vehicle_results) > 0:
                    result = vehicle_results[0]
                    
                    # Draw only Vehicle detections - triple verify each box is a vehicle
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            cls_id = int(box.cls[0].cpu().numpy())
                            cls_name = models['general'].names.get(cls_id, 'Unknown')
                            
                            # STRICT CHECK: Only process if it's a vehicle class
                            # This ensures no humans, animals, or other objects slip through
                            # Double verification: check class name is in our vehicle list
                            if cls_name.lower() in vehicle_classes_list:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                conf = float(box.conf[0].cpu().numpy())
                                
                                # Draw bounding box (yellow/orange for vehicles)
                                color = (0, 165, 255)  # Orange color for vehicles
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                                
                                # Draw label background
                                label = f"{cls_name} {conf:.2f}"
                                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                label_y = max(y1, label_size[1] + 10)
                                
                                cv2.rectangle(annotated_frame, 
                                             (x1, label_y - label_size[1] - 10), 
                                             (x1 + label_size[0], label_y + baseline), 
                                             color, -1)
                                
                                # Draw label text
                                cv2.putText(annotated_frame, label, (x1, label_y - 5), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                
                                if cls_name not in vehicle_types:
                                    vehicle_types.append(cls_name)
                                    vehicle_detected = True
            except Exception as e:
                # Fallback: if class filtering fails, filter manually - STRICTLY ONLY Vehicles
                # This ensures that even if model-level filtering fails, we still only show vehicles
                try:
                    all_results = models['general'](frame, conf=0.5, verbose=False)
                    if all_results and len(all_results) > 0:
                        result = all_results[0]
                        if result.boxes is not None:
                            # Filter to STRICTLY ONLY vehicle classes - ignore everything else
                            for box in result.boxes:
                                cls_id = int(box.cls[0].cpu().numpy())
                                cls_name = models['general'].names.get(cls_id, 'Unknown')
                                
                                # STRICT FILTER: Only draw if it's a vehicle
                                # This ensures NO humans, animals, or other objects are displayed
                                if cls_name.lower() in vehicle_classes_list:
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                    conf = float(box.conf[0].cpu().numpy())
                                    
                                    # Draw bounding box (orange for vehicles)
                                    color = (0, 165, 255)  # Orange color for vehicles
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                                    
                                    # Draw label background
                                    label = f"{cls_name} {conf:.2f}"
                                    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                    label_y = max(y1, label_size[1] + 10)
                                    
                                    cv2.rectangle(annotated_frame, 
                                                 (x1, label_y - label_size[1] - 10), 
                                                 (x1 + label_size[0], label_y + baseline), 
                                                 color, -1)
                                    
                                    # Draw label text
                                    cv2.putText(annotated_frame, label, (x1, label_y - 5), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                    
                                    if cls_name not in vehicle_types:
                                        vehicle_types.append(cls_name)
                                        vehicle_detected = True
                except:
                    pass
        
        # Update stats
        with stats_lock:
            detection_stats['vehicle']['detected'] = vehicle_detected
            if vehicle_detected:
                detection_stats['vehicle']['count'] += len(vehicle_types)
                detection_stats['vehicle']['types'] = list(set(detection_stats['vehicle']['types'] + vehicle_types))[-5:]
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    vehicle_camera.release()

def generate_plate_frames():
    """
    Generate video frames with number plate detection only
    Only detects and displays number plates, filters out all other detections
    """
    global camera, detection_active, ocr_reader
    
    plate_camera = cv2.VideoCapture(0)
    plate_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    plate_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Vehicle classes for finding vehicles (to locate plates)
    vehicle_class_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck in COCO
    
    while detection_active:
        ret, frame = plate_camera.read()
        if not ret:
            break
        
        annotated_frame = frame.copy()
        plate_detected = False
        plate_texts = []
        
        # Number plate detection only - STRICTLY ONLY detect number plates
        # This ensures NO humans, animals, or other objects are detected
        if 'general' in models and ocr_reader:
            try:
                # First detect vehicles to find potential plate regions
                vehicle_results = models['general'](frame, conf=0.3, verbose=False, classes=vehicle_class_ids)
                
                plates_detected = []
                if vehicle_results[0].boxes is not None:
                    for box in vehicle_results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        vehicle_roi = frame[y1:y2, x1:x2]
                        if vehicle_roi.size > 0:
                            # Use edge detection to find plate-like regions
                            gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
                            edges = cv2.Canny(gray, 50, 150)
                            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            for cnt in contours:
                                px, py, pw, ph = cv2.boundingRect(cnt)
                                aspect = pw / float(ph) if ph > 0 else 0
                                
                                # License plates typically have aspect ratio between 2:1 and 5:1
                                if 2.0 <= aspect <= 5.0 and cv2.contourArea(cnt) > 100:
                                    plate_x = x1 + px
                                    plate_y = y1 + py
                                    plate_w = pw
                                    plate_h = ph
                                    
                                    plate_roi = vehicle_roi[py:py+ph, px:px+pw]
                                    if plate_roi.size > 0:
                                        try:
                                            # Read text from plate using OCR
                                            ocr_results = ocr_reader.readtext(plate_roi)
                                            if ocr_results:
                                                text = ocr_results[0][1]
                                                text = ''.join(c for c in text if c.isalnum() or c in ['-', ' '])
                                                if text.strip() and text not in plates_detected:
                                                    plates_detected.append({
                                                        'bbox': (plate_x, plate_y, plate_w, plate_h),
                                                        'text': text.strip()
                                                    })
                                                    
                                                    # Draw bounding box (purple for plates)
                                                    color = (255, 0, 255)  # Magenta/purple color
                                                    cv2.rectangle(annotated_frame, 
                                                                 (plate_x, plate_y), 
                                                                 (plate_x + plate_w, plate_y + plate_h), 
                                                                 color, 3)
                                                    
                                                    # Draw label background
                                                    label = f"PLATE: {text.strip()}"
                                                    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                                    label_y = max(plate_y, label_size[1] + 10)
                                                    
                                                    cv2.rectangle(annotated_frame, 
                                                                 (plate_x, label_y - label_size[1] - 10), 
                                                                 (plate_x + label_size[0], label_y + baseline), 
                                                                 color, -1)
                                                    
                                                    # Draw label text
                                                    cv2.putText(annotated_frame, label, (plate_x, label_y - 5), 
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                                    
                                                    if text.strip() not in plate_texts:
                                                        plate_texts.append(text.strip())
                                                        plate_detected = True
                                        except:
                                            pass
                
                # Also try edge detection on entire frame for plates not on vehicles
                if not plate_detected:
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                        edges = cv2.Canny(blurred, 50, 150)
                        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            aspect_ratio = w / float(h) if h > 0 else 0
                            area = cv2.contourArea(contour)
                            
                            # License plates typically have aspect ratio between 2:1 and 5:1
                            if 2.0 <= aspect_ratio <= 5.0 and area > 500:
                                # Check if it's in lower portion of frame (where plates usually are)
                                if y > frame.shape[0] * 0.3:  # Lower 70% of frame
                                    plate_roi = frame[y:y+h, x:x+w]
                                    if plate_roi.size > 0:
                                        try:
                                            ocr_results = ocr_reader.readtext(plate_roi)
                                            if ocr_results:
                                                text = ocr_results[0][1]
                                                text = ''.join(c for c in text if c.isalnum() or c in ['-', ' '])
                                                if text.strip() and text.strip() not in plate_texts:
                                                    # Draw bounding box
                                                    color = (255, 0, 255)  # Magenta/purple color
                                                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 3)
                                                    
                                                    # Draw label
                                                    label = f"PLATE: {text.strip()}"
                                                    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                                    label_y = max(y, label_size[1] + 10)
                                                    
                                                    cv2.rectangle(annotated_frame, 
                                                                 (x, label_y - label_size[1] - 10), 
                                                                 (x + label_size[0], label_y + baseline), 
                                                                 color, -1)
                                                    cv2.putText(annotated_frame, label, (x, label_y - 5), 
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                                    
                                                    plate_texts.append(text.strip())
                                                    plate_detected = True
                                        except:
                                            pass
                    except:
                        pass
            except:
                pass
        
        # Update stats
        with stats_lock:
            detection_stats['plate']['detected'] = plate_detected
            if plate_detected:
                detection_stats['plate']['count'] += len(plate_texts)
                detection_stats['plate']['texts'] = list(set(detection_stats['plate']['texts'] + plate_texts))[:10]
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    plate_camera.release()

def generate_frames():
    """
    Generate video frames - always shows camera feed
    Shows raw feed when detection is off, annotated feed when detection is on
    """
    global camera, detection_active
    
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Always run the loop to show camera feed continuously
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        # If detection is active, run detections and annotate
        if detection_active:
            # Run all detections
            annotated_frame = detect_all(frame)
            
            # Add status overlay
            h, w = annotated_frame.shape[:2]
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
            annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0)
            
            # Status text
            status_lines = []
            if detection_stats['fire']['detected']:
                status_lines.append("🔥 FIRE")
            if detection_stats['weapon']['detected']:
                status_lines.append("🔫 WEAPON")
            if detection_stats['human']['detected']:
                status_lines.append("👤 HUMAN")
            if detection_stats['animal']['detected']:
                status_lines.append("🐾 ANIMAL")
            if detection_stats['vehicle']['detected']:
                status_lines.append("🚗 VEHICLE")
            if detection_stats['plate']['detected']:
                status_lines.append("🚗 PLATE")
            
            if status_lines:
                status_text = " | ".join(status_lines)
                cv2.putText(annotated_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(annotated_frame, "All Systems Active - No Threats", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            output_frame = annotated_frame
        else:
            # Detection is off - just show raw camera feed with status message
            output_frame = frame.copy()
            h, w = output_frame.shape[:2]
            
            # Add simple status overlay
            cv2.rectangle(output_frame, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.putText(output_frame, "Camera Active - Detection Off", (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Authentication Routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required.', 'error')
            return render_template('register.html')
        
        if len(username) < 3:
            flash('Username must be at least 3 characters long.', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        
        # Check if user exists
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # Check username
        c.execute('SELECT id FROM users WHERE username = ?', (username,))
        if c.fetchone():
            conn.close()
            flash('Username already exists.', 'error')
            return render_template('register.html')
        
        # Check email
        c.execute('SELECT id FROM users WHERE email = ?', (email,))
        if c.fetchone():
            conn.close()
            flash('Email already registered.', 'error')
            return render_template('register.html')
        
        # Create user
        password_hash = generate_password_hash(password)
        c.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                  (username, email, password_hash))
        conn.commit()
        conn.close()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = bool(request.form.get('remember'))
        
        if not username or not password:
            flash('Please enter both username and password.', 'error')
            return render_template('login.html')
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT id, username, email, password_hash FROM users WHERE username = ?', (username,))
        user_data = c.fetchone()
        conn.close()
        
        if user_data and check_password_hash(user_data[3], password):
            user = User(user_data[0], user_data[1], user_data[2])
            login_user(user, remember=remember)
            flash(f'Welcome back, {username}!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

# Main Application Routes
@app.route('/')
@login_required
def dashboard():
    """Dashboard page - requires login"""
    return render_template('dashboard.html', username=current_user.username)

@app.route('/detection')
@login_required
def detection():
    """Detection page - requires login"""
    return render_template('index.html', username=current_user.username)

@app.route('/video-upload')
@login_required
def video_upload():
    """Video upload page - requires login"""
    return render_template('video_upload.html', username=current_user.username)

def process_video_with_detections(video_path, output_path, enabled_detections):
    """
    Process video with selected detection types
    enabled_detections: dict with keys 'fire', 'weapon', 'human', 'animal', 'vehicle', 'plate'
    """
    global detection_enabled
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Could not open video file"
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    detection_summary = {
        'fire': {'count': 0, 'detected': False},
        'weapon': {'count': 0, 'detected': False, 'types': []},
        'human': {'count': 0, 'detected': False},
        'animal': {'count': 0, 'detected': False, 'types': []},
        'vehicle': {'count': 0, 'detected': False, 'types': []},
        'plate': {'count': 0, 'detected': False, 'texts': []}
    }
    
    # Temporary enable detections
    original_enabled = detection_enabled.copy()
    detection_enabled.update(enabled_detections)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processing frame {frame_count}/{total_frames}")
            
            # Process frame with detections
            results = {}
            annotated_frame = frame.copy()
            
            # Fire detection
            if enabled_detections.get('fire') and 'fire' in models:
                try:
                    fire_results = models['fire'](frame, conf=0.25, verbose=False)
                    for result in fire_results:
                        if result.boxes is not None and len(result.boxes) > 0:
                            detection_summary['fire']['detected'] = True
                            detection_summary['fire']['count'] += len(result.boxes)
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(annotated_frame, 'Fire', (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    results['fire'] = fire_results
                except:
                    pass
            
            # Weapon detection
            if enabled_detections.get('weapon') and 'weapon' in models:
                try:
                    weapon_results = models['weapon'](frame, conf=0.5, verbose=False)
                    weapons = []
                    for result in weapon_results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                cls_id = int(box.cls[0].cpu().numpy())
                                cls_name = models['weapon'].names.get(cls_id, 'Unknown')
                                if cls_name.lower() in ['knife', 'pistol']:
                                    if cls_name not in weapons:
                                        weapons.append(cls_name)
                                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                                    cv2.putText(annotated_frame, cls_name, (x1, y1-10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    if weapons:
                        detection_summary['weapon']['detected'] = True
                        detection_summary['weapon']['types'] = list(set(detection_summary['weapon']['types'] + weapons))
                        detection_summary['weapon']['count'] += len(weapons)
                except:
                    pass
            
            # Human detection
            if enabled_detections.get('human') and 'general' in models:
                try:
                    person_class_id = None
                    for cls_id, cls_name in models['general'].names.items():
                        if cls_name.lower() == 'person':
                            person_class_id = cls_id
                            break
                    if person_class_id is not None:
                        human_results = models['general'](frame, conf=0.5, verbose=False, classes=[person_class_id])
                        if human_results[0].boxes is not None and len(human_results[0].boxes) > 0:
                            detection_summary['human']['detected'] = True
                            detection_summary['human']['count'] += len(human_results[0].boxes)
                            for box in human_results[0].boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(annotated_frame, 'Person', (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except:
                    pass
            
            # Animal detection
            if enabled_detections.get('animal') and 'general' in models:
                try:
                    animal_classes = []
                    for cls_id, cls_name in models['general'].names.items():
                        if cls_name.lower() in ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']:
                            animal_classes.append(cls_id)
                    if animal_classes:
                        animal_results = models['general'](frame, conf=0.5, verbose=False, classes=animal_classes)
                        animals = []
                        if animal_results[0].boxes is not None:
                            for box in animal_results[0].boxes:
                                cls_id = int(box.cls[0].cpu().numpy())
                                cls_name = models['general'].names.get(cls_id, 'Unknown')
                                if cls_name not in animals:
                                    animals.append(cls_name)
                                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 192, 203), 2)
                                cv2.putText(annotated_frame, cls_name, (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 192, 203), 2)
                        if animals:
                            detection_summary['animal']['detected'] = True
                            detection_summary['animal']['types'] = list(set(detection_summary['animal']['types'] + animals))
                            detection_summary['animal']['count'] += len(animals)
                except:
                    pass
            
            # Vehicle detection
            if enabled_detections.get('vehicle') and 'general' in models:
                try:
                    vehicle_classes = []
                    for cls_id, cls_name in models['general'].names.items():
                        if cls_name.lower() in ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train', 'boat', 'airplane']:
                            vehicle_classes.append(cls_id)
                    if vehicle_classes:
                        vehicle_results = models['general'](frame, conf=0.5, verbose=False, classes=vehicle_classes)
                        vehicles = []
                        if vehicle_results[0].boxes is not None:
                            for box in vehicle_results[0].boxes:
                                cls_id = int(box.cls[0].cpu().numpy())
                                cls_name = models['general'].names.get(cls_id, 'Unknown')
                                if cls_name not in vehicles:
                                    vehicles.append(cls_name)
                                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                                cv2.putText(annotated_frame, cls_name, (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        if vehicles:
                            detection_summary['vehicle']['detected'] = True
                            detection_summary['vehicle']['types'] = list(set(detection_summary['vehicle']['types'] + vehicles))
                            detection_summary['vehicle']['count'] += len(vehicles)
                except:
                    pass
            
            # Number plate detection
            if enabled_detections.get('plate') and 'general' in models and ocr_reader:
                try:
                    vehicle_classes = [2, 3, 5, 7]
                    vehicle_results = models['general'](frame, conf=0.3, verbose=False, classes=vehicle_classes)
                    plates_detected = []
                    if vehicle_results[0].boxes is not None:
                        for box in vehicle_results[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            vehicle_roi = frame[y1:y2, x1:x2]
                            if vehicle_roi.size > 0:
                                gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
                                edges = cv2.Canny(gray, 50, 150)
                                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for cnt in contours:
                                    px, py, pw, ph = cv2.boundingRect(cnt)
                                    aspect = pw / float(ph) if ph > 0 else 0
                                    if 2.0 <= aspect <= 5.0 and cv2.contourArea(cnt) > 100:
                                        plate_roi = vehicle_roi[py:py+ph, px:px+pw]
                                        if plate_roi.size > 0:
                                            try:
                                                ocr_results = ocr_reader.readtext(plate_roi)
                                                if ocr_results:
                                                    text = ocr_results[0][1]
                                                    text = ''.join(c for c in text if c.isalnum() or c in ['-', ' '])
                                                    if text.strip() and text not in plates_detected:
                                                        plates_detected.append(text.strip())
                                                        cv2.rectangle(annotated_frame, (x1+px, y1+py), (x1+px+pw, y1+py+ph), (255, 0, 255), 2)
                                                        cv2.putText(annotated_frame, text, (x1+px, y1+py-10), 
                                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                                            except:
                                                pass
                    if plates_detected:
                        detection_summary['plate']['detected'] = True
                        detection_summary['plate']['texts'] = list(set(detection_summary['plate']['texts'] + plates_detected))
                        detection_summary['plate']['count'] += len(plates_detected)
                except:
                    pass
            
            out.write(annotated_frame)
    
    finally:
        # Restore original detection settings
        detection_enabled.update(original_enabled)
        cap.release()
        out.release()
    
    return True, detection_summary

@app.route('/upload-video', methods=['POST'])
@login_required
def upload_video():
    """Handle video upload and processing"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_video_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv, flv, wmv, webm'}), 400
    
    # Get enabled detections from form
    enabled_detections = {
        'fire': 'fire' in request.form,
        'weapon': 'weapon' in request.form,
        'human': 'human' in request.form,
        'animal': 'animal' in request.form,
        'vehicle': 'vehicle' in request.form,
        'plate': 'plate' in request.form
    }
    
    # Check if at least one detection is enabled
    if not any(enabled_detections.values()):
        return jsonify({'error': 'Please select at least one detection type'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        upload_path = os.path.join(UPLOAD_VIDEO_FOLDER, unique_filename)
        file.save(upload_path)
        
        # Process video
        output_filename = f"processed_{timestamp}_{filename}"
        output_path = os.path.join(PROCESSED_VIDEO_FOLDER, output_filename)
        
        success, result = process_video_with_detections(upload_path, output_path, enabled_detections)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Video processed successfully',
                'video_url': url_for('download_processed_video', filename=output_filename),
                'detection_summary': result
            })
        else:
            return jsonify({'error': result}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload-face', methods=['POST'])
@login_required
def upload_face():
    """
    Face Registration:
    - Detect face (Haar Cascade)
    - Extract embedding (DeepFace Facenet512)
    - Save embedding to embeddings/{person_name}.npy
    """
    if 'face_image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    person_name = request.form.get('person_name', '').strip()
    if not person_name:
        return jsonify({'error': 'Please enter a person name'}), 400

    file = request.files['face_image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_image_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp, webp'}), 400

    try:
        ensure_dirs(FACE_CFG)

        # Save uploaded file (optional: keep as evidence/debug)
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{current_user.id}_{timestamp}_{filename}"
        upload_path = os.path.join(USER_FACES_FOLDER, unique_filename)
        file.save(upload_path)

        if not os.path.exists(upload_path) or os.path.getsize(upload_path) == 0:
            return jsonify({'error': 'Failed to save uploaded image'}), 500

        # Read with OpenCV -> RGB
        bgr = cv2.imread(upload_path)
        if bgr is None:
            return jsonify({'error': 'Could not read uploaded image'}), 400
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Detect face
        face_box = detect_face_haar(rgb, FACE_CFG)
        if face_box is None:
            return jsonify({'error': 'No face detected in the uploaded image'}), 400

        # Crop face and embed
        face_rgb = crop_rgb(rgb, face_box)
        emb = embedding_facenet512(face_rgb, FACE_CFG)
        if emb is None:
            return jsonify({'error': 'Face detected but embedding extraction failed'}), 500

        emb_file = save_embedding(person_name, emb, FACE_CFG)

        return jsonify({
            'status': 'success',
            'message': f'Embedding created for {person_name}',
            'embedding_file': emb_file,
            'face_box': {
                'x1': int(face_box[0]), 
                'y1': int(face_box[1]), 
                'x2': int(face_box[2]), 
                'y2': int(face_box[3])
            },
        })
    except Exception as e:
        return jsonify({'error': f'Error processing face image: {str(e)}'}), 500

@app.route('/get-user-faces')
@login_required
def get_user_faces():
    """Get all face images uploaded by the current user"""
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('''SELECT id, image_path, face_location, timestamp 
                     FROM user_faces 
                     WHERE user_id = ? 
                     ORDER BY timestamp DESC''', (current_user.id,))
        faces = c.fetchall()
        conn.close()
        
        faces_list = []
        for face in faces:
            faces_list.append({
                'id': face[0],
                'image_path': face[1],
                'face_location': json.loads(face[2]) if face[2] else None,
                'timestamp': face[3]
            })
        
        return jsonify({'status': 'success', 'faces': faces_list})
    except Exception as e:
        return jsonify({'error': f'Error retrieving faces: {str(e)}'}), 500

def generate_live_collection_frames():
    """
    Generate video frames for live image collection
    Returns both main frame and detected face view
    """
    global live_collection_active, live_collection_camera, live_collection_face_images
    global live_collection_count, live_collection_person_name, live_collection_max_samples
    
    # Load Haar Cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
    if not os.path.exists(cascade_path):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if live_collection_camera is None:
        live_collection_camera = cv2.VideoCapture(0)
        live_collection_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        live_collection_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while live_collection_active:
        ret, frame = live_collection_camera.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        detected_face_frame = None
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (200, 200))
            
            # Add to collection if not at max (thread-safe)
            with live_collection_lock:
                if live_collection_count < live_collection_max_samples:
                    live_collection_face_images.append(face_resized)
                    live_collection_count += 1
            
            # Draw rectangle on main frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Create detected face frame (color version for display)
            face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2BGR)
            detected_face_frame = face_bgr.copy()
            
            # Add border to face frame
            detected_face_frame = cv2.copyMakeBorder(
                detected_face_frame, 10, 10, 10, 10, 
                cv2.BORDER_CONSTANT, value=(0, 255, 0)
            )
        
        # Show count and status on main frame
        status_text = f"Count: {live_collection_count}/{live_collection_max_samples}"
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if live_collection_person_name:
            name_text = f"Collecting: {live_collection_person_name}"
            cv2.putText(frame, name_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Create combined frame: main frame on left, face view on right (or below if no face)
        if detected_face_frame is not None:
            # Resize face frame to match height
            face_height = frame.shape[0]
            face_width = int(detected_face_frame.shape[1] * face_height / detected_face_frame.shape[0])
            detected_face_frame = cv2.resize(detected_face_frame, (face_width, face_height))
            
            # Combine frames side by side
            combined_frame = np.hstack([frame, detected_face_frame])
        else:
            # No face detected, show placeholder
            placeholder = np.zeros((frame.shape[0], 220, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No Face", (30, frame.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
            cv2.putText(placeholder, "Detected", (20, frame.shape[0]//2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
            combined_frame = np.hstack([frame, placeholder])
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', combined_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/live-collection-video-feed')
@login_required
def live_collection_video_feed():
    """Video stream for live image collection"""
    return Response(generate_live_collection_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start-live-collection', methods=['POST'])
@login_required
def start_live_collection():
    """Start live image collection"""
    global live_collection_active, live_collection_camera, live_collection_face_images
    global live_collection_count, live_collection_person_name
    
    try:
        data = request.get_json()
        person_name = data.get('person_name', '').strip()
        
        if not person_name:
            return jsonify({'error': 'Please enter a person name'}), 400
        
        # Sanitize person name
        safe_name = "".join(c for c in person_name if c.isalnum() or c in ("_", "-")).strip()
        if not safe_name:
            return jsonify({'error': 'Invalid person name'}), 400
        
        with live_collection_lock:
            if live_collection_active:
                return jsonify({'error': 'Collection is already active'}), 400
            
            # Initialize camera
            if live_collection_camera is None:
                live_collection_camera = cv2.VideoCapture(0)
                if not live_collection_camera.isOpened():
                    return jsonify({'error': 'Could not open camera. Make sure your webcam is connected.'}), 500
            
            # Reset collection state
            live_collection_active = True
            live_collection_person_name = person_name
            live_collection_face_images = []
            live_collection_count = 0
        
        return jsonify({
            'status': 'success',
            'message': f'Started collecting images for {person_name}'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error starting collection: {str(e)}'}), 500

@app.route('/stop-live-collection', methods=['POST'])
@login_required
def stop_live_collection():
    """Stop live image collection and save collected images"""
    global live_collection_active, live_collection_camera, live_collection_face_images
    global live_collection_count, live_collection_person_name
    
    try:
        with live_collection_lock:
            if not live_collection_active:
                return jsonify({'error': 'Collection is not active'}), 400
            
            person_name = live_collection_person_name
            face_images = live_collection_face_images.copy()
            count = live_collection_count
            
            # Stop collection
            live_collection_active = False
            
            # Release camera
            if live_collection_camera is not None:
                live_collection_camera.release()
                live_collection_camera = None
        
        if count == 0:
            return jsonify({'error': 'No face images collected. Make sure your face is visible to the camera.'}), 400
        
        # Sanitize person name for filename
        safe_name = "".join(c for c in person_name if c.isalnum() or c in ("_", "-")).strip()
        
        # Save collected images
        dataset_path = "./face_dataset/"
        os.makedirs(dataset_path, exist_ok=True)
        dataset_file = os.path.join(dataset_path, f"{safe_name}_faces.npy")
        np.save(dataset_file, np.array(face_images))
        
        print(f"Saved {count} images to {dataset_file}")
        
        return jsonify({
            'status': 'success',
            'message': f'Successfully collected {count} face images',
            'person_name': person_name,
            'count': count,
            'dataset_file': dataset_file
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error stopping collection: {str(e)}'}), 500

@app.route('/get-collection-status')
@login_required
def get_collection_status():
    """Get current collection status"""
    global live_collection_active, live_collection_count, live_collection_person_name, live_collection_max_samples
    
    with live_collection_lock:
        return jsonify({
            'active': live_collection_active,
            'count': live_collection_count,
            'max_samples': live_collection_max_samples,
            'person_name': live_collection_person_name if live_collection_active else None
        })

@app.route('/download-processed-video/<filename>')
@login_required
def download_processed_video(filename):
    """Download processed video"""
    try:
        return send_file(
            os.path.join(PROCESSED_VIDEO_FOLDER, filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/user_faces/<path:filename>')
@login_required
def serve_face_image(filename):
    """Serve user face images"""
    try:
        # Security: Ensure the file belongs to the current user
        file_path = os.path.join(USER_FACES_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Verify file belongs to current user (filename starts with user_id)
        if not filename.startswith(f"{current_user.id}_"):
            return jsonify({'error': 'Access denied'}), 403
        
        return send_file(file_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/fire-detection')
@login_required
def fire_detection():
    """Fire detection only page - requires login"""
    return render_template('fire_detection.html')

@app.route('/fire_video_feed')
@login_required
def fire_video_feed():
    """Fire detection video streaming route - requires login"""
    return Response(generate_fire_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fire_stats')
@login_required
def fire_stats():
    """Get fire detection statistics - requires login"""
    with stats_lock:
        return jsonify({'fire': detection_stats['fire']})

@app.route('/fire_start')
@login_required
def fire_start():
    """Start fire detection - requires login"""
    global detection_active, camera
    
    if not detection_active:
        detection_active = True
        if camera is None:
            camera = cv2.VideoCapture(0)
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'})

@app.route('/fire_stop')
@login_required
def fire_stop():
    """Stop fire detection - requires login"""
    global detection_active, camera
    
    detection_active = False
    if camera:
        camera.release()
        camera = None
    return jsonify({'status': 'stopped'})

@app.route('/fire_reset')
@login_required
def fire_reset():
    """Reset fire detection statistics - requires login"""
    global detection_stats
    with stats_lock:
        detection_stats['fire'] = {'count': 0, 'detected': False}
    return jsonify({'status': 'reset'})

@app.route('/weapon-detection')
@login_required
def weapon_detection():
    """Weapon detection only page - requires login"""
    return render_template('weapon_detection.html')

@app.route('/weapon_video_feed')
@login_required
def weapon_video_feed():
    """Weapon detection video streaming route - requires login"""
    return Response(generate_weapon_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/weapon_stats')
@login_required
def weapon_stats():
    """Get weapon detection statistics - requires login"""
    with stats_lock:
        return jsonify({'weapon': detection_stats['weapon']})

@app.route('/weapon_start')
@login_required
def weapon_start():
    """Start weapon detection - requires login"""
    global detection_active, camera
    
    if not detection_active:
        detection_active = True
        if camera is None:
            camera = cv2.VideoCapture(0)
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'})

@app.route('/weapon_stop')
@login_required
def weapon_stop():
    """Stop weapon detection - requires login"""
    global detection_active, camera
    
    detection_active = False
    if camera:
        camera.release()
        camera = None
    return jsonify({'status': 'stopped'})

@app.route('/weapon_reset')
@login_required
def weapon_reset():
    """Reset weapon detection statistics - requires login"""
    global detection_stats
    with stats_lock:
        detection_stats['weapon'] = {'count': 0, 'detected': False, 'types': []}
    return jsonify({'status': 'reset'})

@app.route('/human-detection')
@login_required
def human_detection():
    """Human detection only page - requires login"""
    return render_template('human_detection.html')

@app.route('/human_video_feed')
@login_required
def human_video_feed():
    """Human detection video streaming route - requires login"""
    return Response(generate_human_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/human_stats')
@login_required
def human_stats():
    """Get human detection statistics - requires login"""
    with stats_lock:
        return jsonify({'human': detection_stats['human']})

@app.route('/human_start')
@login_required
def human_start():
    """Start human detection - requires login"""
    global detection_active, camera
    
    if not detection_active:
        detection_active = True
        if camera is None:
            camera = cv2.VideoCapture(0)
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'})

@app.route('/human_stop')
@login_required
def human_stop():
    """Stop human detection - requires login"""
    global detection_active, camera
    
    detection_active = False
    if camera:
        camera.release()
        camera = None
    return jsonify({'status': 'stopped'})

@app.route('/human_reset')
@login_required
def human_reset():
    """Reset human detection statistics - requires login"""
    global detection_stats
    with stats_lock:
        detection_stats['human'] = {'count': 0, 'detected': False}
    return jsonify({'status': 'reset'})

@app.route('/animal-detection')
@login_required
def animal_detection():
    """Animal detection only page - requires login"""
    return render_template('animal_detection.html')

@app.route('/animal_video_feed')
@login_required
def animal_video_feed():
    """Animal detection video streaming route - requires login"""
    return Response(generate_animal_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/animal_stats')
@login_required
def animal_stats():
    """Get animal detection statistics - requires login"""
    with stats_lock:
        return jsonify({'animal': detection_stats['animal']})

@app.route('/animal_start')
@login_required
def animal_start():
    """Start animal detection - requires login"""
    global detection_active, camera
    
    if not detection_active:
        detection_active = True
        if camera is None:
            camera = cv2.VideoCapture(0)
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'})

@app.route('/animal_stop')
@login_required
def animal_stop():
    """Stop animal detection - requires login"""
    global detection_active, camera
    
    detection_active = False
    if camera:
        camera.release()
        camera = None
    return jsonify({'status': 'stopped'})

@app.route('/animal_reset')
@login_required
def animal_reset():
    """Reset animal detection statistics - requires login"""
    global detection_stats
    with stats_lock:
        detection_stats['animal'] = {'count': 0, 'detected': False, 'types': []}
    return jsonify({'status': 'reset'})

@app.route('/vehicle-detection')
@login_required
def vehicle_detection():
    """Vehicle detection only page - requires login"""
    return render_template('vehicle_detection.html')

@app.route('/vehicle_video_feed')
@login_required
def vehicle_video_feed():
    """Vehicle detection video streaming route - requires login"""
    return Response(generate_vehicle_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vehicle_stats')
@login_required
def vehicle_stats():
    """Get vehicle detection statistics - requires login"""
    with stats_lock:
        return jsonify({'vehicle': detection_stats['vehicle']})

@app.route('/vehicle_start')
@login_required
def vehicle_start():
    """Start vehicle detection - requires login"""
    global detection_active, camera
    
    if not detection_active:
        detection_active = True
        if camera is None:
            camera = cv2.VideoCapture(0)
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'})

@app.route('/vehicle_stop')
@login_required
def vehicle_stop():
    """Stop vehicle detection - requires login"""
    global detection_active, camera
    
    detection_active = False
    if camera:
        camera.release()
        camera = None
    return jsonify({'status': 'stopped'})

@app.route('/vehicle_reset')
@login_required
def vehicle_reset():
    """Reset vehicle detection statistics - requires login"""
    global detection_stats
    with stats_lock:
        detection_stats['vehicle'] = {'count': 0, 'detected': False, 'types': []}
    return jsonify({'status': 'reset'})

@app.route('/number-plate-detection')
@login_required
def number_plate_detection():
    """Number plate detection only page - requires login"""
    return render_template('number_plate_detection.html')

@app.route('/plate_video_feed')
@login_required
def plate_video_feed():
    """Number plate detection video streaming route - requires login"""
    return Response(generate_plate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/plate_stats')
@login_required
def plate_stats():
    """Get number plate detection statistics - requires login"""
    with stats_lock:
        return jsonify({'plate': detection_stats['plate']})

@app.route('/plate_start')
@login_required
def plate_start():
    """Start number plate detection - requires login"""
    global detection_active, camera
    
    if not detection_active:
        detection_active = True
        if camera is None:
            camera = cv2.VideoCapture(0)
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'})

@app.route('/plate_stop')
@login_required
def plate_stop():
    """Stop number plate detection - requires login"""
    global detection_active, camera
    
    detection_active = False
    if camera:
        camera.release()
        camera = None
    return jsonify({'status': 'stopped'})

@app.route('/plate_reset')
@login_required
def plate_reset():
    """Reset number plate detection statistics - requires login"""
    global detection_stats
    with stats_lock:
        detection_stats['plate'] = {'count': 0, 'detected': False, 'texts': []}
    return jsonify({'status': 'reset'})

@app.route('/video_feed')
@login_required
def video_feed():
    """Video streaming route - requires login"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
@login_required
def stats():
    """Get detection statistics - requires login"""
    with stats_lock:
        return jsonify(detection_stats)

@app.route('/start')
@login_required
def start_detection():
    """Start detection - requires login"""
    global detection_active, camera
    
    if not detection_active:
        detection_active = True
        if camera is None:
            camera = cv2.VideoCapture(0)
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'})

@app.route('/stop')
@login_required
def stop_detection():
    """Stop detection - requires login (but keep camera running for video feed)"""
    global detection_active
    
    detection_active = False
    # Don't release camera - keep it running to show video feed
    return jsonify({'status': 'stopped'})

@app.route('/stop-camera', methods=['GET', 'POST'])
@login_required
def stop_camera():
    """Stop camera completely - releases camera resource"""
    global camera, detection_active
    
    detection_active = False
    if camera is not None:
        camera.release()
        camera = None
        print("Camera released")
    return jsonify({'status': 'camera_stopped'})

@app.route('/reset')
@login_required
def reset_stats():
    """Reset detection statistics - requires login"""
    global detection_stats, previous_detection_state
    with stats_lock:
        detection_stats = {
            'fire': {'count': 0, 'detected': False},
            'weapon': {'count': 0, 'detected': False, 'types': []},
            'human': {'count': 0, 'detected': False},
            'animal': {'count': 0, 'detected': False, 'types': []},
            'vehicle': {'count': 0, 'detected': False, 'types': []},
            'plate': {'count': 0, 'detected': False, 'texts': []}
        }
        # Reset previous detection state to allow saving new detections after reset
        previous_detection_state = {
            'fire': False,
            'weapon': False,
            'human': False,
            'animal': False,
            'vehicle': False,
            'plate': False
        }
    return jsonify({'status': 'reset'})

@app.route('/toggle_detection/<detection_type>', methods=['POST'])
@login_required
def toggle_detection(detection_type):
    """Toggle detection on/off for a specific type - requires login"""
    global detection_enabled
    
    if detection_type not in detection_enabled:
        return jsonify({'error': 'Invalid detection type'}), 400
    
    detection_enabled[detection_type] = not detection_enabled[detection_type]
    return jsonify({
        'status': 'success',
        'detection_type': detection_type,
        'enabled': detection_enabled[detection_type]
    })

@app.route('/detection_status')
@login_required
def detection_status():
    """Get detection enable/disable status - requires login"""
    return jsonify(detection_enabled)

@app.route('/toggle_email_notifications', methods=['POST'])
@login_required
def toggle_email_notifications():
    """Toggle email notifications on/off - requires login"""
    global email_notifications_enabled
    email_notifications_enabled = not email_notifications_enabled
    return jsonify({
        'status': 'success',
        'enabled': email_notifications_enabled
    })

@app.route('/email_notifications_status')
@login_required
def email_notifications_status():
    """Get email notifications status - requires login"""
    global email_notifications_enabled
    return jsonify({'enabled': email_notifications_enabled})

@app.route('/history')
@login_required
def history():
    """Detection history page - requires login"""
    # Get detection history from database
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Get history for current user or all if admin
    c.execute('''SELECT id, detection_type, frame_path, timestamp, details 
                 FROM detection_history 
                 WHERE user_id = ? OR user_id IS NULL
                 ORDER BY timestamp DESC 
                 LIMIT 100''', (current_user.id,))
    
    history_records = []
    for row in c.fetchall():
        record = {
            'id': row[0],
            'detection_type': row[1],
            'frame_path': row[2],
            'frame_filename': os.path.basename(row[2]),  # Extract just filename for URL
            'timestamp': row[3],
            'details': row[4]
        }
        # Parse details if available
        if record['details']:
            try:
                record['details'] = json.loads(record['details'])
            except:
                pass
        history_records.append(record)
    
    conn.close()
    
    return render_template('history.html', history=history_records, username=current_user.username)

@app.route('/history_image/<path:filename>')
@login_required
def history_image(filename):
    """Serve detection history images - requires login"""
    from flask import send_from_directory
    # Extract just the filename if full path is provided
    filename_only = os.path.basename(filename)
    filepath = os.path.join(DETECTION_HISTORY_FOLDER, filename_only)
    
    # Security check - ensure file is in the history folder
    abs_history_path = os.path.abspath(DETECTION_HISTORY_FOLDER)
    abs_filepath = os.path.abspath(filepath)
    
    if os.path.exists(filepath) and abs_filepath.startswith(abs_history_path):
        return send_from_directory(DETECTION_HISTORY_FOLDER, filename_only)
    return jsonify({'error': 'File not found'}), 404

@app.route('/delete_history/<int:history_id>', methods=['POST'])
@login_required
def delete_history(history_id):
    """Delete a history record - requires login"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Get the record to check ownership
    c.execute('SELECT frame_path, user_id FROM detection_history WHERE id = ?', (history_id,))
    record = c.fetchone()
    
    if not record:
        conn.close()
        return jsonify({'error': 'Record not found'}), 404
    
    frame_path, user_id = record
    
    # Check if user owns this record or it's a shared record
    if user_id is not None and user_id != current_user.id:
        conn.close()
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Delete the file if it exists
    if os.path.exists(frame_path):
        try:
            os.remove(frame_path)
        except:
            pass
    
    # Delete from database
    c.execute('DELETE FROM detection_history WHERE id = ?', (history_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'status': 'deleted'})


if __name__ == '__main__':
    detection_active = True
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
