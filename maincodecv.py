import cv2
import numpy as np
from flask import Flask, jsonify, render_template
import threading
import time
from collections import deque

app = Flask(__name__)

class EnhancedGridMotionDetector:
    def __init__(self, camera_url, grid_size=(3, 2), min_activity_threshold=1000, fps_limit=30, window_size=(1920, 1080)):
        self.camera_url = camera_url
        self.grid_size = grid_size
        self.min_activity_threshold = min_activity_threshold
        self.fps_limit = fps_limit
        self.window_size = window_size
        
        # Enhanced frame handling
        self.previous_frames = deque(maxlen=3)  # Store last 3 frames for better motion detection
        self.grid_activity = {}
        self.human_detected = False
        self.frame_counter = 0
        
        # Improved camera settings
        self.camera_settings = {
            'brightness': 1.2,      # Slightly increase brightness
            'contrast': 1.1,        # Slightly increase contrast
            'saturation': 1.2,      # Enhance colors
            'sharpness': 1.3        # Increase sharpness
        }
        
        # Initialize enhanced HOG detector with better parameters
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )

    def connect_camera(self):
        """Connect to camera with enhanced settings"""
        cap = cv2.VideoCapture(self.camera_url)
        if cap.isOpened():
            # Set camera properties for better quality
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.window_size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window_size[1])
            cap.set(cv2.CAP_PROP_FPS, self.fps_limit)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure
            
        return cap

    def enhance_frame(self, frame):
        """Apply image enhancement techniques"""
        # Convert to float32 for processing
        frame_float = frame.astype(np.float32) / 255.0
        
        # Apply brightness and contrast adjustments
        frame_enhanced = cv2.multiply(frame_float, self.camera_settings['brightness'])
        frame_enhanced = cv2.multiply(frame_enhanced, self.camera_settings['contrast'])
        
        # Apply sharpening
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        frame_enhanced = cv2.filter2D(frame_enhanced, -1, kernel)
        
        # Apply noise reduction
        frame_enhanced = cv2.GaussianBlur(frame_enhanced, (3, 3), 0)
        
        # Convert back to uint8
        frame_enhanced = np.clip(frame_enhanced * 255, 0, 255).astype(np.uint8)
        
        return frame_enhanced

    def detect_humans(self, frame):
        """Enhanced human detection with multiple scale detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect humans with multiple scales
        humans, weights = self.hog.detectMultiScale(
            gray,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05,
            useMeanshiftGrouping=True
        )
        
        # Filter detections based on confidence
        filtered_humans = []
        for (x, y, w, h), weight in zip(humans, weights):
            if weight > 0.3:  # Confidence threshold
                filtered_humans.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        return frame, len(filtered_humans) > 0

    def process_frame(self, frame):
        """Enhanced frame processing with improved motion detection"""
        # Resize and enhance frame
        frame_resized = cv2.resize(frame, self.window_size)
        frame_enhanced = self.enhance_frame(frame_resized)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame_enhanced)
        
        # Store frame history
        self.previous_frames.append(cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2GRAY))
        
        if len(self.previous_frames) < 3:
            return frame_enhanced, {}
            
        # Calculate motion using frame difference of multiple frames
        diff1 = cv2.absdiff(self.previous_frames[-1], self.previous_frames[-2])
        diff2 = cv2.absdiff(self.previous_frames[-2], self.previous_frames[-3])
        motion_mask = cv2.bitwise_or(diff1, diff2)
        
        # Combine motion detection with background subtraction
        combined_mask = cv2.bitwise_and(motion_mask, fg_mask)
        
        # Process grid cells
        height, width = frame_enhanced.shape[:2]
        cell_height = height // self.grid_size[0]
        cell_width = width // self.grid_size[1]
        
        grid_activity = {}
        
        # Detect humans with improved detection
        frame_enhanced, human_detected = self.detect_humans(frame_enhanced)
        
        # Process each grid cell
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                x1 = j * cell_width
                y1 = i * cell_height
                x2 = (j + 1) * cell_width
                y2 = (i + 1) * cell_height
                
                cell_mask = combined_mask[y1:y2, x1:x2]
                activity = np.sum(cell_mask)
                
                grid_index = i * self.grid_size[1] + j
                is_active = activity > self.min_activity_threshold
                grid_activity[grid_index] = is_active
                
                # Dynamic color based on activity level
                if is_active or human_detected:
                    intensity = min(255, int(activity / 1000))
                    color = (0, 255 - intensity, intensity)
                else:
                    color = (0, 255, 0)
                
                cv2.rectangle(frame_enhanced, (x1, y1), (x2, y2), color, 2)
        
        self.grid_activity = grid_activity
        self.human_detected = human_detected
        
        return frame_enhanced, grid_activity

    def run(self):
        """Main processing loop with improved frame handling"""
        cap = self.connect_camera()
        
        if not cap.isOpened():
            print("Error: Could not connect to camera")
            return
            
        last_frame_time = time.time()
        
        try:
            while True:
                # Implement frame rate control
                current_time = time.time()
                elapsed = current_time - last_frame_time
                
                if elapsed < 1.0/self.fps_limit:
                    time.sleep(1.0/self.fps_limit - elapsed)
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, grid_activity = self.process_frame(frame)
                
                # Calculate and display FPS
                fps = 1.0 / (time.time() - last_frame_time)
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Enhanced Grid Motion Detection', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                last_frame_time = time.time()
                
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def get_grid_activity(self):
        return self.grid_activity

    def get_human_detection_status(self):
        return {"human_detected": self.human_detected}

# Flask routes remain the same
@app.route('/status')
def status():
    activity = detector.get_grid_activity()
    human_status = detector.get_human_detection_status()
    return jsonify({**activity, **human_status})

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

def run_web_server():
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    DROID_CAM_URL = "http://10.9.97.32:4747/video"
    
    detector = EnhancedGridMotionDetector(
        camera_url=DROID_CAM_URL,
        grid_size=(3, 2),
        min_activity_threshold=1000,
        fps_limit=30,
        window_size=(1920, 1080)  # Full HD resolution
    )

    web_server_thread = threading.Thread(target=run_web_server)
    web_server_thread.daemon = True
    web_server_thread.start()

    detector.run()