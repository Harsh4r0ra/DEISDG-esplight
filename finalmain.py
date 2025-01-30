import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
import threading
import time
import requests
import logging
from urllib.parse import urlparse
import os

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app with fixed configuration
app = Flask(__name__)
app.config['PROVIDE_AUTOMATIC_OPTIONS'] = True
app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

class GridMotionDetector:
    def __init__(self, camera_url, esp_urls, grid_size=(2, 2), 
                 min_activity_threshold=1000, fps_limit=10, max_retries=3):
        self.camera_url = camera_url
        self.esp_urls = esp_urls
        self.grid_size = grid_size  # 2x2 grid
        self.min_activity_threshold = min_activity_threshold
        self.previous_frame = None
        self.grid_activity = {}
        self.fps_limit = fps_limit
        self.frame_counter = 0
        self.previous_led_states = {}
        self.max_retries = max_retries
        self.human_detected = False
        self.camera = None
        self.manual_override = {}
        
        # Initialize HOG detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Validate URLs
        self._validate_urls()

    def _validate_urls(self):
        """Validate the format of camera and ESP8266 URLs"""
        try:
            camera_parsed = urlparse(self.camera_url)
            if not all([camera_parsed.scheme, camera_parsed.netloc]):
                raise ValueError(f"Invalid camera URL format: {self.camera_url}")
                
            for esp_id, url in self.esp_urls.items():
                esp_parsed = urlparse(url)
                if not all([esp_parsed.scheme, esp_parsed.netloc]):
                    raise ValueError(f"Invalid ESP8266 URL format for ESP {esp_id}: {url}")
                    
        except Exception as e:
            logger.error(f"URL validation failed: {str(e)}")
            raise

    def connect_camera(self):
        """Connect to camera stream with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempting to connect to camera (attempt {attempt + 1}/{self.max_retries})")
                camera = cv2.VideoCapture(self.camera_url)
                
                if camera.isOpened():
                    logger.info("Successfully connected to camera")
                    return camera
                else:
                    logger.warning(f"Failed to open camera on attempt {attempt + 1}")
                    if camera is not None:
                        camera.release()
                    
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Error connecting to camera: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                    
        raise ConnectionError("Failed to connect to camera after maximum retries")

    def send_esp_command(self, esp_number, state):
        """Send command directly to ESP8266 with retry mechanism"""
        if esp_number not in self.esp_urls:
            logger.error(f"No URL configured for ESP {esp_number}")
            return False
            
        url = self.esp_urls[esp_number]
        command = "on" if state else "off"
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(f"{url}/{command}", timeout=0.5)
                
                if response.status_code == 200:
                    logger.info(f"Successfully sent command {command} to ESP {esp_number}")
                    return True
                else:
                    logger.warning(f"Failed to send command to ESP {esp_number}. Status: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Error sending command to ESP {esp_number}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    
        return False

    def set_manual_override(self, esp_number, state):
        """Set manual override for a specific ESP8266"""
        self.manual_override[esp_number] = state
        self.send_esp_command(esp_number, state)

    def clear_manual_override(self, esp_number):
        """Clear manual override for a specific ESP8266"""
        if esp_number in self.manual_override:
            del self.manual_override[esp_number]

    def process_frame(self, frame):
        """Process frame and divide into grid"""
        height, width = frame.shape[:2]
        cell_height = height // self.grid_size[0]
        cell_width = width // self.grid_size[1]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.previous_frame is None:
            self.previous_frame = gray
            return frame, {}
            
        frame_diff = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        self.previous_frame = gray
        
        grid_activity = {}
        frame, human_detected = self.detect_humans(frame)

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                x1 = j * cell_width
                y1 = i * cell_height
                x2 = (j + 1) * cell_width
                y2 = (i + 1) * cell_height
                
                cell_thresh = thresh[y1:y2, x1:x2]
                activity = np.sum(cell_thresh)
                
                grid_index = i * self.grid_size[1] + j
                is_active = activity > self.min_activity_threshold
                grid_activity[grid_index] = is_active
                
                color = (0, 0, 255) if (is_active or human_detected) else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        self.update_esp_states(grid_activity, human_detected)
        self.grid_activity = grid_activity
        self.human_detected = human_detected
        
        return frame, grid_activity

    def detect_humans(self, frame):
        """Detect humans using HOG + SVM"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        humans, _ = self.hog.detectMultiScale(gray, winStride=(8, 8), padding=(16, 16), scale=1.05)
        
        for (x, y, w, h) in humans:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
        return frame, len(humans) > 0

    def update_esp_states(self, grid_activity, human_detected):
        """Update ESP8266 LED states based on grid activity and manual overrides"""
        for grid_index, is_active in grid_activity.items():
            esp_number = grid_index + 1
            
            # Skip if manual override is active
            if esp_number in self.manual_override:
                continue
                
            # Turn on LED if there's motion in the grid or a human is detected
            new_state = is_active or human_detected
            
            # Only send command if state has changed
            if self.previous_led_states.get(esp_number) != new_state:
                self.send_esp_command(esp_number, new_state)
                self.previous_led_states[esp_number] = new_state

    def run(self):
        """Main loop for video processing"""
        while True:
            try:
                if self.camera is None or not self.camera.isOpened():
                    self.camera = self.connect_camera()
                
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to read frame, attempting to reconnect...")
                    self.camera.release()
                    self.camera = None
                    continue
                
                self.frame_counter += 1
                if self.frame_counter % (30 // self.fps_limit) != 0:
                    continue
                
                processed_frame, grid_activity = self.process_frame(frame)
                cv2.imshow('Grid Motion Detection with Human Detection', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(1 / self.fps_limit)
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                if self.camera is not None:
                    self.camera.release()
                    self.camera = None
                time.sleep(2)
                
        self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        try:
            # Turn off all LEDs
            for esp_number in self.esp_urls.keys():
                self.send_esp_command(esp_number, False)
            
            if self.camera is not None:
                self.camera.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def get_grid_activity(self):
        """Return current grid activity"""
        return self.grid_activity

    def get_human_detection_status(self):
        """Return human detection status"""
        return {"human_detected": self.human_detected}

# Global detector instance
detector = None

@app.route('/status')
def status():
    """API endpoint to get current status"""
    if detector is None:
        return jsonify({"error": "Detector not initialized"}), 500
    
    activity = detector.get_grid_activity()
    human_status = detector.get_human_detection_status()
    manual_overrides = detector.manual_override
    
    return jsonify({
        "grid_activity": activity,
        "human_detected": human_status["human_detected"],
        "manual_overrides": manual_overrides
    })

@app.route('/override', methods=['POST'])
def override():
    """API endpoint to set manual override"""
    if detector is None:
        return jsonify({"error": "Detector not initialized"}), 500
        
    try:
        data = request.get_json()
        esp_number = int(data['esp_number'])
        
        if data.get('clear', False):
            detector.clear_manual_override(esp_number)
            return jsonify({"message": f"Manual override cleared for ESP {esp_number}"})
        else:
            state = bool(data['state'])
            detector.set_manual_override(esp_number, state)
            return jsonify({"message": f"Manual override set for ESP {esp_number}"})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/')
def index():
    """Serve the main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving index page: {str(e)}")
        return f"An error occurred: {str(e)}", 500

def run_web_server():
    """Run the Flask web server"""
    try:
        logger.info("Starting web server...")
        app.run(host='0.0.0.0', port=5000, use_reloader=False)
    except Exception as e:
        logger.error(f"Web server error: {str(e)}")
        raise

def run_with_error_handling():
    """Run the application with error handling"""
    global detector
    
    try:
        # Configuration
        DROID_CAM_URL = "http://192.168.137.94:4747/video"
        ESP_URLS = {
            1: "http://192.168.137.101",  # ESP8266 #1
            2: "http://192.168.137.102",  # ESP8266 #2
            3: "http://192.168.137.103",  # ESP8266 #3
            4: "http://192.168.137.104",  # ESP8266 #4
        }
        
        # Initialize detector
        detector = GridMotionDetector(
            camera_url=DROID_CAM_URL,
            esp_urls=ESP_URLS,
            grid_size=(2, 2),  # 2x2 grid
            min_activity_threshold=1000,
            fps_limit=10,
            max_retries=3
        )

        # Start Flask web server
        web_server_thread = threading.Thread(target=run_web_server)
        web_server_thread.daemon = True
        web_server_thread.start()

        # Start motion detection
        detector.run()
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    run_with_error_handling()