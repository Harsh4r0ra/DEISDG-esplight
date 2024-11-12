import cv2
import numpy as np
import requests  # To send HTTP requests to ESP8266 devices
from flask import Flask, jsonify, render_template
import threading
import time

# Initialize Flask app for serving grid data
app = Flask(__name__)

class GridMotionDetector:
    def __init__(self, camera_url, grid_size=(3, 2), min_activity_threshold=1000, fps_limit=10):
        self.camera_url = camera_url
        self.grid_size = grid_size
        self.min_activity_threshold = min_activity_threshold
        self.previous_frame = None
        self.grid_activity = {}  # This will hold the grid activity state
        self.fps_limit = fps_limit  # Limit FPS
        self.frame_counter = 0  # Frame counter to skip frames if needed

        # IP addresses of each ESP8266 device
        self.esp_ips = [
            "http://192.168.1.101",  # Replace with actual IP addresses for each ESP8266
            "http://192.168.1.102",  #replace all the ip of the esp8266
            "http://192.168.1.103",
            "http://192.168.1.104",
            "http://192.168.1.105",
            "http://192.168.1.106"
        ]

        # Initialize HOG + SVM for human detection
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def connect_camera(self):
        """Connect to DroidCam stream"""
        return cv2.VideoCapture(self.camera_url)

    def control_esp(self, grid_index, state):
        """Send on/off command to the ESP8266 device based on the grid activity"""
        url = f"{self.esp_ips[grid_index]}/control"
        try:
            if state:
                requests.get(f"{url}/on")  # Send 'on' request if grid is red
            else:
                requests.get(f"{url}/off")  # Send 'off' request if grid is green
        except requests.exceptions.RequestException as e:
            print(f"Error controlling ESP8266 {grid_index + 1}: {e}")

    def detect_humans(self, frame):
        """Detect humans using HOG + SVM"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        humans, _ = self.hog.detectMultiScale(gray, winStride=(8, 8), padding=(16, 16), scale=1.05)

        for (x, y, w, h) in humans:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        return frame, len(humans) > 0

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

                if is_active or human_detected:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                
                # Control ESP8266 light for each grid cell
                self.control_esp(grid_index, is_active or human_detected)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        self.grid_activity = grid_activity
        self.human_detected = human_detected
        
        return frame, grid_activity

    def run(self):
        """Main loop for video processing"""
        cap = self.connect_camera()
        
        if not cap.isOpened():
            print("Error: Could not connect to DroidCam")
            return
            
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_counter += 1
                if self.frame_counter % (30 // self.fps_limit) != 0:
                    continue
                
                processed_frame, grid_activity = self.process_frame(frame)
                cv2.imshow('Grid Motion Detection with Human Detection', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(1 / self.fps_limit)
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def get_grid_activity(self):
        """Return the current grid activity (for web access)"""
        return self.grid_activity

    def get_human_detection_status(self):
        """Return the human detection status"""
        return {"human_detected": self.human_detected}

@app.route('/status')
def status():
    activity = detector.get_grid_activity()
    human_status = detector.get_human_detection_status()
    return jsonify({**activity, **human_status})

@app.route('/')
def index():
    return render_template('index.html')

def run_web_server():
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    DROID_CAM_URL = "http://10.9.97.32:4747/video"  # Your DroidCam's IP address #srmisty
    
    detector = GridMotionDetector(
        camera_url=DROID_CAM_URL,
        grid_size=(3, 2),
        min_activity_threshold=1000,
        fps_limit=10
    )

    web_server_thread = threading.Thread(target=run_web_server)
    web_server_thread.daemon = True
    web_server_thread.start()

    detector.run()
