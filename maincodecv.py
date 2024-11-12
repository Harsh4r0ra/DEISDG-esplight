import cv2
import numpy as np
from flask import Flask, jsonify, render_template
import threading

# Initialize Flask app for serving grid data
app = Flask(__name__)

class GridMotionDetector:
    def __init__(self, camera_url, grid_size=(2, 2), min_activity_threshold=1000):
        self.camera_url = camera_url
        self.grid_size = grid_size
        self.min_activity_threshold = min_activity_threshold
        self.previous_frame = None
        self.grid_activity = {}  # This will hold the grid activity state

    def connect_camera(self):
        """Connect to DroidCam stream"""
        return cv2.VideoCapture(self.camera_url)
        
    def process_frame(self, frame):
        """Process frame and divide into grid"""
        height, width = frame.shape[:2]
        cell_height = height // self.grid_size[0]
        cell_width = width // self.grid_size[1]
        
        # Convert frame to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.previous_frame is None:
            self.previous_frame = gray
            return frame, {}
            
        # Calculate frame difference
        frame_diff = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Store current frame for next iteration
        self.previous_frame = gray
        
        # Initialize activity dictionary
        grid_activity = {}
        
        # Process each grid cell
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                # Calculate cell coordinates
                x1 = j * cell_width
                y1 = i * cell_height
                x2 = (j + 1) * cell_width
                y2 = (i + 1) * cell_height
                
                # Extract cell region
                cell_thresh = thresh[y1:y2, x1:x2]
                activity = np.sum(cell_thresh)
                
                # Check if activity exceeds threshold
                grid_index = i * self.grid_size[1] + j
                is_active = activity > self.min_activity_threshold
                grid_activity[grid_index] = is_active
                
                # Draw grid and activity indicator
                color = (0, 255, 0) if is_active else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Store grid activity for web display
        self.grid_activity = grid_activity
        
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
                    
                processed_frame, grid_activity = self.process_frame(frame)
                
                # Display the processed frame
                cv2.imshow('Grid Motion Detection', processed_frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def get_grid_activity(self):
        """Return the current grid activity (for web access)"""
        return self.grid_activity

# Web server route to display grid activity
@app.route('/status')
def status():
    activity = detector.get_grid_activity()
    return jsonify(activity)  # Sends the activity as a JSON response

@app.route('/')
def index():
    # Serve a simple HTML page (can be expanded later)
    return render_template('index.html')  # Create an HTML page to visualize activity

def run_web_server():
    """Start the Flask web server on a separate thread"""
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    # DroidCam URL with Wi-Fi IP address (adjust according to your setup)
    DROID_CAM_URL = "http://10.9.97.32:4747/video"  # Your DroidCam's IP address
    
    # Initialize and run detector
    detector = GridMotionDetector(
        camera_url=DROID_CAM_URL,
        grid_size=(2, 2),  # 2x2 grid for 4 ESP8266 devices
        min_activity_threshold=1000
    )

    # Start Flask web server in a separate thread
    web_server_thread = threading.Thread(target=run_web_server)
    web_server_thread.daemon = True
    web_server_thread.start()

    # Start motion detection
    detector.run()
