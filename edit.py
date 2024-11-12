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

# Initialize Flask app
app = Flask(__name__)
app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

class GridMotionDetector:
    def __init__(self, camera_url, esp_urls, grid_size=(2, 2), 
                 min_activity_threshold=1000, fps_limit=10, max_retries=3):
        self.camera_url = camera_url
        self.esp_urls = esp_urls
        self.grid_size = grid_size
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
        
        # Initialize CUDA-enabled HOG detector
        self.init_cuda_hog()
        
        # Create CUDA-enabled GPU matrices
        self.cuda_stream = cv2.cuda.Stream()
        
        # Validate URLs
        self._validate_urls()

    def init_cuda_hog(self):
        """Initialize CUDA-enabled HOG detector"""
        try:
            # Check if CUDA is available
            if not cv2.cuda.getCudaEnabledDeviceCount():
                logger.warning("No CUDA devices available, falling back to CPU")
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                self.use_cuda = False
            else:
                logger.info("Initializing CUDA-enabled HOG detector")
                self.hog = cv2.cuda_HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                self.use_cuda = True
        except Exception as e:
            logger.error(f"Error initializing CUDA HOG: {str(e)}")
            logger.warning("Falling back to CPU HOG detector")
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.use_cuda = False

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

    def process_frame(self, frame):
        """Process frame using CUDA acceleration"""
        try:
            height, width = frame.shape[:2]
            cell_height = height // self.grid_size[0]
            cell_width = width // self.grid_size[1]
            
            # Convert frame to CUDA GpuMat
            if self.use_cuda:
                cuda_frame = cv2.cuda_GpuMat()
                cuda_frame.upload(frame)
                
                # Convert to grayscale on GPU
                cuda_gray = cv2.cuda.cvtColor(cuda_frame, cv2.COLOR_BGR2GRAY, stream=self.cuda_stream)
                cuda_gray = cv2.cuda.GaussianBlur(cuda_gray, (21, 21), 0, stream=self.cuda_stream)
                
                if self.previous_frame is None:
                    self.previous_frame = cuda_gray
                    return frame, {}
                
                # Compute frame difference on GPU
                cuda_diff = cv2.cuda.absdiff(self.previous_frame, cuda_gray, stream=self.cuda_stream)
                self.previous_frame = cuda_gray
                
                # Threshold on GPU
                _, cuda_thresh = cv2.cuda.threshold(cuda_diff, 25, 255, cv2.THRESH_BINARY, stream=self.cuda_stream)
                
                # Download results back to CPU for grid processing
                thresh = cuda_thresh.download()
            else:
                # Fall back to CPU processing if CUDA is not available
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
            
        except cv2.error as e:
            logger.error(f"OpenCV error in process_frame: {str(e)}")
            return frame, {}
        except Exception as e:
            logger.error(f"Error in process_frame: {str(e)}")
            return frame, {}

    def detect_humans(self, frame):
        """Detect humans using CUDA-accelerated HOG + SVM"""
        try:
            if self.use_cuda:
                # Upload frame to GPU
                cuda_frame = cv2.cuda_GpuMat()
                cuda_frame.upload(frame)
                
                # Convert to grayscale on GPU
                cuda_gray = cv2.cuda.cvtColor(cuda_frame, cv2.COLOR_BGR2GRAY, stream=self.cuda_stream)
                
                # Detect humans using CUDA HOG
                humans = self.hog.detectMultiScale(cuda_gray, hit_threshold=0, 
                                                 win_stride=(8, 8),
                                                 padding=(16, 16),
                                                 scale=1.05)
            else:
                # Fall back to CPU detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                humans, _ = self.hog.detectMultiScale(gray, 
                                                    winStride=(8, 8),
                                                    padding=(16, 16),
                                                    scale=1.05)

            # Draw rectangles around detected humans
            for (x, y, w, h) in humans:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                
            return frame, len(humans) > 0
            
        except cv2.error as e:
            logger.error(f"OpenCV error in detect_humans: {str(e)}")
            return frame, False
        except Exception as e:
            logger.error(f"Error in detect_humans: {str(e)}")
            return frame, False

    def cleanup(self):
        """Cleanup resources including CUDA resources"""
        logger.info("Cleaning up resources...")
        try:
            # Turn off all LEDs
            for esp_number in self.esp_urls.keys():
                self.send_esp_command(esp_number, False)
            
            if self.camera is not None:
                self.camera.release()
            
            # Release CUDA resources
            if self.use_cuda:
                self.cuda_stream.free()
                cv2.cuda.destroyAllWindows()
            else:
                cv2.destroyAllWindows()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    # [Rest of the class methods remain unchanged]

# [Rest of the Flask routes and main execution code remain unchanged]