#!/usr/bin/env python3
"""
Computer Vision System for PiCar

This module provides advanced computer vision capabilities for the PiCar,
including object detection, line following, obstacle avoidance, and
integration with reinforcement learning.

"""

import cv2
import numpy as np
import threading
import time
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
from picarx import Picarx
import sys
sys.path.append('.')
from config import get_config

class VisionMode(Enum):
    OBSTACLE_DETECTION = "obstacle_detection"
    LINE_FOLLOWING = "line_following"
    OBJECT_TRACKING = "object_tracking"
    LANE_DETECTION = "lane_detection"
    COLOR_DETECTION = "color_detection"

@dataclass
class DetectionResult:
    """Represents a detection result from computer vision"""
    object_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[int, int]
    distance: Optional[float] = None

@dataclass
class VisionState:
    """Represents the current state of the vision system"""
    mode: VisionMode
    frame: np.ndarray
    detections: List[DetectionResult]
    processed_frame: np.ndarray
    timestamp: float

class VisionSystem:
    """
    Advanced computer vision system for PiCar
    """
    
    def __init__(self, car: Picarx, camera_index: int = None):
        self.car = car
        
        # Use configuration if available
        config = get_config()
        self.camera_index = camera_index if camera_index is not None else config.camera_index
        self.vision_enabled = config.vision_enabled
        
        self.cap = None
        
        # Vision state
        self.current_state: Optional[VisionState] = None
        self.vision_mode = VisionMode.OBSTACLE_DETECTION
        
        # Threading
        self.is_running = False
        self.vision_thread = None
        self.stop_vision = threading.Event()
        
        # Configuration
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30
        
        # Detection parameters
        self.obstacle_threshold = 0.5
        self.line_threshold = 100
        self.color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'green': ([40, 100, 100], [80, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255]),
            'yellow': ([20, 100, 100], [30, 255, 255])
        }
        
        # Neural network models (placeholder for future implementation)
        self.object_detection_model = None
        self.lane_detection_model = None
        
        # Initialize camera
        self._init_camera()
        
        # Load pre-trained models if available
        self._load_models()
    
    def _init_camera(self):
        """Initialize camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera")
            
            print("Camera initialized successfully")
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.cap = None
    
    def _load_models(self):
        """Load pre-trained neural network models"""
        try:
            # Load YOLO model for object detection (if available)
            model_path = "models/yolov5s.pt"
            if os.path.exists(model_path):
                self.object_detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                print("Object detection model loaded")
            
            # Load lane detection model (placeholder)
            # self.lane_detection_model = load_lane_detection_model()
            
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def start_vision(self):
        """Start vision processing in a separate thread"""
        if self.is_running or self.cap is None:
            return
        
        self.is_running = True
        self.stop_vision.clear()
        self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.vision_thread.start()
        print("Vision system started")
    
    def stop_vision(self):
        """Stop vision processing"""
        self.is_running = False
        self.stop_vision.set()
        if self.vision_thread:
            self.vision_thread.join()
        if self.cap:
            self.cap.release()
        print("Vision system stopped")
    
    def _vision_loop(self):
        """Main vision processing loop"""
        while self.is_running and not self.stop_vision.is_set():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Process frame based on current mode
                processed_frame, detections = self._process_frame(frame)
                
                # Update current state
                self.current_state = VisionState(
                    mode=self.vision_mode,
                    frame=frame,
                    detections=detections,
                    processed_frame=processed_frame,
                    timestamp=time.time()
                )
                
                # Control frame rate
                time.sleep(1.0 / self.fps)
                
            except Exception as e:
                print(f"Error in vision loop: {e}")
                time.sleep(0.1)
    
    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectionResult]]:
        """Process a single frame based on current vision mode"""
        if self.vision_mode == VisionMode.OBSTACLE_DETECTION:
            return self._detect_obstacles(frame)
        elif self.vision_mode == VisionMode.LINE_FOLLOWING:
            return self._detect_lines(frame)
        elif self.vision_mode == VisionMode.OBJECT_TRACKING:
            return self._track_objects(frame)
        elif self.vision_mode == VisionMode.LANE_DETECTION:
            return self._detect_lanes(frame)
        elif self.vision_mode == VisionMode.COLOR_DETECTION:
            return self._detect_colors(frame)
        else:
            return frame, []
    
    def _detect_obstacles(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectionResult]]:
        """Detect obstacles using computer vision"""
        detections = []
        processed_frame = frame.copy()
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    center = (x + w // 2, y + h // 2)
                    
                    # Calculate distance (simplified)
                    distance = self._estimate_distance(w, h)
                    
                    detection = DetectionResult(
                        object_type="obstacle",
                        confidence=min(area / 10000, 1.0),  # Simple confidence based on area
                        bbox=(x, y, w, h),
                        center=center,
                        distance=distance
                    )
                    detections.append(detection)
                    
                    # Draw bounding box
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(processed_frame, f"Obstacle: {distance:.1f}cm", 
                              (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        except Exception as e:
            print(f"Error in obstacle detection: {e}")
        
        return processed_frame, detections
    
    def _detect_lines(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectionResult]]:
        """Detect lines for line following"""
        detections = []
        processed_frame = frame.copy()
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Threshold to get binary image
            _, binary = cv2.threshold(blurred, self.line_threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Find lines using Hough transform
            lines = cv2.HoughLinesP(binary, 1, np.pi/180, threshold=50, 
                                  minLineLength=50, maxLineGap=10)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate line properties
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    detection = DetectionResult(
                        object_type="line",
                        confidence=min(length / 200, 1.0),
                        bbox=(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)),
                        center=center
                    )
                    detections.append(detection)
                    
                    # Draw line
                    cv2.line(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        except Exception as e:
            print(f"Error in line detection: {e}")
        
        return processed_frame, detections
    
    def _track_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectionResult]]:
        """Track objects using neural network models"""
        detections = []
        processed_frame = frame.copy()
        
        try:
            if self.object_detection_model is not None:
                # Use YOLO model for object detection
                results = self.object_detection_model(frame)
                
                for det in results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                    
                    if conf > self.obstacle_threshold:
                        bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        
                        detection = DetectionResult(
                            object_type=results.names[int(cls)],
                            confidence=float(conf),
                            bbox=bbox,
                            center=center
                        )
                        detections.append(detection)
                        
                        # Draw bounding box
                        cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                    (0, 255, 0), 2)
                        cv2.putText(processed_frame, f"{results.names[int(cls)]}: {conf:.2f}", 
                                  (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"Error in object tracking: {e}")
        
        return processed_frame, detections
    
    def _detect_lanes(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectionResult]]:
        """Detect lanes for autonomous driving"""
        detections = []
        processed_frame = frame.copy()
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply region of interest mask
            height, width = gray.shape
            roi_vertices = np.array([
                [(0, height), (width/2, height/2), (width, height)]
            ], dtype=np.int32)
            
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, roi_vertices, 255)
            masked_image = cv2.bitwise_and(gray, mask)
            
            # Edge detection
            edges = cv2.Canny(masked_image, 50, 150)
            
            # Find lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                  minLineLength=100, maxLineGap=50)
            
            if lines is not None:
                left_lines = []
                right_lines = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
                    
                    if slope < -0.5:  # Left lane
                        left_lines.append(line)
                    elif slope > 0.5:  # Right lane
                        right_lines.append(line)
                
                # Process left lane
                if left_lines:
                    left_line = np.mean(left_lines, axis=0)
                    x1, y1, x2, y2 = left_line[0]
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    detection = DetectionResult(
                        object_type="left_lane",
                        confidence=0.8,
                        bbox=(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)),
                        center=center
                    )
                    detections.append(detection)
                    cv2.line(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
                
                # Process right lane
                if right_lines:
                    right_line = np.mean(right_lines, axis=0)
                    x1, y1, x2, y2 = right_line[0]
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    detection = DetectionResult(
                        object_type="right_lane",
                        confidence=0.8,
                        bbox=(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)),
                        center=center
                    )
                    detections.append(detection)
                    cv2.line(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            
        except Exception as e:
            print(f"Error in lane detection: {e}")
        
        return processed_frame, detections
    
    def _detect_colors(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectionResult]]:
        """Detect specific colors in the frame"""
        detections = []
        processed_frame = frame.copy()
        
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            for color_name, (lower, upper) in self.color_ranges.items():
                # Create mask for color
                lower = np.array(lower)
                upper = np.array(upper)
                mask = cv2.inRange(hsv, lower, upper)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:  # Minimum area threshold
                        x, y, w, h = cv2.boundingRect(contour)
                        center = (x + w // 2, y + h // 2)
                        
                        detection = DetectionResult(
                            object_type=f"{color_name}_object",
                            confidence=min(area / 5000, 1.0),
                            bbox=(x, y, w, h),
                            center=center
                        )
                        detections.append(detection)
                        
                        # Draw bounding box
                        color_bgr = self._get_color_bgr(color_name)
                        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color_bgr, 2)
                        cv2.putText(processed_frame, color_name, (x, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
            
        except Exception as e:
            print(f"Error in color detection: {e}")
        
        return processed_frame, detections
    
    def _estimate_distance(self, width: int, height: int) -> float:
        """Estimate distance to object based on bounding box size"""
        # Simplified distance estimation
        # In a real implementation, this would use camera calibration
        area = width * height
        distance = 1000 / (area ** 0.5)  # Inverse relationship
        return max(10, min(200, distance))  # Clamp between 10-200cm
    
    def _get_color_bgr(self, color_name: str) -> Tuple[int, int, int]:
        """Get BGR color tuple for drawing"""
        color_map = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255)
        }
        return color_map.get(color_name, (255, 255, 255))
    
    def set_vision_mode(self, mode: VisionMode):
        """Set the current vision mode"""
        self.vision_mode = mode
        print(f"Vision mode changed to: {mode.value}")
    
    def get_current_state(self) -> Optional[VisionState]:
        """Get the current vision state"""
        return self.current_state
    
    def get_detections(self) -> List[DetectionResult]:
        """Get current detections"""
        if self.current_state:
            return self.current_state.detections
        return []
    
    def get_processed_frame(self) -> Optional[np.ndarray]:
        """Get the current processed frame"""
        if self.current_state:
            return self.current_state.processed_frame
        return None
    
    def save_frame(self, filename: str = "vision_frame.jpg"):
        """Save current frame to file"""
        if self.current_state:
            cv2.imwrite(filename, self.current_state.processed_frame)
            print(f"Frame saved to {filename}")
    
    def get_vision_statistics(self) -> Dict[str, Any]:
        """Get statistics about vision processing"""
        if not self.current_state:
            return {}
        
        return {
            "mode": self.vision_mode.value,
            "detection_count": len(self.current_state.detections),
            "timestamp": self.current_state.timestamp,
            "frame_shape": self.current_state.frame.shape
        }


def main():
    """Demo function for vision system"""
    try:
        car = Picarx()
        vision = VisionSystem(car)
        
        print("Vision System Demo")
        print("Press 'q' to quit, 's' to save frame")
        
        vision.start_vision()
        
        while True:
            # Get current state
            state = vision.get_current_state()
            if state:
                # Display frame
                cv2.imshow("Vision System", state.processed_frame)
                
                # Show statistics
                stats = vision.get_vision_statistics()
                print(f"Mode: {stats.get('mode', 'unknown')}, "
                      f"Detections: {stats.get('detection_count', 0)}")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                vision.save_frame()
            elif key == ord('1'):
                vision.set_vision_mode(VisionMode.OBSTACLE_DETECTION)
            elif key == ord('2'):
                vision.set_vision_mode(VisionMode.LINE_FOLLOWING)
            elif key == ord('3'):
                vision.set_vision_mode(VisionMode.OBJECT_TRACKING)
            elif key == ord('4'):
                vision.set_vision_mode(VisionMode.LANE_DETECTION)
            elif key == ord('5'):
                vision.set_vision_mode(VisionMode.COLOR_DETECTION)
        
        vision.stop_vision()
        cv2.destroyAllWindows()
        print("Demo ended")
    
    except Exception as e:
        print(f"Error in demo: {e}")


if __name__ == "__main__":
    main() 