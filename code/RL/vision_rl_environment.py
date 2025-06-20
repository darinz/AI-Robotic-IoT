#!/usr/bin/env python3
"""
Vision-Enhanced Reinforcement Learning Environment for KITT

This module provides a reinforcement learning environment that integrates
computer vision capabilities with the existing RL system for advanced
autonomous driving and navigation.

"""

import numpy as np
import time
import cv2
import torch
import torch.nn as nn
from picarx import Picarx
from enum import Enum
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque

# Import vision system
import sys
sys.path.append('../NLP-CV')
from computer_vision import VisionSystem, VisionMode, DetectionResult

class VisionAction(Enum):
    FORWARD = 0
    FORWARD_LEFT = 1
    FORWARD_RIGHT = 2
    BACKWARD = 3
    BACKWARD_LEFT = 4
    BACKWARD_RIGHT = 5
    STOP = 6
    TURN_LEFT = 7
    TURN_RIGHT = 8
    FOLLOW_LINE = 9
    AVOID_OBSTACLE = 10

@dataclass
class VisionState:
    """Enhanced state representation including vision data"""
    sensor_state: Tuple[int, int]  # (distance_state, cliff_state)
    vision_features: np.ndarray    # Vision-based features
    object_detections: List[DetectionResult]
    lane_detection: Optional[Tuple[float, float]]  # (left_lane_angle, right_lane_angle)
    timestamp: float

class VisionRLEnvironment:
    """
    Vision-enhanced reinforcement learning environment for PiCar
    """
    
    def __init__(self, 
                 power=30,
                 safe_distance=40,
                 danger_distance=20,
                 cliff_reference=[200, 200, 200],
                 max_steps=1000,
                 vision_enabled=True,
                 camera_index=0):
        
        self.px = Picarx()
        self.power = power
        self.safe_distance = safe_distance
        self.danger_distance = danger_distance
        self.cliff_reference = cliff_reference
        self.max_steps = max_steps
        self.vision_enabled = vision_enabled
        
        # Set up cliff detection
        self.px.set_cliff_reference(cliff_reference)
        
        # Initialize vision system if enabled
        if self.vision_enabled:
            self.vision_system = VisionSystem(self.px, camera_index)
            self.vision_system.start_vision()
        else:
            self.vision_system = None
        
        # Environment state
        self.step_count = 0
        self.total_reward = 0
        self.collision_count = 0
        self.success_count = 0
        self.vision_success_count = 0
        
        # Action mappings
        self.action_map = {
            VisionAction.FORWARD: (0, self.power),
            VisionAction.FORWARD_LEFT: (-30, self.power),
            VisionAction.FORWARD_RIGHT: (30, self.power),
            VisionAction.BACKWARD: (0, -self.power),
            VisionAction.BACKWARD_LEFT: (-30, -self.power),
            VisionAction.BACKWARD_RIGHT: (30, -self.power),
            VisionAction.STOP: (0, 0),
            VisionAction.TURN_LEFT: (-45, 0),
            VisionAction.TURN_RIGHT: (45, 0),
            VisionAction.FOLLOW_LINE: None,  # Special handling
            VisionAction.AVOID_OBSTACLE: None  # Special handling
        }
        
        # State discretization
        self.distance_bins = [0, 10, 20, 30, 40, 50, 100, float('inf')]
        self.cliff_bins = [False, True]
        
        # Vision feature extraction
        self.vision_feature_size = 64  # Size of vision feature vector
        self.object_history = deque(maxlen=10)  # Track object detections over time
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.vision_accuracy = []
        
    def _extract_vision_features(self) -> np.ndarray:
        """Extract features from vision system"""
        if not self.vision_enabled or self.vision_system is None:
            return np.zeros(self.vision_feature_size)
        
        try:
            # Get current vision state
            vision_state = self.vision_system.get_current_state()
            if vision_state is None:
                return np.zeros(self.vision_feature_size)
            
            features = []
            
            # Object detection features
            detections = vision_state.detections
            self.object_history.append(detections)
            
            # Count objects by type
            object_counts = {}
            for detection in detections:
                obj_type = detection.object_type
                object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
            
            # Add object count features
            for obj_type in ['obstacle', 'line', 'left_lane', 'right_lane', 'red_object', 'green_object']:
                features.append(object_counts.get(obj_type, 0))
            
            # Add distance features (closest objects)
            distances = [det.distance for det in detections if det.distance is not None]
            if distances:
                features.extend([min(distances), max(distances), np.mean(distances)])
            else:
                features.extend([200, 200, 200])  # Default values
            
            # Add confidence features
            confidences = [det.confidence for det in detections]
            if confidences:
                features.extend([min(confidences), max(confidences), np.mean(confidences)])
            else:
                features.extend([0, 0, 0])
            
            # Add position features (center points)
            centers = [det.center for det in detections]
            if centers:
                x_coords = [c[0] for c in centers]
                y_coords = [c[1] for c in centers]
                features.extend([np.mean(x_coords), np.mean(y_coords), np.std(x_coords), np.std(y_coords)])
            else:
                features.extend([320, 240, 0, 0])  # Default to image center
            
            # Add temporal features from object history
            if len(self.object_history) > 1:
                prev_detections = self.object_history[-2]
                movement_features = self._calculate_object_movement(detections, prev_detections)
                features.extend(movement_features)
            else:
                features.extend([0] * 10)  # Default movement features
            
            # Pad or truncate to fixed size
            if len(features) < self.vision_feature_size:
                features.extend([0] * (self.vision_feature_size - len(features)))
            else:
                features = features[:self.vision_feature_size]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error extracting vision features: {e}")
            return np.zeros(self.vision_feature_size)
    
    def _calculate_object_movement(self, current_detections: List[DetectionResult], 
                                 prev_detections: List[DetectionResult]) -> List[float]:
        """Calculate movement features between consecutive frames"""
        features = []
        
        # Match objects between frames (simple center-based matching)
        for curr_det in current_detections:
            min_distance = float('inf')
            best_match = None
            
            for prev_det in prev_detections:
                if curr_det.object_type == prev_det.object_type:
                    distance = np.sqrt((curr_det.center[0] - prev_det.center[0])**2 + 
                                     (curr_det.center[1] - prev_det.center[1])**2)
                    if distance < min_distance:
                        min_distance = distance
                        best_match = prev_det
            
            if best_match is not None:
                # Calculate movement
                dx = curr_det.center[0] - best_match.center[0]
                dy = curr_det.center[1] - best_match.center[1]
                features.extend([dx, dy, np.sqrt(dx**2 + dy**2)])
            else:
                features.extend([0, 0, 0])
        
        # Pad to fixed size
        while len(features) < 10:
            features.append(0)
        
        return features[:10]
    
    def _get_lane_angles(self) -> Optional[Tuple[float, float]]:
        """Extract lane angles from vision system"""
        if not self.vision_enabled or self.vision_system is None:
            return None
        
        try:
            detections = self.vision_system.get_detections()
            left_lane_angle = None
            right_lane_angle = None
            
            for detection in detections:
                if detection.object_type == "left_lane":
                    # Calculate angle from bounding box
                    x, y, w, h = detection.bbox
                    if w > 0:
                        left_lane_angle = np.arctan2(h, w) * 180 / np.pi
                
                elif detection.object_type == "right_lane":
                    x, y, w, h = detection.bbox
                    if w > 0:
                        right_lane_angle = np.arctan2(h, w) * 180 / np.pi
            
            if left_lane_angle is not None or right_lane_angle is not None:
                return (left_lane_angle or 0, right_lane_angle or 0)
            
        except Exception as e:
            print(f"Error getting lane angles: {e}")
        
        return None
    
    def get_state(self) -> VisionState:
        """
        Get current state including vision data
        Returns: VisionState object with sensor and vision information
        """
        try:
            # Get sensor state
            distance = self.px.ultrasonic.read()
            grayscale_data = self.px.get_grayscale_data()
            cliff_status = self.px.get_cliff_status(grayscale_data)
            
            # Discretize distance
            distance_state = 0
            for i, threshold in enumerate(self.distance_bins[:-1]):
                if distance <= threshold:
                    distance_state = i
                    break
            
            # Discretize cliff status
            cliff_state = 1 if cliff_status else 0
            sensor_state = (distance_state, cliff_state)
            
            # Get vision features
            vision_features = self._extract_vision_features()
            
            # Get object detections
            object_detections = []
            if self.vision_enabled and self.vision_system:
                object_detections = self.vision_system.get_detections()
            
            # Get lane detection
            lane_detection = self._get_lane_angles()
            
            return VisionState(
                sensor_state=sensor_state,
                vision_features=vision_features,
                object_detections=object_detections,
                lane_detection=lane_detection,
                timestamp=time.time()
            )
            
        except Exception as e:
            print(f"Error reading sensors: {e}")
            return VisionState(
                sensor_state=(0, 0),
                vision_features=np.zeros(self.vision_feature_size),
                object_detections=[],
                lane_detection=None,
                timestamp=time.time()
            )
    
    def _execute_vision_action(self, action: VisionAction) -> bool:
        """Execute vision-specific actions"""
        if action == VisionAction.FOLLOW_LINE:
            return self._follow_line_action()
        elif action == VisionAction.AVOID_OBSTACLE:
            return self._avoid_obstacle_action()
        else:
            return False
    
    def _follow_line_action(self) -> bool:
        """Execute line following action using vision"""
        try:
            if not self.vision_enabled or self.vision_system is None:
                return False
            
            detections = self.vision_system.get_detections()
            line_detections = [d for d in detections if d.object_type == "line"]
            
            if not line_detections:
                # No line detected, stop
                self.px.stop()
                return True
            
            # Find the most prominent line (highest confidence)
            best_line = max(line_detections, key=lambda x: x.confidence)
            center_x = best_line.center[0]
            
            # Calculate steering based on line position
            image_center = 320  # Assuming 640x480 image
            steering_offset = (center_x - image_center) / image_center
            
            # Apply steering
            steering_angle = int(steering_offset * 45)  # Max 45 degrees
            self.px.set_dir_servo_angle(steering_angle)
            self.px.forward(self.power)
            
            return True
            
        except Exception as e:
            print(f"Error in line following: {e}")
            return False
    
    def _avoid_obstacle_action(self) -> bool:
        """Execute obstacle avoidance action using vision"""
        try:
            if not self.vision_enabled or self.vision_system is None:
                return False
            
            detections = self.vision_system.get_detections()
            obstacles = [d for d in detections if d.object_type == "obstacle"]
            
            if not obstacles:
                # No obstacles, continue forward
                self.px.forward(self.power)
                return True
            
            # Find closest obstacle
            closest_obstacle = min(obstacles, key=lambda x: x.distance or float('inf'))
            
            if closest_obstacle.distance and closest_obstacle.distance < self.danger_distance:
                # Obstacle too close, stop and turn
                self.px.stop()
                time.sleep(0.1)
                
                # Turn away from obstacle
                center_x = closest_obstacle.center[0]
                image_center = 320
                
                if center_x < image_center:
                    # Obstacle on left, turn right
                    self.px.set_dir_servo_angle(30)
                    self.px.backward(self.power // 2)
                else:
                    # Obstacle on right, turn left
                    self.px.set_dir_servo_angle(-30)
                    self.px.backward(self.power // 2)
                
                time.sleep(0.5)
                return True
            
            return True
            
        except Exception as e:
            print(f"Error in obstacle avoidance: {e}")
            return False
    
    def step(self, action):
        """
        Execute action and return (next_state, reward, done, info)
        """
        if self.step_count >= self.max_steps:
            return self.get_state(), 0, True, {"reason": "max_steps"}
        
        # Execute action
        if action in [VisionAction.FOLLOW_LINE, VisionAction.AVOID_OBSTACLE]:
            success = self._execute_vision_action(action)
            if not success:
                # Fallback to basic action
                action = VisionAction.STOP
                steering_angle, power = self.action_map[action]
        else:
            steering_angle, power = self.action_map[VisionAction(action)]
        
        # Apply basic movement if not vision action
        if action not in [VisionAction.FOLLOW_LINE, VisionAction.AVOID_OBSTACLE]:
            self.px.set_dir_servo_angle(steering_angle)
            
            if power > 0:
                self.px.forward(abs(power))
            elif power < 0:
                self.px.backward(abs(power))
            else:
                self.px.stop()
        
        # Wait for action to take effect
        time.sleep(0.1)
        
        # Get new state
        next_state = self.get_state()
        
        # Calculate reward
        reward = self._calculate_vision_reward(next_state, action)
        
        # Check if episode is done
        done = self._is_done(next_state)
        
        # Update counters
        self.step_count += 1
        self.total_reward += reward
        
        if done and reward < -50:  # Collision
            self.collision_count += 1
        elif done and reward > 50:  # Success
            self.success_count += 1
            if self.vision_enabled:
                self.vision_success_count += 1
        
        info = {
            "step": self.step_count,
            "total_reward": self.total_reward,
            "collision_count": self.collision_count,
            "success_count": self.success_count,
            "vision_success_count": self.vision_success_count,
            "vision_enabled": self.vision_enabled
        }
        
        return next_state, reward, done, info
    
    def _calculate_vision_reward(self, state: VisionState, action) -> float:
        """
        Calculate reward based on state and action, including vision information
        """
        distance_state, cliff_state = state.sensor_state
        reward = 0
        
        # Basic sensor rewards
        if cliff_state == 0:
            reward += 1
        else:
            reward -= 50  # Heavy penalty for cliff detection
        
        if distance_state >= 4:  # Safe distance (>40cm)
            reward += 2
        elif distance_state >= 2:  # Moderate distance (20-40cm)
            reward += 0
        else:  # Close distance (<20cm)
            reward -= 30
        
        # Vision-based rewards
        if self.vision_enabled:
            # Reward for successful object detection
            if state.object_detections:
                reward += len(state.object_detections) * 0.5
            
            # Reward for lane detection
            if state.lane_detection:
                left_angle, right_angle = state.lane_detection
                # Reward for staying in lane (angles should be reasonable)
                if abs(left_angle) < 45 and abs(right_angle) < 45:
                    reward += 3
            
            # Reward for vision-based actions
            if action in [VisionAction.FOLLOW_LINE, VisionAction.AVOID_OBSTACLE]:
                reward += 2  # Bonus for using vision
        
        # Action penalties
        if action == VisionAction.STOP.value:
            reward -= 1
        
        if action in [VisionAction.BACKWARD.value, VisionAction.BACKWARD_LEFT.value, 
                     VisionAction.BACKWARD_RIGHT.value]:
            reward -= 2
        
        return reward
    
    def _is_done(self, state: VisionState) -> bool:
        """
        Check if episode should end
        """
        distance_state, cliff_state = state.sensor_state
        
        # Episode ends if cliff detected or very close to obstacle
        if cliff_state == 1 or distance_state == 0:
            return True
        
        # Additional vision-based termination conditions
        if self.vision_enabled:
            # Check if too many obstacles detected
            obstacle_count = len([d for d in state.object_detections if d.object_type == "obstacle"])
            if obstacle_count > 5:  # Too many obstacles
                return True
            
            # Check if no vision features available (camera failure)
            if np.all(state.vision_features == 0):
                return True
        
        return False
    
    def reset(self):
        """
        Reset environment for new episode
        """
        self.px.stop()
        time.sleep(0.2)
        
        # Reset counters
        self.step_count = 0
        self.total_reward = 0
        
        # Clear object history
        self.object_history.clear()
        
        return self.get_state()
    
    def close(self):
        """
        Clean up resources
        """
        self.px.stop()
        time.sleep(0.2)
        
        if self.vision_enabled and self.vision_system:
            self.vision_system.stop_vision()
    
    def get_state_space_size(self):
        """Get size of state space"""
        # Sensor state (2) + vision features (64) + additional features
        return 2 + self.vision_feature_size + 10  # Additional features for object tracking
    
    def get_action_space_size(self):
        """Get size of action space"""
        return len(VisionAction)
    
    def get_vision_statistics(self) -> Dict[str, Any]:
        """Get statistics about vision system performance"""
        if not self.vision_enabled or self.vision_system is None:
            return {"vision_enabled": False}
        
        try:
            vision_stats = self.vision_system.get_vision_statistics()
            vision_stats.update({
                "vision_enabled": True,
                "vision_success_count": self.vision_success_count,
                "object_history_length": len(self.object_history)
            })
            return vision_stats
        except Exception as e:
            return {"vision_enabled": True, "error": str(e)}


def main():
    """Demo function for vision RL environment"""
    try:
        env = VisionRLEnvironment(vision_enabled=True)
        
        print("Vision RL Environment Demo")
        print("Testing different actions...")
        
        # Test basic actions
        for action in [VisionAction.FORWARD, VisionAction.STOP, VisionAction.TURN_LEFT]:
            print(f"Testing action: {action.name}")
            state, reward, done, info = env.step(action.value)
            print(f"Reward: {reward}, Done: {done}")
            time.sleep(1)
        
        # Test vision actions
        if env.vision_enabled:
            print("Testing vision actions...")
            for action in [VisionAction.FOLLOW_LINE, VisionAction.AVOID_OBSTACLE]:
                print(f"Testing vision action: {action.name}")
                state, reward, done, info = env.step(action.value)
                print(f"Reward: {reward}, Done: {done}")
                time.sleep(2)
        
        env.close()
        print("Demo completed")
    
    except Exception as e:
        print(f"Error in demo: {e}")


if __name__ == "__main__":
    main() 