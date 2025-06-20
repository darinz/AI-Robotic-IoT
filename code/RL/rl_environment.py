import numpy as np
import time
from picarx import Picarx
from enum import Enum
import json
import os

class Action(Enum):
    FORWARD = 0
    FORWARD_LEFT = 1
    FORWARD_RIGHT = 2
    BACKWARD = 3
    BACKWARD_LEFT = 4
    BACKWARD_RIGHT = 5
    STOP = 6

class CarEnvironment:
    """
    Reinforcement Learning Environment for PicarX autonomous car
    Provides a gym-like interface for RL training
    """
    
    def __init__(self, 
                 power=30,  # Reduced power for safety during learning
                 safe_distance=40,
                 danger_distance=20,
                 cliff_reference=[200, 200, 200],
                 max_steps=1000):
        
        self.px = Picarx()
        self.power = power
        self.safe_distance = safe_distance
        self.danger_distance = danger_distance
        self.cliff_reference = cliff_reference
        self.max_steps = max_steps
        
        # Set up cliff detection
        self.px.set_cliff_reference(cliff_reference)
        
        # Environment state
        self.step_count = 0
        self.total_reward = 0
        self.collision_count = 0
        self.success_count = 0
        
        # Action mappings
        self.action_map = {
            Action.FORWARD: (0, self.power),      # (steering_angle, power)
            Action.FORWARD_LEFT: (-30, self.power),
            Action.FORWARD_RIGHT: (30, self.power),
            Action.BACKWARD: (0, -self.power),
            Action.BACKWARD_LEFT: (-30, -self.power),
            Action.BACKWARD_RIGHT: (30, -self.power),
            Action.STOP: (0, 0)
        }
        
        # State discretization
        self.distance_bins = [0, 10, 20, 30, 40, 50, 100, float('inf')]
        self.cliff_bins = [False, True]  # Safe, Danger
        
    def get_state(self):
        """
        Get current state from sensors
        Returns: discretized state tuple (distance_state, cliff_state)
        """
        try:
            # Get ultrasonic distance
            distance = self.px.ultrasonic.read()
            
            # Get cliff/grayscale status
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
            
            return (distance_state, cliff_state)
            
        except Exception as e:
            print(f"Error reading sensors: {e}")
            return (0, 0)  # Default safe state
    
    def step(self, action):
        """
        Execute action and return (next_state, reward, done, info)
        """
        if self.step_count >= self.max_steps:
            return self.get_state(), 0, True, {"reason": "max_steps"}
        
        # Execute action
        steering_angle, power = self.action_map[Action(action)]
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
        reward = self._calculate_reward(next_state, action)
        
        # Check if episode is done
        done = self._is_done(next_state)
        
        # Update counters
        self.step_count += 1
        self.total_reward += reward
        
        if done and reward < -50:  # Collision
            self.collision_count += 1
        elif done and reward > 50:  # Success
            self.success_count += 1
        
        info = {
            "step": self.step_count,
            "total_reward": self.total_reward,
            "collision_count": self.collision_count,
            "success_count": self.success_count
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, state, action):
        """
        Calculate reward based on state and action
        """
        distance_state, cliff_state = state
        
        reward = 0
        
        # Reward for staying safe (no cliff detected)
        if cliff_state == 0:
            reward += 1
        else:
            reward -= 50  # Heavy penalty for cliff detection
        
        # Reward based on distance to obstacles
        if distance_state >= 4:  # Safe distance (>40cm)
            reward += 2
        elif distance_state >= 2:  # Moderate distance (20-40cm)
            reward += 0
        else:  # Close distance (<20cm)
            reward -= 30
        
        # Small penalty for stopping to encourage movement
        if action == Action.STOP.value:
            reward -= 1
        
        # Small penalty for backward movement to encourage forward progress
        if action in [Action.BACKWARD.value, Action.BACKWARD_LEFT.value, Action.BACKWARD_RIGHT.value]:
            reward -= 2
        
        return reward
    
    def _is_done(self, state):
        """
        Check if episode should end
        """
        distance_state, cliff_state = state
        
        # Episode ends if cliff detected or very close to obstacle
        if cliff_state == 1 or distance_state == 0:
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
        
        return self.get_state()
    
    def close(self):
        """
        Clean up resources
        """
        self.px.stop()
        time.sleep(0.2)
    
    def get_state_space_size(self):
        """
        Get size of state space for Q-learning
        """
        return len(self.distance_bins) - 1, len(self.cliff_bins)
    
    def get_action_space_size(self):
        """
        Get number of possible actions
        """
        return len(Action) 