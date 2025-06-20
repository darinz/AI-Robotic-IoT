import numpy as np
import json
import os
import pickle
from datetime import datetime
import random

class QLearningAgent:
    """
    Q-Learning agent for autonomous car navigation
    Implements traditional Q-learning with persistent storage
    """
    
    def __init__(self, 
                 state_space_size,
                 action_space_size,
                 learning_rate=0.1,
                 discount_factor=0.95,
                 epsilon=0.1,
                 epsilon_decay=0.995,
                 epsilon_min=0.01,
                 model_path="q_learning_model.pkl"):
        
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model_path = model_path
        
        # Initialize Q-table
        self.q_table = self._initialize_q_table()
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.collision_rates = []
        self.success_rates = []
        
        # Load existing model if available
        self.load_model()
    
    def _initialize_q_table(self):
        """
        Initialize Q-table with zeros
        """
        # Q-table shape: (distance_states, cliff_states, actions)
        distance_states, cliff_states = self.state_space_size
        return np.zeros((distance_states, cliff_states, self.action_space_size))
    
    def _state_to_index(self, state):
        """
        Convert state tuple to Q-table indices
        """
        distance_state, cliff_state = state
        return distance_state, cliff_state
    
    def choose_action(self, state):
        """
        Choose action using epsilon-greedy policy
        """
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.action_space_size - 1)
        else:
            # Exploitation: best action
            distance_idx, cliff_idx = self._state_to_index(state)
            return np.argmax(self.q_table[distance_idx, cliff_idx])
    
    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-table using Q-learning update rule
        """
        distance_idx, cliff_idx = self._state_to_index(state)
        next_distance_idx, next_cliff_idx = self._state_to_index(next_state)
        
        # Current Q-value
        current_q = self.q_table[distance_idx, cliff_idx, action]
        
        # Next state's maximum Q-value
        if done:
            next_max_q = 0
        else:
            next_max_q = np.max(self.q_table[next_distance_idx, next_cliff_idx])
        
        # Q-learning update rule (Bellman equation)
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        
        # Update Q-table
        self.q_table[distance_idx, cliff_idx, action] = new_q
    
    def update_epsilon(self):
        """
        Decay epsilon for exploration-exploitation balance
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self):
        """
        Save Q-table and training statistics to disk
        """
        model_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'collision_rates': self.collision_rates,
            'success_rates': self.success_rates,
            'timestamp': datetime.now().isoformat(),
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min
            }
        }
        
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {self.model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self):
        """
        Load Q-table and training statistics from disk
        """
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.q_table = model_data['q_table']
                self.epsilon = model_data['epsilon']
                self.episode_rewards = model_data.get('episode_rewards', [])
                self.episode_lengths = model_data.get('episode_lengths', [])
                self.collision_rates = model_data.get('collision_rates', [])
                self.success_rates = model_data.get('success_rates', [])
                
                print(f"Model loaded from {self.model_path}")
                print(f"Previous training: {len(self.episode_rewards)} episodes")
                print(f"Current epsilon: {self.epsilon:.4f}")
                
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Starting with fresh Q-table")
        else:
            print("No existing model found. Starting fresh.")
    
    def get_policy(self, state):
        """
        Get the best action for a given state (no exploration)
        """
        distance_idx, cliff_idx = self._state_to_index(state)
        return np.argmax(self.q_table[distance_idx, cliff_idx])
    
    def get_q_values(self, state):
        """
        Get Q-values for all actions in a given state
        """
        distance_idx, cliff_idx = self._state_to_index(state)
        return self.q_table[distance_idx, cliff_idx].copy()
    
    def update_statistics(self, episode_reward, episode_length, collision, success):
        """
        Update training statistics
        """
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        # Calculate rates over last 100 episodes
        window = min(100, len(self.episode_rewards))
        if window > 0:
            recent_collisions = sum(1 for i in range(max(0, len(self.episode_rewards) - window), len(self.episode_rewards)) 
                                  if self.episode_rewards[i] < -50)
            recent_successes = sum(1 for i in range(max(0, len(self.episode_rewards) - window), len(self.episode_rewards)) 
                                 if self.episode_rewards[i] > 50)
            
            self.collision_rates.append(recent_collisions / window)
            self.success_rates.append(recent_successes / window)
    
    def print_statistics(self):
        """
        Print current training statistics
        """
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-100:]
            recent_lengths = self.episode_lengths[-100:]
            
            print(f"\nTraining Statistics:")
            print(f"Total Episodes: {len(self.episode_rewards)}")
            print(f"Current Epsilon: {self.epsilon:.4f}")
            print(f"Recent Avg Reward: {np.mean(recent_rewards):.2f}")
            print(f"Recent Avg Length: {np.mean(recent_lengths):.2f}")
            
            if len(self.collision_rates) > 0:
                print(f"Recent Collision Rate: {self.collision_rates[-1]:.2%}")
            if len(self.success_rates) > 0:
                print(f"Recent Success Rate: {self.success_rates[-1]:.2%}")
    
    def export_q_table(self, filename="q_table_export.json"):
        """
        Export Q-table to human-readable JSON format
        """
        try:
            # Convert numpy array to list for JSON serialization
            q_table_list = self.q_table.tolist()
            
            export_data = {
                'q_table': q_table_list,
                'state_space_size': self.state_space_size,
                'action_space_size': self.action_space_size,
                'timestamp': datetime.now().isoformat(),
                'epsilon': self.epsilon
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"Q-table exported to {filename}")
            
        except Exception as e:
            print(f"Error exporting Q-table: {e}") 