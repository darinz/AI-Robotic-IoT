import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import pickle
import os
from datetime import datetime

class DQN(nn.Module):
    """
    Deep Q-Network architecture
    """
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """
    Experience replay buffer for DQN
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), action, np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    Deep Q-Network agent for autonomous car navigation
    Implements DQN with experience replay and target network
    """
    
    def __init__(self, 
                 state_size,
                 action_size,
                 learning_rate=0.001,
                 discount_factor=0.95,
                 epsilon=1.0,
                 epsilon_decay=0.995,
                 epsilon_min=0.01,
                 memory_size=10000,
                 batch_size=32,
                 target_update=100,
                 model_path="dqn_model.pkl"):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update = target_update
        self.model_path = model_path
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Networks
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = ReplayBuffer(memory_size)
        
        # Training variables
        self.step_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        
        # Load existing model if available
        self.load_model()
        
        # Initialize target network
        self.update_target_network()
    
    def _state_to_tensor(self, state):
        """
        Convert state tuple to tensor
        """
        # Flatten state tuple to single vector
        state_vector = np.array([state[0], state[1]], dtype=np.float32)
        return torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
    
    def choose_action(self, state):
        """
        Choose action using epsilon-greedy policy
        """
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploitation: best action
            state_tensor = self._state_to_tensor(state)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def learn(self, state, action, reward, next_state, done):
        """
        Train the DQN on a single experience
        """
        # Store experience in replay buffer
        self.memory.push(state, action, reward, next_state, done)
        
        # Only learn if we have enough samples
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.update_target_network()
        
        # Store loss for monitoring
        self.losses.append(loss.item())
    
    def update_target_network(self):
        """
        Update target network with current Q-network weights
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_epsilon(self):
        """
        Decay epsilon for exploration-exploitation balance
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self):
        """
        Save DQN model and training statistics to disk
        """
        model_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'memory': self.memory,
            'timestamp': datetime.now().isoformat(),
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'batch_size': self.batch_size,
                'target_update': self.target_update
            }
        }
        
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"DQN model saved to {self.model_path}")
        except Exception as e:
            print(f"Error saving DQN model: {e}")
    
    def load_model(self):
        """
        Load DQN model and training statistics from disk
        """
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.q_network.load_state_dict(model_data['q_network_state_dict'])
                self.target_network.load_state_dict(model_data['target_network_state_dict'])
                self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
                self.epsilon = model_data['epsilon']
                self.step_count = model_data['step_count']
                self.episode_rewards = model_data.get('episode_rewards', [])
                self.episode_lengths = model_data.get('episode_lengths', [])
                self.losses = model_data.get('losses', [])
                self.memory = model_data.get('memory', ReplayBuffer(self.memory_size))
                
                print(f"DQN model loaded from {self.model_path}")
                print(f"Previous training: {len(self.episode_rewards)} episodes")
                print(f"Current epsilon: {self.epsilon:.4f}")
                print(f"Step count: {self.step_count}")
                
            except Exception as e:
                print(f"Error loading DQN model: {e}")
                print("Starting with fresh DQN")
        else:
            print("No existing DQN model found. Starting fresh.")
    
    def get_policy(self, state):
        """
        Get the best action for a given state (no exploration)
        """
        state_tensor = self._state_to_tensor(state)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def get_q_values(self, state):
        """
        Get Q-values for all actions in a given state
        """
        state_tensor = self._state_to_tensor(state)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.cpu().numpy().flatten()
    
    def update_statistics(self, episode_reward, episode_length):
        """
        Update training statistics
        """
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
    
    def print_statistics(self):
        """
        Print current training statistics
        """
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-100:]
            recent_lengths = self.episode_lengths[-100:]
            recent_losses = self.losses[-100:] if self.losses else []
            
            print(f"\nDQN Training Statistics:")
            print(f"Total Episodes: {len(self.episode_rewards)}")
            print(f"Total Steps: {self.step_count}")
            print(f"Current Epsilon: {self.epsilon:.4f}")
            print(f"Recent Avg Reward: {np.mean(recent_rewards):.2f}")
            print(f"Recent Avg Length: {np.mean(recent_lengths):.2f}")
            if recent_losses:
                print(f"Recent Avg Loss: {np.mean(recent_losses):.4f}")
            print(f"Memory Size: {len(self.memory)}")
    
    def export_model(self, filename="dqn_model_export.pt"):
        """
        Export DQN model to PyTorch format
        """
        try:
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'model_architecture': {
                    'state_size': self.state_size,
                    'action_size': self.action_size
                },
                'timestamp': datetime.now().isoformat(),
                'epsilon': self.epsilon
            }, filename)
            print(f"DQN model exported to {filename}")
        except Exception as e:
            print(f"Error exporting DQN model: {e}") 