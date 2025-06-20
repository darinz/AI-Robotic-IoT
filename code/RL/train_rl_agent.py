#!/usr/bin/env python3
"""
Reinforcement Learning Training Script for Autonomous Car
Implements both Q-Learning and DQN with safety features and persistent storage
"""

import argparse
import time
import signal
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from rl_environment import CarEnvironment
from q_learning_agent import QLearningAgent
from dqn_agent import DQNAgent

class RLTrainer:
    """
    Main training class for reinforcement learning
    """
    
    def __init__(self, algorithm='q_learning', **kwargs):
        self.algorithm = algorithm
        self.training = True
        self.save_interval = kwargs.get('save_interval', 50)
        self.print_interval = kwargs.get('print_interval', 10)
        self.max_episodes = kwargs.get('max_episodes', 1000)
        
        # Initialize environment
        print("Initializing car environment...")
        self.env = CarEnvironment(
            power=kwargs.get('power', 30),  # Reduced power for safety
            safe_distance=kwargs.get('safe_distance', 40),
            danger_distance=kwargs.get('danger_distance', 20),
            max_steps=kwargs.get('max_steps', 500)
        )
        
        # Get state and action space sizes
        state_space_size = self.env.get_state_space_size()
        action_space_size = self.env.get_action_space_size()
        
        print(f"State space: {state_space_size}")
        print(f"Action space: {action_space_size}")
        
        # Initialize agent
        if algorithm == 'q_learning':
            self.agent = QLearningAgent(
                state_space_size=state_space_size,
                action_space_size=action_space_size,
                learning_rate=kwargs.get('learning_rate', 0.1),
                discount_factor=kwargs.get('discount_factor', 0.95),
                epsilon=kwargs.get('epsilon', 0.1),
                epsilon_decay=kwargs.get('epsilon_decay', 0.995),
                epsilon_min=kwargs.get('epsilon_min', 0.01)
            )
        elif algorithm == 'dqn':
            self.agent = DQNAgent(
                state_size=2,  # Flattened state size
                action_size=action_space_size,
                learning_rate=kwargs.get('learning_rate', 0.001),
                discount_factor=kwargs.get('discount_factor', 0.95),
                epsilon=kwargs.get('epsilon', 1.0),
                epsilon_decay=kwargs.get('epsilon_decay', 0.995),
                epsilon_min=kwargs.get('epsilon_min', 0.01),
                batch_size=kwargs.get('batch_size', 32),
                memory_size=kwargs.get('memory_size', 10000)
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Training statistics
        self.episode_count = 0
        self.total_steps = 0
        self.start_time = time.time()
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """
        Handle Ctrl+C for graceful shutdown
        """
        print("\nReceived interrupt signal. Saving model and shutting down...")
        self.training = False
        self.save_model()
        self.env.close()
        sys.exit(0)
    
    def train_episode(self):
        """
        Train for one episode
        """
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        collision = False
        success = False
        
        while True:
            # Choose action
            action = self.agent.choose_action(state)
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # Learn from experience
            self.agent.learn(state, action, reward, next_state, done)
            
            # Update episode statistics
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            # Check for collision or success
            if done:
                if reward < -50:
                    collision = True
                elif reward > 50:
                    success = True
                break
            
            # Safety check - stop if episode is too long
            if episode_length > 1000:
                print("Episode too long, stopping for safety")
                break
        
        # Update agent statistics
        if hasattr(self.agent, 'update_statistics'):
            self.agent.update_statistics(episode_reward, episode_length, collision, success)
        else:
            self.agent.update_statistics(episode_reward, episode_length)
        
        # Update epsilon
        self.agent.update_epsilon()
        
        return episode_reward, episode_length, collision, success
    
    def train(self):
        """
        Main training loop
        """
        print(f"\nStarting {self.algorithm.upper()} training...")
        print("Press Ctrl+C to stop training and save model")
        
        try:
            while self.training and self.episode_count < self.max_episodes:
                self.episode_count += 1
                
                # Train one episode
                episode_reward, episode_length, collision, success = self.train_episode()
                
                # Print progress
                if self.episode_count % self.print_interval == 0:
                    elapsed_time = time.time() - self.start_time
                    print(f"Episode {self.episode_count:4d} | "
                          f"Reward: {episode_reward:6.1f} | "
                          f"Length: {episode_length:3d} | "
                          f"Collision: {collision} | "
                          f"Success: {success} | "
                          f"Time: {elapsed_time:.1f}s")
                    
                    # Print agent statistics
                    self.agent.print_statistics()
                
                # Save model periodically
                if self.episode_count % self.save_interval == 0:
                    self.save_model()
                
                # Safety pause between episodes
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            print("\nTraining completed. Saving final model...")
            self.save_model()
            self.env.close()
    
    def save_model(self):
        """
        Save the trained model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save agent model
        self.agent.save_model()
        
        # Export additional formats
        if self.algorithm == 'q_learning':
            self.agent.export_q_table(f"q_table_export_{timestamp}.json")
        elif self.algorithm == 'dqn':
            self.agent.export_model(f"dqn_model_export_{timestamp}.pt")
        
        print(f"Model saved at {timestamp}")
    
    def test_policy(self, num_episodes=5):
        """
        Test the trained policy without exploration
        """
        print(f"\nTesting trained policy for {num_episodes} episodes...")
        
        test_rewards = []
        test_lengths = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Use policy without exploration
                action = self.agent.get_policy(state)
                
                # Execute action
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
                
                # Safety timeout
                if episode_length > 500:
                    break
            
            test_rewards.append(episode_reward)
            test_lengths.append(episode_length)
            
            print(f"Test Episode {episode + 1}: Reward = {episode_reward:.1f}, Length = {episode_length}")
        
        avg_reward = np.mean(test_rewards)
        avg_length = np.mean(test_lengths)
        
        print(f"\nTest Results:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Length: {avg_length:.2f}")
        
        return test_rewards, test_lengths

def plot_training_results(agent, algorithm):
    """
    Plot training results
    """
    if not agent.episode_rewards:
        print("No training data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{algorithm.upper()} Training Results')
    
    # Episode rewards
    axes[0, 0].plot(agent.episode_rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    
    # Episode lengths
    axes[0, 1].plot(agent.episode_lengths)
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Length')
    
    # Moving average rewards
    window = min(50, len(agent.episode_rewards))
    if window > 0:
        moving_avg = np.convolve(agent.episode_rewards, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(moving_avg)
        axes[1, 0].set_title(f'Moving Average Reward (window={window})')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Average Reward')
    
    # Collision/Success rates (if available)
    if hasattr(agent, 'collision_rates') and agent.collision_rates:
        axes[1, 1].plot(agent.collision_rates, label='Collision Rate')
        axes[1, 1].plot(agent.success_rates, label='Success Rate')
        axes[1, 1].set_title('Collision and Success Rates')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Rate')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{algorithm}_training_results.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train RL agent for autonomous car')
    parser.add_argument('--algorithm', choices=['q_learning', 'dqn'], 
                       default='q_learning', help='RL algorithm to use')
    parser.add_argument('--episodes', type=int, default=1000, 
                       help='Maximum number of training episodes')
    parser.add_argument('--power', type=int, default=30, 
                       help='Car power (reduced for safety)')
    parser.add_argument('--save-interval', type=int, default=50, 
                       help='Save model every N episodes')
    parser.add_argument('--print-interval', type=int, default=10, 
                       help='Print progress every N episodes')
    parser.add_argument('--test', action='store_true', 
                       help='Test trained policy after training')
    parser.add_argument('--plot', action='store_true', 
                       help='Plot training results')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = RLTrainer(
        algorithm=args.algorithm,
        max_episodes=args.episodes,
        power=args.power,
        save_interval=args.save_interval,
        print_interval=args.print_interval
    )
    
    # Train
    trainer.train()
    
    # Test if requested
    if args.test:
        trainer.test_policy()
    
    # Plot if requested
    if args.plot:
        plot_training_results(trainer.agent, args.algorithm)

if __name__ == "__main__":
    main() 