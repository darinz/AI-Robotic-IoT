#!/usr/bin/env python3
"""
Test script for trained RL policy
Demonstrates autonomous driving using a pre-trained model
"""

import argparse
import time
import signal
import sys
from rl_environment import CarEnvironment
from q_learning_agent import QLearningAgent
from dqn_agent import DQNAgent

class PolicyTester:
    """
    Test class for trained RL policies
    """
    
    def __init__(self, algorithm='q_learning', model_path=None, power=40):
        self.algorithm = algorithm
        self.power = power
        
        # Initialize environment
        print("Initializing car environment for testing...")
        self.env = CarEnvironment(
            power=power,
            safe_distance=40,
            danger_distance=20,
            max_steps=1000
        )
        
        # Get state and action space sizes
        state_space_size = self.env.get_state_space_size()
        action_space_size = self.env.get_action_space_size()
        
        # Initialize agent
        if algorithm == 'q_learning':
            self.agent = QLearningAgent(
                state_space_size=state_space_size,
                action_space_size=action_space_size,
                model_path=model_path or "q_learning_model.pkl"
            )
        elif algorithm == 'dqn':
            self.agent = DQNAgent(
                state_size=2,
                action_size=action_space_size,
                model_path=model_path or "dqn_model.pkl"
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Set up signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        self.running = True
    
    def signal_handler(self, signum, frame):
        """
        Handle Ctrl+C for graceful shutdown
        """
        print("\nReceived interrupt signal. Stopping test...")
        self.running = False
        self.env.close()
        sys.exit(0)
    
    def test_single_episode(self, max_steps=500):
        """
        Test policy for one episode
        """
        state = self.env.reset()
        total_reward = 0
        step_count = 0
        
        print(f"\nStarting test episode with {self.algorithm.upper()} policy...")
        print("Press Ctrl+C to stop")
        
        while self.running and step_count < max_steps:
            # Get action from trained policy (no exploration)
            action = self.agent.get_policy(state)
            
            # Get Q-values for debugging
            q_values = self.agent.get_q_values(state)
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # Print current state and action
            distance_state, cliff_state = state
            action_names = ['FORWARD', 'FORWARD_LEFT', 'FORWARD_RIGHT', 
                          'BACKWARD', 'BACKWARD_LEFT', 'BACKWARD_RIGHT', 'STOP']
            
            print(f"Step {step_count:3d} | "
                  f"State: ({distance_state}, {cliff_state}) | "
                  f"Action: {action_names[action]} | "
                  f"Reward: {reward:6.1f} | "
                  f"Q-values: {q_values}")
            
            total_reward += reward
            step_count += 1
            state = next_state
            
            if done:
                print(f"Episode finished after {step_count} steps")
                break
            
            # Small delay for readability
            time.sleep(0.2)
        
        print(f"Episode completed: Total reward = {total_reward:.1f}, Steps = {step_count}")
        return total_reward, step_count
    
    def test_multiple_episodes(self, num_episodes=5):
        """
        Test policy for multiple episodes
        """
        print(f"\nTesting {self.algorithm.upper()} policy for {num_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            reward, length = self.test_single_episode()
            episode_rewards.append(reward)
            episode_lengths.append(length)
            
            # Pause between episodes
            if episode < num_episodes - 1:
                print("Pausing 3 seconds before next episode...")
                time.sleep(3)
        
        # Print summary
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        avg_length = sum(episode_lengths) / len(episode_lengths)
        
        print(f"\n=== Test Summary ===")
        print(f"Algorithm: {self.algorithm.upper()}")
        print(f"Episodes: {num_episodes}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Length: {avg_length:.2f}")
        print(f"Best Episode: {max(episode_rewards):.1f}")
        print(f"Worst Episode: {min(episode_rewards):.1f}")
        
        return episode_rewards, episode_lengths
    
    def continuous_test(self):
        """
        Run continuous testing until interrupted
        """
        print(f"\nStarting continuous testing with {self.algorithm.upper()} policy...")
        print("Press Ctrl+C to stop")
        
        episode_count = 0
        
        while self.running:
            episode_count += 1
            print(f"\n--- Continuous Test Episode {episode_count} ---")
            
            reward, length = self.test_single_episode()
            
            print(f"Episode {episode_count} completed: Reward = {reward:.1f}, Length = {length}")
            
            # Ask user if they want to continue
            try:
                response = input("\nContinue testing? (y/n): ").lower().strip()
                if response != 'y':
                    break
            except KeyboardInterrupt:
                break
    
    def close(self):
        """
        Clean up resources
        """
        self.env.close()

def main():
    parser = argparse.ArgumentParser(description='Test trained RL policy')
    parser.add_argument('--algorithm', choices=['q_learning', 'dqn'], 
                       default='q_learning', help='RL algorithm to test')
    parser.add_argument('--model-path', type=str, 
                       help='Path to trained model file')
    parser.add_argument('--episodes', type=int, default=5, 
                       help='Number of test episodes')
    parser.add_argument('--power', type=int, default=40, 
                       help='Car power for testing')
    parser.add_argument('--continuous', action='store_true', 
                       help='Run continuous testing')
    
    args = parser.parse_args()
    
    # Create tester
    tester = PolicyTester(
        algorithm=args.algorithm,
        model_path=args.model_path,
        power=args.power
    )
    
    try:
        if args.continuous:
            tester.continuous_test()
        else:
            tester.test_multiple_episodes(args.episodes)
    finally:
        tester.close()

if __name__ == "__main__":
    main() 