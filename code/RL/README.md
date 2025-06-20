# Reinforcement Learning Implementation for Autonomous Car

This directory contains a complete reinforcement learning implementation for the KITT autonomous car, featuring both Q-Learning and Deep Q-Network (DQN) algorithms, plus a vision-enhanced environment for advanced autonomous driving.

## Overview

The implementation provides:
- **Q-Learning Agent**: Traditional tabular Q-learning for discrete state spaces
- **DQN Agent**: Deep Q-Network for more complex scenarios with neural network approximation
- **Vision-Enhanced Environment**: Computer vision integration for advanced perception
- **Car Environment**: Gym-like interface that wraps the PicarX hardware
- **Training Scripts**: Complete training loops with safety features
- **Testing Scripts**: Policy evaluation and demonstration tools
- **Persistent Storage**: Model checkpointing and resume capabilities

## File Structure

```
code/RL/
├── rl_environment.py           # Basic car environment wrapper
├── vision_rl_environment.py    # Vision-enhanced RL environment
├── q_learning_agent.py         # Q-Learning implementation
├── dqn_agent.py                # DQN implementation
├── train_rl_agent.py           # Main training script
├── test_trained_policy.py      # Policy testing script
├── requirements_rl.txt         # Dependencies
└── README.md                   # This file
```

## Key Features

### Safety Features
- **Reduced Power**: Lower motor power during early learning phases
- **Episode Limits**: Maximum step limits to prevent infinite loops
- **Graceful Shutdown**: Ctrl+C handling for safe interruption
- **Collision Detection**: Automatic episode termination on dangerous states

### Vision Integration
- **Computer Vision**: Real-time image processing and object detection
- **Multi-modal State**: Combines sensor data with vision features
- **Advanced Actions**: Vision-based actions like line following and obstacle avoidance
- **Enhanced Perception**: Object tracking, lane detection, and color recognition

### Persistent Learning
- **Model Checkpointing**: Automatic saving every N episodes
- **Resume Training**: Load previous models and continue learning
- **Export Formats**: Multiple export formats (pickle, JSON, PyTorch)
- **Training Statistics**: Comprehensive logging and analysis

### State Space
- **Distance States**: Discretized ultrasonic sensor readings (0-100+ cm)
- **Cliff States**: Binary cliff detection (safe/danger)
- **Vision Features**: 64-dimensional vision feature vector
- **Object Detections**: Real-time object tracking and classification
- **Lane Information**: Lane angle detection for autonomous driving
- **Action Space**: 11 discrete actions including vision-based actions

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_rl.txt
```

### 2. Train Q-Learning Agent (Basic)

```bash
# Basic training with sensor-only environment
python train_rl_agent.py --algorithm q_learning --episodes 500

# With custom parameters
python train_rl_agent.py \
    --algorithm q_learning \
    --episodes 1000 \
    --power 25 \
    --save-interval 25 \
    --print-interval 5
```

### 3. Train DQN Agent (Basic)

```bash
# DQN training (requires more episodes)
python train_rl_agent.py \
    --algorithm dqn \
    --episodes 2000 \
    --power 30 \
    --save-interval 100
```

### 4. Vision-Enhanced Training

```bash
# Train with vision-enhanced environment
python train_rl_agent.py \
    --algorithm dqn \
    --episodes 3000 \
    --vision-enabled \
    --power 25 \
    --save-interval 100
```

### 5. Test Trained Policy

```bash
# Test Q-Learning policy
python test_trained_policy.py --algorithm q_learning --episodes 3

# Test DQN policy
python test_trained_policy.py --algorithm dqn --episodes 3

# Test vision-enhanced policy
python test_trained_policy.py --algorithm dqn --vision-enabled --episodes 3

# Continuous testing
python test_trained_policy.py --algorithm q_learning --continuous
```

## Action Space

The car can perform 11 discrete actions:

| Action ID | Name | Description |
|-----------|------|-------------|
| 0 | FORWARD | Move straight forward |
| 1 | FORWARD_LEFT | Turn left while moving forward |
| 2 | FORWARD_RIGHT | Turn right while moving forward |
| 3 | BACKWARD | Move straight backward |
| 4 | BACKWARD_LEFT | Turn left while moving backward |
| 5 | BACKWARD_RIGHT | Turn right while moving backward |
| 6 | STOP | Stop all movement |
| 7 | TURN_LEFT | Turn left in place |
| 8 | TURN_RIGHT | Turn right in place |
| 9 | FOLLOW_LINE | Vision-based line following |
| 10 | AVOID_OBSTACLE | Vision-based obstacle avoidance |

## State Space

### Basic Environment (2-dimensional)
1. **Distance State** (0-6): Discretized ultrasonic sensor reading
2. **Cliff State** (0-1): Binary cliff detection

### Vision-Enhanced Environment (Multi-dimensional)
1. **Sensor State** (2): Distance and cliff states
2. **Vision Features** (64): Extracted from camera feed
3. **Object Detections**: List of detected objects with properties
4. **Lane Information**: Left and right lane angles
5. **Temporal Features**: Object movement tracking

## Reward Function

### Basic Environment
- **+1**: Safe state (no cliff detected)
- **+2**: Safe distance to obstacles (>40cm)
- **-1**: Stopping (encourages movement)
- **-2**: Backward movement (encourages forward progress)
- **-30**: Close to obstacles (<20cm)
- **-50**: Cliff detected (episode termination)

### Vision-Enhanced Environment
- **+0.5**: Successful object detection
- **+3**: Proper lane detection (angles < 45°)
- **+2**: Vision-based actions (line following, obstacle avoidance)
- **+1**: Safe state (no cliff detected)
- **+2**: Safe distance to obstacles (>40cm)
- **-1**: Stopping (encourages movement)
- **-2**: Backward movement (encourages forward progress)
- **-30**: Close to obstacles (<20cm)
- **-50**: Cliff detected (episode termination)
- **-50**: Too many obstacles detected (>5)
- **-10**: Camera failure (no vision features)

## Training Parameters

### Q-Learning Parameters
- `learning_rate`: 0.1 (how much to update Q-values)
- `discount_factor`: 0.95 (future reward importance)
- `epsilon`: 0.1 (exploration rate)
- `epsilon_decay`: 0.995 (exploration decay)
- `epsilon_min`: 0.01 (minimum exploration)

### DQN Parameters
- `learning_rate`: 0.001 (neural network learning rate)
- `discount_factor`: 0.95 (future reward importance)
- `epsilon`: 1.0 (initial exploration rate)
- `epsilon_decay`: 0.995 (exploration decay)
- `batch_size`: 32 (experience replay batch size)
- `memory_size`: 10000 (experience replay buffer size)

### Environment Parameters
- `power`: 30 (motor power, reduced for safety)
- `safe_distance`: 40 (cm, safe zone)
- `danger_distance`: 20 (cm, danger zone)
- `max_steps`: 500 (episode length limit)
- `vision_enabled`: true (enable computer vision)
- `camera_index`: 0 (camera device index)

## Advanced Usage

### Vision-Enhanced Environment

```python
from vision_rl_environment import VisionRLEnvironment

# Create vision-enhanced environment
env = VisionRLEnvironment(
    power=25,
    vision_enabled=True,
    camera_index=0
)

# Get state with vision features
state = env.get_state()
print(f"Sensor state: {state.sensor_state}")
print(f"Vision features shape: {state.vision_features.shape}")
print(f"Object detections: {len(state.object_detections)}")
print(f"Lane detection: {state.lane_detection}")
```

### Custom Training Configuration

```python
from train_rl_agent import RLTrainer

# Create custom trainer with vision
trainer = RLTrainer(
    algorithm='dqn',
    max_episodes=3000,
    power=25,
    vision_enabled=True,
    learning_rate=0.001,
    discount_factor=0.95,
    epsilon=0.2
)

# Train with custom parameters
trainer.train()
```

### Model Analysis

```python
from q_learning_agent import QLearningAgent

# Load trained model
agent = QLearningAgent(state_space_size=(7, 2), action_space_size=7)
agent.load_model()

# Analyze Q-values for specific state
state = (3, 0)  # Safe distance, no cliff
q_values = agent.get_q_values(state)
print(f"Q-values for state {state}: {q_values}")
```

### Vision Statistics

```python
from vision_rl_environment import VisionRLEnvironment

env = VisionRLEnvironment(vision_enabled=True)
stats = env.get_vision_statistics()
print(f"Vision mode: {stats['mode']}")
print(f"Detection count: {stats['detection_count']}")
print(f"Vision success count: {stats['vision_success_count']}")
```

## Safety Guidelines

1. **Start with Low Power**: Begin training with power=20-30
2. **Use Controlled Environment**: Train in open, obstacle-free areas
3. **Monitor Closely**: Watch the car during early training phases
4. **Emergency Stop**: Keep Ctrl+C ready for immediate shutdown
5. **Gradual Progression**: Increase power only after successful training
6. **Vision Safety**: Ensure camera is properly mounted and calibrated
7. **Lighting Conditions**: Train in well-lit environments for best vision performance

## Troubleshooting

### Common Issues

1. **Car not moving**: Check power settings and motor connections
2. **Sensors not reading**: Verify ultrasonic and grayscale sensor connections
3. **Camera not working**: Check camera connections and permissions
4. **Vision features empty**: Ensure proper lighting and camera calibration
5. **Model not loading**: Ensure model file exists and is not corrupted
6. **Training not improving**: Adjust learning rate or reward function

### Debug Mode

Enable verbose output for debugging:

```bash
python train_rl_agent.py --algorithm q_learning --print-interval 1
```

### Vision Debugging

```python
# Test vision system separately
from computer_vision import VisionSystem
from picarx import Picarx

car = Picarx()
vision = VisionSystem(car)
vision.start_vision()

# Check vision statistics
stats = vision.get_vision_statistics()
print(stats)
```

## Algorithm Details

### Q-Learning
- **Type**: Tabular, model-free
- **Best for**: Small state spaces, quick learning
- **Memory**: O(states × actions)
- **Convergence**: Guaranteed under certain conditions

### DQN
- **Type**: Neural network approximation
- **Best for**: Large state spaces, complex environments
- **Memory**: Experience replay buffer
- **Features**: Target network, experience replay

### Vision-Enhanced RL
- **Type**: Multi-modal reinforcement learning
- **Best for**: Complex environments requiring visual perception
- **Features**: Object detection, lane following, obstacle avoidance
- **Integration**: Seamless combination of sensor and vision data

## Future Enhancements

- [ ] PPO (Proximal Policy Optimization) implementation
- [ ] Advanced vision models (YOLO, SSD integration)
- [ ] Multi-agent training
- [ ] Transfer learning capabilities
- [ ] Real-time visualization
- [ ] Advanced reward shaping
- [ ] Semantic segmentation for better environment understanding
- [ ] Attention mechanisms for vision processing
- [ ] Multi-camera support
- [ ] SLAM integration for mapping

## License

This implementation follows the same license as the main project.

---

**Note**: This RL implementation is designed for educational and research purposes. Always ensure safe operation when training autonomous vehicles. The vision-enhanced environment requires proper camera setup and lighting conditions for optimal performance. 