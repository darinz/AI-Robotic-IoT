# AI-Robotic-IoT: Comprehensive AI-Powered Robotics Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-OpenCV-orange.svg)
![NLP](https://img.shields.io/badge/NLP-Whisper%20%7C%20TTS-purple.svg)
![Hardware](https://img.shields.io/badge/Hardware-Picar--X-red.svg)
![IoT](https://img.shields.io/badge/IoT-MQTT%20%7C%20Smart%20Home-cyan.svg)
![RL](https://img.shields.io/badge/Reinforcement%20Learning-PyTorch-yellow.svg)

**A comprehensive platform combining IoT, Natural Language Processing, Computer Vision, and Reinforcement Learning for intelligent robotic systems**

[Overview](#overview) • [Components](#components) • [Quick Start](#quick-start) • [Architecture](#architecture) • [Installation](#installation) • [Usage](#usage) • [Contributing](#contributing)

</div>


## Overview

AI-Robotic-IoT is a comprehensive platform that demonstrates the integration of multiple AI technologies with robotics and IoT systems. The project consists of three main components, each showcasing different aspects of modern AI and robotics:

1. **IoT Smart Home System** - Clap detection and smart home automation
2. **NLP-CV Robotic Car** - AI-powered conversational robot with computer vision
3. **RL Autonomous Navigation** - Reinforcement learning for autonomous driving

## Components

### NLP-CV: AI-Powered Robotic Car

**Location**: `NLP-CV/`

An intelligent robotic car that combines Natural Language Processing, Computer Vision, and Voice Commands for interactive AI experiences.

**Key Features**:
- **AI Conversations**: Real-time dialogue using OpenAI's GPT-4o
- **Voice Commands**: Advanced voice command system with natural language processing
- **Computer Vision**: Real-time image analysis, object detection, and visual understanding
- **Speech Interaction**: Speech-to-text and text-to-speech with Whisper API
- **Environment Variables**: Secure configuration management
- **Multiple Vision Modes**: Obstacle detection, line following, object tracking, lane detection, color detection

**Technologies**: OpenAI GPT-4o, Whisper API, OpenCV, SpeechRecognition, Picar-X

### RL: Reinforcement Learning for Autonomous Navigation

**Location**: `RL/`

A complete reinforcement learning implementation for autonomous car navigation, featuring both traditional Q-Learning and modern Deep Q-Network (DQN) algorithms.

**Key Features**:
- **Q-Learning Agent**: Traditional tabular Q-learning for discrete state spaces
- **DQN Agent**: Deep Q-Network for complex scenarios with neural network approximation
- **Vision-Enhanced Environment**: Computer vision integration for advanced perception
- **Safety Features**: Reduced power, episode limits, collision detection
- **Persistent Learning**: Model checkpointing and resume capabilities
- **Multi-modal State**: Combines sensor data with vision features

**Technologies**: PyTorch, NumPy, OpenCV, Picar-X, Gym-like interface

### IoT: Smart Home Clap Detection System

**Location**: `IoT/`

A sophisticated IoT project that combines clap detection with smart home automation, featuring both a Raspberry Pi backend and a Godot frontend application.

**Key Features**:
- **Dual Control Methods**: Both clap detection and mobile app control
- **Smart Light Integration**: Controls smart light devices via MQTT
- **Real-time Status**: Live updates of device states
- **Robust Communication**: Reliable MQTT protocol implementation
- **Mobile App**: Godot-based mobile application for remote control

**Technologies**: MQTT, Raspberry Pi GPIO, Godot Engine, Python

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/darinz/AI-Robotic-IoT.git
cd AI-Robotic-IoT
```

### 2. Choose Your Component

Each component can be used independently:

#### For AI Robotic Car (NLP-CV)
```bash
cd code/NLP-CV
python setup_env.py  # Configure environment
python gpt_car.py    # Run the main application
```

#### For Reinforcement Learning (RL)
```bash
cd code/RL
pip install -r requirements_rl.txt
python train_rl_agent.py --algorithm q_learning --episodes 500
```

#### For IoT Smart Home (IoT)
```bash
cd code/IoT
chmod +x install.sh
./install.sh
python3 clap_detector.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI-Robotic-IoT Platform                      │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   IoT System    │   NLP-CV Car    │      RL Environment         │
│                 │                 │                             │
│ ┌─────────────┐ │ ┌─────────────┐ │ ┌─────────────────────────┐ │
│ │ Clap Sensor │ │ │ Voice Input │ │ │ Sensor State            │ │
│ │ Raspberry Pi│ │ │ Camera      │ │ │ Vision Features         │ │
│ │ MQTT Broker │ │ │ OpenAI API  │ │ │ Q-Learning/DQN          │ │
│ │ Godot App   │ │ │ Picar-X     │ │ │ Training Loop           │ │
│ └─────────────┘ │ └─────────────┘ │ └─────────────────────────┘ │
│                 │                 │                             │
│ Smart Home      │ AI Conversations│ Autonomous Navigation      │
│ Automation      │ Computer Vision │ Reinforcement Learning     │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## Installation

### Prerequisites

- **Python 3.8+**
- **Raspberry Pi** (for IoT and some robotics components)
- **Picar-X Robotic Car** (for NLP-CV and RL components)
- **OpenAI API Key** (for NLP-CV component)
- **MQTT Broker** (for IoT component)
- **Camera and Microphone** (for computer vision and voice features)

### System Requirements

#### For NLP-CV Component
- **Hardware**: Picar-X robotic car with Robot HAT
- **OS**: Raspberry Pi OS or compatible Linux distribution
- **Camera**: USB camera or Pi Camera
- **Microphone**: USB microphone or built-in microphone
- **Internet**: Required for OpenAI API calls

#### For RL Component
- **Hardware**: Picar-X robotic car with Robot HAT
- **OS**: Raspberry Pi OS or compatible Linux distribution
- **Camera**: USB camera or Pi Camera (for vision-enhanced RL)
- **Storage**: Sufficient space for model checkpoints

#### For IoT Component
- **Hardware**: Raspberry Pi with sound sensor
- **OS**: Raspberry Pi OS
- **Network**: MQTT broker access
- **Smart Lights**: MQTT-compatible smart light devices

### Installation Steps

#### 1. Base System Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install common dependencies
sudo apt install python3-pip python3-venv git
```

#### 2. Component-Specific Installation

**NLP-CV Component**:
```bash
cd code/NLP-CV
pip install -r requirements.txt
python setup_env.py
```

**RL Component**:
```bash
cd code/RL
pip install -r requirements_rl.txt
```

**IoT Component**:
```bash
cd code/IoT
chmod +x install.sh
./install.sh
```

## Usage

### NLP-CV: AI Robotic Car

#### Basic Usage
```bash
cd code/NLP-CV

# Voice interaction mode (default)
sudo python3 gpt_car.py

# Keyboard interaction mode
sudo python3 gpt_car.py --keyboard

# Voice commands demo
python3 voice_commander.py

# Computer vision demo
python3 computer_vision.py
```

#### Configuration
```bash
# Set up environment variables
export OPENAI_API_KEY="your-api-key"
export OPENAI_ASSISTANT_ID="your-assistant-id"

# Or use .env file
cp env.template .env
# Edit .env with your keys
```

#### Voice Commands
- **Movement**: "forward", "backward", "left", "right", "stop"
- **Speed Control**: "speed up", "slow down"
- **System**: "status", "help"
- **Vision**: "look around", "follow line"

### RL: Reinforcement Learning

#### Training
```bash
cd code/RL

# Train Q-Learning agent
python train_rl_agent.py --algorithm q_learning --episodes 500

# Train DQN agent
python train_rl_agent.py --algorithm dqn --episodes 2000

# Train with vision enhancement
python train_rl_agent.py --algorithm dqn --vision-enabled --episodes 3000
```

#### Testing
```bash
# Test trained policy
python test_trained_policy.py --algorithm q_learning --episodes 3

# Continuous testing
python test_trained_policy.py --algorithm dqn --continuous
```

#### Available Actions
- **Basic**: forward, forward_left, forward_right, backward, stop
- **Advanced**: turn_left, turn_right, follow_line, avoid_obstacle

### IoT: Smart Home System

#### Raspberry Pi Setup
```bash
cd code/IoT

# Run clap detection system
python3 clap_detector.py
```

#### Mobile App
1. Open `iot-godot-app/` in Godot Engine
2. Configure MQTT settings in `mqtt.gd`
3. Export to Android APK
4. Install on mobile device

#### Control Methods
- **Single Clap**: Cycle through light states
- **Double Clap**: Master toggle (all lights on/off)
- **Mobile App**: Manual control via touch interface

## Configuration

### Environment Variables

The platform supports comprehensive configuration via environment variables:

```bash
# OpenAI Configuration (NLP-CV)
OPENAI_API_KEY=your-api-key
OPENAI_ASSISTANT_ID=your-assistant-id

# Speech and Audio (NLP-CV)
SPEECH_LANGUAGE=en-US
AUDIO_VOLUME_DB=3
TTS_VOICE=echo

# Vision Configuration (NLP-CV, RL)
CAMERA_INDEX=0
VISION_ENABLED=true

# Voice Command Configuration (NLP-CV)
VOICE_ENABLED=true
VOICE_CONFIDENCE_THRESHOLD=0.7

# RL Configuration
RL_POWER=30
RL_SAFE_DISTANCE=40
RL_DANGER_DISTANCE=20

# System Configuration
DEBUG_MODE=false
LOG_LEVEL=INFO
```

### Configuration Files

- **NLP-CV**: `env.template`, `config.py`, `keys_template.py`
- **RL**: Training parameters in `train_rl_agent.py`
- **IoT**: MQTT settings in `clap_detector.py` and `mqtt.gd`

## Development

### Project Structure

```
AI-Robotic-IoT/code/
├── README.md                      # This file
├── setup_env.py                   # Environment setup script
├── env.template                   # Environment variables template
├── ENVIRONMENT_UPDATE_SUMMARY.md  # Configuration update summary
│
├── NLP-CV/                 # AI Robotic Car Component
│   ├── gpt_car.py          # Main application
│   ├── config.py           # Configuration management
│   ├── voice_commander.py  # Voice command system
│   ├── computer_vision.py  # Computer vision system
│   ├── openai_helper.py    # OpenAI API integration
│   ├── preset_actions.py   # Predefined actions
│   ├── utils.py            # Utility functions
│   ├── keys_template.py    # Configuration template
│   ├── requirements.txt    # Dependencies
│   └── README.md           # Component documentation
│
├── RL/                            # Reinforcement Learning Component
│   ├── rl_environment.py          # Basic car environment
│   ├── vision_rl_environment.py   # Vision-enhanced environment
│   ├── q_learning_agent.py        # Q-Learning implementation
│   ├── dqn_agent.py               # DQN implementation
│   ├── train_rl_agent.py          # Training script
│   ├── test_trained_policy.py     # Testing script
│   ├── requirements_rl.txt        # Dependencies
│   └── README.md                  # Component documentation
│
└── IoT/                    # IoT Smart Home Component
    ├── clap_detector.py    # Raspberry Pi clap detection
    ├── config.py           # IoT configuration
    ├── install.sh          # Installation script
    ├── requirements.txt    # Python dependencies
    ├── README.md           # Component documentation
    └── iot-godot-app/      # Godot mobile application
        ├── project.godot   # Godot project configuration
        ├── main.gd         # Main application controller
        ├── mqtt.gd         # MQTT protocol helper
        ├── main.tscn       # Main scene file
        └── assets/         # Application assets
```

### Adding New Features

#### For NLP-CV Component
1. **New Voice Commands**: Add to `voice_commander.py`
2. **New Vision Modes**: Add to `computer_vision.py`
3. **New Actions**: Add to `preset_actions.py`

#### For RL Component
1. **New Algorithms**: Create new agent class
2. **New Environments**: Extend environment classes
3. **New Features**: Add to training scripts

#### For IoT Component
1. **New Sensors**: Extend sensor classes
2. **New Devices**: Add MQTT device handlers
3. **New UI**: Modify Godot application

## Troubleshooting

### Common Issues

#### Configuration Issues
```bash
# Run setup script for guidance
python setup_env.py

# Check configuration
python code/NLP-CV/config.py
```

#### Hardware Issues
- **Picar-X**: Check motor and sensor connections
- **Camera**: Verify camera permissions and connections
- **Microphone**: Test with `python3 -c "import speech_recognition as sr; print('OK')"`

#### Network Issues
- **MQTT**: Check broker connectivity and credentials
- **OpenAI API**: Verify API key and internet connection

### Debug Mode

Enable debug mode for detailed logging:

```bash
export DEBUG_MODE=true
# Run your component
```

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Add tests** if applicable
5. **Submit a pull request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include error handling for robustness
- Test on actual hardware when possible
- Update documentation for new features

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **OpenAI** for GPT-4o and Whisper APIs
- **SunFounder** for Picar-X hardware platform
- **OpenCV** community for computer vision tools
- **PyTorch** team for deep learning framework
- **Godot Engine** for mobile app development
- **MQTT** community for IoT communication protocols

## Support

- **Issues**: [GitHub Issues](https://github.com/darinz/AI-Robotic-IoT/issues)
- **Discussions**: [GitHub Discussions](https://github.com/darinz/AI-Robotic-IoT/discussions)
- **Documentation**: See individual component README files

---

<div align="center">

**Made for the Robotics, AI, and IoT community**

[Star this repo](https://github.com/darinz/AI-Robotic-IoT) • [Report Issues](https://github.com/darinz/AI-Robotic-IoT/issues) • [View Documentation](../docs)

</div> 