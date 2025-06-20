# AI-Powered Robotic Car with NLP and Computer Vision

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-OpenCV-orange.svg)
![NLP](https://img.shields.io/badge/NLP-Whisper%20%7C%20TTS-purple.svg)
![Hardware](https://img.shields.io/badge/Hardware-Picar--X-red.svg)
![Voice Commands](https://img.shields.io/badge/Voice%20Commands-SpeechRecognition-yellow.svg)

**An intelligent robotic car that combines Natural Language Processing, Computer Vision, and Voice Commands for interactive AI experiences**

[Features](#features) • [Installation](#installation) • [Configuration](#configuration) • [Usage](#usage) • [Architecture](#architecture) • [Contributing](#contributing)

</div>

---

## Features

- **AI-Powered Conversations**: Real-time dialogue using OpenAI's GPT-4o with multimodal capabilities
- **Voice Commands**: Advanced voice command system with natural language processing
- **Computer Vision**: Real-time image analysis, object detection, and visual understanding
- **Speech Interaction**: Speech-to-text and text-to-speech with Whisper API
- **Expressive Actions**: 10+ predefined robotic movements and gestures
- **Audio Feedback**: Sound effects and voice synthesis
- **Real-time Processing**: Low-latency audio and visual processing
- **Multi-language Support**: Configurable language detection and processing
- **Dual Input Modes**: Voice and keyboard interaction options
- **Environment Variables**: Secure configuration management
- **Vision Modes**: Multiple computer vision modes (obstacle detection, line following, object tracking, lane detection, color detection)

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Voice Input   │    │  Camera Input   │    │ Keyboard Input  │
│   (Whisper)     │    │   (OpenCV)      │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    OpenAI GPT-4o API      │
                    │  (Multimodal Processing)  │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   Response Processing     │
                    │  (Actions + Speech)       │
                    └─────────────┬─────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
┌─────────▼────────┐    ┌─────────▼────────┐    ┌─────────▼────────┐
│  Robotic Actions │    │  Text-to-Speech  │    │  Sound Effects   │
│   (Picar-X)      │    │   (TTS API)      │    │   (Audio)        │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

## Prerequisites

- **Hardware**: Picar-X robotic car with Robot HAT
- **OS**: Raspberry Pi OS or compatible Linux distribution
- **Python**: 3.8 or higher
- **OpenAI API Key**: Valid API key with GPT-4o access
- **Internet Connection**: Required for API calls
- **Camera**: USB camera or Pi Camera for computer vision features
- **Microphone**: USB microphone or built-in microphone for voice commands

## Installation

### 1. Install Picar-X Dependencies

First, install the Picar-X and Robot HAT dependencies:

```bash
# Follow the official Picar-X installation guide
# https://docs.sunfounder.com/projects/picar-x-v20/en/latest/python/python_start/install_all_modules.html
```

### 2. Install Python Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install system dependencies (if needed)
sudo apt install python3-pyaudio
sudo apt install sox
```

### 3. Clone the Repository

```bash
git clone https://github.com/darinz/AI-Robotic-IoT.git
cd AI-Robotic-IoT
```

## Configuration

### 1. Environment Setup

The application now supports multiple ways to configure your API keys:

#### Option A: Environment Variables (Recommended)
```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_ASSISTANT_ID="your-assistant-id-here"
```

#### Option B: .env File (Recommended for Development)
```bash
# Copy the template and edit with your keys
cp env.template .env
# Edit .env file with your actual API keys
```

#### Option C: keys.py File (Legacy)
```python
# Create keys.py from keys_template.py and add your credentials
OPENAI_API_KEY = "your-api-key-here"
OPENAI_ASSISTANT_ID = "your-assistant-id-here"
```

### 2. Quick Setup

Use the setup script to configure your environment:

```bash
# Run the setup script
python setup_env.py

# Test your configuration
python setup_env.py --test
```

### 3. OpenAI API Setup

1. **Get API Key**: Visit [OpenAI Platform](https://platform.openai.com/api-keys) to obtain your API key
2. **Create Assistant**: Go to [OpenAI Assistants](https://platform.openai.com/assistants) to create a custom assistant
3. **Configure Keys**: Use one of the methods above to set your credentials

### 4. Assistant Configuration

Configure your OpenAI Assistant with the following settings:

**Name**: `KITT`

**Description**:
```markdown
You are a small car with AI capabilities named KITT. You can engage in conversations with people and react accordingly to different situations with actions or sounds. You are driven by two rear wheels, with two front wheels that can turn left and right, and equipped with a camera mounted on a 2-axis gimbal.

## Response Format (JSON):
{"actions": ["start engine", "honking", "wave hands"], "answer": "Hello, I am PaiCar-X, your good friend."}

## Response Style:
- Tone: Cheerful, optimistic, humorous, childlike
- Style: Enjoys jokes, metaphors, and playful banter from a robotic perspective
- Detail Level: Moderately detailed

## Available Actions:
["shake head", "nod", "wave hands", "resist", "act cute", "rub hands", "think", "twist body", "celebrate", "depressed"]

## Sound Effects:
["honking", "start engine"]
```

**Model**: Select `gpt-4o` or `gpt-4o-mini` for image analysis capabilities

### 5. Optional Configuration

The system supports many configurable parameters via environment variables:

```bash
# Speech and Audio
SPEECH_LANGUAGE=en-US
AUDIO_VOLUME_DB=3
TTS_VOICE=echo

# Vision Configuration
CAMERA_INDEX=0
VISION_ENABLED=true

# Voice Command Configuration
VOICE_ENABLED=true
VOICE_CONFIDENCE_THRESHOLD=0.7

# System Configuration
DEBUG_MODE=false
LOG_LEVEL=INFO
```

## Usage

### Basic Usage

```bash
# Voice interaction mode (default)
sudo python3 gpt_car.py

# Keyboard interaction mode
sudo python3 gpt_car.py --keyboard

# Keyboard mode without image analysis
sudo python3 gpt_car.py --keyboard --no-img
```

### Voice Commands

The system includes an advanced voice command system:

```bash
# Run voice commander demo
python3 voice_commander.py
```

**Available Voice Commands**:
- **Movement**: "forward", "backward", "left", "right", "stop"
- **Speed Control**: "speed up", "slow down"
- **System**: "status", "help"
- **Vision**: "look around", "follow line"

### Computer Vision

The system includes comprehensive computer vision capabilities:

```bash
# Run computer vision demo
python3 computer_vision.py
```

**Vision Modes**:
- **Obstacle Detection**: Real-time obstacle detection and distance estimation
- **Line Following**: Line detection for autonomous line following
- **Object Tracking**: Neural network-based object detection and tracking
- **Lane Detection**: Lane detection for autonomous driving
- **Color Detection**: Color-based object detection and classification

### Command Line Options

| Option | Description |
|--------|-------------|
| `--keyboard` | Use keyboard input instead of voice |
| `--no-img` | Disable image analysis (faster processing) |

### Interaction Examples

**Voice Mode**: Simply speak to the car - it will listen, process your request, and respond with actions and speech.

**Keyboard Mode**: Type your messages and press Enter for instant responses.

**Voice Commands**: Use natural language commands to control the car directly.

**Example Conversations**:
- "Hello, how are you today?" → Car waves and responds cheerfully
- "What do you see?" → Car analyzes camera feed and describes the environment
- "Show me a celebration!" → Car performs celebration dance with sound effects
- "Go forward" → Car moves forward (voice command)
- "Turn left" → Car turns left (voice command)
- "Stop" → Car stops (voice command)

## Available Actions

### Physical Movements
- **shake head**: Side-to-side head movement
- **nod**: Up-and-down nodding motion
- **wave hands**: Steering wheel movement simulation
- **resist**: Defensive movement pattern
- **act cute**: Gentle forward-backward motion
- **rub hands**: Subtle steering adjustments
- **think**: Contemplative head and body movement
- **twist body**: Rotational body movement
- **celebrate**: Joyful celebration dance
- **depressed**: Slow, sad movement pattern

### Sound Effects
- **honking**: Car horn sound effect
- **start engine**: Engine startup sound

### Voice Commands
- **Movement Commands**: forward, backward, left, right, stop
- **Speed Commands**: speed up, slow down
- **System Commands**: status, help
- **Vision Commands**: look around, follow line

## Computer Vision Features

### Vision Modes

1. **Obstacle Detection**
   - Real-time obstacle detection using edge detection
   - Distance estimation based on object size
   - Bounding box visualization

2. **Line Following**
   - Line detection using Hough transform
   - Binary image processing
   - Line center calculation

3. **Object Tracking**
   - Neural network-based object detection (YOLO support)
   - Real-time object classification
   - Confidence scoring

4. **Lane Detection**
   - Lane line detection for autonomous driving
   - Left and right lane separation
   - Lane angle calculation

5. **Color Detection**
   - HSV-based color detection
   - Support for red, green, blue, yellow objects
   - Contour-based object detection

### Vision Integration

The computer vision system integrates seamlessly with:
- **Reinforcement Learning**: Provides vision features for RL training
- **Voice Commands**: Enables vision-based voice commands
- **AI Conversations**: Provides visual context for AI responses

## Advanced Features

### Configuration Management

The system uses a sophisticated configuration management system:

```python
from config import get_config, validate_required_config

# Get configuration
config = get_config()

# Validate configuration
if validate_required_config():
    print("Configuration is valid")
```

### Voice Command System

Advanced voice command system with features:

```python
from voice_commander import VoiceCommander

# Create voice commander
commander = VoiceCommander(car)

# Register custom commands
commander.register_command(
    "custom_action",
    CommandType.CUSTOM,
    ["custom", "action"],
    custom_function,
    "Custom action description"
)

# Start listening
commander.start_listening()
```

### Computer Vision System

Comprehensive computer vision system:

```python
from computer_vision import VisionSystem, VisionMode

# Create vision system
vision = VisionSystem(car)

# Set vision mode
vision.set_vision_mode(VisionMode.OBSTACLE_DETECTION)

# Start vision processing
vision.start_vision()

# Get detections
detections = vision.get_detections()
```

## Troubleshooting

### Common Issues

1. **Configuration Errors**
   ```bash
   # Run setup script for guidance
   python setup_env.py
   
   # Check configuration
   python config.py
   ```

2. **Voice Recognition Issues**
   - Check microphone permissions
   - Ensure quiet environment
   - Adjust confidence threshold

3. **Camera Issues**
   - Check camera connections
   - Verify camera permissions
   - Adjust camera index in configuration

4. **Hardware Issues**
   - Verify Picar-X connections
   - Check motor and sensor connections
   - Ensure proper power supply

### Debug Mode

Enable debug mode for detailed logging:

```bash
export DEBUG_MODE=true
python gpt_car.py
```

## Development

### Project Structure

```
code/NLP-CV/
├── gpt_car.py              # Main application
├── config.py               # Configuration management
├── voice_commander.py      # Voice command system
├── computer_vision.py      # Computer vision system
├── openai_helper.py        # OpenAI API integration
├── preset_actions.py       # Predefined actions
├── utils.py                # Utility functions
├── keys_template.py        # Configuration template
├── requirements.txt        # Dependencies
└── README.md               # This file
```

### Adding New Features

1. **New Voice Commands**
   ```python
   # Add to voice_commander.py
   self.register_command(
       "new_command",
       CommandType.CUSTOM,
       ["keywords"],
       action_function,
       "Description"
   )
   ```

2. **New Vision Modes**
   ```python
   # Add to computer_vision.py
   def _new_vision_mode(self, frame):
       # Implementation
       return processed_frame, detections
   ```

3. **New Actions**
   ```python
   # Add to preset_actions.py
   def new_action(car):
       # Implementation
       pass
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT-4o and Whisper APIs
- SunFounder for Picar-X hardware
- OpenCV community for computer vision tools
- SpeechRecognition library contributors

---

**Note**: This implementation is designed for educational and research purposes. Always ensure safe operation when using autonomous vehicles.
