# Code Directory

This folder contains the code for the **AI-Robotic-IoT** project, organized into key technology areas.

> *Note: This project is a work in progress. Code for each module will continue to be added and updated as development progresses.*

## `IoT/` – Remote IoT Integration

This module enables the robotic car to communicate with and control smart home devices remotely.

**Contents:**
- Scripts for device control using MQTT/HTTP
- Setup for Wi-Fi-enabled microcontrollers (e.g., ESP32, Raspberry Pi)
- Basic dashboard or mobile app interface (if applicable)

## `RL/` – Reinforcement Learning for Self-Driving

This module contains the logic for training the robotic vehicle using reinforcement learning.

**Contents:**
- Training environments
- Reward function design
- Policy networks and model checkpoints
- Offline logs and analysis tools

## `NLP-CV/` – Voice Interaction & Computer Vision

This module combines NLP for voice command interpretation and CV for perception tasks like lane detection and obstacle recognition.

**Contents:**
- NLP: Voice command recognition pipeline
- CV: Real-time camera-based detection and analysis
- Integration scripts for communication with the RL module

## Usage Notes

Each module is self-contained and can be tested independently or as part of the full robotic system. Make sure to follow setup instructions and install dependencies as listed in each subdirectory.

## Directory Tree

```bash
code/
├── IoT/
│   └── ...  # IoT control scripts and configs
├── RL/
│   └── ...  # RL model training, policy files
└── NLP-CV/
    └── ...  # NLP and CV modules
```

Stay tuned as more functionality and integration code is added!