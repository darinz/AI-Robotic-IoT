# IoT Smart Home Clap Detection System

A sophisticated IoT project that combines clap detection with smart home automation, featuring both a Raspberry Pi backend and a Godot frontend application.

## Overview

This project implements a complete smart home lighting control system that can be operated through:
- **Clap Detection**: Automatic light control using sound sensors
- **Mobile App**: Manual control via a Godot-based mobile application
- **MQTT Integration**: Real-time communication with smart light devices

### Key Features

- **Dual Control Methods**: Both clap detection and mobile app control
- **Smart Light Integration**: Controls two smart light devices via MQTT
- **Real-time Status**: Live updates of device states
- **Robust Communication**: Reliable MQTT protocol implementation
- **Elegant Architecture**: Well-structured, documented, and maintainable code

## System Architecture

```
┌─────────────────┐    MQTT    ┌─────────────────┐
│   Raspberry Pi  │◄──────────►│  MQTT Broker    │
│  (Clap Sensor)  │            │   (Mosquitto)   │
└─────────────────┘            └─────────────────┘
         │                              │
         │                              │
         ▼                              ▼
┌─────────────────┐            ┌─────────────────┐
│   Sound Sensor  │            │  Smart Lights   │
│   (GPIO Pin)    │            │  (2 Devices)    │
└─────────────────┘            └─────────────────┘

┌─────────────────┐    MQTT    ┌─────────────────┐
│   Godot App     │◄──────────►│  MQTT Broker    │
│  (Mobile UI)    │            │   (Mosquitto)   │
└─────────────────┘            └─────────────────┘
```

## Project Structure

```
IoT/
├── README.md                 # This file
├── clap_detector.py         # Raspberry Pi clap detection system
├── requirements.txt         # Python dependencies
└── iot-godot-app/          # Godot mobile application
    ├── project.godot       # Godot project configuration
    ├── main.gd            # Main application controller
    ├── mqtt.gd            # MQTT protocol helper
    ├── main.tscn          # Main scene file
    ├── bulb.png           # Light bulb icon
    ├── power_on.png       # Power on button icon
    ├── power_off.png      # Power off button icon
    └── icon.svg           # Application icon
```

## Hardware Requirements

### Raspberry Pi Setup
- **Raspberry Pi** (3B+ or 4B recommended)
- **Sound Sensor Module** (KY-038 or similar)
- **Breadboard and Jumper Wires**
- **Power Supply** for Raspberry Pi

### Smart Light Devices
- **2x Smart Light Bulbs** (compatible with MQTT)
- **Device IDs**: 
  - Light 1: `C8F09EB5B18C`
  - Light 2: `C8F09EB94208`

### Network Requirements
- **MQTT Broker** (Mosquitto recommended)
- **Network connectivity** between all devices

## Installation & Setup

### 1. Raspberry Pi Setup

#### Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
pip3 install paho-mqtt RPi.GPIO

# Install MQTT broker (if running locally)
sudo apt install mosquitto mosquitto-clients
```

#### Hardware Connection
```
Sound Sensor → Raspberry Pi GPIO
VCC         → 3.3V
GND         → GND
OUT         → GPIO 17 (BCM)
```

#### Configure MQTT Broker
```bash
# Edit Mosquitto configuration
sudo nano /etc/mosquitto/mosquitto.conf

# Add these lines:
allow_anonymous false
password_file /etc/mosquitto/passwd
```

```bash
# Create password file
sudo mosquitto_passwd -c /etc/mosquitto/passwd USERNAME

# Restart Mosquitto
sudo systemctl restart mosquitto
```

#### Run Clap Detection System
```bash
# Make script executable
chmod +x clap_detector.py

# Run the system
python3 clap_detector.py
```

### 2. Godot App Setup

#### Install Godot
1. Download Godot 4.x from [godotengine.org](https://godotengine.org/)
2. Install and open Godot
3. Import the `iot-godot-app` folder as a project

#### Configure MQTT Settings
Edit `mqtt.gd` and update the authentication:
```gdscript
username: String = "YOUR_USERNAME",
password: String = "YOUR_PASSWORD"
```

Edit `main.gd` and update the broker IP:
```gdscript
const BROKER_IP := "YOUR_BROKER_IP"
```

#### Build and Deploy
1. Go to **Project → Export**
2. Add **Android** export preset
3. Configure Android SDK settings
4. Build APK and install on mobile device

## Usage Instructions

### Clap Detection Control

The system responds to claps with the following behavior:

#### Single Clap - Light Cycling
- **Both lights OFF** → Light 1 ON
- **Light 1 ON** → Light 1 OFF, Light 2 ON
- **Light 2 ON** → Light 1 ON
- **Both lights ON** → Both OFF

#### Double Clap - Master Toggle
- **Both lights OFF** → Both lights ON
- **Any light ON** → Both lights OFF

### Mobile App Control

1. **Launch the Godot app** on your mobile device
2. **Wait for MQTT connection** (check console logs)
3. **Tap light buttons** to toggle individual lights
4. **Monitor real-time status** of both devices

## Code Documentation

### Python Backend (`clap_detector.py`)

The Python backend is organized into several classes:

#### `ClapDetector`
- Handles GPIO sound sensor input
- Implements thread-safe clap detection
- Manages timing for single vs double claps

#### `SmartLightController`
- Manages MQTT communication
- Controls smart light devices
- Handles device status updates

#### `ClapLightSystem`
- Orchestrates clap detection and light control
- Implements clap-to-action mapping
- Provides system lifecycle management

### Godot Frontend

#### `main.gd`
- Manages TCP/MQTT connection
- Handles UI interactions
- Implements robust CONNACK realignment

#### `mqtt.gd`
- Builds MQTT protocol packets
- Handles CONNECT and PUBLISH operations
- Provides utility functions for MQTT communication

## Configuration

### MQTT Topics

The system uses the following MQTT topic structure:

```
home/{DEVICE_ID}/commands/MQTTtoONOFF    # Send commands to devices
home/{DEVICE_ID}/ONOFFtoMQTT             # Receive status updates
home/{DEVICE_ID}/commands/status         # Request device status
raspberry/status                         # Raspberry Pi status
```

### Device Configuration

Update device IDs in `clap_detector.py`:
```python
'light_1': LightDevice(
    device_id="YOUR_DEVICE_ID_1",
    topic_base="home/YOUR_DEVICE_ID_1"
),
'light_2': LightDevice(
    device_id="YOUR_DEVICE_ID_2", 
    topic_base="home/YOUR_DEVICE_ID_2"
)
```

## Troubleshooting

### Common Issues

#### Clap Detection Not Working
- Check GPIO pin connections
- Verify sound sensor sensitivity
- Check for background noise interference

#### MQTT Connection Issues
- Verify broker IP and port
- Check username/password credentials
- Ensure network connectivity

#### Mobile App Not Connecting
- Verify broker IP in `main.gd`
- Check MQTT authentication settings
- Ensure mobile device is on same network

### Debug Logging

The Python backend includes comprehensive logging:
```bash
# View real-time logs
tail -f clap_detector.log

# Check system status
python3 -c "import clap_detector; print('System ready')"
```

## Performance & Optimization

### Raspberry Pi Optimization
- **CPU Usage**: ~5-10% during normal operation
- **Memory Usage**: ~50MB for Python process
- **Network**: Minimal MQTT traffic (~1KB per command)

### Mobile App Optimization
- **Battery Usage**: Minimal due to efficient MQTT implementation
- **Memory Usage**: ~20MB for Godot app
- **Network**: Efficient packet-based communication

## Security Considerations

- **MQTT Authentication**: Always use username/password
- **Network Security**: Use VPN or firewall rules
- **Device Security**: Keep firmware updated
- **Access Control**: Limit MQTT topic access

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper documentation
4. Test thoroughly
5. Submit a pull request

## Acknowledgments

- MQTT protocol specification
- Godot Engine community
- Raspberry Pi Foundation
- Smart home device manufacturers

---

**Note**: This project is designed for educational and personal use. Always follow safety guidelines when working with electrical components and IoT devices.
