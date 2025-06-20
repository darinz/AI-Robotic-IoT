#!/usr/bin/env python3
"""
IoT Smart Home Clap Detection System
====================================

This module implements a clap detection system that controls smart lights via MQTT.
It uses a sound sensor connected to a Raspberry Pi GPIO pin to detect claps and
sends commands to control two smart light devices.

Features:
- Single clap: Cycles through light states (Light 1 → Light 2 → Both Off)
- Double clap: Toggles both lights on/off
- MQTT integration for smart device control
- Thread-safe clap detection with debouncing
- Graceful shutdown handling

Author: IoT Project Team
Date: 2025
"""

import RPi.GPIO as GPIO
import time
import threading
import json
import logging
from typing import Optional
from dataclasses import dataclass
import paho.mqtt.client as mqtt

# Import configuration
try:
    import config
except ImportError:
    print("Warning: config.py not found. Using default settings.")
    # Fallback configuration
    class Config:
        MQTT_BROKER_HOST = "localhost"
        MQTT_BROKER_PORT = 1883
        MQTT_USERNAME = "USERNAME"
        MQTT_PASSWORD = "PASSWORD"
        SOUND_SENSOR_GPIO_CHANNEL = 17
        SOUND_SENSOR_BOUNCE_TIME = 300
        SOUND_SENSOR_CLAP_TIMEOUT = 1.0
        LOGGING_CONFIG = {
            'level': 'INFO',
            'format': '%(asctime)s - %(levelname)s - %(message)s',
            'file': 'clap_detector.log'
        }
        SMART_LIGHTS = {
            'light_1': {'device_id': "C8F09EB5B18C", 'name': "Light 1"},
            'light_2': {'device_id': "C8F09EB94208", 'name': "Light 2"}
        }
    config = Config()

# Configure logging based on configuration
logging.basicConfig(
    level=getattr(logging, config.LOGGING_CONFIG.get('level', 'INFO')),
    format=config.LOGGING_CONFIG.get('format', '%(asctime)s - %(levelname)s - %(message)s'),
    handlers=[
        logging.FileHandler(config.LOGGING_CONFIG.get('file', 'clap_detector.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class LightDevice:
    """Represents a smart light device with its MQTT configuration."""
    device_id: str
    topic_base: str
    name: str
    is_on: bool = False
    
    @property
    def status_topic(self) -> str:
        """Get the MQTT topic for device status updates."""
        return f"{self.topic_base}/ONOFFtoMQTT"
    
    @property
    def command_topic(self) -> str:
        """Get the MQTT topic for sending commands to the device."""
        return f"{self.topic_base}/commands/MQTTtoONOFF"
    
    @property
    def status_request_topic(self) -> str:
        """Get the MQTT topic for requesting device status."""
        return f"{self.topic_base}/commands/status"

class ClapDetector:
    """
    Handles clap detection using GPIO sound sensor with thread-safe operations.
    
    This class manages the detection of claps using a sound sensor connected to
    a GPIO pin. It implements debouncing and timing logic to distinguish between
    single and double claps.
    """
    
    def __init__(self, gpio_channel: int = None, clap_timeout: float = None, 
                 bounce_time: int = None):
        """
        Initialize the clap detector.
        
        Args:
            gpio_channel: GPIO pin number for the sound sensor
            clap_timeout: Time window to detect multiple claps (seconds)
            bounce_time: Debounce time for GPIO events (milliseconds)
        """
        # Use configuration values with fallback to parameters
        self.gpio_channel = gpio_channel or config.SOUND_SENSOR_GPIO_CHANNEL
        self.clap_timeout = clap_timeout or config.SOUND_SENSOR_CLAP_TIMEOUT
        self.bounce_time = bounce_time or config.SOUND_SENSOR_BOUNCE_TIME
        
        # Thread-safe state management
        self.clap_count = 0
        self.clap_timer: Optional[threading.Timer] = None
        self.lock = threading.Lock()
        
        # Setup GPIO
        self._setup_gpio()
    
    def _setup_gpio(self) -> None:
        """Configure GPIO pin for sound sensor input."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.gpio_channel, GPIO.IN)
        
        # Add event detection for falling edge (sound detected)
        GPIO.add_event_detect(
            self.gpio_channel, 
            GPIO.FALLING, 
            bouncetime=self.bounce_time
        )
        GPIO.add_event_callback(self.gpio_channel, self._clap_callback)
        
        logger.info(f"GPIO channel {self.gpio_channel} configured for clap detection")
    
    def _clap_callback(self, channel: int) -> None:
        """
        Callback function triggered when a clap is detected.
        
        This function is called by the GPIO event system when the sound sensor
        detects a sound above the threshold (falling edge).
        
        Args:
            channel: GPIO channel that triggered the event
        """
        with self.lock:
            self.clap_count += 1
            logger.debug(f"Clap detected! Count: {self.clap_count}")
            
            if self.clap_count == 1:
                # First clap: start timer to wait for potential second clap
                self.clap_timer = threading.Timer(
                    self.clap_timeout, 
                    self._process_clap_sequence
                )
                self.clap_timer.start()
                
            elif self.clap_count == 2:
                # Second clap detected: cancel timer and process immediately
                if self.clap_timer:
                    self.clap_timer.cancel()
                # Process in a separate thread to avoid blocking GPIO
                threading.Thread(target=self._process_clap_sequence).start()
    
    def _process_clap_sequence(self) -> None:
        """
        Process the detected clap sequence and trigger appropriate actions.
        
        This method is called after the clap timeout or when a second clap
        is detected. It determines whether it was a single or double clap
        and calls the appropriate handler.
        """
        with self.lock:
            current_count = self.clap_count
            self.clap_count = 0
            self.clap_timer = None
        
        # Determine clap type and handle accordingly
        if current_count == 1:
            logger.info("Single clap detected - cycling light states")
            self._handle_single_clap()
        elif current_count == 2:
            logger.info("Double clap detected - toggling both lights")
            self._handle_double_clap()
        else:
            logger.warning(f"Unexpected clap count: {current_count}")
    
    def _handle_single_clap(self) -> None:
        """Handle single clap: cycle through light states."""
        # This method will be implemented by the main controller
        pass
    
    def _handle_double_clap(self) -> None:
        """Handle double clap: toggle both lights on/off."""
        # This method will be implemented by the main controller
        pass
    
    def cleanup(self) -> None:
        """Clean up GPIO resources."""
        GPIO.cleanup()
        logger.info("GPIO cleanup completed")

class SmartLightController:
    """
    Manages smart light devices via MQTT communication.
    
    This class handles the MQTT connection and provides methods to control
    smart light devices by sending commands and monitoring their status.
    """
    
    def __init__(self, broker_host: str = None, broker_port: int = None,
                 username: str = None, password: str = None):
        """
        Initialize the MQTT client and smart light devices.
        
        Args:
            broker_host: MQTT broker hostname or IP address
            broker_port: MQTT broker port number
            username: MQTT broker username
            password: MQTT broker password
        """
        # Use configuration values with fallback to parameters
        self.broker_host = broker_host or config.MQTT_BROKER_HOST
        self.broker_port = broker_port or config.MQTT_BROKER_PORT
        self.username = username or config.MQTT_USERNAME
        self.password = password or config.MQTT_PASSWORD
        
        # Initialize smart light devices from configuration
        self.lights = {}
        for light_name, light_config in config.SMART_LIGHTS.items():
            device_id = light_config['device_id']
            self.lights[light_name] = LightDevice(
                device_id=device_id,
                topic_base=f"home/{device_id}",
                name=light_config.get('name', light_name)
            )
        
        # Setup MQTT client
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        
        # Set will message for graceful disconnection
        self.client.will_set(
            'raspberry/status', 
            json.dumps({"status": "Off"}).encode()
        )
        
        # Set authentication
        self.client.username_pw_set(self.username, self.password)
        
        logger.info("Smart light controller initialized")
    
    def connect(self) -> None:
        """Establish connection to MQTT broker."""
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
            logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise
    
    def _on_connect(self, client, userdata, flags, rc) -> None:
        """
        MQTT connection callback.
        
        Args:
            client: MQTT client instance
            userdata: User data passed to callback
            flags: Connection flags
            rc: Return code (0 = success)
        """
        if rc == 0:
            logger.info("MQTT connection established successfully")
            
            # Subscribe to device status topics
            for light in self.lights.values():
                client.subscribe(light.status_topic)
                # Request initial status
                client.publish(light.status_request_topic, '{}')
                logger.debug(f"Subscribed to {light.status_topic}")
        else:
            logger.error(f"MQTT connection failed with code {rc}")
    
    def _on_message(self, client, userdata, msg) -> None:
        """
        MQTT message callback for handling device status updates.
        
        Args:
            client: MQTT client instance
            userdata: User data passed to callback
            flags: Message flags
            msg: MQTT message object
        """
        try:
            # Extract device ID from topic
            device_id = msg.topic.split('/')[1]
            payload = msg.payload.decode('utf-8')
            
            # Parse status from payload (format: "cmd:0" or "cmd:1")
            onoff_status = payload.split(',')[0].split(':')[1]
            is_on = bool(int(onoff_status))
            
            # Update corresponding light status
            for light in self.lights.values():
                if light.device_id == device_id:
                    light.is_on = is_on
                    logger.info(f"Light {light.name} ({device_id}) status updated: {'ON' if is_on else 'OFF'}")
                    break
                    
        except (IndexError, ValueError) as e:
            logger.error(f"Failed to parse MQTT message: {e}")
    
    def toggle_light(self, light_name: str) -> None:
        """
        Toggle the state of a specific light.
        
        Args:
            light_name: Name of the light to toggle ('light_1' or 'light_2')
        """
        if light_name not in self.lights:
            logger.error(f"Unknown light: {light_name}")
            return
        
        light = self.lights[light_name]
        new_state = not light.is_on
        light.is_on = new_state
        
        command = json.dumps({"cmd": int(new_state)})
        self.client.publish(light.command_topic, command)
        
        logger.info(f"Toggled {light.name} to {'ON' if new_state else 'OFF'}")
    
    def set_light_state(self, light_name: str, is_on: bool) -> None:
        """
        Set the state of a specific light.
        
        Args:
            light_name: Name of the light to control ('light_1' or 'light_2')
            is_on: True to turn on, False to turn off
        """
        if light_name not in self.lights:
            logger.error(f"Unknown light: {light_name}")
            return
        
        light = self.lights[light_name]
        light.is_on = is_on
        
        command = json.dumps({"cmd": int(is_on)})
        self.client.publish(light.command_topic, command)
        
        logger.info(f"Set {light.name} to {'ON' if is_on else 'OFF'}")
    
    def get_light_status(self) -> dict:
        """
        Get the current status of all lights.
        
        Returns:
            Dictionary with light names as keys and boolean status as values
        """
        return {name: light.is_on for name, light in self.lights.items()}
    
    def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Disconnected from MQTT broker")

class ClapLightSystem:
    """
    Main system class that combines clap detection with smart light control.
    
    This class orchestrates the interaction between the clap detector and
    the smart light controller to provide a complete clap-controlled
    lighting system.
    """
    
    def __init__(self):
        """Initialize the complete clap-controlled lighting system."""
        self.clap_detector = ClapDetector()
        self.light_controller = SmartLightController()
        
        # Connect clap handlers to light controller
        self.clap_detector._handle_single_clap = self._handle_single_clap
        self.clap_detector._handle_double_clap = self._handle_double_clap
        
        logger.info("Clap-controlled lighting system initialized")
    
    def start(self) -> None:
        """Start the system by connecting to MQTT broker."""
        try:
            self.light_controller.connect()
            logger.info("Clap-controlled lighting system started")
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            raise
    
    def _handle_single_clap(self) -> None:
        """
        Handle single clap: cycle through light states.
        
        State transitions:
        - Both off → Light 1 on
        - Light 1 on → Light 1 off, Light 2 on  
        - Light 2 on → Light 1 on
        - Both on → Both off
        """
        status = self.light_controller.get_light_status()
        light_1_on = status['light_1']
        light_2_on = status['light_2']
        
        if not light_1_on and not light_2_on:
            # Both off → Light 1 on
            self.light_controller.set_light_state('light_1', True)
        elif light_1_on and not light_2_on:
            # Light 1 on → Light 1 off, Light 2 on
            self.light_controller.set_light_state('light_1', False)
            self.light_controller.set_light_state('light_2', True)
        elif not light_1_on and light_2_on:
            # Light 2 on → Light 1 on
            self.light_controller.set_light_state('light_1', True)
        else:
            # Both on → Both off
            self.light_controller.set_light_state('light_1', False)
            self.light_controller.set_light_state('light_2', False)
    
    def _handle_double_clap(self) -> None:
        """
        Handle double clap: toggle both lights on/off.
        
        If both lights are off, turn both on.
        If any light is on, turn both off.
        """
        status = self.light_controller.get_light_status()
        light_1_on = status['light_1']
        light_2_on = status['light_2']
        
        if not light_1_on and not light_2_on:
            # Both off → Both on
            self.light_controller.set_light_state('light_1', True)
            self.light_controller.set_light_state('light_2', True)
        else:
            # Any on → Both off
            self.light_controller.set_light_state('light_1', False)
            self.light_controller.set_light_state('light_2', False)
    
    def stop(self) -> None:
        """Stop the system and cleanup resources."""
        self.light_controller.disconnect()
        self.clap_detector.cleanup()
        logger.info("Clap-controlled lighting system stopped")

def main():
    """
    Main entry point for the clap detection system.
    
    This function initializes and runs the complete system, handling
    graceful shutdown on keyboard interrupt.
    """
    system = None
    
    try:
        # Validate configuration if available
        if hasattr(config, 'validate_configuration'):
            if not config.validate_configuration():
                logger.error("Configuration validation failed")
                return
        
        # Initialize and start the system
        system = ClapLightSystem()
        system.start()
        
        logger.info("Clap detection system running. Press Ctrl+C to stop.")
        
        # Keep the main thread alive
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        # Ensure proper cleanup
        if system:
            system.stop()
        logger.info("System shutdown complete")

if __name__ == "__main__":
    main()