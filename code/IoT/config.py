#!/usr/bin/env python3
"""
IoT Smart Home Configuration
============================

This module contains all configuration settings for the IoT smart home system.
Modify these values to customize the system for your specific setup.

Author: IoT Project Team
Date: 2025
"""

# MQTT Broker Configuration
# =========================

# Broker connection settings
MQTT_BROKER_HOST = "localhost"      # MQTT broker IP address or hostname
MQTT_BROKER_PORT = 1883            # MQTT broker port (default: 1883)
MQTT_KEEP_ALIVE = 60               # Keep-alive interval in seconds
MQTT_USERNAME = "USERNAME"         # MQTT broker username
MQTT_PASSWORD = "PASSWORD"         # MQTT broker password

# MQTT Topic Configuration
# ========================

# Topic structure for device communication
MQTT_TOPIC_BASE = "home"           # Base topic for all home devices
MQTT_COMMAND_TOPIC = "commands/MQTTtoONOFF"    # Topic for sending commands
MQTT_STATUS_TOPIC = "ONOFFtoMQTT"              # Topic for receiving status
MQTT_STATUS_REQUEST_TOPIC = "commands/status"  # Topic for requesting status

# Device Configuration
# ====================

# Smart light device IDs and topics
SMART_LIGHTS = {
    'light_1': {
        'device_id': "C8F09EB5B18C",
        'name': "Living Room Light",
        'description': "Primary smart light in living room"
    },
    'light_2': {
        'device_id': "C8F09EB94208", 
        'name': "Bedroom Light",
        'description': "Secondary smart light in bedroom"
    }
}

# Raspberry Pi Status Topic
RASPBERRY_STATUS_TOPIC = "raspberry/status"

# GPIO Configuration
# ==================

# Sound sensor GPIO settings
SOUND_SENSOR_GPIO_CHANNEL = 17     # GPIO pin for sound sensor (BCM numbering)
SOUND_SENSOR_BOUNCE_TIME = 300     # Debounce time in milliseconds
SOUND_SENSOR_CLAP_TIMEOUT = 1.0    # Time window for detecting multiple claps (seconds)

# Clap Detection Configuration
# ===========================

# Clap detection sensitivity and timing
CLAP_DETECTION = {
    'single_clap_timeout': 1.0,     # Seconds to wait for second clap
    'min_clap_interval': 0.1,       # Minimum time between claps (seconds)
    'max_clap_interval': 2.0,       # Maximum time between claps (seconds)
    'clap_threshold': 0.5,          # Sound threshold for clap detection
}

# Logging Configuration
# ====================

# Logging settings
LOGGING_CONFIG = {
    'level': 'INFO',                # Log level: DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'file': 'clap_detector.log',    # Log file name
    'max_size': 10 * 1024 * 1024,  # Maximum log file size (10MB)
    'backup_count': 5,              # Number of backup log files
}

# System Configuration
# ===================

# General system settings
SYSTEM_CONFIG = {
    'debug_mode': False,            # Enable debug mode for development
    'auto_reconnect': True,         # Automatically reconnect to MQTT broker
    'reconnect_interval': 5,        # Reconnection interval in seconds
    'max_reconnect_attempts': 10,   # Maximum reconnection attempts
    'graceful_shutdown_timeout': 5, # Timeout for graceful shutdown (seconds)
}

# Light Control Logic
# ===================

# Single clap behavior configuration
SINGLE_CLAP_BEHAVIOR = {
    'cycle_mode': True,             # Enable cycling through light states
    'cycle_order': ['light_1', 'light_2', 'both_off'],  # Cycling order
    'skip_off_state': False,        # Skip the all-off state in cycle
}

# Double clap behavior configuration  
DOUBLE_CLAP_BEHAVIOR = {
    'master_toggle': True,          # Enable master toggle functionality
    'toggle_both_lights': True,     # Toggle both lights together
    'preserve_individual_control': False,  # Allow individual control after double clap
}

# Network Configuration
# ====================

# Network and connectivity settings
NETWORK_CONFIG = {
    'connection_timeout': 10,       # Connection timeout in seconds
    'read_timeout': 30,             # Read timeout in seconds
    'write_timeout': 10,            # Write timeout in seconds
    'retry_on_failure': True,       # Retry operations on failure
    'max_retries': 3,               # Maximum retry attempts
}

# Security Configuration
# =====================

# Security and authentication settings
SECURITY_CONFIG = {
    'use_tls': False,               # Use TLS/SSL encryption
    'verify_certificate': True,     # Verify SSL certificates
    'certificate_path': None,       # Path to SSL certificate
    'private_key_path': None,       # Path to private key
    'ca_certificate_path': None,    # Path to CA certificate
}

# Performance Configuration
# ========================

# Performance and optimization settings
PERFORMANCE_CONFIG = {
    'max_concurrent_operations': 5, # Maximum concurrent MQTT operations
    'operation_timeout': 5,         # Timeout for individual operations
    'buffer_size': 1024,            # Network buffer size
    'enable_compression': False,    # Enable MQTT message compression
}

# Development Configuration
# ========================

# Development and testing settings
DEVELOPMENT_CONFIG = {
    'enable_mock_mode': False,      # Enable mock mode for testing
    'mock_device_responses': True,  # Mock device responses in test mode
    'enable_profiling': False,      # Enable performance profiling
    'log_mqtt_messages': False,     # Log all MQTT messages (verbose)
}

# Helper Functions
# ================

def get_device_topic(device_id: str, topic_type: str) -> str:
    """
    Generate MQTT topic for a specific device and topic type.
    
    Args:
        device_id: The device ID
        topic_type: Type of topic ('command', 'status', 'status_request')
    
    Returns:
        Complete MQTT topic string
    """
    topic_map = {
        'command': MQTT_COMMAND_TOPIC,
        'status': MQTT_STATUS_TOPIC,
        'status_request': MQTT_STATUS_REQUEST_TOPIC
    }
    
    if topic_type not in topic_map:
        raise ValueError(f"Invalid topic type: {topic_type}")
    
    return f"{MQTT_TOPIC_BASE}/{device_id}/{topic_map[topic_type]}"

def get_all_device_topics() -> dict:
    """
    Get all MQTT topics for all configured devices.
    
    Returns:
        Dictionary with device names as keys and topic dictionaries as values
    """
    topics = {}
    
    for light_name, light_config in SMART_LIGHTS.items():
        device_id = light_config['device_id']
        topics[light_name] = {
            'command': get_device_topic(device_id, 'command'),
            'status': get_device_topic(device_id, 'status'),
            'status_request': get_device_topic(device_id, 'status_request'),
            'device_id': device_id,
            'name': light_config['name']
        }
    
    return topics

def validate_configuration() -> bool:
    """
    Validate the configuration settings.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    # Check required settings
    if not MQTT_BROKER_HOST:
        print("ERROR: MQTT_BROKER_HOST is not set")
        return False
    
    if not MQTT_USERNAME or not MQTT_PASSWORD:
        print("WARNING: MQTT credentials are not set")
    
    if SOUND_SENSOR_GPIO_CHANNEL < 1 or SOUND_SENSOR_GPIO_CHANNEL > 40:
        print("ERROR: Invalid GPIO channel number")
        return False
    
    # Check device configuration
    for light_name, light_config in SMART_LIGHTS.items():
        if not light_config['device_id']:
            print(f"ERROR: Device ID not set for {light_name}")
            return False
    
    print("Configuration validation passed")
    return True

# Configuration validation on import
if __name__ == "__main__":
    validate_configuration() 