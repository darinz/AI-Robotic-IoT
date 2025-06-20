#!/bin/bash
# IoT Smart Home Clap Detection System - Installation Script
# =========================================================
#
# This script automates the installation and setup of the IoT smart home system.
# It installs dependencies, configures the system, and provides setup guidance.
#
# Author: IoT Project Team
# Date: 2025

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if running on Raspberry Pi
check_raspberry_pi() {
    if [[ -f /proc/cpuinfo ]] && grep -q "Raspberry Pi" /proc/cpuinfo; then
        print_success "Raspberry Pi detected"
        return 0
    else
        print_warning "This script is designed for Raspberry Pi. Some features may not work on other systems."
        return 1
    fi
}

# Function to check Python version
check_python_version() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        print_success "Python $PYTHON_VERSION found"
        
        # Check if version is 3.7 or higher
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 7) else 1)"; then
            print_success "Python version is compatible (3.7+)"
        else
            print_error "Python 3.7 or higher is required. Current version: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 is not installed"
        exit 1
    fi
}

# Function to install system dependencies
install_system_dependencies() {
    print_status "Installing system dependencies..."
    
    # Update package list
    sudo apt update
    
    # Install required packages
    sudo apt install -y python3-pip python3-venv git curl wget
    
    # Install MQTT broker (Mosquitto)
    print_status "Installing MQTT broker (Mosquitto)..."
    sudo apt install -y mosquitto mosquitto-clients
    
    print_success "System dependencies installed"
}

# Function to install Python dependencies
install_python_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Install required Python packages
    pip3 install -r requirements.txt
    
    print_success "Python dependencies installed"
}

# Function to configure MQTT broker
configure_mqtt_broker() {
    print_status "Configuring MQTT broker..."
    
    # Create backup of original config
    sudo cp /etc/mosquitto/mosquitto.conf /etc/mosquitto/mosquitto.conf.backup
    
    # Create new configuration
    cat << EOF | sudo tee /etc/mosquitto/mosquitto.conf
# IoT Smart Home MQTT Configuration
# =================================

# Network settings
listener 1883
allow_anonymous false
password_file /etc/mosquitto/passwd

# Logging
log_type all
log_dest file /var/log/mosquitto/mosquitto.log
log_dest stdout

# Security
persistence true
persistence_location /var/lib/mosquitto/

# Performance
max_queued_messages 200
max_inflight_messages 20
EOF
    
    # Create password file
    print_status "Creating MQTT user account..."
    read -p "Enter MQTT username: " MQTT_USERNAME
    read -s -p "Enter MQTT password: " MQTT_PASSWORD
    echo
    
    # Create password file
    echo "$MQTT_USERNAME:$MQTT_PASSWORD" | sudo mosquitto_passwd -c /etc/mosquitto/passwd -
    
    # Set proper permissions
    sudo chown mosquitto:mosquitto /etc/mosquitto/passwd
    sudo chmod 600 /etc/mosquitto/passwd
    
    # Restart Mosquitto
    sudo systemctl restart mosquitto
    sudo systemctl enable mosquitto
    
    print_success "MQTT broker configured"
    
    # Save credentials to config file
    update_config_file "$MQTT_USERNAME" "$MQTT_PASSWORD"
}

# Function to update configuration file
update_config_file() {
    local username="$1"
    local password="$2"
    
    print_status "Updating configuration file..."
    
    # Update config.py with MQTT credentials
    if [[ -f config.py ]]; then
        sed -i "s/MQTT_USERNAME = \"USERNAME\"/MQTT_USERNAME = \"$username\"/" config.py
        sed -i "s/MQTT_PASSWORD = \"PASSWORD\"/MQTT_PASSWORD = \"$password\"/" config.py
        print_success "Configuration file updated"
    else
        print_warning "config.py not found. Please update MQTT credentials manually."
    fi
}

# Function to setup GPIO permissions
setup_gpio_permissions() {
    print_status "Setting up GPIO permissions..."
    
    # Add user to gpio group
    sudo usermod -a -G gpio $USER
    
    # Create udev rules for GPIO access
    cat << EOF | sudo tee /etc/udev/rules.d/99-gpio.rules
# GPIO access for IoT project
SUBSYSTEM=="bcm2835-gpiomem", GROUP="gpio", MODE="0660"
SUBSYSTEM=="gpio", GROUP="gpio", MODE="0660"
EOF
    
    # Reload udev rules
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    
    print_success "GPIO permissions configured"
}

# Function to create systemd service
create_systemd_service() {
    print_status "Creating systemd service..."
    
    # Get current directory
    CURRENT_DIR=$(pwd)
    
    # Create service file
    cat << EOF | sudo tee /etc/systemd/system/clap-detector.service
[Unit]
Description=IoT Smart Home Clap Detection System
After=network.target mosquitto.service
Wants=mosquitto.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$CURRENT_DIR
ExecStart=/usr/bin/python3 $CURRENT_DIR/clap_detector.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and enable service
    sudo systemctl daemon-reload
    sudo systemctl enable clap-detector.service
    
    print_success "Systemd service created and enabled"
}

# Function to setup log rotation
setup_log_rotation() {
    print_status "Setting up log rotation..."
    
    cat << EOF | sudo tee /etc/logrotate.d/clap-detector
$PWD/clap_detector.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
    postrotate
        systemctl reload clap-detector.service
    endscript
}
EOF
    
    print_success "Log rotation configured"
}

# Function to run system tests
run_system_tests() {
    print_status "Running system tests..."
    
    # Test Python imports
    if python3 -c "import RPi.GPIO, paho.mqtt.client, json, logging, threading, time, dataclasses" 2>/dev/null; then
        print_success "Python dependencies test passed"
    else
        print_error "Python dependencies test failed"
        return 1
    fi
    
    # Test MQTT connection
    if mosquitto_pub -h localhost -u "$MQTT_USERNAME" -P "$MQTT_PASSWORD" -t "test/topic" -m "test" 2>/dev/null; then
        print_success "MQTT connection test passed"
    else
        print_warning "MQTT connection test failed"
    fi
    
    # Test GPIO access (if on Raspberry Pi)
    if check_raspberry_pi; then
        if python3 -c "import RPi.GPIO; RPi.GPIO.setmode(RPi.GPIO.BCM); print('GPIO test passed')" 2>/dev/null; then
            print_success "GPIO access test passed"
        else
            print_warning "GPIO access test failed - check permissions"
        fi
    fi
    
    print_success "System tests completed"
}

# Function to display setup instructions
display_setup_instructions() {
    echo
    echo "=========================================="
    echo "IoT Smart Home System Setup Complete!"
    echo "=========================================="
    echo
    echo "Next steps:"
    echo "1. Connect your sound sensor to GPIO 17"
    echo "2. Configure your smart light devices"
    echo "3. Update device IDs in config.py if needed"
    echo "4. Test the system: python3 clap_detector.py"
    echo "5. Start the service: sudo systemctl start clap-detector"
    echo
    echo "Useful commands:"
    echo "- Check service status: sudo systemctl status clap-detector"
    echo "- View logs: sudo journalctl -u clap-detector -f"
    echo "- Stop service: sudo systemctl stop clap-detector"
    echo "- Restart service: sudo systemctl restart clap-detector"
    echo
    echo "For the Godot mobile app:"
    echo "1. Open iot-godot-app/ in Godot"
    echo "2. Update MQTT settings in mqtt.gd"
    echo "3. Build and deploy to your mobile device"
    echo
}

# Main installation function
main() {
    echo "IoT Smart Home Clap Detection System - Installation"
    echo "=================================================="
    echo
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        print_error "This script should not be run as root"
        exit 1
    fi
    
    # Check system requirements
    check_raspberry_pi
    check_python_version
    
    # Install dependencies
    install_system_dependencies
    install_python_dependencies
    
    # Configure system
    configure_mqtt_broker
    setup_gpio_permissions
    create_systemd_service
    setup_log_rotation
    
    # Run tests
    run_system_tests
    
    # Display instructions
    display_setup_instructions
    
    print_success "Installation completed successfully!"
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 