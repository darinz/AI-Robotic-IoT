#!/usr/bin/env python3
"""
Setup script for AI Robotic Car Project

This script helps users set up the project by:
1. Checking system requirements
2. Installing dependencies
3. Setting up configuration
4. Validating the setup

"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_step(step: str) -> None:
    """Print a step message."""
    print(f"\n▶ {step}")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"✅ {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"❌ {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"⚠️  {message}")


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    print_step("Checking Python version...")
    
    if sys.version_info < (3, 8):
        print_error("Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print_success(f"Python {sys.version.split()[0]} is compatible")
    return True


def check_system_requirements() -> bool:
    """Check system requirements."""
    print_step("Checking system requirements...")
    
    # Check if running on Linux/Raspberry Pi
    if not sys.platform.startswith('linux'):
        print_warning("This project is designed for Linux/Raspberry Pi")
        print_warning("Some features may not work on other platforms")
    
    # Check for sudo access
    if os.geteuid() != 0:
        print_warning("Not running as root (sudo)")
        print_warning("Some hardware features may require sudo access")
    
    # Check for required system packages
    required_packages = ['python3-pip', 'python3-venv']
    missing_packages = []
    
    for package in required_packages:
        if not shutil.which(package.replace('python3-', '')):
            missing_packages.append(package)
    
    if missing_packages:
        print_warning(f"Missing system packages: {', '.join(missing_packages)}")
        print("You may need to install them manually:")
        print("  sudo apt update")
        print("  sudo apt install python3-pip python3-venv")
    else:
        print_success("System requirements met")
    
    return True


def install_python_dependencies() -> bool:
    """Install Python dependencies."""
    print_step("Installing Python dependencies...")
    
    try:
        # Install core dependencies
        dependencies = [
            'openai>=1.0.0',
            'openai-whisper>=20231117',
            'SpeechRecognition>=3.10.0',
            'PyAudio>=0.2.11',
            'sox>=1.4.1',
            'opencv-python>=4.8.0',
            'numpy>=1.24.0',
            'Pillow>=10.0.0'
        ]
        
        for dep in dependencies:
            print(f"Installing {dep}...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', dep, '--break-system-packages'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print_error(f"Failed to install {dep}")
                print(result.stderr)
                return False
        
        print_success("Python dependencies installed successfully")
        return True
        
    except Exception as e:
        print_error(f"Failed to install dependencies: {e}")
        return False


def setup_configuration() -> bool:
    """Set up configuration files."""
    print_step("Setting up configuration...")
    
    # Check if keys.py exists
    if os.path.exists('keys.py'):
        print_success("Configuration file already exists")
        return True
    
    # Check if template exists
    if not os.path.exists('keys_template.py'):
        print_error("Configuration template not found")
        return False
    
    # Copy template to keys.py
    try:
        shutil.copy('keys_template.py', 'keys.py')
        print_success("Configuration file created from template")
        print_warning("Please edit keys.py with your actual API credentials")
        return True
    except Exception as e:
        print_error(f"Failed to create configuration file: {e}")
        return False


def validate_setup() -> bool:
    """Validate the setup."""
    print_step("Validating setup...")
    
    # Check if keys.py exists and has content
    if not os.path.exists('keys.py'):
        print_error("Configuration file not found")
        return False
    
    # Check if main files exist
    required_files = ['gpt_car.py', 'openai_helper.py', 'preset_actions.py', 'utils.py']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print_error(f"Missing required files: {', '.join(missing_files)}")
        return False
    
    # Test imports
    try:
        import openai
        import speech_recognition
        print_success("Core dependencies imported successfully")
    except ImportError as e:
        print_error(f"Import test failed: {e}")
        return False
    
    print_success("Setup validation completed")
    return True


def create_directories() -> bool:
    """Create necessary directories."""
    print_step("Creating directories...")
    
    directories = ['tts', 'sounds', 'logs']
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print_success(f"Created directory: {directory}")
        except Exception as e:
            print_error(f"Failed to create directory {directory}: {e}")
            return False
    
    return True


def print_next_steps() -> None:
    """Print next steps for the user."""
    print_header("Setup Complete!")
    
    print("\nNext steps:")
    print("1. Edit keys.py with your OpenAI API credentials")
    print("2. Create an OpenAI Assistant at https://platform.openai.com/assistants")
    print("3. Configure your Assistant with the provided description")
    print("4. Run the application:")
    print("   sudo python3 gpt_car.py")
    
    print("\nFor more information, see the README.md file")
    print("\nHappy coding! 🚗🤖")


def main() -> None:
    """Main setup function."""
    print_header("AI Robotic Car Setup")
    
    print("This script will help you set up the AI Robotic Car project.")
    print("Make sure you have:")
    print("- A Raspberry Pi with Picar-X hardware")
    print("- Internet connection")
    print("- OpenAI API key")
    
    # Check if user wants to continue
    response = input("\nContinue with setup? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Setup cancelled.")
        return
    
    # Run setup steps
    steps = [
        ("Python Version Check", check_python_version),
        ("System Requirements", check_system_requirements),
        ("Python Dependencies", install_python_dependencies),
        ("Configuration Setup", setup_configuration),
        ("Directory Creation", create_directories),
        ("Setup Validation", validate_setup)
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print_error(f"Setup failed at: {step_name}")
            print("\nPlease fix the issues and run setup again.")
            return
    
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
    except Exception as e:
        print_error(f"Setup failed with unexpected error: {e}")
        sys.exit(1) 