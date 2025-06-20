#!/usr/bin/env python3
"""
Configuration Management for AI Robotic Car

This module handles configuration management using environment variables
with fallback to local configuration files for development.

"""

import os
import sys
from typing import Optional
from pathlib import Path

# Try to import dotenv for .env file support
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

class Config:
    """
    Configuration manager that prioritizes environment variables
    with fallback to local configuration files
    """
    
    def __init__(self):
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment variables and fallback sources"""
        # Load .env file if dotenv is available
        if DOTENV_AVAILABLE:
            # Try to load from project root first
            project_root = Path(__file__).parent.parent
            env_path = project_root / '.env'
            if env_path.exists():
                load_dotenv(env_path)
            else:
                # Try current directory
                load_dotenv()
        
        # OpenAI API Configuration
        self.openai_api_key = self._get_openai_api_key()
        self.openai_assistant_id = self._get_openai_assistant_id()
        
        # Speech and Audio Configuration
        self.language = os.getenv('SPEECH_LANGUAGE', 'en-US')
        self.volume_db = float(os.getenv('AUDIO_VOLUME_DB', '3'))
        self.tts_voice = os.getenv('TTS_VOICE', 'echo')
        
        # Vision Configuration
        self.camera_index = int(os.getenv('CAMERA_INDEX', '0'))
        self.vision_enabled = os.getenv('VISION_ENABLED', 'true').lower() == 'true'
        
        # RL Configuration
        self.rl_power = int(os.getenv('RL_POWER', '30'))
        self.rl_safe_distance = int(os.getenv('RL_SAFE_DISTANCE', '40'))
        self.rl_danger_distance = int(os.getenv('RL_DANGER_DISTANCE', '20'))
        
        # Voice Command Configuration
        self.voice_enabled = os.getenv('VOICE_ENABLED', 'true').lower() == 'true'
        self.voice_confidence_threshold = float(os.getenv('VOICE_CONFIDENCE_THRESHOLD', '0.7'))
        
        # System Configuration
        self.debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    def _get_openai_api_key(self) -> str:
        """Get OpenAI API key from environment or fallback sources"""
        # First priority: Environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            return api_key
        
        # Second priority: Local keys file
        api_key = self._load_from_keys_file('OPENAI_API_KEY')
        if api_key:
            return api_key
        
        # Third priority: .env file
        api_key = self._load_from_env_file('OPENAI_API_KEY')
        if api_key:
            return api_key
        
        # Final fallback: Empty string (will cause error when used)
        return ""
    
    def _get_openai_assistant_id(self) -> str:
        """Get OpenAI Assistant ID from environment or fallback sources"""
        # First priority: Environment variable
        assistant_id = os.getenv('OPENAI_ASSISTANT_ID')
        if assistant_id:
            return assistant_id
        
        # Second priority: Local keys file
        assistant_id = self._load_from_keys_file('OPENAI_ASSISTANT_ID')
        if assistant_id:
            return assistant_id
        
        # Third priority: .env file
        assistant_id = self._load_from_env_file('OPENAI_ASSISTANT_ID')
        if assistant_id:
            return assistant_id
        
        # Final fallback: Empty string (will cause error when used)
        return ""
    
    def _load_from_keys_file(self, key_name: str) -> Optional[str]:
        """Load configuration from local keys.py file"""
        try:
            # Try to import from keys.py
            keys_path = Path(__file__).parent / 'keys.py'
            if keys_path.exists():
                # Add current directory to path temporarily
                sys.path.insert(0, str(keys_path.parent))
                try:
                    from keys import OPENAI_API_KEY, OPENAI_ASSISTANT_ID
                    if key_name == 'OPENAI_API_KEY':
                        return OPENAI_API_KEY if OPENAI_API_KEY != "your-openai-api-key-here" else None
                    elif key_name == 'OPENAI_ASSISTANT_ID':
                        return OPENAI_ASSISTANT_ID if OPENAI_ASSISTANT_ID != "your-assistant-id-here" else None
                except ImportError:
                    pass
                finally:
                    # Remove from path
                    sys.path.pop(0)
        except Exception as e:
            if self.debug_mode:
                print(f"Warning: Could not load from keys file: {e}")
        return None
    
    def _load_from_env_file(self, key_name: str) -> Optional[str]:
        """Load configuration from .env file"""
        try:
            env_path = Path(__file__).parent.parent / '.env'
            if env_path.exists():
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            env_key, env_value = line.split('=', 1)
                            if env_key == key_name:
                                return env_value.strip('"\'')
        except Exception as e:
            if self.debug_mode:
                print(f"Warning: Could not load from .env file: {e}")
        return None
    
    def validate_config(self) -> bool:
        """Validate that required configuration is present"""
        errors = []
        
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is not set")
        
        if not self.openai_assistant_id:
            errors.append("OPENAI_ASSISTANT_ID is not set")
        
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            print("\nPlease set the required environment variables or create a keys.py file.")
            return False
        
        return True
    
    def get_config_summary(self) -> dict:
        """Get a summary of current configuration (without sensitive data)"""
        return {
            'openai_api_key_set': bool(self.openai_api_key),
            'openai_assistant_id_set': bool(self.openai_assistant_id),
            'language': self.language,
            'volume_db': self.volume_db,
            'tts_voice': self.tts_voice,
            'camera_index': self.camera_index,
            'vision_enabled': self.vision_enabled,
            'rl_power': self.rl_power,
            'rl_safe_distance': self.rl_safe_distance,
            'rl_danger_distance': self.rl_danger_distance,
            'voice_enabled': self.voice_enabled,
            'voice_confidence_threshold': self.voice_confidence_threshold,
            'debug_mode': self.debug_mode,
            'log_level': self.log_level
        }
    
    def print_config_summary(self):
        """Print a summary of current configuration"""
        summary = self.get_config_summary()
        print("Configuration Summary:")
        print("=" * 50)
        for key, value in summary.items():
            print(f"{key}: {value}")
        print("=" * 50)


# Global configuration instance
config = Config()


def get_openai_api_key() -> str:
    """Get OpenAI API key from configuration"""
    return config.openai_api_key


def get_openai_assistant_id() -> str:
    """Get OpenAI Assistant ID from configuration"""
    return config.openai_assistant_id


def validate_required_config() -> bool:
    """Validate that required configuration is present"""
    return config.validate_config()


def print_config_info():
    """Print configuration information"""
    config.print_config_summary()


# Backward compatibility functions
def get_config():
    """Get the global configuration instance"""
    return config


if __name__ == "__main__":
    # Test configuration loading
    print("Testing configuration loading...")
    print_config_info()
    
    if validate_required_config():
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration has errors")
        print("\nTo fix this, you can:")
        print("1. Set environment variables:")
        print("   export OPENAI_API_KEY='your-api-key'")
        print("   export OPENAI_ASSISTANT_ID='your-assistant-id'")
        print("2. Create a .env file in the project root:")
        print("   OPENAI_API_KEY=your-api-key")
        print("   OPENAI_ASSISTANT_ID=your-assistant-id")
        print("3. Create a keys.py file in the NLP-CV directory:")
        print("   OPENAI_API_KEY = 'your-api-key'")
        print("   OPENAI_ASSISTANT_ID = 'your-assistant-id'") 