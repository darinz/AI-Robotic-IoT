# Environment Variable Configuration Update

This document summarizes the changes made to implement environment variable configuration for the AI Robotic Car project.

## Overview

The project has been updated to use environment variables for sensitive configuration data (API keys) instead of hardcoded values in Python files. This improves security and follows best practices for configuration management.

## Changes Made

### 1. New Configuration System (`code/NLP-CV/config.py`)

- **Created**: `NLP-CV/config.py` - A comprehensive configuration management system
- **Features**:
  - Environment variable support with fallback to local files
  - Support for `.env` files using python-dotenv
  - Backward compatibility with existing `keys.py` files
  - Configuration validation and error reporting
  - Centralized configuration for all components

### 2. Updated Main Application (`code/NLP-CV/gpt_car.py`)

- **Modified**: Import statements to use new configuration system
- **Added**: Configuration validation in main function
- **Enhanced**: Error messages with setup instructions
- **Improved**: Startup process with configuration summary

### 3. Enhanced Voice Commander (`code/NLP-CV/voice_commander.py`)

- **Updated**: To use configuration system for language and confidence settings
- **Added**: Integration with centralized configuration
- **Improved**: Default parameter handling

### 4. Enhanced Computer Vision (`code/NLP-CV/computer_vision.py`)

- **Updated**: To use configuration system for camera settings
- **Added**: Vision parameters from configuration
- **Improved**: Camera initialization with configurable parameters

### 5. Environment Template (`code/env.template`)

- **Created**: Template file for environment variables
- **Includes**: All configurable parameters with default values
- **Provides**: Clear documentation for each setting

### 6. Setup Script (`code/setup_env.py`)

- **Created**: Automated setup and testing script
- **Features**:
  - Creates `.env` file from template
  - Validates environment variables
  - Tests configuration system
  - Provides setup instructions

### 7. Updated Requirements (`code/NLP-CV/requirements.txt`)

- **Added**: `python-dotenv>=1.0.0` for `.env` file support
- **Added**: `pyttsx3>=2.90` for text-to-speech functionality

### 8. Updated Documentation (`code/NLP-CV/README.md`)

- **Updated**: Configuration section with new environment variable options
- **Added**: Quick setup instructions
- **Improved**: Installation and setup documentation

## Configuration Priority

The system now loads configuration in the following priority order:

1. **Environment Variables** (Highest Priority)
   - `OPENAI_API_KEY`
   - `OPENAI_ASSISTANT_ID`
   - Other configurable parameters

2. **`.env` File** (Second Priority)
   - Loaded automatically if python-dotenv is available
   - Supports all configuration parameters

3. **`keys.py` File** (Third Priority)
   - Legacy support for existing installations
   - Only for API keys and Assistant ID

4. **Default Values** (Lowest Priority)
   - Fallback values for all parameters

## New Configuration Parameters

The system now supports the following configurable parameters:

### Required Parameters
- `OPENAI_API_KEY` - Your OpenAI API key
- `OPENAI_ASSISTANT_ID` - Your OpenAI Assistant ID

### Optional Parameters
- `SPEECH_LANGUAGE` - Speech recognition language (default: 'en-US')
- `AUDIO_VOLUME_DB` - Audio volume gain (default: 3)
- `TTS_VOICE` - Text-to-speech voice (default: 'echo')
- `CAMERA_INDEX` - Camera device index (default: 0)
- `VISION_ENABLED` - Enable computer vision (default: true)
- `RL_POWER` - RL motor power (default: 30)
- `RL_SAFE_DISTANCE` - Safe distance for RL (default: 40)
- `RL_DANGER_DISTANCE` - Danger distance for RL (default: 20)
- `VOICE_ENABLED` - Enable voice commands (default: true)
- `VOICE_CONFIDENCE_THRESHOLD` - Voice recognition confidence (default: 0.7)
- `DEBUG_MODE` - Enable debug mode (default: false)
- `LOG_LEVEL` - Logging level (default: 'INFO')

## Migration Guide

### For Existing Users

1. **Option A: Use Environment Variables**
   ```bash
   export OPENAI_API_KEY="your-api-key"
   export OPENAI_ASSISTANT_ID="your-assistant-id"
   ```

2. **Option B: Create .env File**
   ```bash
   cp env.template .env
   # Edit .env with your actual keys
   ```

3. **Option C: Keep Using keys.py**
   - Your existing `keys.py` file will continue to work
   - No changes needed

### For New Users

1. **Run Setup Script**
   ```bash
   python setup_env.py
   ```

2. **Follow Instructions**
   - The script will guide you through the setup process
   - It will create necessary files and validate configuration

3. **Test Configuration**
   ```bash
   python setup_env.py --test
   ```

## Benefits

### Security
- API keys are no longer hardcoded in source files
- Environment variables are not committed to version control
- Multiple configuration options for different deployment scenarios

### Flexibility
- Easy to switch between different API keys for testing
- Support for different environments (development, production)
- Centralized configuration management

### Maintainability
- Single source of truth for configuration
- Clear separation of code and configuration
- Easy to add new configuration parameters

### Developer Experience
- Automated setup process
- Clear error messages and instructions
- Backward compatibility with existing installations

## Testing

To test the new configuration system:

```bash
# Test configuration loading
python code/NLP-CV/config.py

# Test setup script
python code/setup_env.py --test

# Test main application
cd code/NLP-CV
python gpt_car.py
```

## Troubleshooting

### Common Issues

1. **"Configuration validation failed"**
   - Ensure you have set either environment variables or created a `.env` file
   - Check that your API keys are valid

2. **"Could not import configuration"**
   - Ensure you're running from the correct directory
   - Check that all dependencies are installed

3. **"Camera not found"**
   - Adjust `CAMERA_INDEX` in your configuration
   - Ensure camera is properly connected

### Getting Help

1. Run the setup script for guidance: `python setup_env.py`
2. Check the configuration summary: `python NLP-CV/config.py`
3. Review the README for detailed instructions
4. Ensure all dependencies are installed: `pip install -r NLP-CV/requirements.txt`

## Future Enhancements

- Support for encrypted configuration files
- Integration with cloud configuration services
- Configuration validation schemas
- Hot-reloading of configuration changes
- Configuration backup and restore functionality 