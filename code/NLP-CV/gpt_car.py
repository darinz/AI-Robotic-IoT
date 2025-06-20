#!/usr/bin/env python3
"""
AI-Powered Robotic Car with NLP and Computer Vision

This module provides the main entry point for the AI robotic car application,
integrating OpenAI's GPT-4o with speech recognition, computer vision, and
robotic movement control.

"""

import argparse
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Union

# Third-party imports
import speech_recognition as sr
from picarx import Picarx
from robot_hat import Music, Pin

# Local imports
from openai_helper import OpenAiHelper
from config import get_openai_api_key, get_openai_assistant_id, validate_required_config, print_config_info
from preset_actions import actions_dict, sounds_dict
from utils import (
    gray_print, 
    redirect_error_2_null, 
    cancel_redirect_error, 
    speak_block, 
    sox_volume
)

# Configuration constants
DEFAULT_HEAD_TILT = 20
LED_DOUBLE_BLINK_INTERVAL = 0.8
LED_BLINK_INTERVAL = 0.1
ACTION_INTERVAL = 5
CHUNK_SIZE = 8192

# Audio processing settings
LANGUAGE = []  # Empty for auto-detection
VOLUME_DB = 3  # Volume gain (0-5 recommended)
TTS_VOICE = 'echo'  # Options: alloy, echo, fable, onyx, nova, shimmer

# Sound effect actions that trigger audio playback
SOUND_EFFECT_ACTIONS = ["honking", "start engine"]


class RoboticCarController:
    """
    Main controller class for the AI-powered robotic car.
    
    Handles speech recognition, AI processing, robotic actions, and audio output
    in a coordinated manner using multiple threads.
    """
    
    def __init__(self, api_key: str, assistant_id: str, enable_vision: bool = True):
        """
        Initialize the robotic car controller.
        
        Args:
            api_key: OpenAI API key
            assistant_id: OpenAI Assistant ID
            enable_vision: Whether to enable computer vision features
        """
        self.enable_vision = enable_vision
        self.input_mode = 'voice'
        
        # Initialize OpenAI helper
        self.openai_helper = OpenAiHelper(api_key, assistant_id, 'picarx')
        
        # Initialize hardware components
        self._init_hardware()
        
        # Initialize speech recognition
        self._init_speech_recognition()
        
        # Initialize vision system if enabled
        if self.enable_vision:
            self._init_vision_system()
        
        # Threading state management
        self.speech_loaded = False
        self.speech_lock = threading.Lock()
        self.tts_file = None
        
        self.action_status = 'standby'  # 'standby', 'think', 'actions', 'actions_done'
        self.led_status = 'standby'
        self.last_action_status = 'standby'
        self.last_led_status = 'standby'
        self.actions_to_be_done = []
        self.action_lock = threading.Lock()
        
        # Start background threads
        self._start_background_threads()
    
    def _init_hardware(self):
        """Initialize hardware components with error handling."""
        try:
            self.car = Picarx()
            time.sleep(1)  # Allow hardware to stabilize
            self.music = Music()
            self.led = Pin('LED')
            
            # Enable speaker switch
            os.popen("pinctrl set 20 op dh")
            
            # Set initial position
            self.car.reset()
            self.car.set_cam_tilt_angle(DEFAULT_HEAD_TILT)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize hardware: {e}")
    
    def _init_speech_recognition(self):
        """Initialize speech recognition system."""
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_adjustment_damping = 0.16
        self.recognizer.dynamic_energy_ratio = 1.6
    
    def _init_vision_system(self):
        """Initialize computer vision system."""
        try:
            from vilib import Vilib
            import cv2
            
            Vilib.camera_start(vflip=False, hflip=False)
            Vilib.show_fps()
            Vilib.display(local=False, web=True)
            
            # Wait for camera to initialize
            while not Vilib.flask_start:
                time.sleep(0.01)
            
            time.sleep(0.5)
            self.vilib = Vilib
            self.cv2 = cv2
            gray_print("Vision system initialized successfully")
            
        except ImportError as e:
            gray_print(f"Warning: Vision system not available: {e}")
            self.enable_vision = False
        except Exception as e:
            gray_print(f"Warning: Failed to initialize vision system: {e}")
            self.enable_vision = False
    
    def _start_background_threads(self):
        """Start background threads for speech and action processing."""
        # Speech processing thread
        self.speak_thread = threading.Thread(target=self._speech_handler, daemon=True)
        self.speak_thread.start()
        
        # Action processing thread
        self.action_thread = threading.Thread(target=self._action_handler, daemon=True)
        self.action_thread.start()
    
    def _speech_handler(self):
        """Background thread for handling text-to-speech output."""
        while True:
            with self.speech_lock:
                is_loaded = self.speech_loaded
            
            if is_loaded:
                speak_block(self.music, self.tts_file)
                with self.speech_lock:
                    self.speech_loaded = False
            
            time.sleep(0.05)
    
    def _action_handler(self):
        """Background thread for handling robotic actions and LED indicators."""
        last_action_time = time.time()
        last_led_time = time.time()
        
        while True:
            with self.action_lock:
                state = self.action_status
            
            # LED status management
            self.led_status = state
            
            if self.led_status != self.last_led_status:
                last_led_time = 0
                self.last_led_status = self.led_status
            
            # LED behavior based on state
            self._update_led_behavior(last_led_time)
            
            # Action execution
            if state == 'standby':
                self.last_action_status = 'standby'
                if time.time() - last_action_time > ACTION_INTERVAL:
                    last_action_time = time.time()
                    # TODO: Implement idle actions
            elif state == 'think':
                if self.last_action_status != 'think':
                    self.last_action_status = 'think'
                    self._perform_thinking_action()
            elif state == 'actions':
                self.last_action_status = 'actions'
                self._execute_actions()
                with self.action_lock:
                    self.action_status = 'actions_done'
                last_action_time = time.time()
            
            time.sleep(0.01)
    
    def _update_led_behavior(self, last_led_time: float):
        """Update LED behavior based on current status."""
        if self.led_status == 'standby':
            if time.time() - last_led_time > LED_DOUBLE_BLINK_INTERVAL:
                self.led.off()
                self.led.on()
                time.sleep(0.1)
                self.led.off()
                time.sleep(0.1)
                self.led.on()
                time.sleep(0.1)
                self.led.off()
        elif self.led_status == 'think':
            if time.time() - last_led_time > LED_BLINK_INTERVAL:
                self.led.off()
                time.sleep(LED_BLINK_INTERVAL)
                self.led.on()
                time.sleep(LED_BLINK_INTERVAL)
        elif self.led_status == 'actions':
            self.led.on()
    
    def _perform_thinking_action(self):
        """Perform thinking animation."""
        try:
            from preset_actions import keep_think
            keep_think(self.car)
        except Exception as e:
            gray_print(f"Thinking action error: {e}")
    
    def _execute_actions(self):
        """Execute queued robotic actions."""
        with self.action_lock:
            actions = self.actions_to_be_done.copy()
        
        for action in actions:
            try:
                if action in actions_dict:
                    actions_dict[action](self.car)
                else:
                    gray_print(f"Unknown action: {action}")
                time.sleep(0.5)
            except Exception as e:
                gray_print(f"Action execution error: {e}")
    
    def set_input_mode(self, mode: str):
        """Set the input mode (voice or keyboard)."""
        if mode not in ['voice', 'keyboard']:
            raise ValueError("Input mode must be 'voice' or 'keyboard'")
        self.input_mode = mode
    
    def listen_for_voice(self) -> Optional[str]:
        """
        Listen for voice input and convert to text.
        
        Returns:
            Transcribed text or None if failed
        """
        gray_print("Listening...")
        
        with self.action_lock:
            self.action_status = 'standby'
        
        # Redirect stderr to suppress ALSA errors
        stderr_backup = redirect_error_2_null()
        
        try:
            with sr.Microphone(chunk_size=CHUNK_SIZE) as source:
                cancel_redirect_error(stderr_backup)
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source)
            
            # Speech-to-text conversion
            start_time = time.time()
            result = self.openai_helper.stt(audio, language=LANGUAGE)
            gray_print(f"STT processing time: {time.time() - start_time:.3f}s")
            
            return result if result else None
            
        except Exception as e:
            gray_print(f"Voice recognition error: {e}")
            return None
        finally:
            cancel_redirect_error(stderr_backup)
    
    def get_keyboard_input(self) -> Optional[str]:
        """
        Get input from keyboard.
        
        Returns:
            User input or None if empty
        """
        with self.action_lock:
            self.action_status = 'standby'
        
        try:
            user_input = input('\033[1;30mInput: \033[0m').encode(sys.stdin.encoding).decode('utf-8')
            return user_input if user_input.strip() else None
        except (EOFError, KeyboardInterrupt):
            return None
    
    def process_user_input(self, user_input: str) -> Dict[str, Union[List[str], str]]:
        """
        Process user input through OpenAI API.
        
        Args:
            user_input: User's text or voice input
            
        Returns:
            Dictionary containing actions and response text
        """
        with self.action_lock:
            self.action_status = 'think'
        
        start_time = time.time()
        
        try:
            if self.enable_vision:
                # Capture and process image
                img_path = './img_input.jpg'
                self.cv2.imwrite(img_path, self.vilib.img)
                response = self.openai_helper.dialogue_with_img(user_input, img_path)
            else:
                response = self.openai_helper.dialogue(user_input)
            
            gray_print(f'AI processing time: {time.time() - start_time:.3f}s')
            return self._parse_response(response)
            
        except Exception as e:
            gray_print(f"AI processing error: {e}")
            return {"actions": [], "answer": "I'm sorry, I encountered an error processing your request."}
    
    def _parse_response(self, response: Union[Dict, str]) -> Dict[str, Union[List[str], str]]:
        """
        Parse AI response into structured format.
        
        Args:
            response: Raw response from OpenAI
            
        Returns:
            Parsed response with actions and answer
        """
        try:
            if isinstance(response, dict):
                actions = response.get('actions', [])
                answer = response.get('answer', '')
            else:
                actions = []
                answer = str(response) if response else ''
            
            return {"actions": actions, "answer": answer}
            
        except Exception as e:
            gray_print(f"Response parsing error: {e}")
            return {"actions": [], "answer": str(response) if response else ""}
    
    def execute_response(self, response: Dict[str, Union[List[str], str]]):
        """
        Execute the AI response (actions and speech).
        
        Args:
            response: Parsed response containing actions and answer
        """
        actions = response.get('actions', [])
        answer = response.get('answer', '')
        
        # Separate sound effects from physical actions
        sound_actions = []
        physical_actions = []
        
        for action in actions:
            if action in SOUND_EFFECT_ACTIONS:
                sound_actions.append(action)
            else:
                physical_actions.append(action)
        
        try:
            # Generate speech if there's an answer
            tts_status = False
            if answer:
                tts_status = self._generate_speech(answer)
            
            # Execute physical actions
            with self.action_lock:
                self.actions_to_be_done = physical_actions
                gray_print(f'Actions to execute: {physical_actions}')
                self.action_status = 'actions'
            
            # Play sound effects
            for sound_action in sound_actions:
                try:
                    sounds_dict[sound_action](self.music)
                except Exception as e:
                    gray_print(f'Sound effect error: {e}')
            
            # Trigger speech playback
            if tts_status:
                with self.speech_lock:
                    self.speech_loaded = True
            
            # Wait for speech to complete
            if tts_status:
                while True:
                    with self.speech_lock:
                        if not self.speech_loaded:
                            break
                    time.sleep(0.01)
            
            # Wait for actions to complete
            while True:
                with self.action_lock:
                    if self.action_status != 'actions':
                        break
                time.sleep(0.01)
            
            print()  # New line for readability
            
        except Exception as e:
            gray_print(f'Response execution error: {e}')
    
    def _generate_speech(self, text: str) -> bool:
        """
        Generate speech from text using OpenAI TTS.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            True if successful, False otherwise
        """
        try:
            start_time = time.time()
            timestamp = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
            
            raw_file = f"./tts/{timestamp}_raw.wav"
            final_file = f"./tts/{timestamp}_{VOLUME_DB}dB.wav"
            
            # Generate TTS
            tts_status = self.openai_helper.text_to_speech(
                text, raw_file, TTS_VOICE, response_format='wav'
            )
            
            if tts_status:
                # Apply volume adjustment
                tts_status = sox_volume(raw_file, final_file, VOLUME_DB)
                if tts_status:
                    self.tts_file = final_file
            
            gray_print(f'TTS processing time: {time.time() - start_time:.3f}s')
            return tts_status
            
        except Exception as e:
            gray_print(f'TTS generation error: {e}')
            return False
    
    def run(self):
        """Main application loop."""
        try:
            while True:
                # Reset camera position
                self.car.set_cam_tilt_angle(DEFAULT_HEAD_TILT)
                
                # Get user input
                if self.input_mode == 'voice':
                    user_input = self.listen_for_voice()
                else:
                    user_input = self.get_keyboard_input()
                
                if not user_input:
                    continue
                
                # Process through AI
                response = self.process_user_input(user_input)
                
                # Execute response
                self.execute_response(response)
                
        except KeyboardInterrupt:
            gray_print("\nShutting down gracefully...")
        except Exception as e:
            gray_print(f"Application error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources and reset hardware."""
        try:
            if self.enable_vision and hasattr(self, 'vilib'):
                self.vilib.camera_close()
            self.car.reset()
        except Exception as e:
            gray_print(f"Cleanup error: {e}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Robotic Car with NLP and Computer Vision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Voice interaction with vision
  %(prog)s --keyboard         # Keyboard interaction with vision
  %(prog)s --keyboard --no-img # Keyboard interaction without vision
        """
    )
    
    parser.add_argument(
        '--keyboard',
        action='store_true',
        help='Use keyboard input instead of voice recognition'
    )
    
    parser.add_argument(
        '--no-img',
        action='store_true',
        help='Disable image analysis (faster processing)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    # Set up working directory
    current_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_path)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate configuration
    print("Validating configuration...")
    if not validate_required_config():
        print("\nConfiguration validation failed!")
        print("Please set the required environment variables or create a keys.py file.")
        print("\nYou can set environment variables with:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("  export OPENAI_ASSISTANT_ID='your-assistant-id'")
        print("\nOr create a .env file in the project root with:")
        print("  OPENAI_API_KEY=your-api-key")
        print("  OPENAI_ASSISTANT_ID=your-assistant-id")
        sys.exit(1)
    
    # Print configuration summary
    print_config_info()
    
    try:
        # Initialize controller
        controller = RoboticCarController(
            api_key=get_openai_api_key(),
            assistant_id=get_openai_assistant_id(),
            enable_vision=not args.no_img
        )
        
        # Set input mode
        controller.set_input_mode('keyboard' if args.keyboard else 'voice')
        
        # Print startup information
        gray_print("AI Robotic Car initialized successfully!")
        gray_print(f"Input mode: {controller.input_mode}")
        gray_print(f"Vision enabled: {controller.enable_vision}")
        gray_print("Press Ctrl+C to exit")
        print()
        
        # Run the application
        controller.run()
        
    except Exception as e:
        print(f"\033[31mERROR: {e}\033[m")
        sys.exit(1)


if __name__ == "__main__":
    main()

