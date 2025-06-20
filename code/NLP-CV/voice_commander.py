#!/usr/bin/env python3
"""
Voice Command System for PiCar

This module provides advanced voice command capabilities for controlling the PiCar,
including natural language processing, command recognition, and voice feedback.

"""

import speech_recognition as sr
import pyttsx3
import threading
import time
import json
import os
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from picarx import Picarx
import sys
sys.path.append('.')
from config import get_config

class CommandType(Enum):
    MOVEMENT = "movement"
    VISION = "vision"
    SYSTEM = "system"
    CUSTOM = "custom"

@dataclass
class VoiceCommand:
    """Represents a voice command with its metadata"""
    command_type: CommandType
    keywords: List[str]
    action: Callable
    description: str
    confidence_threshold: float = 0.7
    requires_confirmation: bool = False

class VoiceCommander:
    """
    Advanced voice command system for PiCar control
    """
    
    def __init__(self, car: Picarx, language: str = None):
        self.car = car
        # Use configuration if available, otherwise use provided language
        config = get_config()
        self.language = language or config.language
        self.confidence_threshold = config.voice_confidence_threshold
        
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        
        # Configure speech recognition
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        
        # Configure text-to-speech
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.8)
        
        # Voice commands registry
        self.commands: Dict[str, VoiceCommand] = {}
        self.command_history: List[Tuple[str, float]] = []
        
        # Threading
        self.is_listening = False
        self.listen_thread = None
        self.stop_listening = threading.Event()
        
        # Initialize commands
        self._register_default_commands()
    
    def _register_default_commands(self):
        """Register default voice commands"""
        
        # Movement commands
        self.register_command(
            "forward",
            CommandType.MOVEMENT,
            ["forward", "go forward", "move forward", "drive forward"],
            self._move_forward,
            "Move the car forward"
        )
        
        self.register_command(
            "backward",
            CommandType.MOVEMENT,
            ["backward", "go backward", "move backward", "reverse", "back up"],
            self._move_backward,
            "Move the car backward"
        )
        
        self.register_command(
            "left",
            CommandType.MOVEMENT,
            ["left", "turn left", "go left", "steer left"],
            self._turn_left,
            "Turn the car left"
        )
        
        self.register_command(
            "right",
            CommandType.MOVEMENT,
            ["right", "turn right", "go right", "steer right"],
            self._turn_right,
            "Turn the car right"
        )
        
        self.register_command(
            "stop",
            CommandType.MOVEMENT,
            ["stop", "halt", "brake", "stop moving"],
            self._stop_car,
            "Stop the car"
        )
        
        # Speed control
        self.register_command(
            "speed_up",
            CommandType.MOVEMENT,
            ["speed up", "faster", "accelerate", "increase speed"],
            self._speed_up,
            "Increase car speed"
        )
        
        self.register_command(
            "slow_down",
            CommandType.MOVEMENT,
            ["slow down", "slower", "decelerate", "reduce speed"],
            self._slow_down,
            "Decrease car speed"
        )
        
        # System commands
        self.register_command(
            "status",
            CommandType.SYSTEM,
            ["status", "how are you", "car status", "system status"],
            self._get_status,
            "Get car status"
        )
        
        self.register_command(
            "help",
            CommandType.SYSTEM,
            ["help", "commands", "what can you do", "available commands"],
            self._show_help,
            "Show available commands"
        )
        
        # Vision commands
        self.register_command(
            "look_around",
            CommandType.VISION,
            ["look around", "scan", "obstacle detection", "what do you see"],
            self._look_around,
            "Scan surroundings for obstacles"
        )
        
        self.register_command(
            "follow_line",
            CommandType.VISION,
            ["follow line", "line following", "track line"],
            self._follow_line,
            "Enable line following mode"
        )
    
    def register_command(self, name: str, command_type: CommandType, 
                        keywords: List[str], action: Callable, 
                        description: str, confidence_threshold: float = 0.7,
                        requires_confirmation: bool = False):
        """Register a new voice command"""
        self.commands[name] = VoiceCommand(
            command_type=command_type,
            keywords=keywords,
            action=action,
            description=description,
            confidence_threshold=confidence_threshold,
            requires_confirmation=requires_confirmation
        )
    
    def speak(self, text: str):
        """Convert text to speech"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
    
    def listen(self) -> Optional[str]:
        """Listen for voice input and return transcribed text"""
        try:
            with sr.Microphone() as source:
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                text = self.recognizer.recognize_google(audio, language=self.language)
                print(f"Recognized: {text}")
                return text.lower()
                
        except sr.WaitTimeoutError:
            print("No speech detected")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results: {e}")
            return None
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return None
    
    def match_command(self, text: str) -> Optional[Tuple[str, VoiceCommand, float]]:
        """Match spoken text to registered commands"""
        best_match = None
        best_confidence = 0.0
        
        for command_name, command in self.commands.items():
            for keyword in command.keywords:
                # Simple keyword matching (can be enhanced with fuzzy matching)
                if keyword in text:
                    confidence = len(keyword) / len(text)  # Simple confidence metric
                    if confidence > best_confidence and confidence >= command.confidence_threshold:
                        best_confidence = confidence
                        best_match = (command_name, command, confidence)
        
        return best_match
    
    def execute_command(self, command_name: str, command: VoiceCommand) -> bool:
        """Execute a voice command"""
        try:
            print(f"Executing command: {command_name}")
            self.speak(f"Executing {command_name}")
            
            # Execute the command
            result = command.action()
            
            # Log command execution
            self.command_history.append((command_name, time.time()))
            
            return True
            
        except Exception as e:
            print(f"Error executing command {command_name}: {e}")
            self.speak(f"Error executing {command_name}")
            return False
    
    def process_voice_input(self) -> bool:
        """Process a single voice input"""
        text = self.listen()
        if not text:
            return False
        
        # Match command
        match = self.match_command(text)
        if not match:
            self.speak("Command not recognized")
            return False
        
        command_name, command, confidence = match
        
        # Check if confirmation is required
        if command.requires_confirmation:
            self.speak(f"Confirm {command_name}")
            confirmation = self.listen()
            if not confirmation or "yes" not in confirmation.lower():
                self.speak("Command cancelled")
                return False
        
        # Execute command
        return self.execute_command(command_name, command)
    
    def start_listening(self):
        """Start continuous voice listening in a separate thread"""
        if self.is_listening:
            return
        
        self.is_listening = True
        self.stop_listening.clear()
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        self.speak("Voice commander activated")
    
    def stop_listening(self):
        """Stop continuous voice listening"""
        self.is_listening = False
        self.stop_listening.set()
        if self.listen_thread:
            self.listen_thread.join()
        self.speak("Voice commander deactivated")
    
    def _listen_loop(self):
        """Main listening loop"""
        while self.is_listening and not self.stop_listening.is_set():
            try:
                self.process_voice_input()
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in listening loop: {e}")
                time.sleep(1)
    
    # Movement command implementations
    def _move_forward(self):
        """Move car forward"""
        self.car.forward(50)
        self.speak("Moving forward")
        return True
    
    def _move_backward(self):
        """Move car backward"""
        self.car.backward(50)
        self.speak("Moving backward")
        return True
    
    def _turn_left(self):
        """Turn car left"""
        self.car.set_dir_servo_angle(-30)
        self.car.forward(40)
        self.speak("Turning left")
        return True
    
    def _turn_right(self):
        """Turn car right"""
        self.car.set_dir_servo_angle(30)
        self.car.forward(40)
        self.speak("Turning right")
        return True
    
    def _stop_car(self):
        """Stop the car"""
        self.car.stop()
        self.speak("Car stopped")
        return True
    
    def _speed_up(self):
        """Increase car speed"""
        # Implementation depends on current speed tracking
        self.speak("Increasing speed")
        return True
    
    def _slow_down(self):
        """Decrease car speed"""
        # Implementation depends on current speed tracking
        self.speak("Decreasing speed")
        return True
    
    # System command implementations
    def _get_status(self):
        """Get car status"""
        # Get sensor readings
        try:
            distance = self.car.ultrasonic.read()
            grayscale_data = self.car.get_grayscale_data()
            
            status_text = f"Distance to obstacle: {distance} cm. "
            status_text += f"Grayscale readings: {grayscale_data}"
            
            self.speak(status_text)
            return True
        except Exception as e:
            self.speak("Unable to read sensor status")
            return False
    
    def _show_help(self):
        """Show available commands"""
        help_text = "Available commands: "
        for command_name, command in self.commands.items():
            help_text += f"{command_name}, "
        
        self.speak(help_text)
        return True
    
    # Vision command implementations
    def _look_around(self):
        """Scan surroundings"""
        self.speak("Scanning surroundings for obstacles")
        # Implementation would integrate with computer vision
        return True
    
    def _follow_line(self):
        """Enable line following mode"""
        self.speak("Enabling line following mode")
        # Implementation would integrate with line following algorithm
        return True
    
    def get_command_statistics(self) -> Dict:
        """Get statistics about command usage"""
        if not self.command_history:
            return {}
        
        command_counts = {}
        for command_name, _ in self.command_history:
            command_counts[command_name] = command_counts.get(command_name, 0) + 1
        
        return {
            "total_commands": len(self.command_history),
            "command_counts": command_counts,
            "last_command": self.command_history[-1] if self.command_history else None
        }
    
    def save_command_history(self, filename: str = "voice_commands.json"):
        """Save command history to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.command_history, f, indent=2)
            print(f"Command history saved to {filename}")
        except Exception as e:
            print(f"Error saving command history: {e}")
    
    def load_command_history(self, filename: str = "voice_commands.json"):
        """Load command history from file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    self.command_history = json.load(f)
                print(f"Command history loaded from {filename}")
        except Exception as e:
            print(f"Error loading command history: {e}")


def main():
    """Demo function for voice commander"""
    try:
        car = Picarx()
        commander = VoiceCommander(car)
        
        print("Voice Commander Demo")
        print("Say 'help' to see available commands")
        print("Say 'stop listening' to exit")
        
        commander.start_listening()
        
        # Keep running until stopped
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            commander.stop_listening()
            car.stop()
            print("Demo ended")
    
    except Exception as e:
        print(f"Error in demo: {e}")


if __name__ == "__main__":
    main() 