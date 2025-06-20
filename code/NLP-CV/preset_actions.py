#!/usr/bin/env python3
"""
Preset Robotic Actions for AI Car

This module defines a comprehensive set of predefined robotic movements and
actions that the AI-powered car can perform. Each action is designed to
convey specific emotions or responses through physical movement.
"""

from time import sleep
import random
from math import sin, cos, pi
from typing import Any, Dict, List


class RoboticAction:
    """
    Base class for robotic actions with common functionality.
    
    Provides a standardized interface for all robotic movements
    with error handling and safety checks.
    """
    
    def __init__(self, name: str, description: str, duration: float = 1.0):
        """
        Initialize a robotic action.
        
        Args:
            name: Action name
            description: Human-readable description
            duration: Typical duration in seconds
        """
        self.name = name
        self.description = description
        self.duration = duration
    
    def execute(self, car: Any) -> bool:
        """
        Execute the robotic action.
        
        Args:
            car: Picar-X car instance
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._perform_action(car)
            return True
        except Exception as e:
            print(f"Action '{self.name}' failed: {e}")
            return False
    
    def _perform_action(self, car: Any) -> None:
        """
        Perform the actual action. Must be implemented by subclasses.
        
        Args:
            car: Picar-X car instance
        """
        raise NotImplementedError("Subclasses must implement _perform_action")


class PhysicalAction(RoboticAction):
    """Base class for physical movement actions."""
    
    def __init__(self, name: str, description: str, duration: float = 1.0):
        super().__init__(name, description, duration)
    
    def _safe_reset(self, car: Any) -> None:
        """Safely reset the car to neutral position."""
        try:
            car.reset()
        except Exception as e:
            print(f"Reset failed: {e}")


class SoundAction(RoboticAction):
    """Base class for sound-producing actions."""
    
    def __init__(self, name: str, description: str, duration: float = 0.5):
        super().__init__(name, description, duration)


# Physical Movement Actions
class WaveHands(PhysicalAction):
    """Wave hands action - steering wheel movement simulation."""
    
    def __init__(self):
        super().__init__(
            name="wave hands",
            description="Steering wheel movement to simulate waving",
            duration=1.5
        )
    
    def _perform_action(self, car: Any) -> None:
        """Perform waving motion."""
        self._safe_reset(car)
        car.set_cam_tilt_angle(20)
        
        for _ in range(2):
            car.set_dir_servo_angle(-25)
            sleep(0.1)
            car.set_dir_servo_angle(25)
            sleep(0.1)
        
        car.set_dir_servo_angle(0)


class Resist(PhysicalAction):
    """Resist action - defensive movement pattern."""
    
    def __init__(self):
        super().__init__(
            name="resist",
            description="Defensive movement pattern",
            duration=1.0
        )
    
    def _perform_action(self, car: Any) -> None:
        """Perform resistance motion."""
        self._safe_reset(car)
        car.set_cam_tilt_angle(10)
        
        for _ in range(3):
            car.set_dir_servo_angle(-15)
            car.set_cam_pan_angle(15)
            sleep(0.1)
            car.set_dir_servo_angle(15)
            car.set_cam_pan_angle(-15)
            sleep(0.1)
        
        car.stop()
        car.set_dir_servo_angle(0)
        car.set_cam_pan_angle(0)


class ActCute(PhysicalAction):
    """Act cute action - gentle forward-backward motion."""
    
    def __init__(self):
        super().__init__(
            name="act cute",
            description="Gentle forward-backward motion",
            duration=1.2
        )
    
    def _perform_action(self, car: Any) -> None:
        """Perform cute motion."""
        self._safe_reset(car)
        car.set_cam_tilt_angle(-20)
        
        for i in range(15):
            car.forward(5)
            sleep(0.02)
            car.backward(5)
            sleep(0.02)
        
        car.set_cam_tilt_angle(0)
        car.stop()


class RubHands(PhysicalAction):
    """Rub hands action - subtle steering adjustments."""
    
    def __init__(self):
        super().__init__(
            name="rub hands",
            description="Subtle steering adjustments",
            duration=2.5
        )
    
    def _perform_action(self, car: Any) -> None:
        """Perform hand rubbing motion."""
        self._safe_reset(car)
        
        for i in range(5):
            car.set_dir_servo_angle(-6)
            sleep(0.5)
            car.set_dir_servo_angle(6)
            sleep(0.5)
        
        car.reset()


class Think(PhysicalAction):
    """Think action - contemplative head and body movement."""
    
    def __init__(self):
        super().__init__(
            name="think",
            description="Contemplative head and body movement",
            duration=2.0
        )
    
    def _perform_action(self, car: Any) -> None:
        """Perform thinking motion."""
        self._safe_reset(car)
        
        # Gradual movement sequence
        for i in range(11):
            car.set_cam_pan_angle(i * 3)
            car.set_cam_tilt_angle(-i * 2)
            car.set_dir_servo_angle(i * 2)
            sleep(0.05)
        
        sleep(1)
        car.set_cam_pan_angle(15)
        car.set_cam_tilt_angle(-10)
        car.set_dir_servo_angle(10)
        sleep(0.1)
        car.reset()


class KeepThink(PhysicalAction):
    """Keep thinking action - continuous thinking motion."""
    
    def __init__(self):
        super().__init__(
            name="keep think",
            description="Continuous thinking motion",
            duration=1.0
        )
    
    def _perform_action(self, car: Any) -> None:
        """Perform continuous thinking motion."""
        self._safe_reset(car)
        
        for i in range(11):
            car.set_cam_pan_angle(i * 3)
            car.set_cam_tilt_angle(-i * 2)
            car.set_dir_servo_angle(i * 2)
            sleep(0.05)


class ShakeHead(PhysicalAction):
    """Shake head action - side-to-side head movement."""
    
    def __init__(self):
        super().__init__(
            name="shake head",
            description="Side-to-side head movement",
            duration=1.5
        )
    
    def _perform_action(self, car: Any) -> None:
        """Perform head shaking motion."""
        car.stop()
        car.set_cam_pan_angle(0)
        
        # Gradual shaking motion
        angles = [60, -50, 40, -30, 20, -10, 10, -5, 0]
        for angle in angles:
            car.set_cam_pan_angle(angle)
            sleep(0.1)


class Nod(PhysicalAction):
    """Nod action - up-and-down nodding motion."""
    
    def __init__(self):
        super().__init__(
            name="nod",
            description="Up-and-down nodding motion",
            duration=1.0
        )
    
    def _perform_action(self, car: Any) -> None:
        """Perform nodding motion."""
        self._safe_reset(car)
        car.set_cam_tilt_angle(0)
        
        # Nodding sequence
        angles = [5, -30, 5, -30, 0]
        for angle in angles:
            car.set_cam_tilt_angle(angle)
            sleep(0.1)


class Depressed(PhysicalAction):
    """Depressed action - sad, drooping movement."""
    
    def __init__(self):
        super().__init__(
            name="depressed",
            description="Sad, drooping movement",
            duration=3.0
        )
    
    def _perform_action(self, car: Any) -> None:
        """Perform depressed motion."""
        self._safe_reset(car)
        car.set_cam_tilt_angle(0)
        
        # Gradual drooping motion
        angles = [20, -22, 10, -22, 0, -22, -10, -22, -15, -22, -19, -22]
        for angle in angles:
            car.set_cam_tilt_angle(angle)
            sleep(0.1)
        
        sleep(1.5)
        car.reset()


class TwistBody(PhysicalAction):
    """Twist body action - rotational body movement."""
    
    def __init__(self):
        super().__init__(
            name="twist body",
            description="Rotational body movement",
            duration=1.5
        )
    
    def _perform_action(self, car: Any) -> None:
        """Perform body twisting motion."""
        self._safe_reset(car)
        
        for i in range(3):
            # Forward twist
            car.set_motor_speed(1, 20)
            car.set_motor_speed(2, 20)
            car.set_cam_pan_angle(-20)
            car.set_dir_servo_angle(-10)
            sleep(0.1)
            
            # Stop
            car.set_motor_speed(1, 0)
            car.set_motor_speed(2, 0)
            car.set_cam_pan_angle(0)
            car.set_dir_servo_angle(0)
            sleep(0.1)
            
            # Backward twist
            car.set_motor_speed(1, -20)
            car.set_motor_speed(2, -20)
            car.set_cam_pan_angle(20)
            car.set_dir_servo_angle(10)
            sleep(0.1)
            
            # Stop
            car.set_motor_speed(1, 0)
            car.set_motor_speed(2, 0)
            car.set_cam_pan_angle(0)
            car.set_dir_servo_angle(0)
            sleep(0.1)


class Celebrate(PhysicalAction):
    """Celebrate action - joyful celebration dance."""
    
    def __init__(self):
        super().__init__(
            name="celebrate",
            description="Joyful celebration dance",
            duration=2.5
        )
    
    def _perform_action(self, car: Any) -> None:
        """Perform celebration motion."""
        self._safe_reset(car)
        car.set_cam_tilt_angle(20)
        
        # Right side celebration
        car.set_dir_servo_angle(30)
        car.set_cam_pan_angle(60)
        sleep(0.3)
        car.set_dir_servo_angle(10)
        car.set_cam_pan_angle(30)
        sleep(0.1)
        car.set_dir_servo_angle(30)
        car.set_cam_pan_angle(60)
        sleep(0.3)
        car.set_dir_servo_angle(0)
        car.set_cam_pan_angle(0)
        sleep(0.2)
        
        # Left side celebration
        car.set_dir_servo_angle(-30)
        car.set_cam_pan_angle(-60)
        sleep(0.3)
        car.set_dir_servo_angle(-10)
        car.set_cam_pan_angle(-30)
        sleep(0.1)
        car.set_dir_servo_angle(-30)
        car.set_cam_pan_angle(-60)
        sleep(0.3)
        car.set_dir_servo_angle(0)
        car.set_cam_pan_angle(0)
        sleep(0.2)


# Sound Actions
class Honking(SoundAction):
    """Honking action - car horn sound."""
    
    def __init__(self):
        super().__init__(
            name="honking",
            description="Car horn sound effect",
            duration=0.5
        )
    
    def _perform_action(self, music: Any) -> None:
        """Play honking sound."""
        try:
            music.sound_play_threading("../sounds/car-double-horn.wav", 100)
        except Exception as e:
            print(f"Honking sound failed: {e}")


class StartEngine(SoundAction):
    """Start engine action - engine startup sound."""
    
    def __init__(self):
        super().__init__(
            name="start engine",
            description="Engine startup sound effect",
            duration=1.0
        )
    
    def _perform_action(self, music: Any) -> None:
        """Play engine startup sound."""
        try:
            music.sound_play_threading("../sounds/car-start-engine.wav", 50)
        except Exception as e:
            print(f"Engine sound failed: {e}")


# Action Registry
class ActionRegistry:
    """
    Registry for all available robotic actions.
    
    Provides a centralized way to manage and access all actions
    with proper categorization and error handling.
    """
    
    def __init__(self):
        """Initialize the action registry."""
        self._physical_actions = {}
        self._sound_actions = {}
        self._register_actions()
    
    def _register_actions(self) -> None:
        """Register all available actions."""
        # Physical actions
        physical_action_classes = [
            WaveHands, Resist, ActCute, RubHands, Think, KeepThink,
            ShakeHead, Nod, Depressed, TwistBody, Celebrate
        ]
        
        for action_class in physical_action_classes:
            action = action_class()
            self._physical_actions[action.name] = action
        
        # Sound actions
        sound_action_classes = [Honking, StartEngine]
        
        for action_class in sound_action_classes:
            action = action_class()
            self._sound_actions[action.name] = action
    
    def get_physical_action(self, name: str) -> PhysicalAction:
        """
        Get a physical action by name.
        
        Args:
            name: Action name
            
        Returns:
            PhysicalAction instance
            
        Raises:
            KeyError: If action not found
        """
        if name not in self._physical_actions:
            raise KeyError(f"Physical action '{name}' not found")
        return self._physical_actions[name]
    
    def get_sound_action(self, name: str) -> SoundAction:
        """
        Get a sound action by name.
        
        Args:
            name: Action name
            
        Returns:
            SoundAction instance
            
        Raises:
            KeyError: If action not found
        """
        if name not in self._sound_actions:
            raise KeyError(f"Sound action '{name}' not found")
        return self._sound_actions[name]
    
    def get_all_physical_actions(self) -> List[str]:
        """Get list of all physical action names."""
        return list(self._physical_actions.keys())
    
    def get_all_sound_actions(self) -> List[str]:
        """Get list of all sound action names."""
        return list(self._sound_actions.keys())
    
    def get_all_actions(self) -> List[str]:
        """Get list of all action names."""
        return self.get_all_physical_actions() + self.get_all_sound_actions()
    
    def execute_physical_action(self, name: str, car: Any) -> bool:
        """
        Execute a physical action by name.
        
        Args:
            name: Action name
            car: Picar-X car instance
            
        Returns:
            True if successful, False otherwise
        """
        try:
            action = self.get_physical_action(name)
            return action.execute(car)
        except KeyError as e:
            print(f"Action not found: {e}")
            return False
        except Exception as e:
            print(f"Action execution failed: {e}")
            return False
    
    def execute_sound_action(self, name: str, music: Any) -> bool:
        """
        Execute a sound action by name.
        
        Args:
            name: Action name
            music: Music controller instance
            
        Returns:
            True if successful, False otherwise
        """
        try:
            action = self.get_sound_action(name)
            return action.execute(music)
        except KeyError as e:
            print(f"Action not found: {e}")
            return False
        except Exception as e:
            print(f"Action execution failed: {e}")
            return False


# Global action registry instance
action_registry = ActionRegistry()

# Legacy function mappings for backward compatibility
def wave_hands(car: Any) -> None:
    """Legacy function for wave hands action."""
    action_registry.execute_physical_action("wave hands", car)


def resist(car: Any) -> None:
    """Legacy function for resist action."""
    action_registry.execute_physical_action("resist", car)


def act_cute(car: Any) -> None:
    """Legacy function for act cute action."""
    action_registry.execute_physical_action("act cute", car)


def rub_hands(car: Any) -> None:
    """Legacy function for rub hands action."""
    action_registry.execute_physical_action("rub hands", car)


def think(car: Any) -> None:
    """Legacy function for think action."""
    action_registry.execute_physical_action("think", car)


def keep_think(car: Any) -> None:
    """Legacy function for keep think action."""
    action_registry.execute_physical_action("keep think", car)


def shake_head(car: Any) -> None:
    """Legacy function for shake head action."""
    action_registry.execute_physical_action("shake head", car)


def nod(car: Any) -> None:
    """Legacy function for nod action."""
    action_registry.execute_physical_action("nod", car)


def depressed(car: Any) -> None:
    """Legacy function for depressed action."""
    action_registry.execute_physical_action("depressed", car)


def twist_body(car: Any) -> None:
    """Legacy function for twist body action."""
    action_registry.execute_physical_action("twist body", car)


def celebrate(car: Any) -> None:
    """Legacy function for celebrate action."""
    action_registry.execute_physical_action("celebrate", car)


def honking(music: Any) -> None:
    """Legacy function for honking action."""
    action_registry.execute_sound_action("honking", music)


def start_engine(music: Any) -> None:
    """Legacy function for start engine action."""
    action_registry.execute_sound_action("start engine", music)


# Legacy dictionary mappings for backward compatibility
actions_dict = {
    "shake head": shake_head,
    "nod": nod,
    "wave hands": wave_hands,
    "resist": resist,
    "act cute": act_cute,
    "rub hands": rub_hands,
    "think": think,
    "twist body": twist_body,
    "celebrate": celebrate,
    "depressed": depressed,
}

sounds_dict = {
    "honking": honking,
    "start engine": start_engine,
}


# Interactive testing function
def test_actions():
    """Interactive testing function for all actions."""
    try:
        from picarx import Picarx
        from robot_hat import Music
        import os
        
        # Enable speaker
        os.popen("pinctrl set 20 op dh")
        
        # Initialize hardware
        car = Picarx()
        car.reset()
        music = Music()
        sleep(0.5)
        
        # Get all actions
        physical_actions = action_registry.get_all_physical_actions()
        sound_actions = action_registry.get_all_sound_actions()
        
        print("Available Physical Actions:")
        for i, action in enumerate(physical_actions):
            print(f"  {i}: {action}")
        
        print("\nAvailable Sound Actions:")
        for i, action in enumerate(sound_actions):
            print(f"  {len(physical_actions) + i}: {action}")
        
        print("\nEnter action number to test (or press Enter to repeat last action):")
        
        last_action = None
        
        while True:
            try:
                user_input = input("Action: ").strip()
                
                if user_input == "":
                    if last_action is not None:
                        if last_action < len(physical_actions):
                            action_name = physical_actions[last_action]
                            print(f"Executing: {action_name}")
                            action_registry.execute_physical_action(action_name, car)
                        else:
                            action_name = sound_actions[last_action - len(physical_actions)]
                            print(f"Executing: {action_name}")
                            action_registry.execute_sound_action(action_name, music)
                else:
                    action_num = int(user_input)
                    if 0 <= action_num < len(physical_actions):
                        action_name = physical_actions[action_num]
                        print(f"Executing: {action_name}")
                        action_registry.execute_physical_action(action_name, car)
                        last_action = action_num
                    elif action_num < len(physical_actions) + len(sound_actions):
                        action_name = sound_actions[action_num - len(physical_actions)]
                        print(f"Executing: {action_name}")
                        action_registry.execute_sound_action(action_name, music)
                        last_action = action_num
                    else:
                        print("Invalid action number")
                        
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
        
    except ImportError as e:
        print(f"Hardware not available: {e}")
    except Exception as e:
        print(f"Testing failed: {e}")
    finally:
        try:
            car.reset()
            sleep(0.1)
        except:
            pass


if __name__ == "__main__":
    test_actions()




