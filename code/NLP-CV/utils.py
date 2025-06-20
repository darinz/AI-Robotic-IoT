#!/usr/bin/env python3
"""
Utility Functions for AI Robotic Car

This module provides utility functions for audio processing, system operations,
and terminal output formatting used throughout the AI robotic car application.

"""

import os
import sys
import subprocess
from typing import Tuple, Optional, Union


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output formatting."""
    GRAY = '1;30'
    RED = '0;31'
    GREEN = '0;32'
    YELLOW = '0;33'
    BLUE = '0;34'
    PURPLE = '0;35'
    DARK_GREEN = '0;36'
    WHITE = '0;37'
    BOLD = '1'
    RESET = '0'


def print_color(
    message: str, 
    color: str = '', 
    end: str = '\n', 
    file=sys.stdout, 
    flush: bool = False
) -> None:
    """
    Print a colored message to the terminal.
    
    Args:
        message: Text to print
        color: ANSI color code
        end: String appended after the message
        file: Output file object
        flush: Whether to flush the output buffer
    """
    if color:
        print(f'\033[{color}m{message}\033[0m', end=end, file=file, flush=flush)
    else:
        print(message, end=end, file=file, flush=flush)


def gray_print(
    message: str, 
    end: str = '\n', 
    file=sys.stdout, 
    flush: bool = False
) -> None:
    """Print a gray-colored message."""
    print_color(message, color=Colors.GRAY, end=end, file=file, flush=flush)


def warn(
    message: str, 
    end: str = '\n', 
    file=sys.stdout, 
    flush: bool = False
) -> None:
    """Print a yellow warning message."""
    print_color(message, color=Colors.YELLOW, end=end, file=file, flush=flush)


def error(
    message: str, 
    end: str = '\n', 
    file=sys.stdout, 
    flush: bool = False
) -> None:
    """Print a red error message."""
    print_color(message, color=Colors.RED, end=end, file=file, flush=flush)


def success(
    message: str, 
    end: str = '\n', 
    file=sys.stdout, 
    flush: bool = False
) -> None:
    """Print a green success message."""
    print_color(message, color=Colors.GREEN, end=end, file=file, flush=flush)


def info(
    message: str, 
    end: str = '\n', 
    file=sys.stdout, 
    flush: bool = False
) -> None:
    """Print a blue info message."""
    print_color(message, color=Colors.BLUE, end=end, file=file, flush=flush)


class ErrorRedirector:
    """
    Context manager for temporarily redirecting stderr to suppress unwanted output.
    
    Useful for suppressing ALSA errors and other system messages during audio operations.
    """
    
    def __init__(self):
        self.devnull = None
        self.old_stderr = None
    
    def __enter__(self):
        """Redirect stderr to /dev/null."""
        try:
            self.devnull = os.open(os.devnull, os.O_WRONLY)
            self.old_stderr = os.dup(2)
            sys.stderr.flush()
            os.dup2(self.devnull, 2)
        except Exception as e:
            warn(f"Failed to redirect stderr: {e}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore stderr to original state."""
        try:
            if self.old_stderr is not None:
                os.dup2(self.old_stderr, 2)
                os.close(self.old_stderr)
            if self.devnull is not None:
                os.close(self.devnull)
        except Exception as e:
            warn(f"Failed to restore stderr: {e}")


def redirect_error_2_null() -> int:
    """
    Legacy function: Redirect stderr to /dev/null.
    
    Returns:
        File descriptor for the original stderr
        
    Note:
        This function is maintained for backward compatibility.
        Consider using ErrorRedirector context manager for new code.
    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        sys.stderr.flush()
        os.dup2(devnull, 2)
        os.close(devnull)
        return old_stderr
    except Exception as e:
        warn(f"Failed to redirect stderr: {e}")
        return 2


def cancel_redirect_error(old_stderr: int) -> None:
    """
    Legacy function: Restore stderr from backup.
    
    Args:
        old_stderr: File descriptor for the original stderr
        
    Note:
        This function is maintained for backward compatibility.
        Consider using ErrorRedirector context manager for new code.
    """
    try:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
    except Exception as e:
        warn(f"Failed to restore stderr: {e}")


def run_command(command: str, timeout: Optional[int] = None) -> Tuple[int, str]:
    """
    Execute a shell command and return the result.
    
    Args:
        command: Shell command to execute
        timeout: Command timeout in seconds (None for no timeout)
        
    Returns:
        Tuple of (exit_code, output)
        
    Raises:
        subprocess.TimeoutExpired: If command times out
        subprocess.SubprocessError: For other subprocess errors
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout
    except subprocess.TimeoutExpired:
        raise
    except subprocess.SubprocessError as e:
        raise subprocess.SubprocessError(f"Command execution failed: {e}")


def run_command_safe(command: str, timeout: Optional[int] = None) -> Tuple[int, str]:
    """
    Execute a shell command safely, catching all exceptions.
    
    Args:
        command: Shell command to execute
        timeout: Command timeout in seconds (None for no timeout)
        
    Returns:
        Tuple of (exit_code, output). Returns (-1, error_message) on failure.
    """
    try:
        return run_command(command, timeout)
    except subprocess.TimeoutExpired:
        return -1, f"Command timed out after {timeout} seconds"
    except Exception as e:
        return -1, f"Command execution error: {e}"


class AudioProcessor:
    """
    Audio processing utilities using SoX (Sound eXchange).
    
    Provides functions for audio manipulation like volume adjustment,
    format conversion, and audio effects.
    """
    
    @staticmethod
    def adjust_volume(
        input_file: str, 
        output_file: str, 
        volume_db: float
    ) -> bool:
        """
        Adjust audio volume using SoX.
        
        Args:
            input_file: Path to input audio file
            output_file: Path for output audio file
            volume_db: Volume adjustment in decibels
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import sox
            
            # Validate input file
            if not os.path.isfile(input_file):
                error(f"Input file not found: {input_file}")
                return False
            
            # Create SoX transformer
            transformer = sox.Transformer()
            transformer.vol(volume_db)
            
            # Process audio
            transformer.build(input_file, output_file)
            
            return True
            
        except ImportError:
            error("SoX library not available. Install with: pip install sox")
            return False
        except Exception as e:
            error(f"Volume adjustment error: {e}")
            return False
    
    @staticmethod
    def get_audio_info(file_path: str) -> Optional[dict]:
        """
        Get audio file information using SoX.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio information or None if failed
        """
        try:
            import sox
            
            if not os.path.isfile(file_path):
                return None
            
            # Get file info
            info = sox.file_info.info(file_path)
            return {
                'duration': info.get('duration'),
                'sample_rate': info.get('sample_rate'),
                'channels': info.get('channels'),
                'bit_depth': info.get('bit_depth'),
                'file_size': os.path.getsize(file_path)
            }
            
        except ImportError:
            warn("SoX library not available for audio info")
            return None
        except Exception as e:
            warn(f"Failed to get audio info: {e}")
            return None


# Legacy function aliases for backward compatibility
def sox_volume(input_file: str, output_file: str, volume: float) -> bool:
    """
    Legacy function: Adjust audio volume using SoX.
    
    Args:
        input_file: Path to input audio file
        output_file: Path for output audio file
        volume: Volume adjustment in decibels
        
    Returns:
        True if successful, False otherwise
    """
    return AudioProcessor.adjust_volume(input_file, output_file, volume)


class AudioPlayer:
    """
    Audio playback utilities for the robotic car.
    
    Handles audio file playback with proper error handling and
    system integration.
    """
    
    def __init__(self, music_controller):
        """
        Initialize audio player.
        
        Args:
            music_controller: Robot HAT Music controller instance
        """
        self.music = music_controller
        self.speak_first = False
    
    def play_audio_blocking(
        self, 
        file_path: str, 
        volume: int = 100
    ) -> bool:
        """
        Play audio file with blocking behavior.
        
        Args:
            file_path: Path to audio file
            volume: Volume level (0-100)
            
        Returns:
            True if successful, False otherwise
        """
        # Check for root privileges
        is_root = (os.geteuid() == 0)
        if not is_root and not self.speak_first:
            self.speak_first = True
            warn("Audio playback requires root privileges (sudo)")
        
        try:
            # Kill pulseaudio to avoid conflicts in VNC environments
            status, _ = run_command_safe('sudo killall pulseaudio')
            if status != 0:
                warn("Failed to kill pulseaudio (this is usually OK)")
            
            # Validate file
            if not os.path.isfile(file_path):
                warn(f"Audio file not found: {file_path}")
                return False
            
            # Play audio
            self.music.sound_play(file_path, volume)
            return True
            
        except Exception as e:
            error(f"Audio playback error: {e}")
            return False
    
    def play_audio_non_blocking(
        self, 
        file_path: str, 
        volume: int = 100
    ) -> bool:
        """
        Play audio file without blocking.
        
        Args:
            file_path: Path to audio file
            volume: Volume level (0-100)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.isfile(file_path):
                warn(f"Audio file not found: {file_path}")
                return False
            
            # Play audio in background
            self.music.sound_play_threading(file_path, volume)
            return True
            
        except Exception as e:
            error(f"Non-blocking audio playback error: {e}")
            return False


# Legacy function for backward compatibility
def speak_block(music_controller, file_path: str, volume: int = 100) -> bool:
    """
    Legacy function: Play audio file with blocking behavior.
    
    Args:
        music_controller: Robot HAT Music controller instance
        file_path: Path to audio file
        volume: Volume level (0-100)
        
    Returns:
        True if successful, False otherwise
    """
    player = AudioPlayer(music_controller)
    return player.play_audio_blocking(file_path, volume)


class SystemUtils:
    """System utility functions for the robotic car."""
    
    @staticmethod
    def is_root() -> bool:
        """Check if the current process is running as root."""
        return os.geteuid() == 0
    
    @staticmethod
    def enable_speaker() -> bool:
        """
        Enable the robot speaker switch.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try different methods to enable speaker
            commands = [
                "pinctrl set 20 op dh",
                "robot-hat enable_speaker"
            ]
            
            for cmd in commands:
                status, _ = run_command_safe(cmd)
                if status == 0:
                    return True
            
            warn("Failed to enable speaker with any method")
            return False
            
        except Exception as e:
            error(f"Speaker enable error: {e}")
            return False
    
    @staticmethod
    def get_system_info() -> dict:
        """
        Get basic system information.
        
        Returns:
            Dictionary with system information
        """
        try:
            info = {
                'platform': sys.platform,
                'python_version': sys.version,
                'is_root': SystemUtils.is_root(),
                'current_dir': os.getcwd(),
                'user': os.getenv('USER', 'unknown')
            }
            
            # Try to get more system info on Linux
            if sys.platform.startswith('linux'):
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        cpu_info = f.read()
                        if 'Raspberry Pi' in cpu_info:
                            info['hardware'] = 'Raspberry Pi'
                        else:
                            info['hardware'] = 'Unknown Linux'
                except:
                    info['hardware'] = 'Unknown'
            
            return info
            
        except Exception as e:
            return {'error': str(e)}


# Convenience functions
def check_dependencies() -> dict:
    """
    Check if all required dependencies are available.
    
    Returns:
        Dictionary with dependency status
    """
    dependencies = {
        'openai': False,
        'speech_recognition': False,
        'sox': False,
        'picarx': False,
        'robot_hat': False,
        'vilib': False,
        'cv2': False
    }
    
    try:
        import openai
        dependencies['openai'] = True
    except ImportError:
        pass
    
    try:
        import speech_recognition
        dependencies['speech_recognition'] = True
    except ImportError:
        pass
    
    try:
        import sox
        dependencies['sox'] = True
    except ImportError:
        pass
    
    try:
        import picarx
        dependencies['picarx'] = True
    except ImportError:
        pass
    
    try:
        import robot_hat
        dependencies['robot_hat'] = True
    except ImportError:
        pass
    
    try:
        import vilib
        dependencies['vilib'] = True
    except ImportError:
        pass
    
    try:
        import cv2
        dependencies['cv2'] = True
    except ImportError:
        pass
    
    return dependencies


def print_dependency_status() -> None:
    """Print the status of all dependencies."""
    deps = check_dependencies()
    
    info("Dependency Status:")
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        color = Colors.GREEN if available else Colors.RED
        print_color(f"  {status} {dep}", color=color)
