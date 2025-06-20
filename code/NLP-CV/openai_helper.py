#!/usr/bin/env python3
"""
OpenAI API Integration Helper

This module provides a comprehensive interface for OpenAI's API services,
including GPT-4o conversations, Whisper speech-to-text, and text-to-speech
capabilities with support for multimodal interactions.

"""

import os
import shutil
import time
from typing import Dict, List, Optional, Union, Any
from io import BytesIO

from openai import OpenAI


class ChatLogger:
    """
    Utility class for formatting and displaying chat messages with timestamps.
    
    Provides both simple and table-formatted output options for better readability.
    """
    
    def __init__(self, enable_table_format: bool = False):
        """
        Initialize the chat logger.
        
        Args:
            enable_table_format: Whether to use table formatting for long messages
        """
        self.enable_table_format = enable_table_format
    
    def log_message(self, label: str, message: str) -> None:
        """
        Log a chat message with timestamp and formatting.
        
        Args:
            label: Sender label (e.g., 'user', 'assistant')
            message: Message content
        """
        if not self.enable_table_format:
            self._simple_log(label, message)
        else:
            self._table_log(label, message)
    
    def _simple_log(self, label: str, message: str) -> None:
        """Simple timestamped log format."""
        timestamp = time.time()
        print(f'{timestamp:.3f} {label:>6} >>> {message}')
    
    def _table_log(self, label: str, message: str) -> None:
        """Table-formatted log for long messages."""
        width = shutil.get_terminal_size().columns
        msg_len = len(message)
        line_len = width - 27
        
        if width < 38 or msg_len <= line_len:
            self._simple_log(label, message)
            return
        
        # Split message into lines
        lines = []
        for i in range(0, len(message), line_len):
            lines.append(message[i:i+line_len])
        
        # Print with proper indentation
        for i, line in enumerate(lines):
            if i == 0:
                self._simple_log(label, line)
            else:
                print(f'{"":>26} {line}')


class OpenAiHelper:
    """
    Comprehensive OpenAI API integration helper.
    
    Provides interfaces for:
    - GPT-4o conversations (text and multimodal)
    - Whisper speech-to-text
    - OpenAI text-to-speech
    - Assistant API integration
    """
    
    # Default configuration
    STT_OUTPUT_FILE = "stt_output.wav"
    TTS_OUTPUT_FILE = 'tts_output.mp3'
    DEFAULT_TIMEOUT = 30  # seconds
    SUPPORTED_TTS_VOICES = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
    
    def __init__(
        self, 
        api_key: str, 
        assistant_id: str, 
        assistant_name: str, 
        timeout: int = DEFAULT_TIMEOUT
    ) -> None:
        """
        Initialize the OpenAI helper.
        
        Args:
            api_key: OpenAI API key
            assistant_id: OpenAI Assistant ID
            assistant_name: Name for the assistant in logs
            timeout: API request timeout in seconds
        """
        self.api_key = api_key
        self.assistant_id = assistant_id
        self.assistant_name = assistant_name
        self.timeout = timeout
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        
        # Initialize chat logger
        self.logger = ChatLogger(enable_table_format=False)
        
        # Initialize assistant thread
        self._init_assistant_thread()
    
    def _init_assistant_thread(self) -> None:
        """Initialize the assistant thread for conversations."""
        try:
            self.thread = self.client.beta.threads.create()
            self.run = self.client.beta.threads.runs.create_and_poll(
                thread_id=self.thread.id,
                assistant_id=self.assistant_id,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize assistant thread: {e}")
    
    def speech_to_text(self, audio, language: Optional[Union[str, List[str]]] = None) -> Optional[str]:
        """
        Convert speech audio to text using OpenAI's Whisper API.
        
        Args:
            audio: Audio data from speech recognition
            language: Language code(s) for improved accuracy
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            # Prepare audio data
            wav_data = BytesIO(audio.get_wav_data())
            wav_data.name = self.STT_OUTPUT_FILE
            
            # Prepare language parameter
            lang_param = None
            if language:
                if isinstance(language, list) and len(language) > 0:
                    lang_param = language[0]  # Whisper API only supports single language
                elif isinstance(language, str):
                    lang_param = language
            
            # Create transcription
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=wav_data,
                language=lang_param,
                prompt="This is a conversation between a human and an AI-powered robotic car."
            )
            
            return transcript.text
            
        except Exception as e:
            self.logger.log_message("ERROR", f"Speech-to-text error: {e}")
            return None
    
    def speech_to_text_legacy(self, recognizer, audio) -> Optional[str]:
        """
        Legacy speech-to-text using speech_recognition library.
        
        Args:
            recognizer: Speech recognition recognizer instance
            audio: Audio data
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            return recognizer.recognize_whisper_api(audio, api_key=self.api_key)
        except Exception as e:
            self.logger.log_message("ERROR", f"Legacy STT error: {e}")
            return None
    
    def dialogue(self, message: str) -> Union[Dict[str, Any], str]:
        """
        Conduct a text-only dialogue with the OpenAI assistant.
        
        Args:
            message: User's text message
            
        Returns:
            Assistant's response (dict for structured responses, str for plain text)
        """
        self.logger.log_message("user", message)
        
        try:
            # Create user message
            user_message = self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=message
            )
            
            # Run assistant
            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=self.thread.id,
                assistant_id=self.assistant_id,
            )
            
            if run.status == 'completed':
                return self._extract_assistant_response()
            else:
                self.logger.log_message("ERROR", f"Assistant run failed: {run.status}")
                return "I'm sorry, I encountered an error processing your request."
                
        except Exception as e:
            self.logger.log_message("ERROR", f"Dialogue error: {e}")
            return "I'm sorry, I encountered an error processing your request."
    
    def dialogue_with_image(
        self, 
        message: str, 
        image_path: str
    ) -> Union[Dict[str, Any], str]:
        """
        Conduct a multimodal dialogue with image analysis.
        
        Args:
            message: User's text message
            image_path: Path to the image file
            
        Returns:
            Assistant's response (dict for structured responses, str for plain text)
        """
        self.logger.log_message("user", f"{message} [with image: {image_path}]")
        
        try:
            # Upload image file
            image_file = self.client.files.create(
                file=open(image_path, "rb"),
                purpose="vision"
            )
            
            # Create multimodal message
            user_message = self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=[
                    {
                        "type": "text",
                        "text": message
                    },
                    {
                        "type": "image_file",
                        "image_file": {"file_id": image_file.id}
                    }
                ],
            )
            
            # Run assistant
            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=self.thread.id,
                assistant_id=self.assistant_id,
            )
            
            if run.status == 'completed':
                return self._extract_assistant_response()
            else:
                self.logger.log_message("ERROR", f"Assistant run failed: {run.status}")
                return "I'm sorry, I encountered an error processing your request."
                
        except Exception as e:
            self.logger.log_message("ERROR", f"Multimodal dialogue error: {e}")
            return "I'm sorry, I encountered an error processing your request."
    
    def _extract_assistant_response(self) -> Union[Dict[str, Any], str]:
        """
        Extract and parse the assistant's response from the thread.
        
        Returns:
            Parsed response (dict for structured responses, str for plain text)
        """
        try:
            messages = self.client.beta.threads.messages.list(
                thread_id=self.thread.id
            )
            
            for message in messages.data:
                if message.role == 'assistant':
                    for block in message.content:
                        if block.type == 'text':
                            value = block.text.value
                            self.logger.log_message(self.assistant_name, value)
                            
                            # Try to parse as structured response
                            try:
                                parsed_value = eval(value)  # Convert string dict to dict
                                if isinstance(parsed_value, dict):
                                    return parsed_value
                                else:
                                    return str(parsed_value)
                            except (SyntaxError, NameError, TypeError):
                                # Return as plain text if not a valid dict
                                return str(value)
                    break  # Only process the latest assistant message
            
            return "I'm sorry, I couldn't generate a proper response."
            
        except Exception as e:
            self.logger.log_message("ERROR", f"Response extraction error: {e}")
            return "I'm sorry, I encountered an error processing the response."
    
    def text_to_speech(
        self, 
        text: str, 
        output_file: str, 
        voice: str = 'alloy', 
        response_format: str = "mp3", 
        speed: float = 1.0
    ) -> bool:
        """
        Convert text to speech using OpenAI's TTS API.
        
        Args:
            text: Text to convert to speech
            output_file: Path for the output audio file
            voice: TTS voice to use (alloy, echo, fable, onyx, nova, shimmer)
            response_format: Audio format (mp3, wav, etc.)
            speed: Speech speed (0.25 to 4.0)
            
        Returns:
            True if successful, False otherwise
        """
        # Validate parameters
        if voice not in self.SUPPORTED_TTS_VOICES:
            self.logger.log_message("ERROR", f"Unsupported voice: {voice}")
            return False
        
        if not (0.25 <= speed <= 4.0):
            self.logger.log_message("ERROR", f"Speed must be between 0.25 and 4.0")
            return False
        
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            elif output_dir and not os.path.isdir(output_dir):
                raise FileExistsError(f"'{output_dir}' is not a directory")
            
            # Generate speech
            with self.client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice=voice,
                input=text,
                response_format=response_format,
                speed=speed,
            ) as response:
                response.stream_to_file(output_file)
            
            return True
            
        except Exception as e:
            self.logger.log_message("ERROR", f"Text-to-speech error: {e}")
            return False
    
    def get_available_voices(self) -> List[str]:
        """
        Get list of available TTS voices.
        
        Returns:
            List of supported voice names
        """
        return self.SUPPORTED_TTS_VOICES.copy()
    
    def validate_configuration(self) -> bool:
        """
        Validate the OpenAI configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Test API key by making a simple request
            models = self.client.models.list()
            return True
        except Exception as e:
            self.logger.log_message("ERROR", f"Configuration validation failed: {e}")
            return False
    
    def get_usage_info(self) -> Dict[str, Any]:
        """
        Get API usage information (if available).
        
        Returns:
            Usage information dictionary
        """
        try:
            # Note: OpenAI doesn't provide real-time usage in the standard API
            # This is a placeholder for future implementation
            return {
                "status": "Usage information not available in standard API",
                "timestamp": time.time()
            }
        except Exception as e:
            return {"error": str(e)}


# Backward compatibility aliases
def chat_print(label: str, message: str) -> None:
    """
    Legacy function for backward compatibility.
    
    Args:
        label: Message label
        message: Message content
    """
    logger = ChatLogger()
    logger.log_message(label, message)


# Convenience functions for common operations
def create_openai_helper(
    api_key: str, 
    assistant_id: str, 
    assistant_name: str = "assistant"
) -> OpenAiHelper:
    """
    Create a new OpenAI helper instance.
    
    Args:
        api_key: OpenAI API key
        assistant_id: OpenAI Assistant ID
        assistant_name: Assistant name for logging
        
    Returns:
        Configured OpenAiHelper instance
    """
    return OpenAiHelper(api_key, assistant_id, assistant_name)


def test_openai_connection(api_key: str) -> bool:
    """
    Test OpenAI API connection.
    
    Args:
        api_key: OpenAI API key to test
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        return True
    except Exception:
        return False
