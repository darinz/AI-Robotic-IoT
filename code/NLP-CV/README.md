# NLP-CV Module – Voice Interaction and Visual Perception

This module provides natural language and computer vision capabilities for KITT. It enables the robot to understand voice commands and perceive its environment in real time.

## NLP (Voice Command Processing)

- Voice recognition using speech-to-text
- Intent classification (e.g., move forward, stop)
- Text-to-speech (TTS) responses

## CV (Computer Vision)

- Lane detection using OpenCV
- Object detection (YOLOv5/TensorFlow Lite)
- Traffic sign recognition and obstacle awareness

## Contents

- Voice command handling and NLP pipeline
- Camera-based lane and object detection scripts
- Unified pipeline that ties voice + vision into action

## Example Commands

* "KITT, move forward"
* "Stop"
* "Turn left"
* "What do you see?"

## Notes

* Can operate independently for testing
* Easily extendable to support more commands or more vision tasks
* Integrates with RL module to enable context-aware driving

> This module forms the "brain" of KITT's user interaction and perception.