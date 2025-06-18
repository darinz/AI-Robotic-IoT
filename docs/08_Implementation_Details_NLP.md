# IMPLEMENTATION DETAILS: NLP - NATURAL LANGUAGE PROCESSING AND CONVERSATIONAL INTERFACE

## IMPLEMENTATION DETAILS

The conversational interface was implemented as a modular, extensible pipeline centered around the Raspberry Pi 4, which served as the primary control hub. This design allowed for a seamless fusion of real-time sensory input, language processing, and robotic actuation, while distributing computational workload across both onboard and networked resources. The core of the natural language processing (NLP) system integrated a suite of open-source software components and powerful, pretrained large language models (LLMs). Python served as the glue language for orchestrating this ecosystem, enabling rapid prototyping, customization, and hardware integration.

Audio-to-text transcription began with the SpeechRecognition library, chosen for its high-level abstraction and compatibility with multiple speech engines. This was paired with PyAudio to interface directly with the USB microphone, and sox (Sound eXchange) to preprocess audio signals, including downsampling and noise reduction when needed. These tools allowed for efficient capture and preprocessing of speech on the Raspberry Pi, which has limited computational headroom.

To minimize latency and maintain responsiveness, the system adopted a hybrid inference model. Lightweight preprocessing, audio handling, and response parsing were kept onboard the Pi, while the more computationally demanding natural language inference was offloaded. RESTful API requests were dispatched using the requests Python library to either cloud-based models such as OpenAI's GPT-4 or to open-source LLMs (e.g., LLaMA) hosted on a local machine. This architecture enabled a flexible trade-off between performance, cost, and data privacy—automatically falling back to local models in the event of network latency or API throttling.

This modular pipeline was designed with clear interfaces between components, allowing for future improvements, such as switching TTS engines or plugging in new models, without requiring a full system rewrite. Each stage in the pipeline—from audio acquisition to NLP processing to physical actuation—communicated through shared state variables, function dispatch tables, and JSON-based message structures to ensure extensibility and maintainability.
TWO DIFFERENT LLM BACKENDS WERE TESTED:

Two distinct backend architectures for large language models (LLMs) were implemented and evaluated to balance performance, latency, and control over data privacy. These included both commercial cloud-hosted models and self-hosted open-source alternatives.

The first and most performant option involved leveraging OpenAI's GPT-4 and GPT-4o via the official openai Python SDK. These models were accessed through HTTPS requests using an API key, and were chosen for their state-of-the-art natural language fluency, multi-turn conversation memory, and nuanced contextual understanding. The cloud API returned well-structured completions with very low error rates and high consistency, which made them ideal for conversational tasks that demanded personality, wit, or emotional tone matching (such as simulating KITT-like behavior). However, reliance on an internet connection and per-token costs made this option less desirable in resource-constrained or privacy-sensitive scenarios.

To provide an offline and cost-free alternative, a local inference server was configured on a machine within the local area network (LAN). This server hosted open-source LLMs such as quantized LLaMA variants, deployed via HuggingFace’s transformers library. To expose these models to the Raspberry Pi in a RESTful format, lightweight web frameworks such as FastAPI and Flask were used to wrap inference calls into simple HTTP POST endpoints. These endpoints accepted JSON payloads containing user input and system prompt templates, and returned model completions as structured JSON responses.

To optimize performance, models were served in quantized form, with response times ranging from 5 to 10 seconds depending on prompt length and system load. This setup offered flexibility for use cases where latency tolerance was higher, and where complete control over the language model was desired (e.g., in educational or research deployments). The local models also allowed for fine-tuning or prompt injection strategies that would otherwise be restricted or rate-limited by commercial APIs.

This dual-backend approach—cloud-first, LAN-fallback—was complemented by a confidence-aware switching mechanism that dynamically chose the optimal model based on network availability, task complexity, and latency tolerances.

The returned natural language responses were parsed using custom-built regex-based heuristics that extracted action intents from key phrases (e.g., “turn right,” “say hello,” “think about it”). These were mapped to function calls on the Raspberry Pi via a dictionary-based command dispatcher. The re module facilitated robust action parsing across variable phrasings.

For text-to-speech (TTS), we tested lightweight onboard solutions. Onboard audio output was routed through an I2S audio amplifier connected to the Raspberry Pi’s GPIO pins.

Hardware integration was managed through the PiCar-X’s Robot HAT and controlled with SunFounder_PiCar-X Python SDK. Servo motors (used for gimbal movement and gestures) and DC motors (used for directional movement) were triggered based on parsed intent. A state manager maintained the current posture of the robot to avoid redundant or conflicting commands.

### List of Referenced Materials and Tools
- SpeechRecognition: https://github.com/Uberi/speech_recognition
- PyAudio: https://people.csail.mit.edu/hubert/pyaudio/
- sox (Sound eXchange): http://sox.sourceforge.net/
- OpenAI GPT-4 / GPT-4o API: https://platform.openai.com/docs
- HuggingFace Transformers (LLaMA): https://huggingface.co/transformers/
