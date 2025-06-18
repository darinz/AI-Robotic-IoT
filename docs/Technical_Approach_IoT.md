# TECHNICAL APPROACH: IoT - REMOTE HOME AUTOMATION AND CONTROL

## TECHNICAL APPROACH

To implement our remote home automation system, we followed a modular IoT architecture focused on practical device integration, local and cloud communication, and real-time responsiveness. The system was divided into three main components—smart plugs, sound-based control, and a custom mobile application—each communicating via the MQTT protocol to ensure lightweight and efficient message passing across devices.

## OVERALL ARCHITECTURE AND DATA FLOW

The system's core functionality centers around controlling bedroom lighting via two user interfaces: physical sound interaction (clapping) and remote home automation and control using mobile app commands and in-car system integration. A Raspberry Pi 4 Model B served as the primary hub for sensor input processing and local device coordination. Smart plugs (Theengs Plug), chosen for their native MQTT support and energy monitoring features, connected directly to the MQTT broker hosted on the Raspberry Pi in the first version. This allowed local MQTT messages to turn devices on or off based on received commands.

The KY-037 sound sensor was wired to the Raspberry Pi to enable hands-free control. We developed a Python script using the RPi.GPIO library and paho-mqtt client to continuously monitor the sensor for sharp audio peaks that signify claps. When specific patterns (e.g., two rapid claps) were detected, the Raspberry Pi published a control message to a predefined MQTT topic. Smart plugs subscribed to these topics and responded instantly to the message, activating or deactivating the lights.

## REMOTE/MOBILE APPLICATION AND MQTT COMMUNICATION

The remote/mobile interface was built using Godot 4, selected for its lightweight, cross-platform capabilities. Since Godot lacks native MQTT support, we designed a custom MQTT packet builder and established a raw TCP socket to manually construct and send MQTT packets. This allowed the mobile app to interact with the local broker hosted on the Raspberry Pi over Wi-Fi, offering users a touch-based way to toggle lights and view real-time system status.

## CLOUD-ENABLED SECOND ITERATION

To extend system access beyond the home network, we developed a second version that leveraged AWS IoT Core as a secure, cloud-hosted MQTT broker. The Raspberry Pi was reconfigured as a persistent MQTT client to AWS IoT Core, using TLS encryption and X.509 certificates to ensure secure communication. It subscribed to cloud-based command topics and forwarded received messages to the local smart plugs using a second local MQTT client.

The remote/mobile app was also updated to connect directly to AWS IoT Core over the internet. We implemented secure packet handling, persistent MQTT sessions, and reconnection strategies to ensure stable performance even under variable network conditions. This allowed full control of the system from anywhere, improving the overall reliability and user convenience of the smart home system.

## CIRCUIT DESIGN

The KY-037 sound sensor was connected to the Raspberry Pi using GPIO pins—specifically, the analog output was tied to a digital input pin via a voltage divider to ensure safe input levels. A simple debounce logic in software filtered out false triggers. The Raspberry Pi then used the paho-mqtt library to send messages corresponding to detected audio events.

This architecture successfully demonstrates a robust, extensible, and secure IoT solution by combining off-the-shelf hardware with custom-built software logic. It showcases local responsiveness, remote accessibility, and real-time feedback, all centered around lightweight MQTT messaging and modular design principles.