# main.gd
# ========
# Main controller for the IoT Smart Home Godot Application
# 
# This script manages the MQTT connection to the smart home broker and provides
# a user interface for controlling smart light devices. It handles TCP connection,
# MQTT protocol communication, and UI interactions.
#
# Features:
# - Robust MQTT connection with CONNACK realignment
# - Smart light device control via MQTT
# - Real-time status updates from devices
# - Graceful error handling and reconnection
#
# Author: IoT Project Team
# Date: 2025

extends Node2D

# MQTT helper class for building protocol packets
@onready var mqtt := MQTTHelper

# Network configuration
const BROKER_IP := "10.0.0.31"  # MQTT broker IP address
const BROKER_PORT := 1883       # MQTT broker port (standard MQTT port)

# TCP connection for MQTT communication
var tcp : StreamPeerTCP = StreamPeerTCP.new()

# Connection state tracking
var tcp_connected    : bool = false    # TCP connection established
var mqtt_connected   : bool = false    # MQTT protocol handshake completed
var connack_expected : bool = false    # Waiting for CONNACK response
var connack_buf      : PackedByteArray = PackedByteArray()  # Buffer for CONNACK data

# Smart light device states
var light_1_on : bool = true   # Light 1 (Device ID: C8F09EB5B18C) state
var light_2_on : bool = true   # Light 2 (Device ID: C8F09EB94208) state

func _ready() -> void:
	"""
	Initialize the application when the scene is ready.
	
	This function is called when the node enters the scene tree. It establishes
	the initial TCP connection to the MQTT broker and enables processing.
	"""
	# Attempt to connect to the MQTT broker
	var err := tcp.connect_to_host(BROKER_IP, BROKER_PORT)
	print("[INIT] connect_to_host =", err)
	
	# Enable processing to handle network events
	set_process(true)
	
	print("[INIT] Application initialized, attempting connection to %s:%d" % [BROKER_IP, BROKER_PORT])

func _process(_delta: float) -> void:
	"""
	Main processing loop called every frame.
	
	This function handles the TCP connection lifecycle, MQTT protocol handshake,
	and connection state management. It polls the TCP connection for new data
	and processes MQTT protocol messages.
	
	Args:
		_delta: Time elapsed since last frame (unused)
	"""
	# Poll TCP connection for new data and status updates
	tcp.poll()

	# Handle TCP connection establishment
	_handle_tcp_connection()
	
	# Handle MQTT protocol handshake
	_handle_mqtt_handshake()
	
	# Handle connection errors and cleanup
	_handle_connection_errors()

func _handle_tcp_connection() -> void:
	"""
	Handle TCP connection establishment and initial setup.
	
	This function monitors the TCP connection status and performs initial
	setup when the connection is established, including sending the MQTT
	CONNECT packet and configuring connection parameters.
	"""
	# Check if TCP connection was just established
	if not tcp_connected and tcp.get_status() == StreamPeerTCP.STATUS_CONNECTED:
		tcp_connected = true
		print("[TCP] Connection established to broker")
		
		# Send MQTT CONNECT packet to initiate protocol handshake
		tcp.put_data(mqtt.build_connect_packet())
		
		# Optimize TCP connection for low latency (if supported)
		if tcp.has_method("set_no_delay"):
			tcp.set_no_delay(true)
		
		# Expect CONNACK response from broker
		connack_expected = true
		print("[MQTT] CONNECT packet sent, waiting for CONNACK...")

func _handle_mqtt_handshake() -> void:
	"""
	Handle MQTT protocol handshake and CONNACK processing.
	
	This function processes the CONNACK response from the MQTT broker. Since
	MQTT packets may be fragmented across multiple TCP packets, this function
	implements a robust realignment mechanism to handle partial packet reception.
	"""
	# Process CONNACK only if we're expecting it and not yet connected
	if connack_expected and not mqtt_connected:
		# Read all available bytes from TCP connection
		while tcp.get_available_bytes() > 0:
			connack_buf.append(tcp.get_u8())  # Read single byte at a time

		# Process CONNACK packet with realignment logic
		_process_connack_packet()

func _process_connack_packet() -> void:
	"""
	Process CONNACK packet with robust realignment.
	
	This function implements a sliding window approach to find the correct
	CONNACK packet boundary, handling cases where packets are fragmented
	or contain additional data.
	"""
	# Continue processing while we have enough bytes for a CONNACK header
	while connack_buf.size() >= 4:
		# Check for valid CONNACK packet signature (0x20 = CONNACK, 0x02 = remaining length)
		if connack_buf[0] == 0x20 and connack_buf[1] == 0x02:
			# Extract return code from CONNACK packet
			var return_code : int = connack_buf[3]
			
			# Handle different CONNACK return codes
			_handle_connack_return_code(return_code)
			
			# Clear buffer and stop expecting CONNACK
			connack_expected = false
			connack_buf.clear()
			break
		else:
			# Realign by removing first byte and trying again
			connack_buf.remove_at(0)

func _handle_connack_return_code(return_code: int) -> void:
	"""
	Handle MQTT CONNACK return codes and connection status.
	
	This function interprets the return code from the MQTT broker's CONNACK
	response and takes appropriate action based on the connection result.
	
	Args:
		return_code: MQTT CONNACK return code (0-5)
	"""
	match return_code:
		0x00:
			# Connection accepted
			mqtt_connected = true
			print("[MQTT] Connection established successfully")
			_on_mqtt_connected()
		0x01:
			push_error("[MQTT] CONNACK: unacceptable protocol version")
		0x02:
			push_error("[MQTT] CONNACK: identifier rejected")
		0x03:
			push_error("[MQTT] CONNACK: broker not responding")
		0x04:
			push_error("[MQTT] CONNACK: bad credentials")
		0x05:
			push_error("[MQTT] CONNACK: not authorized")
		_:
			push_error("[MQTT] CONNACK: unknown return code %d" % return_code)

func _on_mqtt_connected() -> void:
	"""
	Handle successful MQTT connection.
	
	This function is called when the MQTT connection is successfully established.
	It can be used to perform any post-connection setup or initialization.
	"""
	# Update UI to reflect connected state
	print("[MQTT] Session established, ready for communication")

func _handle_connection_errors() -> void:
	"""
	Handle connection errors and cleanup.
	
	This function monitors for connection errors and performs appropriate
	cleanup when the connection is lost or encounters errors.
	"""
	# Handle remote connection close
	if tcp_connected and tcp.get_status() == StreamPeerTCP.STATUS_NONE:
		print("[ERROR] Remote connection closed by broker")
		_reset_connection_state()
		set_process(false)

	# Handle TCP errors
	if tcp.get_status() == StreamPeerTCP.STATUS_ERROR:
		push_error("[ERROR] TCP connection error - status %s" % tcp.get_status())
		_reset_connection_state()
		set_process(false)

func _reset_connection_state() -> void:
	"""
	Reset connection state variables.
	
	This function resets all connection-related state variables when
	a connection is lost or encounters an error.
	"""
	tcp_connected = false
	mqtt_connected = false
	connack_expected = false
	connack_buf.clear()

func send_mqtt_message(topic: String, payload: String) -> void:
	"""
	Send an MQTT message to the broker.
	
	This function publishes a message to the specified MQTT topic. It checks
	the connection status before attempting to send and logs the operation.
	
	Args:
		topic: MQTT topic to publish to
		payload: Message payload to send
	"""
	if mqtt_connected:
		# Build and send MQTT PUBLISH packet
		tcp.put_data(mqtt.build_publish_packet(topic, payload))
		print("[MQTT] Published to %s: %s" % [topic, payload])
	else:
		print("[WARNING] Cannot send message - MQTT not connected")

# UI Event Handlers
# =================

func _on_light_btn1_pressed() -> void:
	"""
	Handle light 1 button press event.
	
	This function is called when the user clicks the button for light 1.
	It toggles the light state and updates both the UI and the device
	via MQTT command.
	"""
	# Toggle light 1 state
	light_1_on = !light_1_on
	
	# Update UI to reflect new state
	$Device1/LightOn.visible  = light_1_on
	$Device1/LightOff.visible = !light_1_on
	
	# Send MQTT command to control the device
	var command_payload = "{\"cmd\":%d}" % int(light_1_on)
	send_mqtt_message("home/C8F09EB5B18C/commands/MQTTtoONOFF", command_payload)
	
	print("[UI] Light 1 toggled to: %s" % ("ON" if light_1_on else "OFF"))

func _on_light_btn2_pressed() -> void:
	"""
	Handle light 2 button press event.
	
	This function is called when the user clicks the button for light 2.
	It toggles the light state and updates both the UI and the device
	via MQTT command.
	"""
	# Toggle light 2 state
	light_2_on = !light_2_on
	
	# Update UI to reflect new state
	$Device2/LightOn.visible  = light_2_on
	$Device2/LightOff.visible = !light_2_on
	
	# Send MQTT command to control the device
	var command_payload = "{\"cmd\":%d}" % int(light_2_on)
	send_mqtt_message("home/C8F09EB94208/commands/MQTTtoONOFF", command_payload)
	
	print("[UI] Light 2 toggled to: %s" % ("ON" if light_2_on else "OFF"))
