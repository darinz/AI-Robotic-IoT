# mqtt.gd
# ========
# MQTT Protocol Helper for Godot IoT Application
# 
# This class provides static methods for building MQTT protocol packets
# according to the MQTT 3.1.1 specification. It handles packet construction
# for CONNECT and PUBLISH operations used by the smart home application.
#
# MQTT Protocol Overview:
# - CONNECT: Establishes connection to broker with authentication
# - PUBLISH: Sends messages to specific topics
# - Packet structure: Fixed header + Variable header + Payload
#
# Author: IoT Project Team
# Date: 2024

class_name MQTTHelper

# MQTT Protocol Constants
const MQTT_PROTOCOL_NAME := "MQTT"
const MQTT_PROTOCOL_LEVEL := 0x04  # MQTT 3.1.1
const MQTT_KEEP_ALIVE := 60        # Keep-alive interval in seconds

# MQTT Packet Types
const PACKET_TYPE_CONNECT := 0x10
const PACKET_TYPE_PUBLISH := 0x30

# MQTT Connect Flags
const CONNECT_FLAG_CLEAN_SESSION := 0x02
const CONNECT_FLAG_USERNAME := 0x80
const CONNECT_FLAG_PASSWORD := 0x40

static func build_connect_packet(
		client_id: String = "godot-client",
		username: String = "USERNAME HERE",
		password: String = "PASSWORD HERE"
	) -> PackedByteArray:
	"""
	Build an MQTT CONNECT packet for establishing connection to broker.
	
	This function constructs a complete MQTT CONNECT packet according to the
	MQTT 3.1.1 specification. The packet includes protocol information,
	connection flags, authentication credentials, and client identification.
	
	Args:
		client_id: Unique identifier for this MQTT client
		username: MQTT broker username for authentication
		password: MQTT broker password for authentication
	
	Returns:
		PackedByteArray containing the complete CONNECT packet
	
	Packet Structure:
		Fixed Header: [Packet Type][Remaining Length]
		Variable Header: [Protocol Name][Protocol Level][Connect Flags][Keep Alive]
		Payload: [Client ID][Username][Password]
	"""
	
	# Initialize packet buffer
	var packet = PackedByteArray()
	
	# Convert strings to UTF-8 byte arrays for packet construction
	var client_id_utf = client_id.to_utf8_buffer()
	var username_utf = username.to_utf8_buffer()
	var password_utf = password.to_utf8_buffer()
	var protocol_name_utf = MQTT_PROTOCOL_NAME.to_utf8_buffer()

	# Build Variable Header
	var variable_header = PackedByteArray()
	
	# Protocol name (2-byte length + name)
	variable_header.append_array([0x00, protocol_name_utf.size()])
	variable_header.append_array(protocol_name_utf)
	
	# Protocol level (MQTT 3.1.1 = 0x04)
	variable_header.append(MQTT_PROTOCOL_LEVEL)
	
	# Connect flags (username, password, clean session)
	var connect_flags = CONNECT_FLAG_USERNAME | CONNECT_FLAG_PASSWORD | CONNECT_FLAG_CLEAN_SESSION
	variable_header.append(connect_flags)
	
	# Keep alive (2-byte big-endian)
	variable_header.append_array([MQTT_KEEP_ALIVE >> 8, MQTT_KEEP_ALIVE & 0xFF])

	# Build Payload
	var payload = PackedByteArray()
	
	# Client ID (2-byte length + ID)
	payload.append_array([0x00, client_id_utf.size()])
	payload.append_array(client_id_utf)

	# Username (2-byte length + username)
	payload.append_array([0x00, username_utf.size()])
	payload.append_array(username_utf)

	# Password (2-byte length + password)
	payload.append_array([0x00, password_utf.size()])
	payload.append_array(password_utf)

	# Calculate remaining length (variable header + payload)
	var remaining_length = variable_header.size() + payload.size()

	# Build Fixed Header
	packet.append(PACKET_TYPE_CONNECT)  # CONNECT packet type
	packet.append(remaining_length)     # Remaining length
	
	# Append variable header and payload
	packet.append_array(variable_header)
	packet.append_array(payload)

	return packet

static func build_publish_packet(topic: String, payload: String) -> PackedByteArray:
	"""
	Build an MQTT PUBLISH packet for sending messages to topics.
	
	This function constructs a complete MQTT PUBLISH packet according to the
	MQTT 3.1.1 specification. The packet includes the topic name and message
	payload for publishing to the MQTT broker.
	
	Args:
		topic: MQTT topic to publish to (e.g., "home/device/commands")
		payload: Message payload to send (typically JSON string)
	
	Returns:
		PackedByteArray containing the complete PUBLISH packet
	
	Packet Structure:
		Fixed Header: [Packet Type][Remaining Length]
		Variable Header: [Topic Name]
		Payload: [Message Payload]
	"""
	
	# Initialize packet buffer
	var packet = PackedByteArray()
	
	# Convert topic and payload to UTF-8 byte arrays
	var topic_utf = topic.to_utf8_buffer()
	var payload_bytes = payload.to_utf8_buffer()
	
	# Build Variable Header (topic name)
	var variable_header = PackedByteArray()
	
	# Topic name (2-byte length + topic)
	var topic_length = topic_utf.size()
	variable_header.append_array([topic_length >> 8, topic_length & 0xFF])
	variable_header.append_array(topic_utf)
	
	# Note: For QoS 0 (fire and forget), no packet ID is needed
	
	# Calculate remaining length (variable header + payload)
	var remaining_length = variable_header.size() + payload_bytes.size()
	
	# Build Fixed Header
	packet.append(PACKET_TYPE_PUBLISH)  # PUBLISH packet type with QoS 0
	packet.append(remaining_length)     # Remaining length
	
	# Append variable header and payload
	packet.append_array(variable_header)
	packet.append_array(payload_bytes)
	
	return packet

# Utility Functions
# =================

static func calculate_remaining_length(variable_header_size: int, payload_size: int) -> int:
	"""
	Calculate the remaining length field for MQTT packets.
	
	The remaining length field indicates the number of bytes in the variable
	header and payload sections of the MQTT packet.
	
	Args:
		variable_header_size: Size of the variable header in bytes
		payload_size: Size of the payload in bytes
	
	Returns:
		Total remaining length in bytes
	"""
	return variable_header_size + payload_size

static func encode_utf8_string(text: String) -> PackedByteArray:
	"""
	Encode a string as UTF-8 with length prefix for MQTT packets.
	
	MQTT strings are encoded as 2-byte length followed by UTF-8 bytes.
	This is used for topic names, client IDs, usernames, and passwords.
	
	Args:
		text: String to encode
	
	Returns:
		PackedByteArray with length prefix and UTF-8 bytes
	"""
	var text_utf = text.to_utf8_buffer()
	var result = PackedByteArray()
	
	# Add 2-byte length prefix (big-endian)
	result.append_array([text_utf.size() >> 8, text_utf.size() & 0xFF])
	
	# Add UTF-8 bytes
	result.append_array(text_utf)
	
	return result
