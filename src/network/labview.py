"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/network/labview.py - LabVIEW Protocol & Network Interface
Unified LabVIEW network driver handling command parsing, socket communications,
and simulated command sequence replays.
"""
from __future__ import annotations

import collections
from enum import Enum
import socket
from typing import Optional


# =============================================================================
# COMMAND SCHEMA & PARSER
# =============================================================================

class LabViewCommand(Enum):
	"""Commands received over TCP from LabVIEW or simulation queue."""

	RUN_OCR = "RUN_OCR"
	SHUTDOWN = "SHUTDOWN"
	UP = "UP"
	LEFT = "LEFT"
	SELECT = "SELECT"
	RIGHT = "RIGHT"
	BACK = "BACK"
	DOWN = "DOWN"
	MENU = "MENU"


LABVIEW_COMMANDS: tuple[str, ...] = tuple(command.value for command in LabViewCommand)


def parse_labview_command(command: str) -> LabViewCommand | None:
	"""Normalize a raw LabVIEW TCP command string into a known command token."""
	if not command:
		return None

	normalized = command.strip().upper()
	try:
		return LabViewCommand(normalized)
	except ValueError:
		return None


# =============================================================================
# SIMULATION & RESPONSE DATA
# =============================================================================

_SIMULATED_COMMANDS: collections.deque[str] = collections.deque(
	[
		# Homescreen Verification
		"BACK", "RUN_OCR", "BACK", "RUN_OCR", "BACK", "RUN_OCR", "SHUTDOWN",
		# Location Selection Test Sequence
		"MENU", "RIGHT", "DOWN", "SELECT", "DOWN", "DOWN", "SELECT", "DOWN", "DOWN", "SELECT",
		# USA
		"SELECT", "SELECT", "RUN_OCR",
		# CANADA
		"LEFT", "SELECT", "DOWN", "SELECT", "RUN_OCR",
		# MEXICO
		"LEFT", "SELECT", "DOWN", "DOWN", "SELECT", "RUN_OCR",
		# OTHER
		"LEFT", "SELECT", "DOWN", "DOWN", "DOWN", "SELECT", "RUN_OCR", "SHUTDOWN",
		"BACK", "BACK", "BACK", "BACK",
		# User Schedules Test Sequence
		"MENU", "RIGHT", "RIGHT", "SELECT", "SELECT", "SELECT", "RUN_OCR",
		"DOWN", "SELECT", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "SELECT", "SELECT", "RUN_OCR",
		"SELECT", "BACK", "BACK", "BACK", "BACK",
		# Mode Transition Test Sequence
		# HEAT_PUMP
		"BACK", "MENU", "SELECT", "LEFT", "LEFT", "SELECT", "DOWN", "DOWN", "SELECT", "RIGHT", "RIGHT", "RUN_OCR",
		# HYBRID_STANDARD
		"BACK", "MENU", "SELECT", "LEFT", "LEFT", "SELECT", "SELECT", "RIGHT", "RIGHT", "RUN_OCR",
		# HYBRID_PLUS
		"BACK", "MENU", "SELECT", "LEFT", "LEFT", "SELECT", "DOWN", "SELECT", "RIGHT", "RIGHT", "RUN_OCR",
		# ELECTRIC
		"BACK", "MENU", "SELECT", "LEFT", "LEFT", "SELECT", "DOWN", "DOWN", "DOWN", "SELECT", "RIGHT", "RIGHT", "RUN_OCR",
		# VACATION
		"BACK", "MENU", "SELECT", "LEFT", "LEFT", "SELECT", "DOWN", "DOWN", "DOWN", "DOWN", "SELECT", "RIGHT", "RIGHT", "RUN_OCR",
		"SHUTDOWN",
	]
)

_COMMAND_RESPONSES: dict[str, str] = {
	"UP": "Button_Pressed",
	"SELECT": "Button_Pressed",
	"DOWN": "Button_Pressed",
	"LEFT": "Button_Pressed",
	"RIGHT": "Button_Pressed",
	"BACK": "Button_Pressed",
	"MENU": "Button_Pressed",
	"RUN_OCR": "Running_OCR123",
	"SEND_OCR_RESULT": "Sending_OCR123",
}


# =============================================================================
# MODULE STATE
# =============================================================================

_SERVER_SOCKET: Optional[socket.socket] = None
_CONN: Optional[socket.socket] = None


# =============================================================================
# SERVER LIFECYCLE & CONNECTION MANAGEMENT
# =============================================================================

def start_tcp_server(host: str = "0.0.0.0", port: int = 5000) -> bool:
	"""Start TCP server to listen for LabVIEW client connections."""
	global _SERVER_SOCKET, _CONN

	try:
		server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		server.bind((host, port))
		server.listen(1)
		server.settimeout(0.1)
		_SERVER_SOCKET = server
		_CONN = None
		print(f"[NETWORK] TCP server listening on {host}:{port}")
		print("[NETWORK] Waiting for connection...")
		return True
	except Exception as exc:
		_SERVER_SOCKET = None
		_CONN = None
		print(f"[ERROR] Failed to start TCP server on {host}:{port}: {exc}")
		return False


def _accept_connection_if_needed() -> bool:
	"""Accept the first LabVIEW client connection without blocking the main loop."""
	global _CONN
	if _CONN is not None or _SERVER_SOCKET is None:
		return _CONN is not None

	try:
		new_conn, addr = _SERVER_SOCKET.accept()
		new_conn.settimeout(0.1)
		_CONN = new_conn
		print(f"[NETWORK] Connected by {addr}")
		return True
	except socket.timeout:
		return False
	except Exception as exc:
		print(f"[WARNING] TCP accept failed: {exc}")
		return False


# =============================================================================
# COMMAND & REPORTING API
# =============================================================================

def get_next_command(simulated: bool = False) -> str:
	"""Receive next command from simulated queue or LabVIEW TCP socket."""
	if simulated:
		if _SIMULATED_COMMANDS:
			command = _SIMULATED_COMMANDS.popleft()
			print(f"[SIMULATION] Replaying command: {command}")
			return command
		print("[SIMULATION] No more simulated commands.")
		return ""

	if not _accept_connection_if_needed() or _CONN is None:
		return ""

	try:
		data = _CONN.recv(1024)
		if data:
			command = data.decode().strip()
			print(f"[NETWORK] Received: {command}")
			response = _COMMAND_RESPONSES.get(command, "Unknown Command\n")
			_CONN.sendall(response.encode())
			return command
		return ""
	except socket.timeout:
		return ""
	except Exception as exc:
		print(f"[WARNING] TCP receive failed: {exc}")
		return ""


def send_report(ocr_result: str, simulated: bool = False) -> None:
	"""Send OCR test result back to LabVIEW over TCP socket or print simulation."""
	if simulated:
		print(f"[SIMULATION] OCR report ready:\n{ocr_result}")
		return

	if _CONN is None:
		print("[WARNING] Cannot send OCR report: no LabVIEW connection yet.")
		return

	try:
		response = f"OCR ready\n{ocr_result}\n"
		_CONN.sendall(response.encode())
		print("[NETWORK] Sent OCR report to LabVIEW.")
	except Exception as exc:
		print(f"[WARNING] Failed to send OCR report: {exc}")
		return
