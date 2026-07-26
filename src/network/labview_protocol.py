"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/network/labview_protocol.py - LabVIEW Protocol Parsing
Defines the network command schema and parsing utilities for LabVIEW TCP communication.
"""
from __future__ import annotations

from enum import Enum


# =============================================================================
# COMMAND SCHEMA
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


# =============================================================================
# PARSER ENGINE
# =============================================================================

def parse_labview_command(command: str) -> LabViewCommand | None:
	"""Normalize a raw LabVIEW TCP command string into a known command token."""
	if not command:
		return None

	normalized = command.strip().upper()
	try:
		return LabViewCommand(normalized)
	except ValueError:
		return None
	