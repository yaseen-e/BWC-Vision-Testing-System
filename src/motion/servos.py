"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/motion/servos.py - Servo Driver
Provides button mapping and servo control hooks for water heater UI actuation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import time
from typing import Any, Optional

try:
	from adafruit_servokit import ServoKit
except Exception:  # Hardware/environment dependent
	ServoKit = None


# =============================================================================
# HARDWARE CONFIGURATION & DATA MODELS
# =============================================================================

BUS_NUM = 1
MIN_PULSE = 500
MAX_PULSE = 2500
STROKE_LENGTH = 55.0

CAL_FILE = Path(__file__).resolve().parents[2] / "playground" / "servo_calibration.json"


@dataclass
class ServoConfig:
	channel: int
	home_angle: float
	press_angle: float


class Button(Enum):
	"""One servo mapping per physical button on the water heater UI."""

	UP = "UP"
	LEFT = "LEFT"
	SELECT = "SELECT"
	RIGHT = "RIGHT"
	BACK = "BACK"
	DOWN = "DOWN"
	MENU = "MENU"


DEFAULT_HOME_ANGLES: dict[Button, float] = {
	Button.UP: 50.0,
	Button.LEFT: 155.0,
	Button.SELECT: 50.0,
	Button.RIGHT: 50.0,
	Button.BACK: 50.0,
	Button.DOWN: 50.0,
	Button.MENU: 155.0,
}


# =============================================================================
# MODULE STATE
# =============================================================================

_kit: Any = None
_servos: dict[int, Any] = {}
_initialized: bool = False
_servo_available: bool = True
BUTTON_SERVO_CONFIG: dict[Button, ServoConfig] = {}


# =============================================================================
# CALIBRATION & CONFIGURATION HELPERS
# =============================================================================

def load_home_angles() -> dict[Button, float]:
	"""Load custom calibrated home angles or default to defaults."""
	home_angles = DEFAULT_HOME_ANGLES.copy()

	if not CAL_FILE.exists():
		return home_angles

	try:
		with open(CAL_FILE, "r") as f:
			data = json.load(f)

		for name, angle in data.items():
			try:
				button = Button[name]
				home_angles[button] = float(angle)
			except KeyError:
				print(f"[WARNING] Unknown button in calibration file: {name}")

	except Exception as exc:
		print(f"[WARNING] Failed to load servo calibration: {exc}")

	return home_angles


def _build_servo_config(
	channel: int, home_angle: float, press_delta: float, press_direction: int
) -> ServoConfig:
	"""Build a ServoConfig given baseline angle, stroke length, and direction."""
	return ServoConfig(
		channel=channel,
		home_angle=home_angle,
		press_angle=home_angle + (press_delta * press_direction),
	)


def build_servo_config() -> None:
	"""Construct default servo configurations for all active buttons."""
	global BUTTON_SERVO_CONFIG

	home_angles = load_home_angles()

	BUTTON_SERVO_CONFIG = {
		Button.UP: _build_servo_config(
			channel=0,
			home_angle=home_angles[Button.UP],
			press_delta=STROKE_LENGTH,
			press_direction=1,
		),
		Button.LEFT: _build_servo_config(
			channel=1,
			home_angle=home_angles[Button.LEFT],
			press_delta=STROKE_LENGTH,
			press_direction=-1,
		),
		Button.SELECT: _build_servo_config(
			channel=2,
			home_angle=home_angles[Button.SELECT],
			press_delta=STROKE_LENGTH,
			press_direction=1,
		),
		Button.RIGHT: _build_servo_config(
			channel=3,
			home_angle=home_angles[Button.RIGHT],
			press_delta=STROKE_LENGTH,
			press_direction=1,
		),
		Button.BACK: _build_servo_config(
			channel=4,
			home_angle=home_angles[Button.BACK],
			press_delta=STROKE_LENGTH,
			press_direction=1,
		),
		Button.DOWN: _build_servo_config(
			channel=5,
			home_angle=home_angles[Button.DOWN],
			press_delta=STROKE_LENGTH,
			press_direction=1,
		),
		Button.MENU: _build_servo_config(
			channel=6,
			home_angle=home_angles[Button.MENU],
			press_delta=STROKE_LENGTH,
			press_direction=-1,
		),
	}


# =============================================================================
# HARDWARE INITIALIZATION & LIFECYCLE
# =============================================================================

def initialize() -> None:
	"""Initialize PCA9685 PWM board and configure all active servo channels."""
	global _kit, _servos, _initialized, _servo_available

	if _initialized or not _servo_available:
		return

	if ServoKit is None:
		_servo_available = False
		print("[WARNING] Servo hardware libraries unavailable; servo control disabled.")
		return

	try:
		build_servo_config()
		_kit = ServoKit(channels=16)

		for button, config in BUTTON_SERVO_CONFIG.items():
			servo = _kit.servo[config.channel]
			servo.set_pulse_width_range(MIN_PULSE, MAX_PULSE)
			_servos[config.channel] = servo

		_initialized = True
	except Exception as exc:
		_servo_available = False
		_kit = None
		_servos = {}
		print(f"[WARNING] Servo initialization skipped: {exc}")


def home_all() -> None:
	"""Park all active servos back to their home resting positions."""
	if not _initialized:
		initialize()

	if not _initialized:
		return

	for button, config in BUTTON_SERVO_CONFIG.items():
		servo = _servos.get(config.channel)
		if servo is None:
			continue

		try:
			servo.angle = config.home_angle
			time.sleep(0.2)
		except Exception as exc:
			print(f"[Home Error] {button.value}: {exc}")


def shutdown() -> None:
	"""Disable PWM output signals to relieve servo motor tension."""
	if not _initialized:
		return

	for servo in _servos.values():
		servo.angle = None


# =============================================================================
# ACTUATION API
# =============================================================================

def press_button(button: Button, press_ready: Optional[list[int]] = None) -> bool:
	"""Actuate a specific button by sweeping to press angle and returning home."""
	if press_ready is not None:
		press_ready[0] = 0

	if not _initialized:
		initialize()

	if not _initialized:
		print(f"[WARNING] Servo control unavailable; skipping {button.value} press.")
		if press_ready is not None:
			press_ready[0] = 1
		return False

	try:
		config = BUTTON_SERVO_CONFIG[button]
		servo = _servos.get(config.channel)

		if servo is None:
			print(f"[Servo Error] Missing servo for {button.value}")
			if press_ready is not None:
				press_ready[0] = 1
			return False

		servo.angle = config.press_angle
		time.sleep(0.5)

		servo.angle = config.home_angle
		time.sleep(0.5)

		if press_ready is not None:
			press_ready[0] = 1

		return True

	except Exception as exc:
		print(f"Servo error ({button.value}): {exc}")
		if press_ready is not None:
			press_ready[0] = 1
		return False
