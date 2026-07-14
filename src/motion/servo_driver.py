"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/motion/servo_driver.py - Servo Driver
Provides button mapping and servo control hooks for water heater UI actuation.
"""
# TODO: Add error handling and limited number of retries. Program breaks when servo not connected (bad I2C bus) or when servo is stuck. Should not crash whole program, just report error and skip actuation.
from __future__ import annotations

from enum import Enum
import time
from typing import Any
import json
from pathlib import Path

try:
    from adafruit_blinka.microcontroller.generic_linux.i2c import I2C as LinuxI2C
except Exception:  # pragma: no cover - hardware/environment dependent
    LinuxI2C = None

try:
    from adafruit_servokit import ServoKit
except Exception:  # pragma: no cover - hardware/environment dependent
    ServoKit = None

from dataclasses import dataclass

"""CONSTANTS"""
BUS_NUM = 1

MIN_PULSE = 500
MAX_PULSE = 2500

@dataclass
class ServoConfig:
    channel: int
    home_angle: float
    press_angle: float

class Button(Enum):
    """One servo per button on the water heater UI."""
    UP = "UP"
    LEFT = "LEFT"
    SELECT = "SELECT"
    RIGHT = "RIGHT"
    BACK = "BACK"
    DOWN = "DOWN"
    MENU = "MENU"


def _build_servo_config(channel: int, home_angle: float, press_delta: float, press_direction: int) -> ServoConfig:
    return ServoConfig(channel=channel, home_angle=home_angle, press_angle=home_angle + (press_delta * press_direction))

CAL_FILE = Path(__file__).parent / "servo_calibration.json"

DEFAULT_HOME_ANGLES = {
    Button.UP: 50,
    Button.LEFT: 155,
    Button.SELECT: 50,
    Button.RIGHT: 50,
    Button.BACK: 50,
    Button.DOWN: 50,
    Button.MENU: 155,
}

def load_home_angles() -> dict[Button, float]:

    home_angles = DEFAULT_HOME_ANGLES.copy()

    if not CAL_FILE.exists():
        return home_angles

    try:
        with open(CAL_FILE, "r") as f:
            data = json.load(f)

        for name, angle in data.items():

            try:
                button = Button[name]
                home_angles[button] = angle

            except KeyError:
                print(f"[WARNING] Unknown button in calibration file: {name}")

    except Exception as exc:
        print(f"[WARNING] Failed to load servo calibration: {exc}")

    return home_angles

def build_servo_config():

    global BUTTON_SERVO_CONFIG

    home_angles = load_home_angles()
    press_delta = 60

    BUTTON_SERVO_CONFIG = {
        Button.UP: _build_servo_config(
            channel=0,
            home_angle=home_angles[Button.UP],
            press_delta=press_delta,
            press_direction=1,
        ),
        Button.LEFT: _build_servo_config(
            channel=1,
            home_angle=home_angles[Button.LEFT],
            press_delta=press_delta,
            press_direction=-1,
        ),
        Button.SELECT: _build_servo_config(
            channel=2,
            home_angle=home_angles[Button.SELECT],
            press_delta=press_delta,
            press_direction=1,
        ),
        Button.RIGHT: _build_servo_config(
            channel=3,
            home_angle=home_angles[Button.RIGHT],
            press_delta=press_delta,
            press_direction=1,
        ),
        Button.BACK: _build_servo_config(
            channel=4,
            home_angle=home_angles[Button.BACK],
            press_delta=press_delta,
            press_direction=1,
        ),
        Button.DOWN: _build_servo_config(
            channel=5,
            home_angle=home_angles[Button.DOWN],
            press_delta=press_delta,
            press_direction=1,
        ),
        Button.MENU: _build_servo_config(
            channel=6,
            home_angle=home_angles[Button.MENU],
            press_delta=press_delta,
            press_direction=-1,
        ),
    }

"""I2C Wrapper for Linux-based servo control. Only needed due to bad pins on Pi."""
class LinuxI2CBus:
    def __init__(self, bus_num):
        if LinuxI2C is None:
            raise RuntimeError("adafruit_blinka I2C support is not available")
        self._i2c = LinuxI2C(bus_num)
        self._locked = False

    def try_lock(self):
        if not self._locked:
            self._locked = True
            return True
        return False

    def unlock(self):
        self._locked = False

    def writeto(self, address, buffer, *, start=0, end=None, stop=True):
        return self._i2c.writeto(
            address,
            buffer,
            start=start,
            end=end,
            stop=stop,
        )

    def readfrom_into(self, address, buffer, *, start=0, end=None, stop=True):
        return self._i2c.readfrom_into(
            address,
            buffer,
            start=start,
            end=end,
            stop=stop,
        )

    def writeto_then_readfrom(
        self,
        address,
        buffer_out,
        buffer_in,
        *,
        out_start=0,
        out_end=None,
        in_start=0,
        in_end=None,
        stop=False,
    ):
        return self._i2c.writeto_then_readfrom(
            address,
            buffer_out,
            buffer_in,
            out_start=out_start,
            out_end=out_end,
            in_start=in_start,
            in_end=in_end,
            stop=stop,
        )

"""Module state"""
_kit = None
_servos: dict[int, Any] = {}
_initialized = False
_servo_available = True


def initialize() -> None:
    """Initialize PCA9685 and configure servos."""

    global _kit
    global _servos
    global _initialized
    global _servo_available

    if _initialized:
        return

    if not _servo_available:
        return

    if ServoKit is None or LinuxI2C is None:
        _servo_available = False
        print("[WARNING] Servo hardware libraries unavailable; servo control disabled.")
        return

    try:
        build_servo_config()
        i2c = LinuxI2CBus(BUS_NUM)

        _kit = ServoKit(
            channels=16,
            i2c=i2c,
        )

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
    """Disable PWM outputs."""

    if not _initialized:
        return

    for servo in _servos.values():
        servo.angle = None


def press_button(button: Button) -> bool:
    if not _initialized:
        initialize()

    if not _initialized:
        print(f"[WARNING] Servo control unavailable; skipping {button.value} press.")
        return False

    try:
        config = BUTTON_SERVO_CONFIG[button]
        servo = _servos.get(config.channel)

        if servo is None:
            print(f"[Servo Error] Missing servo for {button.value}")
            return False

        servo.angle = config.press_angle
        time.sleep(0.5)

        servo.angle = config.home_angle
        time.sleep(0.5)

        return True

    except Exception as exc:
        print(f"Servo error ({button.value}): {exc}")
        return False
