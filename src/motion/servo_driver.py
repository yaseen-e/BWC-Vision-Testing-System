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

from adafruit_blinka.microcontroller.generic_linux.i2c import I2C as LinuxI2C
from adafruit_servokit import ServoKit

"""CONSTANTS"""
BUS_NUM = 1

MIN_PULSE = 500
MAX_PULSE = 2500

HOME_ANGLE = 50
PRESS_ANGLE = 145

BUTTON_CHANNEL_MAP = {
    "UP": 0,
    "LEFT": 1,
    "SELECT": 2,
    "RIGHT": 3,
    "BACK": 4,
    "DOWN": 5,
    "MENU": 6,
}

class Button(Enum):
    """One servo per button on the water heater UI."""
    UP = "UP"
    LEFT = "LEFT"
    SELECT = "SELECT"
    RIGHT = "RIGHT"
    BACK = "BACK"
    DOWN = "DOWN"
    MENU = "MENU"

"""I2C Wrapper for Linux-based servo control. Only needed due to bad pins on Pi."""
class LinuxI2CBus:
    def __init__(self, bus_num):
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
_servos = {}
_initialized = False


def initialize() -> None:
    """Initialize PCA9685 and configure servos."""

    global _kit
    global _servos
    global _initialized

    if _initialized:
        return

    i2c = LinuxI2CBus(BUS_NUM)

    _kit = ServoKit(
        channels=16,
        i2c=i2c,
    )

    for channel in BUTTON_CHANNEL_MAP.values():
        servo = _kit.servo[channel]
        servo.set_pulse_width_range(
            MIN_PULSE,
            MAX_PULSE,
        )
        _servos[channel] = servo

    _initialized = True


def home_all() -> None:
    """Move every servo to its home position.

    One servo at a time.
    """

    if not _initialized:
        initialize()

    for channel in BUTTON_CHANNEL_MAP.values():
        _servos[channel].angle = HOME_ANGLE
        # Interlock startup sequence
        time.sleep(0.5)


def shutdown() -> None:
    """Disable PWM outputs."""

    if not _initialized:
        return

    for servo in _servos.values():
        servo.angle = None


def press_button(button: Button) -> bool:
    """Press a single button.

    Returns:
        True on success
        False on failure
    """

    if not _initialized:
        initialize()

    try:
        channel = BUTTON_CHANNEL_MAP[button.value]
        servo = _servos[channel]
        servo.angle = PRESS_ANGLE
        time.sleep(0.5)
        servo.angle = HOME_ANGLE
        time.sleep(0.5)
        return True

    except Exception as exc:
        print(f"Servo error: {exc}")
        return False
