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

BUTTON_SERVO_CONFIG = {
    Button.UP: ServoConfig(channel=0, home_angle=50, press_angle=home_angle+55),
    Button.LEFT: ServoConfig(channel=1, home_angle=155, press_angle=home_angle-55),
    Button.SELECT: ServoConfig(channel=2, home_angle=50, press_angle=home_angle+55),
    Button.RIGHT: ServoConfig(channel=3, home_angle=50, press_angle=home_angle+55),
    Button.BACK: ServoConfig(channel=4, home_angle=50, press_angle=home_angle+55),
    Button.DOWN: ServoConfig(channel=5, home_angle=50, press_angle=home_angle+55),
    Button.MENU: ServoConfig(channel=6, home_angle=155, press_angle=home_angle-55),
}

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
_servos: dict[int, any] = {}
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

    for button, config in BUTTON_SERVO_CONFIG.items():
        servo = _kit.servo[config.channel]
        servo.set_pulse_width_range(MIN_PULSE, MAX_PULSE)
        _servos[config.channel] = servo

    _initialized = True


def home_all() -> None:
    if not _initialized:
        initialize()

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
