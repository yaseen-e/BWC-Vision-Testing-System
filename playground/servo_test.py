"""
servo_test.py

Interactive tester for:
- PCA9685 Servo HAT
- AGFRC B13DLM V2 servos

Channel Mapping:
UP      -> Channel 0
LEFT    -> Channel 1
SELECT  -> Channel 2
RIGHT   -> Channel 3
BACK    -> Channel 4
DOWN    -> Channel 5
MENU    -> Channel 6
"""

import time
from adafruit_blinka.microcontroller.generic_linux.i2c import I2C as LinuxI2C
from adafruit_servokit import ServoKit


# ==================================================
# Linux I2C Wrapper
# ==================================================

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
            stop=stop
        )

    def readfrom_into(self, address, buffer, *, start=0, end=None, stop=True):
        return self._i2c.readfrom_into(
            address,
            buffer,
            start=start,
            end=end,
            stop=stop
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


# ==================================================
# CONFIGURATION
# ==================================================

BUS_NUM = 4

MIN_PULSE = 500
MAX_PULSE = 2500

HOME_ANGLE = 50
PRESS_ANGLE = 145

SERVO_CHANNELS = [0, 1, 2, 3, 4, 5, 6]

COMMANDS = {
    "UP": 0,
    "LEFT": 1,
    "SELECT": 2,
    "RIGHT": 3,
    "BACK": 4,
    "DOWN": 5,
    "MENU": 6,
}


# ==================================================
# INITIALIZATION
# ==================================================

print("\nInitializing PCA9685...")

i2c = LinuxI2CBus(BUS_NUM)
kit = ServoKit(channels=16, i2c=i2c)

servos = {}

for channel in SERVO_CHANNELS:

    servo = kit.servo[channel]

    servo.set_pulse_width_range(
        MIN_PULSE,
        MAX_PULSE
    )

    servos[channel] = servo

print("\nRunning startup calibration...")
print("Moving one servo at a time.\n")

for channel in SERVO_CHANNELS:

    print(f"Channel {channel} -> {HOME_ANGLE}°")

    servos[channel].angle = HOME_ANGLE

    # Wait for this servo to finish moving
    time.sleep(0.5)

print("\nInitialization complete.\n")

print("Available Commands:")
for cmd, ch in COMMANDS.items():
    print(f"  {cmd:<8} -> Channel {ch}")

print("\nType command or 'q' to quit.\n")


# ==================================================
# SERVO ACTUATION
# ==================================================

def press_button(channel):

    servo = servos[channel]

    print(f"\nChannel {channel}")

    servo.angle = PRESS_ANGLE

    time.sleep(0.5)

    servo.angle = HOME_ANGLE

    # Allow servo to settle before accepting
    # another command
    time.sleep(0.5)


# ==================================================
# MAIN LOOP
# ==================================================

try:

    while True:

        command = input("> ").strip().upper()

        if command == "Q":
            break

        if command not in COMMANDS:
            print("Invalid command")
            continue

        press_button(COMMANDS[command])

except KeyboardInterrupt:

    print("\nInterrupted")

finally:

    print("\nDisabling servos...")

    for channel in SERVO_CHANNELS:
        servos[channel].angle = None

    print("Done.")