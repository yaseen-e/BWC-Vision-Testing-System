"""
servo_test.py

Interactive servo tester for:
- Adafruit PCA9685 Servo HAT
- AGFRC B13DLM V2 servos
"""

import time
from adafruit_blinka.microcontroller.generic_linux.i2c import I2C as LinuxI2C
from adafruit_servokit import ServoKit

# Use a custom Linux I2C bus (e.g. /dev/i2c-4) with Blinka.
# The Adafruit BusDevice layer expects I2C objects to support try_lock()/unlock().
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
        return self._i2c.writeto(address, buffer, start=start, end=end, stop=stop)

    def readfrom_into(self, address, buffer, *, start=0, end=None, stop=True):
        return self._i2c.readfrom_into(address, buffer, start=start, end=end, stop=stop)

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

# ==========================================
# CONFIG
# ==========================================

CHANNEL = 0

# AGFRC digital servos usually tolerate wide pulse ranges
MIN_PULSE = 500
MAX_PULSE = 2500

# Safe initial position
START_ANGLE = 90

# ==========================================
# INITIALIZE
# ==========================================

BUS_NUM = 4

i2c = LinuxI2CBus(BUS_NUM)
kit = ServoKit(channels=16, i2c=i2c)

servo = kit.servo[CHANNEL]

# Configure PWM pulse range
servo.set_pulse_width_range(MIN_PULSE, MAX_PULSE)

print(f"\nInitializing servo on channel {CHANNEL}...")

servo.angle = START_ANGLE
time.sleep(1)

print("\nServo calibration tester")
print("Enter angle values between 0 and 180")
print("Type 'q' to quit\n")

# ==========================================
# MAIN LOOP
# ==========================================

try:
    while True:

        value = input("Angle> ").strip()

        if value.lower() == "q":
            break

        try:
            angle = float(value)

            # Safety clamp
            if angle < 0 or angle > 180:
                print("Angle must be between 0 and 180")
                continue

            print(f"Moving to {angle}°")

            servo.angle = angle

            # Allow servo time to move
            time.sleep(0.3)

        except ValueError:
            print("Invalid number")

except KeyboardInterrupt:
    print("\nInterrupted")

finally:
    print("Releasing servo...")

    # Disable PWM signal
    servo.angle = None

    print("Done.")