"""
servo_test.py

Interactive servo tester for:
- Adafruit PCA9685 Servo HAT
- AGFRC B13DLM V2 servos
"""

import time
from adafruit_servokit import ServoKit

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

kit = ServoKit(channels=16)

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