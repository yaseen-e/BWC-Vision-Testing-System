from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from src.motion import servos as servo_driver

CAL_FILE = Path(__file__).parent / "servo_calibration.json"


def load_current_angles():

    servo_driver.build_servo_config()

    angles = {}

    for button, config in servo_driver.BUTTON_SERVO_CONFIG.items():
        angles[button] = config.home_angle

    return angles


def save_angles(angles):

    data = {
        button.name: angle
        for button, angle in angles.items()
    }

    with open(CAL_FILE, "w") as f:
        json.dump(data, f, indent=4)


def main():

    servo_driver.initialize()

    angles = load_current_angles()

    buttons = list(servo_driver.Button)

    while True:

        print("\n=== Servo Calibration ===")

        for idx, button in enumerate(buttons, start=1):

            print(
                f"{idx}: {button.name:<8} "
                f"({angles[button]}°)"
            )

        print("\nQ: Quit")

        selection = input(
            "\nSelect Servo: "
        ).strip().upper()

        if selection == "Q":
            break

        try:
            button = buttons[int(selection) - 1]
        except Exception:
            continue

        channel = servo_driver.BUTTON_SERVO_CONFIG[button].channel
        servo = servo_driver._servos[channel]

        current = angles[button]

        servo.angle = current

        while True:

            print(
                f"\n{button.name} = {current}°"
            )

            print(
                "a=-1  d=+1  "
                "z=-5  c=+5  "
                "s=save  q=back"
            )

            cmd = input("> ").lower()

            if cmd == "a":
                current -= 1

            elif cmd == "d":
                current += 1

            elif cmd == "z":
                current -= 5

            elif cmd == "c":
                current += 5

            elif cmd == "s":

                angles[button] = current
                save_angles(angles)

                print(
                    f"Saved {button.name} = "
                    f"{current}°"
                )

                continue

            elif cmd == "q":
                break

            else:
                continue

            current = max(0, min(180, current))
            servo.angle = current

    servo_driver.shutdown()


if __name__ == "__main__":
    main()