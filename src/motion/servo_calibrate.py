from pathlib import Path
import json

from servo_driver import (
    Button,
    initialize,
    shutdown,
    _servos,
    BUTTON_SERVO_CONFIG,
    build_servo_config,
)

CAL_FILE = Path(__file__).parent / "servo_calibration.json"


def load_current_angles():

    build_servo_config()

    angles = {}

    for button, config in BUTTON_SERVO_CONFIG.items():
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

    initialize()

    angles = load_current_angles()

    buttons = list(Button)

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

        channel = BUTTON_SERVO_CONFIG[button].channel
        servo = _servos[channel]

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

    shutdown()


if __name__ == "__main__":
    main()