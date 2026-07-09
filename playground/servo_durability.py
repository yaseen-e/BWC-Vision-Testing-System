from src.motion.servo_driver import initialize, home_all, press_button, shutdown, Button
import time


def main() -> None:
    initialize()
    home_all()

    try:
        while True:
            try:
                cmd = input("Press Enter to cycle buttons, or 'Q' then Enter to quit: ").strip().upper()
            except (EOFError, KeyboardInterrupt):
                break

            if cmd == "Q":
                break

            sequence = [
                Button.UP,
                Button.DOWN,
                Button.LEFT,
                Button.RIGHT,
                Button.SELECT,
                Button.MENU,
                Button.BACK,
                Button.UP,
                Button.DOWN,
                Button.LEFT,
                Button.RIGHT,
                Button.SELECT,
                Button.MENU,
                Button.BACK,
                Button.UP,
                Button.DOWN,
                Button.LEFT,
                Button.RIGHT,
                Button.SELECT,
                Button.MENU,
                Button.BACK,
                Button.UP,
                Button.DOWN,
                Button.LEFT,
                Button.RIGHT,
                Button.SELECT,
                Button.MENU,
                Button.BACK,
                Button.UP,
                Button.DOWN,
                Button.LEFT,
                Button.RIGHT,
                Button.SELECT,
                Button.MENU,
                Button.BACK,
                Button.UP,
                Button.DOWN,
                Button.LEFT,
                Button.RIGHT,
                Button.SELECT,
                Button.MENU,
                Button.BACK,
                Button.UP,
                Button.DOWN,
                Button.LEFT,
                Button.RIGHT,
                Button.SELECT,
                Button.MENU,
                Button.BACK,
                Button.UP,
                Button.DOWN,
                Button.LEFT,
                Button.RIGHT,
                Button.SELECT,
                Button.MENU,
                Button.BACK,
                Button.UP,
                Button.DOWN,
                Button.LEFT,
                Button.RIGHT,
                Button.SELECT,
                Button.MENU,
                Button.BACK,
            ]

            for btn in sequence:
                press_button(btn)
                time.sleep(0.25)
    finally:
        shutdown()


if __name__ == "__main__":
    main()