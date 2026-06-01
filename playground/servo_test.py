from src.motion.servo_driver import *

initialize()
home_all()

try:

    while True:

        cmd = input("> ").strip().upper()

        if cmd == "Q":
            break

        try:
            button = Button[cmd]
            press_button(button)

        except KeyError:
            print("Invalid command")

finally:
    shutdown()