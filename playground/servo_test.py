from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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