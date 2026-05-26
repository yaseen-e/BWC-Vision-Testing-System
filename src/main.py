"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/main.py - Main Event Loop
Coordinates startup, command handling, actuation, OCR reads, reporting, and shutdown.
"""

from pathlib import Path
import csv
import select
import sys
import time
import traceback
import termios
import tty
from enum import Enum, auto
from typing import Optional

import data_manager
from motion import servo_driver
from vision import vision_engine
from vision.display_layouts import CURRENT_LAYOUT


# Capture storage layout (repo-relative): data/captures
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIR = PROJECT_ROOT / "data" / "captures"
TEST_REPORT_DIR = PROJECT_ROOT / "data" / "test_reports"
CLEANUP_RETENTION_DAYS = 14


# Whole-second waits keep the state machine readable and avoid rushing through tasks.
STATE_SLEEP_SECONDS = {
    "STARTUP": 2,
    "WAIT_FOR_COMMAND": 1,
    "PRESS_BUTTON": 2,
    "READ_DISPLAY": 3,
    "REPORT_TO_LABVIEW": 1,
    "ERROR": 1,
    "SHUTDOWN": 2,
}

class SystemState(Enum):
    STARTUP = auto()
    WAIT_FOR_COMMAND = auto()
    PRESS_BUTTON = auto()
    READ_DISPLAY = auto()
    REPORT_TO_LABVIEW = auto()
    ERROR = auto()
    SHUTDOWN = auto()


class LabViewCommand(Enum):
    RUN_OCR = "RUN_OCR"
    SHUTDOWN = "SHUTDOWN"
    UP = "UP"
    LEFT = "LEFT"
    SELECT = "SELECT"
    RIGHT = "RIGHT"
    BACK = "BACK"
    DOWN = "DOWN"
    MENU = "MENU"


LABVIEW_COMMANDS: tuple[str, ...] = tuple(command.value for command in LabViewCommand)
LABVIEW_BUTTON_COMMANDS = {
    LabViewCommand.UP: servo_driver.Button.UP,
    LabViewCommand.LEFT: servo_driver.Button.LEFT,
    LabViewCommand.SELECT: servo_driver.Button.SELECT,
    LabViewCommand.RIGHT: servo_driver.Button.RIGHT,
    LabViewCommand.BACK: servo_driver.Button.BACK,
    LabViewCommand.DOWN: servo_driver.Button.DOWN,
    LabViewCommand.MENU: servo_driver.Button.MENU,
}


def _parse_labview_command(command: str) -> Optional[LabViewCommand]:
    """Normalize a LabVIEW TCP/IP command into a known command token."""
    if not command:
        return None

    normalized = command.strip().upper()
    try:
        return LabViewCommand(normalized)
    except ValueError:
        return None


def _run_capture_cleanup() -> None:
    """Run retention cleanup during startup; failures do not stop system startup."""
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        data_manager.cleanup_captures(
            captures_dir=CAPTURE_DIR,
            retention_days=CLEANUP_RETENTION_DAYS,
            apply_changes=True,
            allowed_root=CAPTURE_DIR,
        )
        print(f"[INFO] Capture cleanup applied (retention={CLEANUP_RETENTION_DAYS} days).")
    except Exception as exc:
        print(f"[WARNING] Capture cleanup error: {exc}")


def _open_test_report() -> tuple[object, csv.DictWriter, Path]:
    """Create a timestamped CSV report for this run and write header row."""
    TEST_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_id = data_manager.build_capture_id("test_report")
    report_path = TEST_REPORT_DIR / f"{report_id}.csv"
    report_file = report_path.open("w", newline="", encoding="utf-8")
    fieldnames = ["step", "menu"] + list(CURRENT_LAYOUT.fields.keys())
    writer = csv.DictWriter(report_file, fieldnames=fieldnames)
    writer.writeheader()
    report_file.flush()
    return report_file, writer, report_path


def _enable_single_key_mode() -> tuple[object | None, object | None]:
    """Put terminal in cbreak mode so single key presses are readable."""
    if not sys.stdin.isatty():
        return None, None

    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return fd, previous


def _disable_single_key_mode(fd: object | None, previous: object | None) -> None:
    """Restore terminal mode after single-key capture usage."""
    if fd is None or previous is None:
        return
    termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def _space_pressed() -> bool:
    """Return True when a space key is waiting on stdin."""
    if not sys.stdin.isatty():
        return False

    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return False

    return sys.stdin.read(1) == " "


def main():
    """Main Event Loop (The Orchestrator)"""
    
    # initial state on startup
    current_state = SystemState.STARTUP
    last_command = ""
    pending_command: LabViewCommand | None = None
    ocr_result = ""
    error_message = ""
    readout = None
    step_counter = 0
    report_file, report_writer, report_path = _open_test_report()
    stdin_fd, previous_termios = _enable_single_key_mode()
    print("--- Starting BWC Water Heater Vision Testing System ---")
    print(f"[INFO] Test report CSV: {report_path}")

    try:
        while True:
            if current_state != SystemState.SHUTDOWN and _space_pressed():
                print("[INFO] Space pressed. Entering SHUTDOWN state.")
                current_state = SystemState.SHUTDOWN

            match current_state:
                
                case SystemState.STARTUP:
                    print("[INFO] Cleaning up old captures.")
                    _run_capture_cleanup()
                    # Bring hardware to a known state before first command.
                    print("[INFO] Initializing servos.")
                    servo_driver.initialize()
                    servo_driver.home_all()
                    print("[INFO] Initializing vision engine.")
                    if vision_engine.is_camera_available():
                        print("[INFO] Camera detected.")
                    else:
                        print("[WARNING] Camera NOT found.")
                    current_state = SystemState.WAIT_FOR_COMMAND
                    time.sleep(STATE_SLEEP_SECONDS["STARTUP"])
                    
                case SystemState.WAIT_FOR_COMMAND:
                    last_command = LabViewCommand.UP.value  # simulated command received from LabVIEW
                    print(f"[NETWORK] Received command from LabVIEW: {last_command}")
                    # TODO: listen to LabVIEW via Serial or TCP/IP socket      
                    if last_command: # string is true if not empty = command received
                        # --- COMMAND ROUTING LOGIC ---
                        pending_command = _parse_labview_command(last_command)
                        if pending_command in LABVIEW_BUTTON_COMMANDS:
                            current_state = SystemState.PRESS_BUTTON
                        elif pending_command is LabViewCommand.RUN_OCR:
                            current_state = SystemState.READ_DISPLAY
                        elif pending_command is LabViewCommand.SHUTDOWN:
                            current_state = SystemState.SHUTDOWN
                        else:
                            print(f"[WARNING] Unknown command from LabVIEW: {last_command}")
                            # just ignore it and wait for a valid one
                            last_command = ""
                            pending_command = None
                    time.sleep(STATE_SLEEP_SECONDS["WAIT_FOR_COMMAND"])
                        
                case SystemState.PRESS_BUTTON:
                    print(f"[ACTION] Executing command: {last_command}")
                    # Map the validated LabVIEW command onto one of the physical buttons.
                    button = LABVIEW_BUTTON_COMMANDS.get(pending_command) if pending_command is not None else None
                    if button is None:
                        print(f"[WARNING] Unable to map command to button: {last_command}")
                    else:
                        servo_driver.press_button(button)
                    current_state = SystemState.READ_DISPLAY
                    time.sleep(STATE_SLEEP_SECONDS["PRESS_BUTTON"])
                    
                case SystemState.READ_DISPLAY:
                    print("[ACTION] Reading UI display...")
                    # Capture first, then persist the raw image and run OCR on that same frame.
                    capture_id = data_manager.build_capture_id(last_command)
                    frame = vision_engine.capture_frame()
                    if frame is None:
                        print("[WARNING] No camera frame available; skipping image save.")
                        readout = vision_engine.OCRReadout(
                            display_found=False,
                            current_menu_key="dashboard",
                            fields={"mode": {"raw": "", "value": "UNKNOWN"}, "temperature": {"raw": "", "value": None}},
                        )
                    else:
                        saved_path = data_manager.save_capture_frame(CAPTURE_DIR, frame, capture_id)
                        if saved_path is None:
                            print("[WARNING] Capture frame could not be saved.")
                        else:
                            print(f"[INFO] Saved capture frame: {saved_path}")
                        
                        # LabVIEW commands no longer encode the menu path; default context stays stable.
                        readout = vision_engine.read_display(frame, "dashboard")
                    
                    # Convert the dynamic dictionary of fields into a unified result string
                    fields_str = ";".join(f"{k.upper()}={v.get('value', '')}" for k, v in readout.fields.items())
                    ocr_result = (
                        f"DISPLAY_FOUND={readout.display_found};"
                        f"MENU={readout.current_menu_key};"
                        f"{fields_str}"
                    )
                    step_counter += 1
                    
                    # Extract values for the report safely and dynamically based on the current layout's defined fields.
                    row_data = {"step": step_counter, "menu": readout.current_menu_key}
                    for field_name in CURRENT_LAYOUT.fields:
                        val = readout.fields.get(field_name, {}).get("value")
                        if field_name == "temperature":
                            row_data[field_name] = "" if val is None else int(val)
                        elif field_name == "mode":
                            row_data[field_name] = "UNKNOWN" if val is None else val
                        else:
                            row_data[field_name] = "" if val is None else val
                    
                    report_writer.writerow(row_data)
                    current_state = SystemState.REPORT_TO_LABVIEW
                    time.sleep(STATE_SLEEP_SECONDS["READ_DISPLAY"])
                    
                case SystemState.REPORT_TO_LABVIEW:
                    print(f"[NETWORK] Reporting data to LabVIEW: {ocr_result}")
                    # TODO: send ocr_result back to LabVIEW via Serial or TCP/IP socket
                    # Serial.write(ocr_result)
                    current_state = SystemState.WAIT_FOR_COMMAND
                    time.sleep(STATE_SLEEP_SECONDS["REPORT_TO_LABVIEW"])
                    
                case SystemState.ERROR:
                    print(f"[FATAL] System Faulted: {error_message}")
                    time.sleep(STATE_SLEEP_SECONDS["ERROR"])
                    break
                    
                case SystemState.SHUTDOWN:
                    print("[INFO] LabVIEW requested shutdown. Parking servos, exiting.")
                    report_file.flush()
                    report_file.close()
                    # Leave system in safe state before exit.
                    servo_driver.home_all()
                    servo_driver.shutdown()
                    vision_engine.shutdown()
                    _disable_single_key_mode(stdin_fd, previous_termios)
                    time.sleep(STATE_SLEEP_SECONDS["SHUTDOWN"])
                    break
    
    # alternative for case default - this catches ANY Python crash (divide by zero, camera disconnected, etc.)
    except Exception as e:
        error_message = str(e)
        print("\n[EMERGENCY] Unhandled exception caught!")
        traceback.print_exc() # prints exact line number of crash\

        # Try to park hardware even after unexpected crash.
        report_file.flush()
        report_file.close()
        _disable_single_key_mode(stdin_fd, previous_termios)
        servo_driver.home_all()
        servo_driver.shutdown()
        vision_engine.shutdown()
        
        print("[EMERGENCY] System parked safely. Exiting.")

if __name__ == "__main__":
    main()
