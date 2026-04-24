"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/main.py - Main Event Loop
Coordinates startup, command handling, actuation, OCR reads, reporting, and shutdown.
"""

from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time
import traceback
from enum import Enum, auto

from motion import servo_driver
from vision import vision_engine


# Capture storage layout (repo-relative): data/captures
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIR = PROJECT_ROOT / "data" / "captures"
CLEANUP_SCRIPT = PROJECT_ROOT / "utilities" / "rotate_capture_cleanup.py"
CLEANUP_RETENTION_DAYS = 14


# Small waits per loop reduce CPU usage while keeping response time fast.
STATE_SLEEP_SECONDS = {
    "STARTUP": 0.20,
    "WAIT_FOR_COMMAND": 0.25,
    "PRESS_BUTTON": 0.20,
    "READ_DISPLAY": 0.20,
    "REPORT_TO_LABVIEW": 0.20,
    "ERROR": 0.20,
    "SHUTDOWN": 0.10,
}

class SystemState(Enum):
    STARTUP = auto()
    WAIT_FOR_COMMAND = auto()
    PRESS_BUTTON = auto()
    READ_DISPLAY = auto()
    REPORT_TO_LABVIEW = auto()
    ERROR = auto()
    SHUTDOWN = auto()


def _build_capture_id(command_text: str) -> str:
    """Create a stable, filename-safe ID for one OCR attempt."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    command_tag = "".join(char if char.isalnum() else "_" for char in command_text.lower()).strip("_")
    if not command_tag:
        command_tag = "read"
    return f"{timestamp}_{command_tag}"


def _run_capture_cleanup() -> None:
    """Run retention cleanup in apply mode; failures do not stop system startup."""
    if not CLEANUP_SCRIPT.exists():
        print(f"[WARNING] Cleanup script not found: {CLEANUP_SCRIPT}")
        return

    command = [
        sys.executable,
        str(CLEANUP_SCRIPT),
        "--retention-days",
        str(CLEANUP_RETENTION_DAYS),
        "--apply",
    ]

    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[INFO] Capture cleanup applied (retention={CLEANUP_RETENTION_DAYS} days).")
        else:
            print("[WARNING] Capture cleanup failed; continuing startup.")
    except Exception as exc:
        print(f"[WARNING] Capture cleanup error: {exc}")

def main():
    """Main Event Loop (The Orchestrator)"""
    
    # initial state on startup
    current_state = SystemState.STARTUP
    last_command = ""
    ocr_result = ""
    error_message = ""
    readout = None
    
    print("--- Starting BWC Water Heater Vision Testing System ---")

    try:
        while True:
            match current_state:
                
                case SystemState.STARTUP:
                    print("[INFO] Homing servos, warming up camera...")
                    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
                    _run_capture_cleanup()
                    # Bring hardware to a known state before first command.
                    servo_driver.initialize()
                    servo_driver.home_all()
                    vision_engine.warm_up()
                    current_state = SystemState.WAIT_FOR_COMMAND
                    time.sleep(STATE_SLEEP_SECONDS["STARTUP"])
                    
                case SystemState.WAIT_FOR_COMMAND:
                    last_command = "CMD_PRESS_TEMP_UP" # simulated command received from LabVIEW
                    print(f"[NETWORK] Received command from LabVIEW: {last_command}")
                    # TODO: listen to LabVIEW via Serial or TCP/IP socket      
                    if last_command: # string is true if not empty = command received
                        # --- COMMAND ROUTING LOGIC ---
                        if last_command.startswith("CMD_PRESS"):
                            current_state = SystemState.PRESS_BUTTON
                        elif last_command == "CMD_READ_ONLY":
                            current_state = SystemState.READ_DISPLAY
                        elif last_command == "CMD_SHUTDOWN":
                            current_state = SystemState.SHUTDOWN
                        else:
                            print(f"[WARNING] Unknown command from LabVIEW: {last_command}")
                            # just ignore it and wait for a valid one
                            last_command = ""
                    time.sleep(STATE_SLEEP_SECONDS["WAIT_FOR_COMMAND"])
                        
                case SystemState.PRESS_BUTTON:
                    print(f"[ACTION] Executing command: {last_command}")
                    # Parse command text into one of the seven physical buttons.
                    button = servo_driver.parse_button_from_command(last_command)
                    if button is None:
                        print(f"[WARNING] Unable to map command to button: {last_command}")
                    else:
                        servo_driver.press_button(button)
                    current_state = SystemState.READ_DISPLAY
                    time.sleep(STATE_SLEEP_SECONDS["PRESS_BUTTON"])
                    
                case SystemState.READ_DISPLAY:
                    print("[ACTION] Reading UI display...")
                    # Read both mode and temperature from one capture.
                    capture_id = _build_capture_id(last_command)
                    readout = vision_engine.capture_and_read_display(
                        capture_dir=str(CAPTURE_DIR),
                        capture_id=capture_id,
                    )
                    if not readout.display_found:
                        ocr_result = "DISPLAY_NOT_FOUND"
                    else:
                        ocr_result = (
                            f"MODE={readout.mode};"
                            f"TEMP_F={readout.temperature_f};"
                            f"RAW_MODE={readout.mode_raw};"
                            f"RAW_TEMP={readout.temperature_raw}"
                        )
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
                    # Leave system in safe state before exit.
                    servo_driver.home_all()
                    servo_driver.shutdown()
                    vision_engine.shutdown()
                    time.sleep(STATE_SLEEP_SECONDS["SHUTDOWN"])
                    break
    
    # alternative for case default - this catches ANY Python crash (divide by zero, camera disconnected, etc.)
    except Exception as e:
        error_message = str(e)
        print("\n[EMERGENCY] Unhandled exception caught!")
        traceback.print_exc() # prints exact line number of crash\

        # Try to park hardware even after unexpected crash.
        servo_driver.home_all()
        servo_driver.shutdown()
        vision_engine.shutdown()
        
        print("[EMERGENCY] System parked safely. Exiting.")

if __name__ == "__main__":
    main()
