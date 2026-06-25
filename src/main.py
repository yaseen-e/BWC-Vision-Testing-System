"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/main.py - Main Event Loop
Coordinates startup, command handling, actuation, OCR reads, reporting, and shutdown.
"""

from pathlib import Path
from enum import Enum, auto
import time
import traceback

from src import data_manager
from src.motion import servo_driver
from src.vision import vision_engine
from src.vision.display_layouts import CURRENT_LAYOUT

from src.network.labview_protocol import LabViewCommand, parse_labview_command
from src.network import labview_tcp
from src.network import report_writer
from src.network import terminal_control_temp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_DIR = PROJECT_ROOT / "data" / "captures"
CLEANUP_RETENTION_DAYS = 14


class SystemState(Enum):
    STARTUP = auto()
    WAIT_FOR_COMMAND = auto()
    PRESS_BUTTON = auto()
    READ_DISPLAY = auto()
    REPORT_TO_LABVIEW = auto()
    ERROR = auto()
    SHUTDOWN = auto()


LABVIEW_BUTTON_COMMANDS = {
    LabViewCommand.UP: servo_driver.Button.UP,
    LabViewCommand.LEFT: servo_driver.Button.LEFT,
    LabViewCommand.SELECT: servo_driver.Button.SELECT,
    LabViewCommand.RIGHT: servo_driver.Button.RIGHT,
    LabViewCommand.BACK: servo_driver.Button.BACK,
    LabViewCommand.DOWN: servo_driver.Button.DOWN,
    LabViewCommand.MENU: servo_driver.Button.MENU,
}


def get_button_for_command(command: LabViewCommand) -> servo_driver.Button | None:
    return LABVIEW_BUTTON_COMMANDS.get(command)


def _prompt_for_command_mode() -> bool:
    """Ask whether to use simulated commands or a real LabVIEW connection."""
    while True:
        try:
            answer = input("[PROMPT] Use simulated commands? [y/N]: ").strip().lower()
        except EOFError:
            print("[WARNING] No input available; defaulting to real LabVIEW connection.")
            return False

        if answer in ("y", "yes", "sim", "simulated"):
            print("[INFO] Simulated command mode enabled.")
            return True

        if answer in ("", "n", "no", "real", "labview"):
            print("[INFO] Real LabVIEW connection mode enabled.")
            return False

        print("[WARNING] Invalid selection. Enter 'y' for simulated or 'n' for LabVIEW.")

def _run_capture_cleanup() -> None:
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


def _safe_servo_home_all() -> None:
    """Attempt to park servos without allowing hardware errors to crash shutdown."""
    try:
        servo_driver.home_all()
    except Exception as exc:
        print(f"[WARNING] Servo home_all skipped: {exc}")


def _safe_servo_shutdown() -> None:
    """Attempt servo shutdown even when hardware is missing or unavailable."""
    try:
        servo_driver.shutdown()
    except Exception as exc:
        print(f"[WARNING] Servo shutdown skipped: {exc}")

def main():
    current_state = SystemState.STARTUP
    last_command = ""
    pending_command = None
    ocr_result = ""
    error_message = ""
    step_counter = 0
    use_simulated_commands = _prompt_for_command_mode()

    report_file, report_csv_writer, report_path = report_writer.open_test_report()
    stdin_fd, previous_termios = terminal_control_temp.enable_single_key_mode()
    
    print("--- Starting BWC Water Heater Vision Testing System ---")
    print(f"[INFO] Test report CSV: {report_path}")

    try:
        while True:
            if current_state != SystemState.SHUTDOWN and terminal_control_temp.space_pressed():
                print("[INFO] Space pressed. Entering SHUTDOWN state.")
                current_state = SystemState.SHUTDOWN

            match current_state:
                case SystemState.STARTUP:
                    print("[INFO] Cleaning up old captures.")
                    _run_capture_cleanup()
                    print("[INFO] Initializing servos.")
                    servo_driver.initialize()
                    servo_driver.home_all()
                    print("[INFO] Initializing vision engine.")
                    if vision_engine.is_camera_available():
                        print("[INFO] Camera detected.")
                    else:
                        print("[WARNING] Camera NOT found.")
                    if use_simulated_commands:
                        print("[NETWORK] Simulated command mode active; skipping TCP server startup.")
                    else:
                        if labview_tcp.start_tcp_server():
                            print("[NETWORK] TCP server started successfully.")
                        else:
                            print("[WARNING] TCP server startup failed; continuing to run.")
                    current_state = SystemState.WAIT_FOR_COMMAND
                    
                case SystemState.WAIT_FOR_COMMAND:
                    last_command = labview_tcp.get_next_command(simulated=use_simulated_commands)
                    if last_command:
                        print(f"[NETWORK] Received command from LabVIEW: {last_command}")
                    else:
                        if use_simulated_commands:
                            print("[SIMULATION] Waiting for next simulated command...")
                        else:
                            print("[NETWORK] Waiting for LabVIEW command...")
                        time.sleep(1)  # Avoid busy waiting
                    if last_command:
                        pending_command = parse_labview_command(last_command)
                        if pending_command is not None:
                            if get_button_for_command(pending_command) is not None:
                                current_state = SystemState.PRESS_BUTTON
                            elif pending_command is LabViewCommand.RUN_OCR:
                                current_state = SystemState.READ_DISPLAY
                            elif pending_command is LabViewCommand.SHUTDOWN:
                                current_state = SystemState.SHUTDOWN
                        else:
                            print(f"[WARNING] Unknown command from LabVIEW: {last_command}")
                            last_command = ""
                            pending_command = None
                    pass
                        
                case SystemState.PRESS_BUTTON:
                    print(f"[ACTION] Executing command: {last_command}")
                    button = get_button_for_command(pending_command) if pending_command is not None else None
                    if button is None:
                        print(f"[WARNING] Unable to map command to button: {last_command}")
                    else:
                        print(f"[MANUAL] Please press the {button.name} button on the water heater.")
                        if use_simulated_commands:
                            terminal_control_temp.wait_for_enter()
                        else:
                            servo_driver.press_button(button)

                    current_state = SystemState.WAIT_FOR_COMMAND
                    
                case SystemState.READ_DISPLAY:
                    print("[ACTION] Reading UI display...")
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

                        roi_overlay_path = vision_engine.save_roi_ocr_overlay(CAPTURE_DIR, frame, capture_id)
                        if roi_overlay_path is None:
                            print("[WARNING] ROI calibration image could not be saved.")
                        else:
                            print(f"[INFO] Saved ROI calibration image: {roi_overlay_path}")
                        
                        readout = vision_engine.read_display(frame, "dashboard")
                    
                    fields_str = ";".join(f"{k.upper()}={v.get('value', '')}" for k, v in readout.fields.items())
                    ocr_result = (
                        f"DISPLAY_FOUND={readout.display_found};"
                        f"MENU={readout.current_menu_key};"
                        f"{fields_str}"
                    )
                    step_counter += 1
                    
                    row_data = {"step": step_counter, "menu": readout.current_menu_key}
                    for field_name in CURRENT_LAYOUT.fields:
                        val = readout.fields.get(field_name, {}).get("value")
                        if field_name == "temperature":
                            if val is None or val == "":
                                row_data[field_name] = ""
                            else:
                                try:
                                    row_data[field_name] = int(val)
                                except (TypeError, ValueError):
                                    print(f"[WARNING] Invalid temperature value from OCR: {val}")
                                    row_data[field_name] = ""
                        elif field_name == "mode":
                            row_data[field_name] = "UNKNOWN" if val is None else val
                        else:
                            row_data[field_name] = "" if val is None else val
                    
                    report_csv_writer.writerow(row_data)
                    current_state = SystemState.REPORT_TO_LABVIEW
                    
                case SystemState.REPORT_TO_LABVIEW:
                    print(f"[NETWORK] Reporting data to LabVIEW: {ocr_result}")
                    labview_tcp.send_report(ocr_result, simulated=use_simulated_commands)
                    current_state = SystemState.WAIT_FOR_COMMAND
                    
                case SystemState.ERROR:
                    print(f"[FATAL] System Faulted: {error_message}")
                    break
                    
                case SystemState.SHUTDOWN:
                    print("[INFO] LabVIEW requested shutdown. Parking servos, exiting.")
                    report_file.flush()
                    report_file.close()
                    _safe_servo_home_all()
                    _safe_servo_shutdown()
                    vision_engine.shutdown()
                    terminal_control_temp.disable_single_key_mode(stdin_fd, previous_termios)
                    break
    
    except Exception as e:
        error_message = str(e)
        print("\n[EMERGENCY] Unhandled exception caught!")
        traceback.print_exc()

        report_file.flush()
        report_file.close()
        terminal_control_temp.disable_single_key_mode(stdin_fd, previous_termios)
        _safe_servo_home_all()
        _safe_servo_shutdown()
        vision_engine.shutdown()
        
        print("[EMERGENCY] System parked safely. Exiting.")

if __name__ == "__main__":
    main()
