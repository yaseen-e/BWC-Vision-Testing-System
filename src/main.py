"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/main.py - Main Event Loop
Coordinates startup, command handling, actuation, OCR reads, reporting, and shutdown.
"""
from __future__ import annotations

import os
os.environ["ORT_LOGGING_LEVEL"] = "3"  # Suppress ONNX Runtime warnings before engine imports

import csv
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import select
import sys
import time
import traceback
from typing import Any

from src.motion import servos
from src.network import labview
from src.network.labview import LabViewCommand, parse_labview_command
from src.storage import captures, reports
from src.vision import engine
from src.vision.layouts import (
	HOME_MENU,
	ContextNode,
	apply_navigation_command,
)

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_DIR = PROJECT_ROOT / "data" / "captures"
CLEANUP_RETENTION_DAYS = 14


# =============================================================================
# SYSTEM STATE & CONTEXT MODELS
# =============================================================================

class SystemState(Enum):
	STARTUP = auto()
	WAIT_FOR_COMMAND = auto()
	PRESS_BUTTON = auto()
	READ_DISPLAY = auto()
	REPORT_TO_LABVIEW = auto()
	ERROR = auto()
	SHUTDOWN = auto()


LABVIEW_BUTTON_COMMANDS = {
	LabViewCommand.UP: servos.Button.UP,
	LabViewCommand.LEFT: servos.Button.LEFT,
	LabViewCommand.SELECT: servos.Button.SELECT,
	LabViewCommand.RIGHT: servos.Button.RIGHT,
	LabViewCommand.BACK: servos.Button.BACK,
	LabViewCommand.DOWN: servos.Button.DOWN,
	LabViewCommand.MENU: servos.Button.MENU,
}


@dataclass
class SystemContext:
	"""Encapsulates runtime state and navigation tracking across the test loop."""

	use_simulated: bool
	current_state: SystemState = SystemState.STARTUP
	last_command: str = ""
	pending_command: LabViewCommand | None = None
	ocr_result: str = ""
	step_counter: int = 0
	current_menu: ContextNode = HOME_MENU
	transition_buffer: tuple[str, ...] = ()
	sequence_broken: bool = False
	press_ready: list[int] = field(default_factory=lambda: [1])


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> None:
	"""Main system execution entry point."""
	# use_simulated = _prompt_for_command_mode()
	use_simulated = False
	ctx = SystemContext(use_simulated=use_simulated)
	report_file, report_writer, report_path = reports.open_test_report()

	print("--- Starting BWC Water Heater Vision Testing System ---")
	print(f"[INFO] Test report CSV: {report_path}")
	print(f"[INFO] Initial menu context: {ctx.current_menu.label}")

	try:
		while ctx.current_state != SystemState.SHUTDOWN:
			if _space_pressed():
				print("[INFO] Space pressed. Entering SHUTDOWN state.")
				ctx.current_state = SystemState.SHUTDOWN
				break

			_execute_state_step(ctx, report_writer)

	except Exception:
		print("\n[EMERGENCY] Unhandled exception caught!")
		traceback.print_exc()
	finally:
		_safe_system_cleanup(report_file)
		print("[INFO] System resources freed and parked safely. Exiting.")


# =============================================================================
# HIGH-LEVEL STATE DISPATCHER
# =============================================================================

def _execute_state_step(ctx: SystemContext, report_writer: csv.DictWriter) -> None:
	"""Dispatch execution to the appropriate step handler based on current state."""
	match ctx.current_state:
		case SystemState.STARTUP:
			_handle_startup(ctx)
		case SystemState.WAIT_FOR_COMMAND:
			_handle_wait_for_command(ctx)
		case SystemState.PRESS_BUTTON:
			_handle_press_button(ctx)
		case SystemState.READ_DISPLAY:
			_handle_read_display(ctx, report_writer)
		case SystemState.REPORT_TO_LABVIEW:
			_handle_report_to_labview(ctx)
		case SystemState.ERROR | SystemState.SHUTDOWN:
			pass


# =============================================================================
# INDIVIDUAL STATE HANDLERS
# =============================================================================

def _handle_startup(ctx: SystemContext) -> None:
	"""Initialize hardware drivers, camera vision engine, and network server."""
	print("[INFO] Cleaning up old captures.")
	_run_capture_cleanup()

	print("[INFO] Initializing servos.")
	servos.initialize()
	servos.home_all()

	print("[INFO] Initializing vision engine.")
	if engine.is_camera_available():
		print("[INFO] Camera detected.")
	else:
		print("[WARNING] Camera NOT found.")

	if ctx.use_simulated:
		print("[NETWORK] Simulated command mode active; skipping TCP server startup.")
	else:
		if labview.start_tcp_server():
			print("[NETWORK] TCP server started successfully.")
		else:
			print("[WARNING] TCP server startup failed; continuing to run.")

	ctx.current_state = SystemState.WAIT_FOR_COMMAND


def _handle_wait_for_command(ctx: SystemContext) -> None:
	"""Fetch and parse the next inbound command from TCP or simulation queue."""
	ctx.last_command = labview.get_next_command(simulated=ctx.use_simulated)

	if not ctx.last_command:
		mode_str = "simulated " if ctx.use_simulated else "LabVIEW "
		print(f"[NETWORK] Waiting for next {mode_str}command...")
		time.sleep(1)
		return

	print(f"[NETWORK] Received command: {ctx.last_command}")
	ctx.pending_command = parse_labview_command(ctx.last_command)

	if ctx.pending_command is None:
		print(f"[WARNING] Unknown command from LabVIEW: {ctx.last_command}")
		ctx.last_command = ""
		return

	if get_button_for_command(ctx.pending_command) is not None:
		ctx.current_state = SystemState.PRESS_BUTTON
	elif ctx.pending_command is LabViewCommand.RUN_OCR:
		ctx.transition_buffer = ()
		ctx.sequence_broken = False
		ctx.current_state = SystemState.READ_DISPLAY
	elif ctx.pending_command is LabViewCommand.SHUTDOWN:
		ctx.current_state = SystemState.SHUTDOWN


def _handle_press_button(ctx: SystemContext) -> None:
	"""Actuate the mapped button servo and update expected UI menu context."""
	print(f"[ACTION] Executing command: {ctx.last_command}")
	button = get_button_for_command(ctx.pending_command)

	if button is None:
		print(f"[WARNING] Unable to map command to button: {ctx.last_command}")
		ctx.current_state = SystemState.WAIT_FOR_COMMAND
		return

	_actuate_button(button, ctx.use_simulated, ctx.press_ready)
	_update_menu_navigation(ctx)
	ctx.current_state = SystemState.WAIT_FOR_COMMAND


def _handle_read_display(ctx: SystemContext, report_writer: csv.DictWriter) -> None:
	"""Capture frame, execute OCR read, format response, and record CSV report row."""
	print("[ACTION] Reading UI display...")
	capture_id = captures.build_capture_id(ctx.last_command)
	readout = _capture_and_process_frame(ctx.current_menu, capture_id)

	ctx.ocr_result = _format_ocr_result(readout, ctx.current_menu)
	ctx.step_counter += 1

	_write_step_to_report(report_writer, ctx.step_counter, ctx.current_menu, readout)
	ctx.current_state = SystemState.REPORT_TO_LABVIEW


def _handle_report_to_labview(ctx: SystemContext) -> None:
	"""Transmit formatted OCR results back to LabVIEW or simulation log."""
	print(f"[NETWORK] Reporting data to LabVIEW: {ctx.ocr_result}")
	labview.send_report(ctx.ocr_result, simulated=ctx.use_simulated)
	ctx.current_state = SystemState.WAIT_FOR_COMMAND


# =============================================================================
# DOMAIN HELPER UTILITIES
# =============================================================================

def get_button_for_command(command: LabViewCommand | None) -> servos.Button | None:
	"""Map LabVIEW command enum to physical servo driver button."""
	if command is None:
		return None
	return LABVIEW_BUTTON_COMMANDS.get(command)


def _actuate_button(
	button: servos.Button, use_simulated: bool, press_ready: list[int]
) -> None:
	"""Execute physical servo press or wait for user manual enter key."""
	print(f"[MANUAL] Please press the {button.name} button on the water heater.")
	if use_simulated:
		wait_for_enter()
	else:
		servos.press_button(button, press_ready)

	while press_ready[0] != 1:
		time.sleep(0.01)


def _update_menu_navigation(ctx: SystemContext) -> None:
	"""Update navigation state tracking based on button press."""
	if ctx.sequence_broken:
		print("[NAV] Sequence locked after mismatch; waiting for RUN_OCR reset.")
		return

	previous_menu_key = ctx.current_menu.key
	cmd_val = ctx.pending_command.value if ctx.pending_command else ""

	(
		ctx.current_menu,
		ctx.transition_buffer,
		ctx.sequence_broken,
	) = apply_navigation_command(
		HOME_MENU,
		ctx.current_menu,
		ctx.transition_buffer,
		cmd_val,
	)

	if ctx.sequence_broken:
		print(
			f"[NAV] Sequence mismatch on {ctx.transition_buffer}; "
			"route tracking locked until RUN_OCR."
		)
	elif ctx.current_menu.key != previous_menu_key:
		print(f"[NAV] Current menu updated: {ctx.current_menu.label}")
	elif ctx.transition_buffer:
		print(f"[NAV] Partial sequence: {ctx.transition_buffer}")


def _capture_and_process_frame(
	current_menu: ContextNode, capture_id: str
) -> engine.OCRReadout:
	"""Capture frame from camera and execute OCR engine evaluation."""
	frame = engine.capture_frame()
	if frame is None:
		print("[WARNING] No camera frame available; skipping image save.")
		return _build_empty_readout(current_menu)

	saved_path = captures.save_capture_frame(CAPTURE_DIR, frame, capture_id)
	if saved_path:
		print(f"[INFO] Saved capture frame: {saved_path}")

	roi_overlay_path = engine.save_roi_ocr_overlay(
		CAPTURE_DIR,
		frame,
		capture_id,
		current_menu.fields,
		current_menu_key=current_menu.key,
	)
	if roi_overlay_path:
		print(f"[INFO] Saved ROI calibration image: {roi_overlay_path}")

	return engine.read_display(frame, current_menu.key, current_menu.fields)


def _build_empty_readout(current_menu: ContextNode) -> engine.OCRReadout:
	"""Construct empty readout payload when camera frame is unavailable."""
	empty_fields = {}
	for field_item in current_menu.fields:
		default_val = (
			"UNKNOWN"
			if field_item.name == "mode"
			else None
			if field_item.name == "temperature"
			else ""
		)
		empty_fields[field_item.name] = {"raw": "", "value": default_val}

	return engine.OCRReadout(
		display_found=False,
		current_menu_key=current_menu.key,
		fields=empty_fields,
	)


def _format_ocr_result(
	readout: engine.OCRReadout, current_menu: ContextNode
) -> str:
	"""Format OCR readout into semicolon-delimited string payload for LabVIEW."""
	field_pairs: list[str] = []
	for field_item in current_menu.fields:
		raw_val = readout.fields.get(field_item.name, {}).get("value", "")
		val_str = "" if raw_val is None else str(raw_val)
		field_pairs.append(f"{field_item.name.upper()}={val_str}")

	fields_str = ";".join(field_pairs)
	base_str = f"DISPLAY_FOUND={readout.display_found};MENU={current_menu.label}"
	return f"{base_str};{fields_str}" if fields_str else base_str


def _write_step_to_report(
	writer: csv.DictWriter,
	step: int,
	menu: ContextNode,
	readout: engine.OCRReadout,
) -> None:
	"""Format step metadata and write row entry to CSV test report."""
	row_data = {"step": step, "menu": menu.label}
	for name in reports.get_report_field_names():
		row_data[name] = ""

	for field_item in menu.fields:
		field_name = field_item.name
		val = readout.fields.get(field_name, {}).get("value")

		if field_name == "temperature":
			row_data[field_name] = _parse_report_temperature(val)
		elif field_name == "mode":
			row_data[field_name] = "UNKNOWN" if val is None else val
		else:
			row_data[field_name] = "" if val is None else val

	writer.writerow(row_data)


def _parse_report_temperature(val: Any) -> int | str:
	"""Safely parse temperature raw string into integer for report writer."""
	if val is None or val == "":
		return ""
	try:
		return int(val)
	except (TypeError, ValueError):
		print(f"[WARNING] Invalid temperature value from OCR: {val}")
		return ""


# =============================================================================
# CLI & HARDWARE CLEANUP UTILITIES
# =============================================================================

def _prompt_for_command_mode() -> bool:
	"""Ask user whether to run simulated commands or await TCP connection."""
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
	"""Delete capture image assets older than retention threshold."""
	CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
	try:
		captures.cleanup_captures(
			captures_dir=CAPTURE_DIR,
			retention_days=CLEANUP_RETENTION_DAYS,
			apply_changes=True,
			allowed_root=CAPTURE_DIR,
		)
		print(f"[INFO] Capture cleanup applied (retention={CLEANUP_RETENTION_DAYS} days).")
	except Exception as exc:
		print(f"[WARNING] Capture cleanup error: {exc}")


def _safe_system_cleanup(report_file: Any = None) -> None:
	"""Safely flush report files, park hardware servos, and shutdown camera."""
	if report_file and not report_file.closed:
		try:
			report_file.flush()
			report_file.close()
		except Exception as exc:
			print(f"[WARNING] Report file cleanup error: {exc}")

	try:
		servos.home_all()
	except Exception as exc:
		print(f"[WARNING] Servo home_all skipped: {exc}")

	try:
		servos.shutdown()
	except Exception as exc:
		print(f"[WARNING] Servo shutdown skipped: {exc}")

	try:
		engine.shutdown()
	except Exception as exc:
		print(f"[WARNING] Vision engine shutdown skipped: {exc}")


def wait_for_enter() -> None:
	"""Block until user hits ENTER key in terminal interactive mode."""
	if not sys.stdin.isatty():
		return
	print("Press ENTER to continue...", end="", flush=True)
	while True:
		char = sys.stdin.read(1)
		if char in ("\n", "\r"):
			print()
			break


def _space_pressed() -> bool:
	"""Non-blocking check for spacebar key waiting on stdin."""
	if not sys.stdin.isatty():
		return False

	ready, _, _ = select.select([sys.stdin], [], [], 0)
	if not ready:
		return False

	return sys.stdin.read(1) == " "


if __name__ == "__main__":
	main()
