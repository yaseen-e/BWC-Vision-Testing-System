"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/vision/vision_engine.py - Vision/OCR Engine
Captures camera frames, isolates the display region, and extracts mode/temperature via OCR.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
import re
from typing import Any, Optional

import cv2
import numpy as np
import pytesseract


# Expected mode strings from the display.
KNOWN_MODES = [
	"MODE: HYBRID",
	"MODE: HYBRID PLUS",
	"MODE: HEAT PUMP",
	"MODE: ELECTRIC",
	"MODE: VACATION",
]


# Camera object is created lazily so non-Pi environments still run.
_CAMERA: Optional[Any] = None
_CAMERA_NUM = 0


def get_camera_inventory() -> list[dict[str, Any]]:
	"""Return the libcamera-visible camera list for startup diagnostics."""
	try:
		from picamera2 import Picamera2  # type: ignore
	except Exception:
		return []

	try:
		return Picamera2.global_camera_info()
	except Exception:
		return []


@dataclass(frozen=True)
class OCRReadout:
	"""Single OCR result payload for upstream logic (main/network)."""

	display_found: bool
	mode: str
	mode_raw: str
	temperature_f: Optional[float]
	temperature_raw: str


def _order_points(points: np.ndarray) -> np.ndarray:
	"""Normalize corner order so perspective math is stable."""
	ordered = np.zeros((4, 2), dtype="float32")
	point_sums = points.sum(axis=1)
	ordered[0] = points[np.argmin(point_sums)]
	ordered[2] = points[np.argmax(point_sums)]
	point_diffs = np.diff(points, axis=1)
	ordered[1] = points[np.argmin(point_diffs)]
	ordered[3] = points[np.argmax(point_diffs)]
	return ordered


def _four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
	"""Flatten the angled display into a front-facing view."""
	rect = _order_points(points)
	top_left, top_right, bottom_right, bottom_left = rect

	width_a = np.linalg.norm(bottom_right - bottom_left)
	width_b = np.linalg.norm(top_right - top_left)
	max_width = max(int(width_a), int(width_b))

	height_a = np.linalg.norm(top_right - bottom_right)
	height_b = np.linalg.norm(top_left - bottom_left)
	max_height = max(int(height_a), int(height_b))

	dst = np.array(
		[
			[0, 0],
			[max_width - 1, 0],
			[max_width - 1, max_height - 1],
			[0, max_height - 1],
		],
		dtype="float32",
	)

	transform = cv2.getPerspectiveTransform(rect, dst)
	return cv2.warpPerspective(image, transform, (max_width, max_height))


def _display_mask(frame: np.ndarray) -> np.ndarray:
	"""Keep only the orange display area to reduce OCR noise."""
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower_orange = np.array([5, 150, 150])
	upper_orange = np.array([25, 255, 255])
	mask = cv2.inRange(hsv, lower_orange, upper_orange)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	return mask


def _find_display_contour(mask: np.ndarray, min_area: int = 3000) -> Optional[np.ndarray]:
	"""Find the largest 4-corner shape that looks like the LCD window."""
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	best_contour = None
	best_area = 0.0
	for contour in contours:
		area = cv2.contourArea(contour)
		if area < min_area:
			continue

		perimeter = cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
		if len(approx) == 4 and area > best_area:
			best_contour = approx
			best_area = area

	return best_contour


def _prepare_binary(warped: np.ndarray) -> np.ndarray:
	"""Convert to a high-contrast black/white image for OCR."""
	gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
	_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return thresh


def _extract_mode_roi(binary: np.ndarray) -> np.ndarray:
	"""Crop the top line where mode text appears."""
	height, width = binary.shape
	mode_roi = binary[int(height * 0.00):int(height * 0.14), int(width * 0.20):int(width * 0.80)]
	mode_roi = cv2.bitwise_not(mode_roi)
	return cv2.resize(mode_roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)


def _extract_temp_roi(binary: np.ndarray) -> np.ndarray:
	"""Crop the center section where numeric temperature appears."""
	height, width = binary.shape
	temp_roi = binary[int(height * 0.16):int(height * 0.48), int(width * 0.28):int(width * 0.63)]
	temp_roi = cv2.bitwise_not(temp_roi)

	# Small close operation helps connect broken strokes in digits.
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	temp_roi = cv2.morphologyEx(temp_roi, cv2.MORPH_CLOSE, kernel)
	return cv2.resize(temp_roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)


def _read_mode(mode_roi: np.ndarray) -> tuple[str, str]:
	"""Run OCR on mode text and map it to the nearest known mode."""
	config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ: "
	raw = pytesseract.image_to_string(mode_roi, config=config).strip().upper()
	return raw, parse_mode_text(raw)


def _read_temperature(temp_roi: np.ndarray) -> tuple[str, Optional[float]]:
	"""Run OCR on numeric text and parse a float when possible."""
	config = "--psm 8 -c tessedit_char_whitelist=0123456789."
	raw = pytesseract.image_to_string(temp_roi, config=config).strip()
	return raw, parse_temperature_text(raw)


def parse_mode_text(raw_mode: str) -> str:
	"""Map free-form OCR text to the closest expected mode string."""
	if not raw_mode:
		return "UNKNOWN"

	normalized = raw_mode.strip().upper()
	if not normalized:
		return "UNKNOWN"

	# Keep cutoff high so unrelated text does not map to a real mode.
	match = get_close_matches(normalized, KNOWN_MODES, n=1, cutoff=0.75)
	return match[0] if match else "UNKNOWN"


def parse_temperature_text(raw_temp: str) -> Optional[float]:
	"""Extract numeric temperature from OCR text when available."""
	if not raw_temp:
		return None

	# Replace common OCR confusion before numeric parsing.
	normalized = raw_temp.replace("O", "0").replace("o", "0")
	match = re.search(r"\d+(?:\.\d+)?", normalized)
	return float(match.group()) if match else None


def read_display(
	frame: np.ndarray,
) -> OCRReadout:
	"""
	Read both display mode and temperature from one frame.

	Args:
		frame: BGR image from camera.

	Returns:
		OCRReadout with mode text + temperature float.
	"""
	mask = _display_mask(frame)
	display_contour = _find_display_contour(mask)

	if display_contour is None:
		return OCRReadout(
			display_found=False,
			mode="UNKNOWN",
			mode_raw="",
			temperature_f=None,
			temperature_raw="",
		)

	warped = _four_point_transform(frame, display_contour.reshape(4, 2))
	binary = _prepare_binary(warped)

	mode_roi = _extract_mode_roi(binary)
	mode_raw, mode = _read_mode(mode_roi)

	temp_roi = _extract_temp_roi(binary)
	temp_raw, temp_value = _read_temperature(temp_roi)

	return OCRReadout(
		display_found=True,
		mode=mode,
		mode_raw=mode_raw,
		temperature_f=temp_value,
		temperature_raw=temp_raw,
	)


def capture_frame() -> Optional[np.ndarray]:
	"""Capture one camera frame (returns None when camera is unavailable)."""
	camera = _get_camera()
	if camera is None:
		return None

	return camera.capture_array()


def capture_and_read_display(
) -> OCRReadout:
	"""One-call helper used by main: capture frame then run OCR pipeline."""
	frame = capture_frame()
	if frame is None:
		return OCRReadout(
			display_found=False,
			mode="UNKNOWN",
			mode_raw="",
			temperature_f=None,
			temperature_raw="",
		)

	return read_display(
		frame,
	)


def warm_up() -> None:
	"""Hook for camera warmup work (kept for main loop integration)."""
	_get_camera()


def is_camera_available() -> bool:
	"""Return True when the Pi camera can be initialized successfully."""
	return len(get_camera_inventory()) > _CAMERA_NUM


def _get_camera() -> Optional[Any]:
	"""Initialize camera on first use, but stay safe on non-Pi systems."""
	global _CAMERA
	if _CAMERA is not None:
		return _CAMERA

	try:
		from picamera2 import Picamera2  # type: ignore
	except Exception:
		return None

	try:
		camera = Picamera2(camera_num=_CAMERA_NUM)
		config = camera.create_preview_configuration(main={"size": (1280, 720)})
		camera.configure(config)
		camera.start()
		camera.set_controls({"AfMode": 2})
		_CAMERA = camera
	except Exception:
		return None

	return _CAMERA


def shutdown() -> None:
	"""Hook for camera cleanup work (kept for main loop integration)."""
	global _CAMERA
	if _CAMERA is None:
		return

	try:
		_CAMERA.stop()
	except Exception:
		pass

	_CAMERA = None
