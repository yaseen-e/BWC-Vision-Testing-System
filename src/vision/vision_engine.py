"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/vision/vision_engine.py - Vision/OCR Engine
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
import re
from typing import Optional

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
	if not raw:
		return "", "UNKNOWN"

	match = get_close_matches(raw, KNOWN_MODES, n=1, cutoff=0.5)
	final_mode = match[0] if match else "UNKNOWN"
	return raw, final_mode


def _read_temperature(temp_roi: np.ndarray) -> tuple[str, Optional[float]]:
	"""Run OCR on numeric text and parse a float when possible."""
	config = "--psm 8 -c tessedit_char_whitelist=0123456789."
	raw = pytesseract.image_to_string(temp_roi, config=config).strip()
	if not raw:
		return "", None

	# Replace common OCR confusion before numeric parsing.
	normalized = raw.replace("O", "0").replace("o", "0")
	match = re.search(r"\d+(?:\.\d+)?", normalized)
	return raw, float(match.group()) if match else None


def read_display(frame: np.ndarray, debug_dir: Optional[str] = None) -> OCRReadout:
	"""
	Read both display mode and temperature from one frame.

	Args:
		frame: BGR image from camera.
		debug_dir: Optional folder for saved debug images.

	Returns:
		OCRReadout with mode text + temperature float.
	"""
	mask = _display_mask(frame)
	display_contour = _find_display_contour(mask)

	if display_contour is None:
		_save_debug(debug_dir, frame=frame, mask=mask)
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

	_save_debug(
		debug_dir,
		frame=frame,
		mask=mask,
		warped=warped,
		binary=binary,
		mode_roi=mode_roi,
		temp_roi=temp_roi,
	)

	return OCRReadout(
		display_found=True,
		mode=mode,
		mode_raw=mode_raw,
		temperature_f=temp_value,
		temperature_raw=temp_raw,
	)


def _save_debug(debug_dir: Optional[str], **images: np.ndarray) -> None:
	"""Write intermediate images only when debug output is requested."""
	if not debug_dir:
		return

	output_dir = Path(debug_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	for name, image in images.items():
		cv2.imwrite(str(output_dir / f"{name}.jpg"), image)


def warm_up() -> None:
	"""Hook for camera warmup work (kept for main loop integration)."""
	return


def shutdown() -> None:
	"""Hook for camera cleanup work (kept for main loop integration)."""
	return
