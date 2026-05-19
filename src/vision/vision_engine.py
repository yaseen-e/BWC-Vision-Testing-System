"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/vision/vision_engine.py - Vision/OCR Engine
Captures camera frames, isolates the display region, and extracts mode/temperature via OCR.
"""
# TODO: remove try/except imports and add explicit requirements once dependencies are finalized.
from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
import re
from typing import Any, Optional

try:
	import numpy as np
except Exception:  # pragma: no cover - environment dependent
	np = None

try:
	import cv2
except Exception:  # pragma: no cover - environment dependent
	cv2 = None

try:
	import pytesseract
except Exception:  # pragma: no cover - environment dependent
	pytesseract = None

from .display_layouts import CURRENT_LAYOUT


# Camera object is created lazily so non-Pi environments still run.
_CAMERA: Optional[Any] = None


@dataclass(frozen=True)
class OCRReadout:
	"""Single OCR result payload for upstream logic (main/network)."""

	display_found: bool
	mode: str
	mode_raw: str
	temperature_f: Optional[int]
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


def _require_cv2() -> None:
	if cv2 is None:
		raise RuntimeError("opencv-python is required for OCR image processing")


def _require_pytesseract() -> None:
	if pytesseract is None:
		raise RuntimeError("pytesseract is required for OCR text extraction")


def _four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
	"""Flatten the angled display into a front-facing view."""
	_require_cv2()
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
	_require_cv2()
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
	_require_cv2()
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
	_require_cv2()
	gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
	_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return thresh


def _extract_mode_roi(binary: np.ndarray) -> np.ndarray:
	"""Crop the top line where mode text appears."""
	_require_cv2()
	mode_roi = CURRENT_LAYOUT.fields["mode"].ideal.crop(binary)
	mode_roi = cv2.bitwise_not(mode_roi)
	mode_roi = cv2.GaussianBlur(mode_roi, (3, 3), 0)
	return cv2.resize(mode_roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)


def _extract_mode_variants(binary: np.ndarray) -> list[np.ndarray]:
	"""Build a few cheap OCR variants for the mode line."""
	_require_cv2()
	mode_roi = _extract_mode_roi(binary)
	variants = [mode_roi]

	# Slightly different binarization paths help when the display is dim or skewed.
	_, otsu_inverse = cv2.threshold(mode_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	variants.append(otsu_inverse)
	variants.append(cv2.morphologyEx(mode_roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))))

	return variants


def _fallback_mode_variants(frame: np.ndarray) -> list[np.ndarray]:
	"""Build conservative OCR variants when the display contour is not found."""
	_require_cv2()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
	mode_roi = CURRENT_LAYOUT.fields["mode"].fallback.crop(gray)
	mode_roi = cv2.GaussianBlur(mode_roi, (3, 3), 0)
	_, otsu = cv2.threshold(mode_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return [mode_roi, otsu, cv2.bitwise_not(otsu)]


def _extract_temp_roi(binary: np.ndarray) -> np.ndarray:
	"""Crop the center section where numeric temperature appears."""
	_require_cv2()
	temp_roi = CURRENT_LAYOUT.fields["temperature"].ideal.crop(binary)
	temp_roi = cv2.bitwise_not(temp_roi)
	temp_roi = cv2.GaussianBlur(temp_roi, (3, 3), 0)

	# Small close operation helps connect broken strokes in digits.
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	temp_roi = cv2.morphologyEx(temp_roi, cv2.MORPH_CLOSE, kernel)
	return cv2.resize(temp_roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)


def _extract_temp_variants(binary: np.ndarray) -> list[np.ndarray]:
	"""Build a few cheap OCR variants for the temperature line."""
	_require_cv2()
	temp_roi = _extract_temp_roi(binary)
	variants = [temp_roi]

	_, otsu_inverse = cv2.threshold(temp_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	variants.append(otsu_inverse)
	variants.append(cv2.morphologyEx(temp_roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))))
	variants.append(cv2.dilate(temp_roi, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1))

	return variants


def _fallback_temp_variants(frame: np.ndarray) -> list[np.ndarray]:
	"""Build conservative OCR variants when the display contour is not found."""
	_require_cv2()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
	temp_roi = CURRENT_LAYOUT.fields["temperature"].fallback.crop(gray)
	temp_roi = cv2.GaussianBlur(temp_roi, (3, 3), 0)
	_, otsu = cv2.threshold(temp_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return [temp_roi, otsu, cv2.bitwise_not(otsu)]


def _read_mode(mode_roi: np.ndarray) -> tuple[str, str]:
	"""Run OCR on mode text and map it to the nearest known mode."""
	_require_pytesseract()
	config = CURRENT_LAYOUT.fields["mode"].tesseract_config
	return _read_mode_from_variants([mode_roi], config)


def _read_mode_from_variants(variants: list[np.ndarray], config: str) -> tuple[str, str]:
	"""OCR the mode line using several cheap variants and keep the best text."""
	_require_pytesseract()
	best_raw = ""
	best_mode = CURRENT_LAYOUT.known_modes[0]
	best_score = -1.0

	for variant in variants:
		raw = pytesseract.image_to_string(variant, config=config).strip().upper()
		mode = parse_mode_text(raw)
		score = _sequence_score(raw, mode)
		if score > best_score:
			best_raw = raw
			best_mode = mode
			best_score = score

	return best_raw, best_mode


def _read_temperature(temp_roi: np.ndarray) -> tuple[str, Optional[float]]:
	"""Run OCR on numeric text and parse a float when possible."""
	_require_pytesseract()
	config = CURRENT_LAYOUT.fields["temperature"].tesseract_config
	return _read_temperature_from_variants([temp_roi], config)


def _read_temperature_from_variants(variants: list[np.ndarray], config: str) -> tuple[str, Optional[int]]:
	"""OCR the temperature line using several cheap variants and keep the best integer."""
	_require_pytesseract()
	best_raw = ""
	best_value: Optional[int] = None
	best_score = -1.0

	for variant in variants:
		raw = pytesseract.image_to_string(variant, config=config).strip()
		value = parse_temperature_text(raw)
		if value is None:
			continue

		score = _score_temperature_candidate(raw, value)
		if score > best_score:
			best_raw = raw
			best_value = value
			best_score = score

	return best_raw, best_value


def parse_mode_text(raw_mode: str) -> str:
	"""Map free-form OCR text to the closest expected mode string."""
	if not raw_mode:
		return CURRENT_LAYOUT.known_modes[0]

	normalized = raw_mode.strip().upper()
	if not normalized:
		return CURRENT_LAYOUT.known_modes[0]

	# First try keyword hits because the display text is short and highly structured.
	keyword_match = _score_mode_keywords(normalized)
	if keyword_match is not None:
		return keyword_match

	# Fall back to fuzzy matching, then the closest string by ratio.
	match = get_close_matches(normalized, CURRENT_LAYOUT.known_modes, n=1, cutoff=0.55)
	if match:
		return match[0]

	return max(CURRENT_LAYOUT.known_modes, key=lambda mode: _sequence_score(normalized, mode))


def _normalize_mode_text(raw_mode: str) -> str:
	"""Normalize OCR text into uppercase words separated by single spaces."""
	text = raw_mode.upper()
	text = text.replace("+", " PLUS ")
	text = re.sub(r"[^A-Z]+", " ", text)
	return re.sub(r"\s+", " ", text).strip()


def _score_mode_keywords(normalized: str) -> Optional[str]:
	"""Pick the mode that best matches obvious keywords in the OCR text."""
	words = set(normalized.split())
	for mode, keywords in CURRENT_LAYOUT.mode_keywords.items():
		if all(keyword in words for keyword in keywords):
			return mode

	# Handle common OCR blends like HEATPUMP or HYBRIDPLUS.
	compact = normalized.replace(" ", "")
	if "HYBRIDPLUS" in compact:
		return "HYBRID PLUS"
	if "HEATPUMP" in compact:
		return "HEAT PUMP"
	if "VACATION" in compact:
		return "VACATION"
	if "ELECTRIC" in compact:
		return "ELECTRIC"
	if "HYBRID" in compact:
		return "HYBRID"

	return None


def _sequence_score(raw_mode: str, mode: str) -> float:
	"""Score a candidate mode by rough text similarity and keyword presence."""
	compact_raw = _normalize_mode_text(raw_mode).replace(" ", "")
	compact_mode = mode.replace(" ", "")
	# difflib sequence similarity without importing another helper.
	from difflib import SequenceMatcher
	value = SequenceMatcher(None, compact_raw, compact_mode).ratio()
	if mode == "HYBRID PLUS" and "PLUS" in raw_mode:
		value += 0.08
	if mode == "HEAT PUMP" and ("HEAT" in raw_mode or "PUMP" in raw_mode):
		value += 0.08
	if mode == "HYBRID" and "HYBRID" in raw_mode:
		value += 0.04
	if mode == "ELECTRIC" and "ELECTRIC" in raw_mode:
		value += 0.08
	if mode == "VACATION" and "VACATION" in raw_mode:
		value += 0.08
	return value


def parse_temperature_text(raw_temp: str) -> Optional[int]:
	"""Extract an integer temperature from OCR text when available."""
	if not raw_temp:
		return None

	# Replace common OCR confusion before numeric parsing.
	normalized = raw_temp.upper()
	normalized = normalized.replace("O", "0").replace("I", "1").replace("L", "1")
	normalized = normalized.replace("|", "1").replace("S", "5").replace("B", "8").replace("Z", "2")
	matches = list(re.finditer(r"\d+", normalized))
	if not matches:
		return None

	best_match = max(matches, key=lambda match: (len(match.group()), -match.start()))
	value = int(best_match.group())
	if value <= 0:
		return None

	return value


def _score_temperature_candidate(raw_temp: str, temperature: int) -> float:
	"""Score a candidate temperature by plausibility and OCR shape."""
	score = 0.0
	min_temp, max_temp = CURRENT_LAYOUT.temperature_range_f
	if min_temp <= temperature <= max_temp:
		score += 2.0
	if 100 <= temperature <= 199:
		score += 1.5
	if len(str(temperature)) == 3:
		score += 1.0
	if 60 <= temperature <= 240:
		score += 0.5
	if "." in raw_temp:
		score += 0.2
	if any(char in raw_temp.upper() for char in ("O", "I", "L", "S", "B", "Z", "|")):
		score += 0.1
	return score


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
	_require_cv2()
	_require_pytesseract()
	mask = _display_mask(frame)
	display_contour = _find_display_contour(mask)

	if display_contour is None:
		fallback_raw, fallback_mode = _read_mode_from_variants(
			_fallback_mode_variants(frame),
			"--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ+ ",
		)
		return OCRReadout(
			display_found=False,
			mode=fallback_mode,
			mode_raw=fallback_raw,
			temperature_f=None,
			temperature_raw="",
		)

	warped = _four_point_transform(frame, display_contour.reshape(4, 2))
	binary = _prepare_binary(warped)

	mode_variants = _extract_mode_variants(binary)
	mode_raw, mode = _read_mode_from_variants(mode_variants, "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ+ ")

	temp_raw, temp_value = _read_temperature_from_variants(
		_extract_temp_variants(binary),
		"--psm 7 -c tessedit_char_whitelist=0123456789Ool|SsbZ",
	)

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
	return _get_camera() is not None


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
		# Auto-detect available cameras
		camera_info = Picamera2.global_camera_info()
		if not camera_info:
			return None
		
		# Use the first available camera
		camera_num = camera_info[0]["Num"]
		camera = Picamera2(camera_num=camera_num)
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
