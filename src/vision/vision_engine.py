"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/vision/vision_engine.py - Vision/OCR Engine
Captures camera frames, isolates the display region, and extracts mode/temperature via OCR.
"""
# TODO: remove try/except imports and add explicit requirements once dependencies are finalized.
from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
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
	current_menu_key: str
	fields: dict[str, dict[str, Any]]


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
	"""Keep the LCD body and bright UI content so the icon bar stays attached."""
	_require_cv2()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower_orange = np.array([5, 150, 150])
	upper_orange = np.array([25, 255, 255])
	orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
	bright_mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 90, 255]))
	mask = cv2.bitwise_or(orange_mask, bright_mask)

	# A taller closing kernel helps bridge the darker lower strip back into the screen outline.
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 21))
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
	return mask


def _find_display_contour(mask: np.ndarray, min_area: int = 3000) -> Optional[np.ndarray]:
	"""Find the 4-corner shape that best matches the LCD window geometry."""
	_require_cv2()
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	best_contour = None
	best_score = float("inf")
	target_ratio = CURRENT_LAYOUT.display_aspect_ratio
	ratio_tolerance = 0.22
	for contour in contours:
		area = cv2.contourArea(contour)
		if area < min_area:
			continue

		hull = cv2.convexHull(contour)
		perimeter = cv2.arcLength(hull, True)
		approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
		if len(approx) != 4:
			continue

		(x, y), (width, height), angle = cv2.minAreaRect(approx)
		short_side = min(width, height)
		long_side = max(width, height)
		if short_side <= 0 or long_side <= 0:
			continue

		ratio = short_side / long_side
		ratio_error = abs(ratio - target_ratio)
		if ratio_error > ratio_tolerance:
			continue

		score = ratio_error - (area / 1_000_000.0)
		if score < best_score:
			best_contour = approx
			best_score = score

	if best_contour is not None:
		return best_contour

	# Fallback: choose the largest quadrilateral if nothing fit the expected shape.
	best_area = 0.0
	for contour in contours:
		area = cv2.contourArea(contour)
		if area < min_area:
			continue

		hull = cv2.convexHull(contour)
		perimeter = cv2.arcLength(hull, True)
		approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
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


def _extract_field_variants(binary: np.ndarray, field_name: str) -> list[np.ndarray]:
	"""Build OCR variants for any field using its ROI definitions."""
	_require_cv2()
	field = CURRENT_LAYOUT.fields[field_name]
	roi = field.ideal.crop(binary)
	roi = cv2.bitwise_not(roi)
	roi = cv2.GaussianBlur(roi, (3, 3), 0)
	
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	if field_name == "temperature":
		roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
		
	roi = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
	variants = [roi]
	
	_, otsu_inverse = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	variants.append(otsu_inverse)
	variants.append(cv2.morphologyEx(roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))))
	
	if field_name == "temperature":
		variants.append(cv2.dilate(roi, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1))
		
	return variants


def _fallback_field_variants(frame: np.ndarray, field_name: str) -> list[np.ndarray]:
	"""Build conservative OCR variants when the display contour is not found."""
	_require_cv2()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
	field = CURRENT_LAYOUT.fields[field_name]
	roi = field.fallback.crop(gray)
	roi = cv2.GaussianBlur(roi, (3, 3), 0)
	_, otsu = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return [roi, otsu, cv2.bitwise_not(otsu)]


def _draw_roi_overlay(image: np.ndarray, use_fallback_rois: bool) -> np.ndarray:
	"""Draw every OCR ROI on a frame for calibration and debugging."""
	_require_cv2()
	overlay = image.copy()
	height, width = overlay.shape[:2]
	fields = list(CURRENT_LAYOUT.fields.values())

	for index, field in enumerate(fields):
		box = field.fallback if use_fallback_rois else field.ideal
		top = int(height * box.top)
		bottom = int(height * box.bottom)
		left = int(width * box.left)
		right = int(width * box.right)
		color = (255, 255, 255)
		label_color = (255, 255, 255)

		# High-contrast 0.5 pt-style outline: black shadow plus white edge.
		cv2.rectangle(overlay, (left, top), (right, bottom), (0, 0, 0), 3, cv2.LINE_AA)
		cv2.rectangle(overlay, (left, top), (right, bottom), color, 1, cv2.LINE_AA)

		label_top = max(12, top - 6)
		label_origin = (left + 4, label_top)
		label = field.name.upper()
		cv2.putText(
			overlay,
			label,
			label_origin,
			cv2.FONT_HERSHEY_SIMPLEX,
			0.45,
			(0, 0, 0),
			3,
			cv2.LINE_AA,
		)
		cv2.putText(
			overlay,
			label,
			label_origin,
			cv2.FONT_HERSHEY_SIMPLEX,
			0.45,
			label_color,
			1,
			cv2.LINE_AA,
		)

		# Stagger labels slightly when ROIs overlap near the top edge.
		if index > 0:
			cv2.line(overlay, (left, top), (left + 12, top), color, 1, cv2.LINE_AA)

	return overlay


def _draw_polygon_outline(image: np.ndarray, points: np.ndarray, label: str) -> None:
	"""Draw a high-contrast polygon outline with a readable label."""
	_require_cv2()
	polygon = points.astype("int32")
	cv2.polylines(image, [polygon], True, (0, 0, 0), 3, cv2.LINE_AA)
	cv2.polylines(image, [polygon], True, (255, 255, 255), 1, cv2.LINE_AA)

	anchor_x = int(polygon[:, 0].min()) + 4
	anchor_y = max(18, int(polygon[:, 1].min()) - 8)
	anchor = (anchor_x, anchor_y)
	cv2.putText(
		image,
		label,
		anchor,
		cv2.FONT_HERSHEY_SIMPLEX,
		0.45,
		(0, 0, 0),
		3,
		cv2.LINE_AA,
	)
	cv2.putText(
		image,
		label,
		anchor,
		cv2.FONT_HERSHEY_SIMPLEX,
		0.45,
		(255, 255, 255),
		1,
		cv2.LINE_AA,
	)


def _project_roi_box(box: "ROIBox", inverse_transform: np.ndarray, warped_width: int, warped_height: int) -> np.ndarray:
	"""Project a normalized warped ROI back into source-frame coordinates."""
	_require_cv2()
	warped_points = np.array(
		[
			[warped_width * box.left, warped_height * box.top],
			[warped_width * box.right, warped_height * box.top],
			[warped_width * box.right, warped_height * box.bottom],
			[warped_width * box.left, warped_height * box.bottom],
		],
		dtype="float32",
	)
	projected = cv2.perspectiveTransform(warped_points[None, :, :], inverse_transform)[0]
	return projected


def _safe_capture_stem(capture_id: Optional[str]) -> str:
	"""Normalize optional IDs into filename-safe stems."""
	if not capture_id:
		return "roi_ocr"

	cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", capture_id).strip("._")
	return cleaned or "roi_ocr"


def save_roi_ocr_overlay(capture_dir: Path, frame: np.ndarray, capture_id: Optional[str]) -> Optional[Path]:
	"""Persist a calibration image that shows every OCR ROI on the current frame."""
	_require_cv2()
	if frame is None:
		return None

	capture_dir.mkdir(parents=True, exist_ok=True)
	mask = _display_mask(frame)
	display_contour = _find_display_contour(mask)

	if display_contour is None:
		overlay = _draw_roi_overlay(frame, use_fallback_rois=True)
	else:
		source_points = _order_points(display_contour.reshape(4, 2))
		width_a = np.linalg.norm(source_points[2] - source_points[3])
		width_b = np.linalg.norm(source_points[1] - source_points[0])
		warped_width = max(int(width_a), int(width_b))
		height_a = np.linalg.norm(source_points[1] - source_points[2])
		height_b = np.linalg.norm(source_points[0] - source_points[3])
		warped_height = max(int(height_a), int(height_b))
		dst = np.array(
			[
				[0, 0],
				[warped_width - 1, 0],
				[warped_width - 1, warped_height - 1],
				[0, warped_height - 1],
			],
			dtype="float32",
		)
		transform = cv2.getPerspectiveTransform(source_points, dst)
		inverse_transform = cv2.getPerspectiveTransform(dst, source_points)
		overlay = frame.copy()
		_draw_polygon_outline(overlay, source_points, "DETECTED SCREEN BORDER")

		for field in CURRENT_LAYOUT.fields.values():
			projected = _project_roi_box(field.ideal, inverse_transform, warped_width, warped_height)
			_draw_polygon_outline(overlay, projected, field.name.upper())

	output_path = capture_dir / f"{_safe_capture_stem(capture_id)}_roi_ocr.jpg"
	if not cv2.imwrite(str(output_path), overlay):
		return None

	return output_path


def _read_field_from_variants(field_name: str, variants: list[np.ndarray]) -> tuple[str, Any]:
	"""OCR a field using several cheap variants and keep the best text based on field type."""
	_require_pytesseract()
	field = CURRENT_LAYOUT.fields[field_name]
	best_raw = ""
	best_value: Any = None
	best_score = -1.0

	for variant in variants:
		raw = pytesseract.image_to_string(variant, config=field.tesseract_config).strip()

		if field_name == "temperature":
			value = parse_temperature_text(raw)
			if value is None:
				continue
			score = _score_temperature_candidate(raw, value)
			if score > best_score:
				best_raw = raw
				best_value = value
				best_score = score
		else:
			score = len(raw)
			if score > best_score:
				best_raw = raw
				best_value = raw if raw else ""
				best_score = score

	# Default return values when parsing fully fails
	if best_value is None:
		if field_name != "temperature":
			best_value = ""

	return best_raw, best_value


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
	current_menu_key: str = "dashboard",
) -> OCRReadout:
	"""
	Read fields from the display in one frame based on current context.

	Args:
		frame: BGR image from camera.
		current_menu_key: Current menu state from layout.

	Returns:
		OCRReadout structure containing dynamic fields.
	"""
	_require_cv2()
	_require_pytesseract()
	mask = _display_mask(frame)
	display_contour = _find_display_contour(mask)
	fields_result: dict[str, dict[str, Any]] = {}

	if display_contour is None:
		for field_name in CURRENT_LAYOUT.fields:
			raw, val = _read_field_from_variants(field_name, _fallback_field_variants(frame, field_name))
			fields_result[field_name] = {"raw": raw, "value": val}
		
		return OCRReadout(
			display_found=False,
			current_menu_key=current_menu_key,
			fields=fields_result,
		)

	warped = _four_point_transform(frame, display_contour.reshape(4, 2))
	binary = _prepare_binary(warped)

	for field_name in CURRENT_LAYOUT.fields:
		raw, val = _read_field_from_variants(field_name, _extract_field_variants(binary, field_name))
		fields_result[field_name] = {"raw": raw, "value": val}

	return OCRReadout(
		display_found=True,
		current_menu_key=current_menu_key,
		fields=fields_result,
	)


def capture_frame() -> Optional[np.ndarray]:
	"""Capture one camera frame (returns None when camera is unavailable)."""
	camera = _get_camera()
	if camera is None:
		return None

	return camera.capture_array()


def capture_and_read_display(
	current_menu_key: str = "dashboard",
) -> OCRReadout:
	"""One-call helper used by main: capture frame then run OCR pipeline."""
	frame = capture_frame()
	if frame is None:
		empty_fields = {name: {"raw": "", "value": None} for name in CURRENT_LAYOUT.fields}
		# Ensure defaults
		empty_fields["mode"] = {"raw": "", "value": "UNKNOWN"}
		return OCRReadout(
			display_found=False,
			current_menu_key=current_menu_key,
			fields=empty_fields,
		)

	return read_display(
		frame,
		current_menu_key,
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
