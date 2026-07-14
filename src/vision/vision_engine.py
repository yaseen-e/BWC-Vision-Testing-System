"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/vision/vision_engine.py - Vision/OCR Engine
Captures camera frames, isolates the display region, and extracts mode/temperature via OCR.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import cv2
import pytesseract
from .display_layouts import OCRField, TEMPERATURE_RANGE_F

# Camera is assumed to be connected and initialized directly.
_CAMERA: Any = None


@dataclass(frozen=True)
class OCRReadout:
	"""Single OCR result payload for upstream logic (main/network)."""

	display_found: bool
	current_menu_key: str
	fields: dict[str, dict[str, Any]]


def _require_cv2() -> None:
	if cv2 is None:
		raise RuntimeError("opencv-python is required for OCR image processing")


def _require_pytesseract() -> None:
	if pytesseract is None:
		raise RuntimeError("pytesseract is required for OCR text extraction")


def _order_points(points: np.ndarray) -> np.ndarray:
	"""Normalize corner order so perspective math stays stable."""
	points = np.asarray(points, dtype="float32").reshape(4, 2)
	ordered = np.zeros((4, 2), dtype="float32")
	point_sums = points.sum(axis=1)
	ordered[0] = points[np.argmin(point_sums)]
	ordered[2] = points[np.argmax(point_sums)]
	point_diffs = np.diff(points, axis=1)
	ordered[1] = points[np.argmin(point_diffs)]
	ordered[3] = points[np.argmax(point_diffs)]
	return ordered


def _warp_image(image: np.ndarray, points: np.ndarray) -> np.ndarray:
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

def _build_display_mask(frame: np.ndarray) -> np.ndarray:
	"""Build a single mask for the emissive orange display window."""
	_require_cv2()
	if frame is None or frame.size == 0:
		raise ValueError("frame must be a non-empty image")

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# Secure the vibrant orange/amber background perfectly.
	orange_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([35, 255, 255]))

	# Fuse any inner text gaps horizontally/vertically without spilling into bezels
	mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)))
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15)))

	# Clean up any lingering tiny noise specs
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
	
	return mask


def _find_display_contour(frame: np.ndarray, mask: np.ndarray, min_area: int = 3000) -> Optional[np.ndarray]:
	"""Find the display rectangle and geometrically project the status bar if present."""
	_require_cv2()
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return None

	frame_area = max(1, mask.shape[0] * mask.shape[1])
	min_area = max(min_area, int(frame_area * 0.05))
	
	best_contour: Optional[np.ndarray] = None
	best_score = float("-inf")

	def _contour_box(candidate: np.ndarray) -> np.ndarray:
		rotated = cv2.minAreaRect(candidate)
		points = cv2.boxPoints(rotated)
		return points.astype("float32")

	for contour in contours:
		area = cv2.contourArea(contour)
		if area < min_area or area > frame_area * 0.99:
			continue

		hull = cv2.convexHull(contour)
		hull_area = cv2.contourArea(hull)
		if hull_area <= 0:
			continue

		solidity = area / hull_area
		if solidity < 0.50: 
			continue

		score = area * solidity
		if score > best_score:
			best_score = score
			best_contour = _contour_box(contour)

	# --- GEOMETRIC STATUS BAR EXTENSION ---
	if best_contour is not None:
		ordered = _order_points(best_contour)
		top_left, top_right, bottom_right, bottom_left = ordered

		# Vectors representing the left and right edges moving downward
		v_left = bottom_left - top_left
		v_right = bottom_right - top_right

		# Sample a small region directly below the orange box (2% to 12% below the edge)
		sample_pts = np.array([
			bottom_left + v_left * 0.02,
			bottom_right + v_right * 0.02,
			bottom_right + v_right * 0.12,
			bottom_left + v_left * 0.12
		], dtype="float32")

		# Flatten the sample region to inspect it cleanly
		tw, th = 100, 20
		dst_pts = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype="float32")
		M = cv2.getPerspectiveTransform(sample_pts, dst_pts)
		sample_region = cv2.warpPerspective(frame, M, (tw, th))
		
		# Look for high-contrast white text/icons inside the dark status bar
		sample_gray = cv2.cvtColor(sample_region, cv2.COLOR_BGR2GRAY)
		_, thresh = cv2.threshold(sample_gray, 160, 255, cv2.THRESH_BINARY)
		
		# If white text pixels are detected, the black status bar layout is active
		if np.sum(thresh == 255) > 25:
			# The status bar takes up 9% of total height (orange box is 91%).
			# Extend downward by exactly 9/91 (~9.89%) of the orange box height.
			ext_factor = 9.0 / 91.0
			extended_bottom_left = bottom_left + v_left * ext_factor
			extended_bottom_right = bottom_right + v_right * ext_factor
			
			best_contour = np.array([top_left, top_right, extended_bottom_right, extended_bottom_left], dtype="float32")

	return best_contour


def _prepare_ocr_binary(warped: np.ndarray) -> np.ndarray:
	"""Convert to grayscale, enhance edges, and scale for OCR."""
	_require_cv2()
	gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	
	# 1. Bilateral Filter: Melts away the LCD pixel grid noise but PRESERVES hard edges.
	# This is much safer than Median Blur for images that are already out of focus.
	filtered = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
	
	# 2. Unsharp Masking: Artificially sharpen the image to combat camera lens blur.
	gaussian_blur = cv2.GaussianBlur(filtered, (0, 0), 2.0)
	sharpened = cv2.addWeighted(filtered, 1.5, gaussian_blur, -0.5, 0)
	
	# 3. Moderate Upscaling: 2.5x gives Tesseract enough pixel density without bloating.
	scaled = cv2.resize(sharpened, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
	
	return scaled


def _extract_warped_variants(binary: np.ndarray, field: OCRField) -> list[np.ndarray]:
	"""Build robust OCR variants utilizing adaptive thresholding and morphological operations to bridge LCD segment gaps."""
	_require_cv2()
	roi = field.ideal.crop(binary)
	
	if np.std(roi) < 15.0:
		return []
	
	# 1. Generate baseline standard and inverted threshold options
	_, otsu_standard = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	otsu_inverted = cv2.bitwise_not(otsu_standard)
	
	# 2. Dynamically enforce dark-on-light polarity.
	# Since display backgrounds occupy far more area than text lines, the background
	# color will consistently hold the dominant pixel count.
	if np.sum(otsu_standard == 255) > np.sum(otsu_standard == 0):
		dark_on_light = otsu_standard
		light_on_dark = otsu_inverted
	else:
		dark_on_light = otsu_inverted
		light_on_dark = otsu_standard

	variants = []
	
	# Variant 1: Clean, high-contrast dark text on a light background (Standard Otsu)
	variants.append(dark_on_light)
	
	# Variant 2: Morphological Closing to bridge gaps in segmented LCD text.
	# Closing (dilation followed by erosion) fills the small inner gaps of white-on-black 
	# character pieces. Then we invert back to dark-on-light for Tesseract.
	kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
	closed_segments = cv2.morphologyEx(light_on_dark, cv2.MORPH_CLOSE, kernel_bridge)
	variants.append(cv2.bitwise_not(closed_segments))
	
	# Variant 3: Adaptive Threshold with specialized dilation to handle backlight glare.
	inverted_roi = cv2.bitwise_not(roi)
	adaptive = cv2.adaptiveThreshold(
		inverted_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
	)
	# Dilating white-on-black text patches smears adjacent segment segments together
	kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
	dilated_adaptive = cv2.dilate(adaptive, kernel_dilate, iterations=1)
	variants.append(cv2.bitwise_not(dilated_adaptive))
	
	return variants

def _extract_fallback_variants(frame: np.ndarray, field: OCRField) -> list[np.ndarray]:
	"""Build conservative OCR variants when the display contour is not found."""
	_require_cv2()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
	roi = field.fallback.crop(gray)
	
	# --- FIX: PHANTOM TEXT HALLUCINATION CHECK ---
	if np.std(roi) < 15.0:
		return []

	roi = cv2.GaussianBlur(roi, (3, 3), 0)
	_, otsu = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return [roi, otsu, cv2.bitwise_not(otsu)]


def _clean_text(raw_text: str) -> str:
	"""Collapse OCR whitespace into a stable single-space form."""
	return re.sub(r"\s+", " ", raw_text).strip()


def _parse_mode(raw_text: str) -> str:
    cleaned = _clean_text(raw_text).upper().strip()
    
    # Split the string using a space as the separator. 
    # maxsplit=1 ensures we only split at the very first space.
    parts = cleaned.split(" ", 1)
    
    # If there are multiple parts, return everything after the first space.
    if len(parts) > 1:
        return parts[1]
        
    # If there was no space, return an empty string (no "UNKNOWN" fallback).
    return ""


def _parse_temperature(raw_text: str) -> Optional[int]:
	if not raw_text:
		return None

	normalized = raw_text.upper()
	normalized = normalized.replace("O", "0").replace("I", "1").replace("L", "1")
	normalized = normalized.replace("|", "1").replace("S", "5").replace("B", "8").replace("Z", "2")
	all_matches = list(re.finditer(r"\d+", normalized))
	if not all_matches:
		return None

	best_match = max(all_matches, key=lambda match: (len(match.group()), -match.start()))
	value = int(best_match.group())
	if value <= 0:
		return None

	return value


def _parse_info_line(raw_text: str) -> str:
	# Strictly enforce only letters, numbers, spaces, and periods
	text = re.sub(r"[^a-zA-Z0-9 \.]", "", raw_text)
	
	# Clean up any duplicated whitespace and strip trailing/leading spaces or periods
	text = _clean_text(text).strip(" .")
	
	if not text:
		return ""

	alpha_count = sum(1 for char in text if char.isalpha())
	alnum_count = sum(1 for char in text if char.isalnum())
	if alpha_count < 3 or alnum_count <= 0:
		return ""

	if (alpha_count / max(1, len(text))) < 0.35:
		return ""

	return text


def _field_empty_value(field_name: str) -> Any:
	if field_name == "mode":
		return "UNKNOWN"
	if field_name == "temperature":
		return None
	return ""


def _parse_time(raw_text: str) -> str:
	# Normalize to uppercase for uniform string analysis
	text = raw_text.upper().strip()
	
	# Self-heal dropped or misinterpreted 'M' characters at the end of the timestamp
	if text.endswith("A") or text.endswith("AN") or text.endswith("AH"):
		text = text.split("A")[0] + "AM"
	elif text.endswith("P") or text.endswith("PN") or text.endswith("PH"):
		text = text.split("P")[0] + "PM"

	# Strictly enforce only numbers, spaces, A, M, P, and colon ':'
	text = re.sub(r"[^0-9 AMP:]", "", text)
	
	return _clean_text(text)


def _parse_field_value(field_name: str, raw_text: str) -> Any:
	if field_name == "mode":
		return _parse_mode(raw_text)
	if field_name == "temperature":
		return _parse_temperature(raw_text)
	if field_name.startswith("dashboard_info_line_"):
		return _parse_info_line(raw_text)
	if field_name == "time_field":
		return _parse_time(raw_text)
	return _clean_text(raw_text)


def _score_temperature(raw_text: str, value: Any) -> float:
	if not isinstance(value, int):
		return -1.0

	score = 0.0
	min_temp, max_temp = TEMPERATURE_RANGE_F
	if min_temp <= value <= max_temp:
		score += 2.0
	if 100 <= value <= 199:
		score += 1.5
	if len(str(value)) == 3:
		score += 1.0
	if 60 <= value <= 240:
		score += 0.5
	if "." in raw_text:
		score += 0.2
	if any(char in raw_text.upper() for char in ("O", "I", "L", "S", "B", "Z", "|")):
		score += 0.1
	return score


def _score_info_line(raw_text: str, parsed_value: Any) -> float:
	if not isinstance(parsed_value, str) or not parsed_value:
		return -1.0

	alpha_count = sum(1 for char in parsed_value if char.isalpha())
	word_count = len([word for word in parsed_value.split(" ") if any(char.isalpha() for char in word)])
	punct_count = sum(1 for char in parsed_value if not char.isalnum() and char != " ")
	if alpha_count < 3 or word_count == 0:
		return -1.0

	junk_cluster_penalty = 0.8 if re.search(r"[^A-Za-z0-9\s]{3,}", raw_text) else 0.0

	score = 0.0
	score += min(3.0, alpha_count * 0.10)
	score += min(2.0, word_count * 0.70)
	score += min(1.5, len(parsed_value) * 0.04)
	score -= min(2.0, punct_count * 0.15)
	score -= junk_cluster_penalty
	return score


def _score_field_candidate(field_name: str, raw_text: str, value: Any) -> float:
	if field_name == "temperature":
		return _score_temperature(raw_text, value)
	if field_name.startswith("dashboard_info_line_"):
		return _score_info_line(raw_text, value)
	if isinstance(value, str):
		return float(len(value))
	if value is None:
		return -1.0
	return 0.0


def _ocr_field(field: OCRField, variants: list[np.ndarray]) -> tuple[str, Any]:
	"""OCR a field from multiple variants and keep the best-scoring candidate."""
	_require_pytesseract()
	field_name = field.name
	best_raw = ""
	best_value: Any = _field_empty_value(field_name)
	best_score = -1.0
	candidate_debug: list[tuple[float, str, Any]] = []

	# 1. Parse and cache the whitelist set once per field invocation (O(1) lookups)
	whitelist_set = None
	if "tessedit_char_whitelist=" in field.tesseract_config:
		whitelist_set = set(field.tesseract_config.split("tessedit_char_whitelist=")[1])

	for variant in variants:
		raw = _clean_text(pytesseract.image_to_string(variant, config=field.tesseract_config))
		
		# 2. Filter raw text early for non-temperature fields to strip noise 
		# and prevent junk characters from spoiling the candidate score.
		if whitelist_set and field_name != "temperature":
			raw = "".join(c for c in raw if c in whitelist_set)

		value = _parse_field_value(field_name, raw)
		score = _score_field_candidate(field_name, raw, value)

		candidate_debug.append((score, raw, value))

		if score > best_score:
			best_raw = raw
			best_value = value
			best_score = score

	# 3. Apply strict final enforcement on the chosen best candidate outputs
	if whitelist_set:
		best_raw = "".join(c for c in best_raw if c in whitelist_set)
		if isinstance(best_value, str):
			best_value = "".join(c for c in best_value if c in whitelist_set)

	return best_raw, best_value


def _draw_roi_overlay(image: np.ndarray, fields: tuple[OCRField, ...], use_fallback_rois: bool) -> np.ndarray:
	"""Draw every OCR ROI on a frame for calibration and debugging."""
	_require_cv2()
	overlay = image.copy()
	height, width = overlay.shape[:2]

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


def save_roi_ocr_overlay(
	capture_dir: Path,
	frame: np.ndarray,
	capture_id: Optional[str],
	menu_fields: tuple[OCRField, ...],
) -> Optional[Path]:
	"""Persist a calibration image that shows every OCR ROI on the current frame."""
	_require_cv2()
	if frame is None:
		return None

	capture_dir.mkdir(parents=True, exist_ok=True)
	mask = _build_display_mask(frame)
	display_contour = _find_display_contour(frame, mask)

	if display_contour is None:
		overlay = _draw_roi_overlay(frame, menu_fields, use_fallback_rois=True)
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

		for field in menu_fields:
			projected = _project_roi_box(field.ideal, inverse_transform, warped_width, warped_height)
			_draw_polygon_outline(overlay, projected, field.name.upper())

	output_path = capture_dir / f"{_safe_capture_stem(capture_id)}_roi_ocr.jpg"
	if not cv2.imwrite(str(output_path), overlay):
		return None

	return output_path


def read_display(
	frame: np.ndarray,
	current_menu_key: str,
	menu_fields: tuple[OCRField, ...],
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
	mask = _build_display_mask(frame)
	display_contour = _find_display_contour(frame, mask)
	fields_result: dict[str, dict[str, Any]] = {}

	if display_contour is None:
		for field in menu_fields:
			raw, val = _ocr_field(field, _extract_fallback_variants(frame, field))
			fields_result[field.name] = {"raw": raw, "value": val}
		
		return OCRReadout(
			display_found=False,
			current_menu_key=current_menu_key,
			fields=fields_result,
		)

	warped = _warp_image(frame, display_contour.reshape(4, 2))
	binary = _prepare_ocr_binary(warped)

	for field in menu_fields:
		raw, val = _ocr_field(field, _extract_warped_variants(binary, field))
		fields_result[field.name] = {"raw": raw, "value": val}

	return OCRReadout(
		display_found=True,
		current_menu_key=current_menu_key,
		fields=fields_result,
	)


def capture_frame() -> Optional[np.ndarray]:
	"""Capture one camera frame."""
	camera = _get_camera()
	return camera.capture_array()


def capture_and_read_display(
	current_menu_key: str,
	menu_fields: tuple[OCRField, ...],
) -> OCRReadout:
	"""One-call helper used by main: capture frame then run OCR pipeline."""
	frame = capture_frame()
	return read_display(
		frame,
		current_menu_key,
		menu_fields,
	)


def warm_up() -> None:
	"""Hook for camera warmup work."""
	camera = _get_camera()
	if camera is not None:
		import time
		# 1. Allow the physical focus motor and auto-exposure time to stabilize
		time.sleep(0.5)
		
		# 2. Capture and discard an initial frame to flush the stale pipeline buffer
		try:
			camera.capture_array()
		except Exception:
			pass


def is_camera_available() -> bool:
	"""Return True when the camera can be used."""
	return True


def _get_camera() -> Any:
	"""Initialize and return the camera object."""
	global _CAMERA
	if _CAMERA is not None:
		return _CAMERA

	from picamera2 import Picamera2  # type: ignore

	camera_info = Picamera2.global_camera_info()
	camera_num = camera_info[0]["Num"]
	camera = Picamera2(camera_num=camera_num)
	config = camera.create_preview_configuration(main={"size": (1280, 720)})
	camera.configure(config)
	camera.start()
	camera.set_controls({
		"AfMode": 0,
		"LensPosition": 9,
	})

	_CAMERA = camera
	return _CAMERA


def shutdown() -> None:
	"""Stop the camera cleanly."""
	global _CAMERA
	if _CAMERA is None:
		return

	_CAMERA.stop()
	_CAMERA = None


def get_warped_display(frame: np.ndarray) -> Optional[np.ndarray]:
	"""Extract a front-facing view of the display from a camera frame, or None if not found."""
	mask = _build_display_mask(frame)
	display_contour = _find_display_contour(frame, mask)

	if display_contour is None:
		return None

	return _warp_image(frame, display_contour.reshape(4, 2))
