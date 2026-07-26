"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/vision/vision_engine.py - Vision/OCR Engine
Captures camera frames, isolates the display region, and extracts mode/temperature via RapidOCR (ONNX Runtime).
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Optional
import time

import numpy as np
import cv2
from rapidocr_onnxruntime import RapidOCR

from .layouts import OCRField, ROIBox, STATUS_BAR

# Camera and OCR Engine Singletons
_CAMERA: Any = None
_RAPID_OCR: Any = None


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass(frozen=True)
class OCRReadout:
	"""Single OCR result payload for upstream logic (main/network)."""

	display_found: bool
	current_menu_key: str
	fields: dict[str, dict[str, Any]]


# =============================================================================
# PUBLIC API & LIFECYCLE
# =============================================================================

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


def read_display(
	frame: np.ndarray,
	current_menu_key: str,
	menu_fields: tuple[OCRField, ...],
) -> OCRReadout:
	"""Read fields from the display in one frame based on current context using RapidOCR."""
	_require_cv2()
	_require_rapidocr()
	mask = _build_display_mask(frame)
	display_contour = _find_display_contour(frame, mask)

	if display_contour is None:
		return OCRReadout(
			display_found=False,
			current_menu_key=current_menu_key,
			fields={
				field.name: {"raw": "", "value": _field_empty_value(field.name)}
				for field in menu_fields
			},
		)

	final_contour, warped = _process_display_contour_and_warp(
		frame,
		display_contour,
		menu_fields=menu_fields,
	)
	binary = _prepare_ocr_binary(warped)

	fields_result: dict[str, dict[str, Any]] = {}
	for field in menu_fields:
		raw, val = _ocr_field(field, _extract_warped_variants(binary, field))
		fields_result[field.name] = {"raw": raw, "value": val}

	return OCRReadout(
		display_found=True,
		current_menu_key=current_menu_key,
		fields=fields_result,
	)


def capture_frame() -> Optional[np.ndarray]:
	"""Capture one camera frame and convert from Picamera2 RGB to OpenCV BGR."""
	camera = _get_camera()
	frame = camera.capture_array()
	if frame is not None:
		return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	return None


def get_warped_display(
	frame: np.ndarray,
	current_menu_key: Optional[str] = None,
	menu_fields: Optional[tuple[OCRField, ...]] = None,
) -> Optional[np.ndarray]:
	"""Extract a front-facing view of the display from a camera frame, or None if not found."""
	_ = current_menu_key
	mask = _build_display_mask(frame)
	display_contour = _find_display_contour(frame, mask)

	if display_contour is None:
		return None

	_, warped = _process_display_contour_and_warp(frame, display_contour, menu_fields=menu_fields)
	return warped


def save_roi_ocr_overlay(
	capture_dir: Path,
	frame: np.ndarray,
	capture_id: Optional[str],
	menu_fields: tuple[OCRField, ...],
	current_menu_key: Optional[str] = None,
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
		final_contour, warped = _process_display_contour_and_warp(
			frame,
			display_contour,
			menu_fields=menu_fields,
		)
		source_points = _order_points(final_contour.reshape(4, 2))

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


def warm_up() -> None:
	"""Hook for camera and OCR engine warmup work to fully flush hardware pipelines."""
	_ = _get_camera()
	_ = _get_rapid_ocr()


def is_camera_available() -> bool:
	"""Return True when the camera can be used."""
	return True


def shutdown() -> None:
	"""Stop the camera cleanly."""
	global _CAMERA
	if _CAMERA is None:
		return

	_CAMERA.stop()
	_CAMERA = None


# =============================================================================
# SINGLETONS & SYSTEM REQUIREMENT CHECKS
# =============================================================================

def _require_cv2() -> None:
	if cv2 is None:
		raise RuntimeError("opencv-python is required for OCR image processing")


def _require_rapidocr() -> None:
	if RapidOCR is None:
		raise RuntimeError(
			"rapidocr_onnxruntime is required for OCR text extraction. "
			"Run: pip install rapidocr_onnxruntime onnxruntime"
		)


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
		"LensPosition": 9.1,
	})

	time.sleep(0.6)
	for _ in range(6):
		try:
			camera.capture_array()
		except Exception:
			pass

	try:
		converged = camera.capture_metadata()
		lock_controls: dict[str, Any] = {"AeEnable": False, "AwbEnable": False}
		for key in ("ExposureTime", "AnalogueGain", "ColourGains"):
			if key in converged:
				lock_controls[key] = converged[key]
		camera.set_controls(lock_controls)
	except Exception:
		pass

	_CAMERA = camera
	return _CAMERA


def _get_rapid_ocr() -> Any:
	"""Initialize and return the RapidOCR (ONNX Runtime) engine singleton."""
	global _RAPID_OCR
	if _RAPID_OCR is not None:
		return _RAPID_OCR

	_require_rapidocr()
	_RAPID_OCR = RapidOCR()
	return _RAPID_OCR


# =============================================================================
# DISPLAY SEGMENTATION & GEOMETRY
# =============================================================================

def _build_display_mask(frame: np.ndarray) -> np.ndarray:
	"""Build a mask for the emissive display window, regardless of its backlight color."""
	_require_cv2()
	if frame is None or frame.size == 0:
		raise ValueError("frame must be a non-empty image")

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	saturation = hsv[:, :, 1]
	value = hsv[:, :, 2]

	_, saturated_mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	_, bright_mask = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	display_mask = cv2.bitwise_and(saturated_mask, bright_mask)

	mask = cv2.morphologyEx(display_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)))
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15)))

	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

	return mask


def _find_display_contour(frame: np.ndarray, mask: np.ndarray, min_area: int = 3000) -> Optional[np.ndarray]:
	"""Find the raw display rectangle without geometric status bar projection."""
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

	return best_contour


def _should_use_extended_warp(menu_fields: Optional[tuple[OCRField, ...]]) -> bool:
	"""Only extend when the current ContextNode includes the full status bar field set."""
	if not menu_fields:
		return False

	status_bar_names = {field.name for field in STATUS_BAR}
	current_field_names = {field.name for field in menu_fields}
	return status_bar_names.issubset(current_field_names)


def _process_display_contour_and_warp(
	frame: np.ndarray,
	orange_contour: np.ndarray,
	menu_fields: Optional[tuple[OCRField, ...]] = None,
) -> tuple[np.ndarray, np.ndarray]:
	"""Warp the display area once and optionally extend the contour for ContextNodes that include status bar fields."""
	_require_cv2()
	if not _should_use_extended_warp(menu_fields):
		return orange_contour, _warp_image(frame, orange_contour)

	ordered = _order_points(orange_contour)
	top_left, top_right, bottom_right, bottom_left = ordered

	v_left = bottom_left - top_left
	v_right = bottom_right - top_right

	ext_factor = 9.0 / 91.0
	extended_bottom_left = bottom_left + v_left * ext_factor
	extended_bottom_right = bottom_right + v_right * ext_factor
	extended_contour = np.array([top_left, top_right, extended_bottom_right, extended_bottom_left], dtype="float32")

	warped_extended = _warp_image(frame, extended_contour)

	return extended_contour, warped_extended


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


# =============================================================================
# IMAGE PREPARATION & VARIANT GENERATION
# =============================================================================

def _prepare_ocr_binary(warped: np.ndarray) -> np.ndarray:
	"""Convert the warped display to grayscale."""
	_require_cv2()
	return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)


def _denoise_roi_grayscale(roi: np.ndarray) -> np.ndarray:
	"""Scrub sensor speckle/static at native resolution, before any upscaling."""
	despeckled = cv2.medianBlur(roi, 3)
	return cv2.bilateralFilter(despeckled, d=5, sigmaColor=60, sigmaSpace=60)


def _crop_and_clean_roi(prepared: np.ndarray, field: Any) -> np.ndarray:
	"""Crop field ROI with a safe inner margin to strip surrounding box border line bleed."""
	if field is not None and hasattr(field, "ideal"):
		roi = field.ideal.crop(prepared)
	else:
		roi = prepared

	if roi is None or roi.size == 0:
		return roi

	h, w = roi.shape[:2]
	margin_y = max(1, int(h * 0.02))
	margin_x = max(1, int(w * 0.02))

	if h > 2 * margin_y and w > 2 * margin_x:
		roi = roi[margin_y : h - margin_y, margin_x : w - margin_x]

	return roi


def _has_text_signal(img_gray: np.ndarray) -> bool:
	"""Return False if the ROI image is blank, uniform, or lacks valid text-like contours."""
	if img_gray is None or img_gray.size == 0:
		return False

	# Standard deviation check: Blank or purely noisy ROIs have extremely low variance
	std_dev = float(np.std(img_gray))
	if std_dev < 7.0:
		return False

	# Contour geometry check: Search for structures matching standard text stroke sizes
	_, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	if np.mean(thresh) > 127:
		thresh = cv2.bitwise_not(thresh)

	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return False

	img_h, img_w = img_gray.shape[:2]
	min_text_h = max(3, int(img_h * 0.12))

	for c in contours:
		_, _, w, h = cv2.boundingRect(c)
		if min_text_h <= h <= int(img_h * 0.95) and w >= 3 and w <= int(img_w * 0.98):
			return True

	return False


def _extract_warped_variants(prepared: np.ndarray, field: Any = None) -> list[np.ndarray]:
	"""Extract denoised, normalized, and scale-adjusted variants optimal for RapidOCR."""
	_require_cv2()

	roi = _crop_and_clean_roi(prepared, field)

	if roi is None or roi.size == 0:
		return []

	if len(roi.shape) == 3:
		roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

	field_name = getattr(field, "name", "")
	denoised_roi = _denoise_roi_grayscale(roi)

	# --- GIANT DIGIT HANDLING (e.g., TEMPERATURE) ---
	if field_name == "temperature":
		return _generate_digit_variants(denoised_roi)

	# --- STANDARD TEXT LINES (MODE, TIME, DATE, DASHBOARD LINES) ---
	h, w = denoised_roi.shape[:2]
	target_height = 180
	if h < target_height:
		scale = target_height / float(h)
		new_w = max(1, int(w * scale))
		denoised_roi = cv2.resize(denoised_roi, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

	variants: list[np.ndarray] = []

	v1 = cv2.copyMakeBorder(denoised_roi, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)
	variants.append(v1)

	v2 = cv2.copyMakeBorder(cv2.bitwise_not(denoised_roi), 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)
	variants.append(v2)

	return variants


def _generate_digit_variants(roi_gray: np.ndarray) -> list[np.ndarray]:
	"""Locate digit contours, crop tightly, and scale down to standard OCR height (48px)."""
	_require_cv2()
	variants: list[np.ndarray] = []

	_, thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	digit_boxes = []
	roi_h = roi_gray.shape[0]
	min_digit_height = int(roi_h * 0.25)

	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		if h >= min_digit_height and w >= 5:
			digit_boxes.append((x, y, w, h))

	if digit_boxes:
		min_x = min(box[0] for box in digit_boxes)
		min_y = min(box[1] for box in digit_boxes)
		max_x = max(box[0] + box[2] for box in digit_boxes)
		max_y = max(box[1] + box[3] for box in digit_boxes)
		digit_crop = roi_gray[min_y:max_y, min_x:max_x]
	else:
		digit_crop = roi_gray

	ch, cw = digit_crop.shape[:2]
	if ch > 0 and cw > 0:
		target_h = 48
		scale = target_h / float(ch)
		target_w = max(10, int(cw * scale))
		scaled_digits = cv2.resize(digit_crop, (target_w, target_h), interpolation=cv2.INTER_AREA)

		inverted = cv2.bitwise_not(scaled_digits)
		v1 = cv2.copyMakeBorder(inverted, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
		variants.append(v1)

		_, bin_inv = cv2.threshold(scaled_digits, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		v2 = cv2.copyMakeBorder(bin_inv, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
		variants.append(v2)

		v3 = cv2.copyMakeBorder(scaled_digits, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
		variants.append(v3)

	return variants


# =============================================================================
# OCR PROCESSING & TEXT PARSING
# =============================================================================

def _ocr_field(field: OCRField, variants: list[np.ndarray]) -> tuple[str, Any]:
	"""Classify icons or extract text using RapidOCR with multi-layer noise rejection."""
	field_name = field.name

	# --- ICON CLASSIFICATION BYPASS ---
	if field_name in ("wifi_icon", "schedule_icon"):
		if not variants:
			return "UNKNOWN", "UNKNOWN"
		first_variant = variants[0]
		if len(first_variant.shape) == 3:
			first_variant = cv2.cvtColor(first_variant, cv2.COLOR_BGR2GRAY)
		inverted_variant = cv2.bitwise_not(first_variant)
		return _classify_icon_field(inverted_variant, field_name)

	# --- RAPIDOCR FOR TEXT/DIGITS ---
	_require_rapidocr()
	ocr_engine = _get_rapid_ocr()

	best_raw = ""
	best_value: Any = _field_empty_value(field_name)
	best_score = 0.0  # Require candidate score > 0.0 to accept

	for variant in variants:
		if not _has_text_signal(variant):
			continue

		rapid_output, _ = ocr_engine(variant)
		raw, conf = _parse_rapidocr_data(rapid_output)

		cleaned_raw = _sanitize_ocr_text(raw)
		if not cleaned_raw:
			continue

		value = _parse_field_value(field_name, cleaned_raw)
		score = _score_field_candidate(field_name, cleaned_raw, value, conf)

		if score > best_score:
			best_raw = cleaned_raw
			best_value = value
			best_score = score

	return best_raw, best_value


def _parse_rapidocr_data(rapid_output: Optional[list[Any]]) -> tuple[str, float]:
	"""Extract text and average confidence score from RapidOCR result array."""
	if not rapid_output:
		return "", 0.0

	cleaned_tokens: list[str] = []
	confidences: list[float] = []

	for line in rapid_output:
		if not line or len(line) < 3:
			continue

		text, conf = line[1], float(line[2])
		stripped = str(text).strip()
		if not stripped or conf < 0.45:  # Reject low-confidence neural hallucinations
			continue

		cleaned_tokens.append(stripped)
		confidences.append(conf)

	if not cleaned_tokens:
		return "", 0.0

	avg_conf = float(np.mean(confidences))
	return " ".join(cleaned_tokens), avg_conf


def _score_field_candidate(field_name: str, raw_text: str, value: Any, conf: float) -> float:
	"""Score candidate value based on validity, length, and RapidOCR model confidence."""
	if value is None or (isinstance(value, str) and not value) or value == "UNKNOWN":
		return 0.0

	if conf < 0.45:
		return 0.0

	score = conf * 2.0

	if field_name == "temperature" and isinstance(value, int):
		if 60 <= value <= 199:
			score += 3.0
		else:
			return 0.0
	elif isinstance(value, str):
		score += min(len(value) * 0.1, 1.5)

	return score


def _parse_field_value(field_name: str, raw_text: str) -> Any:
	if field_name == "mode":
		return _parse_mode(raw_text)
	if field_name == "temperature":
		return _parse_temperature(raw_text)
	if field_name == "time_field":
		return _parse_time(raw_text)
	return _sanitize_ocr_text(raw_text)


def _parse_mode(raw_text: str) -> str:
	cleaned = _sanitize_ocr_text(raw_text)
	if not cleaned:
		return "UNKNOWN"

	# Generically strip label prefix "MODE:" or "MODE" if present
	cleaned = re.sub(r"^(?i)MODE\s*[:;\-\.]*\s*", "", cleaned).strip()
	cleaned = _sanitize_ocr_text(cleaned).upper()

	return cleaned if cleaned else "UNKNOWN"


def _parse_temperature(raw_text: str) -> Optional[int]:
	"""Extract integer temperature value from raw OCR string."""
	if not raw_text:
		return None

	substitutions = {
		"O": "0", "o": "0", "Q": "0", "D": "0",
		"I": "1", "l": "1", "|": "1", "!": "1",
		"Z": "2", "z": "2",
		"S": "5", "s": "5",
		"B": "8", "g": "9",
	}
	cleaned = raw_text
	for char, digit in substitutions.items():
		cleaned = cleaned.replace(char, digit)

	digits = re.sub(r"[^\d]", "", cleaned)
	if not digits:
		return None

	val = int(digits)
	return val if 50 <= val <= 199 else None


def _parse_time(raw_text: str) -> str:
	text = _sanitize_ocr_text(raw_text).upper()
	if "A" in text:
		text = text.split("A")[0] + "AM"
	elif "P" in text:
		text = text.split("P")[0] + "PM"
	return _sanitize_ocr_text(text)


def _sanitize_ocr_text(raw_text: str) -> str:
	"""
	Generically clean OCR text output by removing surrounding noise symbols,
	edge border artifacts, and isolated single-character tokens.
	"""
	if not raw_text:
		return ""

	text = re.sub(r"\s+", " ", raw_text).strip()

	# Strip non-alphanumeric noise at start/end
	text = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9.\%]+$", "", text).strip()
	if not text:
		return ""

	# Reject standalone single non-digit noise characters (e.g. ".", "-", "a")
	if len(text) == 1 and not text.isdigit():
		return ""

	# Remove stray isolated single-character letter artifacts attached to multi-word text
	# Example: "a Fan On." -> "Fan On.", "H HEAT PUMP" -> "HEAT PUMP"
	text = re.sub(r"^[a-zA-Z]\s+(?=[a-zA-Z0-9]{2,})", "", text).strip()
	text = re.sub(r"\s+[a-zA-Z]$", "", text).strip()

	return text


def _field_empty_value(field_name: str) -> Any:
	if field_name == "mode":
		return "UNKNOWN"
	if field_name == "temperature":
		return None
	return ""


# =============================================================================
# ICON MATCHING HELPERS
# =============================================================================

_ICON_TEMPLATES: dict[str, dict[str, np.ndarray]] = {}

ICON_STATE_MAPPINGS = {
	"wifi_icon": {
		"wifi_connected": "CONNECTED",
		"wifi_not_connected": "NOT_CONNECTED",
	},
	"schedule_icon": {
		"schedule_running": "RUNNING",
		"schedule_not_running": "NOT_RUNNING",
	},
}


def _classify_icon_field(roi_gray: np.ndarray, field_name: str) -> tuple[str, str]:
	"""Classify an icon by structural XOR overlap against normalized templates."""
	_require_cv2()
	templates_dict = _load_icon_templates().get(field_name, {})

	if not templates_dict or roi_gray is None or roi_gray.size == 0:
		return "UNKNOWN", "UNKNOWN"

	if len(roi_gray.shape) == 3:
		roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)

	_, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return "UNKNOWN", "UNKNOWN"

	valid_contours = [c for c in contours if cv2.contourArea(c) > 5]
	if not valid_contours:
		valid_contours = [max(contours, key=cv2.contourArea)]

	x, y, w, h = cv2.boundingRect(np.concatenate(valid_contours))
	if w < 5 or h < 5:
		return "UNKNOWN", "UNKNOWN"

	crop = roi_thresh[y : y + h, x : x + w]
	max_dim = max(w, h)
	pad_top = (max_dim - h) // 2
	pad_bottom = max_dim - h - pad_top
	pad_left = (max_dim - w) // 2
	pad_right = max_dim - w - pad_left
	squared_crop = cv2.copyMakeBorder(
		crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
	)

	roi_norm = cv2.resize(squared_crop, (64, 64), interpolation=cv2.INTER_AREA)
	_, roi_norm = cv2.threshold(roi_norm, 127, 255, cv2.THRESH_BINARY)

	scores: dict[str, float] = {}

	for state_key, template_norm in templates_dict.items():
		if template_norm.shape != (64, 64):
			scores[state_key] = float("inf")
			continue

		diff = cv2.bitwise_xor(roi_norm, template_norm)
		score = np.sum(diff == 255) / (64.0 * 64.0)
		scores[state_key] = float(score)

	best_state_key = min(scores, key=scores.get)
	parsed_state = ICON_STATE_MAPPINGS.get(field_name, {}).get(best_state_key, "UNKNOWN")
	return best_state_key, parsed_state


def _load_icon_templates() -> dict[str, dict[str, np.ndarray]]:
	"""Lazy load icon templates from ./templates directory into grayscale memory cache."""
	global _ICON_TEMPLATES
	if _ICON_TEMPLATES:
		return _ICON_TEMPLATES

	base_dir = Path(__file__).resolve().parent / "templates"
	if not base_dir.exists():
		base_dir = Path("./templates")

	template_files = {
		"wifi_icon": {
			"wifi_connected": base_dir / "wifi_connected.png",
			"wifi_not_connected": base_dir / "wifi_not_connected.png",
		},
		"schedule_icon": {
			"schedule_running": base_dir / "schedule_running.png",
			"schedule_not_running": base_dir / "schedule_not_running.png",
		},
	}

	_require_cv2()
	for field_key, templates in template_files.items():
		_ICON_TEMPLATES[field_key] = {}
		for state_key, file_path in templates.items():
			if not file_path.exists():
				continue

			img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
			if img is not None:
				_, template_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
				t_contours, _ = cv2.findContours(template_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

				if t_contours:
					tx, ty, tw, th = cv2.boundingRect(np.concatenate(t_contours))
					t_crop = template_thresh[ty : ty + th, tx : tx + tw]

					t_max_dim = max(tw, th)
					t_pad_top = (t_max_dim - th) // 2
					t_pad_bottom = t_max_dim - th - t_pad_top
					t_pad_left = (t_max_dim - tw) // 2
					t_pad_right = t_max_dim - tw - t_pad_left
					t_squared_crop = cv2.copyMakeBorder(
						t_crop, t_pad_top, t_pad_bottom, t_pad_left, t_pad_right, cv2.BORDER_CONSTANT, value=0
					)

					template_norm = cv2.resize(t_squared_crop, (64, 64), interpolation=cv2.INTER_AREA)
					_, template_norm = cv2.threshold(template_norm, 127, 255, cv2.THRESH_BINARY)
					_ICON_TEMPLATES[field_key][state_key] = template_norm
				else:
					_ICON_TEMPLATES[field_key][state_key] = img

	return _ICON_TEMPLATES


# =============================================================================
# VISUALIZATION & OVERLAY HELPERS
# =============================================================================

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
