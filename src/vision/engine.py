"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/vision/engine.py - Vision/OCR Engine

Captures camera frames, isolates the outermost blue LCD display region, extracts 
text via RapidOCR, and classifies icon states via multi-metric template matching.
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
	"""Single OCR result payload for upstream system logic."""

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
	"""One-call helper: captures camera frame and runs OCR display reading pipeline."""
	frame = capture_frame()
	return read_display(
		frame,
		current_menu_key,
		menu_fields,
	)


def read_display(
	frame: Optional[np.ndarray],
	current_menu_key: str,
	menu_fields: tuple[OCRField, ...],
) -> OCRReadout:
	"""Read fields from the display in one frame based on current layout context."""
	_require_cv2()
	_require_rapidocr()

	if frame is None or frame.size == 0:
		return OCRReadout(
			display_found=False,
			current_menu_key=current_menu_key,
			fields={
				field.name: {"raw": "", "value": _field_empty_value(field.name)}
				for field in menu_fields
			},
		)

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
	prepared_gray = _prepare_ocr_grayscale(warped)

	fields_result: dict[str, dict[str, Any]] = {}
	for field in menu_fields:
		if field.name in ("wifi_icon", "schedule_icon") or field.name.endswith("_icon"):
			unpadded_roi = _crop_roi(prepared_gray, field)
			raw, val = _classify_icon_field(unpadded_roi, field.name)
		else:
			raw, val = _ocr_field(field, _extract_warped_variants(prepared_gray, field))

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
	"""Extract a front-facing warped view of the display from a frame."""
	_ = current_menu_key
	if frame is None or frame.size == 0:
		return None

	mask = _build_display_mask(frame)
	display_contour = _find_display_contour(frame, mask)

	if display_contour is None:
		return None

	_, warped = _process_display_contour_and_warp(frame, display_contour, menu_fields=menu_fields)
	return warped


def save_roi_ocr_overlay(
	capture_dir: Path,
	frame: Optional[np.ndarray],
	capture_id: Optional[str],
	menu_fields: tuple[OCRField, ...],
	current_menu_key: Optional[str] = None,
) -> Optional[Path]:
	"""Persist a calibration image showing bounding display polygons and field ROIs."""
	_require_cv2()
	if frame is None or frame.size == 0:
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
	"""Warm up camera and OCR singletons to flush initial cold-start delays."""
	_ = _get_camera()
	_ = _get_rapid_ocr()


def is_camera_available() -> bool:
	return True


def shutdown() -> None:
	"""Stop camera device cleanly."""
	global _CAMERA
	if _CAMERA is None:
		return

	_CAMERA.stop()
	_CAMERA = None


# =============================================================================
# SINGLETONS & HARDWARE INITIALIZATION
# =============================================================================

def _require_cv2() -> None:
	if cv2 is None:
		raise RuntimeError("opencv-python is required for OCR image processing")


def _require_rapidocr() -> None:
	if RapidOCR is None:
		raise RuntimeError(
			"rapidocr_onnxruntime is required for OCR text extraction."
		)


def _get_camera() -> Any:
	"""Initialize Picamera2 hardware controls and lock optical parameters."""
	global _CAMERA
	if _CAMERA is not None:
		return _CAMERA

	from picamera2 import Picamera2  # type: ignore

	camera_info = Picamera2.global_camera_info()
	camera_num = camera_info[0]["Num"]
	camera = Picamera2(camera_num=camera_num)
	config = camera.create_preview_configuration(main={"size": (1280, 720)})  # Native frame size: 1280x720 pixels (16:9 aspect)
	camera.configure(config)
	camera.start()
	camera.set_controls({
		"AfMode": 0,          # 0 = Manual AF mode (disables continuous auto-focus search jitter)
		"LensPosition": 9.1,  # LensPosition = 1 / focal_distance_meters = 1 / 0.110m (11cm target distance) = 9.09 ~ 9.1 diopters
	})

	time.sleep(0.6)  # 0.6s delay allows camera hardware exposure & gain auto-convergence to settle
	for _ in range(6):  # Read 6 frame buffers to drain stale/underexposed images from libcamera ISP queue
		try:
			camera.capture_array()
		except Exception:
			pass

	try:
		# Lock current converged AE/AWB parameters to prevent illumination shifting during testing
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
	"""Initialize RapidOCR singleton tuned for tight text box bounding."""
	global _RAPID_OCR
	if _RAPID_OCR is not None:
		return _RAPID_OCR

	_require_rapidocr()
	try:
		_RAPID_OCR = RapidOCR(
			params={
				"Det.unclip_ratio": 1.1,  # Unclip ratio scales box expansion: area_unclipped = area * (1 + 1.1). Reduced from 1.5 to keep words distinct
				"Det.box_thresh": 0.30,   # Minimum score threshold (0.30) to accept a candidate bounding box
				"Det.thresh": 0.20,       # Binarization segmentation threshold (0.20) for DBNet text detector
				"Global.text_score": 0.20, # Text confidence threshold (0.20) to accept recognized character sequence
			}
		)
	except Exception:
		try:
			_RAPID_OCR = RapidOCR(
				det_unclip_ratio=1.1,
				det_box_thresh=0.30,
				det_thresh=0.20,
				text_score=0.20,
			)
		except Exception:
			_RAPID_OCR = RapidOCR()
	return _RAPID_OCR


# =============================================================================
# DISPLAY SEGMENTATION & GEOMETRY
# =============================================================================

def _build_display_mask(frame: np.ndarray) -> np.ndarray:
	"""Build a clean mask targeting dim or bright blue LCD screen backgrounds."""
	_require_cv2()
	if frame is None or frame.size == 0:
		raise ValueError("frame must be a non-empty image")

	b_chan = frame[:, :, 0]
	r_chan = frame[:, :, 2]

	# 1. BGR Blue Dominance: I_blue_diff = max(0, Blue - Red).
	# For blue screen backgrounds (dim or bright) and cyan boxes, Blue > Red (diff > 15).
	# Neutral gray glare, white reflections, and black chassis yield Blue ~ Red (diff ~ 0).
	blue_diff = cv2.subtract(b_chan, r_chan)
	_, blue_diff_thresh = cv2.threshold(blue_diff, 15, 255, cv2.THRESH_BINARY)  # Threshold = 15 units blue dominance

	# 2. Targeted HSV Blue Masking
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	# OpenCV HSV bounds: Hue in [80, 135] covers cyan (~90) to deep blue (~120).
	# Saturation >= 25 captures desaturated/dim blue edges. Value >= 20 captures dim LCD backlights.
	lower_blue = np.array([80, 25, 20], dtype=np.uint8)
	upper_blue = np.array([135, 255, 255], dtype=np.uint8)
	blue_hsv_mask = cv2.inRange(hsv, lower_blue, upper_blue)

	# Combine blue dominance and HSV blue detection (OR operation between targeted blue features)
	combined_blue = cv2.bitwise_or(blue_diff_thresh, blue_hsv_mask)

	# Morphological Closing: Mask_closed = (Mask (+) Kernel) (-) Kernel
	# 25x25 kernel bridges white text characters (S ~ 0) and dark UI gaps (<20px) inside the blue screen
	kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
	mask = cv2.morphologyEx(combined_blue, cv2.MORPH_CLOSE, kernel_close)

	# Morphological Opening: Removes isolated camera noise and small specular reflections (<5px)
	kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

	return mask


def _find_display_contour(frame: np.ndarray, mask: np.ndarray, min_area: int = 3000) -> Optional[np.ndarray]:
	"""Extract the true outermost rectangular LCD display contour from the screen mask."""
	_require_cv2()
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return None

	frame_area = max(1, mask.shape[0] * mask.shape[1])
	# Screen must occupy at least 5% of total camera frame area (1280x720 * 0.05 = 46,080 px^2)
	min_area = max(min_area, int(frame_area * 0.05))

	valid_contours: list[np.ndarray] = []

	for contour in contours:
		area = cv2.contourArea(contour)
		# Filter out candidate regions smaller than 5% or larger than 98% of total frame area
		if area < min_area or area > frame_area * 0.98:
			continue

		hull = cv2.convexHull(contour)
		hull_area = cv2.contourArea(hull)
		if hull_area <= 0:
			continue

		# Solidity = ContourArea / ConvexHullArea. Screen bounds have high solidity (>= 0.60)
		solidity = area / hull_area
		if solidity < 0.60:
			continue

		valid_contours.append(contour)

	if not valid_contours:
		return None

	# Merge point clouds of all valid display regions to guarantee the OUTERMOST bounding rectangle is chosen
	# (Prevents collapsing onto an internal active focus box if the outer dim blue border is segmented in pieces)
	all_screen_points = np.concatenate(valid_contours)
	rotated_rect = cv2.minAreaRect(all_screen_points)
	box_points = cv2.boxPoints(rotated_rect)

	return box_points.astype("float32")


def _should_use_extended_warp(menu_fields: Optional[tuple[OCRField, ...]]) -> bool:
	"""Check whether the active layout includes status bar fields requiring downward extension."""
	if not menu_fields:
		return False

	status_bar_names = {field.name for field in STATUS_BAR}
	current_field_names = {field.name for field in menu_fields}
	return status_bar_names.issubset(current_field_names)


def _process_display_contour_and_warp(
	frame: np.ndarray,
	display_contour: np.ndarray,
	menu_fields: Optional[tuple[OCRField, ...]] = None,
) -> tuple[np.ndarray, np.ndarray]:
	"""Warp display area, extending downward for status bar fields when required."""
	_require_cv2()
	if not _should_use_extended_warp(menu_fields):
		return display_contour, _warp_image(frame, display_contour)

	ordered = _order_points(display_contour)
	top_left, top_right, bottom_right, bottom_left = ordered

	# Compute left and right top-to-bottom edge vectors
	v_left = bottom_left - top_left
	v_right = bottom_right - top_right

	# Status bar extension factor: Height_status_bar / Height_main_blue_screen = 10px / 90px = 0.1111 (11.1% extension)
	ext_factor = 10.0 / 90.0
	extended_bottom_left = bottom_left + (v_left * ext_factor)
	extended_bottom_right = bottom_right + (v_right * ext_factor)
	extended_contour = np.array([top_left, top_right, extended_bottom_right, extended_bottom_left], dtype="float32")

	warped_extended = _warp_image(frame, extended_contour)
	return extended_contour, warped_extended


def _warp_image(image: np.ndarray, points: np.ndarray) -> np.ndarray:
	"""Flatten angled LCD display polygon into a perspective-corrected front-facing view."""
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
	"""Order 4 quad points: [top-left, top-right, bottom-right, bottom-left]."""
	points = np.asarray(points, dtype="float32").reshape(4, 2)
	ordered = np.zeros((4, 2), dtype="float32")
	
	point_sums = points.sum(axis=1)
	ordered[0] = points[np.argmin(point_sums)]  # Top-left has minimum (x + y) sum
	ordered[2] = points[np.argmax(point_sums)]  # Bottom-right has maximum (x + y) sum

	point_diffs = np.diff(points, axis=1)
	ordered[1] = points[np.argmin(point_diffs)] # Top-right has minimum (y - x) difference
	ordered[3] = points[np.argmax(point_diffs)] # Bottom-left has maximum (y - x) difference

	return ordered


# =============================================================================
# IMAGE PREPARATION & VARIANT GENERATION
# =============================================================================

def _prepare_ocr_grayscale(warped: np.ndarray) -> np.ndarray:
	"""Convert warped color display to 8-bit single-channel grayscale."""
	_require_cv2()
	if len(warped.shape) == 3:
		return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	return warped


def _crop_roi(prepared: np.ndarray, field: Any) -> np.ndarray:
	"""Crop field region of interest from normalized grayscale image."""
	if field is not None and hasattr(field, "ideal"):
		return field.ideal.crop(prepared)
	return prepared


def _normalize_backlit_crop(crop: np.ndarray) -> np.ndarray:
	"""Normalize uneven backlighting across ROI using Gaussian background estimation division."""
	if len(crop.shape) == 3:
		l_channel = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
	else:
		l_channel = crop.copy()

	h, w = l_channel.shape[:2]
	if h < 5 or w < 5:
		return l_channel

	# Kernel size forced to odd integer roughly half of minimum dimension (minimum 15px)
	ksize = max(15, (min(h, w) // 2) | 1)
	bg = cv2.GaussianBlur(l_channel, (ksize, ksize), 0)

	# Normalized = (L_channel / max(BG, 1)) * 255
	norm = cv2.divide(l_channel, np.maximum(bg, 1), scale=255)

	# Percentile contrast stretch (stretching 2nd to 98th percentile range to [0, 255])
	p2, p98 = np.percentile(norm, (2, 98))
	if p98 > p2:
		norm = np.clip((norm - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8)
	else:
		norm = norm.astype(np.uint8)

	return norm


def _get_border_bg_color(img: np.ndarray) -> int:
	"""Sample border pixels around image edge to calculate background padding fill value."""
	if img is None or img.size == 0:
		return 255
	border = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
	return int(np.median(border))


def _extract_warped_variants(prepared: np.ndarray, field: Any = None) -> list[np.ndarray]:
	"""Generate grayscale image variants optimized for OCR character recognition."""
	_require_cv2()
	roi = _crop_roi(prepared, field)

	if roi is None or roi.size == 0:
		return []

	h, w = roi.shape[:2]
	if h == 0 or w == 0:
		return []

	target_h = 48  # Target height = 48px (native input pixel height for CRNN text recognition networks)
	scale = target_h / float(h)
	new_w = max(16, int(w * scale))

	interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
	resized = cv2.resize(roi, (new_w, target_h), interpolation=interp)

	pad = 16  # 16px constant background border padding prevents edge character cropping during CRNN convolutions

	# Variant 1: Soft Normalized Grayscale (Preserves character stroke geometry)
	norm_gray = _normalize_backlit_crop(resized)
	bg1 = _get_border_bg_color(norm_gray)
	v1 = cv2.copyMakeBorder(norm_gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg1)

	# Variant 2: Clean Linear Rescale (Preserves original camera illumination gradients)
	bg2 = _get_border_bg_color(resized)
	v2 = cv2.copyMakeBorder(resized, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg2)

	# Variant 3: Soft Edge-Preserving Denoised Grayscale (d=5, sigma=30 filter suppresses sensor noise)
	denoised = cv2.bilateralFilter(resized, d=5, sigmaColor=30, sigmaSpace=30)
	bg3 = _get_border_bg_color(denoised)
	v3 = cv2.copyMakeBorder(denoised, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg3)

	return [v1, v2, v3]


# =============================================================================
# OCR PROCESSING & TEXT RECONSTRUCTION
# =============================================================================

def _ocr_field(field: OCRField, variants: list[np.ndarray]) -> tuple[str, Any]:
	"""Evaluate candidate OCR text across all extracted image variants."""
	field_name = field.name
	_require_rapidocr()
	ocr_engine = _get_rapid_ocr()

	best_raw = ""
	best_value: Any = _field_empty_value(field_name)
	best_score = -1.0

	for variant in variants:
		rapid_output, _ = ocr_engine(variant)
		raw, conf = _parse_and_filter_rapidocr_boxes(rapid_output)

		if not raw:
			continue

		# Character whitelist filtering
		if field.whitelisted_chars:
			allowed = set(field.whitelisted_chars)
			cleaned_chars = []
			for char in raw:
				if char.isspace():
					cleaned_chars.append(" ")
				elif char in allowed:
					cleaned_chars.append(char)
				elif char.upper() in allowed:
					cleaned_chars.append(char.upper())
				elif char.lower() in allowed:
					cleaned_chars.append(char.lower())
			raw = "".join(cleaned_chars)
			raw = re.sub(r"\s+", " ", raw).strip()

		value = _parse_field_value(field_name, raw)
		score = _score_field_candidate(field_name, raw, value, conf)

		if score > best_score:
			best_raw = raw
			best_value = value
			best_score = score

	return best_raw, best_value


def _parse_and_filter_rapidocr_boxes(
	rapid_output: Optional[list[Any]]
) -> tuple[str, float]:
	"""Sort detected bounding boxes top-to-bottom, left-to-right, and reconstruct text."""
	if not rapid_output:
		return "", 0.0

	items_to_keep: list[dict[str, Any]] = []

	for item in rapid_output:
		if not item or len(item) < 3:
			continue

		box, text, conf = item[0], str(item[1]).strip(), float(item[2])

		# Retain valid text tokens meeting minimum confidence threshold (0.20)
		if not text or conf < 0.20:
			continue

		box_points = np.array(box, dtype=np.float32)
		min_x = float(np.min(box_points[:, 0]))
		min_y = float(np.min(box_points[:, 1]))

		items_to_keep.append({
			"text": text,
			"conf": conf,
			"min_x": min_x,
			"min_y": min_y,
		})

	if not items_to_keep:
		return "", 0.0

	# Sort top-to-bottom within 12px line bands (round(min_y / 12.0)), then left-to-right by min_x
	items_to_keep.sort(key=lambda k: (round(k["min_y"] / 12.0), k["min_x"]))

	full_text = " ".join(item["text"] for item in items_to_keep)
	full_text = re.sub(r"\s+", " ", full_text).strip()
	mean_conf = float(np.mean([item["conf"] for item in items_to_keep]))

	return full_text, mean_conf


def _score_field_candidate(field_name: str, raw_text: str, value: Any, conf: float) -> float:
	"""Score candidate OCR result using domain rules and confidence metrics."""
	if value is None or (isinstance(value, str) and (value == "UNKNOWN" or not value)):
		return -1.0

	# Base score = 2.0 + (confidence * 2.0)
	score = 2.0 + (conf * 2.0)

	if field_name == "temperature" and isinstance(value, int):
		# BWC Water Heater temperature valid bounds: [50°F, 199°F]
		if 50 <= value <= 199:
			score += 3.0
		else:
			return -1.0
	elif field_name == "mode" and isinstance(value, str) and value != "UNKNOWN":
		score += 2.0
	elif field_name == "time_field":
		if isinstance(value, str) and re.search(r"\d{1,2}:\d{2}", value):
			score += 3.0
		else:
			return -1.0
	elif field_name == "date_field":
		if isinstance(value, str) and re.search(r"\d{2}/\d{2}/\d{2}", value):
			score += 3.0

	return score


# =============================================================================
# VALUE PARSING & SANITIZATION
# =============================================================================

def _parse_field_value(field_name: str, raw_text: str) -> Any:
	if field_name == "mode":
		return _parse_mode(raw_text)
	if field_name == "temperature":
		return _parse_temperature(raw_text)
	if field_name == "time_field":
		return _parse_time(raw_text)
	if field_name.startswith("dashboard_info_line") or "schedule" in field_name:
		return _parse_dashboard_info(raw_text)
	return raw_text.strip()


def _parse_mode(raw_text: str) -> str:
	if not raw_text:
		return "UNKNOWN"

	text = raw_text.strip()
	cleaned = re.sub(r"^(MODE|M0DE)\s*[:\-]?\s*", "", text, flags=re.IGNORECASE).strip()

	if not cleaned:
		return "UNKNOWN"

	return cleaned.upper()


def _parse_temperature(raw_text: str) -> Optional[int]:
	if not raw_text:
		return None

	matches = re.findall(r"\d+", raw_text)
	for match in matches:
		val = int(match)
		if 50 <= val <= 199:
			return val

	return None


def _parse_time(raw_text: str) -> str:
	if not raw_text:
		return ""

	text = raw_text.strip().upper()
	text = re.sub(r"(\d{1,2})[;\.\-\s]+(\d{2})", r"\1:\2", text)

	time_match = re.search(r"(\d{1,2}):(\d{2})", text)
	if time_match:
		hour = int(time_match.group(1))
		minute = time_match.group(2)
		if 1 <= hour <= 12:
			period = "PM" if "P" in text else "AM"
			return f"{hour}:{minute} {period}"

	return ""


def _parse_dashboard_info(raw_text: str) -> str:
	if not raw_text:
		return ""

	cleaned = raw_text.strip()
	cleaned = re.sub(r"^[^\w\s\.\,\!\?]+", "", cleaned)
	cleaned = re.sub(r"\s+", " ", cleaned).strip()
	return cleaned


def _field_empty_value(field_name: str) -> Any:
	if field_name == "mode":
		return "UNKNOWN"
	if field_name == "temperature":
		return None
	return ""


# =============================================================================
# MULTI-METRIC ICON CLASSIFICATION
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
	"""Classify icon state using composite shape distance, correlation, and XOR error."""
	_require_cv2()
	templates_dict = _load_icon_templates().get(field_name, {})

	if not templates_dict or roi_gray is None or roi_gray.size == 0:
		return "UNKNOWN", "UNKNOWN"

	if len(roi_gray.shape) == 3:
		roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)

	# 3x3 Gaussian blur smooths high-frequency noise prior to Otsu binarization
	blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)
	_, roi_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	# Invert polarity if image background is light (mean perimeter border value > 127)
	perimeter = np.concatenate([
		roi_thresh[0, :], roi_thresh[-1, :],
		roi_thresh[:, 0], roi_thresh[:, -1]
	])
	if np.mean(perimeter) > 127:
		roi_thresh = cv2.bitwise_not(roi_thresh)

	contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return "UNKNOWN", "UNKNOWN"

	# Filter out speckle noise contours (<5px area)
	valid_contours = [c for c in contours if cv2.contourArea(c) > 5]
	if not valid_contours:
		return "UNKNOWN", "UNKNOWN"

	all_pts = np.concatenate(valid_contours)
	x, y, w, h = cv2.boundingRect(all_pts)
	if w < 4 or h < 4:
		return "UNKNOWN", "UNKNOWN"

	crop = roi_thresh[y : y + h, x : x + w]

	# Pad cropped icon into a 1:1 square canvas
	max_dim = max(w, h)
	pad_top = (max_dim - h) // 2
	pad_bottom = max_dim - h - pad_top
	pad_left = (max_dim - w) // 2
	pad_right = max_dim - w - pad_left
	squared_crop = cv2.copyMakeBorder(
		crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
	)

	# Resize canvas to standard 64x64 pixel matrix (64x64 = 4096 px total area)
	roi_norm = cv2.resize(squared_crop, (64, 64), interpolation=cv2.INTER_AREA)
	_, roi_norm = cv2.threshold(roi_norm, 127, 255, cv2.THRESH_BINARY)

	scores: dict[str, float] = {}

	for state_key, template_norm in templates_dict.items():
		if template_norm.shape != (64, 64):
			scores[state_key] = float("inf")
			continue

		# 1. Hu Moments Shape Distance (CONTOURS_MATCH_I1 invariant to rotation and scale)
		shape_diff = cv2.matchShapes(roi_norm, template_norm, cv2.CONTOURS_MATCH_I1, 0.0)
		
		# 2. Normalized Cross-Correlation Distance: Distance = 1.0 - max(0, NCC_score)
		corr_matrix = cv2.matchTemplate(roi_norm, template_norm, cv2.TM_CCOEFF_NORMED)
		max_corr = float(np.max(corr_matrix)) if corr_matrix is not None else 0.0
		corr_dist = 1.0 - max(0.0, max_corr)

		# 3. Normalized Hamming Pixel XOR Difference Ratio: Ratio = sum(XOR == 255) / (64 * 64)
		xor_diff = cv2.bitwise_xor(roi_norm, template_norm)
		xor_dist = np.sum(xor_diff == 255) / (64.0 * 64.0)

		# Composite Score = 0.45 * Shape_Diff + 0.35 * Corr_Dist + 0.20 * XOR_Dist
		composite = (0.45 * shape_diff) + (0.35 * corr_dist) + (0.20 * xor_dist)
		scores[state_key] = float(composite)

	best_state_key = min(scores, key=scores.get)
	parsed_state = ICON_STATE_MAPPINGS.get(field_name, {}).get(best_state_key, "UNKNOWN")
	return best_state_key, parsed_state


def _load_icon_templates() -> dict[str, dict[str, np.ndarray]]:
	"""Lazy-load reference icon templates into memory cache."""
	global _ICON_TEMPLATES
	if _ICON_TEMPLATES:
		return _ICON_TEMPLATES

	base_dir = Path(__file__).resolve().parent / "templates"
	if not base_dir.exists():
		base_dir = Path("./templates")

	template_files = {
		"wifi_icon": {
			"wifi_connected": ["wifi_connected.png", "wifi_connected.jpg", "wifi_connected.jpeg"],
			"wifi_not_connected": ["wifi_not_connected.png", "wifi_not_connected.jpg", "wifi_not_connected.jpeg"],
		},
		"schedule_icon": {
			"schedule_running": ["schedule_running.png", "schedule_running.jpg", "schedule_running.jpeg"],
			"schedule_not_running": ["schedule_not_running.png", "schedule_not_running.jpg", "schedule_not_running.jpeg"],
		},
	}

	_require_cv2()
	for field_key, templates in template_files.items():
		_ICON_TEMPLATES[field_key] = {}
		for state_key, filenames in templates.items():
			file_path = None
			for fname in filenames:
				candidate = base_dir / fname
				if candidate.exists():
					file_path = candidate
					break

			if file_path is None:
				continue

			img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
			if img is None:
				continue

			blurred = cv2.GaussianBlur(img, (3, 3), 0)
			_, template_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

			perimeter = np.concatenate([
				template_thresh[0, :], template_thresh[-1, :],
				template_thresh[:, 0], template_thresh[:, -1]
			])
			if np.mean(perimeter) > 127:
				template_thresh = cv2.bitwise_not(template_thresh)

			t_contours, _ = cv2.findContours(template_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			valid_cnts = [c for c in t_contours if cv2.contourArea(c) > 5]

			if valid_cnts:
				tx, ty, tw, th = cv2.boundingRect(np.concatenate(valid_cnts))
				t_crop = template_thresh[ty : ty + th, tx : tx + tw]

				t_max_dim = max(tw, th)
				t_pad_top = (t_max_dim - th) // 2
				t_pad_bottom = t_max_dim - th - t_pad_top
				t_pad_left = (t_max_dim - tw) // 2
				t_pad_right = t_max_dim - tw - t_pad_left
				t_squared = cv2.copyMakeBorder(
					t_crop, t_pad_top, t_pad_bottom, t_pad_left, t_pad_right, cv2.BORDER_CONSTANT, value=0
				)

				template_norm = cv2.resize(t_squared, (64, 64), interpolation=cv2.INTER_AREA)
				_, template_norm = cv2.threshold(template_norm, 127, 255, cv2.THRESH_BINARY)
				_ICON_TEMPLATES[field_key][state_key] = template_norm
			else:
				_ICON_TEMPLATES[field_key][state_key] = cv2.resize(template_thresh, (64, 64))

	return _ICON_TEMPLATES


# =============================================================================
# VISUALIZATION & OVERLAY HELPERS
# =============================================================================

def _draw_roi_overlay(image: np.ndarray, fields: tuple[OCRField, ...], use_fallback_rois: bool) -> np.ndarray:
	"""Draw OCR ROI boundaries on camera frame for visual calibration and debugging."""
	_require_cv2()
	overlay = image.copy()
	height, width = overlay.shape[:2]

	for index, field in enumerate(fields):
		box = field.fallback if use_fallback_rois else field.ideal
		top = int(height * box.top)
		bottom = int(height * box.bottom)
		left = int(width * box.left)
		right = int(width * box.right)

		cv2.rectangle(overlay, (left, top), (right, bottom), (0, 0, 0), 3, cv2.LINE_AA)
		cv2.rectangle(overlay, (left, top), (right, bottom), (255, 255, 255), 1, cv2.LINE_AA)

		label_top = max(12, top - 6)
		label_origin = (left + 4, label_top)
		label = field.name.upper()
		cv2.putText(overlay, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
		cv2.putText(overlay, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

	return overlay


def _draw_polygon_outline(image: np.ndarray, points: np.ndarray, label: str) -> None:
	"""Draw high-contrast polygon outlines with text label overlay."""
	_require_cv2()
	polygon = points.astype("int32")
	cv2.polylines(image, [polygon], True, (0, 0, 0), 3, cv2.LINE_AA)
	cv2.polylines(image, [polygon], True, (255, 255, 255), 1, cv2.LINE_AA)

	anchor_x = int(polygon[:, 0].min()) + 4
	anchor_y = max(18, int(polygon[:, 1].min()) - 8)
	anchor = (anchor_x, anchor_y)
	cv2.putText(image, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
	cv2.putText(image, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _project_roi_box(box: ROIBox, inverse_transform: np.ndarray, warped_width: int, warped_height: int) -> np.ndarray:
	"""Project a normalized ROI box back into original unwarped frame coordinates."""
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
	"""Sanitize optional string identifiers into safe filename stems."""
	if not capture_id:
		return "roi_ocr"

	cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", capture_id).strip("._")
	return cleaned or "roi_ocr"