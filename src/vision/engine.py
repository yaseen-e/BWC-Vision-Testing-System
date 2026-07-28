"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/vision/engine.py - Vision/OCR Engine
Captures camera frames, isolates the display region, extracts mode/temperature via RapidOCR,
and classifies icon states via multi-metric shape/template matching.
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
	frame: Optional[np.ndarray],
	current_menu_key: str,
	menu_fields: tuple[OCRField, ...],
) -> OCRReadout:
	"""Read fields from the display in one frame based on current context using RapidOCR."""
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
			# Pass clean crop directly to the icon classifier
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
	"""Extract a front-facing view of the display from a camera frame, or None if not found."""
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
	"""Persist a calibration image that shows every OCR ROI on the current frame."""
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
	"""Initialize RapidOCR singleton letting DBNet handle word segmentation natively."""
	global _RAPID_OCR
	if _RAPID_OCR is not None:
		return _RAPID_OCR

	_require_rapidocr()
	try:
		_RAPID_OCR = RapidOCR(
			params={
				"Det.unclip_ratio": 1.5,
				"Det.use_dilation": False,
				"Det.box_thresh": 0.45,
				"Det.thresh": 0.25,
				"Global.text_score": 0.50,
			}
		)
	except Exception:
		try:
			_RAPID_OCR = RapidOCR(
				det_unclip_ratio=1.5,
				det_use_dilation=False,
				det_box_thresh=0.45,
				det_thresh=0.25,
				text_score=0.50,
			)
		except Exception:
			_RAPID_OCR = RapidOCR()
	return _RAPID_OCR


# =============================================================================
# DISPLAY SEGMENTATION & GEOMETRY
# =============================================================================

def _build_display_mask(frame: np.ndarray) -> np.ndarray:
	"""Build a mask for the emissive display window."""
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
	"""Find the raw display rectangle."""
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
	"""Only extend when the current ContextNode includes status bar fields."""
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
	"""Warp display area into flat front-facing view."""
	_require_cv2()
	if not _should_use_extended_warp(menu_fields):
		return orange_contour, _warp_image(frame, orange_contour)

	ordered = _order_points(orange_contour)
	top_left, top_right, bottom_right, bottom_left = ordered

	v_left = bottom_left - top_left
	v_right = bottom_right - top_right

	ext_factor = 10.0 / 90.0
	extended_bottom_left = bottom_left + v_left * ext_factor
	extended_bottom_right = bottom_right + v_right * ext_factor
	extended_contour = np.array([top_left, top_right, extended_bottom_right, extended_bottom_left], dtype="float32")

	warped_extended = _warp_image(frame, extended_contour)
	return extended_contour, warped_extended


def _warp_image(image: np.ndarray, points: np.ndarray) -> np.ndarray:
	"""Flatten angled display into front-facing view."""
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
# HIGH-PRECISION IMAGE PREPARATION
# =============================================================================

def _prepare_ocr_grayscale(warped: np.ndarray) -> np.ndarray:
	"""Convert warped image to grayscale."""
	_require_cv2()
	if len(warped.shape) == 3:
		return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	return warped


def _crop_roi(prepared: np.ndarray, field: Any) -> np.ndarray:
	"""Crop field ROI from image array."""
	if field is not None and hasattr(field, "ideal"):
		return field.ideal.crop(prepared)
	return prepared


def _normalize_backlit_crop(crop: np.ndarray) -> np.ndarray:
	"""Normalize backlit LCD illumination using soft background division while preserving original polarity."""
	if len(crop.shape) == 3:
		l_channel = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
	else:
		l_channel = crop.copy()

	h, w = l_channel.shape[:2]
	if h < 5 or w < 5:
		return l_channel

	ksize = max(15, (min(h, w) // 2) | 1)
	bg = cv2.GaussianBlur(l_channel, (ksize, ksize), 0)

	norm = cv2.divide(l_channel, np.maximum(bg, 1), scale=255)

	p2, p98 = np.percentile(norm, (2, 98))
	if p98 > p2:
		norm = np.clip((norm - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8)
	else:
		norm = norm.astype(np.uint8)

	return norm


def _generate_digit_variants(roi_gray: np.ndarray) -> list[np.ndarray]:
	"""
	Locate digit contours, crop tightly, and scale down to standard OCR height (48px).
	Guarantees DBNet detection on giant LCD numbers.
	"""
	_require_cv2()
	variants: list[np.ndarray] = []

	# Isolate bright digits on dark background
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

	# Scale tight digit crop to standard 48px OCR height
	ch, cw = digit_crop.shape[:2]
	if ch > 0 and cw > 0:
		target_h = 48
		scale = target_h / float(ch)
		target_w = max(10, int(cw * scale))
		scaled_digits = cv2.resize(digit_crop, (target_w, target_h), interpolation=cv2.INTER_AREA)

		# Variant 1: Inverted Black-on-White with padding
		inverted = cv2.bitwise_not(scaled_digits)
		v1 = cv2.copyMakeBorder(inverted, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
		variants.append(v1)

		# Variant 2: Binary Inverted
		_, bin_inv = cv2.threshold(scaled_digits, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		v2 = cv2.copyMakeBorder(bin_inv, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
		variants.append(v2)

		# Variant 3: Standard White-on-Black
		v3 = cv2.copyMakeBorder(scaled_digits, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
		variants.append(v3)

	return variants


def _extract_warped_variants(prepared: np.ndarray, field: Any = None) -> list[np.ndarray]:
	"""Extract clean grayscale variants scaled and softly padded using border replication."""
	_require_cv2()
	roi = _crop_roi(prepared, field)

	if roi is None or roi.size == 0:
		return []

	field_name = getattr(field, "name", "")

	# Check for giant temperature digits
	if field_name == "temperature":
		if len(roi.shape) == 3:
			roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		return _generate_digit_variants(roi)
	
	h, w = roi.shape[:2]
	if h == 0 or w == 0:
		return []

	# Target standard OCR height (~48px)
	target_h = 48
	scale = target_h / float(h)
	new_w = max(16, int(w * scale))

	interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
	resized = cv2.resize(roi, (new_w, target_h), interpolation=interp)

	norm_gray = _normalize_backlit_crop(resized)

	# Use BORDER_REPLICATE rather than hard white, preventing fake contrast borders at edges
	pad = 12
	bg_color = int(np.median(norm_gray))
	v1 = cv2.copyMakeBorder(norm_gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg_color)

	# Variant 2: Simple contrast stretch
	p2, p98 = np.percentile(resized, (2, 98))
	if p98 > p2:
		v2_base = np.clip((resized.astype(np.float32) - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8)
	else:
		v2_base = resized.copy()
	v2 = cv2.copyMakeBorder(v2_base, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

	# Variant 3: Bilateral filter to smooth LCD backlight pixel noise
	v3_base = cv2.bilateralFilter(resized, d=5, sigmaColor=50, sigmaSpace=50)
	v3 = cv2.copyMakeBorder(v3_base, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

	return [v1, v2, v3]


# =============================================================================
# OCR PROCESSING & SPATIAL LINE RECONSTRUCTION
# =============================================================================

def _ocr_field(field: OCRField, variants: list[np.ndarray]) -> tuple[str, Any]:
	"""Extract text using RapidOCR with geometry & confidence filtering."""
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

		# Whitelist character filtering
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

		if best_score > 6.0:
			break

	return best_raw, best_value


def _parse_and_filter_rapidocr_boxes(
	rapid_output: Optional[list[Any]]
) -> tuple[str, float]:
	"""Filter boxes and group by line, relying purely on OCR model predictions."""
	if not rapid_output:
		return "", 0.0

	items_to_keep: list[dict[str, Any]] = []

	for item in rapid_output:
		if not item or len(item) < 3:
			continue

		box, text, conf = item[0], str(item[1]).strip(), float(item[2])

		if not text or conf < 0.30:
			continue
		if len(text) == 1 and conf < 0.45:
			continue

		box_points = np.array(box, dtype=np.float32)
		min_x = float(np.min(box_points[:, 0]))
		min_y, max_y = float(np.min(box_points[:, 1])), float(np.max(box_points[:, 1]))
		height = max_y - min_y
		center_y = (min_y + max_y) / 2.0

		items_to_keep.append({
			"text": text,
			"conf": conf,
			"min_x": min_x,
			"center_y": center_y,
			"height": height,
		})

	if not items_to_keep:
		return "", 0.0

	# Group items into visual rows by Y-center
	items_to_keep.sort(key=lambda k: k["center_y"])

	rows: list[list[dict[str, Any]]] = []
	for item in items_to_keep:
		placed = False
		for row in rows:
			avg_cy = sum(r["center_y"] for r in row) / len(row)
			avg_h = sum(r["height"] for r in row) / len(row)
			if abs(item["center_y"] - avg_cy) < max(8.0, avg_h * 0.6):
				row.append(item)
				placed = True
				break
		if not placed:
			rows.append([item])

	line_strings: list[str] = []
	all_confidences: list[float] = []

	for row in rows:
		row.sort(key=lambda k: k["min_x"])
		row_text = " ".join(item["text"] for item in row)
		all_confidences.extend(item["conf"] for item in row)

		if row_text.strip():
			line_strings.append(row_text.strip())

	full_text = " ".join(line_strings)
	full_text = re.sub(r"\s+", " ", full_text).strip()
	mean_conf = float(np.mean(all_confidences)) if all_confidences else 0.0

	return full_text, mean_conf


def _score_field_candidate(field_name: str, raw_text: str, value: Any, conf: float) -> float:
	"""Score candidate value based on domain validity and model confidence."""
	if value is None or (isinstance(value, str) and (value == "UNKNOWN" or not value)):
		return -1.0

	score = conf * 2.0

	if field_name == "temperature" and isinstance(value, int):
		if 60 <= value <= 199:
			score += 5.0
		else:
			return -1.0
	elif field_name == "mode" and isinstance(value, str) and value != "UNKNOWN":
		if len(value) >= 3:
			score += 4.0
		else:
			score += 1.0
	elif field_name.startswith("dashboard_info_line"):
		if isinstance(value, str) and len(value) >= 2:
			score += 2.0
			if len(value) >= 5:
				score += 2.0
	elif field_name == "time_field":
		if isinstance(value, str) and re.search(r"\d{1,2}:\d{2}", value):
			score += 4.0
		else:
			return -1.0
	elif field_name == "date_field":
		if isinstance(value, str) and re.search(r"\d{2}/\d{2}/\d{2}", value):
			score += 4.0

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
	if field_name.startswith("dashboard_info_line"):
		return _parse_dashboard_info(raw_text)
	return raw_text.strip()


def _parse_mode(raw_text: str) -> str:
	"""Extract mode by stripping leading 'MODE:' label prefix cleanly."""
	if not raw_text:
		return "UNKNOWN"

	text = raw_text.strip()
	cleaned = re.sub(r"^(MODE|M0DE)\s*[:\-]?\s*", "", text, flags=re.IGNORECASE).strip()

	if not cleaned:
		return "UNKNOWN"

	return cleaned.upper()


def _parse_temperature(raw_text: str) -> Optional[int]:
	"""Extract integer temperature value from raw OCR string."""
	if not raw_text:
		return None

	matches = re.findall(r"\d+", raw_text)
	for match in matches:
		val = int(match)
		if 50 <= val <= 199:
			return val

	return None


def _parse_time(raw_text: str) -> str:
	"""Extract and normalize time string (e.g., '10:27 AM', '12:45 PM')."""
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
	"""Clean dashboard info line strings by removing border artifacts."""
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
# GENERALIZED MULTI-METRIC ICON CLASSIFIER
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
	"""Classify icon via shape/template matching."""
	_require_cv2()
	templates_dict = _load_icon_templates().get(field_name, {})

	if not templates_dict or roi_gray is None or roi_gray.size == 0:
		return "UNKNOWN", "UNKNOWN"

	if len(roi_gray.shape) == 3:
		roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)

	blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)
	_, roi_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	perimeter = np.concatenate([
		roi_thresh[0, :], roi_thresh[-1, :],
		roi_thresh[:, 0], roi_thresh[:, -1]
	])
	if np.mean(perimeter) > 127:
		roi_thresh = cv2.bitwise_not(roi_thresh)

	contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return "UNKNOWN", "UNKNOWN"

	valid_contours = [c for c in contours if cv2.contourArea(c) > 5]
	if not valid_contours:
		return "UNKNOWN", "UNKNOWN"

	all_pts = np.concatenate(valid_contours)
	x, y, w, h = cv2.boundingRect(all_pts)
	if w < 4 or h < 4:
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

		shape_diff = cv2.matchShapes(roi_norm, template_norm, cv2.CONTOURS_MATCH_I1, 0.0)
		corr_matrix = cv2.matchTemplate(roi_norm, template_norm, cv2.TM_CCOEFF_NORMED)
		max_corr = float(np.max(corr_matrix)) if corr_matrix is not None else 0.0
		corr_dist = 1.0 - max(0.0, max_corr)

		xor_diff = cv2.bitwise_xor(roi_norm, template_norm)
		xor_dist = np.sum(xor_diff == 255) / (64.0 * 64.0)

		composite = (0.45 * shape_diff) + (0.35 * corr_dist) + (0.20 * xor_dist)
		scores[state_key] = float(composite)

	best_state_key = min(scores, key=scores.get)
	parsed_state = ICON_STATE_MAPPINGS.get(field_name, {}).get(best_state_key, "UNKNOWN")
	return best_state_key, parsed_state


def _load_icon_templates() -> dict[str, dict[str, np.ndarray]]:
	"""Lazy load icon templates into memory cache."""
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
			"schedule_not_running": ["schedule_not_running.png", "schedule_not_running.jpeg", "schedule_not_running.jpeg"],
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


def _project_roi_box(box: ROIBox, inverse_transform: np.ndarray, warped_width: int, warped_height: int) -> np.ndarray:
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
