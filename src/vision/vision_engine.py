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
DEBUG_OCR_CANDIDATES = False


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


def _expand_display_bottom(points: np.ndarray, extra_height_ratio: float = 0.12) -> np.ndarray:
	"""Extend the detected border downward so bottom-row content stays inside the warp."""
	rect = _order_points(points).copy()
	top_left, top_right, bottom_right, bottom_left = rect
	display_height = max(
		np.linalg.norm(top_right - bottom_right),
		np.linalg.norm(top_left - bottom_left),
	)
	if display_height <= 0:
		return rect

	extension = max(1, int(round(display_height * extra_height_ratio)))
	rect[2][1] += extension
	rect[3][1] += extension
	return rect


def _build_display_mask(frame: np.ndarray) -> np.ndarray:
	"""Keep the LCD body and near-edge signal while suppressing bezel noise."""
	_require_cv2()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Orange LCD body.
	lower_orange = np.array([5, 85, 60])
	upper_orange = np.array([28, 255, 255])
	orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

	# Bright pixels are useful for the illuminated border and white text, but
	# only when they are near the orange LCD body. This prevents bezel glare
	# from expanding the contour outward into the black frame.
	_, bright_mask = cv2.threshold(gray, 132, 255, cv2.THRESH_BINARY)
	bright_mask = cv2.GaussianBlur(bright_mask, (3, 3), 0)
	edge_mask = cv2.Canny(gray, 35, 110)
	edge_mask = cv2.dilate(edge_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

	anchor = cv2.dilate(
		orange_mask,
		cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
		iterations=1,
	)
	bright_near_anchor = cv2.bitwise_and(bright_mask, anchor)
	edge_near_anchor = cv2.bitwise_and(edge_mask, anchor)

	# Start from the orange body, then add only nearby bright/edge evidence.
	mask = cv2.bitwise_or(orange_mask, bright_near_anchor)
	mask = cv2.bitwise_or(mask, edge_near_anchor)
	kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
	kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
	mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
	return mask


def _find_display_contour(mask: np.ndarray, min_area: int = 3000) -> Optional[np.ndarray]:
	"""Find the 4-corner shape that best matches the LCD window geometry."""
	_require_cv2()
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return None

	best_contour = None
	best_score = float("inf")
	target_ratio = CURRENT_LAYOUT.display_aspect_ratio
	ratio_tolerance = 0.32
	area_tolerance = 0.55

	def _contour_box(candidate: np.ndarray) -> np.ndarray:
		rotated = cv2.minAreaRect(candidate)
		points = cv2.boxPoints(rotated)
		return points.reshape(4, 1, 2).astype("float32")

	for contour in contours:
		area = cv2.contourArea(contour)
		if area < min_area:
			continue

		hull = cv2.convexHull(contour)
		contour_to_score = hull if cv2.contourArea(hull) >= area else contour

		perimeter = cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour_to_score, 0.018 * perimeter, True)
		if len(approx) < 4:
			approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)

		candidate = approx if len(approx) >= 4 else hull
		(x, y), (width, height), angle = cv2.minAreaRect(candidate)
		short_side = min(width, height)
		long_side = max(width, height)
		if short_side <= 0 or long_side <= 0:
			continue

		ratio = short_side / long_side
		ratio_error = abs(ratio - target_ratio)
		if ratio_error > ratio_tolerance:
			continue

		box_area = width * height
		if box_area <= 0:
			continue
		fill_ratio = area / box_area
		if fill_ratio < area_tolerance:
			continue

		# Prefer the largest stable outer shell because the LCD border is the
		# actual warp anchor; the bright interior should not clip the edges.
		score = ratio_error - (area / 1_000_000.0) - (fill_ratio * 0.15)
		if score < best_score:
			best_contour = _contour_box(candidate)
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
		candidate = approx if len(approx) >= 4 else hull
		if area > best_area:
			best_contour = _contour_box(candidate)
			best_area = area

	if best_contour is not None:
		return best_contour

	# Final fallback: fit a rotated rectangle to the largest viable contour.
	# This recovers detection when contour simplification misses exactly 4 points.
	largest_contour = None
	largest_area = 0.0
	for contour in contours:
		area = cv2.contourArea(contour)
		if area < max(600, min_area // 3):
			continue
		if area > largest_area:
			largest_area = area
			largest_contour = cv2.convexHull(contour)

	if largest_contour is not None:
		return _contour_box(largest_contour)

	return best_contour


def _prepare_ocr_binary(warped: np.ndarray) -> np.ndarray:
	"""Convert to a high-contrast black/white image for OCR."""
	_require_cv2()
	gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
	_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return thresh


def _extract_warped_variants(binary: np.ndarray, field_name: str) -> list[np.ndarray]:
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


def _extract_fallback_variants(frame: np.ndarray, field_name: str) -> list[np.ndarray]:
	"""Build conservative OCR variants when the display contour is not found."""
	_require_cv2()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
	field = CURRENT_LAYOUT.fields[field_name]
	roi = field.fallback.crop(gray)
	roi = cv2.GaussianBlur(roi, (3, 3), 0)
	_, otsu = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return [roi, otsu, cv2.bitwise_not(otsu)]


def _clean_text(raw_text: str) -> str:
	"""Collapse OCR whitespace into a stable single-space form."""
	return re.sub(r"\s+", " ", raw_text).strip()


def _parse_mode(raw_text: str) -> str:
	cleaned = _clean_text(raw_text).upper()
	if not cleaned or cleaned == "UNKNOWN":
		return "UNKNOWN"

	cleaned = re.sub(r"^MODE\s*:\s*", "", cleaned)
	cleaned = re.sub(r"^MODE\s+", "", cleaned).strip()
	return cleaned or "UNKNOWN"


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
	text = _clean_text(raw_text)
	text = text.replace("_", " ")
	text = _clean_text(text).strip(" |:;,_-.")
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


def _parse_field_value(field_name: str, raw_text: str) -> Any:
	if field_name == "mode":
		return _parse_mode(raw_text)
	if field_name == "temperature":
		return _parse_temperature(raw_text)
	if field_name.startswith("dashboard_info_line_"):
		return _parse_info_line(raw_text)
	return _clean_text(raw_text)


def _score_temperature(raw_text: str, value: Any) -> float:
	if not isinstance(value, int):
		return -1.0

	score = 0.0
	min_temp, max_temp = CURRENT_LAYOUT.temperature_range_f
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


def _ocr_field(field_name: str, variants: list[np.ndarray]) -> tuple[str, Any]:
	"""OCR a field from multiple variants and keep the best-scoring candidate."""
	_require_pytesseract()
	field = CURRENT_LAYOUT.fields[field_name]
	best_raw = ""
	best_value: Any = _field_empty_value(field_name)
	best_score = -1.0
	candidate_debug: list[tuple[float, str, Any]] = []

	for variant in variants:
		raw = _clean_text(pytesseract.image_to_string(variant, config=field.tesseract_config))
		value = _parse_field_value(field_name, raw)
		score = _score_field_candidate(field_name, raw, value)

		if DEBUG_OCR_CANDIDATES:
			candidate_debug.append((score, raw, value))

		if score > best_score:
			best_raw = raw
			best_value = value
			best_score = score

	if DEBUG_OCR_CANDIDATES:
		top = sorted(candidate_debug, key=lambda item: item[0], reverse=True)[:2]
		print(f"[OCR DEBUG] {field_name} top candidates: {top}")

	return best_raw, best_value


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
	mask = _build_display_mask(frame)
	display_contour = _find_display_contour(mask)

	if display_contour is None:
		overlay = _draw_roi_overlay(frame, use_fallback_rois=True)
	else:
		source_points = _expand_display_bottom(display_contour.reshape(4, 2))
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
	mask = _build_display_mask(frame)
	display_contour = _find_display_contour(mask)
	fields_result: dict[str, dict[str, Any]] = {}

	if display_contour is None:
		for field_name in CURRENT_LAYOUT.fields:
			raw, val = _ocr_field(field_name, _extract_fallback_variants(frame, field_name))
			fields_result[field_name] = {"raw": raw, "value": val}
		
		return OCRReadout(
			display_found=False,
			current_menu_key=current_menu_key,
			fields=fields_result,
		)

	warped = _warp_image(frame, _expand_display_bottom(display_contour.reshape(4, 2)))
	binary = _prepare_ocr_binary(warped)

	for field_name in CURRENT_LAYOUT.fields:
		raw, val = _ocr_field(field_name, _extract_warped_variants(binary, field_name))
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
		empty_fields = {
			name: {"raw": "", "value": _field_empty_value(name)}
			for name in CURRENT_LAYOUT.fields
		}
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


def get_warped_display(frame: np.ndarray) -> Optional[np.ndarray]:
	"""Extract a front-facing view of the display from a camera frame, or None if not found."""
	mask = _build_display_mask(frame)
	display_contour = _find_display_contour(mask)

	if display_contour is None:
		return None

	return _warp_image(frame, _expand_display_bottom(display_contour.reshape(4, 2)))
