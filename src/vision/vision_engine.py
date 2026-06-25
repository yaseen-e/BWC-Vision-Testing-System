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

from .display_layouts import CURRENT_LAYOUT, OCRField


# Camera object is created lazily so non-Pi environments still run.
_CAMERA: Optional[Any] = None
DEBUG_OCR_CANDIDATES = False
ICON_STATE_TEMPLATES: dict[str, tuple[str, str]] = {
	"wifi_icon": ("vision/templates/wifi_on.png", "vision/templates/wifi_off.png"),
	"calendar_icon": ("vision/templates/schedule_running.png", "vision/templates/schedule_not_running.png"),
}


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


def _elongate_bottom_edge(points: np.ndarray, extra_height_ratio: float = 0.12) -> np.ndarray:
	"""Extend the detected border downward so the slim bottom icon bar stays inside the warp."""
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


def _display_mask(frame: np.ndarray) -> np.ndarray:
	"""Keep the back-lit LCD body and its bright edges while suppressing the bezel."""
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


def _resolve_template_path(template_path: str) -> Optional[Path]:
	"""Resolve template paths robustly across differing runtime working directories."""
	path_obj = Path(template_path)
	candidates: list[Path] = []

	if path_obj.is_absolute():
		candidates.append(path_obj)
	else:
		module_dir = Path(__file__).resolve().parent
		src_dir = module_dir.parent
		repo_root = src_dir.parent
		candidates.extend(
			[
				Path.cwd() / path_obj,
				module_dir / path_obj,
				src_dir / path_obj,
				repo_root / path_obj,
				module_dir / "templates" / path_obj.name,
			]
		)

	for candidate in candidates:
		if candidate.is_file():
			return candidate

	return None


def _apply_allowed_pattern(text: str, allowed_pattern: Optional[str]) -> str:
	"""Filter text by a field-provided allow-list regex pattern."""
	if not text:
		return ""
	if not allowed_pattern:
		return text
	filtered = "".join(re.findall(allowed_pattern, text))
	return re.sub(r"\s+", " ", filtered).strip()


def _trim_text_suffix_artifacts(text: str) -> str:
	"""Trim common OCR punctuation tails left at line ends."""
	if not text:
		return ""
	trimmed = re.sub(r"[|~`_^]+$", "", text)
	trimmed = re.sub(r"(?:\s+[\-_,;:])+$", "", trimmed)
	trimmed = re.sub(r"([.]){2,}$", ".", trimmed)
	return re.sub(r"\s+", " ", trimmed).strip()


def _crop_box_gray(image: np.ndarray, box: "ROIBox") -> Optional[np.ndarray]:
	"""Crop a normalized ROIBox from a grayscale image with bounds checks."""
	height, width = image.shape[:2]
	top = int(height * box.top)
	bottom = int(height * box.bottom)
	left = int(width * box.left)
	right = int(width * box.right)

	if top >= bottom or left >= right:
		return None

	roi_img = image[top:bottom, left:right]
	if roi_img.size == 0:
		return None

	return roi_img


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
		source_points = _elongate_bottom_edge(display_contour.reshape(4, 2))
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
	candidate_debug: list[tuple[float, str, Any]] = []

	for variant in variants:
		raw = pytesseract.image_to_string(variant, config=field.tesseract_config).strip()

		if field.value_parser is not None:
			value = field.value_parser(raw)
		else:
			value = raw if raw else ""

		if isinstance(value, str):
			value = _apply_allowed_pattern(value, field.allowed_pattern)
			if field.strip_trailing_garbage:
				value = _trim_text_suffix_artifacts(value)

		score = field.value_scorer(raw, value) if field.value_scorer is not None else len(raw)
		if DEBUG_OCR_CANDIDATES:
			candidate_debug.append((score, raw, value))
		if score > best_score:
			best_raw = raw
			best_value = value
			best_score = score

	if best_value is None:
		best_value = field.empty_value
	elif field.blank_score_threshold is not None and best_score < field.blank_score_threshold:
		best_value = field.empty_value

	if DEBUG_OCR_CANDIDATES:
		top = sorted(candidate_debug, key=lambda item: item[0], reverse=True)[:2]
		print(f"[OCR DEBUG] {field_name} top candidates: {top}")

	return best_raw, best_value


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
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	if display_contour is None:
		for field_name in CURRENT_LAYOUT.fields:
			field = CURRENT_LAYOUT.fields[field_name]
			if field.icon_template_path is not None:
				if field_name in ICON_STATE_TEMPLATES:
					on_path, off_path = ICON_STATE_TEMPLATES[field_name]
					detected, on_score, off_score = detect_icon_state(
						frame=frame,
						field=field,
						on_template_path=on_path,
						off_template_path=off_path,
						threshold=field.icon_match_threshold,
						warped=None,
						gray_frame=gray_frame,
					)
					if field_name == "wifi_icon":
						state_value = "wifi_on" if detected else "wifi_off"
					elif field_name == "calendar_icon":
						state_value = "schedule_running" if detected else "schedule_not_running"
					else:
						state_value = detected
					fields_result[field_name] = {"raw": f"on={on_score:.3f},off={off_score:.3f}", "value": state_value}
				else:
					detected, score = detect_icon(
						frame=frame,
						template_path=field.icon_template_path,
						field=field,
						threshold=field.icon_match_threshold,
						warped=None,
						gray_frame=gray_frame,
					)
					fields_result[field_name] = {"raw": f"{score:.3f}", "value": detected}
			else:
				raw, val = _read_field_from_variants(field_name, _fallback_field_variants(frame, field_name))
				fields_result[field_name] = {"raw": raw, "value": val}
		
		return OCRReadout(
			display_found=False,
			current_menu_key=current_menu_key,
			fields=fields_result,
		)

	warped = _four_point_transform(frame, _elongate_bottom_edge(display_contour.reshape(4, 2)))
	binary = _prepare_binary(warped)
	gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

	for field_name in CURRENT_LAYOUT.fields:
		field = CURRENT_LAYOUT.fields[field_name]
		if field.icon_template_path is not None:
			if field_name in ICON_STATE_TEMPLATES:
				on_path, off_path = ICON_STATE_TEMPLATES[field_name]
				detected, on_score, off_score = detect_icon_state(
					frame=frame,
					field=field,
					on_template_path=on_path,
					off_template_path=off_path,
					threshold=field.icon_match_threshold,
					warped=warped,
					gray_frame=gray_frame,
					gray_warped=gray_warped,
				)
				if field_name == "wifi_icon":
					state_value = "wifi_on" if detected else "wifi_off"
				elif field_name == "calendar_icon":
					state_value = "schedule_running" if detected else "schedule_not_running"
				else:
					state_value = detected
				fields_result[field_name] = {"raw": f"on={on_score:.3f},off={off_score:.3f}", "value": state_value}
			else:
				detected, score = detect_icon(
					frame=frame,
					template_path=field.icon_template_path,
					field=field,
					threshold=field.icon_match_threshold,
					warped=warped,
					gray_frame=gray_frame,
					gray_warped=gray_warped,
				)
				fields_result[field_name] = {"raw": f"{score:.3f}", "value": detected}
		else:
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


def get_warped_display(frame: np.ndarray) -> Optional[np.ndarray]:
	"""Extract a front-facing view of the display from a camera frame, or None if not found."""
	mask = _display_mask(frame)
	display_contour = _find_display_contour(mask)

	if display_contour is None:
		return None

	return _four_point_transform(frame, _elongate_bottom_edge(display_contour.reshape(4, 2)))


def detect_icon(
	frame: np.ndarray,
	template_path: str,
	field: OCRField,
	threshold: float = 0.80,
	warped: Optional[np.ndarray] = None,
	gray_frame: Optional[np.ndarray] = None,
	gray_warped: Optional[np.ndarray] = None,
) -> tuple[bool, float]:

	_require_cv2()

	if gray_frame is None:
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	if warped is None:
		warped = get_warped_display(frame)

	if warped is not None and gray_warped is None:
		gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

	template = cv2.imread(str(_resolve_template_path(template_path) or ""), cv2.IMREAD_GRAYSCALE)
	if template is None:
		return False, 0.0

	roi_img: Optional[np.ndarray]
	if gray_warped is not None:
		roi_img = _crop_box_gray(gray_warped, field.ideal)
	else:
		roi_img = _crop_box_gray(gray_frame, field.fallback)

	if roi_img is None:
		return False, 0.0

	t_h, t_w = template.shape[:2]
	if t_h <= 0 or t_w <= 0:
		return False, 0.0

	# Work at a normalized size so huge upscaled templates do not amplify camera noise.
	compare_width = 120
	compare_height = max(64, int(round(compare_width * (t_h / max(1, t_w)))))
	compare_size = (compare_width, compare_height)
	roi_resized = cv2.resize(roi_img, compare_size, interpolation=cv2.INTER_AREA)
	template_resized = cv2.resize(template, compare_size, interpolation=cv2.INTER_AREA)

	# Normalize contrast and denoise before matching.
	roi_norm = cv2.GaussianBlur(cv2.equalizeHist(roi_resized), (3, 3), 0)
	template_norm = cv2.GaussianBlur(cv2.equalizeHist(template_resized), (3, 3), 0)

	_, template_bin = cv2.threshold(template_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	_, roi_bin_otsu = cv2.threshold(roi_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	roi_bin_adaptive = cv2.adaptiveThreshold(
		roi_norm,
		255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY,
		11,
		2,
	)
	template_edges = cv2.Canny(template_norm, 40, 120)

	def _shift_image(image: np.ndarray, dx: int, dy: int, interpolation: int) -> np.ndarray:
		transform = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
		return cv2.warpAffine(
			image,
			transform,
			compare_size,
			flags=interpolation,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=0,
		)

	def _score_variant(roi_bin_variant: np.ndarray) -> float:
		roi_edges_variant = cv2.Canny(roi_bin_variant, 40, 120)
		best_variant_score = -1.0
		for dy in range(-2, 3):
			for dx in range(-2, 3):
				shifted_gray = _shift_image(roi_norm, dx, dy, cv2.INTER_LINEAR)
				shifted_bin = _shift_image(roi_bin_variant, dx, dy, cv2.INTER_NEAREST)
				shifted_edges = _shift_image(roi_edges_variant, dx, dy, cv2.INTER_NEAREST)

				gray_ncc = float(cv2.matchTemplate(shifted_gray, template_norm, cv2.TM_CCOEFF_NORMED)[0, 0])
				gray_score = max(0.0, min(1.0, (gray_ncc + 1.0) / 2.0))

				shifted_edges_nz = cv2.countNonZero(shifted_edges)
				template_edges_nz = cv2.countNonZero(template_edges)
				if shifted_edges_nz == 0 and template_edges_nz == 0:
					edge_score = 1.0
				elif shifted_edges_nz == 0 or template_edges_nz == 0:
					edge_score = 0.0
				else:
					edge_ncc = float(cv2.matchTemplate(shifted_edges, template_edges, cv2.TM_CCOEFF_NORMED)[0, 0])
					edge_score = max(0.0, min(1.0, (edge_ncc + 1.0) / 2.0))

				intersection = cv2.countNonZero(cv2.bitwise_and(shifted_bin, template_bin))
				union = cv2.countNonZero(cv2.bitwise_or(shifted_bin, template_bin))
				binary_iou = float(intersection / union) if union > 0 else 0.0

				xor_pixels = cv2.countNonZero(cv2.bitwise_xor(shifted_bin, template_bin))
				xor_similarity = 1.0 - (float(xor_pixels) / float(compare_width * compare_height))

				score = (
					0.40 * gray_score
					+ 0.25 * binary_iou
					+ 0.25 * edge_score
					+ 0.10 * xor_similarity
				)
				if score > best_variant_score:
					best_variant_score = score

		return best_variant_score

	score = max(_score_variant(roi_bin_otsu), _score_variant(roi_bin_adaptive))

	return score >= threshold, score


def detect_icon_state(
	frame: np.ndarray,
	field: OCRField,
	on_template_path: str,
	off_template_path: str,
	threshold: float = 0.80,
	warped: Optional[np.ndarray] = None,
	gray_frame: Optional[np.ndarray] = None,
	gray_warped: Optional[np.ndarray] = None,
) -> tuple[bool, float, float]:
	"""Classify icon state using baseline template scores plus pairwise discriminative scoring."""
	on_detected, on_score = detect_icon(
		frame=frame,
		template_path=on_template_path,
		field=field,
		threshold=threshold,
		warped=warped,
		gray_frame=gray_frame,
		gray_warped=gray_warped,
	)
	off_detected, off_score = detect_icon(
		frame=frame,
		template_path=off_template_path,
		field=field,
		threshold=threshold,
		warped=warped,
		gray_frame=gray_frame,
		gray_warped=gray_warped,
	)

	# Pairwise discriminative comparison focuses on template-difference pixels
	# (calendar checkmark region, wifi slash region), not shared icon outlines.
	try:
		if gray_frame is None:
			gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if warped is None:
			warped = get_warped_display(frame)
		if warped is not None and gray_warped is None:
			gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

		roi_img: Optional[np.ndarray]
		if gray_warped is not None:
			roi_img = _crop_box_gray(gray_warped, field.ideal)
		else:
			roi_img = _crop_box_gray(gray_frame, field.fallback)

		on_template = cv2.imread(str(_resolve_template_path(on_template_path) or ""), cv2.IMREAD_GRAYSCALE)
		off_template = cv2.imread(str(_resolve_template_path(off_template_path) or ""), cv2.IMREAD_GRAYSCALE)

		if roi_img is not None and on_template is not None and off_template is not None:
			t_h = max(on_template.shape[0], off_template.shape[0])
			t_w = max(on_template.shape[1], off_template.shape[1])
			compare_width = 120
			compare_height = max(64, int(round(compare_width * (t_h / max(1, t_w)))))
			compare_size = (compare_width, compare_height)

			roi_resized = cv2.resize(roi_img, compare_size, interpolation=cv2.INTER_AREA)
			on_resized = cv2.resize(on_template, compare_size, interpolation=cv2.INTER_AREA)
			off_resized = cv2.resize(off_template, compare_size, interpolation=cv2.INTER_AREA)

			roi_norm = cv2.GaussianBlur(cv2.equalizeHist(roi_resized), (3, 3), 0)
			on_norm = cv2.GaussianBlur(cv2.equalizeHist(on_resized), (3, 3), 0)
			off_norm = cv2.GaussianBlur(cv2.equalizeHist(off_resized), (3, 3), 0)

			template_diff = cv2.absdiff(on_norm, off_norm)
			_, diff_mask = cv2.threshold(template_diff, 24, 255, cv2.THRESH_BINARY)
			diff_mask = cv2.dilate(diff_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

			min_diff_pixels = int(0.015 * compare_width * compare_height)
			if cv2.countNonZero(diff_mask) < min_diff_pixels:
				diff_mask = np.full((compare_height, compare_width), 255, dtype=np.uint8)

			mask_idx = diff_mask > 0

			def _shift_gray(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
				transform = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
				return cv2.warpAffine(
					image,
					transform,
					compare_size,
					flags=cv2.INTER_LINEAR,
					borderMode=cv2.BORDER_CONSTANT,
					borderValue=0,
				)

			best_pair_on = 0.0
			best_pair_off = 0.0
			best_objective = -1.0

			for dy in range(-2, 3):
				for dx in range(-2, 3):
					shifted = _shift_gray(roi_norm, dx, dy)

					on_error = float(np.mean(np.abs(shifted[mask_idx].astype(np.float32) - on_norm[mask_idx].astype(np.float32))))
					off_error = float(np.mean(np.abs(shifted[mask_idx].astype(np.float32) - off_norm[mask_idx].astype(np.float32))))

					on_sim = max(0.0, min(1.0, 1.0 - (on_error / 255.0)))
					off_sim = max(0.0, min(1.0, 1.0 - (off_error / 255.0)))

					objective = max(on_sim, off_sim) + 0.35 * abs(on_sim - off_sim)
					if objective > best_objective:
						best_objective = objective
						best_pair_on = on_sim
						best_pair_off = off_sim

			on_score = (0.55 * on_score) + (0.45 * best_pair_on)
			off_score = (0.55 * off_score) + (0.45 * best_pair_off)
			on_detected = on_score >= threshold
			off_detected = off_score >= threshold
	except Exception:
		# If pairwise enhancement fails, keep baseline detect_icon scores.
		pass

	if on_detected and not off_detected:
		return True, on_score, off_score
	if off_detected and not on_detected:
		return False, on_score, off_score

	return on_score >= off_score, on_score, off_score


def detect_status_icons(
 	frame: np.ndarray
) -> dict:
	results: dict[str, Any] = {}
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	warped = get_warped_display(frame)
	gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped is not None else None

	wifi_on, wifi_on_score, wifi_off_score = detect_icon_state(
		frame=frame,
		field=CURRENT_LAYOUT.fields["wifi_icon"],
		on_template_path="vision/templates/wifi_on.png",
		off_template_path="vision/templates/wifi_off.png",
		threshold=CURRENT_LAYOUT.fields["wifi_icon"].icon_match_threshold,
		warped=warped,
		gray_frame=gray_frame,
		gray_warped=gray_warped,
	)

	calendar_running, calendar_run_score, calendar_not_run_score = detect_icon_state(
		frame=frame,
		field=CURRENT_LAYOUT.fields["calendar_icon"],
		on_template_path="vision/templates/schedule_running.png",
		off_template_path="vision/templates/schedule_not_running.png",
		threshold=CURRENT_LAYOUT.fields["calendar_icon"].icon_match_threshold,
		warped=warped,
		gray_frame=gray_frame,
		gray_warped=gray_warped,
	)

	results["wifi_on"] = wifi_on
	results["wifi_off"] = not wifi_on
	results["schedule_running"] = calendar_running
	results["schedule_not_running"] = not calendar_running
	results["scores"] = {
		"wifi": {"on": wifi_on_score, "off": wifi_off_score},
		"schedule": {"running": calendar_run_score, "not_running": calendar_not_run_score},
	}

	return results
