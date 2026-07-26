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
from .display_layouts import OCRField, ROIBox, STATUS_BAR
import time

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
	"""Build a mask for the emissive display window, regardless of its backlight color.

	The display's background color changes with operating mode (observed as
	both orange and blue), so detection keys off saturation and brightness --
	properties any vividly backlit screen shares -- instead of a single
	hardcoded hue range that would only match one color.
	"""
	_require_cv2()
	if frame is None or frame.size == 0:
		raise ValueError("frame must be a non-empty image")

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	saturation = hsv[:, :, 1]
	value = hsv[:, :, 2]

	# Otsu finds the saturation/brightness split point per-frame instead of a
	# fixed guess, so the mask adapts to actual lighting/backlight conditions.
	_, saturated_mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	_, bright_mask = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	display_mask = cv2.bitwise_and(saturated_mask, bright_mask)

	# Fuse any inner text gaps horizontally/vertically without spilling into bezels
	mask = cv2.morphologyEx(display_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)))
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15)))

	# Clean up any lingering tiny noise specs
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
	
	return mask


def _find_display_contour(frame: np.ndarray, mask: np.ndarray, min_area: int = 3000) -> Optional[np.ndarray]:
	"""Find the raw orange display rectangle without geometric status bar projection."""
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
	"""
	Warp the display area once and optionally extend the contour for ContextNodes
	that include the status bar fields.
	"""
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


def _prepare_ocr_binary(warped: np.ndarray) -> np.ndarray:
	"""Convert the warped display to grayscale only.

	Heavy denoising is deliberately deferred to `_extract_warped_variants`,
	where it runs on the small per-field crop *before* that crop gets
	upscaled. Doing it here on the full warped frame would smooth the
	whole display uniformly and then let upscaling re-magnify whatever
	speckle survived; doing it per-field, pre-upscale, removes noise while
	it is still small instead of after it has been blown up to text size.
	"""
	_require_cv2()
	return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)


def _ensure_black_text_white_bg(binary_roi: np.ndarray) -> np.ndarray:
	"""Ensure binary image is black text (0) on white background (255)."""
	if np.mean(binary_roi) < 127:
		return cv2.bitwise_not(binary_roi)
	return binary_roi


def _denoise_roi_grayscale(roi: np.ndarray) -> np.ndarray:
	"""Scrub sensor speckle/static at native resolution, before any upscaling.

	The emissive display shows heavy per-pixel speckle (visible as static
	across the orange background). A median blur is the right first tool
	for that: it demolishes salt-and-pepper-style single-pixel noise
	without smearing edges the way a Gaussian blur would. A light
	bilateral pass then knocks down residual LCD dither while keeping
	character strokes crisp.
	"""
	despeckled = cv2.medianBlur(roi, 3)

	return cv2.bilateralFilter(despeckled, d=5, sigmaColor=60, sigmaSpace=60)


def _filter_connected_components(bin_img: np.ndarray, min_height_ratio: float = 0.35, min_area_px: int = 6) -> np.ndarray:
	"""Keep only connected components tall enough to plausibly be text glyphs.

	Character strokes span a substantial fraction of a tightly-cropped text
	row's height, whether the field holds large digits (temperature) or
	medium-sized prose (mode/dashboard lines). Sensor speckle, in contrast,
	survives thresholding as small blobs that stay short relative to the
	crop even after upscaling. A height-ratio cutoff therefore generalizes
	across field sizes automatically, instead of relying on one fixed
	pixel-area guess tuned for a single font size.
	"""
	# Work on inverted image where text/noise is foreground (255)
	fg_mask = cv2.bitwise_not(bin_img)
	num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)

	roi_height = bin_img.shape[0]
	min_height_px = max(3, int(roi_height * min_height_ratio))

	cleaned_fg = np.zeros_like(fg_mask)
	for i in range(1, num_labels):  # Skip background label 0
		area = stats[i, cv2.CC_STAT_AREA]
		height = stats[i, cv2.CC_STAT_HEIGHT]
		if area < min_area_px or height < min_height_px:
			continue
		cleaned_fg[labels == i] = 255

	# Invert back to black text (0) on white background (255)
	return cv2.bitwise_not(cleaned_fg)


def _is_roi_blank(bin_img: np.ndarray, min_text_ratio: float = 0.008) -> bool:
	"""Returns True if foreground text pixels account for less than min_text_ratio of the total crop area."""
	black_pixels = np.count_nonzero(bin_img == 0)
	total_pixels = bin_img.size
	return (black_pixels / float(total_pixels)) < min_text_ratio


def _extract_warped_variants(prepared: np.ndarray, field: Any = None) -> list[np.ndarray]:
	"""Extract denoised-then-upscaled, speckle-filtered variants for OCR."""
	_require_cv2()

	if field is not None and hasattr(field, "ideal"):
		roi = field.ideal.crop(prepared)
	else:
		roi = prepared

	if roi is None or roi.size == 0:
		return []

	denoised_roi = _denoise_roi_grayscale(roi)

	h, w = denoised_roi.shape[:2]
	target_height = 180
	if h < target_height:
		scale = target_height / float(h)
		new_w = max(1, int(w * scale))
		denoised_roi = cv2.resize(denoised_roi, (new_w, target_height), interpolation=cv2.INTER_LINEAR)

	variants: list[np.ndarray] = []

	def _finalize(bin_img: np.ndarray) -> Optional[np.ndarray]:
		bin_img = _ensure_black_text_white_bg(bin_img)

		# Erase blobs too short to be real glyph strokes (sensor speckle).
		bin_img = _filter_connected_components(bin_img)

		# Skip variants that hold no real text after cleaning.
		if _is_roi_blank(bin_img):
			return None

		# Add a white quiet-zone margin around characters.
		return cv2.copyMakeBorder(bin_img, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)

	# --- Variant 1: Denoised Otsu ---
	_, v1 = cv2.threshold(denoised_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	finalized = _finalize(v1)
	if finalized is not None:
		variants.append(finalized)

	# --- Variant 2: CLAHE + Otsu ---
	clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
	enhanced = clahe.apply(denoised_roi)
	_, v2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	finalized = _finalize(v2)
	if finalized is not None:
		variants.append(finalized)

	# --- Variant 3: Adaptive Gaussian Thresholding ---
	curr_h, curr_w = denoised_roi.shape[:2]
	max_block = max(3, (min(curr_h, curr_w) // 2) * 2 - 1)
	block_size = min(41, max_block)
	if block_size % 2 == 0:
		block_size -= 1

	v3 = cv2.adaptiveThreshold(
		denoised_roi,
		255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY,
		blockSize=max(3, block_size),
		C=4,
	)
	finalized = _finalize(v3)
	if finalized is not None:
		variants.append(finalized)

	return variants


# Cache dictionary for loaded template images
_ICON_TEMPLATES: dict[str, dict[str, np.ndarray]] = {}

# Defined state outputs for each template file
ICON_STATE_MAPPINGS = {
    "wifi_icon": {
        "wifi_connected": "CONNECTED",
        "wifi_not_connected": "NOT_CONNECTED",
    },
    "schedule_icon": {
        "schedule_running": "RUNNING",
        "schedule_not_running": "NOT_RUNNING",
    }
}

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
        }
    }

    _require_cv2()
    for field_key, templates in template_files.items():
        _ICON_TEMPLATES[field_key] = {}
        for state_key, file_path in templates.items():
            if not file_path.exists():
                print(f"[WARNING] Missing template file: {file_path}")
                continue
            
            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # OPTIMIZATION: Process the template bounding box and scaling once upon load
                _, template_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                t_contours, _ = cv2.findContours(template_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if t_contours:
                    tx, ty, tw, th = cv2.boundingRect(np.concatenate(t_contours))
                    t_crop = template_thresh[ty:ty+th, tx:tx+tw]
                    
                    t_max_dim = max(tw, th)
                    t_pad_top = (t_max_dim - th) // 2
                    t_pad_bottom = t_max_dim - th - t_pad_top
                    t_pad_left = (t_max_dim - tw) // 2
                    t_pad_right = t_max_dim - tw - t_pad_left
                    t_squared_crop = cv2.copyMakeBorder(t_crop, t_pad_top, t_pad_bottom, t_pad_left, t_pad_right, cv2.BORDER_CONSTANT, value=0)
                    
                    template_norm = cv2.resize(t_squared_crop, (64, 64), interpolation=cv2.INTER_AREA)
                    _, template_norm = cv2.threshold(template_norm, 127, 255, cv2.THRESH_BINARY)
                    _ICON_TEMPLATES[field_key][state_key] = template_norm
                else:
                    # Fallback if the template is somehow entirely blank
                    _ICON_TEMPLATES[field_key][state_key] = img

    return _ICON_TEMPLATES


def _classify_icon_field(roi_gray: np.ndarray, field_name: str) -> tuple[str, str]:
    """
    Classify an icon by cropping out all padding, forcing the crop into a perfect square 
    to preserve aspect ratio, normalizing to 64x64, and comparing structural overlap (XOR).
    Lower score = better match.
    """
    _require_cv2()
    templates_dict = _load_icon_templates().get(field_name, {})
    
    if not templates_dict or roi_gray is None or roi_gray.size == 0:
        print(f"[{field_name.upper()}] ERROR: Empty ROI or missing templates.")
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
    
    crop = roi_thresh[y:y+h, x:x+w]
    max_dim = max(w, h)
    pad_top = (max_dim - h) // 2
    pad_bottom = max_dim - h - pad_top
    pad_left = (max_dim - w) // 2
    pad_right = max_dim - w - pad_left
    squared_crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        
    roi_norm = cv2.resize(squared_crop, (64, 64), interpolation=cv2.INTER_AREA)
    _, roi_norm = cv2.threshold(roi_norm, 127, 255, cv2.THRESH_BINARY)

    scores: dict[str, float] = {}

    for state_key, template_norm in templates_dict.items():
        if template_norm.shape != (64, 64):
            scores[state_key] = float('inf')
            continue
            
        diff = cv2.bitwise_xor(roi_norm, template_norm)
        score = np.sum(diff == 255) / (64.0 * 64.0)
        scores[state_key] = float(score)

    best_state_key = min(scores, key=scores.get)
    parsed_state = ICON_STATE_MAPPINGS.get(field_name, {}).get(best_state_key, "UNKNOWN")
    return best_state_key, parsed_state


def _clean_text(raw_text: str) -> str:
	"""Collapse OCR whitespace into a stable single-space form."""
	return re.sub(r"\s+", " ", raw_text).strip()


def _parse_mode(raw_text: str) -> str:
	cleaned = _clean_text(raw_text).upper().strip()
	parts = cleaned.split(" ", 1)
	if len(parts) > 1:
		return parts[1]
	return ""


def _parse_temperature(raw_text: str) -> Optional[int]:
	# TEMPERATURE_FIELD's tessedit_char_whitelist is digits-only, and that
	# whitelist is applied uniformly before this function ever sees the
	# text, so raw_text is already guaranteed to contain nothing but digits.
	if not raw_text or not raw_text.isdigit():
		return None

	value = int(raw_text)
	if value <= 0:
		return None

	return value


def _field_empty_value(field_name: str) -> Any:
	if field_name == "mode":
		return "UNKNOWN"
	if field_name == "temperature":
		return None
	return ""


def _parse_time(raw_text: str) -> str:
	# The field's tessedit_char_whitelist has already restricted raw_text to
	# legal characters. Normalize case and self-heal a dropped/misread
	# trailing 'M' (a real, recurring OCR failure mode for this font).
	text = raw_text.upper().strip()

	if "A" in text:
		text = text.split("A")[0] + "AM"
	elif "P" in text:
		text = text.split("P")[0] + "PM"

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
	# Standard bounds for a water heater
	if 90 <= value <= 160:
		score += 2.0
	if 60 <= value <= 199:
		score += 1.5
	if len(str(value)) == 3:
		score += 1.0
	return score


def _is_plausible_word(word: str) -> bool:
	"""Check if a token has plausible word structure (vowel ratio, pronounceability, digits)."""
	clean_num = re.sub(r"[^0-9]", "", word)
	clean_alpha = re.sub(r"[^A-Za-z]", "", word)

	# Pure numeric or temperature/percentage tokens (e.g. "100", "80%") are valid
	if clean_num and not clean_alpha:
		return True

	if not clean_alpha:
		return False

	length = len(clean_alpha)
	vowels = sum(1 for c in clean_alpha.lower() if c in "aeiouy")

	# Single-letter words must be standard English words ('A', 'I')
	if length == 1:
		return clean_alpha.upper() in {"A", "I"}

	# 2-letter words must contain a vowel or be common valid words/acronyms
	if length == 2:
		return vowels >= 1 or clean_alpha.upper() in {
			"ON", "NO", "GO", "TO", "IN", "IT", "IS", "AT", "BY", "HE",
			"ME", "WE", "UP", "OR", "IF", "DO", "SO", "AM", "PM", "ID", "OK"
		}

	# 3+ letter words must contain at least 1 vowel and not be >85% vowels
	vowel_ratio = vowels / float(length)
	if vowels == 0:
		return False  # Blocks vowelless gibberish ("CFR", "TRG")
	if vowel_ratio > 0.85 and length >= 3:
		return False  # Blocks vowel-only hallucinations ("aoe")

	# Reject 3+ consecutive repeated characters (e.g., "zzz")
	if re.search(r"(.)\1\1", clean_alpha.lower()):
		return False

	return True


import difflib

# Common English vocabulary pool (or pass standard dictionary words)
COMMON_WORDS = {
	"call", "for", "heat", "setpoint", "satisfied", "electric", "heat", "pump",
	"hybrid", "vacation", "disabled", "enabled", "running", "status", "water",
	"heater", "system", "normal", "error", "warning", "mode", "standby"
}

def _autocorrect_word(word: str) -> str:
	"""Auto-correct 1-character OCR substitutions against general English words."""
	clean_alpha = re.sub(r"[^A-Za-z]", "", word)
	if not clean_alpha or len(clean_alpha) < 3:
		return word

	# Cutoff 0.70 allows 1-char edits on 4-letter words (0.75 similarity ratio)
	matches = difflib.get_close_matches(clean_alpha.lower(), COMMON_WORDS, n=1, cutoff=0.70)
	if matches:
		corrected = matches[0]
		# Preserve original capitalization structure
		if clean_alpha.istitle():
			corrected = corrected.title()
		elif clean_alpha.isupper():
			corrected = corrected.upper()

		# Replace only the alpha part, keeping attached punctuation
		return word.replace(clean_alpha, corrected)

	return word


def _parse_info_line(raw_text: str) -> str:
	"""Parse info line and self-heal 1-character OCR character bleeds."""
	text = _clean_text(raw_text).strip(" .:-_")
	if not text:
		return ""

	words = text.split()
	if not words:
		return ""

	# Auto-correct OCR character bleeds word-by-word
	corrected_words = [_autocorrect_word(w) for w in words]
	
	plausible_words = [w for w in corrected_words if _is_plausible_word(w)]

	# At least 50% of tokens must be valid words
	if len(plausible_words) / float(len(words)) < 0.50:
		return ""

	return " ".join(corrected_words)


def _score_temperature(raw_text: str, value: Any) -> float:
	if not isinstance(value, int):
		return -1.0

	score = 0.0
	if 90 <= value <= 160:
		score += 2.0
	if 60 <= value <= 199:
		score += 1.5
	if len(str(value)) == 3:
		score += 1.0
	return score


def _score_info_line(raw_text: str, parsed_value: Any) -> float:
	"""Score info line quality, heavily penalizing non-word noise clusters."""
	if not isinstance(parsed_value, str) or not parsed_value:
		return -1.0

	words = parsed_value.split()
	if not words:
		return -1.0

	plausible_words = [w for w in words if _is_plausible_word(w)]
	if not plausible_words:
		return -1.0

	plausible_ratio = len(plausible_words) / float(len(words))
	if plausible_ratio < 0.60:
		return -1.0

	alnum_count = sum(1 for char in parsed_value if char.isalnum())
	if alnum_count < 3:
		return -1.0

	score = 0.0
	score += len(plausible_words) * 1.5
	score += alnum_count * 0.15

	punct_count = sum(1 for char in parsed_value if not char.isalnum() and char != " ")
	score -= punct_count * 0.5

	if re.search(r"[^A-Za-z0-9\s]{2,}", raw_text):
		score -= 1.5

	return score


def _score_field_candidate(field_name: str, raw_text: str, value: Any) -> float:
	"""Master scoring router for candidate field OCR values."""
	if field_name.startswith("dashboard_info_line_"):
		return _score_info_line(raw_text, value)
	if field_name == "temperature":
		return _score_temperature(raw_text, value)
	if isinstance(value, str):
		return float(len(value))
	if value is None:
		return -1.0
	return 0.0


def _get_field_scale_bounds(field_name: str) -> tuple[float, float, float]:
	"""
	Return expected (min_relative_height, max_relative_height, max_aspect_ratio)
	relative to the cropped ROI box height.
	"""
	if field_name == "temperature":
		# Prominent digits fill most of the ROI vertically
		return (0.30, 0.98, 8.0)

	if field_name == "mode":
		# Mode header text (e.g. "MODE: ELECTRIC")
		return (0.20, 0.90, 10.0)

	if field_name.startswith("dashboard_info_line_"):
		# Multi-line prose text bounded by horizontal rules
		return (0.32, 0.82, 10.0)

	if field_name in ("time_field", "date_field"):
		# Status bar numbers and text
		return (0.20, 0.90, 10.0)

	# General sensible default for any future text fields
	return (0.15, 0.95, 12.0)


def _parse_tesseract_data(
	tesseract_output: dict[str, Any],
	variant_shape: Optional[tuple[int, int]] = None,
	field_name: str = "",
) -> str:
	"""Extract tokens using relative scale filtering without aggressive confidence drops."""
	texts = tesseract_output.get("text", []) or []
	heights = tesseract_output.get("height", []) or []
	widths = tesseract_output.get("width", []) or []

	cleaned_tokens: list[str] = []
	var_h = variant_shape[0] if variant_shape else 0

	min_rel_h, max_rel_h, max_aspect = _get_field_scale_bounds(field_name)

	for i, token in enumerate(texts):
		if token is None:
			continue
		stripped = str(token).strip()
		if not stripped:
			continue

		# --- UNIVERSAL RELATIVE SCALE FILTER ---
		# Only drop tokens that are physically impossible (speckle noise or UI lines)
		if var_h > 0 and i < len(heights) and i < len(widths):
			try:
				box_h = float(heights[i])
				box_w = float(widths[i])
				rel_height = box_h / float(var_h)
				aspect_ratio = box_w / max(1.0, box_h)

				# Filter out tiny noise (< min_rel_h) or crop border boxes (> max_rel_h)
				if rel_height < min_rel_h or rel_height > max_rel_h:
					continue

				# Filter out horizontal line rules
				if aspect_ratio > max_aspect and rel_height < 0.25:
					continue
			except (ValueError, TypeError):
				pass

		cleaned_tokens.append(stripped)

	return _clean_text(" ".join(cleaned_tokens))


def _ocr_field(field: OCRField, variants: list[np.ndarray]) -> tuple[str, Any]:
	"""OCR or classify a field from image variants and keep the best candidate by score."""
	field_name = field.name

	# --- ICON CLASSIFICATION BYPASS ---
	if field_name in ("wifi_icon", "schedule_icon"):
		if not variants:
			return "UNKNOWN", "UNKNOWN"
		inverted_variant = cv2.bitwise_not(variants[0])
		icon_key, icon_value = _classify_icon_field(inverted_variant, field_name)
		return icon_key, icon_value

	# --- STANDARD TESSERACT OCR FOR TEXT/DIGITS ---
	_require_pytesseract()
	best_raw = ""
	best_value: Any = _field_empty_value(field_name)
	best_score = 0.0  # Threshold <= 0.0 leaves empty fields clean

	whitelist_set = None
	if "tessedit_char_whitelist=" in field.tesseract_config:
		whitelist_set = set(field.tesseract_config.split("tessedit_char_whitelist=")[1])

	config = field.tesseract_config
	if "--oem" not in config:
		config = f"--oem 1 {config}"

	for variant in variants:
		tesseract_output = pytesseract.image_to_data(
			variant,
			config=config,
			output_type=pytesseract.Output.DICT,
		)
		raw = _parse_tesseract_data(
			tesseract_output,
			variant_shape=variant.shape[:2],
			field_name=field_name,
		)

		if whitelist_set:
			raw = "".join(c for c in raw if c in whitelist_set)

		value = _parse_field_value(field_name, raw)
		score = _score_field_candidate(field_name, raw, value)

		if score > best_score:
			best_raw = raw
			best_value = value
			best_score = score

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

	if display_contour is None:
		# Graceful exit without running heavy OCR on full unwarped frames
		return OCRReadout(
			display_found=False,
			current_menu_key=current_menu_key,
			fields={
				field.name: {"raw": "", "value": _field_empty_value(field.name)}
				for field in menu_fields
			},
		)

	# Extend the warp only when this ContextNode includes the status bar fields.
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
	"""Hook for camera warmup work to fully flush the hardware pipeline."""
	camera = _get_camera()


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
		"LensPosition": 9.1, # 1 / 0.11m = 9.1
	})

	# --- HARDWARE STABILIZATION SETTLE ---
	# Ensures the voice coil physical motor has moved to position 9 and 
	# the image sensor has applied initial auto-exposure/white-balance
	# convergence before ANY frame is handed out to the pipeline.
	time.sleep(0.6)
	for _ in range(6):
		try:
			camera.capture_array()
		except Exception:
			pass

	# --- LOCK AUTO-EXPOSURE AND AUTO-WHITE-BALANCE ---
	# The display is a saturated, colored light source, which can fool
	# continuous auto-exposure/auto-white-balance into hunting or drifting
	# between frames (visible as color shifts and grainy noise from the gain
	# compensating). Rather than guessing fixed exposure/gain numbers, let
	# AE/AWB converge during the warmup above, read back whatever values they
	# landed on, and freeze those so every subsequent capture is consistent.
	try:
		converged = camera.capture_metadata()
		lock_controls: dict[str, Any] = {"AeEnable": False, "AwbEnable": False}
		for key in ("ExposureTime", "AnalogueGain", "ColourGains"):
			if key in converged:
				lock_controls[key] = converged[key]
		camera.set_controls(lock_controls)
	except Exception:
		pass  # Fall back to continuous AE/AWB if metadata isn't available.

	_CAMERA = camera
	return _CAMERA


def shutdown() -> None:
	"""Stop the camera cleanly."""
	global _CAMERA
	if _CAMERA is None:
		return

	_CAMERA.stop()
	_CAMERA = None


def get_warped_display(
	frame: np.ndarray,
	current_menu_key: Optional[str] = None,
	menu_fields: Optional[tuple[OCRField, ...]] = None,
) -> Optional[np.ndarray]:
	"""Extract a front-facing view of the display from a camera frame, or None if not found."""
	# Keep current_menu_key for backward compatibility with existing callers.
	_ = current_menu_key
	mask = _build_display_mask(frame)
	display_contour = _find_display_contour(frame, mask)

	if display_contour is None:
		return None

	# Single-pass warp/crop with status-bar-aware extension when appropriate.
	_, warped = _process_display_contour_and_warp(frame, display_contour, menu_fields=menu_fields)
	return warped
