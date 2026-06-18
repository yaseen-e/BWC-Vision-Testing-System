"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/vision/display_layouts.py - Display Layout Schema
Defines the ordered menu tree and OCR regions of interest (ROIs) 
for the LCD UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Mapping

try:
	import numpy as np
except Exception:  # pragma: no cover - environment dependent
	np = None


@dataclass(frozen=True)
class ROIBox:
	top: float
	bottom: float
	left: float
	right: float

	def crop(self, image: np.ndarray) -> np.ndarray:
		"""Crop a relative rectangle from an image."""
		if np is None:
			raise RuntimeError("numpy is required for ROI cropping")
		height, width = image.shape[:2]
		return image[
			int(height * self.top):int(height * self.bottom),
			int(width * self.left):int(width * self.right),
		]


@dataclass(frozen=True)
class OCRField:
	name: str
	ideal: ROIBox
	fallback: ROIBox
	tesseract_config: str
	value_parser: Callable[[str], Any] | None = None
	value_scorer: Callable[[str, Any], float] | None = None
	empty_value: Any = ""
	min_alpha_chars: int = 0
	blank_score_threshold: float | None = None
	strip_trailing_garbage: bool = False
	allowed_pattern: str | None = None


@dataclass(frozen=True)
class ContextNode:
	key: str
	label: str
	route_here: list[set[str]] = field(default_factory=list)
	return_route: set[str] = field(default_factory=set)
	children: tuple["ContextNode", ...] = ()


@dataclass(frozen=True)
class DisplayLayout:
	name: str
	menu_tree: ContextNode
	display_aspect_ratio: float
	temperature_range_f: tuple[int, int]
	fields: Mapping[str, OCRField] = field(default_factory=dict)


ROOT_MENU = ContextNode(
	key="homescreen",
	label="Home Screen",
	children=(
		ContextNode(key="active_faults_screen", label="Active Faults Screen", route_here={"RIGHT"}, return_route={"MENU", "SELECT"}, children=(
			ContextNode(key="active_faults_list", label="Active Faults List", route_here={"SELECT"}, return_route={"BACK"}, children=()),
		)),
        ContextNode(key="system_status_top", label="System Status 1/2", route_here={"MENU", "DOWN", "SELECT"}, return_route={"BACK", "BACK"}, children=(
			ContextNode(key="system_status_bottom", label="System Status 2/2", route_here={"DOWN"}, return_route={"UP"}),
		)),
        ContextNode(key="settings", label="Settings", route_here=[{"MENU", "RIGHT", "DOWN", "SELECT"}, {"MENU", "DOWN", "RIGHT", "SELECT"}], return_route={"MENU"}, children=()),
		ContextNode(key="schedules", label="Schedules", route_here={"MENU", "RIGHT", "RIGHT", "SELECT"}, return_route={"MENU", "SELECT"}),
	),
)


def _format_mode_text(raw_text: str) -> str:
	"""Normalize OCR output into the report-friendly mode format."""
	cleaned = re.sub(r"\s+", " ", raw_text).strip().upper()
	if not cleaned or cleaned == "UNKNOWN":
		return "UNKNOWN"

	cleaned = re.sub(r"^MODE\s*:\s*", "", cleaned)
	cleaned = re.sub(r"^MODE\s+", "", cleaned).strip()
	if not cleaned:
		return "UNKNOWN"

	return cleaned


def _parse_temperature_text(raw_temp: str) -> int | None:
	"""Extract an integer temperature from OCR text when available."""
	if not raw_temp:
		return None

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


def _score_temperature_candidate(raw_temp: str, temperature: Any) -> float:
	"""Score candidate temperatures by plausibility and OCR shape."""
	if not isinstance(temperature, int):
		return -1.0

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


def _normalize_ocr_unicode(raw_text: str) -> str:
	"""Normalize common OCR mojibake and non-printing artifacts."""
	text = raw_text.strip()
	replacements = {
		"â€˜": "'",
		"â€™": "'",
		"â€œ": '"',
		"â€�": '"',
		"â€“": "-",
		"â€”": "-",
		"Â°": "",
		"Â": "",
		"€": "",
	}
	for old, new in replacements.items():
		text = text.replace(old, new)

	text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
	text = re.sub(r"\s+", " ", text).strip()
	return text


def _strip_non_sentence_noise(raw_text: str, allowed_pattern: str | None = None) -> str:
	"""Keep sentence-like characters while dropping OCR junk symbols."""
	if not raw_text:
		return ""

	if allowed_pattern is not None:
		filtered = "".join(re.findall(allowed_pattern, raw_text))
	else:
		filtered = re.sub(r"[^A-Za-z0-9 .,'\-]", " ", raw_text)

	filtered = re.sub(r"\s+", " ", filtered).strip()
	return filtered


def _trim_trailing_garbage(raw_text: str) -> str:
	"""Trim suffix artifacts like repeated punctuation and OCR edge marks."""
	text = re.sub(r"\s+", " ", raw_text).strip()
	text = re.sub(r"[|~`_^]+$", "", text)
	text = re.sub(r"(?:\s+[\-_,;:])+$", "", text)
	text = re.sub(r"([.]){2,}$", ".", text)
	text = re.sub(r"\s+", " ", text).strip()
	return text


def _is_effectively_blank_text(raw_text: str, min_alpha_chars: int = 3) -> bool:
	"""Return True when OCR output has too little language signal to trust."""
	if not raw_text:
		return True

	text = _normalize_ocr_unicode(raw_text)
	if not text:
		return True

	alpha_count = sum(1 for char in text if char.isalpha())
	if alpha_count < min_alpha_chars:
		return True

	allowed_count = sum(1 for char in text if char.isalnum() or char in " .,'-")
	junk_ratio = 1.0 - (allowed_count / max(1, len(text)))
	if junk_ratio > 0.25:
		return True

	return False


def _parse_text_line(raw_text: str, min_alpha_chars: int = 3, allowed_pattern: str | None = None) -> str:
	"""Parse generic sentence-style OCR lines while suppressing blank noise."""
	text = _normalize_ocr_unicode(raw_text)
	text = _strip_non_sentence_noise(text, allowed_pattern=allowed_pattern)
	text = _trim_trailing_garbage(text)
	if _is_effectively_blank_text(text, min_alpha_chars=min_alpha_chars):
		return ""
	return text


def _parse_text_line_standard(raw_text: str) -> str:
	"""Parser for non-empty UI text lines."""
	return _parse_text_line(raw_text, min_alpha_chars=4, allowed_pattern=r"[A-Za-z0-9 .,'\-]")


def _parse_text_line_sparse(raw_text: str) -> str:
	"""Parser for lines that are often blank and should reject weak OCR noise."""
	return _parse_text_line(raw_text, min_alpha_chars=5, allowed_pattern=r"[A-Za-z0-9 .,'\-]")


def _score_text_line_candidate(raw_text: str, parsed_value: Any, min_alpha_chars: int = 3) -> float:
	"""Score line candidates by language-like signal and penalize OCR junk."""
	if not isinstance(parsed_value, str) or not parsed_value:
		return -1.0

	alpha_count = sum(1 for char in parsed_value if char.isalpha())
	word_count = len([word for word in parsed_value.split(" ") if word])
	if alpha_count < min_alpha_chars:
		return -0.8

	raw_normalized = _normalize_ocr_unicode(raw_text)
	raw_junk_count = sum(1 for char in raw_normalized if not (char.isalnum() or char in " .,'-"))
	raw_junk_ratio = raw_junk_count / max(1, len(raw_normalized))

	score = 0.0
	score += min(4.0, alpha_count * 0.12)
	score += min(3.0, word_count * 0.9)
	score += min(2.0, len(parsed_value) * 0.05)
	score -= raw_junk_ratio * 6.0

	if _is_effectively_blank_text(parsed_value, min_alpha_chars=min_alpha_chars):
		score -= 2.5

	if parsed_value.endswith("."):
		score += 0.1

	return score


def _score_text_line_standard(raw_text: str, parsed_value: Any) -> float:
	"""Scoring profile for expected populated lines."""
	return _score_text_line_candidate(raw_text, parsed_value, min_alpha_chars=4)


def _score_text_line_sparse(raw_text: str, parsed_value: Any) -> float:
	"""Stricter scoring profile for usually-empty lines."""
	return _score_text_line_candidate(raw_text, parsed_value, min_alpha_chars=5)

MODE_FIELD = OCRField(
	name="mode",
	ideal=ROIBox(top=0.04, bottom=0.12, left=0.12, right=0.88),
	fallback=ROIBox(top=0.00, bottom=0.18, left=0.18, right=0.82),
	tesseract_config="--psm 7",
	value_parser=_format_mode_text,
)

TEMPERATURE_FIELD = OCRField(
	name="temperature",
	ideal=ROIBox(top=0.16, bottom=0.42, left=0.32, right=0.64),
	fallback=ROIBox(top=0.12, bottom=0.52, left=0.24, right=0.68),
	tesseract_config="--psm 7 -c tessedit_char_whitelist=0123456789O",
	value_parser=_parse_temperature_text,
	value_scorer=_score_temperature_candidate,
	empty_value=None,
)

DASHBOARD_INFO_LINE_1 = OCRField(
	name="dashboard_info_line_1",
	ideal=ROIBox(top=0.47, bottom=0.57, left=0.10, right=0.90),
	fallback=ROIBox(top=0.45, bottom=0.70, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-",
	value_parser=_parse_text_line_standard,
	value_scorer=_score_text_line_standard,
	empty_value="",
	min_alpha_chars=4,
	blank_score_threshold=2.6,
	strip_trailing_garbage=True,
	allowed_pattern=r"[A-Za-z0-9 .,'\-]",
)

DASHBOARD_INFO_LINE_2 = OCRField(
	name="dashboard_info_line_2",
	ideal=ROIBox(top=0.57, bottom=0.66, left=0.10, right=0.90),
	fallback=ROIBox(top=0.55, bottom=0.75, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-",
	value_parser=_parse_text_line_sparse,
	value_scorer=_score_text_line_sparse,
	empty_value="",
	min_alpha_chars=5,
	blank_score_threshold=3.2,
	strip_trailing_garbage=True,
	allowed_pattern=r"[A-Za-z0-9 .,'\-]",
)

DASHBOARD_INFO_LINE_3 = OCRField(
	name="dashboard_info_line_3",
	ideal=ROIBox(top=0.67, bottom=0.76, left=0.10, right=0.90),
	fallback=ROIBox(top=0.65, bottom=0.80, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-",
	value_parser=_parse_text_line_standard,
	value_scorer=_score_text_line_standard,
	empty_value="",
	min_alpha_chars=4,
	blank_score_threshold=2.4,
	strip_trailing_garbage=True,
	allowed_pattern=r"[A-Za-z0-9 .,'\-]",
)

TIME_BAR = OCRField(
	name="time_bar",
	ideal=ROIBox(top=0.91, bottom=1.00, left=0.83, right=1.00),
	fallback=ROIBox(top=0.82, bottom=1.00, left=0.68, right=1.00),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= 0123456789AMP:",
)

DATE_BAR = OCRField(
	name="date_bar",
	ideal=ROIBox(top=0.91, bottom=1.00, left=0.63, right=0.82),
	fallback=ROIBox(top=0.82, bottom=1.00, left=0.68, right=1.00),
	tesseract_config="--psm 7 -c tessedit_char_whitelist=0123456789/",
)

ACTIVE_FAULT_TEXT = OCRField(
	name="active_fault_text",
	ideal=ROIBox(top=0.20, bottom=0.80, left=0.10, right=0.90),
	fallback=ROIBox(top=0.15, bottom=0.85, left=0.05, right=0.95),
	tesseract_config="--psm 6",
)

SCHEDULE_INFO_TEXT = OCRField(
	name="schedule_info_text",
	ideal=ROIBox(top=0.30, bottom=0.70, left=0.10, right=0.90),
	fallback=ROIBox(top=0.25, bottom=0.75, left=0.05, right=0.95),
	tesseract_config="--psm 6",
)

CURRENT_LAYOUT = DisplayLayout(
	name="bwc_water_heater_lcd_v1",
	menu_tree=ROOT_MENU,
	display_aspect_ratio=1.75 / 2.25,
	temperature_range_f=(80, 220),
	fields={
		"mode": MODE_FIELD,
		"temperature": TEMPERATURE_FIELD,
		"dashboard_info_line_1": DASHBOARD_INFO_LINE_1,
		"dashboard_info_line_2": DASHBOARD_INFO_LINE_2,
		"dashboard_info_line_3": DASHBOARD_INFO_LINE_3,
		"time_bar": TIME_BAR,
		"date_bar": DATE_BAR,
		# "active_fault_text": ACTIVE_FAULT_TEXT,
		# "schedule_info_text": SCHEDULE_INFO_TEXT,
	},
)
