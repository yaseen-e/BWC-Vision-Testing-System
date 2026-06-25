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
	icon_template_path: str | None = None
	icon_match_threshold: float = 0.80


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


def _parse_text_line(raw_text: str) -> str:
	"""Parse OCR text lines dynamically without fixed phrase lists."""
	text = re.sub(r"\s+", " ", raw_text).strip()
	text = text.replace("_", " ")
	text = re.sub(r"\s+", " ", text).strip(" |:;,_-.")
	if not text:
		return ""

	alpha_count = sum(1 for char in text if char.isalpha())
	if alpha_count < 3:
		return ""

	alnum_count = sum(1 for char in text if char.isalnum())
	if alnum_count <= 0:
		return ""

	if (alpha_count / max(1, len(text))) < 0.35:
		return ""

	return text


def _score_text_line(raw_text: str, parsed_value: Any) -> float:
	"""Score dynamic text lines so meaningful text beats symbol noise."""
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

MODE_FIELD = OCRField(
	name="mode",
	ideal=ROIBox(top=0.03, bottom=0.14, left=0.12, right=0.88),
	fallback=ROIBox(top=0.00, bottom=0.18, left=0.18, right=0.82),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= ABCDEFGHIJKLMNOPQRSTUVWXYZ:",
	value_parser=_format_mode_text,
	allowed_pattern=r"[ A-Z:]"
)

TEMPERATURE_FIELD = OCRField(
	name="temperature",
	ideal=ROIBox(top=0.16, bottom=0.42, left=0.30, right=0.64),
	fallback=ROIBox(top=0.12, bottom=0.52, left=0.24, right=0.68),
	tesseract_config="--psm 7 -c tessedit_char_whitelist=0123456789",
	value_parser=_parse_temperature_text,
	value_scorer=_score_temperature_candidate,
	empty_value=None,
	allowed_pattern=r"[0-9]"
)

DASHBOARD_INFO_LINE_1 = OCRField(
	name="dashboard_info_line_1",
	ideal=ROIBox(top=0.45, bottom=0.57, left=0.10, right=0.90),
	fallback=ROIBox(top=0.45, bottom=0.70, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.",
	value_parser=_parse_text_line,
	value_scorer=_score_text_line,
	empty_value="",
	min_alpha_chars=0,
	blank_score_threshold=None,
	strip_trailing_garbage=False,
	allowed_pattern=r"[ A-Za-z.]",
)

DASHBOARD_INFO_LINE_2 = OCRField(
	name="dashboard_info_line_2",
	ideal=ROIBox(top=0.55, bottom=0.66, left=0.10, right=0.90),
	fallback=ROIBox(top=0.55, bottom=0.75, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.",
	value_parser=_parse_text_line,
	value_scorer=_score_text_line,
	empty_value="",
	min_alpha_chars=0,
	blank_score_threshold=None,
	strip_trailing_garbage=False,
	allowed_pattern=r"[ A-Za-z.]",
)

DASHBOARD_INFO_LINE_3 = OCRField(
	name="dashboard_info_line_3",
	ideal=ROIBox(top=0.64, bottom=0.77, left=0.10, right=0.90),
	fallback=ROIBox(top=0.65, bottom=0.80, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.",
	value_parser=_parse_text_line,
	value_scorer=_score_text_line,
	empty_value="",
	min_alpha_chars=0,
	blank_score_threshold=None,
	strip_trailing_garbage=False,
	allowed_pattern=r"[ A-Za-z.]",
)

TIME_BAR = OCRField(
	name="time_bar",
	ideal=ROIBox(top=0.91, bottom=1.00, left=0.83, right=1.00),
	fallback=ROIBox(top=0.82, bottom=1.00, left=0.68, right=1.00),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= 0123456789AMP:",
	allowed_pattern=r"[ 0-9AMP:]",
)

DATE_BAR = OCRField(
	name="date_bar",
	ideal=ROIBox(top=0.91, bottom=1.00, left=0.63, right=0.82),
	fallback=ROIBox(top=0.82, bottom=1.00, left=0.68, right=1.00),
	tesseract_config="--psm 7 -c tessedit_char_whitelist=0123456789/",
	allowed_pattern=r"[0-9/]",
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

WIFI_ICON_FIELD = OCRField(
	name="wifi_icon",
	ideal=ROIBox(top=0.91, bottom=1.00, left=0.55, right=0.62),
	fallback=ROIBox(top=0.82, bottom=1.00, left=0.48, right=0.68),
	tesseract_config="",
	empty_value=False,
	icon_template_path="vision/templates/wifi_on.png",
	icon_match_threshold=0.35,
)

CALENDAR_ICON_FIELD = OCRField(
	name="calendar_icon",
	ideal=ROIBox(top=0.91, bottom=1.00, left=0.48, right=0.55),
	fallback=ROIBox(top=0.82, bottom=1.00, left=0.34, right=0.56),
	tesseract_config="",
	empty_value=False,
	icon_template_path="vision/templates/schedule_running.png",
	icon_match_threshold=0.35,
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
		"wifi_icon": WIFI_ICON_FIELD,
		"calendar_icon": CALENDAR_ICON_FIELD,
		# "active_fault_text": ACTIVE_FAULT_TEXT,
		# "schedule_info_text": SCHEDULE_INFO_TEXT,
	},
)
