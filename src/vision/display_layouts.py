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
)

DASHBOARD_INFO_LINE_1 = OCRField(
	name="dashboard_info_line_1",
	ideal=ROIBox(top=0.47, bottom=0.57, left=0.10, right=0.90),
	fallback=ROIBox(top=0.45, bottom=0.70, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.",
)

DASHBOARD_INFO_LINE_2 = OCRField(
	name="dashboard_info_line_2",
	ideal=ROIBox(top=0.57, bottom=0.66, left=0.10, right=0.90),
	fallback=ROIBox(top=0.55, bottom=0.75, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
)

DASHBOARD_INFO_LINE_3 = OCRField(
	name="dashboard_info_line_3",
	ideal=ROIBox(top=0.67, bottom=0.76, left=0.10, right=0.90),
	fallback=ROIBox(top=0.65, bottom=0.80, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
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
