"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/vision/display_layouts.py - Display Layout Schema
Defines the ordered menu tree and OCR regions of interest (ROIs) 
for the LCD UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

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
	tesseract_config: str


@dataclass(frozen=True)
class MenuNode:
	key: str
	label: str
	route_here: list[set[str]] = field(default_factory=list)
	return_route: set[str] = field(default_factory=set)
	children: tuple["MenuNode", ...] = ()


@dataclass(frozen=True)
class DisplayLayout:
	name: str
	menu_tree: MenuNode
	display_aspect_ratio: float
	temperature_range_f: tuple[int, int]
	fields: Mapping[str, OCRField] = field(default_factory=dict)


ROOT_MENU = MenuNode(
	key="homescreen",
	label="Home Screen",
	children=(
		MenuNode(key="active_faults_screen", label="Active Faults Screen", route_here={"RIGHT"}, return_route={"MENU", "SELECT"}, children=(
			MenuNode(key="active_faults_list", label="Active Faults List", route_here={"SELECT"}, return_route={"BACK"}, children=()),
		)),
        MenuNode(key="system_status_top", label="System Status 1/2", route_here={"MENU", "DOWN", "SELECT"}, return_route={"BACK", "BACK"}, children=(
			MenuNode(key="system_status_bottom", label="System Status 2/2", route_here={"DOWN"}, return_route={"UP"}),
		)),
        MenuNode(key="settings", label="Settings", route_here=[{"MENU", "RIGHT", "DOWN", "SELECT"}, {"MENU", "DOWN", "RIGHT", "SELECT"}], return_route={"MENU"}, children=()),
		MenuNode(key="schedules", label="Schedules", route_here={"MENU", "RIGHT", "RIGHT", "SELECT"}, return_route={"MENU", "SELECT"}),
	),
)

MODE_FIELD = OCRField(
	name="mode",
	ideal=ROIBox(top=0.05, bottom=0.16, left=0.12, right=0.88),
	tesseract_config="--psm 7",
)

TEMPERATURE_FIELD = OCRField(
	name="temperature",
	ideal=ROIBox(top=0.16, bottom=0.48, left=0.28, right=0.63),
	tesseract_config="--psm 7 -c tessedit_char_whitelist=0123456789Ool|SsbZ",
)

BANNER_SYMBOLS_FIELD = OCRField(
	name="banner_symbols",
	ideal=ROIBox(top=0.84, bottom=0.98, left=0.72, right=0.99),
	tesseract_config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-_:/(). ",
)

DASHBOARD_INFO_LINE_1 = OCRField(
	name="dashboard_info_line_1",
	ideal=ROIBox(top=0.50, bottom=0.65, left=0.10, right=0.90),
	tesseract_config="--psm 7",
)

ACTIVE_FAULT_TEXT = OCRField(
	name="active_fault_text",
	ideal=ROIBox(top=0.20, bottom=0.80, left=0.10, right=0.90),
	tesseract_config="--psm 6",
)

SCHEDULE_INFO_TEXT = OCRField(
	name="schedule_info_text",
	ideal=ROIBox(top=0.30, bottom=0.70, left=0.10, right=0.90),
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
		"banner_symbols": BANNER_SYMBOLS_FIELD,
		"dashboard_info_line_1": DASHBOARD_INFO_LINE_1,
		"active_fault_text": ACTIVE_FAULT_TEXT,
		"schedule_info_text": SCHEDULE_INFO_TEXT,
	},
)
