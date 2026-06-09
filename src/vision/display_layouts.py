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
	fallback: ROIBox
	tesseract_config: str


@dataclass(frozen=True)
class MenuNode:
	key: str
	label: str
	children: tuple["MenuNode", ...] = ()


@dataclass(frozen=True)
class DisplayLayout:
	name: str
	menu_tree: MenuNode
	known_modes: tuple[str, ...]
	mode_keywords: Mapping[str, tuple[str, ...]]
	temperature_range_f: tuple[int, int]
	fields: Mapping[str, OCRField] = field(default_factory=dict)


ROOT_MENU = MenuNode(
	key="dashboard",
	label="Dashboard Home",
	children=(
		MenuNode(key="active_faults", label="Active Faults"),
		MenuNode(key="history", label="History", children=(
			MenuNode(key="historical_faults", label="Historical Faults"),
		)),
		MenuNode(key="first_time_setup", label="First Time Setup"),
		MenuNode(key="get_connected", label="Get Connected"),
		MenuNode(key="pro_tools", label="Pro Tools"),
		MenuNode(key="schedules", label="Schedules", children=(
			MenuNode(key="schedules_current", label="Current"),
			MenuNode(key="schedules_get", label="Get"),
			MenuNode(key="schedules_set", label="Set"),
			MenuNode(key="schedules_status", label="Status"),
		)),
		MenuNode(key="system", label="System", children=(
			MenuNode(key="system_info", label="Info"),
			MenuNode(key="system_fw_version", label="FwVersion"),
		)),
		MenuNode(key="control", label="Control", children=(
			MenuNode(key="control_heat_mode", label="HeatMode"),
			MenuNode(key="control_setpoint", label="Setpoint"),
			MenuNode(key="control_status", label="Status"),
			MenuNode(key="control_states", label="States"),
		)),
		MenuNode(key="io", label="IO", children=(
			MenuNode(key="io_anode", label="Anode"),
			MenuNode(key="io_relays", label="Relays"),
			MenuNode(key="io_temps", label="Temps"),
			MenuNode(key="io_sensors", label="Sensors"),
		)),
		MenuNode(key="energy", label="Energy", children=(
			MenuNode(key="energy_info", label="Info"),
			MenuNode(key="energy_usage", label="Usage"),
		)),
	),
)

MODE_FIELD = OCRField(
	name="mode",
	ideal=ROIBox(top=0.00, bottom=0.16, left=0.12, right=0.88),
	fallback=ROIBox(top=0.00, bottom=0.18, left=0.18, right=0.82),
	tesseract_config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ+ ",
)

TEMPERATURE_FIELD = OCRField(
	name="temperature",
	ideal=ROIBox(top=0.16, bottom=0.48, left=0.28, right=0.63),
	fallback=ROIBox(top=0.12, bottom=0.52, left=0.24, right=0.68),
	tesseract_config="--psm 7 -c tessedit_char_whitelist=0123456789Ool|SsbZ",
)

BANNER_SYMBOLS_FIELD = OCRField(
	name="banner_symbols",
	ideal=ROIBox(top=0.84, bottom=0.98, left=0.72, right=0.99),
	fallback=ROIBox(top=0.82, bottom=1.00, left=0.68, right=1.00),
	tesseract_config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-_:/(). ",
)

DASHBOARD_INFO_LINE_1 = OCRField(
	name="dashboard_info_line_1",
	ideal=ROIBox(top=0.50, bottom=0.65, left=0.10, right=0.90),
	fallback=ROIBox(top=0.45, bottom=0.70, left=0.05, right=0.95),
	tesseract_config="--psm 7",
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
	known_modes=("HYBRID", "HYBRID PLUS", "HEAT PUMP", "ELECTRIC", "VACATION"),
	mode_keywords={
		"HYBRID PLUS": ("HYBRID", "PLUS"),
		"HEAT PUMP": ("HEAT", "PUMP"),
		"HYBRID": ("HYBRID",),
		"ELECTRIC": ("ELECTRIC",),
		"VACATION": ("VACATION",),
	},
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
