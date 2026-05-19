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
	key="root",
	label="Main Menu",
	children=(
		MenuNode(key="active_faults", label="Active Faults"),
		MenuNode(key="historical_faults", label="Historical Faults"),
		MenuNode(key="first_time_setup", label="First Time Setup"),
		MenuNode(key="get_connected", label="Get Connected"),
		MenuNode(key="pro_tools", label="Pro Tools"),
		MenuNode(key="schedules", label="Schedules"),
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
	ideal=ROIBox(top=0.12, bottom=0.56, left=0.22, right=0.70),
	fallback=ROIBox(top=0.12, bottom=0.58, left=0.24, right=0.78),
	tesseract_config="--psm 7 -c tessedit_char_whitelist=0123456789Ool|SsbZ",
)

BANNER_SYMBOLS_FIELD = OCRField(
	name="banner_symbols",
	ideal=ROIBox(top=0.84, bottom=0.98, left=0.72, right=0.99),
	fallback=ROIBox(top=0.82, bottom=1.00, left=0.68, right=1.00),
	tesseract_config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-_:/(). ",
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
	},
)
