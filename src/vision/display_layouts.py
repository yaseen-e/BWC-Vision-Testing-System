"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/vision/display_layouts.py - Display Layout Schema
Defines the ordered menu tree and OCR regions of interest (ROIs) 
for the LCD UI.
"""
# commit test
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
class ContextNode:
	key: str
	label: str
	route_here: tuple[tuple[str, ...], ...] = ()
	return_route: tuple[tuple[str, ...], ...] = ()
	fields: tuple[OCRField, ...] = ()
	children: tuple["ContextNode", ...] = ()


DISPLAY_ASPECT_RATIO = 1.75 / 2.25
TEMPERATURE_RANGE_F = (80, 220)

def iter_context_nodes(root: ContextNode) -> tuple[ContextNode, ...]:
	"""Return nodes in pre-order traversal for deterministic iteration."""
	nodes: list[ContextNode] = []

	def _walk(node: ContextNode) -> None:
		nodes.append(node)
		for child in node.children:
			_walk(child)

	_walk(root)
	return tuple(nodes)


def find_context_node(root: ContextNode, key: str) -> Optional[ContextNode]:
	for node in iter_context_nodes(root):
		if node.key == key:
			return node
	return None


def find_parent_context(root: ContextNode, child_key: str) -> Optional[ContextNode]:
	def _walk(node: ContextNode, parent: Optional[ContextNode]) -> Optional[ContextNode]:
		if node.key == child_key:
			return parent
		for child in node.children:
			result = _walk(child, node)
			if result is not None:
				return result
		return None

	return _walk(root, None)


def collect_field_names(root: ContextNode) -> tuple[str, ...]:
	"""Collect unique OCR field names in tree traversal order."""
	names: list[str] = []
	for node in iter_context_nodes(root):
		for field_def in node.fields:
			if field_def.name not in names:
				names.append(field_def.name)
	return tuple(names)


def apply_navigation_command(
	root: ContextNode,
	current_menu: ContextNode,
	transition_buffer: tuple[str, ...],
	command_token: str,
) -> tuple[ContextNode, tuple[str, ...], bool]:
	"""
	Apply one command token to menu navigation.

	Returns:
		(new_menu, new_transition_buffer, sequence_broken)
	"""
	working_buffer = transition_buffer + (command_token,)
	parent = find_parent_context(root, current_menu.key)

	candidates: list[tuple[tuple[str, ...], ContextNode]] = []
	for child in current_menu.children:
		for route in child.route_here:
			candidates.append((route, child))

	if parent is not None:
		for route in current_menu.return_route:
			candidates.append((route, parent))

	for route, destination in candidates:
		if route == working_buffer:
			return destination, (), False

	for route, _ in candidates:
		if len(working_buffer) <= len(route) and route[:len(working_buffer)] == working_buffer:
			return current_menu, working_buffer, False

	return current_menu, working_buffer, True

MODE_FIELD = OCRField(
	name="mode",
	ideal=ROIBox(top=0.03, bottom=0.14, left=0.25, right=0.75),
	fallback=ROIBox(top=0.00, bottom=0.18, left=0.18, right=0.82),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= ABCDEFGHIJKLMNOPQRSTUVWXYZ:",
)

TEMPERATURE_FIELD = OCRField(
	name="temperature",
	ideal=ROIBox(top=0.16, bottom=0.42, left=0.28, right=0.64),
	fallback=ROIBox(top=0.12, bottom=0.52, left=0.24, right=0.68),
	tesseract_config="--psm 7 -c tessedit_char_whitelist=0123456789",
)

DASHBOARD_INFO_LINE_1 = OCRField(
	name="dashboard_info_line_1",
	ideal=ROIBox(top=0.45, bottom=0.56, left=0.10, right=0.90),
	fallback=ROIBox(top=0.45, bottom=0.70, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.",
)

DASHBOARD_INFO_LINE_2 = OCRField(
	name="dashboard_info_line_2",
	ideal=ROIBox(top=0.56, bottom=0.65, left=0.10, right=0.90),
	fallback=ROIBox(top=0.55, bottom=0.75, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.",
)

DASHBOARD_INFO_LINE_3 = OCRField(
	name="dashboard_info_line_3",
	ideal=ROIBox(top=0.65, bottom=0.77, left=0.10, right=0.90),
	fallback=ROIBox(top=0.65, bottom=0.80, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.",
)

TIME_FIELD = OCRField(
	name="time_field",
	ideal=ROIBox(top=0.91, bottom=1.00, left=0.83, right=1.00),
	fallback=ROIBox(top=0.82, bottom=1.00, left=0.68, right=1.00),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= 0123456789AMP:",
)

DATE_FIELD = OCRField(
	name="date_field",
	ideal=ROIBox(top=0.91, bottom=1.00, left=0.64, right=0.82),
	fallback=ROIBox(top=0.82, bottom=1.00, left=0.63, right=0.82),
	tesseract_config="--psm 7 -c tessedit_char_whitelist=0123456789/",
)

WIFI_ICON_FIELD = OCRField(
	name="wifi_icon",
	ideal=ROIBox(top=0.91, bottom=1.00, left=0.56, right=0.63),
	fallback=ROIBox(top=0.82, bottom=1.00, left=0.48, right=0.68),
	tesseract_config="",
)

CALENDAR_ICON_FIELD = OCRField(
	name="calendar_icon",
	ideal=ROIBox(top=0.91, bottom=1.00, left=0.49, right=0.56),
	fallback=ROIBox(top=0.82, bottom=1.00, left=0.34, right=0.56),
	tesseract_config="",
)

STATUS_BAR = tuple((
	TIME_FIELD,
	DATE_FIELD,
	WIFI_ICON_FIELD,
	CALENDAR_ICON_FIELD,
))

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

ACTIVE_FAULTS_ERROR_CODE_1 = OCRField(
	name="active_faults_error_code_1",
	ideal=ROIBox(top=0.15, bottom=0.30, left=0.05, right=0.85),
	fallback=ROIBox(top=0.15, bottom=0.35, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist=abcdefgheijklmnopqrstuvwxyz0123456789. ",
)

ACTIVE_FAULTS_ERROR_CODE_2 = OCRField(
	name="active_faults_error_code_2",
	ideal=ROIBox(top=0.30, bottom=0.45, left=0.05, right=0.85),
	fallback=ROIBox(top=0.15, bottom=0.35, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist=abcdefgheijklmnopqrstuvwxyz0123456789. ",
)

ACTIVE_FAULTS_ERROR_CODE_3 = OCRField(
	name="active_faults_error_code_3",
	ideal=ROIBox(top=0.45, bottom=0.60, left=0.05, right=0.85),
	fallback=ROIBox(top=0.15, bottom=0.35, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist=abcdefgheijklmnopqrstuvwxyz0123456789. ",
)

ACTIVE_FAULTS_ERROR_CODE_4 = OCRField(
	name="active_faults_error_code_4",
	ideal=ROIBox(top=0.60, bottom=0.75, left=0.05, right=0.85),
	fallback=ROIBox(top=0.15, bottom=0.35, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist=abcdefgheijklmnopqrstuvwxyz0123456789. ",
)

ACTIVE_FAULTS_ERROR_CODE_5 = OCRField(
	name="active_faults_error_code_5",
	ideal=ROIBox(top=0.75, bottom=0.90, left=0.05, right=0.85),
	fallback=ROIBox(top=0.15, bottom=0.35, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist=abcdefgheijklmnopqrstuvwxyz0123456789. ",
)

LOCATION_FIELD = OCRField(
	name="location_field",
	ideal=ROIBox(top=0.30, bottom=0.70, left=0.10, right=0.90),
	fallback=ROIBox(top=0.25, bottom=0.75, left=0.05, right=0.95),
	tesseract_config="--psm 6",
)

HOME_MENU = ContextNode(
	key="homescreen",
	label="Home Screen",
	fields=(
		MODE_FIELD,
		TEMPERATURE_FIELD,
		DASHBOARD_INFO_LINE_1,
		DASHBOARD_INFO_LINE_2,
		DASHBOARD_INFO_LINE_3,
		*STATUS_BAR,
	),
	children=(
		ContextNode(key="active_faults_screen", label="Active Faults Screen", route_here=(("RIGHT",),), return_route=(("MENU", "SELECT"),), children=(
			ContextNode(key="active_faults_list", label="Active Faults List", route_here=(("SELECT",),), return_route=(("BACK",),),
			fields=(
				ACTIVE_FAULTS_ERROR_CODE_1,
				ACTIVE_FAULTS_ERROR_CODE_2,
				ACTIVE_FAULTS_ERROR_CODE_3,
				ACTIVE_FAULTS_ERROR_CODE_4,
				ACTIVE_FAULTS_ERROR_CODE_5,
			),
			children=()),
		)),
		ContextNode(key="system_status_top", label="System Status 1/2", route_here=(("MENU", "DOWN", "SELECT"),), return_route=(("BACK", "BACK"),), children=(
			ContextNode(key="system_status_bottom", label="System Status 2/2", route_here=(("DOWN",),), return_route=(("UP",),), ),
		)),
		ContextNode(key="settings", label="Settings", route_here=(("MENU", "RIGHT", "DOWN", "SELECT"), ("MENU", "DOWN", "RIGHT", "SELECT")), return_route=(("MENU",),), children=(
			ContextNode(key="location", label="Location", route_here=(("DOWN", "DOWN", "SELECT", "DOWN", "DOWN", "SELECT"),), return_route=(("BACK", "BACK"),), fields=(
				LOCATION_FIELD,
			)),
		)),
		ContextNode(key="schedules", label="Schedules", route_here=(("MENU", "RIGHT", "RIGHT", "SELECT"),), return_route=(("MENU", "SELECT"),), children=()),
	),
)
