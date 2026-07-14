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
import numpy as np


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


def _matches_route_exact(route: tuple[str, ...], stream: tuple[str, ...]) -> bool:
	"""Return True when the stream matches the route pattern exactly."""
	if not route:
		return not stream

	token = route[0]
	if token.endswith("*"):
		base_token = token[:-1]
		repeat_count = 0
		while repeat_count < len(stream) and stream[repeat_count] == base_token:
			repeat_count += 1
		for consumed_count in range(repeat_count + 1):
			if _matches_route_exact(route[1:], stream[consumed_count:]):
				return True
		return False

	if not stream or stream[0] != token:
		return False
	return _matches_route_exact(route[1:], stream[1:])


def _matches_route_prefix(route: tuple[str, ...], stream: tuple[str, ...]) -> bool:
	"""Return True when the stream is a valid prefix of the route pattern."""
	if not stream:
		return True
	if not route:
		return False

	token = route[0]
	if token.endswith("*"):
		base_token = token[:-1]
		repeat_count = 0
		while repeat_count < len(stream) and stream[repeat_count] == base_token:
			repeat_count += 1
		for consumed_count in range(repeat_count + 1):
			if _matches_route_prefix(route[1:], stream[consumed_count:]):
				return True
		return False

	if stream[0] != token:
		return False
	return _matches_route_prefix(route[1:], stream[1:])


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
		if _matches_route_exact(route, working_buffer):
			return destination, (), False

	for route, _ in candidates:
		if _matches_route_prefix(route, working_buffer):
			return current_menu, working_buffer, False

	return current_menu, working_buffer, True

# Create OCRFields

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
	tesseract_config="--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.",
)

DASHBOARD_INFO_LINE_3 = OCRField(
	name="dashboard_info_line_3",
	ideal=ROIBox(top=0.65, bottom=0.77, left=0.10, right=0.90),
	fallback=ROIBox(top=0.65, bottom=0.80, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.",
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
	ideal=ROIBox(top=0.91, bottom=1.00, left=0.48, right=0.55),
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
	tesseract_config="--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.",
)

ACTIVE_FAULTS_ERROR_CODE_2 = OCRField(
	name="active_faults_error_code_2",
	ideal=ROIBox(top=0.30, bottom=0.45, left=0.05, right=0.85),
	fallback=ROIBox(top=0.15, bottom=0.35, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.",
)

ACTIVE_FAULTS_ERROR_CODE_3 = OCRField(
	name="active_faults_error_code_3",
	ideal=ROIBox(top=0.45, bottom=0.60, left=0.05, right=0.85),
	fallback=ROIBox(top=0.15, bottom=0.35, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.",
)

ACTIVE_FAULTS_ERROR_CODE_4 = OCRField(
	name="active_faults_error_code_4",
	ideal=ROIBox(top=0.60, bottom=0.75, left=0.05, right=0.85),
	fallback=ROIBox(top=0.15, bottom=0.35, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.",
)

ACTIVE_FAULTS_ERROR_CODE_5 = OCRField(
	name="active_faults_error_code_5",
	ideal=ROIBox(top=0.75, bottom=0.90, left=0.05, right=0.85),
	fallback=ROIBox(top=0.15, bottom=0.35, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.",
)

LOCATION_FIELD = OCRField(
	name="location_field",
	ideal=ROIBox(top=0.35, bottom=0.50, left=0.30, right=0.70),
	fallback=ROIBox(top=0.25, bottom=0.75, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= ABCDEFGHIJKLMNOPQRSTUVWXYZ",
)

USER_SCHEDULE_NAME_1 = OCRField(
	name="user_schedule_name_1",
	ideal=ROIBox(top=0.28, bottom=0.38, left=0.08, right=0.67),
	fallback=ROIBox(top=0.08, bottom=0.35, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
)

USER_SCHEDULE_DELETED_TEXT = OCRField(
	name="user_schedule_deleted_text",
	ideal=ROIBox(top=0.03, bottom=0.14, left=0.23, right=0.77),
	fallback=ROIBox(top=0.00, bottom=0.18, left=0.18, right=0.82),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= ABCDEFGHIJKLMNOPQRSTUVWXYZ:",
)

TOU_SCHEDULE_NAME_1 = OCRField(
	name="tou_schedule_name_1",
	ideal=ROIBox(top=0.28, bottom=0.38, left=0.08, right=0.67),
	fallback=ROIBox(top=0.08, bottom=0.35, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
)

TOU_SCHEDULE_DELETED_TEXT = OCRField(
	name="tou_schedule_deleted_text",
	ideal=ROIBox(top=0.03, bottom=0.14, left=0.23, right=0.77),
	fallback=ROIBox(top=0.00, bottom=0.18, left=0.18, right=0.82),
	tesseract_config="--psm 7 -c tessedit_char_whitelist= ABCDEFGHIJKLMNOPQRSTUVWXYZ:",
)

COMPRESSOR_RELAY_STATE = OCRField(
	name="compressor_relay_state",
	ideal=ROIBox(top=0.83, bottom=0.92, left=0.06, right=0.21),
	fallback=ROIBox(top=0.15, bottom=0.35, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist=OFN",
)

UPPER_RELAY_STATE = OCRField(
	name="upper_relay_state",
	ideal=ROIBox(top=0.45, bottom=0.54, left=0.06, right=0.21),
	fallback=ROIBox(top=0.15, bottom=0.35, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist=OFN",
)

LOWER_RELAY_STATE = OCRField(
	name="lower_relay_state",
	ideal=ROIBox(top=0.64, bottom=0.73, left=0.06, right=0.21),
	fallback=ROIBox(top=0.15, bottom=0.35, left=0.05, right=0.95),
	tesseract_config="--psm 7 -c tessedit_char_whitelist=OFN",
)

# Create the menu tree

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
        ContextNode(
            key="active_faults_screen",
            label="Active Faults Screen",
            route_here=(("RIGHT",),),
            return_route=(("MENU", "SELECT"), ("LEFT",)),
            children=(
                ContextNode(
                    key="active_faults_list",
                    label="Active Faults List",
                    route_here=(("SELECT",),),
                    return_route=(("BACK",),),
                    fields=(
                        ACTIVE_FAULTS_ERROR_CODE_1,
                        ACTIVE_FAULTS_ERROR_CODE_2,
                        ACTIVE_FAULTS_ERROR_CODE_3,
                        ACTIVE_FAULTS_ERROR_CODE_4,
                        ACTIVE_FAULTS_ERROR_CODE_5,
                    ),
                    children=(),
                ),
            ),
        ),
        ContextNode(
            key="system_status_top",
            label="System Status 1/2",
            route_here=(("MENU", "DOWN", "SELECT"),),
            return_route=(("BACK", "BACK"), ("BACK", "UP", "SELECT"),),
			fields=(
				COMPRESSOR_RELAY_STATE,
			),
            children=(
                ContextNode(
                    key="system_status_bottom",
                    label="System Status 2/2",
                    route_here=(("DOWN",),),
                    return_route=(("UP",),),
					fields=(
						UPPER_RELAY_STATE,
						LOWER_RELAY_STATE,
					),
                    children=(),
                ),
            ),
        ),
        ContextNode(
            key="settings",
            label="Settings",
            route_here=(("MENU", "RIGHT", "DOWN", "SELECT"), ("MENU", "DOWN", "RIGHT", "SELECT")),
            return_route=(("MENU",),),
            children=(
                ContextNode(
                    key="location",
                    label="Location",
                    route_here=(("DOWN", "DOWN", "SELECT", "DOWN", "DOWN", "SELECT"),),
                    return_route=(("BACK", "BACK"),),
                    fields=(
                        LOCATION_FIELD,
                    ),
                    children=(),
                ),
            ),
        ),
        ContextNode(
            key="schedules",
            label="Schedules",
            route_here=(("MENU", "RIGHT", "RIGHT", "SELECT",),),
            return_route=(("BACK","BACK"),),
            fields=(),
			children=(
				ContextNode(
					key="manage_schedules",
					label="Manage Schedules",
					route_here=(("SELECT",),),
					return_route=(("BACK",),),
					fields=(),
					children=(
						ContextNode(
							key="user_schedules_list",
							label="User Schedules List",
							route_here=(("SELECT",),),
							return_route=(("BACK",),),
							fields=(
								USER_SCHEDULE_NAME_1,
								*STATUS_BAR,
							),
							children=(
								ContextNode(
									key="user_schedule_deleted_confirmation",
									label="User Schedule Deleted Confirmation",
									route_here=(("DOWN*", "SELECT") + ("DOWN",) * 7 + ("SELECT", "SELECT"),),
									return_route=(("SELECT",),),
									fields=(
										USER_SCHEDULE_DELETED_TEXT,
									),
									children=(),
								),
							),
						),
						ContextNode(
							key="tou_schedules_list",
							label="TOU Schedules List",
							route_here=(("DOWN", "DOWN", "SELECT"),),
							return_route=(("BACK",),),
							fields=(
								TOU_SCHEDULE_NAME_1,
							),
							children=(
								ContextNode(
									key="tou_schedule_deleted_confirmation",
									label="TOU Schedule Deleted Confirmation",
									route_here=(("DOWN*", "SELECT") + ("DOWN",) * 7 + ("SELECT", "SELECT"),),
									return_route=(("SELECT",),),
									fields=(
										TOU_SCHEDULE_DELETED_TEXT,
									),
									children=(),
								),
							),
						),
					),
				),
			),
		),
	),
)
