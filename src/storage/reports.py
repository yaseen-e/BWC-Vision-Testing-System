"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/storage/reports.py - Test Report CSV Generator
Handles dynamic schema resolution and creation of CSV test reports.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import TextIO

from src.storage import captures
from src.vision.layouts import HOME_MENU, collect_field_names

# =============================================================================
# PATHS & FIELD SCHEMAS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_REPORT_DIR = PROJECT_ROOT / "data" / "test_reports"
REPORT_FIELD_NAMES: tuple[str, ...] = collect_field_names(HOME_MENU)


# =============================================================================
# REPORT WRITER FUNCTIONS
# =============================================================================

def get_report_field_names() -> tuple[str, ...]:
	"""Return ordered OCR field names to include in test report columns."""
	return REPORT_FIELD_NAMES


def open_test_report() -> tuple[TextIO, csv.DictWriter, Path]:
	"""Create a timestamped CSV report for this test run and write header row."""
	TEST_REPORT_DIR.mkdir(parents=True, exist_ok=True)
	report_id = captures.build_capture_id("test_report")
	report_path = TEST_REPORT_DIR / f"{report_id}.csv"

	report_file = report_path.open("w", newline="", encoding="utf-8")
	fieldnames = ["step", "menu"] + list(REPORT_FIELD_NAMES)
	writer = csv.DictWriter(report_file, fieldnames=fieldnames)

	writer.writeheader()
	report_file.flush()

	return report_file, writer, report_path
