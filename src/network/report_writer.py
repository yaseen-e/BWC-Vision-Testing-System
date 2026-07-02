import csv
from pathlib import Path
from src.vision.display_layouts import HOME_MENU, collect_field_names
from src import data_manager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_REPORT_DIR = PROJECT_ROOT / "data" / "test_reports"
REPORT_FIELD_NAMES: tuple[str, ...] = collect_field_names(HOME_MENU)


def get_report_field_names() -> tuple[str, ...]:
    return REPORT_FIELD_NAMES

def open_test_report() -> tuple[object, csv.DictWriter, Path]:
    """Create a timestamped CSV report for this run and write header row."""
    TEST_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_id = data_manager.build_capture_id("test_report")
    report_path = TEST_REPORT_DIR / f"{report_id}.csv"
    report_file = report_path.open("w", newline="", encoding="utf-8")
    fieldnames = ["step", "menu"] + list(REPORT_FIELD_NAMES)
    writer = csv.DictWriter(report_file, fieldnames=fieldnames)
    writer.writeheader()
    report_file.flush()
    return report_file, writer, report_path
