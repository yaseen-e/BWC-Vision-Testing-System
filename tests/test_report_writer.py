import csv
import sys
import types
from pathlib import Path


def test_open_test_report_uses_only_home_screen_columns(tmp_path, monkeypatch):
    cv2_stub = types.ModuleType("cv2")
    numpy_stub = types.ModuleType("numpy")
    sys.modules.setdefault("cv2", cv2_stub)
    sys.modules.setdefault("numpy", numpy_stub)

    from src.network import report_writer

    monkeypatch.setattr(report_writer, "TEST_REPORT_DIR", tmp_path)

    report_file, writer, report_path = report_writer.open_test_report()

    assert writer.fieldnames == ["Mode", "Mode_Conf", "Temp", "Temp_Conf"]

    report_file.close()
    with report_path.open("r", newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))

    assert header == ["Mode", "Mode_Conf", "Temp", "Temp_Conf"]
