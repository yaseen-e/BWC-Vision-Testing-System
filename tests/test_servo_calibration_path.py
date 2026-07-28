from pathlib import Path

from src.motion import servos


def test_servo_calibration_file_is_resolved_from_utilities_folder():
    expected_path = Path(__file__).resolve().parents[1] / "src" / "utilities" / "servo_calibration.json"
    assert servos.CAL_FILE == expected_path
    assert servos.CAL_FILE.exists()
