from pathlib import Path

import cv2

from src.vision.vision_engine import _build_display_mask, _find_display_contour


def test_display_detector_recovers_real_samples():
    sample_dir = Path(__file__).resolve().parents[1] / "src" / "vision" / "capture_samples"
    sample_paths = sorted(sample_dir.glob("fail*.jpg"))
    assert sample_paths, "expected sample captures to be present"

    hits = 0
    for sample_path in sample_paths:
        frame = cv2.imread(str(sample_path))
        assert frame is not None, f"failed to load {sample_path}"
        mask = _build_display_mask(frame)
        contour = _find_display_contour(mask)
        if contour is not None:
            hits += 1

    assert hits >= 2, f"expected at least 2 detected displays, got {hits}"
