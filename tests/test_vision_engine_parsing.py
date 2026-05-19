"""Basic parsing and layout tests for vision engine helper logic."""

import unittest

from src.vision import vision_engine
from src.vision.display_layouts import BANNER_SYMBOLS_FIELD, CURRENT_LAYOUT, MODE_FIELD, TEMPERATURE_FIELD


class TestVisionEngineParsing(unittest.TestCase):
    """Covers temperature extraction, mode parsing, and layout definition."""

    def test_parse_temperature_text_extracts_integer(self) -> None:
        # OCR can include extra symbols around digits.
        value = vision_engine.parse_temperature_text("TEMP: 120.5F")
        self.assertEqual(value, 120)

    def test_parse_mode_text_returns_known_mode(self) -> None:
        # Unknown text should still resolve to one of the supported modes.
        mode = vision_engine.parse_mode_text("MODE: SOMETHING ELSE")
        self.assertIn(mode, CURRENT_LAYOUT.known_modes)

    def test_current_layout_exposes_expected_rois(self) -> None:
        self.assertEqual(MODE_FIELD.ideal.top, 0.00)
        self.assertEqual(MODE_FIELD.ideal.bottom, 0.16)
        self.assertEqual(MODE_FIELD.ideal.left, 0.12)
        self.assertEqual(MODE_FIELD.ideal.right, 0.88)
        self.assertEqual(TEMPERATURE_FIELD.ideal.top, 0.12)
        self.assertEqual(TEMPERATURE_FIELD.ideal.bottom, 0.56)
        self.assertEqual(TEMPERATURE_FIELD.ideal.left, 0.22)
        self.assertEqual(TEMPERATURE_FIELD.ideal.right, 0.70)
        self.assertIn("banner_symbols", CURRENT_LAYOUT.fields)
        self.assertEqual(BANNER_SYMBOLS_FIELD.name, "banner_symbols")
        child_keys = [child.key for child in CURRENT_LAYOUT.menu_tree.children]
        for expected in ["active_faults", "history", "first_time_setup", "system", "control", "energy", "io"]:
            self.assertIn(expected, child_keys)


if __name__ == "__main__":
    unittest.main()
