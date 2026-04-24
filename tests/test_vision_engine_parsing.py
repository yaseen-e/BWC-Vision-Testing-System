"""Basic parsing tests for vision engine helper logic."""

import unittest

from src.vision import vision_engine


class TestVisionEngineParsing(unittest.TestCase):
    """Covers temperature extraction and unknown mode fallback."""

    def test_parse_temperature_text_extracts_float(self) -> None:
        # OCR can include extra symbols around digits.
        value = vision_engine.parse_temperature_text("TEMP: 120.5F")
        self.assertEqual(value, 120.5)

    def test_parse_mode_text_unknown_fallback(self) -> None:
        # Unrecognized text should safely map to UNKNOWN.
        mode = vision_engine.parse_mode_text("MODE: SOMETHING ELSE")
        self.assertEqual(mode, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
