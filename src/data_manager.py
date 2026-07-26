"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/data_manager.py - Data Manager
Centralizes capture naming, persistence, and retention cleanup for project data assets.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
from typing import Optional

import cv2
import numpy as np


# =============================================================================
# DATA MODELS & CONSTANTS
# =============================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class CleanupResult:
	"""Summary statistics from a retention cleanup pass."""

	scanned_files: int
	matched_files: int
	deleted_files: int
	kept_files: int
	matched_bytes: int
	deleted_bytes: int


# =============================================================================
# CAPTURE PERSISTENCE API
# =============================================================================

def build_capture_id(command_text: str) -> str:
	"""Create a stable, filename-safe ID for one capture event."""
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
	command_tag = "".join(
		char if char.isalnum() else "_" for char in command_text.lower()
	).strip("_")
	if not command_tag:
		command_tag = "read"
	return f"{timestamp}_{command_tag}"


def save_capture_frame(
	capture_dir: Path, frame: np.ndarray, capture_id: Optional[str]
) -> Optional[Path]:
	"""Persist one captured frame into the configured capture directory."""
	if frame is None:
		return None

	capture_dir.mkdir(parents=True, exist_ok=True)
	file_stem = _safe_capture_stem(capture_id)
	output_path = capture_dir / f"{file_stem}.jpg"
	if not cv2.imwrite(str(output_path), frame):
		return None

	return output_path


def cleanup_captures(
	captures_dir: Path,
	retention_days: int,
	apply_changes: bool,
	allowed_root: Optional[Path] = None,
) -> CleanupResult:
	"""Delete capture images older than retention_days (or preview in dry run)."""
	if retention_days < 1:
		raise ValueError("retention_days must be >= 1")

	if not captures_dir.exists() or not captures_dir.is_dir():
		raise FileNotFoundError(f"Capture directory not found: {captures_dir}")

	root = allowed_root or captures_dir
	if not _is_within(captures_dir, root):
		raise ValueError(f"Refusing to clean outside {root}: {captures_dir}")

	cutoff_utc = datetime.now(tz=timezone.utc) - timedelta(days=retention_days)
	candidates, scanned_files, matched_bytes = _iter_cleanup_candidates(
		captures_dir, cutoff_utc
	)

	deleted_files = 0
	deleted_bytes = 0

	for file_path in candidates:
		file_size = file_path.stat().st_size
		if apply_changes:
			file_path.unlink(missing_ok=True)
			deleted_files += 1
			deleted_bytes += file_size

	matched_files = len(candidates)
	kept_files = max(0, scanned_files - matched_files)

	return CleanupResult(
		scanned_files=scanned_files,
		matched_files=matched_files,
		deleted_files=deleted_files,
		kept_files=kept_files,
		matched_bytes=matched_bytes,
		deleted_bytes=deleted_bytes,
	)


# =============================================================================
# HELPER UTILITIES
# =============================================================================

def _safe_capture_stem(capture_id: Optional[str]) -> str:
	"""Normalize optional IDs into filename-safe stems."""
	if not capture_id:
		return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

	cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", capture_id).strip("._")
	if cleaned:
		return cleaned

	return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _is_within(path: Path, parent: Path) -> bool:
	"""Check whether path is nested within parent directory."""
	try:
		path.resolve().relative_to(parent.resolve())
		return True
	except ValueError:
		return False


def _iter_cleanup_candidates(
	captures_dir: Path, cutoff_utc: datetime
) -> tuple[list[Path], int, int]:
	"""Iterate capture files older than cutoff timestamp."""
	candidates: list[Path] = []
	scanned_files = 0
	matched_bytes = 0

	for path in captures_dir.rglob("*"):
		if not path.is_file():
			continue

		scanned_files += 1

		if path.name.startswith("."):
			continue

		if path.suffix.lower() not in IMAGE_EXTENSIONS:
			continue

		modified_utc = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
		if modified_utc >= cutoff_utc:
			continue

		candidates.append(path)
		matched_bytes += path.stat().st_size

	return candidates, scanned_files, matched_bytes
