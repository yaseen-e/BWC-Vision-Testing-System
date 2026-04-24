#!/usr/bin/env python3
"""Rotate (delete) old capture images under data/captures by age.

Default behavior is a safe dry run. Use --apply to actually delete files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class CleanupResult:
    scanned_files: int
    matched_files: int
    deleted_files: int
    kept_files: int
    matched_bytes: int
    deleted_bytes: int


def _bytes_to_mib(value: int) -> float:
    return value / (1024 * 1024)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _iter_cleanup_candidates(captures_dir: Path, cutoff_utc: datetime) -> tuple[list[Path], int, int]:
    candidates: list[Path] = []
    scanned_files = 0
    matched_bytes = 0

    for path in captures_dir.rglob("*"):
        if not path.is_file():
            continue

        scanned_files += 1

        # Keep hidden marker files like .gitkeep and unknown artifacts.
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


def rotate_captures(captures_dir: Path, retention_days: int, apply_changes: bool) -> CleanupResult:
    if retention_days < 1:
        raise ValueError("retention_days must be >= 1")

    if not captures_dir.exists() or not captures_dir.is_dir():
        raise FileNotFoundError(f"Capture directory not found: {captures_dir}")

    project_root = Path(__file__).resolve().parents[1]
    allowed_root = project_root / "data" / "captures"
    if not _is_within(captures_dir, allowed_root):
        raise ValueError(f"Refusing to clean outside {allowed_root}: {captures_dir}")

    cutoff_utc = datetime.now(tz=timezone.utc) - timedelta(days=retention_days)
    candidates, scanned_files, matched_bytes = _iter_cleanup_candidates(captures_dir, cutoff_utc)

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


def _build_parser(default_captures_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Delete old capture images under data/captures.",
    )
    parser.add_argument(
        "--captures-dir",
        type=Path,
        default=default_captures_dir,
        help="Capture directory to clean (must be within data/captures).",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=14,
        help="Keep files newer than this many days (default: 14).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply deletion. Without this flag, runs as dry run.",
    )
    return parser


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    default_captures_dir = project_root / "data" / "captures"

    parser = _build_parser(default_captures_dir)
    args = parser.parse_args()

    result = rotate_captures(
        captures_dir=args.captures_dir,
        retention_days=args.retention_days,
        apply_changes=args.apply,
    )

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}] Capture cleanup summary")
    print(f"Directory: {args.captures_dir.resolve()}")
    print(f"Retention days: {args.retention_days}")
    print(f"Scanned files: {result.scanned_files}")
    print(f"Matched old images: {result.matched_files}")
    print(f"Kept files: {result.kept_files}")

    if args.apply:
        print(f"Deleted files: {result.deleted_files}")
        print(f"Freed space (MiB): {_bytes_to_mib(result.deleted_bytes):.2f}")
    else:
        print(f"Would delete files: {result.matched_files}")
        print(f"Would free space (MiB): {_bytes_to_mib(result.matched_bytes):.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
