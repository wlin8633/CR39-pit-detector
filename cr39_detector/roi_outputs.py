"""Allocate a non-overwriting, per-image identifier for an ROI output group."""

from contextlib import contextmanager
import json
import math
from pathlib import Path
import re


SUFFIXES = ("_state.cr39state", "_measurements.csv", "_results.png", "_results.bmp")


def _same_source(reported, current, csv):
    reported, current = Path(reported), Path(current)
    if not reported.is_absolute() or reported.name.casefold() != current.name.casefold():
        return False
    if reported.resolve() == current.resolve():
        return True
    try:
        return reported.samefile(current)
    except FileNotFoundError:
        # Co-located measurements and raw images can be moved together.
        return csv.parent.resolve() == current.parent.resolve()


def _measurement_header(path):
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            if line.startswith("# result_metadata="):
                return json.loads(line.removeprefix("# result_metadata="))
            if not line.startswith("# "):
                break
    return None


def _origin_and_rate(key):
    width, height, row, column, rate = key[1:6]
    if not math.isfinite(rate) or rate < 1 or min(width, height) < 1 or min(row, column) < 0:
        raise ValueError("Invalid tile geometry")
    return round(column * width * rate), round(row * height * rate), rate


def _map_saved_box(metadata, active_key, shape):
    key = metadata["result_key"]
    x0, y0, old_rate = _origin_and_rate(key)
    bbox = metadata.get("result_bbox")
    if bbox is None:
        # Batch Detect's result image is in the full raw frame's extended lattice.
        height, width = metadata["result_shape_hw"]
        bbox = (0, 0, width, height)
        x0, y0 = 0, 0
    x1, y1, x2, y2 = bbox
    if not all(math.isfinite(v) for v in bbox) or not (0 <= x1 < x2 and 0 <= y1 < y2):
        raise ValueError("Invalid saved ROI")
    cx, cy, rate = _origin_and_rate(active_key)
    height, width = shape[:2]
    x1 = max(0, (x1 + x0) * rate / old_rate - cx)
    y1 = max(0, (y1 + y0) * rate / old_rate - cy)
    x2 = min(width, (x2 + x0) * rate / old_rate - cx)
    y2 = min(height, (y2 + y0) * rate / old_rate - cy)
    return (x1, y1, x2, y2) if x1 < x2 and y1 < y2 else None


def read_saved_rois(directory, active_key, shape):
    """Return the visible saved-measurement ROIs in the active extended tile."""
    current = Path(active_key[0])
    pattern = re.compile(rf"{re.escape(current.stem)}(?:__(roi[0-9]+))?_measurements\.csv", re.IGNORECASE)
    regions = []
    for path in sorted(directory.iterdir()):
        match = pattern.fullmatch(path.name)
        if not match or not path.is_file():
            continue
        try:
            metadata = _measurement_header(path)
            if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
                continue
            if Path(metadata["result_key"][0]) != Path(metadata["source_image"]):
                continue
            if not _same_source(metadata["source_image"], current, path):
                continue
            bbox = _map_saved_box(metadata, active_key, shape)
            if bbox is not None:
                regions.append((match[1].casefold() if match[1] else "Saved", bbox))
        except (OSError, TypeError, ValueError, KeyError, IndexError, OverflowError):
            continue
    return regions


@contextmanager
def reserve_next_roi(directory, image_stem):
    aliases = "|".join(re.escape(suffix) for suffix in (*SUFFIXES, ".saving"))
    pattern = re.compile(rf"{re.escape(image_stem)}__(roi[0-9]+)({aliases})", re.IGNORECASE)
    while True:
        numbers = []
        for path in directory.iterdir():
            found = pattern.fullmatch(path.name)
            if found:
                numbers.append(int(found[1][3:]))
        number = max(numbers, default=0) + 1
        roi_id = f"roi{number:02d}"
        stem = f"{image_stem}__{roi_id}"
        lock = directory / f"{stem}.saving"
        try:
            lock.touch(exist_ok=False)
        except FileExistsError:
            continue
        # Another writer may have exported a group between the scan and reservation.
        if any((directory / (stem + suffix)).exists() for suffix in SUFFIXES):
            lock.unlink()
            continue
        break
    try:
        yield stem, roi_id
    finally:
        lock.unlink()
