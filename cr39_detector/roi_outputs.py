"""Allocate a non-overwriting, per-image identifier for an ROI output group."""

from contextlib import contextmanager
import re


SUFFIXES = ("_state.cr39state", "_measurements.csv", "_results.png", "_results.bmp")


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
