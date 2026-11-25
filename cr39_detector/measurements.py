"""Explicit conversions from internal ellipse fits to physical measurements.

The detector stores ellipse semiaxes in pixels because that is the convention
used by the quadratic ellipse equation, OpenCV drawing, radius filtering, and
RMS normalization.  Scientific outputs use full diameters in metres.
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np

from .diagnostics import to_numpy


INTERNAL_OVAL_AXIS_CONVENTION = "semiaxes_px"
PHYSICAL_AXIS_CONVENTION = "full_diameters_m"


def fitted_oval_diameters(fitted_ovals, pixel_size_m, accepted_only=False):
    """Return physical full diameters for fitted ovals.

    Parameters
    ----------
    fitted_ovals:
        ``(N, 6)`` detector array with columns
        ``cx, cy, semi_major_px, semi_minor_px, azimuth_rad, status``.
    pixel_size_m:
        Physical size of one pixel in the image on which fitting was done.
    accepted_only:
        If true, return only rows whose status is zero.

    Returns
    -------
    numpy.ndarray
        ``(N, 2)`` array containing full major and minor diameters in metres.
    """
    ovals = to_numpy(fitted_ovals)
    if ovals.ndim != 2 or ovals.shape[1] != 6:
        raise ValueError("fitted_ovals must have shape (N, 6)")
    pixel_size_m = float(pixel_size_m)
    if not np.isfinite(pixel_size_m) or pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be finite and positive")
    if accepted_only:
        ovals = ovals[ovals[:, 5] == 0]
    return 2.0 * np.asarray(ovals[:, 2:4], dtype=np.float64) * pixel_size_m


def fitted_ovals_to_measurements(fitted_ovals, pixel_size_m, accepted_only=False):
    """Create an auditable physical-measurement table.

    The returned structured array retains the internal semiaxes and adds the
    full physical diameters used by calibration and inventory workflows.
    """
    ovals = to_numpy(fitted_ovals)
    if ovals.ndim != 2 or ovals.shape[1] != 6:
        raise ValueError("fitted_ovals must have shape (N, 6)")
    if accepted_only:
        ovals = ovals[ovals[:, 5] == 0]
    diameters = fitted_oval_diameters(ovals, pixel_size_m)
    table = np.empty(
        len(ovals),
        dtype=[
            ("center_x_px", "f8"),
            ("center_y_px", "f8"),
            ("semi_major_px", "f8"),
            ("semi_minor_px", "f8"),
            ("major_diameter_m", "f8"),
            ("minor_diameter_m", "f8"),
            ("azimuth_rad", "f8"),
            ("status", "i4"),
        ],
    )
    table["center_x_px"] = ovals[:, 0]
    table["center_y_px"] = ovals[:, 1]
    table["semi_major_px"] = ovals[:, 2]
    table["semi_minor_px"] = ovals[:, 3]
    table["major_diameter_m"] = diameters[:, 0]
    table["minor_diameter_m"] = diameters[:, 1]
    table["azimuth_rad"] = ovals[:, 4]
    table["status"] = ovals[:, 5].astype(np.int32)
    return table


def save_measurements_csv(path, fitted_ovals, pixel_size_m, accepted_only=False, *, metadata=None,
                          manual_excluded=None):
    """Atomically save detector measurements with units in every column name."""
    if manual_excluded is not None:
        fitted_ovals = to_numpy(fitted_ovals)
        manual_excluded = np.asarray(manual_excluded, dtype=bool)
        if manual_excluded.shape != (len(fitted_ovals),):
            raise ValueError("Manual exclusion flags must match the number of detector rows")
        if accepted_only:
            keep = (fitted_ovals[:, 5] == 0) & ~manual_excluded
            fitted_ovals, manual_excluded = fitted_ovals[keep], manual_excluded[keep]
            accepted_only = False
    table = fitted_ovals_to_measurements(
        fitted_ovals,
        pixel_size_m,
        accepted_only=accepted_only,
    )
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    columns = np.column_stack(
        [
            table["center_x_px"],
            table["center_y_px"],
            table["semi_major_px"],
            table["semi_minor_px"],
            table["major_diameter_m"],
            table["minor_diameter_m"],
            table["azimuth_rad"],
            table["status"],
        ]
    )
    header_items = [
            f"pixel_size_m={float(pixel_size_m):.17g}",
            f"internal_axis_convention={INTERNAL_OVAL_AXIS_CONVENTION}",
            f"physical_axis_convention={PHYSICAL_AXIS_CONVENTION}",
    ]
    if metadata is not None:
        header_items.append("result_metadata=" + json.dumps(metadata, ensure_ascii=True, allow_nan=False))
    labels = ("center_x_px,center_y_px,semi_major_px,semi_minor_px,"
              "major_diameter_m,minor_diameter_m,azimuth_rad,status")
    fmt = ["%.17g"] * 7 + ["%d"]
    if manual_excluded is not None:
        labels += ",manual_excluded"
        columns = np.column_stack((columns, manual_excluded))
        fmt.append("%d")
    header_items.append(labels)
    try:
        np.savetxt(
            temporary,
            columns,
            delimiter=",",
            header="\n".join(header_items),
            comments="# ",
            fmt=fmt,
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


__all__ = [
    "INTERNAL_OVAL_AXIS_CONVENTION",
    "PHYSICAL_AXIS_CONVENTION",
    "fitted_oval_diameters",
    "fitted_ovals_to_measurements",
    "save_measurements_csv",
]
