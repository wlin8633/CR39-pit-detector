"""Stable pit-status codes and helpers shared by the detector and GUI."""

import numpy as np


# The sixth column of ``fitted_ovals`` is a public, backward-compatible code.
PIT_STATUS_INFO = {
    0: {"name": "valid", "short": "OK", "description": "Ellipse/circle fit accepted.", "bgr": (255, 0, 0)},
    -1: {"name": "fit_failed", "short": "FIT", "description": "Insufficient contour points or no valid ellipse solution.", "bgr": (255, 255, 0)},
    -2: {"name": "density_rejected", "short": "DENS", "description": "Rejected by the optional mean-free-path local-density filter.", "bgr": (255, 0, 255)},
    -3: {"name": "rms_rejected", "short": "RMS", "description": "Normalized ellipse-fit RMS error exceeds eps_RMS_thre.", "bgr": (0, 255, 0)},
    -4: {"name": "size_rejected", "short": "SIZE", "description": "Fitted major or minor radius is outside radius_min/radius_max.", "bgr": (0, 0, 255)},
}


def to_numpy(array):
    """Return a NumPy view/copy for either NumPy or CuPy input."""
    return array.get() if hasattr(array, "get") else np.asarray(array)


def summarize_pit_statuses(fitted_ovals):
    """Count fitted-pit status codes without changing the six-column schema."""
    fitted_ovals = to_numpy(fitted_ovals)
    if fitted_ovals.size == 0:
        return {}
    labels, counts = np.unique(fitted_ovals[:, 5].astype(int), return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def format_pit_status_summary(counts):
    """Create a compact, human-readable status summary."""
    parts = []
    for code in (0, -1, -2, -3, -4):
        count = counts.get(code, 0)
        if count:
            parts.append(f"{PIT_STATUS_INFO[code]['short']}={count}")
    unknown = sum(count for code, count in counts.items() if code not in PIT_STATUS_INFO)
    if unknown:
        parts.append(f"UNKNOWN={unknown}")
    return ", ".join(parts) if parts else "no fitted pits"
