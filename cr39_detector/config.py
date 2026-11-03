"""Typed configuration for pit detection."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class DetectionConfig:
    """Validated detector inputs independent of Tkinter widgets."""

    pixel_size: float
    radius_range: Tuple[float, float]
    mode: Optional[str]
    bg_value: Optional[int] = None
    clips_to_bg: bool = True
    peak_pad_rate: float = 1.1
    peak_charac_len: float = 2e-6
    peak_thre_offset: float = 0.0
    peak_gaussian_sigma: float = 3.0
    peak_edge_std: float = 2.0
    peak_closest_dist: float = 2e-6
    eps_charac_len: float = 2e-6
    eps_product_thre_offset: float = 7.0
    eps_allowed_center_offset: float = 0.2e-6
    eps_RMS_thre: Optional[float] = 8.0
    cir_radii_thre_percent: float = 95.0
    mfp_detection: bool = False
    mfp: Optional[float] = None
    mfp_std: float = 7.0
    timeout: float = 10.0
    verbose: bool = True
    use_cuda: bool = True

    def validate(self) -> "DetectionConfig":
        minimum, maximum = self.radius_range
        if self.pixel_size <= 0:
            raise ValueError("objective_resolution / extended_rate must be positive")
        if minimum <= 0 or maximum <= minimum:
            raise ValueError("radius_range must satisfy 0 < radius_min < radius_max")
        if self.mode not in ("ellipse", "circle", None):
            raise ValueError("mode must be 'ellipse', 'circle', or None")
        if self.peak_gaussian_sigma < 0:
            raise ValueError("peak_gaussian_sigma cannot be negative")
        if self.eps_RMS_thre is not None and self.eps_RMS_thre < 0:
            raise ValueError("eps_RMS_thre cannot be negative")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        return self

    def as_detector_kwargs(self) -> Dict[str, Any]:
        """Return arguments accepted by the existing ``PitDetector`` API."""
        self.validate()
        return {
            "pixel_size": self.pixel_size,
            "radius_range": list(self.radius_range),
            "mode": self.mode,
            "bg_value": self.bg_value,
            "clips_to_bg": self.clips_to_bg,
            "peak_pad_rate": self.peak_pad_rate,
            "peak_charac_len": self.peak_charac_len,
            "peak_thre_offset": self.peak_thre_offset,
            "peak_gaussian_sigma": self.peak_gaussian_sigma,
            "peak_edge_std": self.peak_edge_std,
            "peak_closest_dist": self.peak_closest_dist,
            "eps_charac_len": self.eps_charac_len,
            "eps_product_thre_offset": self.eps_product_thre_offset,
            # Keep the original misspelled keyword for API compatibility.
            "eps_allowed_center_offest": self.eps_allowed_center_offset,
            "eps_RMS_thre": self.eps_RMS_thre,
            "cir_radii_thre_percent": self.cir_radii_thre_percent,
            "mfp_detection": self.mfp_detection,
            "mfp": self.mfp,
            "mfp_std": self.mfp_std,
            "timeout": self.timeout,
            "verbose": self.verbose,
            "use_cuda": self.use_cuda,
        }
