"""Minimal CPU example using the bundled microscope image."""

from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cr39_detector import DetectionConfig, PitDetector, summarize_pit_statuses


image_path = (PROJECT_ROOT / "example.bmp").resolve()
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)

# A small central crop keeps this example quick. Adjust it for your image.
height, width = image.shape[:2]
crop = image[height // 4 : height // 2, width // 4 : width // 2]

config = DetectionConfig(
    pixel_size=0.138e-6,
    radius_range=(0.2e-6, 2.0e-6),
    mode="ellipse",
    eps_product_thre_offset=7,
    eps_RMS_thre=8,
    mfp_detection=False,
    verbose=False,
    use_cuda=False,
)
detector = PitDetector(**config.as_detector_kwargs())
peaks, fitted_ovals = detector.detect(crop)

print(f"Image: {image_path}")
print(f"Peaks: {len(peaks)}")
print(f"Status counts: {summarize_pit_statuses(fitted_ovals)}")
