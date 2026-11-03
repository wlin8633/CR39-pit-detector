import tempfile
import unittest
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cr39_detector.config import DetectionConfig
from cr39_detector.paths import absolute_path
from cr39_detector.state import load_state_bundle, save_state_bundle


class StateAndConfigTests(unittest.TestCase):
    def test_state_round_trip_uses_json_and_safe_npz(self):
        settings = {"path_var": str((ROOT / "example.bmp").resolve()), "eps_rms_var": "8"}
        arrays = {"peaks": np.array([[1, 2], [3, 4]], dtype=np.int32)}
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "test.cr39state"
            saved_path = save_state_bundle(state_path, settings, arrays)
            self.assertTrue(saved_path.is_absolute())
            with ZipFile(saved_path) as archive:
                self.assertEqual(set(archive.namelist()), {"state.json", "arrays.npz"})

            loaded_settings, loaded_arrays = load_state_bundle(saved_path)
            self.assertEqual(loaded_settings, settings)
            np.testing.assert_array_equal(loaded_arrays["peaks"], arrays["peaks"])

    def test_relative_paths_resolve_from_project_root(self):
        self.assertEqual(absolute_path("example.bmp"), (ROOT / "example.bmp").resolve())

    def test_detection_config_maps_legacy_keyword(self):
        config = DetectionConfig(
            pixel_size=0.069e-6,
            radius_range=(0.2e-6, 2e-6),
            mode="ellipse",
        )
        kwargs = config.as_detector_kwargs()
        self.assertEqual(kwargs["eps_product_thre_offset"], 7)
        self.assertEqual(kwargs["eps_RMS_thre"], 8)
        self.assertIn("eps_allowed_center_offest", kwargs)

    def test_invalid_radius_range_is_rejected(self):
        config = DetectionConfig(
            pixel_size=0.069e-6,
            radius_range=(2e-6, 0.2e-6),
            mode="ellipse",
        )
        with self.assertRaises(ValueError):
            config.validate()


if __name__ == "__main__":
    unittest.main()
