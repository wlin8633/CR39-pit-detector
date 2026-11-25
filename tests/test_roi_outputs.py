"""Test ROI header loading and raw-lattice coordinate transformations."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cr39_detector.roi_outputs import read_saved_rois


class SavedRoiTests(unittest.TestCase):
    @staticmethod
    def write(folder, name, key, bbox, shape=(120, 150)):
        info = {"schema_version": 1, "source_image": key[0], "result_key": key,
                "result_bbox": bbox, "result_shape_hw": shape}
        (folder / name).write_text("# result_metadata=" + json.dumps(info) + "\n", encoding="utf-8")

    def test_different_rate_tile_and_clipping_use_original_pixel_edges(self):
        with TemporaryDirectory() as temp:
            base = Path(temp)
            raw = base / "frame.bmp"
            raw.touch()
            key = (str(raw), 100, 80, 1, 2, 1.5)
            self.write(base, "frame__roi02_measurements.csv", key, (15, 30, 75, 90))
            (base / "frame__roi03_state.cr39state").touch()
            other = base / "other"
            other.mkdir()
            (other / raw.name).touch()
            self.write(base, "frame__roi04_measurements.csv", (str(other / raw.name), *key[1:]), (15, 30, 75, 90))
            current = (str(raw), 208, 92, 1, 1, 2.0, "Linear")
            self.assertEqual(read_saved_rois(base, current, (80, 60, 3)), [
                ("roi02", (4.0, 16.0, 60, 80)),
            ])

    def test_whole_image_batch_ignores_selected_slice_origin(self):
        with TemporaryDirectory() as temp:
            base = Path(temp)
            raw = base / "frame.bmp"
            raw.touch()
            key = (str(raw), 100, 80, 2, 3, 2.0, "Cubic")
            self.write(base, "frame_measurements.csv", key, None, shape=(500, 900))
            active = (str(raw), 100, 80, 0, 0, 2.0, "Cubic")
            self.assertEqual(read_saved_rois(base, active, (160, 200, 3)), [
                ("Saved", (0, 0, 200, 160)),
            ])
