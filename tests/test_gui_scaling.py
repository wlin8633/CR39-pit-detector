"""Exercise image and fit geometry through the GUI callbacks without Tk windows."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from cr39_detector.gui import logic as gui
from cr39_detector.state import load_state_bundle


class Variable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class GuiScalingTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(gui, "DETECTION_WORK_DIR", self.folder / "tiles").start()
        image = np.full((103, 105, 3), 185, dtype=np.uint8)
        path = self.folder / "raw.bmp"
        self.assertTrue(cv2.imwrite(str(path), image))
        settings, _ = load_state_bundle("example_state.cr39state")
        settings.update(
            path_var=str(path), ext_rate_var="1.5", dw_var="105", dh_var="103",
            sh_var="0", sw_var="0", bbox_mode="Fraction", x1_var="0", y1_var="0",
            x2_var="0.5", y2_var="0.5", use_cuda_var="False", verbose_var="False",
            bsw_var="2", bsh_var="2", obj_res_var="1.38e-7",
        )
        self.app = SimpleNamespace(
            **{name: Variable(value) for name, value in settings.items()},
            image_window=None, show_canvas_window=Mock(), extended_img=None,
            peaks=None, ovals=None, display_img=None,
        )
        self.logic = gui.AppLogic(self.app)
        self.logic.log = Mock()
        self.logic.display_results = Mock()

    def test_changed_rate_requires_reset_and_leaves_roi_untouched(self):
        self.logic.update_image()
        self.assertEqual(self.app.extended_img.shape[:2], (77, 79))
        self.logic.update_image()
        self.assertEqual(self.app.extended_img.shape[:2], (77, 79))
        self.assertTrue(self.logic.detect_pits())
        original_ovals = self.app.ovals
        self.app.ext_rate_var.set("1.2")
        for action in (
            self.logic.detect_pits, self.logic.update_image, self.logic.replace_detection,
            self.logic.show_current_results, self.logic.batch_detect,
        ):
            action()
            self.assertIn("Reset Image", self.logic.log.call_args.args[0])
        self.assertIsNone(self.logic.save_measurements(self.folder / "stale.csv"))
        self.assertFalse((self.folder / "stale.csv").exists())
        self.assertIs(self.app.ovals, original_ovals)
        self.assertEqual(self.app.x2_var.get(), "0.5")

        self.logic.reset_image()
        self.assertEqual(self.app.extended_img.shape[:2], (124, 126))
        self.assertEqual(self.app.x2_var.get(), "1")
        self.assertIsNone(self.app.ovals)
        self.app.x2_var.set("0.5")
        self.logic.update_image()
        self.assertTrue(self.logic.detect_pits())
        self.assertAlmostEqual(self.logic.display_results.call_args.args[3], 1.15e-7)
        csv = self.folder / "measurements.csv"
        self.assertEqual(self.logic.save_measurements(csv), csv)
        header_scale = csv.read_text().splitlines()[0].split("=")[1]
        self.assertAlmostEqual(float(header_scale), 1.15e-7, places=15)

    def test_replace_translates_absolute_roi_into_the_existing_result_frame(self):
        self.app.bbox_mode.set("Absolute")
        for name, value in zip(("x1_var", "y1_var", "x2_var", "y2_var"), (20, 10, 100, 90)):
            getattr(self.app, name).set(str(value))
        self.logic.update_image()
        original = np.array([[5, 5], [20, 20], [30, 30]])
        original_ovals = np.column_stack((original, np.ones((3, 4))))
        with patch.object(self.logic, "_run_detection", return_value=(original, original_ovals)):
            self.assertTrue(self.logic.detect_pits())
        result_image = self.logic.display_results.call_args.args[0]
        for name, value in zip(("x1_var", "y1_var", "x2_var", "y2_var"), (40, 30, 50, 40)):
            getattr(self.app, name).set(str(value))
        self.logic.update_image()
        replacement = np.array([[2, 3]])
        replacement_oval = np.column_stack((replacement, np.ones((1, 4))))
        with patch.object(self.logic, "_run_detection", return_value=(replacement, replacement_oval)):
            self.logic.replace_detection()
        np.testing.assert_array_equal(self.app.peaks, [[5, 5], [30, 30], [22, 23]])
        np.testing.assert_array_equal(self.app.ovals[:, :2], self.app.peaks)
        self.assertIs(self.logic.display_results.call_args.args[0], result_image)

    def test_fractional_batch_stitches_odd_sized_tiles_and_restores_results(self):
        self.logic.update_image()
        selected = self.app.extended_img
        self.logic.batch_detect()
        stitched, peaks, ovals, pixel_size = self.logic.display_results.call_args.args
        self.assertEqual(stitched.shape[:2], (154, 158))
        self.assertEqual(peaks.shape, (0, 2))
        self.assertEqual(ovals.shape, (0, 6))
        self.assertEqual(pixel_size, 1.38e-7 / 1.5)
        self.assertIs(self.app.extended_img, selected)
        self.assertEqual(self.app.x2_var.get(), "0.5")
        state = self.folder / "batch.cr39state"
        self.logic.save_parameters(state)
        restored = gui.AppLogic(self.app)
        restored.display_results = Mock()
        restored.load_parameters(state)
        restored.show_current_results()
        np.testing.assert_array_equal(restored.display_results.call_args.args[0], stitched)
        np.testing.assert_array_equal(self.app.extended_img, selected)


if __name__ == "__main__":
    unittest.main()
