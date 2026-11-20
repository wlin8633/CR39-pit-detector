"""Exercise image and fit geometry through the GUI callbacks without Tk windows."""

from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Event, get_ident
from time import monotonic, sleep
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from cr39_detector.gui import logic as gui
from cr39_detector import PitDetector
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
        self.callbacks = Queue()
        self.app.master = SimpleNamespace(after=lambda delay, fn: self.callbacks.put(fn))
        self.app.set_job_controls = Mock()

    def finish_background_job(self):
        deadline = monotonic() + 8
        while monotonic() < deadline:
            self.callbacks.get(timeout=5)()
            if not self.logic.busy:
                self.app.set_job_controls.assert_called_with(False)
                return
            sleep(0.01)
        self.fail("The background job did not finish")

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

    def test_stop_during_ellipse_fit_preserves_previous_result_and_allows_retry(self):
        self.logic.update_image()
        self.assertTrue(self.logic.detect_pits())
        previous_ovals = self.app.ovals
        self.app.display_img = self.app.extended_img.copy()
        previous_display = self.app.display_img
        y, x = np.indices((128, 128))
        pit = np.where((x - 64)**2 + (y - 64)**2 < 8**2, 25, 185).astype(np.uint8)
        self.app.extended_img = np.repeat(pit[..., None], 3, axis=2)
        self.app.peak_edge_std_var.set("1")
        fitting = Event()
        main_thread = get_ident()

        def pause_in_fit(detector, *args):
            self.assertNotEqual(get_ident(), main_thread)
            fitting.set()
            self.assertTrue(detector.cancel_event.wait(5))
            gui.check_cancelled(detector.cancel_event)

        with patch.object(PitDetector, "_fit_ovals", pause_in_fit):
            self.assertTrue(self.logic.detect_pits(background=True))
            try:
                self.assertTrue(fitting.wait(5), "The detector never reached ellipse fitting")
                self.assertTrue(self.logic.busy)
                self.app.set_job_controls.assert_called_with(True)
                self.assertIsNone(self.logic.detect_pits(background=True))
            finally:
                self.logic.stop()
                self.finish_background_job()
        self.assertIn("Detect Pits stopped", self.logic.log.call_args.args[0])
        self.assertIs(self.app.ovals, previous_ovals)
        self.assertIs(self.app.display_img, previous_display)

        self.assertTrue(self.logic.detect_pits(background=True))
        self.finish_background_job()
        self.assertIsNot(self.app.ovals, previous_ovals)
        self.assertGreater(len(self.app.ovals), 0)
        background_peaks, background_ovals = self.app.peaks, self.app.ovals
        self.assertTrue(self.logic.detect_pits())
        np.testing.assert_array_equal(self.app.peaks, background_peaks)
        np.testing.assert_allclose(self.app.ovals, background_ovals)

    def test_cancel_replace_and_reset_discard_completed_worker_values(self):
        self.logic.update_image()
        self.assertTrue(self.logic.detect_pits())
        previous_ovals = self.app.ovals
        previous_tile = self.logic._full_tile
        previous_roi = self.app.extended_img
        for action, method, returned in (
            (self.logic.replace_detection, "_run_detection", (np.empty((0, 2)), np.empty((0, 6)))),
            (self.logic.reset_image, "_prepare_tile", (np.empty((8, 8, 3), np.uint8), 1, 1)),
        ):
            entered = Event()

            def complete_after_stop(*args):
                cancel_event = args[2] if method == "_run_detection" else args[1]
                entered.set()
                self.assertTrue(cancel_event.wait(5))
                return returned

            with patch.object(self.logic, method, side_effect=complete_after_stop):
                self.assertTrue(action(background=True))
                try:
                    self.assertTrue(entered.wait(5))
                finally:
                    self.logic.stop()
                    self.finish_background_job()
            self.assertIs(self.app.ovals, previous_ovals)
            self.assertIs(self.logic._full_tile, previous_tile)
            self.assertIs(self.app.extended_img, previous_roi)
            self.assertEqual(self.app.x2_var.get(), "0.5")


if __name__ == "__main__":
    unittest.main()
