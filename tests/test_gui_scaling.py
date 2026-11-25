"""Exercise image and fit geometry through the GUI callbacks without Tk windows."""

import json
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
from cr39_detector.state import load_state_bundle, save_state_bundle


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
            path_var=str(path), ext_rate_var="1.5", ext_interpolation_var="Cubic", dw_var="105", dh_var="103",
            sh_var="0", sw_var="0", bbox_mode="Fraction", x1_var="0", y1_var="0",
            x2_var="0.5", y2_var="0.5", use_cuda_var="False", verbose_var="False",
            bsw_var="2", bsh_var="2", obj_res_var="1.38e-7",
            save_dir_var="",
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

    def test_interpolation_change_requires_reset_and_survives_state_roundtrip(self):
        path = self.app.path_var.get()
        raw = np.full((103, 105, 3), 185, dtype=np.uint8)
        cv2.circle(raw, (47, 44), 6, (30, 30, 30), -1)
        self.assertTrue(cv2.imwrite(path, raw))
        self.app.ext_rate_var.set("2")
        self.logic.reset_image()
        size = (210, 206)
        cubic = cv2.resize(raw, size, interpolation=cv2.INTER_CUBIC)
        linear = cv2.resize(raw, size, interpolation=cv2.INTER_LINEAR)
        self.assertFalse(np.array_equal(cubic, linear))
        np.testing.assert_array_equal(self.logic._full_tile, cubic)
        self.assertTrue(self.logic.detect_pits())
        previous = self.app.ovals
        cubic_state = self.folder / "cubic.cr39state"
        self.logic.save_parameters(cubic_state)
        legacy_settings, _ = load_state_bundle(cubic_state)
        legacy_arrays = {
            "full_tile": self.logic._full_tile, "result_image": self.logic._result_image,
            "peaks": self.app.peaks, "ovals": self.app.ovals,
        }

        self.app.ext_interpolation_var.set("Linear")
        for action in (self.logic.update_image, self.logic.detect_pits, self.logic.replace_detection,
                       self.logic.batch_detect, self.logic.show_current_results):
            action()
            self.assertIn("Reset Image", self.logic.log.call_args.args[0])
        self.assertIs(self.app.ovals, previous)
        self.logic.reset_image()
        self.assertIsNone(self.app.ovals)
        np.testing.assert_array_equal(self.logic._full_tile, linear)
        self.assertTrue(self.logic.detect_pits())
        linear_state = self.folder / "linear.cr39state"
        self.logic.save_parameters(linear_state)

        self.app.ext_interpolation_var.set("Cubic")
        self.logic.load_parameters(linear_state)
        self.assertEqual(self.app.ext_interpolation_var.get(), "Linear")
        self.assertIsNone(self.logic._result_key)
        self.assertIsNone(self.logic._full_tile)
        self.assertEqual(load_state_bundle(linear_state)[1], {})
        self.logic.update_image()
        np.testing.assert_array_equal(self.logic._full_tile, linear)
        self.assertTrue(self.logic.detect_pits())
        self.assertEqual(self.logic._check_result_scale(), 1.38e-7 / 2)

        settings = legacy_settings
        settings.pop("settings_only")
        settings.pop("ext_interpolation_var")
        for key in ("image_key", "result_key"):
            settings[key] = settings[key][:6]
        old_state = self.folder / "old.cr39state"
        save_state_bundle(old_state, settings, legacy_arrays)
        self.logic.load_parameters(old_state)
        self.assertEqual(self.app.ext_interpolation_var.get(), "Cubic")
        np.testing.assert_array_equal(self.logic._full_tile, cubic)
        self.assertEqual(self.logic._check_result_scale(), 1.38e-7 / 2)

    def test_batch_uses_selected_interpolation_for_fractional_edge_tile(self):
        raw = np.full((103, 105, 3), 185, dtype=np.uint8)
        raw[59:69, 54:68] = 30
        self.assertTrue(cv2.imwrite(self.app.path_var.get(), raw))
        self.app.ext_interpolation_var.set("Linear")
        self.logic.update_image()
        self.assertTrue(self.logic.batch_detect())
        stitched = self.logic.display_results.call_args.args[0]
        # The bottom-right raw tile starts at x=ceil(105/2), y=ceil(103/2).
        result_x, result_y = round(53 * 1.5), round(52 * 1.5)
        tile = raw[52:, 53:]
        transform = np.array([[1.5, 0, 53 * 1.5 - result_x + 0.25], [0, 1.5, 0.25]])
        size = (stitched.shape[1] - result_x, stitched.shape[0] - result_y)
        expected = cv2.warpAffine(tile, transform, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        np.testing.assert_array_equal(stitched[result_y:, result_x:], expected)

    def test_fractional_batch_stitches_odd_sized_tiles_and_can_be_replayed(self):
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
        restored.log = Mock()
        restored.load_parameters(state)
        self.assertIsNone(self.app.extended_img)
        self.assertIsNone(self.app.ovals)
        self.assertEqual(load_state_bundle(state)[1], {})
        self.assertTrue(restored.batch_detect())
        np.testing.assert_array_equal(restored.display_results.call_args.args[0], stitched)

    def test_save_selected_separates_settings_measurements_and_display(self):
        self.logic.update_image()
        saved_bbox = self.logic._selected_bbox
        self.assertTrue(self.logic.detect_pits())
        self.app.display_img = self.app.extended_img.copy()
        # A UI edit after detection must not relabel the statistics in the CSV.
        original_rms = self.app.eps_rms_var.get()
        self.app.eps_rms_var.set("999")
        out = self.folder / "out"
        out.mkdir()
        self.app.save_dir_var.set(str(out))
        with patch.object(gui.filedialog, "askdirectory") as picker:
            self.logic.save_selected(status=True, measurements=True, display=True, display_extension=".bmp")
        picker.assert_not_called()
        self.assertEqual({p.name for p in out.iterdir()},
                         {"raw_state.cr39state", "raw_measurements.csv", "raw_results.bmp"})
        settings, arrays = load_state_bundle(out / "raw_state.cr39state")
        self.assertEqual(arrays, {})
        self.assertEqual(settings["result_bbox"], list(saved_bbox))
        self.assertEqual(settings["eps_rms_var"], "999")
        metadata_line = next(line for line in (out / "raw_measurements.csv").read_text().splitlines()
                             if line.startswith("# result_metadata="))
        provenance = json.loads(metadata_line.split("=", 1)[1])
        self.assertEqual(provenance["result_bbox"], list(saved_bbox))
        self.assertEqual(provenance["settings"]["eps_rms_var"], original_rms)
        self.assertEqual(self.logic._recorded_rois, [("Saved", saved_bbox)])

    def test_new_roi_uses_maximum_and_groups_output_types(self):
        self.logic.update_image()
        self.assertTrue(self.logic.detect_pits())
        self.app.display_img = self.app.extended_img.copy()
        out = self.folder / "out"
        out.mkdir()
        self.app.save_dir_var.set(str(out))
        for name in ("raw__roi01_state.cr39state", "raw__roi03_results.bmp", "other__roi99_measurements.csv"):
            (out / name).touch()
        with patch.object(gui.messagebox, "askyesno") as confirm:
            self.logic.save_selected(status=True, measurements=True, display=True, new_roi=True)
            self.logic.save_selected(status=False, measurements=True, display=False, new_roi=True)
        confirm.assert_not_called()
        for suffix in ("_state.cr39state", "_measurements.csv", "_results.png"):
            self.assertTrue((out / ("raw__roi04" + suffix)).is_file())
        self.assertTrue((out / "raw__roi05_measurements.csv").is_file())
        settings, _ = load_state_bundle(out / "raw__roi04_state.cr39state")
        self.assertEqual(settings["roi_id"], "roi04")
        csv = (out / "raw__roi04_measurements.csv").read_text()
        provenance = json.loads(next(s.split("=", 1)[1] for s in csv.splitlines()
                                     if s.startswith("# result_metadata=")))
        self.assertEqual(provenance["roi_id"], settings["roi_id"])
        self.assertEqual(Path(provenance["source_image"]).name, "raw.bmp")
        self.assertFalse(list(out.glob("*.saving")))

    def test_save_dir_follows_image_when_blank_and_is_preserved_as_gui_preference(self):
        store = self.folder / "autosave.cr39state"
        self.assertEqual(self.logic._save_directory(Path(self.app.path_var.get())), self.folder)
        out = self.folder / "out"
        out.mkdir()
        self.app.save_dir_var.set("out")
        self.assertEqual(self.logic._save_directory(Path(self.app.path_var.get())), out)
        self.logic.auto_save(store)
        manual = self.logic.save_parameters(self.folder / "manual.cr39state")
        self.assertNotIn("save_dir_var", load_state_bundle(manual)[0])
        self.app.save_dir_var.set("")
        self.logic.load_parameters(manual)
        self.assertEqual(self.app.save_dir_var.get(), "")
        self.logic.auto_load(store)
        self.assertEqual(self.app.save_dir_var.get(), str(out))

    def test_saved_measurement_boxes_are_reloaded_after_extending_at_a_new_rate(self):
        self.logic.update_image()
        box = self.logic._selected_bbox
        self.assertTrue(self.logic.detect_pits())
        self.logic.save_selected(status=False, measurements=True, display=False, new_roi=True)
        self.assertEqual(self.logic._recorded_rois, [("roi01", box)])
        self.app.ext_rate_var.set("3")
        self.logic.reset_image()
        self.assertEqual(self.logic._recorded_rois, [("roi01", tuple(v * 2 for v in box))])
        self.assertIsNone(self.app.ovals)

    def test_remove_detect_keeps_rows_and_replace_respects_area_masks(self):
        self.app.bbox_mode.set("Absolute")
        variables = (self.app.x1_var, self.app.y1_var, self.app.x2_var, self.app.y2_var)
        for var, value in zip(variables, (20, 10, 100, 90)):
            var.set(str(value))
        self.logic.update_image()
        peaks = np.array([[5, 5], [20, 20], [30, 30], [31, 31]])
        ovals = np.column_stack((peaks, np.ones((4, 3)), [0, 0, 0, -3]))
        with patch.object(self.logic, "_run_detection", return_value=(peaks, ovals)):
            self.assertTrue(self.logic.detect_pits())

        for var, value in zip(variables, (40, 30, 50, 40)):
            var.set(str(value))
        with patch.object(self.logic, "_run_detection") as detect:
            self.assertEqual(self.logic.remove_detection(exclude_area=False), 1)
        detect.assert_not_called()
        np.testing.assert_array_equal(self.logic._manual_excluded, [0, 1, 0, 0])
        np.testing.assert_array_equal(self.logic.display_results.call_args.args[1], peaks[[0, 2, 3]])

        for var, value in zip(variables, (50, 40, 55, 45)):
            var.set(str(value))
        self.assertEqual(self.logic.remove_detection(exclude_area=True), 2)
        csv = self.logic.save_measurements(self.folder / "manual.csv")
        data = np.loadtxt(csv, delimiter=",", ndmin=2)
        np.testing.assert_array_equal(data[:, 7], [0, 0, 0, -3])
        np.testing.assert_array_equal(data[:, 8], [0, 1, 1, 1])
        self.assertEqual(len(self.app.ovals), 4)
        meta = json.loads(next(s.split("=", 1)[1] for s in csv.read_text().splitlines()
                               if s.startswith("# result_metadata=")))
        self.assertEqual(meta["manual_exclusion_regions"], [
            {"bbox": [40, 30, 50, 40], "exclude_area": False},
            {"bbox": [50, 40, 55, 45], "exclude_area": True},
        ])

        new_peaks = np.array([[2, 2]])
        new_ovals = np.column_stack((new_peaks, np.ones((1, 4))))
        with patch.object(self.logic, "_run_detection", return_value=(new_peaks, new_ovals)):
            self.assertTrue(self.logic.replace_detection())
        np.testing.assert_array_equal(self.logic._manual_excluded, [0, 1, 1])

    def test_compact_state_rebuilds_exact_saved_roi_after_moving_source(self):
        self.logic.update_image()
        selected = self.app.extended_img.copy()
        saved_bbox = self.logic._selected_bbox
        # Edited but not applied before saving.
        self.app.x2_var.set("0.8")
        state = self.folder / "raw_state.cr39state"
        self.logic.save_parameters(state)
        moved = self.folder / "moved"
        moved.mkdir()
        state = state.rename(moved / state.name)
        (self.folder / "raw.bmp").rename(moved / "raw.bmp")
        self.logic.load_parameters(state)
        self.assertIsNone(self.app.extended_img)
        self.logic.update_image()
        self.assertEqual(self.logic._selected_bbox, saved_bbox)
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
