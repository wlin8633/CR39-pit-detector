import pickle
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etchingModel import EtchingTrackRatioFitter
from pitDetection import PitDetector, summarize_pit_statuses


class PitDetectionRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "example_state.sav").open("rb") as stream:
            cls.state = pickle.load(stream)

    def test_cpu_example_baseline(self):
        state = self.state
        extended_rate = float(state["ext_rate_var"])
        detector = PitDetector(
            pixel_size=float(state["obj_res_var"]) / extended_rate,
            radius_range=[float(state["radius_min_var"]), float(state["radius_max_var"])],
            mode=state["mode_var"],
            bg_value=None if state["bg_value_var"] == "None" else int(state["bg_value_var"]),
            clips_to_bg=state["clips_to_bg_var"] == "True",
            peak_pad_rate=float(state["peak_pad_rate_var"]),
            peak_charac_len=float(state["peak_charac_len_var"]),
            peak_thre_offset=float(state["peak_thre_offset_var"]),
            peak_gaussian_sigma=float(state["peak_sigma_var"]),
            peak_edge_std=float(state["peak_edge_std_var"]),
            peak_closest_dist=float(state["peak_dist_var"]),
            eps_charac_len=float(state["eps_charac_len_var"]),
            eps_product_thre_offset=7.0,
            eps_allowed_center_offest=float(state["eps_center_var"]),
            eps_RMS_thre=8.0,
            cir_radii_thre_percent=float(state["cir_radii_thre_var"]),
            mfp_detection=state["mfp_detection_var"] == "True",
            mfp=None if state["mfp_var"] == "None" else int(state["mfp_var"]),
            mfp_std=float(state["mfp_std_var"]),
            timeout=float(state["timeout_var"]),
            verbose=False,
            use_cuda=False,
        )
        image = state["extended_img"]
        peaks, fitted_ovals = detector.detect(image, bbox=[0, 0, image.shape[1], image.shape[0]])

        self.assertEqual(peaks.shape, (164, 2))
        self.assertEqual(fitted_ovals.shape, (164, 6))
        self.assertEqual(summarize_pit_statuses(fitted_ovals), {-3: 15, -1: 26, 0: 123})
        accepted = fitted_ovals[:, 5] == 0
        self.assertTrue(np.isrealobj(fitted_ovals))
        self.assertTrue(np.all(np.isfinite(fitted_ovals[accepted, :5])))

    def test_detector_quality_defaults(self):
        detector = PitDetector(
            pixel_size=0.069e-6,
            radius_range=[0.2e-6, 2e-6],
            mode="ellipse",
            verbose=False,
            use_cuda=False,
        )
        self.assertEqual(detector.eps_product_thre_offset, 7)
        self.assertEqual(detector.eps_RMS_thre, 8)

    def test_energy_angle_assets_remain_loadable(self):
        asset_dir = ROOT / "savFiles_etchingModel"
        with np.load(asset_dir / "etching_rate_ratio_fit_results.npz", allow_pickle=False) as data:
            required = {"alpha_fit", "beta_fit", "theta_fit", "E_fit", "a_fit", "b_fit", "E_mesh", "theta_mesh", "a_mesh", "b_mesh"}
            self.assertTrue(required.issubset(data.files))
            self.assertEqual(data["E_mesh"].shape, data["theta_mesh"].shape)
            self.assertEqual(data["a_mesh"].shape, data["b_mesh"].shape)

            fitter = EtchingTrackRatioFitter()
            fitter.provide_results({key: data[key] for key in data.files})
            lookup = fitter.inverse_lookup(a_target=[0.8], b_target=[0.7], error_len=0.2, refine_energy=50, refine_theta=50)
            self.assertEqual(lookup.ndim, 2)
            self.assertEqual(lookup.shape[1], 4)

    def test_blank_image_returns_empty_results(self):
        image = np.full((128, 128, 3), 200, dtype=np.uint8)
        for mode in ("ellipse", "circle"):
            with self.subTest(mode=mode):
                detector = PitDetector(
                    pixel_size=0.069e-6,
                    radius_range=[0.2e-6, 2e-6],
                    mode=mode,
                    mfp_detection=False,
                    verbose=False,
                    use_cuda=False,
                )
                peaks, fitted_ovals = detector.detect(image)
                self.assertEqual(peaks.shape, (0, 2))
                self.assertEqual(fitted_ovals.shape, (0, 6))
                self.assertEqual(detector.last_diagnostics["status_counts"], {})


if __name__ == "__main__":
    unittest.main()
