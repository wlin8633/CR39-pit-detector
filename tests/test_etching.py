"""Fast, repository-safe tests for CR-39 etching-model conventions."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cr39_detector.detection import PitDetector
from cr39_detector.etching import EtchingTrackRatioFitter, fit_quality_metrics


ASSET_DIR = ROOT / "savFiles_etchingModel"
REMOVED_DEPTH_UM = 1.68 * 0.5
COMPUTATIONAL_ALPHA_MEV_PER_CM = 1.0072576602240627e-4
COMPUTATIONAL_BETA = 1.501356499587407


def load_transport_data():
    with np.load(ASSET_DIR / "REL_x_E_ω0350.npz", allow_pickle=False) as data:
        rel_data = tuple(data[key].copy() for key in ("REL_x_E", "x", "E"))
    with np.load(ASSET_DIR / "R0_E_ω0350.npz", allow_pickle=False) as data:
        range_data = tuple(data[key].copy() for key in ("R0_E", "E"))
    return rel_data, range_data


def make_fitter(rel_unit="MeV/cm"):
    rel_data, range_data = load_transport_data()
    exp_data = (
        np.array([1.0]),
        np.array([1.0]),
        np.array([1.0]),
        REMOVED_DEPTH_UM,
    )
    return EtchingTrackRatioFitter(
        REL_data=rel_data,
        R0_data=range_data,
        exp_data=exp_data,
        rel_unit=rel_unit,
    )


class StoredCalibrationTests(unittest.TestCase):
    def test_to2mev_bundle_matches_reproduced_computational_baseline(self):
        with np.load(
            ASSET_DIR / "to2MeV-etching_rate_ratio_fit_results.npz",
            allow_pickle=False,
        ) as stored:
            self.assertAlmostEqual(
                float(stored["alpha_fit"]),
                COMPUTATIONAL_ALPHA_MEV_PER_CM,
                places=14,
            )
            self.assertAlmostEqual(
                float(stored["beta_fit"]),
                COMPUTATIONAL_BETA,
                delta=2.0e-10,
            )

    def test_rel_conversion_is_explicit(self):
        rel_data, _ = load_transport_data()
        fitter_cm = make_fitter("MeV/cm")
        fitter_mm = make_fitter("MeV/mm")
        np.testing.assert_allclose(fitter_cm.REL_x_E, rel_data[0] / 100.0)
        np.testing.assert_allclose(fitter_mm.REL_x_E, rel_data[0] / 1000.0)

    def test_save_results_includes_covariance_and_metadata(self):
        fitter = EtchingTrackRatioFitter()
        fitter.provide_results(
            {
                "alpha_fit": np.array(COMPUTATIONAL_ALPHA_MEV_PER_CM),
                "parameter_covariance": np.eye(3),
            }
        )
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "bundle.npz"
            fitter.save_results(
                output,
                metadata={
                    "REL_unit": "MeV/cm",
                    "bulk_etch_rate_um_per_h": 1.68,
                    "etch_time_h": 0.5,
                },
                verbose=False,
            )
            with np.load(output, allow_pickle=False) as saved:
                self.assertEqual(str(saved["REL_unit"]), "MeV/cm")
                np.testing.assert_array_equal(saved["parameter_covariance"], np.eye(3))
                self.assertEqual(float(saved["bulk_etch_rate_um_per_h"]), 1.68)

    def test_fit_quality_metrics_keep_major_and_minor_residuals_separate(self):
        metrics = fit_quality_metrics(
            observed_a=np.array([1.0, 2.0, 3.0]),
            observed_b=np.array([2.0, 3.0, 4.0]),
            predicted_a=np.array([1.0, 2.0, 3.0]),
            predicted_b=np.array([1.0, 2.0, 3.0]),
        )
        self.assertEqual(metrics["a_rmse_um"], 0.0)
        self.assertEqual(metrics["a_r2"], 1.0)
        self.assertEqual(metrics["b_rmse_um"], 1.0)
        self.assertEqual(metrics["b_bias_um"], -1.0)
        self.assertEqual(metrics["fit_finite_fraction"], 1.0)


class GeometryConventionTests(unittest.TestCase):
    def test_forward_model_outputs_full_diameters(self):
        fitter = make_fitter()
        result = fitter.compute_a_b(
            alpha=COMPUTATIONAL_ALPHA_MEV_PER_CM,
            beta=COMPUTATIONAL_BETA,
            theta=np.pi / 2,
            Em=1.0,
            hm=np.array([REMOVED_DEPTH_UM]),
            output_verbose=True,
        )

        y1 = float(np.asarray(result["y1"]))
        y2 = float(np.asarray(result["y2"]))
        a = float(np.asarray(result["a"]))
        b = float(np.asarray(result["b"]))
        self.assertAlmostEqual(y1, y2, places=8)
        self.assertAlmostEqual(a, 2.0 * y1, places=8)
        self.assertAlmostEqual(b, 2.0 * y1, places=8)

    def test_detector_ellipse_fit_outputs_semiaxes_and_image_azimuth(self):
        detector = PitDetector(
            pixel_size=1.0,
            radius_range=[0.1, 20.0],
            mode="ellipse",
            eps_allowed_center_offest=100.0,
            eps_RMS_thre=None,
            verbose=False,
            use_cuda=False,
        )
        center = np.array([20.0, 30.0])
        semimajor = 4.0
        semiminor = 2.0
        azimuth = 0.35
        phase = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
        x0 = semimajor * np.cos(phase)
        y0 = semiminor * np.sin(phase)
        x = center[0] + x0 * np.cos(azimuth) - y0 * np.sin(azimuth)
        y = center[1] + x0 * np.sin(azimuth) + y0 * np.cos(azimuth)

        xx = x.reshape(1, 1, -1)
        yy = y.reshape(1, 1, -1)
        mask = np.ones_like(xx, dtype=bool)
        fitted = np.asarray(
            detector._fit_ovals(center.reshape(1, 2), mask, xx, yy)
        )[0]

        self.assertAlmostEqual(fitted[2], semimajor, places=6)
        self.assertAlmostEqual(fitted[3], semiminor, places=6)
        angle_error = np.arctan2(
            np.sin(2.0 * (fitted[4] - azimuth)),
            np.cos(2.0 * (fitted[4] - azimuth)),
        ) / 2.0
        self.assertAlmostEqual(angle_error, 0.0, places=6)

    def test_forward_theta_is_measured_from_detector_surface(self):
        fitter = make_fitter()
        normal_a, normal_b = fitter.compute_a_b(
            COMPUTATIONAL_ALPHA_MEV_PER_CM,
            COMPUTATIONAL_BETA,
            np.pi / 2,
            1.0,
            np.array([REMOVED_DEPTH_UM]),
        )
        oblique_a, oblique_b = fitter.compute_a_b(
            COMPUTATIONAL_ALPHA_MEV_PER_CM,
            COMPUTATIONAL_BETA,
            np.deg2rad(70.0),
            1.0,
            np.array([REMOVED_DEPTH_UM]),
        )

        self.assertAlmostEqual(
            float(np.asarray(normal_a)), float(np.asarray(normal_b)), places=8
        )
        self.assertGreater(float(np.asarray(oblique_a)), float(np.asarray(oblique_b)))


class InverseLookupTests(unittest.TestCase):
    def test_stored_mesh_round_trip_returns_known_energy_angle(self):
        with np.load(
            ASSET_DIR / "etching_rate_ratio_fit_results.npz",
            allow_pickle=False,
        ) as stored:
            results = {key: stored[key].copy() for key in stored.files}

        finite = np.isfinite(results["a_mesh"]) & np.isfinite(results["b_mesh"])
        candidate_indices = np.argwhere(finite)
        self.assertGreater(len(candidate_indices), 0)
        i, j = candidate_indices[len(candidate_indices) // 2]
        target_energy = float(results["E_mesh"][i, j])
        target_theta = float(results["theta_mesh"][i, j])
        target_a = float(results["a_mesh"][i, j])
        target_b = float(results["b_mesh"][i, j])

        fitter = EtchingTrackRatioFitter()
        fitter.provide_results(results)
        lookup = fitter.inverse_lookup(
            [target_a],
            [target_b],
            error_len=0.03,
            tol_energy=0.1,
            tol_theta=0.05,
            refine_energy=101,
            refine_theta=101,
        )

        self.assertGreater(len(lookup), 0)
        closest = lookup[np.argmin(lookup[:, 3])]
        self.assertAlmostEqual(closest[1], target_energy, delta=0.09)
        self.assertAlmostEqual(closest[2], target_theta, delta=0.03)


if __name__ == "__main__":
    unittest.main(verbosity=2)
