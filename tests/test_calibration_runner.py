import unittest

from benchmarks.etching_calibration.run_calibration import fit_quality_warnings


class CalibrationRunnerTests(unittest.TestCase):
    def test_strong_pathologies_block_a_production_mesh(self):
        warnings = fit_quality_warnings(
            {
                "fit_finite_fraction": 1.0,
                "a_r2": 0.9,
                "b_r2": -0.1,
                "alpha_relative_uncertainty": 1.8,
                "max_abs_parameter_correlation": 0.998,
            }
        )
        self.assertEqual(len(warnings), 3)
        self.assertTrue(any("b-axis R2" in warning for warning in warnings))
        self.assertTrue(any("alpha uncertainty" in warning for warning in warnings))
        self.assertTrue(any("correlation" in warning for warning in warnings))

    def test_well_identified_fit_has_no_blocking_warning(self):
        warnings = fit_quality_warnings(
            {
                "fit_finite_fraction": 1.0,
                "a_r2": 0.99,
                "b_r2": 0.98,
                "alpha_relative_uncertainty": 0.1,
                "max_abs_parameter_correlation": 0.8,
            }
        )
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
