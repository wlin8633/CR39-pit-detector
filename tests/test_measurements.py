import unittest
import tempfile
from pathlib import Path

import numpy as np

from cr39_detector.measurements import (
    fitted_oval_diameters,
    fitted_ovals_to_measurements,
    save_measurements_csv,
)


class MeasurementConventionTests(unittest.TestCase):
    def setUp(self):
        self.ovals = np.array(
            [
                [10.0, 20.0, 4.0, 2.0, 0.25, 0.0],
                [30.0, 40.0, 3.0, 1.0, 0.50, -3.0],
            ]
        )

    def test_semiaxes_are_converted_to_full_physical_diameters(self):
        diameters = fitted_oval_diameters(self.ovals, pixel_size_m=0.5e-6)
        np.testing.assert_allclose(
            diameters,
            [[4.0e-6, 2.0e-6], [3.0e-6, 1.0e-6]],
        )

    def test_measurement_table_retains_internal_and_physical_conventions(self):
        table = fitted_ovals_to_measurements(
            self.ovals,
            pixel_size_m=0.5e-6,
            accepted_only=True,
        )
        self.assertEqual(len(table), 1)
        self.assertEqual(table.dtype.names[2:6], (
            "semi_major_px",
            "semi_minor_px",
            "major_diameter_m",
            "minor_diameter_m",
        ))
        self.assertEqual(float(table["semi_major_px"][0]), 4.0)
        self.assertEqual(float(table["major_diameter_m"][0]), 4.0e-6)
        self.assertEqual(int(table["status"][0]), 0)

    def test_invalid_shape_and_pixel_size_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            fitted_oval_diameters(np.zeros((2, 5)), 1.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            fitted_oval_diameters(self.ovals, 0.0)

    def test_csv_export_records_units_conventions_and_rejected_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            output = save_measurements_csv(
                Path(folder) / "measurements.csv",
                self.ovals,
                pixel_size_m=0.5e-6,
            )
            text = output.read_text(encoding="utf-8")
            self.assertIn("pixel_size_m=4.9999999999999998e-07", text)
            self.assertIn("internal_axis_convention=semiaxes_px", text)
            self.assertIn("physical_axis_convention=full_diameters_m", text)
            self.assertIn("major_diameter_m", text)
            data = np.loadtxt(output, delimiter=",", comments="#")
            self.assertEqual(data.shape, (2, 8))
            self.assertAlmostEqual(data[0, 4], 4.0e-6)
            self.assertEqual(int(data[1, 7]), -3)


if __name__ == "__main__":
    unittest.main()
