# CR-39 detector code and calibration audit

This inventory separates confirmed software defects, data-contract problems,
and unresolved physics/calibration questions. A plausible explanation is not
marked resolved until it has a test or an independently checkable result.

| Priority | Item | Current evidence | Resolution/status |
|---|---|---|---|
| P0 | Detector-to-calibration axis convention | `fitted_ovals[:,2:4]` and the paper's eigenvalue equations return semiaxes. The forward equations give full lengths (`a=b=2y` at normal incidence). The paper uses “axes” for both without a factor of two. | Internal arrays remain semiaxes; public measurements and GUI statistics now use explicit full diameters. The published transition is acknowledged as inconsistent. |
| P0 | Published calibration provenance | The legacy NPZ is reproducible, but it omitted axis convention, REL unit, `VB`, etch time, covariance, and input identity | New private bundles store these fields and input hashes. Do not replace the legacy NPZ silently. |
| P0 | Corrected fit quality | A simple `2x` axis refit produced poor minor-axis agreement, strong alpha-beta correlation, and alpha uncertainty larger than alpha | Not accepted. Audit the forward equations and run reduced-resolution multi-start fitting before generating a production mesh. |
| P0 | REL and alpha convention | Code baseline uses MeV/cm; the paper figure uses MeV/mm; the paper alpha is not explained by the known conversion alone | Open. Reconstruct the exact paper/Origin conversion and keep `REL_unit` with every alpha. |
| P0 | Incidence-angle reference | The equations/code use angle from the detector surface (`pi/2` normal), while supplementary prose says angle from the surface normal | Code and bundles now state the executable convention. Correct the scientific text and convert external angles explicitly. |
| P0 | New experimental inventory | Existing data do not provide the controlled samples/exposures requested for the collaboration | Planned new samples; export full diameters, counts, conditions, state ID, and rejection counts. |
| P1 | Fit acceptance criteria | The runner previously generated a mesh even when residuals and covariance were poor | Fit bundles now store residual metrics and identifiability diagnostics; strong warnings block lookup generation unless `--force-lookup` is explicit. Bound checks and study-specific acceptance thresholds remain open. |
| P1 | Optimizer speed and local minima | Full three-parameter fitting is slow and sensitive to initial values | Add reduced-resolution multi-start search, checkpointing, and full-resolution refinement. |
| P1 | Measurement export | The GUI saved overlays/states but had no canonical physical measurement table | CSV export now records semiaxes, full diameters, azimuth, status, pixel size, and conventions. Experiment/sample metadata inventory remains to be added. |
| P1 | State provenance | Old states do not declare whether oval columns are radii, semiaxes, or diameters | New states record internal and physical axis conventions; old trusted states are interpreted as semiaxes. |
| P2 | Ambiguous parameter names | `radius_*` means semiaxis bounds; `cir_radii_thre_percent` is actually a circle-mode background-intensity threshold | GUI labels and manual are explicit. Backward-compatible API renaming remains optional. |
| P2 | Uncertainty propagation | Detector rejection, diameter uncertainty, etch-rate uncertainty, REL uncertainty, and fit covariance are not propagated into energy-angle output | Open; required before quantitative reconstructed-spectrum error bars. |

## Confirmed contracts

- Internal detector oval:
  `(center_x_px, center_y_px, semi_major_px, semi_minor_px, azimuth_rad, status)`.
- Public physical measurement:
  `major_diameter_m = 2*semi_major_px*pixel_size_m` and likewise for minor.
- Fitted-image pixel size:
  `objective_resolution / extended_rate`.
- Image azimuth:
  radians in the image plane, modulo pi.
- Forward incidence angle:
  radians from the detector surface; pi/2 is normal incidence.
- Forward-model axes:
  full diameters in micrometres, confirmed by the normal-incidence limit.

## Published convention inconsistency

The detector ellipse equation uses eigenvalues equal to reciprocal squared
semiaxes, so `1/sqrt(lambda)` is a semiaxis. The pit-formation equations use
`a=(y1+y2)/sin(theta)` and `b=2*sqrt(y1*y2)`; at normal incidence these reduce
to `2*y`, a full diameter. The legacy calibration passed the former directly
to the latter. This is the convention error to report and correct; changing the
low-level ellipse fit itself would be the wrong software fix.

## Work order

1. Lock the detector/public measurement contract with regression tests.
2. Add canonical inventory export using full diameters.
3. Re-derive and test forward-model geometry independently of fitting.
4. Reconcile REL units and the paper alpha.
5. Add fit-quality gates and multi-start optimization.
6. Fit the new controlled samples and produce a versioned calibration bundle.
7. Generate lookup meshes only after the fit passes the acceptance checks.
