# Energy-angle data assets

The files below are tracked scientific-development inputs under the absolute checkout-relative directory `savFiles_etchingModel`. The GUI refactor does not regenerate or overwrite them.

| File | Principal arrays and shapes | Role |
|---|---|---|
| `R0_E_ω0350.npz` | `R0_E (800,)`, `E (800,)` | track range versus energy |
| `REL_x_E_ω0350.npz` | `REL_x_E (4999, 800)`, `x (4999,)`, `E (800,)` | REL versus depth and energy |
| `etching_rate_ratio_fit_results.npz` | scalar fit parameters; `E_fit/a_fit/b_fit (800,)`; four `(101, 101)` meshes | current fit and energy-angle lookup |
| `old_etching_rate_ratio_fit_results.npz` | scalar fit parameters; `E_fit/a_fit/b_fit (300,)`; four `(101, 101)` meshes | historical comparison |
| `to2MeV-etching_rate_ratio_fit_results.npz` | scalar fit parameters; `E_fit/a_fit/b_fit (800,)` | restricted-energy fit |
| `to8MeV-etching_rate_ratio_fit_results.npz` | scalar fit parameters; `E_fit/a_fit/b_fit (800,)` | restricted-energy fit |

The regression suite checks the current lookup schema, consistent mesh shapes, and a representative inverse lookup. These checks protect file compatibility; they do not establish experimental validity or transferability between particle species, CR-39 manufacturers/batches, or etching conditions.

The tracked legacy NPZ files do not contain enough provenance to identify the
bulk etch rate, etching time, REL unit, experimental axis convention, angle
reference, input hashes, or full parameter covariance from the file alone.
New private results should be created with
[`benchmarks/etching_calibration`](benchmarks/etching_calibration/README.md),
which stores these fields without publishing the experiment-specific inputs.

The 2026-11-09 rerun reproduces the numerical `to2MeV` baseline, but the legacy
experimental table contains ellipse semiaxes whereas the forward model returns
full diameters. The benchmark records this as a legacy convention mismatch; a
factor-of-two corrected refit must be versioned as a new calibration rather
than replacing the legacy files silently.
