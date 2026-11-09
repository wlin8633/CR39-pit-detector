# Etching calibration benchmark

This benchmark reproduces the proton calibration used to fit
`V = 1 + alpha * REL**beta` without placing experiment-specific images, pit
tables, or local paths in the public repository. It records the conventions
required to interpret the saved fit.

## Reproduced baseline

The 2026-11-09 rerun used `VB = 1.68 um/h`, `t = 0.5 h`, the legacy selection
`data[22:]`, and the internal `MeV/cm` REL convention. It reproduced the stored
`to2MeV` result to numerical precision:

```text
alpha = 1.0072576602240627e-4
beta  = 1.501356499587407
theta = 1.2155977151506772 rad = 69.648 deg from the detector surface
```

This is a legacy computational baseline. This computational alpha must not be compared to the paper's reported
`1.007e-2` until the paper's REL scaling and reporting conversion have been
reconstructed explicitly. The agreement of the rerun establishes the code
baseline; it does not by itself resolve that reporting convention.

The fitted theta is measured from the detector surface. Its corresponding
angle from the detector normal is about 20.35 degrees. This is distinct from
the ellipse azimuth in the microscope image.

## Private data layout

Copy `local_paths.example.json` to `local_paths.json` and edit the absolute
paths. `local_paths.json`, generated output, and raw private data are ignored by
Git. The experimental pickle is loaded only from a trusted local file because
Python pickle files can execute code when opened.

The legacy inputs are:

- Geant4 `eDep` and `trackLen` files used to build the REL and range NPZ files;
- `Results/pit_size_vs_energy.sav`, produced from the pit-detection results;
- the detection state file(s) that record the image-analysis parameters.

The 2025 Results directory is about 25 MiB. It can remain in the experiment
archive and be referenced by absolute path; copying it into Git is not needed.
The WSL Geant4 folder can likewise be referenced or copied into the ignored
`data/raw/` folder when it is accessible.

## Commands

Inspect paths, selected energies, and conventions without fitting:

```powershell
python benchmarks/etching_calibration/run_calibration.py --dry-run
```

Fit only and save immediately (no curve or lookup mesh):

```powershell
python benchmarks/etching_calibration/run_calibration.py
```

For a long corrected fit, give the checkpoint an explicit name:

```powershell
python benchmarks/etching_calibration/run_calibration.py --output benchmarks/etching_calibration/output/20261109-corrected-diameter-fit.npz
```

Fit and generate the 800-point forward curve:

```powershell
python benchmarks/etching_calibration/run_calibration.py --generate-curve
```

Fit, generate the curve, and generate the energy-angle lookup mesh:

```powershell
python benchmarks/etching_calibration/run_calibration.py --generate-lookup
```

Resume curve and mesh generation without repeating a completed fit:

```powershell
python benchmarks/etching_calibration/run_calibration.py --from-fit benchmarks/etching_calibration/output/20261109-corrected-diameter-fit.npz --generate-lookup
```

Parameter fitting and lookup generation are separate stages. The fit is saved
before the optional slow stages begin. The lookup mesh may take tens of minutes
depending on the requested grid and computer. The benchmark config reproduces
the legacy `101 x 101` grid: `0-8 MeV` in `0.08 MeV` steps and `0-pi/2` in
`pi/200` steps.

Every new fit bundle also stores fit-point predictions, residuals, major/minor
RMSE, bias, R-squared, parameter correlation, and alpha relative uncertainty.
The runner skips lookup generation when it finds a negative axis R-squared,
non-finite fit predictions, alpha uncertainty larger than alpha, or parameter
correlation above 0.995. `--force-lookup` overrides this protection for
diagnostic work only; it does not make the calibration acceptable.

Compare two existing bundles without fitting:

```powershell
python benchmarks/etching_calibration/run_calibration.py --compare-only path/to/new.npz path/to/reference.npz
```

## Experimental selection and optical limit

The legacy `data[22:]` selection begins at about 2.056 MeV. It was an index
cut, not an exact `energy <= 2.0 MeV` condition. It excludes the smaller pits at
higher energy because they approach the optical/diffraction limit of the
532 nm imaging system. The index is kept to reproduce the published fit and is
stored in every new bundle.

## Axis conventions

- The forward model returns full major and minor diameters in micrometres.
- The current pit detector returns ellipse semiaxes and an in-plane azimuth.
- The legacy table-building code used `ovals[:, 2:4] * resolution` without a
  factor of two. Therefore `pit_size_vs_energy.sav` contains semiaxes.
- The legacy fit compared those semiaxes directly with the forward model's full
  diameters. The benchmark keeps `experimental_axis_multiplier_for_fit = 1.0`
  solely to reproduce that baseline and labels the mismatch explicitly.
- Set `experimental_axis_multiplier_for_fit = 2.0` to perform a new,
  geometrically matched fit. Treat that as a new calibration result rather
  than silently replacing the legacy NPZ.

## Expected non-finite regions

Some energy-angle pairs have no valid etched-pit solution in the current
forward model, especially at theta equal to zero or outside the simulated
transport range. These points are stored as NaN. The code reports one summary
count after curve or mesh generation instead of printing one warning for every
expected invalid evaluation.
