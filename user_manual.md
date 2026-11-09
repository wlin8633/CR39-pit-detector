# CR-39 Pit Detector User Manual

See [CODE_AUDIT.md](CODE_AUDIT.md) for confirmed convention defects, unresolved
calibration questions, and the ordered remediation plan.

## 1. Installation and first run

Use Python 3.14 for the maintained environment. First activate a Python environment
using Conda, venv, or another environment manager of your choice. Then run these
commands from the repository root:

```powershell
python -m pip install -r requirements.txt
python main.py
```

`python -m cr39_detector` provides the same package entry point.

For CPU analysis, set **Use CUDA = False**. CUDA is optional. If GPU initialization fails while **Use CUDA = True**, the log explains the fallback and the detector continues on the CPU.

A first-time user can analyze the bundled example as follows:

1. Confirm that **Raw Image Path** points to the absolute path of `example.bmp`.
2. Click **Update Image**.
3. Open **Pit Detection** and keep the default physical calibration only if it matches the example.
4. Click **Detect Pits**.
5. Read the status summary before accepting the fitted sizes.
6. Use **Save Display** for an annotated PNG or **Save Status** for a restorable state.

## 2. Path policy and generated files

The application resolves paths before storing or using them. This prevents results from changing merely because the program was launched from another working directory.

- Input image: absolute path shown in **Raw Image Path**.
- Slices: absolute `savFiles_pitDetection\sliced_images` path under the checkout.
- Extended slices: absolute `savFiles_pitDetection\extended_images` path under the checkout.
- Exported overlay: absolute path selected in **Save Display**.
- Measurement table: absolute CSV path selected in **Save Measurements**.
- Manual state: absolute path selected in **Save Status**.
- Autosave: `%LOCALAPPDATA%\CR39PitDetector\autosave.cr39state`.

The autosave occurs during a normal GUI close and is loaded at the next startup. It contains settings only, keeping it small and robust. A manual state also includes available `extended_img`, `old_ext_img`, `display_img`, `ovals`, and `peaks` arrays.

## 3. Image Slicing page

### Image and slice controls

| Control | Unit/type | Definition |
|---|---|---|
| **Raw Image Path** | absolute path | Microscope image in BMP, PNG, JPEG, or TIFF format |
| `desired_width / desired_height` | raw-image pixels | Target size of one unextended slice |
| `extended_rate` | dimensionless integer | Spatial enlargement factor; one fitted-image pixel represents `objective_resolution / extended_rate` metres |
| `slice h / w` | integer index | Row and column of the extended slice to open |
| `Batch scale h / w` | integer count | Number of rows and columns processed by **Batch Detect** |

### Bounding box

- **Fraction**: `x1, y1, x2, y2` are dimensionless fractions from 0 to 1.
- **Absolute**: the values are pixel coordinates in the currently loaded extended image.

Coordinates are sorted and clipped to the image. **Reset Image** clears the current crop and restores the full bounding box.

### Display controls

- **centre_size**: center-marker radius in display pixels.
- **edge_size**: fitted ellipse line thickness in display pixels.
- **font_size**: dimensionless OpenCV font scale.
- **font_edge_size**: label thickness in display pixels.

These change only the overlay, not the numerical fit.

### Actions

- **Browse**: select an image and store its resolved absolute path.
- **Update Image**: slice if required, load the selected extended slice, and apply the bounding box.
- **Detect Pits**: run detection on the current crop.
- **Replace Detect**: re-run a selected subregion and replace results there.
- **Show Display**: reopen the current result window.
- **Save Display**: export the BGR result correctly as a PNG.
- **Save Measurements**: export all candidates to CSV with internal semiaxes,
  physical full diameters, image azimuth, status, pixel size, and conventions.
- **Save Status / Load Status**: save a new `.cr39state` or load a new state/trusted legacy `.sav`.
- **Batch Detect**: process the configured slice grid. Validate a small representative region first because memory use scales with the stitched image size.

## 4. Pit Detection page

### Physical and optical inputs

| Parameter | Unit/type | Definition | Check before use |
|---|---|---|---|
| `objective_resolution` | m/raw-image pixel | Microscope pixel calibration before image extension | Measure for the selected objective and camera |
| `extended_rate` | dimensionless integer | Spatial enlargement factor used by slicing and fitting | Must match the loaded extended image |
| `radius_min / radius_max` | m | Accepted internal semiaxis/radius interval | Set from species, energy, etching, and optics |
| `mode` | `ellipse`, `circle`, or `None` | Select ellipse fitting, circle fitting, or peak finding without shape fitting | Use `ellipse` for the energy-angle path |

The detector pixel size is `objective_resolution / extended_rate`. The configuration layer rejects a non-positive pixel size, reversed radius range, unsupported mode, negative RMS threshold, or non-positive timeout. A conversion/validation problem is reported in the GUI log as **Parameter error**.

### Peak finding

| Parameter | Unit/type | Definition |
|---|---|---|
| `bg_value` | image intensity/ADU or `None` | Fixed background; `None` uses the modal intensity |
| `clips_to_bg` | Boolean | Clip intensities above the selected background before edge detection |
| `peak_pad_rate` | dimensionless ratio | Padding factor applied to the Hough-like peak-search arrays |
| `peak_charac_len` | m | Line-extension distance used to vote for a pit center |
| `peak_thre_offset` | percentage points | Offset applied to the automatically selected gradient percentile |
| `peak_edge_std` | standard deviations | Required peak strength above the smoothed-edge mean |
| `peak_gaussian_sigma` | extended-image pixels | Gaussian smoothing sigma before local-maximum selection |
| `peak_closest_dist` | m | Local-maximum window and approximate minimum peak-center separation |

Changing peak parameters changes the candidate population. Compare total peaks and every status count, not only the accepted count.

### Contour and ellipse quality

| Parameter | Unit/type | Definition |
|---|---|---|
| `eps_charac_len` | m | Maximum semiaxis/contour-search distance around each candidate |
| `eps_product_thre_offset` | percentage points | Offset added to the gradient-product percentile used to keep contour pixels |
| `eps_allowed_center_offset` | m | Maximum center displacement before falling back to the detected peak center; it does not reject the pit |
| `eps_RMS_thre` | percent | Maximum `RMS_px / sqrt(semi_major_px*semi_minor_px) * 100`; failures become `RMS (-3)` |
| `cir_radii_thre_percent` | percent of background | Legacy name for the circle-mode intensity threshold; it is not an ellipse-circularity test |
| `timeout` | s | Time limit for circle-mode contour connection |

For the bundled regression crop, start with `eps_product_thre_offset = 7` and `eps_RMS_thre = 8`. These values are a software regression reference, not a universal physical calibration.

### Optional density rejection

`mfp_detection` is Boolean and enables a local-density filter. `mfp` is a
length in metres; `None` estimates it from accepted peak density. `mfp_std` is
the number of standard deviations above the smoothed-density mean required for
`DENS (-2)`. Keep this disabled unless density rejection is explicitly part of
the analysis.

### Runtime and backend controls

| Parameter | Unit/type | Definition |
|---|---|---|
| `timeout` | s | Maximum contour-connection time per candidate in circle mode |
| `verbose` | Boolean | Print detailed progress and diagnostic messages |
| `use_cuda` | Boolean | Request the optional CuPy/CUDA backend; if unavailable, detection reports the reason and uses NumPy on the CPU |

## 5. Reading the output

### Detector array contract

`peaks` and `fitted_ovals` use the coordinate system of the fitted extended
image: origin at the upper-left, x to the right, and y downward.

| Array/column | Unit | Definition |
|---|---|---|
| `peaks[:,0]`, `peaks[:,1]` | extended-image pixels | Detected center `(x,y)` |
| `fitted_ovals[:,0:2]` | extended-image pixels | Fitted center `(center_x,center_y)` |
| `fitted_ovals[:,2]` | extended-image pixels | Internal semi-major axis |
| `fitted_ovals[:,3]` | extended-image pixels | Internal semi-minor axis |
| `fitted_ovals[:,4]` | rad | In-plane major-axis azimuth, modulo π; not particle incidence angle |
| `fitted_ovals[:,5]` | integer code | Fit status listed below |

The physical fitted-image pixel size is

```text
pixel_size_m = objective_resolution / extended_rate
```

Do not publish or calibrate directly from columns 2 and 3. The physical
measurement contract is:

```text
major_diameter_m = 2 * semi_major_px * pixel_size_m
minor_diameter_m = 2 * semi_minor_px * pixel_size_m
```

Use `fitted_oval_diameters()` or `fitted_ovals_to_measurements()` for this
conversion. The latter retains both the internal semiaxes and physical full
diameters in explicitly named fields. GUI labels beginning with `D=` are full
diameters displayed in micrometres; the green OpenCV overlay still requires
the internal semiaxes.

The CSV created by **Save Measurements** keeps rejected rows rather than only
`OK` rows. This is intentional: an experimental inventory must preserve the
candidate population and status counts needed to quantify detection loss.

### Status codes

The status column is stable:

| Status | Meaning | Common observations | Quantitative use |
|---|---|---|---|
| `OK (0)` | accepted fit | contour, RMS, and size tests passed | include after scientific validation |
| `FIT (-1)` | no valid fit | too few contour points, invalid conic, strong overlap, or artifact | exclude |
| `DENS (-2)` | optional density rejection | unusually dense local region | exclude |
| `RMS (-3)` | fit mismatch too large | overlap, partial pit, scratch, illumination issue, or unsuitable threshold | exclude |
| `SIZE (-4)` | radius outside bounds | wrong species/range or non-pit artifact | exclude |

The code reports the stage of failure, not a unique physical diagnosis. In particular, an `RMS` rejection is not automatically proof of overlapping pits.

The log includes:

- backend and any CUDA fallback reason;
- runtime, peak count, accepted count, and status breakdown;
- lower-level fit-failure categories;
- number of center fallbacks.

## 6. State files and legacy migration

### New format

A `.cr39state` is a versioned ZIP bundle:

- `state.json`: explicit GUI settings, schema version,
  `oval_axis_convention=semiaxes_px`, and
  `measurement_axis_convention=full_diameters_m`;
- `arrays.npz`: optional compressed numerical arrays.

NumPy arrays are loaded with `allow_pickle=False`. Saves use a temporary file plus atomic replacement so an interrupted close is less likely to corrupt the previous autosave.

### Legacy `.sav`

The loader retains one-way compatibility with the previous pickle-based `.sav` files, including `example_state.sav`. Pickle is not safe for untrusted files. Load only a file you created or obtained from a trusted collaborator, then save it as `.cr39state`.

## 7. Reproducible parameter tuning

1. Keep a fixed representative image and a manually saved state.
2. Confirm microscope pixel calibration and etching conditions first.
3. Tune peak-finding parameters while recording total candidate counts.
4. Tune contour and RMS parameters while inspecting examples of every rejection class.
5. Tune physical radius limits only from a defensible species/energy/etching range.
6. Enable density rejection last, if needed.
7. Record the full settings, software revision, and all five status counts.
8. Run the regression tests before and after algorithm or dependency changes.

A larger `OK` count can indicate improved recovery, but it can also indicate weaker rejection of artifacts.

## 8. Energy-angle development assets

`savFiles_etchingModel` contains tracked REL, range, fitted parameter, and lookup-mesh NPZ files. The image GUI does not rewrite them. `EtchingTrackRatioFitter.inverse_lookup()` remains available for development and is covered by a schema/loadability regression test.

For calibration development, use
[`benchmarks/etching_calibration`](benchmarks/etching_calibration/README.md).
The fast `fit_parameters()` stage is separate from `generate_forward_curve()`
and the much slower `generate_lookup_mesh()` stage. The benchmark records the
experiment-specific provenance in a private, ignored NPZ bundle.

The intended chain is:

```text
accepted full pit diameters (major_diameter, minor_diameter)
        -> REL-based etching model
        -> energy-angle lookup mesh
        -> candidate incident energy and angle
```

### Forward-model units and definitions

| Quantity | Unit | Definition |
|---|---|---|
| stored `REL_x_E` | MeV/m | Geant4-derived restricted energy loss before fitting conversion |
| model `REL` | configured; e.g. MeV/cm or MeV/mm | Unit used in `V = 1 + alpha*REL**beta`; it must be stored with every calibration |
| `alpha` | `(REL unit)^(-beta)` | Unit-dependent coefficient; numerical values cannot be compared across REL conventions without conversion |
| `beta` | dimensionless | REL-response exponent |
| `V` | dimensionless | Track etch rate divided by bulk etch rate |
| `E`, `E_exp`, `E_mesh` | MeV | Incident-particle kinetic energy |
| stored depth `x`, range `R0` | m | Raw transport-asset coordinates |
| internal depth and range | µm | Units used by `compute_a_b()` |
| `VB` | µm/h | Bulk etch rate per detector surface |
| `t` | h | Etching duration |
| `h = VB*t` | µm/surface | Removed depth on one surface |
| model `theta` | rad from detector surface | π/2 is normal incidence; this is not image azimuth |
| model `a,b` | µm | Full major and minor pit diameters |
| lookup `a_mesh,b_mesh` | µm | Full diameters over the energy-angle grid |
| `restriction_energy_eV` | eV | Geant4 restricted-energy threshold used to construct REL |

Parameter covariance has heterogeneous units: diagonal entries are the squared
units of `alpha`, `beta`, and `theta`; off-diagonal entries are their products.
Always keep the covariance together with the REL unit and angle convention.

This is not yet a complete experimentally validated GUI workflow. Species-dependent calibration, CR-39 batch/manufacturer differences, etching-condition control, and experimental uncertainty still need to be treated explicitly.

The model and detector's internal array use different geometry conventions:
the forward model returns full ellipse diameters, while `fitted_ovals` stores
semiaxes. The public measurement helpers convert detector results to full
diameters before calibration. The
forward-model incidence angle is measured from the detector surface (`90 deg`
is normal incidence); the detector's ellipse angle is an in-plane azimuth.
Do not exchange these values without an explicit conversion.

## 9. Verification and troubleshooting

Run all tests:

```powershell
python -m unittest discover -s tests -v
```

Expected bundled CPU baseline: 164 peaks = 123 `OK` + 26 `FIT` + 15 `RMS`.

- **Image does not load**: confirm the displayed path is absolute and the file format is supported by OpenCV.
- **No pits detected**: verify the image crop, pixel calibration, peak threshold, contrast, and radius range.
- **Many FIT results**: inspect contour point count and overlap/artifacts; then review `eps_product_thre_offset`.
- **Many RMS results**: inspect real overlays before increasing `eps_RMS_thre`.
- **CUDA fallback**: either install the matching optional CuPy package or set **Use CUDA = False**.
- **Old state warns about pickle**: load it only if trusted, then convert it with **Save Status**.
