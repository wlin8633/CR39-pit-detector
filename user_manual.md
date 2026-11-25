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

For CPU analysis, set **Use CUDA = False**. CUDA is optional. If GPU initialization
fails while **Use CUDA = True**, the log explains the fallback and the detector
continues on the CPU.

For GPU use, first check `nvidia-smi` and `nvcc --version`, install a compatible
NVIDIA driver and CUDA Toolkit, and then install the matching CuPy package:
`cupy-cuda12x` for CUDA 12.x or `cupy-cuda13x` for CUDA 13.x. See the README
[GPU installation section](README.md#cpu-and-optional-gpu) and the official
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) before
installing. Only one CuPy package may be installed in an environment.

A first-time user can analyze the bundled example as follows:

1. Confirm that **Raw Image Path** points to the absolute path of `example.bmp`.
2. Click **Update Image**.
3. Open **Pit Detection** and keep the default physical calibration only if it matches the example.
4. Click **Detect Pits**.
5. Read the status summary before accepting the fitted sizes.
6. In **Save and reopen**, set **Save Directory** (blank uses the raw image's folder), select the outputs you want, and click **Save Selected**.

## 2. Path policy and generated files

The application resolves paths before storing or using them. This prevents results from changing merely because the program was launched from another working directory.

- Input image: absolute path shown in **Raw Image Path**.
- Slices and extended slices: `%LOCALAPPDATA%\CR39PitDetector\image_tiles`.
- Exported overlay, measurement CSV, and manual state: the folder set under **Save Directory**. Use **Browse…** or enter a path; a blank field follows the raw image's folder. Relative paths are resolved from the raw image's folder. The selected folder is remembered by autosave, not by manual states. **Save Selected** no longer opens a folder picker. Unnumbered saves still ask before replacing files.
- Autosave: `%LOCALAPPDATA%\CR39PitDetector\autosave.cr39state`.

Both autosave and manual status contain settings only, with ROI and result provenance in a manual save. They never embed the enlarged image, display, peaks, or fitted ovals. Check **Measurements** to keep completed fits and **Display** to keep the overlay. A manual status requires the original image to rerun detection.

## 3. Image Slicing page

### Image and slice controls

| Control | Unit/type | Definition |
|---|---|---|
| **Raw Image Path** | absolute path | Microscope image in BMP, PNG, JPEG, or TIFF format |
| `desired_width / desired_height` | raw-image pixels | Target size of one unextended slice |
| `extended_rate` | dimensionless number ≥ 1 | Uniform enlargement, e.g. 1, 1.2, 1.5, or 2. One fitted-image pixel represents `objective_resolution / extended_rate` metres. |
| Extension interpolation | method | **Cubic** (default) or **Linear**. Applies to single and batch slicing; no resampling occurs at rate 1. |
| `slice h / w` | integer index | Row and column of the extended slice to open |
| `Batch scale h / w` | integer count | Number of rows and columns processed by **Batch Detect** |

### Bounding box

- **Fraction**: `x1, y1, x2, y2` are dimensionless fractions from 0 to 1.
- **Absolute**: the values are pixel coordinates in the currently loaded extended image.

Coordinates are sorted and clipped against the full extended tile. Repeating **Update Image** with the same bounding box selects the same area. After changing the image path, slice indices, slice dimensions, extension rate, or interpolation, click **Reset Image** to load the new image. This restores the full bounding box and clears previous fits. Reapply the desired box with **Update Image** before detection.

The Canvas shows the whole slice so that you can see where the current
selection (red) is relative to saved measurements (blue). **Zoom to ROI**
restores a cropped preview; the ROI inputs still refer to the full extended
slice. Blue outlines are loaded from matching Measurements CSV metadata in
**Save Directory** after a tile is loaded. They update when measurements are
saved or when you click **Refresh ROIs**. The detector maps old boxes through
raw-image pixels to the current slice and extension rate, and clips them to the
visible image. Status or Display alone does not make a blue outline.

The interpolation method is saved with the analysis state. Older states without this setting use Cubic, the original behavior. Interpolation does not change the physical pixel scale or add optical detail, but it can change the pixel gradients, fitted contours, and pit counts. For a crowded field, compare both modes on the same raw image and ROI before changing the method used across a series.

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
- **Replace Detect**: re-run a subregion of a previous single-image detection and replace results inside that subregion. The original and replacement fits must share the same image geometry and pixel calibration.
- **Remove Detect**: mark detections inside the current box as manually excluded from a previous single-image detection, without rerunning the algorithm. See below for the distinction between excluding detections and masking an area.
- **Show Display**: reopen the current result window.
- **Display**: export the annotated image as a PNG or BMP.
- **Measurements**: export all existing candidates to CSV with internal semiaxes,
  physical full diameters, image azimuth, status, pixel size, and conventions.
- **Status / Load Status**: save settings and ROI as `.cr39state` or load a current state, older array-bearing state, or trusted legacy `.sav`. For a new compact state, use **Update Image** (not **Reset Image**, which resets the ROI) and **Detect Pits**, or run **Batch Detect** again.
- **Batch Detect**: process the configured slice grid. Validate a small representative region first because memory use scales with the stitched image size.
- **Stop** is available on both tabs during Detect Pits, Replace Detect, Batch Detect, and image slicing. It requests cancellation at the next safe checkpoint. The current NumPy, OpenCV, or CUDA call may need to finish first. A stopped job keeps the last completed measurements and overlay. Wait for the stopped message before starting another job.

### Multiple ROIs and manual exclusions

1. Set the first ROI, click **Update Image**, and run **Detect Pits**.
2. Set **Save Directory**, then check **Save as new ROI** and the outputs you want.
   Click **Save Selected**. For example, Status and Measurements may
   save as `image__roi04_state.cr39state` and `image__roi04_measurements.csv`.
3. Set the next ROI, update, detect, and save again. The checkbox stays selected
   until you clear it or restart the GUI: another save creates a new number even
   if the ROI did not change. The next number is based on the highest existing
   state, measurement, or display number for that image, not the number of files.

To clean up a fitted ROI, select a box entirely within it and click
**Remove Detect**. The operation uses candidate peak centers, hides the
excluded detections from the overlay, and preserves their fit rows and original
status codes. In the CSV, a row is accepted for counting when `status=0` and
it is not marked `manual_excluded=1`. The manual-exclusion column is only
added when at least one row is marked. Save **Measurements** to preserve edits.

Leave **Exclude area too** off when removing false positives from an otherwise
readable region. Turn it on when the area itself could not be measured, such as
a scratch that hides real pits. The CSV `result_metadata` records
`manual_exclusion_regions`: each rectangle has `bbox=[x1,y1,x2,y2]` in
full extended-tile pixels and an `exclude_area` flag. An unusable rectangle is
recorded even if it contains no candidates. The GUI does not compute density:
subtract the union of the area-exclusion rectangles within `result_bbox` from
the ROI area, then convert the remaining pixel area using `pixel_size_m²`.
Overlapping rectangles are subtracted only once.

## 4. Pit Detection page

### Physical and optical inputs

| Parameter | Unit/type | Definition | Check before use |
|---|---|---|---|
| `objective_resolution` | m/raw-image pixel | Microscope pixel calibration before image extension | Measure for the selected objective and camera |
| `extended_rate` | dimensionless number ≥ 1 | Spatial enlargement factor used by slicing and fitting | Use **Reset Image** after changing it |
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

The CSV created by the **Measurements** option keeps rejected rows rather than only
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
  `measurement_axis_convention=full_diameters_m`, and saved ROI information in manual states.

An older state may also contain `arrays.npz`; it remains readable with `allow_pickle=False`. A new state does not restore cached fits or reapply manual edits after a new detection. Export **Measurements** before closing a run, especially after **Replace Detect** or **Remove Detect**. The CSV header records its source image, fitted ROI and dimensions, fit settings, and physical pixel size. Saves use a temporary file plus atomic replacement for settings and CSV.

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
- **Old state warns about pickle**: load it only if trusted, then select **Status** in **Save Selected**.
