# CR-39 Pit Detector User Manual

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
- Manual state: absolute path selected in **Save Status**.
- Autosave: `%LOCALAPPDATA%\CR39PitDetector\autosave.cr39state`.

The autosave occurs during a normal GUI close and is loaded at the next startup. It contains settings only, keeping it small and robust. A manual state also includes available `extended_img`, `old_ext_img`, `display_img`, `ovals`, and `peaks` arrays.

## 3. Image Slicing page

### Image and slice controls

- **Raw Image Path**: microscope image in BMP, PNG, JPEG, or TIFF format.
- **desired_width / desired_height**: target size of an unextended slice.
- **extended_rate**: integer spatial enlargement/overlap factor used by the existing slicing algorithm.
- **slice h / w**: row and column index of the extended slice to open.
- **Batch scale h / w**: number of rows and columns used by **Batch Detect**.

### Bounding box

- **Fraction**: `x1, y1, x2, y2` are fractions from 0 to 1.
- **Absolute**: the values are pixel coordinates in the currently loaded extended image.

Coordinates are sorted and clipped to the image. **Reset Image** clears the current crop and restores the full bounding box.

### Display controls

- **centre_size**: center-marker radius in pixels.
- **edge_size**: fitted ellipse line thickness.
- **font_size / font_edge_size**: OpenCV label scale and thickness.

These change only the overlay, not the numerical fit.

### Actions

- **Browse**: select an image and store its resolved absolute path.
- **Update Image**: slice if required, load the selected extended slice, and apply the bounding box.
- **Detect Pits**: run detection on the current crop.
- **Replace Detect**: re-run a selected subregion and replace results there.
- **Show Display**: reopen the current result window.
- **Save Display**: export the BGR result correctly as a PNG.
- **Save Status / Load Status**: save a new `.cr39state` or load a new state/trusted legacy `.sav`.
- **Batch Detect**: process the configured slice grid. Validate a small representative region first because memory use scales with the stitched image size.

## 4. Pit Detection page

### Physical and optical inputs

| Parameter | Meaning | Check before use |
|---|---|---|
| `objective_resolution` | meters per raw-image pixel | microscope calibration |
| `extended_rate` | factor used for the extended image | must match slicing |
| `radius_min / radius_max` | accepted physical fitted-radius interval | species, energy, etching, and optics |
| `mode` | `ellipse`, `circle`, or `None` | ellipse is the normal energy-angle path |

The detector pixel size is `objective_resolution / extended_rate`. The configuration layer rejects a non-positive pixel size, reversed radius range, unsupported mode, negative RMS threshold, or non-positive timeout. A conversion/validation problem is reported in the GUI log as **Parameter error**.

### Peak finding

| Parameter | Effect |
|---|---|
| `bg_value` | fixed background intensity; `None` requests automatic handling |
| `clips_to_bg` | clips selected intensities to the background treatment |
| `peak_pad_rate` | padding around peak-search regions |
| `peak_charac_len` | physical characteristic scale for a peak |
| `peak_thre_offset` | shifts peak validation threshold |
| `peak_edge_std` | edge-statistics factor |
| `peak_gaussian_sigma` | smoothing in pixels before peak selection |
| `peak_closest_dist` | minimum physical separation of two peak centers |

Changing peak parameters changes the candidate population. Compare total peaks and every status count, not only the accepted count.

### Contour and ellipse quality

| Parameter | Effect |
|---|---|
| `eps_charac_len` | physical contour-search scale |
| `eps_product_thre_offset` | gradient-product contour threshold offset |
| `eps_allowed_center_offset` | triggers use of the detected peak center when the fitted center is too far away; it does not reject the pit |
| `eps_RMS_thre` | maximum normalized ellipse mismatch; failures become `RMS (-3)` |
| `cir_radii_thre_percent` | threshold for treating an ellipse as nearly circular |
| `timeout` | time limit for contour connection work |

For the bundled regression crop, start with `eps_product_thre_offset = 7` and `eps_RMS_thre = 8`. These values are a software regression reference, not a universal physical calibration.

### Optional density rejection

`mfp_detection` enables a local-density filter. `mfp` sets its physical scale; `None` estimates a scale from accepted peaks. `mfp_std` sets how unusual a dense region must be before it receives `DENS (-2)`. Keep this disabled unless density rejection is explicitly part of the analysis.

## 5. Reading the output

`peaks` has columns `x, y`. `fitted_ovals` has `cx, cy, a, b, theta, status`. The status column is stable:

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

- `state.json`: explicit GUI settings and schema version;
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

The intended chain is:

```text
accepted pit axes (a, b)
        -> REL-based etching model
        -> energy-angle lookup mesh
        -> candidate incident energy and angle
```

This is not yet a complete experimentally validated GUI workflow. Species-dependent calibration, CR-39 batch/manufacturer differences, etching-condition control, and experimental uncertainty still need to be treated explicitly.

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
