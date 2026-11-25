# CR-39 Pit Detector

This repository detects and fits pits in CR-39 microscope images. It provides:

- a Tkinter GUI for image slicing, parameter tuning, detection, and result export;
- CPU execution with NumPy/SciPy and optional NVIDIA GPU execution with CuPy;
- stable six-column pit output with explicit rejection codes;
- a separate etching/energy-angle model and its preserved NPZ development assets.

The underlying method is described in the open-access AIP Advances article, [Automated CR-39 pit detection for energy and angle reconstruction](https://pubs.aip.org/aip/adv/article/16/7/075051/3399496/Automated-CR-39-pit-detection-for-energy-and).

## Quick start

The maintained environment is Python 3.14. Activate a Python environment using Conda,
venv, or another environment manager of your choice. Then run the following commands
from the repository root:

```powershell
python -m pip install -r requirements.txt
python main.py
```

The package entry point is equivalent:

```powershell
python -m cr39_detector
```

If Tkinter is missing, install a Python distribution that includes Tcl/Tk. No CUDA package is required for CPU use.

## First analysis

1. Start the GUI.
2. On **Image Slicing**, use **Browse** and select an image. The GUI stores and displays the resolved absolute path.
3. For the bundled example, keep the initial slicing values and click **Update Image**.

   Fractional extension rates such as `1.2` and `1.5` are supported. Choose
   **Extension interpolation**: **Cubic** (default) or **Linear**. After
   changing the image path, slice, dimensions, extension rate, or interpolation, click
   **Reset Image**, then reapply any bounding box with **Update Image**.
4. On **Pit Detection**, confirm the microscope resolution and physical radius range.
5. Start with `eps_product_thre_offset = 7` and `eps_RMS_thre = 8`.
6. Click **Detect Pits** and inspect both the overlay and status summary.
7. In **Save Selected**, check **Status**, **Measurements**, or **Display**
   and choose one destination folder. Keep Status and Measurements for quantitative work.
   Check **Save as new ROI** to export another ROI of the same image without overwriting an earlier group.

`example_state.sav` is a legacy demonstration file. Only load legacy `.sav` files that you trust because pickle files can execute code while loading.

### Multiple ROIs and manual cleanup

For each ROI on the same image, set the box, click **Update Image**, run
**Detect Pits**, and use **Save Selected** with **Save as new ROI** checked.
The number is the largest one already saved for that image in the destination
folder, plus one; gaps are not filled. All selected files share the new number.
Leaving the option unchecked uses the original filename and asks before overwriting.

To remove false detections from a completed single-image analysis, set a smaller
box and click **Remove Detect**. It does not refit or delete the original CSV
rows; it flags them as manually excluded. Check **Exclude area too** only if
real pits cannot be observed in that region. Save **Measurements** to keep
manual edits. See [ROI and density details](user_manual.md#multiple-rois-and-manual-exclusions).

## CPU and optional GPU

The base installation uses `requirements.txt` and works without a GPU. Set **Use CUDA = False** to request the CPU backend explicitly.

### Check the GPU, driver, and CUDA version

GPU execution requires a compatible NVIDIA GPU and driver. Use these commands before
choosing a CuPy package:

```powershell
nvidia-smi
nvcc --version
```

- `nvidia-smi` identifies the GPU and reports the **maximum CUDA version supported by
  the installed driver**. It does not necessarily report the CUDA Toolkit installed in
  the active environment.
- `nvcc --version`, when available, reports the installed CUDA Toolkit used by a plain
  `cupy-cudaXX` wheel.
- Check the GPU's compute capability in NVIDIA's
  [current GPU table](https://developer.nvidia.com/cuda/gpus) or
  [legacy GPU table](https://developer.nvidia.com/cuda/gpus/legacy). NVIDIA's
  [Toolkit/driver/architecture matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html)
  shows when support for older architectures ends.
- Check driver compatibility in NVIDIA's
  [CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
  As a major-version guide, CUDA 12.x requires driver branch 525 or newer and CUDA
  13.x requires branch 580 or newer.

### Install CUDA and CuPy

1. Install a compatible NVIDIA driver and CUDA Toolkit. Use NVIDIA's
   [CUDA Toolkit download page](https://developer.nvidia.com/cuda-downloads) and
   [Windows installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
   rather than assuming that every computer needs the same CUDA release.
2. Activate the Python environment intended for this project and install the base
   dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Install exactly one CuPy package matching the installed Toolkit major version. The
   current CuPy 14 stable wheels support CUDA 12.x and 13.x, as listed in the official
   [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html):

   | Installed CUDA Toolkit | CuPy command |
   |---|---|
   | CUDA 12.x | `python -m pip install "cupy-cuda12x>=14,<15"` |
   | CUDA 13.x | `python -m pip install "cupy-cuda13x>=14,<15"` |

   CUDA 11.x and older do not have a CuPy 14 stable wheel. Use the CPU backend or
   update the driver/Toolkit instead of inventing a `cupy-cuda11x>=14` requirement.

Conda users can alternatively let conda-forge select compatible runtime components:

```powershell
conda install -c conda-forge cupy
```

The included `requirements-gpu-cuda12.txt` is only a convenience example:

```text
-r requirements.txt
cupy-cuda12x>=14,<15
```

If a user prefers a requirements file for CUDA 13.x, copy this file and change only
the second line to `cupy-cuda13x>=14,<15`. Do not install multiple CuPy packages in
one environment. The CuPy installation guide also documents optional CUDA component
wheels for users who intentionally do not install a system-wide Toolkit.

Verify the installation before enabling **Use CUDA**:

```powershell
python -c "import cupy as cp; cp.show_config(); print('CUDA devices:', cp.cuda.runtime.getDeviceCount())"
```

When **Use CUDA = True**, the detector checks CuPy and an available CUDA device. If
initialization fails, it records the reason and falls back to NumPy/SciPy.

## State files and absolute paths

New `.cr39state` files are small ZIP bundles of settings and saved ROI geometry. They contain no image pixels or numerical measurements. Keep the original image: after **Load Status**, use **Update Image** to rebuild the ROI and **Detect Pits** to fit it again, or use **Batch Detect** to repeat a batch run.

| Operation | Content | Location |
|---|---|---|
| Autosave on normal GUI close | settings only | `%LOCALAPPDATA%\CR39PitDetector\autosave.cr39state` |
| Automatic startup load | the autosave above, if present | same absolute path |
| **Status** | settings and ROI metadata | `<image>_state.cr39state` |
| **Measurements** | all candidate fits and measurement metadata | `<image>_measurements.csv` |
| **Display** | annotated image | `<image>_results.png` or `.bmp` |
| **Load Status** | current settings, historical array-bearing state, or trusted `.sav` | absolute path chosen by the user |

Writes are atomic: the application finishes a temporary bundle and then replaces the previous autosave. Generated image-slice folders, selected input images, saved states, and result images are all resolved to absolute paths.

## Detection output and diagnostics

`PitDetector.detect()` returns:

- `peaks`: shape `(N, 2)`, with `(x_px, y_px)` in the fitted extended image;
- `fitted_ovals`: shape `(N, 6)`, with
  `(center_x_px, center_y_px, semi_major_px, semi_minor_px, azimuth_rad, status)`.

The raw detector array intentionally stores semiaxes in pixels. Ellipse fitting,
OpenCV drawing, physical semiaxis bounds, and RMS normalization all use this
internal convention. Scientific statistics and calibration must use full
diameters. Convert explicitly with `fitted_oval_diameters()` or
`fitted_ovals_to_measurements()`; the GUI overlay labels full diameters as
`D=(major,minor)um`.

The **Measurements** option exports all candidates, including rejected rows, with
centers and internal semiaxes in pixels, full diameters in metres, azimuth in
radians, and the integer status code. It exports existing results without
running detection. The CSV keeps full floating-point precision and records the
source image, fit ROI, fit settings, physical pixel size and axis conventions.

The fitted-image pixel size is
`objective_resolution / extended_rate` metres per extended-image pixel. Image
coordinates start at the upper-left, with x increasing rightward and y
downward. The ellipse `azimuth_rad` is an in-plane orientation modulo pi and is
not the forward model's particle-incidence angle.

The sixth column is a stable public status code:

| Code | Label | Meaning |
|---:|---|---|
| 0 | OK | fit accepted |
| -1 | FIT | insufficient contour or no valid ellipse solution |
| -2 | DENS | optional local-density rejection |
| -3 | RMS | normalized fit mismatch exceeds `eps_RMS_thre` |
| -4 | SIZE | fitted radius is outside the configured range |

A status identifies the stage that rejected a candidate; it does not uniquely prove the physical cause. The GUI also reports lower-level fit failures and center-fallback counts.

### Core unit conventions

| Quantity | Unit | Definition |
|---|---|---|
| `objective_resolution` | m/raw pixel | Microscope calibration before extension |
| detector `pixel_size` | m/extended pixel | `objective_resolution / extended_rate` |
| fitted centers and semiaxes | extended-image pixel | Internal detector representation |
| reported major/minor size | m or displayed µm | Full diameter: `2 * semiaxis_px * pixel_size` |
| ellipse azimuth | rad | In-plane image orientation, modulo π |
| forward-model `theta` | rad | Track angle from detector surface; π/2 is normal incidence |
| energy | MeV | Incident-particle kinetic energy |
| stored Geant4 REL | MeV/m | Raw transport asset; converted explicitly for fitting |
| `alpha` | `REL_unit**(-beta)` | Unit-dependent coefficient in `V = 1 + alpha*REL**beta` |
| `beta`, `V` | dimensionless | Etch-response exponent and track/bulk etch-rate ratio |

## Project layout

```text
cr39_detector/
  config.py            validated detector configuration
  detection.py         image detection and circle/ellipse fitting
  diagnostics.py       stable status codes and summaries
  measurements.py      explicit semiaxis-to-full-diameter conversion
  etching.py           REL/etching and energy-angle model
  io.py                Geant4/GetData text parsers
  paths.py             central absolute-path policy
  state.py             versioned .cr39state persistence
  gui/                 Tkinter application, pages, and image window
examples/
  detect_example.py    minimal CPU API example
tests/
  test_regression.py   detector and energy-angle asset protection
  test_state.py        state, path, and configuration tests
```

Start the GUI with `python main.py` or `python -m cr39_detector`. For existing
scripts, import detector and etching classes from `cr39_detector` and GUI helpers
from `cr39_detector.gui` and its submodules.

## Verification

Run:

```powershell
python -m unittest discover -s tests -v
```

The bundled CPU regression crop must produce 164 peaks: 123 `OK`, 26 `FIT`, and 15 `RMS`. The tests also verify blank-image handling, safe state round trips, absolute path resolution, configuration validation, NPZ schemas, and an energy-angle inverse lookup.

## Energy-angle reconstruction status

The tracked files in `savFiles_etchingModel` contain REL, track-range, fitted etching parameters, and energy-angle meshes used for continued development. They are protected by regression tests and are not regenerated by the GUI refactor. Image detection is usable now; a complete experimentally validated energy-angle reconstruction workflow is not yet exposed in the GUI.

The private calibration can be reproduced with
[`benchmarks/etching_calibration`](benchmarks/etching_calibration/README.md).
The runner separates parameter fitting from the slower forward-curve and
energy-angle mesh generation, saves the fit before optional slow work, and
records `VB`, etching time, REL unit, axis convention, angle reference, full
fit covariance, selected experimental points, and input hashes. Local paths,
raw experimental data, and generated bundles remain ignored by Git.

See [DATA_ASSETS.md](DATA_ASSETS.md) for the tracked asset schema and [user_manual.md](user_manual.md) for the full operating guide.
The active software/calibration issue inventory and work order are maintained in
[CODE_AUDIT.md](CODE_AUDIT.md).
