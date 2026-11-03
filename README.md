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

The older `main_app.py` entry point remains as a compatibility wrapper.

If Tkinter is missing, install a Python distribution that includes Tcl/Tk. No CUDA package is required for CPU use.

## First analysis

1. Start the GUI.
2. On **Image Slicing**, use **Browse** and select an image. The GUI stores and displays the resolved absolute path.
3. For the bundled example, keep the initial slicing values and click **Update Image**.
4. On **Pit Detection**, confirm the microscope resolution and physical radius range.
5. Start with `eps_product_thre_offset = 7` and `eps_RMS_thre = 8`.
6. Click **Detect Pits** and inspect both the overlay and status summary.
7. Use **Save Display** for the annotated PNG and **Save Status** for a restorable analysis state.

`example_state.sav` is a legacy demonstration file. Only load legacy `.sav` files that you trust because pickle files can execute code while loading.

## CPU and optional GPU

The base installation uses `requirements.txt` and works without a GPU. Set **Use CUDA = False** to request the CPU backend explicitly.

For a compatible CUDA 12.x system:

```powershell
python -m pip install -r requirements-gpu-cuda12.txt
```

When **Use CUDA = True**, the detector checks CuPy and an available CUDA device. If initialization fails, it records the reason and falls back to NumPy/SciPy.

## State files and absolute paths

New state files use the `.cr39state` extension. They are ZIP bundles containing JSON settings and, for a manual save, compressed NumPy arrays loaded with `allow_pickle=False`.

| Operation | Content | Location |
|---|---|---|
| Autosave on normal GUI close | settings only | `%LOCALAPPDATA%\CR39PitDetector\autosave.cr39state` |
| Automatic startup load | the autosave above, if present | same absolute path |
| **Save Status** | settings plus available image/result arrays | absolute path chosen by the user |
| Legacy **Load Status** | trusted `.sav` import | absolute path chosen by the user |

Writes are atomic: the application finishes a temporary bundle and then replaces the previous autosave. Generated image-slice folders, selected input images, saved states, and result images are all resolved to absolute paths.

## Detection output and diagnostics

`PitDetector.detect()` returns:

- `peaks`: shape `(N, 2)`, with detected center coordinates;
- `fitted_ovals`: shape `(N, 6)`, with `cx, cy, a, b, theta, status`.

The sixth column is a stable public status code:

| Code | Label | Meaning |
|---:|---|---|
| 0 | OK | fit accepted |
| -1 | FIT | insufficient contour or no valid ellipse solution |
| -2 | DENS | optional local-density rejection |
| -3 | RMS | normalized fit mismatch exceeds `eps_RMS_thre` |
| -4 | SIZE | fitted radius is outside the configured range |

A status identifies the stage that rejected a candidate; it does not uniquely prove the physical cause. The GUI also reports lower-level fit failures and center-fallback counts.

## Project layout

```text
cr39_detector/
  config.py            validated detector configuration
  detection.py         image detection and circle/ellipse fitting
  diagnostics.py       stable status codes and summaries
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

The old module names (`pitDetection.py`, `etchingModel.py`, `main_app.py`, and GUI helper modules) are thin compatibility wrappers. New users should start with `main.py`.

## Verification

Run:

```powershell
python -m unittest discover -s tests -v
```

The bundled CPU regression crop must produce 164 peaks: 123 `OK`, 26 `FIT`, and 15 `RMS`. The tests also verify blank-image handling, safe state round trips, absolute path resolution, configuration validation, NPZ schemas, and an energy-angle inverse lookup.

## Energy-angle reconstruction status

The tracked files in `savFiles_etchingModel` contain REL, track-range, fitted etching parameters, and energy-angle meshes used for continued development. They are protected by regression tests and are not regenerated by the GUI refactor. Image detection is usable now; a complete experimentally validated energy-angle reconstruction workflow is not yet exposed in the GUI.

See [DATA_ASSETS.md](DATA_ASSETS.md) for the tracked asset schema and [user_manual.md](user_manual.md) for the full operating guide.
