# CR-39 Pit Detection & Etching Physics Model

This repository contains an advanced, GPU-accelerated algorithm for analyzing CR-39 Solid-State Nuclear Track Detectors. It automates the detection of overlapping pits, performs ellipse fitting to extract major/minor axes, and reconstructs incident ion energy and angle using a robust physical etching model.

## Core Features
1. **GPU Acceleration**: Utilizes `cupy` to massively parallelize image filters and morphological operations, enabling real-time processing of high-resolution microscope images.
2. **Advanced Pit Detection**: Handles noisy environments, scratches, dust artifacts, and overlapping pits.
3. **Physical Etching Model**: Maps geometric track properties back to particle stopping power (REL), energy, and incident angle using a lookup-table approach based on Geant4/FLUKA data.
4. **Interactive GUI**: A multi-page Tkinter interface (`main_app.py`) for slicing images, tuning detection thresholds, and previewing results.

## Setup & Requirements
This project relies on NVIDIA GPUs for acceleration. 
Ensure you have a CUDA 12.x compatible GPU and drivers installed.

```bash
pip install -r requirements.txt
```

## Running the Application
To launch the interactive GUI with the provided example:
```bash
python main_app.py
```
You can load the provided `example.bmp` and `example_state.sav` to see the algorithm in action.
