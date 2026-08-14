# CR-39 Pit Detection & Etching Physics Model

<div align="center">
  <a href="https://pubs.aip.org/aip/adv/article/16/7/075051/3399496/Automated-CR-39-pit-detection-for-energy-and">
    <img src="https://img.shields.io/badge/AIP_Advances-Featured_Article-1f5b9d?style=for-the-badge" alt="AIP Advances Featured Article">
  </a>
  <a href="https://pubs.aip.org/aip/adv/article/16/7/075051/3399496/Automated-CR-39-pit-detection-for-energy-and">
    <img src="https://img.shields.io/badge/AIP-Scilight-2ea44f?style=for-the-badge" alt="AIP Scilight">
  </a>
</div>

> 🌟 **Featured Article & Scilight in AIP Advances** 🌟  
> The physical models and algorithms driving this repository have been officially published in **AIP Advances**, and we are honored to be selected as both a **Featured Article** and highlighted in a **Scilight**.  
> 📖 **Read the open-access paper:** [Automated CR-39 pit detection for energy and angle reconstruction](https://pubs.aip.org/aip/adv/article/16/7/075051/3399496/Automated-CR-39-pit-detection-for-energy-and)

This repository contains a GPU-accelerated algorithm for analyzing CR-39 Solid-State Nuclear Track Detectors. It automates the detection of overlapping pits, performs ellipse fitting to extract major/minor axes, and reconstructs incident ion energy and angle using a physical etching model.

## Core Features
1. **GPU Acceleration**: Utilizes `cupy` to parallelize image filters and morphological operations for processing high-resolution microscope images.
2. **Pit Detection Strategy**: Incorporates morphological filters to process images containing scratches, dust artifacts, and overlapping pits.
3. **Physical Etching Model**: Maps geometric track properties back to particle stopping power (REL), energy, and incident angle using a lookup-table approach based on Geant4/FLUKA data.
4. **Interactive GUI**: A multi-page Tkinter interface (`main_app.py`) for slicing images, tuning detection thresholds, and previewing results.

## Setup & Requirements
This project supports both **GPU acceleration (via CuPy)** and **CPU fallback (via NumPy)**. For optimal performance on high-resolution images, an NVIDIA GPU is highly recommended.

**Recommended Environment:**
We highly recommend using **Python 3.9.18** (or Python 3.9.x). This ensures numerical stability and compatibility with the required older NumPy versions (`numpy>=1.20.0, <2.0`).

**CPU Fallback Mode (No GPU Required):**
If you do not have an NVIDIA GPU or if CuPy is not installed, the application will automatically detect this and fall back to CPU execution. You can also manually disable GPU usage in the GUI by setting `Use CUDA = False`.

**CUDA Version Configuration (For GPU Users):**
By default, the `requirements.txt` installs `cupy-cuda12x` for CUDA 12.x. If you have a different CUDA version installed (e.g., CUDA 11.x), please open `requirements.txt` and change `cupy-cuda12x` to match your version (e.g., `cupy-cuda11x`), or install it manually:

```bash
# If you changed the version in requirements.txt, just run:
pip install -r requirements.txt

# Or manually install your specific cupy version (e.g., CUDA 11.x):
pip install cupy-cuda11x
```

## Running the Application
To launch the interactive GUI with the provided example:
```bash
python main_app.py
```
You can load the provided `example.bmp` and `example_state.sav` to see the algorithm in action.

## User Manual
For a detailed breakdown of the GUI parameters, physical models (REL & stopping power), and workflow for large datasets, please refer to the [User Manual (user_manual.md)](user_manual.md).
