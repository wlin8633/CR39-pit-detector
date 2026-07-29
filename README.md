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

## User Manual
For a detailed breakdown of the GUI parameters, physical models (REL & stopping power), and workflow for large datasets, please refer to the [User Manual (user_manual.md)](user_manual.md).
