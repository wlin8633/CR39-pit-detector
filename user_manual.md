# CR-39 Pit Detection & Etching Analysis User Manual

This manual provides a detailed guide on using the GPU-accelerated **CR-39 Pit Detection** application. The software is designed to analyze solid-state nuclear track detector (CR-39) images, identify overlapping pits, fit ellipses to tracks, and back-calculate physical particle properties (such as energy and incident angle) using physical etching models.

---

## 1. Quick Start Workflow

1. **Launch the application:** Run `python main_app.py` in an environment where `cupy` and CUDA are installed.
2. **Load an Image:** On the first tab (`Slicing`), browse for your CR-39 microscope image (e.g., `example.bmp`).
3. **Set Image Resolution:** On the second tab (`Detection`), ensure `objective_resolution (m)` matches your microscope's pixel calibration.
4. **Update Image:** Go back to `Slicing` and click **Update Image**.
5. **Detect Pits:** Click **Detect Pits**. A canvas window will pop up showing the detected tracks, fitted ellipses, and calculated physical sizes.

---

## 2. Tab 1: Image Slicing & Preprocessing (`Slicing`)

Because CR-39 scans are often stitched gigapixel images, loading the entire image into the GPU memory at once is impossible. This tab allows you to slice, crop, and manipulate sub-regions.

### Image Selection & Slicing
* **Raw Image Path**: Absolute path to your `.bmp`, `.png`, or `.tif` file.
* **desired_width / desired_height**: The target resolution to slice the raw image into (e.g., 4096 x 3072).
* **extended_rate**: The overlap ratio to ensure pits on the boundaries are not cut in half during slicing.
* **Slice Indices (h, w)**: Specifies which sub-slice to load if the image was broken into a grid. 

### Region of Interest (BoundingBox)
* **BoundingBox Mode**: 
  * `Fraction`: Define ROI by percentage from 0.0 to 1.0 (e.g., `x1=0, y1=0, x2=0.5, y2=0.5` captures the top-left quarter).
  * `Absolute`: Define ROI by exact pixel coordinates.
* **x1, y1, x2, y2**: The coordinates of the Bounding Box.

### Display Settings
Adjust how the results are rendered in the Canvas window:
* **centre_size**: Radius of the dot marking the center of the pit.
* **edge_size**: Thickness of the fitted ellipse boundary.
* **font_size / font_edge_size**: Text size for the overlay metrics.

### Action Buttons
* **Reset Image**: Clears the current slice and resets crop fractions.
* **Update Image**: Applies the slice/crop and loads it into memory.
* **Detect Pits**: Runs the GPU algorithm on the current slice.
* **Replace Detect**: Re-detects pits specifically within the updated bounding box and replaces only those local results.
* **Show Display / Save Display**: Opens the Matplotlib canvas or exports the annotated image as a `.png`.
* **Save / Load Status**: Saves or loads all current GUI parameters into a `.sav` file.
* **Batch Detect**: Iterates over every slice (defined by `Batch scale w` and `h`), runs detection on each, and stitches the numerical results (ovals/peaks) and images back together.

---

## 3. Tab 2: Pit Detection Physics & Parameters (`Detection`)

This page contains the core hyperparameters sent to the GPU kernel (`cupy`). 

### Optical & Physical Constraints
* **objective_resolution (m)**: Crucial parameter. Defines meters-per-pixel (e.g., `1.2e-7` for 0.12 $\mu$m/px).
* **radius_min (m) / radius_max (m)**: Filters out dust (too small) or scratches (too large). Only pits within this physical major-axis range are kept.

### Gradient & Peak Finding (Detection phase)
The algorithm relies on analyzing the image gradient to find track centers.
* **bg_value**: Background pixel intensity. Can be set manually or determined automatically if set to `None`.
* **clips_to_bg**: If True, forces darker regions to background level.
* **peak_gaussian_sigma (px)**: Smoothing parameter before finding peaks to eliminate high-frequency noise.
* **peak_charac_len (m)**: The characteristic spatial length of the peak region.
* **peak_thre_offset**: Intensity threshold offset for peak validation. Lowering it detects fainter tracks but increases false positives.
* **peak_closest_dist (m)**: The minimum physical distance allowed between two distinct tracks. Fixes over-segmentation of single large pits.

### Ellipse Fitting (Contour phase)
Once a peak is identified, the algorithm fits an ellipse (eps) to the pit contour.
* **eps_charac_len (m)**: Characteristic length defining the search radius for the pit contour.
* **eps_RMS_thre (%)**: The maximum allowed Root-Mean-Square error for a valid ellipse fit. If a contour is too irregular (e.g., overlapping tracks or a scratch), it will be rejected or flagged.
* **cir_radii_thre_percent**: Defines the threshold at which an ellipse is considered a perfect circle (normal incidence).
* **eps_allowed_center_offest (m)**: How far the geometric center of the fitted ellipse can drift from the initial detected peak.

---

## 4. Interpretation of Results

When you click **Detect Pits** or **Show Display**, an annotated image will appear:
1. **Blue Dots**: Valid, perfectly fitted pits.
2. **Yellow/Cyan Dots**: Overlapping pits or pits requiring multi-fit logic.
3. **Green Ellipses**: The fitted contour.
4. **Text Labels**: Displays the Major and Minor axes in $\mu$m. 

### Integration with Etching Physics (Geant4/FLUKA)
Behind the scenes, the fitted Major ($D_{major}$) and Minor ($D_{minor}$) axes are passed to the lookup tables (`E_mesh.txt`, `theta_mesh.txt`). 
These tables were generated by tracking the Restricted Energy Loss (REL) of protons/ions in CR-39 using Monte Carlo simulations. The detector matches the geometry to the REL curve, extracting the original **Particle Energy (MeV)** and **Incident Angle ($\theta$)**.

---

## 5. Automation & Data Persistence
* The application features an **Autosave** mechanism that continuously writes your parameters to `savFiles_pitDetection/autosave.sav`. If the program crashes, it will reload the exact state upon restart.
* For very large datasets, use **Batch Detect** on the `Slicing` tab. Ensure your `ext_rate` (extension rate) is set > 0 to prevent pits on the edges of sub-images from being sliced in half and lost.
