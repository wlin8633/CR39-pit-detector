import os
import time
import tkinter as tk
from tkinter import filedialog
import pickle
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from pitDetection import PitDetector

class AppLogic:
    def __init__(self, app):
        self.app = app

    def auto_save(self, filepath):
        """Save all parameters to a .sav (or .pkl) file automatically."""
        self.save_parameters(filepath)

    def save_parameters(self, filename=None):
        """Save all parameters to a .sav file (using pickle)."""

        # 1) Gather current parameter values into a dictionary
        
        data = {}
        for key in vars(self.app):
            attr = getattr(self.app, key)
            if isinstance(attr, tk.Text):
                data[key] = attr.get("1.0", "end-1c")
            else:
                try:
                    data[key] = attr.get()
                except AttributeError:
                    if isinstance(attr, np.ndarray):
                        data[key] = attr
                    else:
                        data[key] = None
            
        if filename is None:
            # 2) Ask user for the filename
            initial_filename = os.path.splitext(self.app.path_var.get())[0] + "_state.sav"
            initial_dir = os.path.dirname(self.app.path_var.get())
            filename = filedialog.asksaveasfilename(
                defaultextension=".sav",
                filetypes=[("Save Files", "*.sav"), ("All Files", "*.* ")],
                initialdir=initial_dir,
                initialfile=initial_filename,
                title="Save Parameters"
            )
        if not filename:
            return  # user canceled

        # 3) Use pickle to dump the data dictionary to file
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"Parameters saved to {filename}")

    def load_parameters(self, filename=None):
        """Load parameters from a .sav file (using pickle) and update the GUI."""
        # 1) Ask user for the filename
        initial_filename = os.path.splitext(self.app.path_var.get())[0] + "_state.sav"
        initial_dir = os.path.dirname(self.app.path_var.get())
        if filename is None:
            filename = filedialog.askopenfilename(
                defaultextension=".sav",
                filetypes=[("Save Files", "*.sav"), ("All Files", "*.* ")],
                initialdir=initial_dir,
                initialfile=initial_filename,
                title="Load Parameters"
            )
        if not filename:
            return  # user canceled

        # 2) Load data from the file
        with open(filename, "rb") as f:
            data = pickle.load(f)

        # 3) Update each variable in your app
        for key, value in data.items():
            if key in vars(self.app):
                var = getattr(self.app, key)
                try:
                    var.set(value)
                except (AttributeError, ValueError):
                    if type(value) == np.ndarray:
                        setattr(self.app, key, value)
                    else:
                        print(f"Could not load {key}")
                        pass

        print(f"Parameters loaded from {filename}")
        
    def auto_load(self, filepath):
        """Load parameters from file if it exists."""
        if not os.path.exists(filepath):
            print(f"No autosave file found at {filepath}")
            return  # no file to load, do nothing
        
        self.load_parameters(filepath)

    def auto_save(self, filepath):
        """Save all parameters to a .sav file (using pickle)."""
        self.save_parameters(filepath)
    
    def auto_save(self, filepath):
        """Save all parameters to a .sav (or .pkl) file automatically."""
        self.save_parameters(filename=filepath)
    
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files","*.bmp *.png *.jpg *.jpeg *.tif *.tiff"),("All files","*.* ")]
        )
        if filename:
            self.app.path_var.set(filename)

    def reset_image(self, scale=None):
        
        self.app.old_ext_img = None
        
        dw = int(self.app.dw_var.get())
        dh = int(self.app.dh_var.get())
        er = int(self.app.ext_rate_var.get())
        
        if self.app.bbox_mode.get() == "Fraction":
            self.app.x1_var.set("0")
            self.app.y1_var.set("0")
            self.app.x2_var.set("1")
            self.app.y2_var.set("1")
        else:
            self.app.x1_var.set("0")
            self.app.y1_var.set("0")
            self.app.x2_var.set(f"{dw*er}")
            self.app.y2_var.set(f"{dh*er}")
            
        self.app.extended_img = None
        self.update_image(scale)

    def open_canvas_window(self):
        self.app.show_canvas_window()

    def update_image(self, scale=None):
        """
        1) slice_and_extend if needed
        2) crop extended_img by bounding box
        3) push the final image to the separate Canvas window
        """
        path = self.app.path_var.get()
        dw = int(self.app.dw_var.get())
        dh = int(self.app.dh_var.get())
        er = int(self.app.ext_rate_var.get())
        sh = self.app.sh_var.get()
        sw = self.app.sw_var.get()

        x1_val = float(self.app.x1_var.get())
        y1_val = float(self.app.y1_var.get())
        x2_val = float(self.app.x2_var.get())
        y2_val = float(self.app.y2_var.get())
        bbox_mode = self.app.bbox_mode.get()
        
        # If no extended_img loaded, do slice_and_extend
        if self.app.extended_img is None:
            raw_img = cv2.imread(fr"{path}")
            if raw_img is None:
                print("Could not load:", path)
                return
            
            if scale is not None:
                raw_img_shape = raw_img.shape[:2]
                dw = int(np.ceil(raw_img_shape[1] / scale[1]))
                dh = int(np.ceil(raw_img_shape[0] / scale[0]))
                self.app.bdw_var.set(str(dw))
                self.app.bdh_var.set(str(dh))
            
            scale_x, scale_y = PitDetector.slice_and_extend_img(
                raw_img, dw, dh, True, r".\savFiles_pitDetection\sliced_images", r".\savFiles_pitDetection\extended_images", er
            )
            
            self.app.scale_var.set(f"Scale: h{int(scale_y)}, w{int(scale_x)}")
            
            # Load extended slice
            extended_path = fr".\savFiles_pitDetection\extended_images\{sh}_{sw}.bmp"
            ext_img = cv2.imread(extended_path)
            if ext_img is None:
                print("Could not load extended slice:", extended_path)
                return
            self.app.extended_img = ext_img

        # Now we have self.app.extended_img
        ext_img = self.app.extended_img
        h, w = ext_img.shape[:2]

        # Convert bounding box
        if bbox_mode == "Fraction":
            x1 = int(np.clip(x1_val, 0, 1) * w)
            y1 = int(np.clip(y1_val, 0, 1) * h)
            x2 = int(np.clip(x2_val, 0, 1) * w)
            y2 = int(np.clip(y2_val, 0, 1) * h)
        else:
            x1 = int(np.clip(x1_val, 0, w))
            y1 = int(np.clip(y1_val, 0, h))
            x2 = int(np.clip(x2_val, 0, w))
            y2 = int(np.clip(y2_val, 0, h))

        # Crop
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        x2 = min(x2, w)
        y2 = min(y2, h)

        cropped = ext_img[y1:y2, x1:x2]
        self.app.extended_img = cropped

        # Show in separate window
        if not self.app.image_window:
            self.open_canvas_window()
        if self.app.image_window and tk.Toplevel.winfo_exists(self.app.image_window):
            self.app.image_window.show_image(cropped)
        else:
            print("Canvas window is not available.")

    def batch_detect(self):
        # 1) Get the desired width and height
        bsw = int(self.app.bsw_var.get())
        bsh = int(self.app.bsh_var.get())
        er = int(self.app.ext_rate_var.get())
        
        self.reset_image(scale=(bsh, bsw))
        bdh = int(self.app.bdh_var.get())
        bdw = int(self.app.bdw_var.get())
        
        # 3) loop to detect the sliced images
        stitch_extended_img = np.zeros((bdh*er*bsh, bdw*er*bsw, 3), dtype=np.uint8)
        total_peaks = []
        total_ovals = []
        
        self.app.verbose_var.set("False")
        t0 = time.time()
        
        for i in range(bsh):
            for j in range(bsw):
                extended_path = fr".\savFiles_pitDetection\extended_images\{i}_{j}.bmp"
                ext_img = cv2.imread(extended_path)
                
                if ext_img is None:
                    print("Could not load extended slice:", extended_path)
                    continue
                self.app.extended_img = ext_img
                
                # 4) Detect the pits
                self.detect_pits(display=False)
                
                # update the position
                self.app.peaks[:, 0] += j*bdw*er
                self.app.peaks[:, 1] += i*bdh*er
                
                self.app.ovals[:, 0] += j*bdw*er
                self.app.ovals[:, 1] += i*bdh*er
                
                # 5) Save the results
                stitch_extended_img[i*bdh*er:(i+1)*bdh*er, j*bdw*er:(j+1)*bdw*er] = self.app.extended_img
                
                total_peaks = np.append(total_peaks, self.app.peaks).reshape(-1, 2)
                total_ovals = np.append(total_ovals, self.app.ovals).reshape(-1, 6)
                
                print(f"Progress:                     {i+1}/{bsh}, {j+1}/{bsw}", end="\r")
        
        # 6) Save the stitched images
        self.app.extended_img = stitch_extended_img
        self.app.peaks = total_peaks
        self.app.ovals = total_ovals
        
        self.app.verbose_var.set("True")
        dt = time.time() - t0
        print(f"\nBatch detection time: {dt:.3f}s\nFound {len(self.app.peaks)} peaks, {np.sum(self.app.ovals[:, 5] == 0)} ovals\n")
        
        self.display_results(stitch_extended_img, total_peaks, total_ovals, float(self.app.obj_res_var.get()) / float(self.app.ext_rate_var.get()))
        self.reset_image()
        
    def replace_detection(self):
        self.app.old_ext_img = self.app.extended_img.copy()
        
        x1 = int(float(self.app.x1_var.get()) * self.app.old_ext_img.shape[1])
        y1 = int(float(self.app.y1_var.get()) * self.app.old_ext_img.shape[0])
        x2 = int(float(self.app.x2_var.get()) * self.app.old_ext_img.shape[1])
        y2 = int(float(self.app.y2_var.get()) * self.app.old_ext_img.shape[0])
        
        old_peaks = self.app.peaks.copy()
        old_ovals = self.app.ovals.copy()
        
        # detect the pits by the current image
        self.update_image()
        self.detect_pits(display=False)
        self.app.extended_img = self.app.old_ext_img
        new_peaks = self.app.peaks.copy()
        new_ovals = self.app.ovals.copy()
        new_peaks[:, 0] += x1
        new_peaks[:, 1] += y1
        new_ovals[:, 0] += x1
        new_ovals[:, 1] += y1
        
        # replace the old results within the range of the new results
        old_cond = (old_peaks[:, 0] <= x1) | (old_peaks[:, 0] >= x2) | (old_peaks[:, 1] <= y1) | (old_peaks[:, 1] >= y2)
        old_ovals = old_ovals[old_cond]
        old_peaks = old_peaks[old_cond]
        
        new_cond = (new_peaks[:, 0] >= x1) & (new_peaks[:, 0] <= x2) & (new_peaks[:, 1] >= y1) & (new_peaks[:, 1] <= y2)
        new_ovals = new_ovals[new_cond]
        new_peaks = new_peaks[new_cond]
        
        self.app.peaks = np.append(old_peaks, new_peaks).reshape(-1, 2)
        self.app.ovals = np.append(old_ovals, new_ovals).reshape(-1, 6)
        
        # display the results
        self.display_results(self.app.extended_img, self.app.peaks, self.app.ovals, float(self.app.obj_res_var.get()) / float(self.app.ext_rate_var.get()))

    def detect_pits(self, display=True):
        if self.app.extended_img is None:
            self.log("No extended image found! Please slice & show an image first.\n")
            return

        # The entire cropped image is our ROI => bbox = [0,0, width, height]
        h, w = self.app.extended_img.shape[:2]
        bbox = [0, 0, w, h]

        er = float(self.app.ext_rate_var.get())
        obj_res = float(self.app.obj_res_var.get())
        extended_pixel_size = obj_res / er

        radius_min = float(self.app.radius_min_var.get())
        radius_max = float(self.app.radius_max_var.get())
        radius_range = [radius_min, radius_max]

        bg_value = None if self.app.bg_value_var.get() == "None" else int(self.app.bg_value_var.get())
        clips_to_bg = self.app.clips_to_bg_var.get() == "True"
        peak_pad_rate = float(self.app.peak_pad_rate_var.get())
        peak_charac_len = float(self.app.peak_charac_len_var.get())
        peak_thre_offset = float(self.app.peak_thre_offset_var.get())
        peak_edge_std = float(self.app.peak_edge_std_var.get())
        peak_gaussian_sigma = float(self.app.peak_sigma_var.get())
        peak_closest_dist = float(self.app.peak_dist_var.get())
        mode = None if self.app.mode_var.get() == "None" else self.app.mode_var.get()
        eps_charac_len = float(self.app.eps_charac_len_var.get())
        eps_product_thre_offset = float(self.app.eps_product_thre_var.get())
        eps_allowed_center_offest = float(self.app.eps_center_var.get())
        eps_RMS_thre = float(self.app.eps_rms_var.get())
        cir_radii_thre_percent = float(self.app.cir_radii_thre_var.get())
        mfp_detection = self.app.mfp_detection_var.get() == "True"
        mfp = None if self.app.mfp_var.get() == "None" else int(self.app.mfp_var.get())
        mfp_std = float(self.app.mfp_std_var.get())
        detection_timeout = float(self.app.timeout_var.get())
        verbose = self.app.verbose_var.get() == "True"

        img_bgr = self.app.extended_img.copy()

        t0 = time.time()
        
        detector = PitDetector(
            pixel_size=extended_pixel_size,
            radius_range=radius_range,
            mode=mode,
            bg_value=bg_value,
            clips_to_bg=clips_to_bg,
            
            peak_pad_rate=peak_pad_rate,
            peak_charac_len=peak_charac_len,
            peak_thre_offset=peak_thre_offset,
            peak_gaussian_sigma=peak_gaussian_sigma,
            peak_edge_std=peak_edge_std,
            peak_closest_dist=peak_closest_dist,
            eps_charac_len=eps_charac_len,
            eps_product_thre_offset=eps_product_thre_offset,
            eps_allowed_center_offest=eps_allowed_center_offest,
            eps_RMS_thre=eps_RMS_thre,
            cir_radii_thre_percent=cir_radii_thre_percent,
            mfp_detection=mfp_detection,
            mfp=mfp,
            mfp_std=mfp_std,
            timeout=detection_timeout,
            verbose=verbose,
        )
        
        # Run detection
        peaks, fitted_ovals = detector.detect(
            img_bgr=img_bgr,
            bbox=bbox,
        )
        
        dt = time.time() - t0

        self.log(f"Detection time: {dt:.3f}s\nFound {len(peaks)} peaks, {len(fitted_ovals[:, 5] == 0)} ovals\n")
        
        # Save the results
        self.app.peaks = peaks.copy()
        self.app.ovals = fitted_ovals.copy()
        
        # Display the results
        if display:
            self.display_results(img_bgr, peaks, fitted_ovals, extended_pixel_size)
        
    def display_results(self, img_bgr, peaks, fitted_ovals, extended_pixel_size):
        # display the image
        display_img = img_bgr.copy()
        # For thickness scaling
        centre_size = int(self.app.centre_size.get())
        edge_size   = int(self.app.edge_size.get())
        font_size   = float(self.app.font_size.get())
        font_edge_size = int(self.app.font_edge_size.get())

        for idx, (peak, oval) in enumerate(zip(peaks, fitted_ovals)):
            peakx, peaky = peak
            label = oval[5]
            
            if label == 0:
                cv2.circle(display_img, (int(peakx), int(peaky)), centre_size, (255, 0, 0), -1)
            else:
                cv2.circle(display_img, (int(peakx), int(peaky)), centre_size, (0, 255, 255), -1)

            if label == 0:
                cx, cy, a_val, b_val, theta_val = oval[:5]
                cv2.ellipse(
                    display_img,
                    (int(cx), int(cy)),
                    (int(a_val), int(b_val)),
                    int(np.degrees(theta_val)), 0, 360,
                    (0, 255, 0), edge_size
                )
                cv2.putText(
                    display_img,
                    # f"{idx}",
                    f"({a_val*extended_pixel_size*1e6:.2f},{b_val*extended_pixel_size*1e6:.2f})um",
                    (int(peakx), int(peaky)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 255), font_edge_size, cv2.LINE_AA
                )
            elif label == -1:
                cv2.putText(
                    display_img,
                    f'id:{idx}, l1',
                    (int(peakx), int(peaky)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 0), font_edge_size, cv2.LINE_AA
                )
            elif label == -2:
                cv2.putText(
                    display_img,
                    f'id:{idx}, l2',
                    (int(peakx), int(peaky)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 255), font_edge_size, cv2.LINE_AA
                )
            elif label == -3:
                cv2.putText(
                    display_img,
                    f'id:{idx}, l3',
                    (int(peakx), int(peaky)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), font_edge_size, cv2.LINE_AA
                )
            elif label == -4:
                cv2.putText(
                    display_img,
                    f'id:{idx}, l4',
                    (int(peakx), int(peaky)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_edge_size, cv2.LINE_AA
                )
                
        self.app.display_img = display_img
        
        if self.app.result_window is None or not tk.Toplevel.winfo_exists(self.app.result_window):
            # Create a new Toplevel window
            self.app.result_window = tk.Toplevel()
            self.app.result_window.title("Detection Results")

        else:
            # Clear the old canvas
            for widget in self.app.result_window.winfo_children():
                widget.destroy()
            
        if self.app.result_window and tk.Toplevel.winfo_exists(self.app.result_window):
            
            # 3) Create a Matplotlib Figure and Axes
            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)

            # 4) Show the image in the Axes
            ax.imshow(display_img)
            ax.axis("off")  # hide axes if you want

            # 5) Place the FigureCanvasTkAgg in the Tk window
            canvas = FigureCanvasTkAgg(fig, master=self.app.result_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # 6) Add the Matplotlib Navigation Toolbar for Zoom/Pan
            toolbar = NavigationToolbar2Tk(canvas, self.app.result_window)
            toolbar.update()
            canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
        else:
            print("Canvas window is not available.")
            
    def save_results(self, filename=None):
        """Save the results to a .png file."""
        initial_filename = os.path.splitext(self.app.path_var.get())[0] + "_results.png"
        initial_dir = os.path.dirname(self.app.path_var.get())
        if filename is None:
            # 2) Ask user for the filename
            filename = filedialog.asksaveasfilename(
                defaultextension=".sav",
                filetypes=[("Save Files", "*.png"), ("All Files", "*.* ")],
                initialdir=initial_dir,
                initialfile=initial_filename,
                title="Save Parameters"
            )
        if not filename:
            return  # user canceled
        
        cv2.imwrite(filename, cv2.cvtColor(self.app.display_img, cv2.COLOR_BGR2RGB))
    
    def auto_save(self, filepath=r".\savFiles_pitDetection\autosave.sav"):
        """Auto-save the current parameters to a .sav file."""
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        
        self.save_parameters(filename=filepath)
        print(f"Parameters auto-saved to {filepath}")
        
    def log(self, msg):
        self.app.log_text.insert(tk.END, msg)
        self.app.log_text.see(tk.END)
