import time
import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from ..config import DetectionConfig
from ..detection import (
    PIT_STATUS_INFO,
    PitDetector,
    format_pit_status_summary,
    summarize_pit_statuses,
    to_numpy,
)
from ..paths import DEFAULT_AUTOSAVE_PATH, DEFAULT_IMAGE_PATH, DETECTION_WORK_DIR, absolute_path
from ..measurements import (
    INTERNAL_OVAL_AXIS_CONVENTION,
    PHYSICAL_AXIS_CONVENTION,
    fitted_oval_diameters,
    save_measurements_csv,
)
from ..state import load_legacy_pickle, load_state_bundle, save_state_bundle

GUI_SETTING_NAMES = (
    "bbox_mode", "x1_var", "y1_var", "x2_var", "y2_var", "path_var",
    "dw_var", "dh_var", "bdw_var", "bdh_var", "ext_rate_var", "scale_var",
    "sh_var", "sw_var", "bsh_var", "bsw_var", "obj_res_var",
    "radius_min_var", "radius_max_var", "bg_value_var", "clips_to_bg_var",
    "peak_pad_rate_var", "peak_charac_len_var", "peak_thre_offset_var",
    "peak_edge_std_var", "peak_sigma_var", "peak_dist_var", "mode_var",
    "eps_charac_len_var", "eps_product_thre_var", "eps_center_var",
    "eps_rms_var", "cir_radii_thre_var", "mfp_detection_var", "mfp_var",
    "mfp_std_var", "timeout_var", "verbose_var", "use_cuda_var",
    "centre_size", "edge_size", "font_size", "font_edge_size",
)

STATE_ARRAY_NAMES = ("extended_img", "old_ext_img", "display_img", "ovals", "peaks")

class AppLogic:
    def __init__(self, app):
        self.app = app

    def _current_settings(self):
        settings = {}
        for name in GUI_SETTING_NAMES:
            variable = getattr(self.app, name, None)
            if variable is not None and hasattr(variable, "get"):
                settings[name] = variable.get()

        image_path = settings.get("path_var")
        if image_path:
            settings["path_var"] = str(absolute_path(image_path))
        settings["oval_axis_convention"] = INTERNAL_OVAL_AXIS_CONVENTION
        settings["measurement_axis_convention"] = PHYSICAL_AXIS_CONVENTION
        return settings

    def _current_arrays(self):
        return {
            name: value
            for name in STATE_ARRAY_NAMES
            if isinstance((value := getattr(self.app, name, None)), np.ndarray)
        }

    def save_parameters(self, filename=None, include_arrays=True):
        """Save settings to a versioned ``.cr39state`` bundle."""

        if filename is None:
            image_path = absolute_path(self.app.path_var.get() or DEFAULT_IMAGE_PATH)
            filename = filedialog.asksaveasfilename(
                defaultextension=".cr39state",
                filetypes=[("CR-39 state", "*.cr39state"), ("All files", "*.*")],
                initialdir=str(image_path.parent),
                initialfile=f"{image_path.stem}_state.cr39state",
                title="Save Parameters",
            )
        if not filename:
            return None

        arrays = self._current_arrays() if include_arrays else None
        saved_path = save_state_bundle(filename, self._current_settings(), arrays)
        print(f"Parameters saved to {saved_path}")
        return saved_path

    def load_parameters(self, filename=None):
        """Load a safe state bundle or import a trusted legacy ``.sav`` file."""

        if filename is None:
            image_path = absolute_path(self.app.path_var.get() or DEFAULT_IMAGE_PATH)
            filename = filedialog.askopenfilename(
                filetypes=[
                    ("CR-39 state", "*.cr39state"),
                    ("Legacy trusted state", "*.sav"),
                    ("All files", "*.*"),
                ],
                initialdir=str(image_path.parent),
                title="Load Parameters",
            )
        if not filename:
            return None

        source = absolute_path(filename)
        if source.suffix.lower() == ".sav":
            print("WARNING: importing a trusted legacy pickle state file.")
            legacy = load_legacy_pickle(source)
            settings = {name: legacy[name] for name in GUI_SETTING_NAMES if name in legacy}
            arrays = {
                name: legacy[name]
                for name in STATE_ARRAY_NAMES
                if isinstance(legacy.get(name), np.ndarray)
            }
        else:
            settings, arrays = load_state_bundle(source)

        for name, value in settings.items():
            variable = getattr(self.app, name, None)
            if variable is None or not hasattr(variable, "set"):
                continue
            if name == "path_var":
                candidate = absolute_path(value) if value else DEFAULT_IMAGE_PATH
                value = str(candidate if candidate.exists() else DEFAULT_IMAGE_PATH)
            try:
                variable.set(value)
            except (tk.TclError, TypeError, ValueError):
                print(f"Could not load {name}")

        for name, value in arrays.items():
            if name in STATE_ARRAY_NAMES and isinstance(value, np.ndarray):
                setattr(self.app, name, value)

        print(f"Parameters loaded from {source}")
        return source

    def auto_load(self, filepath):
        """Load parameters from file if it exists."""
        source = absolute_path(filepath)
        if source.exists():
            self.load_parameters(source)

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files","*.bmp *.png *.jpg *.jpeg *.tif *.tiff"),("All files","*.* ")]
        )
        if filename:
            self.app.path_var.set(str(absolute_path(filename)))

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
        path = str(absolute_path(self.app.path_var.get()))
        self.app.path_var.set(path)
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

            sliced_dir = (DETECTION_WORK_DIR / "sliced_images").resolve()
            extended_dir = (DETECTION_WORK_DIR / "extended_images").resolve()
            scale_x, scale_y = PitDetector.slice_and_extend_img(
                raw_img, dw, dh, True, str(sliced_dir), str(extended_dir), er
            )

            self.app.scale_var.set(f"Scale: h{int(scale_y)}, w{int(scale_x)}")

            # Load extended slice
            extended_path = (DETECTION_WORK_DIR / "extended_images" / f"{sh}_{sw}.bmp").resolve()
            ext_img = cv2.imread(str(extended_path))
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
                extended_path = (DETECTION_WORK_DIR / "extended_images" / f"{i}_{j}.bmp").resolve()
                ext_img = cv2.imread(str(extended_path))

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

    def _build_detection_config(self):
        er = float(self.app.ext_rate_var.get())
        obj_res = float(self.app.obj_res_var.get())
        extended_pixel_size = obj_res / er
        return DetectionConfig(
            pixel_size=extended_pixel_size,
            radius_range=(
                float(self.app.radius_min_var.get()),
                float(self.app.radius_max_var.get()),
            ),
            mode=None if self.app.mode_var.get() == "None" else self.app.mode_var.get(),
            bg_value=None if self.app.bg_value_var.get() == "None" else int(self.app.bg_value_var.get()),
            clips_to_bg=self.app.clips_to_bg_var.get() == "True",
            peak_pad_rate=float(self.app.peak_pad_rate_var.get()),
            peak_charac_len=float(self.app.peak_charac_len_var.get()),
            peak_thre_offset=float(self.app.peak_thre_offset_var.get()),
            peak_gaussian_sigma=float(self.app.peak_sigma_var.get()),
            peak_edge_std=float(self.app.peak_edge_std_var.get()),
            peak_closest_dist=float(self.app.peak_dist_var.get()),
            eps_charac_len=float(self.app.eps_charac_len_var.get()),
            eps_product_thre_offset=float(self.app.eps_product_thre_var.get()),
            eps_allowed_center_offset=float(self.app.eps_center_var.get()),
            eps_RMS_thre=float(self.app.eps_rms_var.get()),
            cir_radii_thre_percent=float(self.app.cir_radii_thre_var.get()),
            mfp_detection=self.app.mfp_detection_var.get() == "True",
            mfp=None if self.app.mfp_var.get() == "None" else float(self.app.mfp_var.get()),
            mfp_std=float(self.app.mfp_std_var.get()),
            timeout=float(self.app.timeout_var.get()),
            verbose=self.app.verbose_var.get() == "True",
            use_cuda=self.app.use_cuda_var.get() == "True",
        ).validate()

    def detect_pits(self, display=True):
        if self.app.extended_img is None:
            self.log("No extended image found! Please slice & show an image first.\n")
            return

        # The entire cropped image is our ROI => bbox = [0,0, width, height]
        h, w = self.app.extended_img.shape[:2]
        bbox = [0, 0, w, h]

        try:
            config = self._build_detection_config()
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            self.log(f"Parameter error: {exc}\n")
            return
        extended_pixel_size = config.pixel_size

        img_bgr = self.app.extended_img.copy()

        t0 = time.time()

        detector = PitDetector(**config.as_detector_kwargs())
        backend_message = f"Backend: {detector.backend_name}"
        if detector.cuda_fallback_reason:
            backend_message += f" (CUDA fallback: {detector.cuda_fallback_reason})"
        self.log(backend_message + "\n")

        # Run detection
        peaks, fitted_ovals = detector.detect(
            img_bgr=img_bgr,
            bbox=bbox,
        )

        dt = time.time() - t0

        status_counts = summarize_pit_statuses(fitted_ovals)
        valid_count = status_counts.get(0, 0)
        self.log(
            f"Detection time: {dt:.3f}s\n"
            f"Found {len(peaks)} peaks; {valid_count} valid pits\n"
            f"Status summary: {format_pit_status_summary(status_counts)}\n"
        )
        fit_failures = detector.last_diagnostics.get("fit_failure_counts", {})
        if any(fit_failures.values()):
            self.log(
                "Fit failure details: "
                + ", ".join(f"{name}={count}" for name, count in fit_failures.items())
                + "\n"
            )
        center_fallback_count = detector.last_diagnostics.get("center_fallback_count", 0)
        if center_fallback_count:
            self.log(
                "Center fallback: "
                f"{center_fallback_count} fitted centers exceeded eps_allowed_center_offset; "
                "their detected peak centers were used instead.\n"
            )

        # Save the results (ensure they are numpy arrays for compatibility with np.append etc.)
        self.app.peaks = peaks.get() if hasattr(peaks, 'get') else peaks.copy()
        self.app.ovals = fitted_ovals.get() if hasattr(fitted_ovals, 'get') else fitted_ovals.copy()

        # Display the results
        if display:
            self.display_results(img_bgr, peaks, fitted_ovals, extended_pixel_size)

    def display_results(self, img_bgr, peaks, fitted_ovals, extended_pixel_size):
        # display the image
        display_img = img_bgr.copy()
        peaks = to_numpy(peaks)
        fitted_ovals = to_numpy(fitted_ovals)
        diameters_um = fitted_oval_diameters(
            fitted_ovals,
            extended_pixel_size,
        ) * 1e6
        # For thickness scaling
        centre_size = int(self.app.centre_size.get())
        edge_size   = int(self.app.edge_size.get())
        font_size   = float(self.app.font_size.get())
        font_edge_size = int(self.app.font_edge_size.get())

        for idx, (peak, oval) in enumerate(zip(peaks, fitted_ovals)):
            peakx, peaky = peak
            label = oval[5]

            status = PIT_STATUS_INFO.get(int(label), {"short": f"L{int(label)}", "bgr": (255, 255, 255)})
            cv2.circle(display_img, (int(peakx), int(peaky)), centre_size, status["bgr"], -1)

            if label == 0:
                cx, cy, a_val, b_val, theta_val = oval[:5]
                major_diameter_um, minor_diameter_um = diameters_um[idx]
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
                    f"D=({major_diameter_um:.2f},{minor_diameter_um:.2f})um",
                    (int(peakx), int(peaky)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 255), font_edge_size, cv2.LINE_AA
                )
            else:
                cv2.putText(
                    display_img,
                    f'id:{idx}, {status["short"]}',
                    (int(peakx), int(peaky)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, status["bgr"], font_edge_size, cv2.LINE_AA
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
            status_counts = summarize_pit_statuses(fitted_ovals)
            summary_frame = ttk.LabelFrame(self.app.result_window, text="Detection summary")
            summary_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            ttk.Label(
                summary_frame,
                text=f"Total peaks: {len(peaks)} | {format_pit_status_summary(status_counts)}",
            ).pack(anchor=tk.W, padx=5, pady=2)
            for code in (0, -1, -2, -3, -4):
                count = status_counts.get(code, 0)
                if count:
                    info = PIT_STATUS_INFO[code]
                    ttk.Label(
                        summary_frame,
                        text=f"{info['short']} ({code}): {count} - {info['description']}",
                    ).pack(anchor=tk.W, padx=10)

            # 3) Create a Matplotlib Figure and Axes
            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)

            # 4) Show the image in the Axes
            ax.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
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

    def show_current_results(self):
        if self.app.extended_img is None or not isinstance(self.app.peaks, np.ndarray) or not isinstance(self.app.ovals, np.ndarray):
            self.log("No detection results are available. Run Detect Pits first.\n")
            return
        self.display_results(
            self.app.extended_img,
            self.app.peaks,
            self.app.ovals,
            float(self.app.obj_res_var.get()) / float(self.app.ext_rate_var.get()),
        )

    def save_results(self, filename=None):
        """Save the results to a .png file."""
        if not isinstance(self.app.display_img, np.ndarray):
            self.log("No annotated display is available. Run Detect Pits first.\n")
            return None
        image_path = absolute_path(self.app.path_var.get() or DEFAULT_IMAGE_PATH)
        if filename is None:
            # 2) Ask user for the filename
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
                initialdir=str(image_path.parent),
                initialfile=f"{image_path.stem}_results.png",
                title="Save Detection Results"
            )
        if not filename:
            return  # user canceled

        output_path = absolute_path(filename)
        cv2.imwrite(str(output_path), self.app.display_img)
        return output_path

    def save_measurements(self, filename=None):
        """Export internal fits and physical full diameters to an auditable CSV."""
        if not isinstance(self.app.ovals, np.ndarray):
            self.log("No detection measurements are available. Run Detect Pits first.\n")
            return None
        image_path = absolute_path(self.app.path_var.get() or DEFAULT_IMAGE_PATH)
        if filename is None:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV measurement table", "*.csv"), ("All files", "*.*")],
                initialdir=str(image_path.parent),
                initialfile=f"{image_path.stem}_measurements.csv",
                title="Save Pit Measurements",
            )
        if not filename:
            return None
        pixel_size_m = (
            float(self.app.obj_res_var.get()) / float(self.app.ext_rate_var.get())
        )
        output_path = save_measurements_csv(
            filename,
            self.app.ovals,
            pixel_size_m,
            accepted_only=False,
        )
        self.log(
            f"Measurements saved to {output_path}\n"
            "CSV physical axes are full diameters in metres; internal semiaxes in pixels are retained.\n"
        )
        return output_path

    def auto_save(self, filepath=DEFAULT_AUTOSAVE_PATH):
        """Auto-save settings only to the per-user runtime directory."""
        saved_path = self.save_parameters(filename=filepath, include_arrays=False)
        print(f"Parameters auto-saved to {saved_path}")

    def log(self, msg):
        self.app.log_text.insert(tk.END, msg)
        self.app.log_text.see(tk.END)
