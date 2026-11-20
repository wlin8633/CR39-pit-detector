import math
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
    parse_extension_rate,
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

class AppLogic:
    def __init__(self, app):
        self.app = app
        self._image_key = None
        self._full_tile = None
        self._selected_bbox = None
        self._result_image = None
        self._result_key = None
        self._result_bbox = None
        self._result_pixel_size = None

    def _requested_image_key(self):
        path = str(absolute_path(self.app.path_var.get()))
        width, height = int(self.app.dw_var.get()), int(self.app.dh_var.get())
        row, col = int(self.app.sh_var.get()), int(self.app.sw_var.get())
        rate = parse_extension_rate(self.app.ext_rate_var.get())
        if min(width, height) < 1 or min(row, col) < 0:
            raise ValueError("Slice dimensions must be positive and indices non-negative")
        return path, width, height, row, col, rate

    def _check_loaded_image(self):
        key = self._requested_image_key()
        if self._image_key is not None and key != self._image_key:
            raise ValueError(
                "Image settings changed. Press Reset Image, then set the bounding box again if needed."
            )
        return key

    def _invalidate_results(self):
        self._result_image = None
        self._result_key = None
        self._result_bbox = None
        self._result_pixel_size = None
        self.app.ovals = None
        self.app.peaks = None
        self.app.display_img = None

    def _check_result_scale(self):
        key = self._check_loaded_image()
        if self._result_image is None:
            raise ValueError("No detection results are available. Run Detect Pits first.")
        if key != self._result_key:
            raise ValueError("Image settings changed. Press Reset Image and run Detect Pits again.")
        if self._pixel_size(key[-1]) != self._result_pixel_size:
            raise ValueError("Objective resolution changed. Run Detect Pits again before using these results.")
        return self._result_pixel_size

    def _pixel_size(self, rate):
        resolution = float(self.app.obj_res_var.get())
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError("Objective resolution must be a finite positive value in metres per raw pixel")
        return resolution / rate

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
            for name, value in {
                "full_tile": self._full_tile,
                "result_image": self._result_image,
                "display_img": self.app.display_img,
                "ovals": self.app.ovals,
                "peaks": self.app.peaks,
            }.items()
            if isinstance(value, np.ndarray)
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

        settings = self._current_settings()
        arrays = None
        if include_arrays:
            arrays = self._current_arrays()
            settings["image_key"] = self._image_key
            settings["selected_bbox"] = self._selected_bbox
            settings["result_key"] = self._result_key
            settings["result_bbox"] = self._result_bbox
            settings["result_pixel_size_m"] = self._result_pixel_size
        saved_path = save_state_bundle(filename, settings, arrays)
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
            arrays = {name: legacy[name] for name in ("display_img", "ovals") if isinstance(legacy.get(name), np.ndarray)}
        else:
            settings, arrays = load_state_bundle(source)

        for name in GUI_SETTING_NAMES:
            if name not in settings:
                continue
            value = settings[name]
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

        self._invalidate_results()
        self._image_key = None
        self._full_tile = None
        self._selected_bbox = None
        self.app.extended_img = None
        if settings.get("image_key") and "full_tile" in arrays and settings.get("selected_bbox"):
            self._image_key = tuple(settings["image_key"])
            self._full_tile = arrays["full_tile"]
            self._selected_bbox = tuple(settings["selected_bbox"])
            x1, y1, x2, y2 = self._selected_bbox
            self.app.extended_img = self._full_tile[y1:y2, x1:x2]
        if (
            settings.get("result_key")
            and settings.get("result_pixel_size_m")
            and {"result_image", "peaks", "ovals"} <= arrays.keys()
        ):
            self._result_image = arrays["result_image"]
            self._result_key = tuple(settings["result_key"])
            self._result_bbox = tuple(settings["result_bbox"]) if settings.get("result_bbox") else None
            self._result_pixel_size = settings["result_pixel_size_m"]
            self.app.peaks = arrays["peaks"]
            self.app.ovals = arrays["ovals"]
            self.app.display_img = arrays.get("display_img")
        elif "ovals" in arrays or source.suffix.lower() == ".sav":
            self.app.display_img = arrays.get("display_img")
            print("Loaded legacy settings. Use Update Image and Detect Pits to recreate measurements.")

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

    def reset_image(self):
        try:
            key = self._requested_image_key()
            self._prepare_tile(key)
        except (OSError, TypeError, ValueError, cv2.error) as exc:
            self.log(f"Image error: {exc}\n")
            return

        height, width = self._full_tile.shape[:2]
        self._invalidate_results()

        if self.app.bbox_mode.get() == "Fraction":
            self.app.x1_var.set("0")
            self.app.y1_var.set("0")
            self.app.x2_var.set("1")
            self.app.y2_var.set("1")
        else:
            self.app.x1_var.set("0")
            self.app.y1_var.set("0")
            self.app.x2_var.set(str(width))
            self.app.y2_var.set(str(height))

        self.update_image()

    def _prepare_tile(self, key):
        path, dw, dh, sh, sw, er = key
        raw_img = cv2.imread(path)
        if raw_img is None:
            raise ValueError(f"Could not load image: {path}")
        if sh * dh >= raw_img.shape[0] or sw * dw >= raw_img.shape[1]:
            raise ValueError("Selected slice is outside the raw image")
        sliced_dir = (DETECTION_WORK_DIR / "sliced_images").resolve()
        extended_dir = (DETECTION_WORK_DIR / "extended_images").resolve()
        scale_x, scale_y = PitDetector.slice_and_extend_img(
            raw_img, dw, dh, True, str(sliced_dir), str(extended_dir), er
        )
        extended_path = extended_dir / f"{sh}_{sw}.bmp"
        ext_img = cv2.imread(str(extended_path))
        if ext_img is None:
            raise ValueError(f"Could not load extended slice: {extended_path}")
        self.app.path_var.set(path)
        self.app.scale_var.set(f"Scale: h{scale_y}, w{scale_x}")
        self._full_tile = ext_img
        self._image_key = key

    def open_canvas_window(self):
        self.app.show_canvas_window()

    def update_image(self):
        """Apply a bounding box relative to the full extended tile."""
        try:
            key = self._check_loaded_image()
            if self._full_tile is None:
                self._prepare_tile(key)
            bbox = self._requested_bbox()
        except (OSError, TypeError, ValueError, cv2.error) as exc:
            self.log(f"Image error: {exc}\n")
            return

        self._selected_bbox = bbox
        x1, y1, x2, y2 = bbox
        cropped = self._full_tile[y1:y2, x1:x2]
        self.app.extended_img = cropped

        if not self.app.image_window:
            self.open_canvas_window()
        if self.app.image_window and tk.Toplevel.winfo_exists(self.app.image_window):
            self.app.image_window.show_image(cropped)

    def _requested_bbox(self):
        height, width = self._full_tile.shape[:2]
        values = [float(var.get()) for var in (
            self.app.x1_var, self.app.y1_var, self.app.x2_var, self.app.y2_var,
        )]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Bounding box coordinates must be finite")
        x1, y1, x2, y2 = values
        if self.app.bbox_mode.get() == "Fraction":
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        x1, x2 = sorted(int(np.clip(x, 0, width)) for x in (x1, x2))
        y1, y2 = sorted(int(np.clip(y, 0, height)) for y in (y1, y2))
        if x1 == x2 or y1 == y2:
            raise ValueError("Bounding box must cover at least one pixel in each direction")
        return x1, y1, x2, y2

    def batch_detect(self):
        try:
            key = self._check_loaded_image()
            config = self._build_detection_config(key[-1])
            columns, rows = int(self.app.bsw_var.get()), int(self.app.bsh_var.get())
            raw = cv2.imread(key[0])
            if raw is None:
                raise ValueError(f"Could not load image: {key[0]}")
            height, width = raw.shape[:2]
            if not (1 <= columns <= width and 1 <= rows <= height):
                raise ValueError("Batch grid dimensions must be positive and within the raw image")
            dw, dh = math.ceil(width / columns), math.ceil(height / rows)
            rate = key[-1]
            sliced_dir = DETECTION_WORK_DIR / "batch" / "sliced_images"
            extended_dir = DETECTION_WORK_DIR / "batch" / "extended_images"
            columns, rows = PitDetector.slice_and_extend_img(
                raw, dw, dh, True, str(sliced_dir), str(extended_dir), rate,
            )

            options = config.as_detector_kwargs()
            options["verbose"] = False
            detector = PitDetector(**options)
            stitched = np.empty((round(height * rate), round(width * rate), 3), dtype=np.uint8)
            total_peaks, total_ovals = [], []
            for row in range(rows):
                for col in range(columns):
                    path = extended_dir / f"{row}_{col}.bmp"
                    tile = cv2.imread(str(path))
                    if tile is None:
                        raise ValueError(f"Could not load extended slice: {path}")
                    peaks, ovals = map(to_numpy, detector.detect(tile))
                    x, y = round(col * dw * rate), round(row * dh * rate)
                    peaks[:, :2] += (x, y)
                    ovals[:, :2] += (x, y)
                    h, w = tile.shape[:2]
                    stitched[y:y + h, x:x + w] = tile
                    total_peaks.append(peaks)
                    total_ovals.append(ovals)
            peaks, ovals = np.concatenate(total_peaks), np.concatenate(total_ovals)
        except (OSError, TypeError, ValueError, cv2.error) as exc:
            self.log(f"Batch detection error: {exc}\n")
            return

        self.app.bdw_var.set(str(dw))
        self.app.bdh_var.set(str(dh))
        self._remember_results(stitched, key, None, config.pixel_size, peaks, ovals)
        self.log(f"Batch detection: {len(peaks)} peaks | {format_pit_status_summary(summarize_pit_statuses(ovals))}\n")
        self.display_results(stitched, peaks, ovals, config.pixel_size)

    def replace_detection(self):
        try:
            pixel_size = self._check_result_scale()
            if self._result_bbox is None or self._full_tile is None:
                raise ValueError("Replace Detect requires a previous single-image detection.")
            bx, by, bx2, by2 = self._result_bbox
            x1, y1, x2, y2 = self._requested_bbox()
            if not (bx <= x1 < x2 <= bx2 and by <= y1 < y2 <= by2):
                raise ValueError("Replacement bounding box must be within the previously detected region.")
            config = self._build_detection_config(self._image_key[-1])
            roi = self._full_tile[y1:y2, x1:x2]
            new_peaks, new_ovals = self._run_detection(roi, config)
        except (TypeError, ValueError, cv2.error) as exc:
            self.log(f"Replace detection error: {exc}\n")
            return

        x, y = x1 - bx, y1 - by
        new_peaks[:, :2] += (x, y)
        new_ovals[:, :2] += (x, y)
        x_end, y_end = x2 - bx, y2 - by

        def inside(peaks):
            px, py = peaks.T
            return (x <= px) & (px < x_end) & (y <= py) & (py < y_end)

        keep = ~inside(self.app.peaks)
        insert = inside(new_peaks)
        peaks = np.concatenate((self.app.peaks[keep], new_peaks[insert]))
        ovals = np.concatenate((self.app.ovals[keep], new_ovals[insert]))
        self._remember_results(self._result_image, self._image_key, self._result_bbox, pixel_size, peaks, ovals)
        self.display_results(self._result_image, peaks, ovals, pixel_size)

    def _build_detection_config(self, rate):
        extended_pixel_size = self._pixel_size(rate)
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

        try:
            key = self._check_loaded_image()
            config = self._build_detection_config(key[-1])
            img_bgr = self.app.extended_img
            peaks, fitted_ovals = self._run_detection(img_bgr, config)
        except (TypeError, ValueError, cv2.error) as exc:
            self.log(f"Parameter error: {exc}\n")
            return

        self._remember_results(img_bgr, key, self._selected_bbox, config.pixel_size, peaks, fitted_ovals)
        if display:
            self.display_results(img_bgr, peaks, fitted_ovals, config.pixel_size)
        return True

    def _run_detection(self, img_bgr, config):
        t0 = time.time()

        detector = PitDetector(**config.as_detector_kwargs())
        backend_message = f"Backend: {detector.backend_name}"
        if detector.cuda_fallback_reason:
            backend_message += f" (CUDA fallback: {detector.cuda_fallback_reason})"
        self.log(backend_message + "\n")

        # Run detection
        peaks, fitted_ovals = detector.detect(img_bgr)

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

        return to_numpy(peaks), to_numpy(fitted_ovals)

    def _remember_results(self, image, key, bbox, pixel_size, peaks, ovals):
        self._result_image = image
        self._result_key = key
        self._result_bbox = bbox
        self._result_pixel_size = pixel_size
        self.app.peaks = peaks
        self.app.ovals = ovals
        self.app.display_img = None

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
        try:
            pixel_size = self._check_result_scale()
        except (TypeError, ValueError) as exc:
            self.log(f"{exc}\n")
            return
        self.display_results(
            self._result_image,
            self.app.peaks,
            self.app.ovals,
            pixel_size,
        )

    def save_results(self, filename=None):
        """Save the results to a .png file."""
        if not isinstance(self.app.display_img, np.ndarray):
            self.log("No annotated display is available. Run Detect Pits first.\n")
            return None
        if self._result_image is not None:
            try:
                self._check_result_scale()
            except (TypeError, ValueError) as exc:
                self.log(f"{exc}\n")
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
        if not cv2.imwrite(str(output_path), self.app.display_img):
            self.log(f"Could not save result image: {output_path}\n")
            return None
        return output_path

    def save_measurements(self, filename=None):
        """Export internal fits and physical full diameters to an auditable CSV."""
        try:
            pixel_size_m = self._check_result_scale()
        except (TypeError, ValueError) as exc:
            self.log(f"{exc}\n")
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
