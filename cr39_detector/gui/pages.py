import tkinter as tk
from tkinter import ttk

from ..detection import EXTENSION_INTERPOLATIONS, PIT_STATUS_INFO

class PageSlicing:
    def __init__(self, app, logic):
        self.app = app
        self.logic = logic
        self.frame = ttk.Frame(app.notebook, padding=(8, 5))

        def section(title):
            group = ttk.LabelFrame(self.frame, text=title, padding=(6, 3))
            group.pack(fill=tk.X, pady=3)
            return group

        def row(parent):
            line = ttk.Frame(parent)
            line.pack(fill=tk.X, pady=2)
            return line

        def field(parent, label, variable, width=7):
            ttk.Label(parent, text=label).pack(side=tk.LEFT, padx=(4, 2))
            ttk.Entry(parent, textvariable=variable, width=width).pack(side=tk.LEFT, padx=(0, 5))

        def button(parent, label, command):
            control = ttk.Button(parent, text=label, command=command)
            control.pack(side=tk.LEFT, padx=4)
            return control

        image = section("1. Image")
        line = row(image)
        ttk.Label(line, text="Raw image").pack(side=tk.LEFT, padx=(4, 5))
        ttk.Entry(line, textvariable=app.path_var, width=56).pack(side=tk.LEFT, fill=tk.X, expand=True)
        button(line, "Browse…", logic.browse_file)
        line = row(image)
        field(line, "Slice width", app.dw_var)
        field(line, "Height", app.dh_var)
        field(line, "Row", app.sh_var, 4)
        field(line, "Column", app.sw_var, 4)
        field(line, "Extension ×", app.ext_rate_var, 5)
        ttk.Combobox(line, textvariable=app.ext_interpolation_var, values=tuple(EXTENSION_INTERPOLATIONS),
                     state="readonly", width=8).pack(side=tk.LEFT, padx=4)
        line = row(image)
        button(line, "Reset Image", lambda: logic.reset_image(background=True))
        button(line, "Open Canvas", logic.open_canvas_window)
        ttk.Label(line, textvariable=app.scale_var).pack(side=tk.LEFT, padx=6)

        roi = section("2. Select ROI")
        line = row(roi)
        for mode in ("Fraction", "Absolute"):
            ttk.Radiobutton(line, text=mode, variable=app.bbox_mode, value=mode).pack(side=tk.LEFT, padx=4)
        for name, var in (("x1", app.x1_var), ("y1", app.y1_var), ("x2", app.x2_var), ("y2", app.y2_var)):
            field(line, name, var)
        button(line, "Update Image", lambda: logic.update_image(background=True))
        line = row(roi)
        ttk.Checkbutton(line, text="Zoom to ROI", variable=app.zoom_roi_var, command=logic.refresh_canvas).pack(
            side=tk.LEFT, padx=4,
        )
        ttk.Label(line, text="Red: current ROI", foreground="#b00020").pack(side=tk.LEFT, padx=8)
        ttk.Label(line, text="Blue: saved measurements", foreground="#0059bb").pack(side=tk.LEFT, padx=8)

        detect = section("3. Detect and edit")
        line = row(detect)
        button(line, "Detect Pits", lambda: logic.detect_pits(background=True))
        button(line, "Replace Detect", lambda: logic.replace_detection(background=True))
        self.exclude_area = tk.BooleanVar(value=False)
        button(line, "Remove Detect", lambda: logic.remove_detection(exclude_area=self.exclude_area.get()))
        ttk.Checkbutton(line, text="Exclude area too", variable=self.exclude_area).pack(side=tk.LEFT, padx=4)
        button(line, "Show Display", logic.show_current_results)
        self.stop_button = button(line, "Stop", logic.stop)
        self.stop_button.state(["disabled"])
        line = row(detect)
        field(line, "Batch columns", app.bsw_var, 4)
        field(line, "Rows", app.bsh_var, 4)
        button(line, "Batch Detect", lambda: logic.batch_detect(background=True))
        field(line, "Marker", app.centre_size, 4)
        field(line, "Outline", app.edge_size, 4)
        field(line, "Font", app.font_size, 4)
        field(line, "Font stroke", app.font_edge_size, 4)

        save = section("4. Save and reopen")
        line = row(save)
        ttk.Label(line, text="Save Directory").pack(side=tk.LEFT, padx=(4, 5))
        entry = ttk.Entry(line, textvariable=app.save_dir_var, width=44)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entry.bind("<Return>", logic.refresh_saved_rois)
        entry.bind("<FocusOut>", logic.refresh_saved_rois)
        button(line, "Browse…", logic.browse_save_directory)
        button(line, "Refresh ROIs", logic.refresh_saved_rois)
        line = row(save)
        ttk.Label(line, text="Blank uses the original image folder.").pack(side=tk.LEFT, padx=4)
        line = row(save)
        self.save_status = tk.BooleanVar(value=True)
        self.save_measurements = tk.BooleanVar(value=True)
        self.save_display = tk.BooleanVar(value=False)
        self.new_roi = tk.BooleanVar(value=False)
        self.display_format = tk.StringVar(value=".png")
        for label, variable in (
            ("Status", self.save_status), ("Measurements", self.save_measurements), ("Display", self.save_display),
        ):
            ttk.Checkbutton(line, text=label, variable=variable).pack(side=tk.LEFT, padx=4)
        ttk.Combobox(line, textvariable=self.display_format, values=(".png", ".bmp"),
                     state="readonly", width=5).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(line, text="Save as new ROI", variable=self.new_roi).pack(side=tk.LEFT, padx=4)
        button(line, "Save Selected", lambda: logic.save_selected(
            status=self.save_status.get(), measurements=self.save_measurements.get(),
            display=self.save_display.get(), display_extension=self.display_format.get(), new_roi=self.new_roi.get(),
        ))
        button(line, "Load Status", logic.load_parameters)


class PageDetection:
    def __init__(self, app, logic):
        self.app = app
        self.logic = logic
        self.frame = ttk.Frame(app.notebook)

        rowp = 0
        ttk.Label(self.frame, text="objective_resolution (m):").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.obj_res_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)

        rowp += 1
        ttk.Label(self.frame, text="bg_value:").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.bg_value_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)

        ttk.Label(self.frame, text="clips_to_bg:").grid(row=rowp, column=2, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.clips_to_bg_var, width=12).grid(row=rowp, column=3, padx=5, sticky=tk.W)

        rowp += 1
        ttk.Label(self.frame, text="semi_axis_min (m):").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.radius_min_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)

        ttk.Label(self.frame, text="semi_axis_max (m):").grid(row=rowp, column=2, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.radius_max_var, width=12).grid(row=rowp, column=3, padx=5, sticky=tk.W)

        rowp += 1
        ttk.Label(self.frame, text="peak_charac_len (m):").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.peak_charac_len_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)

        ttk.Label(self.frame, text="peak_pad_rate (ratio):").grid(row=rowp, column=2, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.peak_pad_rate_var, width=12).grid(row=rowp, column=3, padx=5, sticky=tk.W)

        rowp += 1
        ttk.Label(self.frame, text="peak_thre_offset (pp):").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.peak_thre_offset_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)

        ttk.Label(self.frame, text="peak_edge_std (sigma):").grid(row=rowp, column=2, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.peak_edge_std_var, width=12).grid(row=rowp, column=3, padx=5, sticky=tk.W)

        rowp += 1
        ttk.Label(self.frame, text="peak_gaussian_sigma (px):").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.peak_sigma_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)

        ttk.Label(self.frame, text="peak_closest_dist (m):").grid(row=rowp, column=2, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.peak_dist_var, width=12).grid(row=rowp, column=3, padx=5, sticky=tk.W)

        rowp += 1
        ttk.Label(self.frame, text="eps_charac_len (m):").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.eps_charac_len_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)

        ttk.Label(self.frame, text="mode:").grid(row=rowp, column=2, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.mode_var, width=12).grid(row=rowp, column=3, padx=5, sticky=tk.W)

        rowp += 1
        ttk.Label(self.frame, text="eps_product_thre_offset (pp):").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.eps_product_thre_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)

        ttk.Label(self.frame, text="eps_allowed_center_offset (m):").grid(row=rowp, column=2, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.eps_center_var, width=12).grid(row=rowp, column=3, padx=5, sticky=tk.W)

        rowp += 1
        ttk.Label(self.frame, text="eps_RMS_thre (%):").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.eps_rms_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)

        ttk.Label(self.frame, text="circle_bg_threshold (%):").grid(row=rowp, column=2, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.cir_radii_thre_var, width=12).grid(row=rowp, column=3, padx=5, sticky=tk.W)

        rowp += 1
        ttk.Label(self.frame, text="mfp_detection:").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.mfp_detection_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)

        ttk.Label(self.frame, text="mfp (m):").grid(row=rowp, column=2, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.mfp_var, width=12).grid(row=rowp, column=3, padx=5, sticky=tk.W)

        rowp += 1
        ttk.Label(self.frame, text="mfp_std (sigma):").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.mfp_std_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)

        rowp += 1
        ttk.Label(self.frame, text="timeout (s):").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.timeout_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)

        ttk.Label(self.frame, text="verbose:").grid(row=rowp, column=2, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.verbose_var, width=12).grid(row=rowp, column=3, padx=5, sticky=tk.W)

        rowp += 1
        ttk.Label(self.frame, text="use CUDA:").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.use_cuda_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)

        rowp += 1
        ttk.Button(self.frame, text="Detect Pits", command=lambda: self.logic.detect_pits(background=True)).grid(row=rowp, column=3, padx=5, pady=5)
        self.stop_button = ttk.Button(self.frame, text="Stop", command=self.logic.stop, state="disabled")
        self.stop_button.grid(row=rowp, column=2, padx=5, pady=5)

        rowp += 1
        help_frame = ttk.LabelFrame(self.frame, text="Fit diagnostics")
        help_frame.grid(row=rowp, column=0, columnspan=4, padx=5, pady=8, sticky=tk.EW)
        ttk.Label(
            help_frame,
            text=(
                "eps_product_thre_offset controls how strictly gradient pixels are kept for the contour. "
                "eps_RMS_thre rejects irregular ellipse fits. eps_allowed_center_offset does not reject a pit; "
                "it falls back to the detected peak center when the fitted center moves too far."
            ),
            wraplength=760,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=6, pady=4)
        for code in (0, -1, -2, -3, -4):
            info = PIT_STATUS_INFO[code]
            ttk.Label(
                help_frame,
                text=f"{info['short']} ({code}): {info['description']}",
            ).pack(anchor=tk.W, padx=12)
