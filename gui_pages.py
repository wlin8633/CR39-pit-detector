import tkinter as tk
from tkinter import ttk

class PageSlicing:
    def __init__(self, app, logic):
        self.app = app
        self.logic = logic
        self.frame = ttk.Frame(self.app.notebook)

        rowp = 0
        # Raw image
        ttk.Label(self.frame, text="Raw Image Path:").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.path_var, width=60).grid(row=rowp, column=1, columnspan=3, sticky=tk.W, padx=5)
        ttk.Button(self.frame, text="Browse...", command=self.logic.browse_file).grid(row=rowp, column=4, padx=5, sticky=tk.W)

        rowp += 1
        ttk.Label(self.frame, text="desired_width:").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.dw_var, width=10).grid(row=rowp, column=1, sticky=tk.W, padx=5)

        rowp += 1
        ttk.Label(self.frame, text="desired_height:").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.dh_var, width=10).grid(row=rowp, column=1, sticky=tk.W, padx=5)

        rowp += 1
        ttk.Label(self.frame, text="extended_rate:").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.ext_rate_var, width=10).grid(row=rowp, column=1, sticky=tk.W, padx=5)

        # slice indices
        ttk.Label(self.frame, textvariable=self.app.scale_var).grid(row=1, column=2, sticky=tk.W)
        
        ttk.Label(self.frame, text="h:").grid(row=2, column=2, sticky=tk.W)
        ttk.Entry(self.frame, textvariable=self.app.sh_var, width=10).grid(row=2, column=2, sticky=tk.W, padx=25)

        ttk.Label(self.frame, text="w:").grid(row=3, column=2, sticky=tk.W)
        ttk.Entry(self.frame, textvariable=self.app.sw_var, width=10).grid(row=3, column=2, sticky=tk.W, padx=25)

        # Bbox mode
        rowp += 1
        ttk.Label(self.frame, text="BoundingBox Mode:").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Radiobutton(self.frame, text="Fraction", variable=self.app.bbox_mode, value="Fraction").grid(row=rowp, column=1, sticky=tk.W, padx=5)
        ttk.Radiobutton(self.frame, text="Absolute", variable=self.app.bbox_mode, value="Absolute").grid(row=rowp, column=2, sticky=tk.W, padx=5)

        # x1, y1, x2, y2
        rowp += 1
        ttk.Label(self.frame, text="x1:").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.x1_var, width=10).grid(row=rowp, column=1, sticky=tk.W, padx=5)

        rowp += 1
        ttk.Label(self.frame, text="y1:").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.y1_var, width=10).grid(row=rowp, column=1, sticky=tk.W, padx=5)

        rowp += 1
        ttk.Label(self.frame, text="x2:").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.x2_var, width=10).grid(row=rowp, column=1, sticky=tk.W, padx=5)

        rowp += 1
        ttk.Label(self.frame, text="y2:").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.y2_var, width=10).grid(row=rowp, column=1, sticky=tk.W, padx=5)
        
        rowp += 1
        ttk.Label(self.frame, text="Parameters for Display:").grid(row=rowp, column=0, sticky=tk.E)
        
        rowp += 1
        ttk.Label(self.frame, text="centre_size:").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.centre_size, width=10).grid(row=rowp, column=1, sticky=tk.W, padx=5)

        ttk.Label(self.frame, text="edge_size:").grid(row=rowp, column=2, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.edge_size, width=10).grid(row=rowp, column=3, sticky=tk.W, padx=5)
        
        rowp += 1
        ttk.Label(self.frame, text="font_size:").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.font_size, width=10).grid(row=rowp, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(self.frame, text="font_edge_size:").grid(row=rowp, column=2, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.font_edge_size, width=10).grid(row=rowp, column=3, sticky=tk.W, padx=5)
        
        rowp += 1
        # Buttons
        ttk.Button(self.frame, text="Reset Image", command=self.logic.reset_image).grid(row=rowp, column=0, pady=5)
        ttk.Button(self.frame, text="Update Image", command=self.logic.update_image).grid(row=rowp, column=1, pady=5)
        ttk.Button(self.frame, text="Open Canvas Window", command=self.logic.open_canvas_window).grid(row=rowp, column=2, pady=5)
        
        rowp += 1
        # Buttons to Save / Load
        ttk.Button(self.frame, text="Detect Pits", command=self.logic.detect_pits).grid(row=rowp, column=0, padx=5, pady=5)
        ttk.Button(self.frame, text="Replace Detect", command=self.logic.replace_detection).grid(row=rowp, column=1, padx=5, pady=5)
        ttk.Button(self.frame, text="Show Display", command=lambda: self.logic.display_results(self.app.extended_img, self.app.peaks, self.app.ovals, float(self.app.obj_res_var.get()) / float(self.app.ext_rate_var.get()))).grid(row=rowp, column=2, padx=5, pady=5)
        ttk.Button(self.frame, text="Save Display", command=self.logic.save_results).grid(row=rowp, column=3, padx=5, pady=5)        
        
        rowp += 1
        ttk.Button(self.frame, text="Load Status", command=self.logic.load_parameters).grid(row=rowp, column=0, pady=5)
        ttk.Button(self.frame, text="Save Status", command=self.logic.save_parameters).grid(row=rowp, column=1, pady=5)
        
        rowp += 1
        ttk.Button(self.frame, text="Batch Detect", command=self.logic.batch_detect).grid(row=rowp, column=0, padx=5, pady=5)
        
        rowp += 1
        ttk.Label(self.frame, text="Batch scale w:").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.bsw_var, width=10).grid(row=rowp, column=1, sticky=tk.W, padx=5)
        
        rowp += 1
        ttk.Label(self.frame, text="Batch scale h:").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.bsh_var, width=10).grid(row=rowp, column=1, sticky=tk.W, padx=5)

        # Load any existing data at startup
        self.logic.auto_load(self.app.AUTOSAVE_FILE)

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
        ttk.Label(self.frame, text="radius_min (m):").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.radius_min_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)

        ttk.Label(self.frame, text="radius_max (m):").grid(row=rowp, column=2, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.radius_max_var, width=12).grid(row=rowp, column=3, padx=5, sticky=tk.W)

        rowp += 1
        ttk.Label(self.frame, text="peak_charac_len (m):").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.peak_charac_len_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)        
        
        ttk.Label(self.frame, text="peak_pad_rate:").grid(row=rowp, column=2, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.peak_pad_rate_var, width=12).grid(row=rowp, column=3, padx=5, sticky=tk.W)

        rowp += 1
        ttk.Label(self.frame, text="peak_thre_offset:").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.peak_thre_offset_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(self.frame, text="peak_edge_std:").grid(row=rowp, column=2, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.peak_edge_std_var, width=12).grid(row=rowp, column=3, padx=5, sticky=tk.W)
        
        rowp += 1
        ttk.Label(self.frame, text="peak_gaussian_sigma (px):").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.peak_sigma_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)

        ttk.Label(self.frame, text="peak_closest_dist (m):").grid(row=rowp, column=2, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.peak_dist_var, width=12).grid(row=rowp, column=3, padx=5, sticky=tk.W)
        
        rowp += 1
        ttk.Label(self.frame, text="rad_charac_len (m):").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.eps_charac_len_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(self.frame, text="mode:").grid(row=rowp, column=2, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.mode_var, width=12).grid(row=rowp, column=3, padx=5, sticky=tk.W)
        
        rowp += 1
        ttk.Label(self.frame, text="eps_product_thre_offset:").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.eps_product_thre_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(self.frame, text="eps_allowed_center_offest (m):").grid(row=rowp, column=2, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.eps_center_var, width=12).grid(row=rowp, column=3, padx=5, sticky=tk.W)
        
        rowp += 1
        ttk.Label(self.frame, text="eps_RMS_thre (%):").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.eps_rms_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(self.frame, text="cir_radii_thre_percent:").grid(row=rowp, column=2, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.cir_radii_thre_var, width=12).grid(row=rowp, column=3, padx=5, sticky=tk.W)
        
        rowp += 1
        ttk.Label(self.frame, text="mfp_detection:").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.mfp_detection_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(self.frame, text="mfp:").grid(row=rowp, column=2, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.mfp_var, width=12).grid(row=rowp, column=3, padx=5, sticky=tk.W)
        
        rowp += 1
        ttk.Label(self.frame, text="mfp_std:").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.mfp_std_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)
        
        rowp += 1
        ttk.Label(self.frame, text="timeout (s):").grid(row=rowp, column=0, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.timeout_var, width=12).grid(row=rowp, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(self.frame, text="verbose:").grid(row=rowp, column=2, sticky=tk.E)
        ttk.Entry(self.frame, textvariable=self.app.verbose_var, width=12).grid(row=rowp, column=3, padx=5, sticky=tk.W)
        
        rowp += 1
        ttk.Button(self.frame, text="Detect Pits", command=self.logic.detect_pits).grid(row=rowp, column=3, padx=5, pady=5)
