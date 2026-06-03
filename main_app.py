# %%
import tkinter as tk
from tkinter import ttk
import numpy as np

from image_window import ImageDisplayWindow
from gui_pages import PageSlicing, PageDetection
from app_logic import AppLogic

class PitDetectionNotebookApp:
    def __init__(self, master):
        self.master = master
        master.title("CR39 Pit Detection - Multi-Page GUI")

        # Create a Notebook
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill='both', expand=True)

        # ---------------------------------------------------------------------
        # Shared Data
        # ---------------------------------------------------------------------
        self.AUTOSAVE_FILE = "./savFiles_pitDetection/autosave.sav"
        self.extended_img = None
        self.old_ext_img = None
        self.bbox_mode = tk.StringVar(value="Fraction")
        self.x1_var = tk.StringVar(value="0")
        self.y1_var = tk.StringVar(value="0")
        self.x2_var = tk.StringVar(value="1.0")
        self.y2_var = tk.StringVar(value="1.0")

        # For slicing
        self.path_var = tk.StringVar(value=r"./example.bmp")
        self.dw_var = tk.StringVar(value="2048")
        self.dh_var = tk.StringVar(value="1536")
        self.bdw_var = tk.StringVar(value="2048")
        self.bdh_var = tk.StringVar(value="1536")
        self.ext_rate_var = tk.StringVar(value="2")
        self.scale_var = tk.StringVar(value="Scale: ")
        self.sh_var = tk.StringVar(value="0")
        self.sw_var = tk.StringVar(value="0")
        self.bsh_var = tk.StringVar(value="0")
        self.bsw_var = tk.StringVar(value="0")

        # For detection
        self.obj_res_var = tk.StringVar(value="1.38e-07")
        self.radius_min_var = tk.StringVar(value="0.2e-6")
        self.radius_max_var = tk.StringVar(value="2e-6")
        self.bg_value_var = tk.StringVar(value="None")
        self.clips_to_bg_var = tk.StringVar(value="True")
        self.peak_pad_rate_var = tk.StringVar(value="1.1")
        self.peak_charac_len_var = tk.StringVar(value="2e-6")
        self.peak_thre_offset_var = tk.StringVar(value="1")
        self.peak_edge_std_var = tk.StringVar(value="2")
        self.peak_sigma_var = tk.StringVar(value="3")
        self.peak_dist_var = tk.StringVar(value="2e-6")
        self.mode_var = tk.StringVar(value="ellipse")
        self.eps_charac_len_var = tk.StringVar(value="2e-6")
        self.eps_product_thre_var = tk.StringVar(value="1")
        self.eps_center_var = tk.StringVar(value="0.2e-6")
        self.eps_rms_var = tk.StringVar(value="3")
        self.cir_radii_thre_var = tk.StringVar(value="95")
        self.mfp_detection_var = tk.StringVar(value="False")
        self.mfp_var = tk.StringVar(value="None")
        self.mfp_std_var = tk.StringVar(value="7")
        self.timeout_var = tk.StringVar(value="10")
        self.verbose_var = tk.StringVar(value="True")
        
        # For display
        self.img_bgr = None
        self.display_img = None
        self.ovals = []
        self.peaks = []
        self.centre_size = tk.StringVar(value="3")
        self.edge_size = tk.StringVar(value="2")
        self.font_size = tk.StringVar(value="0.5")
        self.font_edge_size = tk.StringVar(value="1")

        # Create the logic handler
        self.logic = AppLogic(self)

        # Create pages
        self.page_slicing = PageSlicing(self, self.logic)
        self.page_detection = PageDetection(self, self.logic)

        self.notebook.add(self.page_slicing.frame, text="Image Slicing")
        self.notebook.add(self.page_detection.frame, text="Pit Detection")

        # Text area for logs
        self.log_text = tk.Text(master, height=10)
        self.log_text.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # Create the separate window for image display (initially hidden)
        self.image_window = None
        self.result_window = None
        
        # Override the close event so we can auto‐save before exit
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    def show_canvas_window(self):
        """Create the Toplevel window if not exists, or deiconify it."""
        if self.image_window is None or not tk.Toplevel.winfo_exists(self.image_window):
            # Create a new Toplevel window
            self.image_window = ImageDisplayWindow(self)
        else:
            self.image_window.deiconify()
    
    def on_close(self):
        """Called when the user attempts to close the window."""
        self.logic.auto_save(self.AUTOSAVE_FILE)
        self.master.destroy()

def main():
    root = tk.Tk()
    app = PitDetectionNotebookApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

# %%
