import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

class ImageDisplayWindow(tk.Toplevel):
    def __init__(self, parent_app, *args, **kwargs):
        """
        parent_app: reference to PitDetectionNotebookApp, so we can read/write
                    bounding box variables, extended_img, etc.
        """
        super().__init__(*args, **kwargs)
        self.title("Image Display Window")

        self.parent_app = parent_app
        self.canvas_width  = 720
        self.canvas_height = 560

        # Canvas for the image
        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(padx=5, pady=5)

        # Mouse-based bounding box
        self.rect_id = None
        self.start_x = None
        self.start_y = None

        # Offsets & ratio for coordinate transforms
        self.x0 = 0
        self.y0 = 0
        self.ratio_x = 1.0
        self.ratio_y = 1.0
        self.ext_w = 1
        self.ext_h = 1

        # Keep reference to PhotoImage
        self.imgtk = None

        # Bind events
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def show_image(self, extended_img):
        """
        Display the given extended_img (BGR) on the canvas,
        resetting mouse bounding box etc.
        """
        if extended_img is None:
            self.clear_canvas()
            return

        self.ext_w = extended_img.shape[1]
        self.ext_h = extended_img.shape[0]

        # Convert BGR -> RGB
        rgb_img = cv2.cvtColor(extended_img, cv2.COLOR_BGR2RGB)

        # Fit image into canvas
        ratio = min(self.canvas_width / self.ext_w, self.canvas_height / self.ext_h)
        disp_w = int(self.ext_w * ratio)
        disp_h = int(self.ext_h * ratio)
        disp_img = cv2.resize(rgb_img, (disp_w, disp_h))

        self.ratio_x = self.ext_w / disp_w
        self.ratio_y = self.ext_h / disp_h

        pil_img = Image.fromarray(disp_img)
        self.imgtk = ImageTk.PhotoImage(image=pil_img)

        self.x0 = (self.canvas_width - disp_w) // 2
        self.y0 = (self.canvas_height - disp_h) // 2

        # Clear old
        self.canvas.delete("all")
        self.rect_id = None

        # Draw
        self.canvas.create_image(self.x0, self.y0, image=self.imgtk, anchor=tk.NW)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.rect_id = None
        self.imgtk = None

    # ------------------------------------------------------------------------
    # Mouse events for bounding box
    # ------------------------------------------------------------------------
    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

    def on_move_press(self, event):
        if self.start_x is None or self.start_y is None:
            return
        cur_x, cur_y = event.x, event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, cur_x, cur_y, outline='red')

    def on_button_release(self, event):
        if self.start_x is None or self.start_y is None:
            return
        x1c = min(self.start_x, event.x)
        y1c = min(self.start_y, event.y)
        x2c = max(self.start_x, event.x)
        y2c = max(self.start_y, event.y)

        # Convert to extended coords
        x1_ext = (x1c - self.x0) * self.ratio_x
        y1_ext = (y1c - self.y0) * self.ratio_y
        x2_ext = (x2c - self.x0) * self.ratio_x
        y2_ext = (y2c - self.y0) * self.ratio_y
        
        x1_ext = np.clip(x1_ext, 0, self.ext_w)
        y1_ext = np.clip(y1_ext, 0, self.ext_h)
        x2_ext = np.clip(x2_ext, 0, self.ext_w)
        y2_ext = np.clip(y2_ext, 0, self.ext_h)
        
        # judge if the bounding box is at which canvas
        
        # Update the parent's bounding box fields
        bbox_mode = self.parent_app.bbox_mode.get()
        w = self.ext_w
        h = self.ext_h
        if bbox_mode == "Fraction":
            fx1 = x1_ext / w
            fy1 = y1_ext / h
            fx2 = x2_ext / w
            fy2 = y2_ext / h
            self.parent_app.x1_var.set(f"{fx1:.3f}")
            self.parent_app.y1_var.set(f"{fy1:.3f}")
            self.parent_app.x2_var.set(f"{fx2:.3f}")
            self.parent_app.y2_var.set(f"{fy2:.3f}")
        else:
            self.parent_app.x1_var.set(str(int(round(x1_ext))))
            self.parent_app.y1_var.set(str(int(round(y1_ext))))
            self.parent_app.x2_var.set(str(int(round(x2_ext))))
            self.parent_app.y2_var.set(str(int(round(y2_ext))))

        # Reset
        self.start_x = None
        self.start_y = None
