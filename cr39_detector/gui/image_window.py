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
        self.origin = (0, 0)
        self.full_shape = (1, 1)
        self._shown_image = None

        # Keep reference to PhotoImage
        self.imgtk = None

        # Bind events
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def show_image(self, extended_img, *, origin=(0, 0), full_shape=None):
        """
        Display the given extended_img (BGR) on the canvas,
        resetting mouse bounding box etc.
        """
        if extended_img is None:
            self.clear_canvas()
            return

        full_shape = full_shape or extended_img.shape[:2]
        if extended_img is self._shown_image and origin == self.origin and full_shape == self.full_shape:
            return
        self._shown_image = extended_img
        self.origin, self.full_shape = origin, full_shape
        self.ext_w = extended_img.shape[1]
        self.ext_h = extended_img.shape[0]

        # Fit image into canvas
        ratio = min(self.canvas_width / self.ext_w, self.canvas_height / self.ext_h)
        disp_w = max(1, int(self.ext_w * ratio))
        disp_h = max(1, int(self.ext_h * ratio))
        # Convert the small canvas image, not the full 8x-sized input.
        disp_img = cv2.cvtColor(cv2.resize(extended_img, (disp_w, disp_h)), cv2.COLOR_BGR2RGB)

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

    def _canvas_box(self, box):
        ox, oy = self.origin
        x1, y1, x2, y2 = box
        x1, y1 = max(x1 - ox, 0), max(y1 - oy, 0)
        x2, y2 = min(x2 - ox, self.ext_w), min(y2 - oy, self.ext_h)
        if x1 >= x2 or y1 >= y2:
            return None
        return (self.x0 + x1 / self.ratio_x, self.y0 + y1 / self.ratio_y,
                self.x0 + x2 / self.ratio_x, self.y0 + y2 / self.ratio_y)

    def set_saved_rois(self, regions):
        self.canvas.delete("saved-roi")
        if self.imgtk is None:
            return
        for label, bbox in regions:
            coords = self._canvas_box(bbox)
            if coords:
                self.canvas.create_rectangle(*coords, outline="#0065e0", width=3, tags="saved-roi")
                self.canvas.create_text(coords[0] + 3, coords[1] + 3, text=label, anchor=tk.NW,
                                        fill="#0065e0", tags="saved-roi")
        if self.rect_id is not None:
            self.canvas.tag_raise(self.rect_id)

    def set_current_roi(self, bbox):
        self.canvas.delete("current-roi")
        self.rect_id = None
        # The full tile shown after Reset is the default, not an active selection.
        h, w = self.full_shape
        if bbox is None or tuple(bbox) == (0, 0, w, h):
            return
        coords = self._canvas_box(bbox)
        if coords:
            self.rect_id = self.canvas.create_rectangle(*coords, outline="red", dash=(4, 2), tags="current-roi")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.rect_id = None
        self.imgtk = None
        self._shown_image = None

    # ------------------------------------------------------------------------
    # Mouse events for bounding box
    # ------------------------------------------------------------------------
    def on_button_press(self, event):
        if self.parent_app.logic.busy or self.imgtk is None:
            return
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

    def on_move_press(self, event):
        if self.parent_app.logic.busy:
            return
        if self.start_x is None or self.start_y is None:
            return
        cur_x, cur_y = event.x, event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, cur_x, cur_y, outline="red", dash=(4, 2), tags="current-roi",
        )

    def on_button_release(self, event):
        if self.parent_app.logic.busy:
            self.start_x = self.start_y = None
            return
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

        # Update the parent's bounding box fields
        ox, oy = self.origin
        x1_ext, x2_ext = x1_ext + ox, x2_ext + ox
        y1_ext, y2_ext = y1_ext + oy, y2_ext + oy
        bbox_mode = self.parent_app.bbox_mode.get()
        h, w = self.full_shape
        if bbox_mode == "Fraction":
            fx1 = x1_ext / w
            fy1 = y1_ext / h
            fx2 = x2_ext / w
            fy2 = y2_ext / h
            self.parent_app.x1_var.set(f"{fx1:.10f}")
            self.parent_app.y1_var.set(f"{fy1:.10f}")
            self.parent_app.x2_var.set(f"{fx2:.10f}")
            self.parent_app.y2_var.set(f"{fy2:.10f}")
        else:
            self.parent_app.x1_var.set(str(int(round(x1_ext))))
            self.parent_app.y1_var.set(str(int(round(y1_ext))))
            self.parent_app.x2_var.set(str(int(round(x2_ext))))
            self.parent_app.y2_var.set(str(int(round(y2_ext))))

        # Reset
        self.start_x = None
        self.start_y = None
