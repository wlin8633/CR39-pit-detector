# %%
import os
import shutil
import time
import numpy as np
import cupy as cp
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from cupyx.scipy.ndimage import gaussian_filter, maximum_filter, sobel

def batch_least_squares_general(X, Y):
    """
    Fully vectorized batch homogeneous least squares solver for conic/ellipse fitting.
    Uses batch SVD to find the non-trivial solution to X @ beta = 0 for each dataset.
    Suitable for fitting ellipses (ovals) with the general conic equation.
    Requires NumPy >=1.20.0 for batch SVD.
    
    Parameters:
    X (np.ndarray): A 3D array where each slice represents the design matrix for a dataset.
                    Shape: (num_datasets, num_points, num_features)
    Y (np.ndarray): (Unused, kept for compatibility)

    Returns:
    np.ndarray: Array of coefficients for each dataset. Shape: (num_datasets, num_features)
    """
    # Batch SVD
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    # The solution is the last row of Vh for each dataset
    params = Vh[:, -1, :]
    # Normalize so that last coefficient is -1 (if possible)
    mask = np.abs(params[:, -1]) > 1e-8
    params[mask] = params[mask] / -params[mask, -1][:, None]
    params[mask, -1] = -1.0
    # For the rest, normalize by norm
    mask2 = ~mask
    if np.any(mask2):
        norms = np.linalg.norm(params[mask2], axis=1, keepdims=True)
        params[mask2] = params[mask2] / norms
    return params

class PitDetector:
    """
    A class to detect pits (as circles or ellipses) in an image.
    """
    def __init__(self,
                 pixel_size,
                 radius_range,
                 mode,
                 bg_value=None,
                 clips_to_bg=True,
                 peak_pad_rate=1.1,
                 peak_charac_len=2e-6,
                 peak_thre_offset=0,
                 peak_gaussian_sigma=3,
                 peak_edge_std=2,
                 peak_closest_dist=2e-6,
                 eps_charac_len=2e-6,
                 eps_product_thre_offset=90,
                 eps_allowed_center_offest=0.2e-6,
                 eps_RMS_thre=None,
                 cir_radii_thre_percent=95,
                 mfp_detection=True,
                 mfp=None,
                 mfp_std=7,
                 timeout=60,
                 verbose=True):
        """
        Initializes the PitDetector with configuration parameters.
        """
        self.pixel_size = pixel_size
        self.radius_range = radius_range
        self.mode = mode
        self.bg_value = bg_value
        self.clips_to_bg = clips_to_bg
        self.peak_pad_rate = peak_pad_rate
        self.peak_charac_len = peak_charac_len
        self.peak_thre_offset = peak_thre_offset
        self.peak_gaussian_sigma = peak_gaussian_sigma
        self.peak_edge_std = peak_edge_std
        self.peak_closest_dist = peak_closest_dist
        self.eps_charac_len = eps_charac_len
        self.eps_product_thre_offset = eps_product_thre_offset
        self.eps_allowed_center_offest = eps_allowed_center_offest
        self.eps_RMS_thre = eps_RMS_thre
        self.cir_radii_thre_percent = cir_radii_thre_percent
        self.mfp_detection = mfp_detection
        self.mfp = mfp
        self.mfp_std = mfp_std
        self.timeout = timeout
        self.verbose = verbose

    @staticmethod
    def slice_and_extend_img(image, desired_width, desired_height,
                             remove_existing=True,
                             sliced_image_dir=None,
                             extend_image_dir=None,
                             extended_rate=None):
        """
        Slices a large image into smaller tiles of size (desired_height, desired_width),
        optionally up-samples them by 'extended_rate', and saves them to disk.
        Uses Cupy for minimal overhead, but the main image read/resize (cv2) still uses CPU.

        image: Loaded by cv2, shape (H, W, C).
        desired_width: width of slice in pixels.
        desired_height: height of slice in pixels.
        remove_existing: whether to remove existing directory before saving.
        sliced_image_dir: folder to save the sliced images.
        extend_image_dir: folder to save the extended (zoomed) images.
        extended_rate: scale factor for up-sampling. E.g., 2 => 2x dimension.

        Returns: scale_x, scale_y for the image.
        """
        height, width, _ = image.shape

        # Cupy usage to compute scale_x, scale_y
        scale_x = cp.ceil(width / desired_width).item()
        scale_y = cp.ceil(height / desired_height).item()

        # Resizing the image is still on CPU
        res_height = int(desired_height * scale_y)
        res_width  = int(desired_width * scale_x)
        image_resized = cv2.resize(image, (res_width, res_height))

        if remove_existing:
            shutil.rmtree(sliced_image_dir, ignore_errors=True)
            shutil.rmtree(extend_image_dir, ignore_errors=True)
            
        if sliced_image_dir is not None:
            os.makedirs(sliced_image_dir, exist_ok=True)
        else:
            print("No sliced image directory specified. Exiting...")
            return scale_x, scale_y
        
        if extend_image_dir is not None:
            os.makedirs(extend_image_dir, exist_ok=True)
        else:
            print("No extended image directory specified. Skipping...")

        # Slice and optionally up-sample
        for i in range(0, res_height, desired_height):
            for j in range(0, res_width, desired_width):
                slice_ij = image_resized[i:i+desired_height, j:j+desired_width]
                
                # Save the slices
                slice_path = os.path.join(sliced_image_dir,
                                f"{int(i/desired_height)}_{int(j/desired_width)}.bmp")
                cv2.imwrite(slice_path, slice_ij)
                
                if extended_rate is not None and extend_image_dir is not None:
                    slice_ij_extended = cv2.resize(slice_ij,
                                                   (desired_width * extended_rate,
                                                    desired_height * extended_rate),
                                                   interpolation=cv2.INTER_CUBIC)
                else:
                    print("No extended rate or directory specified. Skipping...")
                    return scale_x, scale_y

                # Save the extended slices
                extend_path = os.path.join(extend_image_dir,
                                f"{int(i/desired_height)}_{int(j/desired_width)}.bmp")
                cv2.imwrite(extend_path, slice_ij_extended)
                    
        return scale_x, scale_y

    @staticmethod
    def _bbox_from_img(image, bbox, verbose=False):
        x1, y1, x2, y2 = map(int, bbox)
        
        if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
            print("Bounding box out of image range. Exiting...")
            return
        
        if verbose:
            print(f"Using bounding box: {bbox}")
        
        return image[y1:y2, x1:x2]

    @staticmethod
    def _otsu_from_hist(hist):
        """
        Helper function that, given a 1D CuPy histogram array of length 256,
        computes the Otsu threshold index on the GPU, returning it as a CuPy scalar.
        """
        # Convert histogram to float and compute total counts
        hist = hist.astype(cp.float32)
        total = cp.sum(hist)
        
        # Avoid divide-by-zero if total=0
        if total <= 0:
            return cp.float32(0)

        # Probability of each intensity
        p = hist / total

        # Cumulative sums
        omega = cp.cumsum(p)  # cumulative probability
        mu = cp.cumsum(p * cp.arange(256))  # cumulative mean

        # Global mean
        mu_t = mu[-1]

        # Between-class variance
        sigma_b_squared = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)

        # Best threshold is the argmax of between-class variance
        best_idx = cp.argmax(sigma_b_squared)
        return best_idx

    def _otsu_threshold(self, arr, batch=False):
        """
        Compute Otsu threshold(s) fully on the GPU using CuPy.
        """
        if not batch:
            # --- Single threshold for entire array ---
            hist, _ = cp.histogram(arr, bins=256, range=(0, 255))
            return self._otsu_from_hist(hist)
        else:
            # --- Batch mode: arr is shape (N, ...), compute one threshold per sub-array (0th axis) ---
            N = arr.shape[0]
            arr_flat = arr.reshape((N, -1))
            histograms = cp.apply_along_axis(
                lambda x: cp.histogram(x, bins=256, range=(0, 255))[0],
                axis=1,
                arr=arr_flat
            )
            thresholds = cp.zeros(N, dtype=cp.float32)
            for i in range(N):
                h = histograms[i]
                best_idx = self._otsu_from_hist(h)
                thresholds[i] = best_idx
            return thresholds

    def _mask_for_hough_lines(self, mask, thre_precent):
        """
        Thresholds the gradient magnitude mask for Hough lines.
        """
        mask_min, mask_max = mask.min(), mask.max()
        norm_mask = ((mask - mask_min) * 255.0 / (mask_max - mask_min)).astype(cp.uint8)
        threshold = cp.percentile(norm_mask, thre_precent)
        thre_mask = cp.zeros_like(norm_mask)
        thre_mask[norm_mask >= threshold] = 255

        if self.verbose:
            print(f"Threshold for gradient magnitude: {thre_precent}% => {threshold:.2f}")
        
        return thre_mask

    def _padding(self, image, pad_rate_x, pad_rate_y, dtype=cp.uint8):
        """
        Pads the image with specified rates.
        """
        pad_h = int(image.shape[0] * pad_rate_y)
        pad_w = int(image.shape[1] * pad_rate_x)
        extended_image = cp.zeros((pad_h, pad_w), dtype=dtype)
        start_y = int((pad_rate_y - 1)/2 * image.shape[0])
        start_x = int((pad_rate_x - 1)/2 * image.shape[1])
        extended_image[start_y:start_y+image.shape[0], start_x:start_x+image.shape[1]] = image
        
        if self.verbose:
            print(f"Padding: {pad_h} x {pad_w}")
        
        return extended_image, start_x, start_y

    def _parallel_bresenham(self, mask_shape, points, dx, dy, dtype=cp.uint16):
        """
        Draws lines using Bresenham's algorithm in parallel on the GPU.
        """
        mask = cp.zeros(mask_shape, dtype=dtype)
        x0, y0 = points[:, 1], points[:, 0]
        x1_, y1_ = x0 + dx, y0 + dy

        ddx, ddy = cp.abs(x1_ - x0), cp.abs(y1_ - y0)
        sx = cp.where(x0 < x1_, 1, -1)
        sy = cp.where(y0 < y1_, 1, -1)
        err = ddx - ddy

        max_steps = int(cp.maximum(ddx, ddy).max().item())
        x_coords = cp.zeros((len(x0), max_steps + 1), dtype=cp.int32)
        y_coords = cp.zeros((len(x0), max_steps + 1), dtype=cp.int32)
        x_coords[:, 0], y_coords[:, 0] = x0, y0

        step = 0
        while True:
            if step >= max_steps: break
            valid_mask = (y_coords[:, step] >= 0) & (y_coords[:, step] < mask.shape[0]) & \
                         (x_coords[:, step] >= 0) & (x_coords[:, step] < mask.shape[1])
            if not cp.any(valid_mask): break
            e2 = 2 * err
            x_step = cp.where((e2 > -ddy) & valid_mask, sx, 0)
            y_step = cp.where((e2 < ddx) & valid_mask, sy, 0)
            err = cp.where((e2 > -ddy) & valid_mask, err - ddy, err)
            err = cp.where((e2 < ddx) & valid_mask, err + ddx, err)

            x_coords[:, step+1] = x_coords[:, step] + x_step
            y_coords[:, step+1] = y_coords[:, step] + y_step
            step += 1

        valid_coords = (y_coords >= 0) & (y_coords < mask.shape[0]) & \
                       (x_coords >= 0) & (x_coords < mask.shape[1])
        cp.add.at(mask, (y_coords[valid_coords], x_coords[valid_coords]), 1)
        
        if self.verbose:
            print(f"Parallel Bresenham completed with {step} steps.")
        
        return mask

    def _find_overlap_peaks(self, peaks, distance):
        """
        Find the overlapping peaks within a distance.
        """
        sq_norms = cp.sum(peaks * peaks, axis=1, keepdims=True)
        dist_sq = sq_norms - 2 * peaks @ peaks.T + sq_norms.T
        dist_sq = cp.maximum(dist_sq, 0)
        D = cp.sqrt(dist_sq)
        D = cp.triu(D)
        
        if self.verbose:
            print(f"Distance matrix shape: {D.shape}")
        
        return cp.argwhere((D <= distance) & (D > 0))

    @staticmethod
    def _pit_separation_line(pts):
        """
        Calculate separation line for overlapping pits.
        """
        center = pts.mean(axis=1)
        slope = (pts[:,1,1] - pts[:,0,1]) / (pts[:,1,0] - pts[:,0,0])
        sep_m = -1.0 / slope
        sep_b = center[:,1] - sep_m * center[:,0]
        sep_b = cp.where(cp.isinf(sep_m), center[:,0], sep_b)
        return cp.column_stack((sep_m, sep_b))

    @staticmethod
    def _filter_out_overlap_region_and_cal_product(sobelx, sobely, xx, yy, peaks, overlap_peaks, max_radius_px, peak_pairs, seg_m, seg_b):
        """
        Filter out overlap regions and calculate dot product for ellipse fitting.
        """
        norm_dir = cp.arctan2(yy - peaks[:, 1, None, None], xx - peaks[:, 0, None, None])
        norm_dir_x, norm_dir_y = cp.cos(norm_dir), cp.sin(norm_dir)
        norm = cp.sqrt(norm_dir_x**2 + norm_dir_y**2)
        norm = cp.where(cp.sqrt((xx - peaks[:, 0, None, None])**2 + (yy - peaks[:, 1, None, None])**2) <= max_radius_px, norm, cp.inf)
        det = (peak_pairs[:, :, 1].T - seg_m.T[0] * peak_pairs[:, :, 0].T - seg_b.T[0]).T
        
        for i in range(len(overlap_peaks)):
            if cp.isinf(det[i, 0]):
                norm[overlap_peaks[i, 0]] = cp.where((xx[overlap_peaks[i, 0]] - seg_b[i, 0]) < 0, norm[overlap_peaks[i, 0]], cp.inf)
                norm[overlap_peaks[i, 1]] = cp.where((xx[overlap_peaks[i, 1]] - seg_b[i, 1]) > 0, norm[overlap_peaks[i, 1]], cp.inf)
            elif det[i, 0] > 0:
                norm[overlap_peaks[i, 0]] = cp.where((yy[overlap_peaks[i, 0]] - seg_m[i, 0] * xx[overlap_peaks[i, 0]] - seg_b[i, 0]) > 0, norm[overlap_peaks[i, 0]], cp.inf)
                norm[overlap_peaks[i, 1]] = cp.where((yy[overlap_peaks[i, 1]] - seg_m[i, 1] * xx[overlap_peaks[i, 1]] - seg_b[i, 1]) < 0, norm[overlap_peaks[i, 1]], cp.inf)
            else:
                norm[overlap_peaks[i, 0]] = cp.where((yy[overlap_peaks[i, 0]] - seg_m[i, 1] * xx[overlap_peaks[i, 0]] - seg_b[i, 1]) < 0, norm[overlap_peaks[i, 0]], cp.inf)
                norm[overlap_peaks[i, 1]] = cp.where((yy[overlap_peaks[i, 1]] - seg_m[i, 0] * xx[overlap_peaks[i, 1]] - seg_b[i, 0]) > 0, norm[overlap_peaks[i, 1]], cp.inf)

        norm_dir_x /= norm
        norm_dir_y /= norm
        return sobelx[yy, xx] * norm_dir_x + sobely[yy, xx] * norm_dir_y

    def _fit_ovals(self, peaks, product_thre, xx, yy):
        """
        Fit ellipses to the detected peaks using batch least squares (vectorized), with vectorized post-processing.
        """
        def RMS_distance_vectorized(points, h, k, a, b, theta, n_grid=1000):
            if len(points) == 0 or np.isnan(a) or np.isnan(b) or np.isnan(h) or np.isnan(k) or np.isnan(theta):
                return np.nan
            phi = np.linspace(0, 2*np.pi, n_grid)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            X, Y = points[:, 0] - h, points[:, 1] - k  # (N,)
            x_ellipse = a * np.cos(phi)
            y_ellipse = b * np.sin(phi)
            x_rot = x_ellipse * cos_t - y_ellipse * sin_t
            y_rot = x_ellipse * sin_t + y_ellipse * cos_t
            dists = (X[:, None] - x_rot[None, :])**2 + (Y[:, None] - y_rot[None, :])**2
            min_dists = np.sqrt(np.min(dists, axis=1))
            return np.sqrt(np.mean(min_dists**2)) * self.pixel_size

        # Collect all valid points for each peak
        x_list, y_list = [], []
        for i in range(len(peaks)):
            mask_ij = product_thre[i]
            x_list.append(xx[i][mask_ij])
            y_list.append(yy[i][mask_ij])
        max_len = max(len(xi) for xi in x_list)
        num_peaks = len(peaks)
        X_pad = np.full((num_peaks, max_len), np.nan, dtype=np.float64)
        Y_pad = np.full((num_peaks, max_len), np.nan, dtype=np.float64)
        for i, (xi, yi) in enumerate(zip(x_list, y_list)):
            X_pad[i, :len(xi)] = xi.get() if hasattr(xi, 'get') else xi
            Y_pad[i, :len(yi)] = yi.get() if hasattr(yi, 'get') else yi
        valid_mask = ~np.isnan(X_pad)
        X_design = np.zeros((num_peaks, max_len, 6), dtype=np.float64)
        X_design[:, :, 0] = np.where(valid_mask, X_pad**2, 0)
        X_design[:, :, 1] = np.where(valid_mask, X_pad*Y_pad, 0)
        X_design[:, :, 2] = np.where(valid_mask, Y_pad**2, 0)
        X_design[:, :, 3] = np.where(valid_mask, X_pad, 0)
        X_design[:, :, 4] = np.where(valid_mask, Y_pad, 0)
        X_design[:, :, 5] = np.where(valid_mask, 1, 0)
        Y_target = np.zeros((num_peaks, max_len), dtype=np.float64)
        X_design[~valid_mask] = 0
        try:
            params = batch_least_squares_general(X_design, Y_target)
        except Exception as e:
            print(f"Batch ellipse fit failed: {e}")
            params = np.full((num_peaks, 6), np.nan)
        A_ = params[:, 0]
        B_ = params[:, 1]
        C__ = params[:, 2]
        D_ = params[:, 3]
        E_ = params[:, 4]
        F_ = params[:, 5]
        Delta = 4.0*A_*C__ - B_**2
        valid = (Delta > 0) & (~np.all(np.isnan(X_pad[:, :10]), axis=1))
        center_x = np.full(num_peaks, np.nan)
        center_y = np.full(num_peaks, np.nan)
        center_x[valid] = (B_[valid]*E_[valid] - 2.0*C__[valid]*D_[valid]) / Delta[valid]
        center_y[valid] = (B_[valid]*D_[valid] - 2.0*A_[valid]*E_[valid]) / Delta[valid]
        peaks_np = peaks.get() if hasattr(peaks, 'get') else peaks
        offset = np.full(num_peaks, np.nan)
        offset[valid] = np.sqrt((center_x[valid] - peaks_np[valid, 0])**2 + (center_y[valid] - peaks_np[valid, 1])**2) * self.pixel_size
        use_peak = (offset > self.eps_allowed_center_offest) & valid
        center_x[use_peak] = peaks_np[use_peak, 0]
        center_y[use_peak] = peaks_np[use_peak, 1]
        a_val = np.full(num_peaks, np.nan)
        b_val = np.full(num_peaks, np.nan)
        theta_val = np.full(num_peaks, np.nan)
        for i in range(num_peaks):
            if not valid[i]:
                continue
            xi, yi = X_pad[i][valid_mask[i]], Y_pad[i][valid_mask[i]]
            Xc, Yc = xi - center_x[i], yi - center_y[i]
            A_mat = np.column_stack([Xc**2, Xc*Yc, Yc**2])
            try:
                Q, R_, S_ = np.linalg.lstsq(A_mat, np.ones_like(Xc), rcond=None)[0]
            except Exception:
                continue
            evals_np, evecs_np = np.linalg.eig(np.array([[Q, R_/2.0], [R_/2.0, S_]]))
            idx_sorted = np.argsort(evals_np)
            lam0, lam1 = evals_np[idx_sorted[0]], evals_np[idx_sorted[1]]
            if lam0 <= 0 or lam1 <= 0:
                continue
            a_tmp, b_tmp = 1.0/np.sqrt(lam0), 1.0/np.sqrt(lam1)
            major_eigenvector = evecs_np[:, idx_sorted[0]] if b_tmp <= a_tmp else evecs_np[:, idx_sorted[1]]
            theta_tmp = np.arctan2(major_eigenvector[1], major_eigenvector[0])
            if b_tmp > a_tmp:
                a_tmp, b_tmp = b_tmp, a_tmp
            a_val[i], b_val[i], theta_val[i] = a_tmp, b_tmp, theta_tmp

        # Vectorized RMS calculation for all valid ellipses
        label = np.full(num_peaks, -1, dtype=np.float64)  # Default: invalid
        valid_idx = np.where(valid & ~np.isnan(a_val) & ~np.isnan(b_val))[0]
        if len(valid_idx) > 0 and self.eps_RMS_thre is not None:
            # Prepare all points for all valid ellipses
            all_points = [np.column_stack((X_pad[i][valid_mask[i]], Y_pad[i][valid_mask[i]])) for i in valid_idx]
            # Compute RMS for all valid ellipses
            rms_vals = np.array([
                RMS_distance_vectorized(
                    pts, center_x[i], center_y[i], a_val[i], b_val[i], theta_val[i]
                ) for i, pts in zip(valid_idx, all_points)
            ])
            # Assign label -3 if RMS too large, else 0
            label[valid_idx] = np.where(rms_vals / self.pixel_size / np.sqrt(a_val[valid_idx] * b_val[valid_idx]) > self.eps_RMS_thre / 100, -3, 0)
        else:
            label[valid_idx] = 0
        fitted_ovals = np.column_stack([center_x, center_y, a_val, b_val, theta_val, label])
        return cp.asarray(fitted_ovals)

    def _radii_mask_based_on_bg_value(self, gpu_gray, bg_value):
        """
        Threshold the radii based on the background value.
        """
        thre_intens = (self.cir_radii_thre_percent / 100) * bg_value
        if self.verbose:
            print(f"Threshold for intensity: {self.cir_radii_thre_percent} = {thre_intens:.2e}")
        return gpu_gray > thre_intens

    def _edge_connection(self, det_rad):
        """
        Connects edges in the radii mask.
        """
        edge_connected = cp.zeros_like(det_rad, dtype=cp.uint8)
        edge_connected[0,:], edge_connected[-1,:], edge_connected[:,0], edge_connected[:,-1] = \
            det_rad[0,:], det_rad[-1,:], det_rad[:,0], det_rad[:,-1]

        loop_start = time.time()
        print("Start edge connection...")
        while True:
            dilated = cp.roll(edge_connected,1,0)|cp.roll(edge_connected,-1,0)| \
                      cp.roll(edge_connected,1,1)|cp.roll(edge_connected,-1,1)
            dilated &= det_rad
            
            if self.verbose:
                print(f"\rTime elapsed: {time.time() - loop_start:.2f} s", end="")
            
            if cp.array_equal(dilated, edge_connected) or (time.time() - loop_start) > self.timeout:
                if (time.time() - loop_start) > self.timeout:
                    print(f"\nTimeout reached. Unable to complete edge connection.")
                break
            edge_connected = dilated.copy()
            
        print("\nEdge connection completed.")
        return edge_connected

    def _fit_circles(self, det_rad_ROIs, peaks, charac_len_px, min_r, max_r, dist_sq_template, x1, y1):
        """
        Fit circles to detected regions of interest.
        """
        int_radii = cp.zeros((len(peaks), charac_len_px))
        for r in range(1, charac_len_px + 1):
            ring_mask = (dist_sq_template <= r**2) & (dist_sq_template > (r-1)**2)
            int_radii[:, r-1] = cp.sum(det_rad_ROIs[:, ring_mask], axis=1) / (2 * cp.pi * r)

        min_indices = cp.argmax(cp.where(int_radii == cp.min(int_radii, axis=1)[:,None], cp.arange(charac_len_px), -1), axis=1)
        radii = cp.where((min_indices >= min_r) & (min_indices <= max_r), min_indices, np.nan)
        
        if self.verbose:
            print(f"Number of detected peaks: {len(peaks)}")
            print(f"Number of detected pits: {cp.sum(~cp.isnan(radii))}")
            print(f"{cp.sum(cp.isnan(radii))} peaks are removed due to the radius range.")
        
        peaks_shifted = cp.column_stack((peaks[:, 0] + x1, peaks[:, 1] + y1))
        fitted_circles = cp.array([peaks_shifted[:, 0], peaks_shifted[:, 1], radii, radii, cp.zeros(len(peaks)), cp.zeros(len(peaks))]).T
        fitted_circles[:, 5] = cp.where(cp.isnan(fitted_circles[:, 2]), -1, fitted_circles[:, 5])
        return fitted_circles

    def _ellipse_radii_detection(self, peaks, sobelx, sobely):
        """
        Radii detection for ellipse mode.
        """
        max_radius_px = int(cp.ceil(self.eps_charac_len / self.pixel_size).item())
        yy, xx = cp.meshgrid(cp.arange(-max_radius_px, max_radius_px+1), cp.arange(-max_radius_px, max_radius_px+1), indexing='ij')
        yy, xx = yy + peaks[:, 1, None, None], xx + peaks[:, 0, None, None]
        
        overlap_peaks = self._find_overlap_peaks(peaks, 2 * max_radius_px)
        peak_pairs = peaks[overlap_peaks]
        lines = self._pit_separation_line(peak_pairs)
        seg_m = cp.column_stack([lines[:, 0], lines[:, 0]])
        seg_b = cp.column_stack([lines[:, 1], lines[:, 1]])
        
        product = self._filter_out_overlap_region_and_cal_product(sobelx, sobely, xx, yy, peaks, overlap_peaks, max_radius_px, peak_pairs, seg_m, seg_b)
        
        thre_otsu = self._otsu_threshold(product, batch=True)
        product_thre_precent = 100 - cp.sum(product > thre_otsu[:, None, None], axis=(1,2)) / (product.shape[1]*product.shape[2]) * 100 + self.eps_product_thre_offset
        product_thre_precent = cp.clip(product_thre_precent, 0, 100)
        thre_vals = cp.diag(cp.percentile(product, product_thre_precent, axis=(1,2)))
        product_thre = product > thre_vals[:, None, None]
        
        fitted_ovals = self._fit_ovals(peaks, product_thre, xx, yy)
        
        if self.verbose:
            print(f"Number of detected pits: {cp.sum(fitted_ovals[:, 5] == 0)}")
        
        min_r, max_r = self.radius_range[0]/self.pixel_size, self.radius_range[1]/self.pixel_size
        cond_invalid = ((fitted_ovals[:,2]<=min_r) | (fitted_ovals[:,2]>=max_r) | \
                        (fitted_ovals[:,3]<=min_r) | (fitted_ovals[:,3]>=max_r)) & \
                        (fitted_ovals[:,5]==0)
        fitted_ovals[:,5] = cp.where(cond_invalid, -4, fitted_ovals[:,5])
        
        return fitted_ovals

    def _circle_radii_detection(self, peaks, gpu_gray, bg_value):
        """
        Radii detection for circle mode.
        """
        min_r, max_r = int(cp.ceil(self.radius_range[0] / self.pixel_size)), int(cp.floor(self.radius_range[1] / self.pixel_size))
        charac_len_px = int(cp.ceil(self.eps_charac_len / self.pixel_size))
        peakx, peaky = peaks.get().T

        ROI_x0, ROI_y0 = peakx - charac_len_px, peaky - charac_len_px
        ROI_x1, ROI_y1 = peakx + charac_len_px, peaky + charac_len_px

        padx0, pady0 = cp.maximum(0, -cp.asarray(ROI_x0)), cp.maximum(0, -cp.asarray(ROI_y0))
        padx1, pady1 = cp.maximum(0, cp.asarray(ROI_x1) - gpu_gray.shape[1]), cp.maximum(0, cp.asarray(ROI_y1) - gpu_gray.shape[0])

        ROI_x0, ROI_y0 = cp.clip(ROI_x0, 0, gpu_gray.shape[1]), cp.clip(ROI_y0, 0, gpu_gray.shape[0])
        ROI_x1, ROI_y1 = cp.clip(ROI_x1, 0, gpu_gray.shape[1]), cp.clip(ROI_y1, 0, gpu_gray.shape[0])
        
        det_rad = self._radii_mask_based_on_bg_value(gpu_gray, bg_value)
        det_rad = self._edge_connection(det_rad)
        
        ROI_x, ROI_y = cp.mgrid[0:2*charac_len_px, 0:2*charac_len_px]
        dist_sq_template = (ROI_x-charac_len_px)**2 + (ROI_y-charac_len_px)**2

        det_rad_ROIs = cp.zeros((len(peaks), 2*charac_len_px, 2*charac_len_px))
        if self.verbose:
            print(f"{len(peaks)} ROIs created with size {2*charac_len_px}x{2*charac_len_px}. Processing...")

        for i in range(len(peaks)):
            det_rad_ROIs[i, pady0[i]:2*charac_len_px-pady1[i], padx0[i]:2*charac_len_px-padx1[i]] = det_rad[ROI_y0[i]:ROI_y1[i], ROI_x0[i]:ROI_x1[i]]
        
        return self._fit_circles(det_rad_ROIs, peaks, charac_len_px, min_r, max_r, dist_sq_template, ROI_x0[0], ROI_y0[0])

    def detect(self, img_bgr, bbox=None):
        """
        Main detection pipeline.
        """
        if self.mode not in ['ellipse', 'circle']:
            print("Please specify a valid mode: 'ellipse' or 'circle'")
            return None, None

        gpu_gray = cp.mean(cp.asarray(img_bgr), axis=2).astype(cp.uint8) if img_bgr.ndim == 3 else cp.asarray(img_bgr).astype(cp.uint8)

        if bbox is None:
            if self.verbose: print("No bounding box. Using entire image.")
            bbox = [0, 0, gpu_gray.shape[1], gpu_gray.shape[0]]
        cropped_image = self._bbox_from_img(gpu_gray, bbox, self.verbose)
        
        bg_value = self.bg_value
        if bg_value is None:
            unique, counts = cp.unique(cropped_image, return_counts=True)
            bg_value = unique[cp.argmax(counts)]
            if self.verbose: print(f"Detected background value: {bg_value}")

        if self.clips_to_bg:
            cropped_image = cp.where(cropped_image > bg_value, bg_value, cropped_image)

        sobelx = sobel(cropped_image.astype(cp.float64), axis=1, mode='constant', cval=0.0)
        sobely = sobel(cropped_image.astype(cp.float64), axis=0, mode='constant', cval=0.0)
        edge_mask = cp.ones_like(cropped_image); edge_mask[1:-1, 1:-1] = 0
        sobelx[edge_mask == 1], sobely[edge_mask == 1] = 0, 0
        magnitude = cp.sqrt(sobelx**2 + sobely**2)
        
        otsu = self._otsu_threshold(cropped_image, batch=False)
        thre_precent = 100 - cp.sum(cropped_image < otsu) / cp.multiply(*cropped_image.shape) * 100 - self.peak_thre_offset
        contour_image = self._mask_for_hough_lines(magnitude, cp.clip(thre_precent, 0, 100))

        extended_contour, start_x, start_y = self._padding(contour_image, self.peak_pad_rate, self.peak_pad_rate, dtype=cp.uint16)
        extended_angle, _, _ = self._padding(cp.arctan2(sobely, sobelx) + cp.pi, self.peak_pad_rate, self.peak_pad_rate, dtype=cp.float32)
        
        edge_points = cp.argwhere(extended_contour > 0)
        angles = extended_angle[edge_points[:, 0], edge_points[:, 1]]
        extension_length = self.peak_charac_len / self.pixel_size
        if self.verbose: print(f"Extension length: {int(extension_length)} px")
        dx, dy = (extension_length * cp.cos(angles)).astype(cp.int32), (extension_length * cp.sin(angles)).astype(cp.int32)
        extended_edges = self._parallel_bresenham(extended_contour.shape, edge_points, dx, dy, dtype=cp.float32)

        smoothed_edges = gaussian_filter(extended_edges, sigma=self.peak_gaussian_sigma)
        local_max = maximum_filter(smoothed_edges, size=int(self.peak_closest_dist/self.pixel_size))
        peaks_mask = (smoothed_edges == local_max) & (extended_edges > 0)
        extended_peaks = cp.argwhere(peaks_mask)
        if self.verbose: print(f"Detected peaks before filtering: {len(extended_peaks)}")
        
        edge_thre = smoothed_edges.mean() + self.peak_edge_std * smoothed_edges.std()
        extended_peaks = extended_peaks[smoothed_edges[extended_peaks[:, 0], extended_peaks[:, 1]] > edge_thre]
        peaks = cp.column_stack((extended_peaks[:, 1] - start_x, extended_peaks[:, 0] - start_y))
        if self.verbose:
            print(f"Peak detection threshold: {float(edge_thre):.2f}")
            print(f"Number of detected peaks: {len(peaks)}")

        if self.mode == 'ellipse':
            fitted_ovals = self._ellipse_radii_detection(peaks, sobelx, sobely)
        elif self.mode == 'circle':
            fitted_ovals = self._circle_radii_detection(peaks, gpu_gray, bg_value)

        # Shift final results by bbox offset
        peaks[:,0] += bbox[0]; peaks[:,1] += bbox[1]
        fitted_ovals[:,0] += bbox[0]; fitted_ovals[:,1] += bbox[1]

        if self.mfp_detection:
            if self.verbose: print(f"Performing MFP-based filtering...")
            good_mask = (fitted_ovals[:,5] == 0)
            good_peaks = peaks[good_mask]
            peak_density = cp.zeros_like(gpu_gray, dtype=cp.float32)
            for pk in good_peaks:
                px, py = int(pk[0].item()), int(pk[1].item())
                if 0 <= px < peak_density.shape[1] and 0 <= py < peak_density.shape[0]:
                    peak_density[py, px] += 1
            
            mfp = self.mfp
            if mfp is None:
                eff_num = len(good_peaks)
                size_of_domain = cp.sqrt(cp.prod(cp.array(gpu_gray.shape))) * self.pixel_size
                mfp = size_of_domain / cp.sqrt(eff_num) if eff_num > 0 else 0
                if self.verbose: print(f"Auto-calculated MFP: {mfp:.2e} m")

            peak_density_thre_sigma = int(mfp/self.pixel_size) if self.pixel_size > 0 else 0
            if peak_density_thre_sigma > 0:
                smoothed_density = gaussian_filter(peak_density, sigma=peak_density_thre_sigma)
                dense_regions = smoothed_density > (smoothed_density.mean() + self.mfp_std*smoothed_density.std())
                for i, pk in enumerate(good_peaks):
                    px, py = int(pk[0].item()), int(pk[1].item())
                    if 0 <= px < dense_regions.shape[1] and 0 <= py < dense_regions.shape[0] and dense_regions[py, px]:
                        global_idx = cp.argwhere(good_mask).ravel()[i]
                        fitted_ovals[global_idx, 5] = -2

        if self.verbose:
            n_valid = cp.sum(fitted_ovals[:,5]==0)
            print(f"Total peaks: {len(peaks)}, valid pits: {int(n_valid.item())}")

        return peaks, fitted_ovals

if __name__ == "__main__":
    # Example usage with your existing data
    rawimage_name = r".\example.bmp"
    
    desired_width = 2048
    desired_height = 1536
    extended_rate = 2
    objective_resolution = 0.138e-6 # 50X objective
    image_name = "0_0.bmp"
    
    extw, exth = desired_width*extended_rate, desired_height*extended_rate
    x1, y1, x2, y2 = int(0.3*extw), int(0*exth), int(0.6*extw), int(0.3*exth)
    vx1, vy1, vx2, vy2 = x1, y1, x2, y2
    
    # Load and preprocess the image
    rawimage = cv2.imread(rawimage_name)
    sliced_image_dir  = r".\savFiles_pitDetection\sliced_images"
    extend_image_dir  = r".\savFiles_pitDetection\extended_images"
    
    PitDetector.slice_and_extend_img(rawimage,
                                     desired_width, desired_height,
                                     remove_existing=True,
                                     sliced_image_dir=sliced_image_dir,
                                     extend_image_dir=extend_image_dir,
                                     extended_rate=extended_rate)

    extended_pixel_size = objective_resolution / extended_rate
    img_path = os.path.join(extend_image_dir, image_name)
    img_cv2 = cv2.imread(img_path)
    
    # Show bounding box on the image
    image_display = img_cv2.copy()
    plt.figure(figsize=(6, 5))
    cv2.rectangle(image_display, (x1, y1), (x2, y2), (255, 0, 0), 2)
    plt.imshow(cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB))
    plt.show()

    # %% Detect ovals
    start_time = time.time()
    
    # Initialize the detector
    detector = PitDetector(
        pixel_size=extended_pixel_size,
        radius_range=[0.2e-6, 4e-6],
        mode='ellipse',
        eps_allowed_center_offest=0.2e-6,
        eps_RMS_thre=3,
        peak_closest_dist=1e-6,
        eps_product_thre_offset=2,
        mfp_detection=False,
        verbose=True
    )
    
    # Run detection
    peaks, fitted_ovals = detector.detect(
        img_bgr=img_cv2,
        bbox=[x1, y1, x2, y2]
    )
    
    end_time = time.time()
    print(f"Detection pipeline time: {end_time - start_time:.3f} s")

    # %% Display final result
    display_peaks_only = False
    if peaks is not None and fitted_ovals is not None:
        peaks_np = cp.asnumpy(peaks)
        fitted_ovals_np = cp.asnumpy(fitted_ovals)
        overlay_img = img_cv2.copy()

        circle_size = max(1, int(1.5 * extended_rate * 0.128e-6 / objective_resolution))
        font_size = 0.6

        for idx, (peak, oval) in enumerate(zip(peaks_np, fitted_ovals_np)):
            peakx, peaky = int(peak[0]), int(peak[1])
            label = oval[5]
            
            color = (0, 255, 255) # Default for non-zero labels
            if label == 0: color = (255, 0, 0)
            cv2.circle(overlay_img, (peakx, peaky), circle_size, color, circle_size)
            
            if not display_peaks_only:
                if label == 0:
                    cx, cy, a_val, b_val, theta_val = oval[:5]
                    cv2.ellipse(overlay_img, (int(cx), int(cy)), (int(a_val), int(b_val)),
                                np.degrees(theta_val), 0, 360, (0, 255, 0), circle_size)
                    text = f"({a_val*extended_pixel_size*1e6:.2f},{b_val*extended_pixel_size*1e6:.2f})um"
                    cv2.putText(overlay_img, text, (peakx, peaky), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 255), 1, cv2.LINE_AA)
                else:
                    colors = {-1: (255,255,0), -2: (255,0,255), -3: (0,255,0), -4: (0,0,255)}
                    cv2.putText(overlay_img, f'id:{idx}, l{abs(int(label))}', (peakx, peaky),
                                cv2.FONT_HERSHEY_SIMPLEX, font_size, colors.get(label, (255,255,255)), 1, cv2.LINE_AA)

        plt.figure(figsize=(10,8))
        plt.imshow(overlay_img)
        plt.xlim(vx1, vx2); plt.ylim(vy2, vy1)
        plt.title("Final Pit Detection Overlay")
        plt.show()

# %% (new) solve E, theta by a, b
if __name__ == "__main__":
    from etchingModel import EtchingTrackRatioFitter
    fitter = EtchingTrackRatioFitter()

    a_arr, b_arr = fitted_ovals_np[:, 2:4].T * extended_pixel_size * 1e6  # Convert to micrometers
    results = np.load(os.path.join(os.path.dirname(__file__), "savFiles_etchingModel\\etching_rate_ratio_fit_results.npz"))
    fitter.provide_results(results)

    tol_energy    = 0.1
    tol_theta     = np.radians(10)
    error_len     = 0.138
    refine_energy = 200
    refine_theta  = 200
    mode          = "ml" # "ml" for most likely, "all" for all possible solutions
    lookup = fitter.inverse_lookup(a_arr, b_arr,
                        tol_energy=tol_energy, tol_theta=tol_theta, error_len=error_len,
                        refine_energy=refine_energy, refine_theta=refine_theta)

        
    # if lookup[:, 3] is the same, merge them by averaging
    uniq_lookup = np.array([np.mean(lookup[lookup[:, 3] == d, :], axis=0) for d in np.unique(lookup[:, 3])])

    if mode == "ml":
        # if uniq_lookup[:, 0] is the same, keep the one with the smallest uniq_lookup[:, 0]
        sorted_indices = np.lexsort((uniq_lookup[:, 3], uniq_lookup[:, 0]))
        sorted_array = uniq_lookup[sorted_indices]
        _, unique_indices = np.unique(sorted_array[:, 0], return_index=True)
        uniq_lookup = sorted_array[unique_indices]
    
    elif mode == "all":
        # # convert the distance to the probability
        # lookup_prob = 1 / (1 + lookup[:, 3])
        # # normalize the probability
        # lookup_prob = lookup_prob / lookup_prob.sum()
        # # sort the lookup by the probability
        # lookup_prob_sorted = lookup_prob[np.argsort(lookup_prob)]
        # # keep the top 10%
        # lookup_prob_sorted = lookup_prob_sorted[:int(lookup_prob_sorted.shape[0] * 0.1)]
        # uniq_lookup_mode = lookup[np.argsort(lookup_prob)]
        pass
        
    # sort uniq_lookup by uniq_lookup[:, 1]
    uniq_lookup = uniq_lookup[np.argsort(uniq_lookup[:, 0])]
    E_arr = uniq_lookup[:, 1]
    theta_arr = uniq_lookup[:, 2]
    target_idx = uniq_lookup[:, 0].astype(int)

    for i in range(uniq_lookup.shape[0]):
        print(f"E: {E_arr[i]:.2f} eV, theta: {np.degrees(theta_arr[i]):.2f} degrees, target_idx: {target_idx[i]}")
        
    # %% Display final result
    display_peaks_only = False
    if peaks is not None and fitted_ovals is not None and uniq_lookup is not None:
        peaks_np = cp.asnumpy(peaks)
        fitted_ovals_np = cp.asnumpy(fitted_ovals)
        overlay_img = img_cv2.copy()

        circle_size = max(1, int(1.5 * extended_rate * 0.128e-6 / objective_resolution))
        font_size = 0.6

        for idx, (peak, oval) in enumerate(zip(peaks_np, fitted_ovals_np)):
            peakx, peaky = int(peak[0]), int(peak[1])
            label = oval[5]
            
            color = (0, 255, 255) # Default for non-zero labels
            if label == 0: color = (255, 0, 0)
            cv2.circle(overlay_img, (peakx, peaky), circle_size, color, circle_size)
            
            if not display_peaks_only:
                if label == 0 and np.any(uniq_lookup[:, 0] == idx) != False:
                    cx, cy, a_val, b_val, theta_val = oval[:5]
                    cv2.ellipse(overlay_img, (int(cx), int(cy)), (int(a_val), int(b_val)),
                                np.degrees(theta_val), 0, 360, (0, 255, 0), circle_size)
                    
                    # Find the corresponding E and theta from uniq_lookup
                    E_arr = uniq_lookup[:, 1][uniq_lookup[:, 0] == idx]
                    theta_arr = uniq_lookup[:, 2][uniq_lookup[:, 0] == idx]
                    
                    (text_width, text_height), baseline = cv2.getTextSize("E", cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
                    for i, (E, theta) in enumerate(zip(E_arr, theta_arr)):
                        y_pos = int(peaky + i * text_height * 1.1)
                        text = f"E:{E:.2f} MeV, {np.degrees(theta):.2f} deg"
                        cv2.putText(overlay_img, text, (peakx, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 255), 1, cv2.LINE_AA)
                else:
                    colors = {-1: (255,255,0), -2: (255,0,255), -3: (0,255,0), -4: (0,0,255)}
                    cv2.putText(overlay_img, f'id:{idx}, l{abs(int(label))}', (peakx, peaky),
                                cv2.FONT_HERSHEY_SIMPLEX, font_size, colors.get(label, (255,255,255)), 1, cv2.LINE_AA)

        plt.figure(figsize=(10,8))
        plt.imshow(overlay_img)
        plt.xlim(vx1, vx2); plt.ylim(vy2, vy1)
        plt.title("Final Pit Detection Overlay")
        plt.show()

# %%
