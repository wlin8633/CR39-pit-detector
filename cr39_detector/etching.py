# -*- coding: utf-8 -*-
# @file etchingModel.py
# @brief Models and utilities for CR39 etching process analysis.
# @date 2025-06-24
# @author Wei Lin
# @license MIT License
"""
etchingModel.py

This module provides models and utilities for describing the etching process of CR39 nuclear track detectors.
It is intended to help analyze the relationship between etching conditions, particle energy/angle, and the resulting
track (pit) geometry observed in CR39. The models here can be used to relate physical etching parameters to
measured pit sizes, and to perform forward or inverse calculations for track analysis.

Typical uses include:
- Modeling the etching rate as a function of energy loss (REL) or other physical parameters.
- Predicting pit dimensions (major/minor axes) for given particle energies and angles.
- Inverse lookup: estimating incident particle properties from measured pit sizes.
- Supporting data analysis and visualization for CR39 etching experiments.

This file is meant to be imported and used by higher-level analysis scripts or GUIs.
"""
# %% ----------------------------------------------------------------------
# ---> Imports
# -------------------------------------------------------------------------
import time
import cv2
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import minimum_filter

from .io import get_data_from_Geant4_output, get_data_from_GetData
from .paths import ETCHING_DATA_DIR, absolute_path

__all__ = ["EtchingTrackRatioFitter", "R0_extraction", "REL_x_E_extraction"]
# %% ----------------------------------------------------------------------
# ---> Build data from Geant4 Simulation
# -------------------------------------------------------------------------

# --- Restricted Energy Loss (REL) data extraction ---
def REL_x_E_extraction(
        folder_path,
        total_depth,
        energy_range,
        total_primary,
        name_include=None,
        omega_0=None,
        is_save=False,
        save_path=None
            ):
    """
    Extracts REL (Restricted Energy Loss) data from Geant4 simulation output files.
    This function reads the output files from a Geant4 simulation, extracts the REL data, and optionally saves it
    to a specified path. The REL data is normalized by the total primary particles and converted to units of MeV/m.
    The depth is calculated based on the total depth and the number of columns in the REL data.
    The energy axis is created based on the specified energy range.

    Parameters:
    folder_path (str): Path to the folder containing Geant4 output files.
    total_depth (float): Total depth of the material in meters.
    energy_range (tuple): A tuple specifying the start, end, and step of the energy range (in MeV).
    total_primary (float): Total number of primary particles used in the simulation.
    name_include (str, optional): If specified, only files containing this string in their names will be processed.
    omega_0 (string, optional): If specified, used to name the saved REL data file (in eV).
    is_save (bool, optional): If True, saves the REL data to a file. Default is False.
    save_path (str, optional): If specified, saves the REL data to this path. If None, uses a default path.

    Returns:
    REL_x_E (np.ndarray): The extracted REL data as a 2D NumPy array (in units of MeV/m).
    depth_mid (np.ndarray): The midpoints of the depth intervals.
    energy_list (np.ndarray): The energy axis created based on the specified energy range.
    """
    # --- Get the list of REL files ---
    folder_path = absolute_path(folder_path)
    file_list = [path.name for path in folder_path.iterdir() if path.is_file()]
    if name_include is not None:
        file_list = [file for file in file_list if name_include in file]
    else:
        pass
    file_list.sort()

    # --- Create the energy axis ---
    start, end, step = energy_range
    energy_list =  np.arange(start, end + step, step)

    # --- Build the REL data ---
    REL_x_E = []
    for file in file_list:
        file_path = folder_path / file
        REL_data = get_data_from_Geant4_output(file_path, (2, 3)).T

        # --- Change the unit ---
        depth     = REL_data[0] * total_depth / REL_data.shape[1] # unit: m
        depth_mid = (depth[1:] + depth[:-1]) / 2
        REL       = REL_data[1, :-1] / (np.diff(depth)) / total_primary     # unit: MeV/m
        REL_x_E.append(REL)
    REL_x_E = np.array(REL_x_E).T

    # --- Save the REL data if required ---
    if is_save:
        if save_path is None:
            if omega_0 is not None:
                save_path = ETCHING_DATA_DIR / f"REL_x_E_ω0{omega_0}.npz"
            else:
                save_path = ETCHING_DATA_DIR / "REL_x_E.npz"
        save_path = absolute_path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(save_path), REL_x_E=REL_x_E, x=depth_mid, E=energy_list)
    else:
        pass
    return REL_x_E, depth_mid, energy_list

# --- Track length (R0) data extraction ---
def R0_extraction(
        folder_path,
        energy_range,
        total_primary,
        name_include=None,
        omega_0=None,
        is_save=False,
        save_path=None
            ):
    """
    Extracts R0 (track length) data from Geant4 simulation output files.
    This function reads the output files from a Geant4 simulation, extracts the R0 data, and optionally saves it
    to a specified path. The R0 data is normalized by the total primary particles and converted to units of meters.
    The energy axis is created based on the specified energy range.

    Parameters:
    folder_path (str): Path to the folder containing Geant4 output files.
    energy_range (tuple): A tuple specifying the start, end, and step of the energy range (in MeV).
    total_primary (float): Total number of primary particles used in the simulation.
    name_include (str, optional): If specified, only files containing this string in their names will be processed.
    omega_0 (string, optional): If specified, used to name the saved R0 data file (in eV).
    is_save (bool, optional): If True, saves the R0 data to a file. Default is False.
    save_path (str, optional): If specified, saves the R0 data to this path. If None, uses a default path.

    Returns:
    R0_E (np.ndarray): The extracted R0 data as a 2D NumPy array (in units of m).
    energy_list (np.ndarray): The energy axis created based on the specified energy range.
    """
    # --- Get the list of R0 files ---
    folder_path = absolute_path(folder_path)
    file_list = [path.name for path in folder_path.iterdir() if path.is_file()]
    if name_include is not None:
        file_list = [file for file in file_list if name_include in file]
    else:
        pass
    file_list.sort()

    # --- Create the energy axis ---
    start, end, step = energy_range
    energy_list =  np.arange(start, end + step, step)

    # --- Build the R0 data ---
    R0_E = []
    for file in file_list:
        R0_file_path = folder_path / file
        R0_data = get_data_from_Geant4_output(R0_file_path, 3).T
        # change the unit
        R0 = R0_data / total_primary / 1e3 # unit: m
        R0_E.append(R0)
    R0_E = np.array(R0_E)

    # --- Save the R0 data if required ---
    if is_save:
        if save_path is None:
            if omega_0 is not None:
                save_path = ETCHING_DATA_DIR / f"R0_E_ω0{omega_0}.npz"
            else:
                save_path = ETCHING_DATA_DIR / "R0_E.npz"
        save_path = absolute_path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(save_path), R0_E=R0_E, E=energy_list)
    else:
        pass
    return R0_E, energy_list
# %% ----------------------------------------------------------------------
# ---> EtchingTrackRatioFitter Class
# -------------------------------------------------------------------------
class EtchingTrackRatioFitter:
    def __init__(self, REL_data=None, R0_data=None, exp_data=None):
        if REL_data is None or R0_data is None or exp_data is None:
            self.fit_results = None
        else:
            self.REL_x_E, self.x, self.E_for_REL = REL_data[0]/1e2, REL_data[1]*1e6, REL_data[2] # unit: MeV, um, MeV
            self.R0_E, self.E_for_R0 = R0_data[0]*1e6, R0_data[1]  # unit: m, MeV
            self.E_exp, self.a_exp, self.b_exp, self.h_exp = exp_data
            if np.isscalar(self.h_exp):
                self.h_exp = np.array([self.h_exp])
            # Placeholders for fit results
            self.fit_results = None

    def compute_a_b(self, alpha, beta, theta, Em, hm, x0_reso=10000, output_verbose=False):
        REL_x_E, x, E_for_REL = self.REL_x_E, self.x, self.E_for_REL
        R0_E, E_for_R0 = self.R0_E, self.E_for_R0
        # --- Create interpolators based on alphai and betai ---
        invV_x_E = 1 / (1 + alpha * REL_x_E**beta); invV_x_E[np.isnan(invV_x_E)] = 0
        h_x_E = (np.cumsum(invV_x_E, axis=0) * np.diff(x)[0]); h_x_E[invV_x_E == 0] = np.nan
        REL_interp = RegularGridInterpolator((x, E_for_REL), REL_x_E, method='linear', bounds_error=False)
        h_interp   = RegularGridInterpolator((x, E_for_REL), h_x_E, method='linear', bounds_error=False)
        R0_interp  = RegularGridInterpolator((E_for_R0,), R0_E, method='linear', bounds_error=False)
        def REL(xi, Ei):
            xi = (xi, ) if not hasattr(xi, "__iter__") else xi
            Ei = (Ei, ) if not hasattr(Ei, "__iter__") else Ei
            xxi, EEi = np.meshgrid(xi, Ei, indexing='ij')
            points   = np.column_stack([xxi.ravel(), EEi.ravel()])
            RELi = REL_interp(points).reshape(xxi.shape)
            return flat_nested(RELi)
        def R0(Ei):
            Ei = (Ei, ) if not hasattr(Ei, "__iter__") else Ei
            points = np.array(Ei).reshape(-1, 1)
            Ri = R0_interp(points)
            return flat_nested(Ri)
        def V(xi, Ei, alpha=alpha, beta=beta):
            return 1 + alpha * REL(xi, Ei)**beta
        def invV(xi, Ei, alpha=alpha, beta=beta):
            return 1 / V(xi, Ei, alpha, beta)
        def hh(xi, Ei, alpha=alpha, beta=beta):
            xi = (xi, ) if not hasattr(xi, "__iter__") else xi
            Ei = (Ei, ) if not hasattr(Ei, "__iter__") else Ei
            xxi, EEi = np.meshgrid(xi, Ei, indexing='ij')
            points   = np.column_stack([xxi.ravel(), EEi.ravel()])
            hi = h_interp(points).reshape(xxi.shape)
            return flat_nested(hi)
        def flat_nested(array):
            array = np.array(array)
            while array.ndim >= 1 and array.shape[0] == 1:
                array = array[0]
            return array
        # --- Calculate the x(x0), y(x0) based on V(x, E) ---
        x0 = np.linspace(0, R0(Em), x0_reso)  # unit: um
        xx = x0 + ( (hm - hh(x0, Em))*invV(x0, Em) ).T  # x(x0)
        yy = ( (hm - hh(x0, Em))*invV(x0, Em)*np.sqrt(V(x0, Em)**2 - 1.0) ).T  # y(x0)
        xx_cond = (xx > x0)
        xx[~xx_cond] = np.nan
        yy[np.isnan(xx)] = np.nan
        P1 = xx*np.sin(theta) + yy*np.cos(theta) - hm[:, None]
        P2 = xx*np.sin(theta) - yy*np.cos(theta) - hm[:, None]
        x01 = np.full(hm.shape, np.nan)
        x02 = np.full(hm.shape, np.nan)
        idx1, idy1 = np.where(np.diff(np.sign(P1)) > 0)
        idx2, idy2 = np.where(np.diff(np.sign(P2)) > 0)

        x01[idx1] = x0[idy1] + (-P1[idx1, idy1]) / (P1[idx1, idy1] - P1[idx1, idy1+1]) * (x0[idy1+1] - x0[idy1])
        x02[idx2] = x0[idy2] + (-P2[idx2, idy2]) / (P2[idx2, idy2] - P2[idx2, idy2+1]) * (x0[idy2+1] - x0[idy2])

        y1 = flat_nested((hm - hh(x01, Em).T)*invV(x01, Em).T*np.sqrt((V(x01, Em).T)**2 - 1.0))
        y2 = flat_nested((hm - hh(x02, Em).T)*invV(x02, Em).T*np.sqrt((V(x02, Em).T)**2 - 1.0))
        a = (y1+y2) / np.sin(theta)
        b = 2*np.sqrt(y1*y2)
        if not output_verbose:
            return a, b
        else:
            x1 = flat_nested(x01 + ( (hm - hh(x01, Em).T)*invV(x01, Em).T ))
            x2 = flat_nested(x02 + ( (hm - hh(x02, Em).T)*invV(x02, Em).T ))
            comp_results = {
                "x01": x01, "x02": x02,
                "xx": xx, "yy": yy,
                "x1": x1, "x2": x2,
                "y1": y1, "y2": y2,
                "a": a, "b": b,
                "hm": hm, "Em": Em,
                "theta": theta
            }
            return comp_results

    def model(self, E_array, alpha, beta, theta):
        E_array = np.asarray(E_array)
        a_model = np.zeros_like(E_array)
        b_model = np.zeros_like(E_array)
        for i, E in enumerate(E_array):
            a_i, b_i = self.compute_a_b(alpha, beta, theta, E, self.h_exp)
            a_model[i] = a_i
            b_model[i] = b_i
        model_output = np.concatenate([a_model, b_model])
        if not np.all(np.isfinite(model_output)):
            print(f"Non-finite model output for alpha={alpha}, beta={beta}, theta={theta}")
        return model_output

    def fit(self, curve_fit_params, verbose=True, return_ab=True, return_ab_E_range=(None, None, None), return_ab_theta_range=(0, np.pi/2, None), is_save=False, save_path=None):
        E_exp, a_exp, b_exp, h_exp = self.E_exp, self.a_exp, self.b_exp, self.h_exp
        xdata = E_exp
        ydata = np.concatenate([a_exp, b_exp])
        if verbose:
            start = time.time()
        popt, pcov = curve_fit(lambda E, alpha, beta, theta: self.model(E, alpha, beta, theta), xdata, ydata, **curve_fit_params)
        alpha_fit, beta_fit, theta_fit = popt
        alpha_dev, beta_dev, theta_dev = np.sqrt(np.diag(pcov))
        if verbose:
            print(f"alpha = {alpha_fit:.3e}, beta = {beta_fit:.3f}, theta = {np.degrees(theta_fit):.2f} deg")
            print(f"alpha_dev = {alpha_dev:.3e}, beta_dev = {beta_dev:.3f}, theta_dev = {np.degrees(theta_dev):.2f} deg")
            print(f"time taken: {time.time() - start:.2f} seconds")
        if verbose:
            print("Fitting completed successfully. Now generating fitting results.")
        E_for_REL = self.E_for_REL
        ab_fit = self.model(E_for_REL, alpha_fit, beta_fit, theta_fit)
        a_fit = ab_fit[:len(E_for_REL)]  # Major axis (a)
        b_fit = ab_fit[len(E_for_REL):]  # Minor axis (b)
        Results = {
            "alpha_fit": alpha_fit,
            "beta_fit": beta_fit,
            "theta_fit": theta_fit,
            "alpha_dev": alpha_dev,
            "beta_dev": beta_dev,
            "theta_dev": theta_dev,
            "E_fit": E_for_REL,
            "a_fit": a_fit,
            "b_fit": b_fit,
        }
        if return_ab:
            if verbose:
                print("Returning fitted a and b values versus energy and theta.")
            E_start = return_ab_E_range[0] if return_ab_E_range[0] is not None else E_for_REL[0]
            E_stop  = return_ab_E_range[1] if return_ab_E_range[1] is not None else E_for_REL[-1]
            E_step = return_ab_E_range[2] if return_ab_E_range[2] is not None else (E_for_REL[1] - E_for_REL[0])
            E_fit = np.arange(E_start, E_stop + E_step, E_step)
            theta_start = return_ab_theta_range[0] if return_ab_theta_range[0] is not None else 0
            theta_stop  = return_ab_theta_range[1] if return_ab_theta_range[1] is not None else np.pi/2
            theta_step = return_ab_theta_range[2] if return_ab_theta_range[2] is not None else (np.pi/2) / len(E_for_REL)
            theta_fit = np.arange(theta_start, theta_stop + theta_step, theta_step)
            E_mesh, theta_mesh = np.meshgrid(E_fit, theta_fit, indexing='ij')
            start = time.time()
            ab_E_theta = np.zeros((len(E_fit), len(theta_fit), 2))  # shape (N, M, 2) for a and b
            for i in range(len(theta_fit)):
                ab_fit = self.model(E_fit, alpha_fit, beta_fit, theta_fit[i])
                ab_E_theta[:, i, 0] = ab_fit[:len(E_fit)]  # a_fit
                ab_E_theta[:, i, 1] = ab_fit[len(E_fit):]  # b_fit
            a_mesh = ab_E_theta[:, :, 0]
            b_mesh = ab_E_theta[:, :, 1]
            Results["E_mesh"] = E_mesh
            Results["theta_mesh"] = theta_mesh
            Results["a_mesh"] = a_mesh
            Results["b_mesh"] = b_mesh
            if verbose:
                print(f"Constructed a and b meshes in {time.time() - start:.2f} seconds.")
        else:
            if verbose:
                print("Returning only fitted parameters without a and b meshes.")
        if is_save:
            if save_path is None:
                save_path = ETCHING_DATA_DIR / "etching_rate_ratio_fit_results.npz"
            save_path = absolute_path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(save_path), **Results)
            if verbose:
                print(f"Results saved to {save_path}")
        self.fit_results = Results
        return Results

    def inverse_lookup(self, a_target, b_target, error_len=None, tol_energy=0.05, tol_theta=0.05, refine_energy=1500, refine_theta=1500):
        """
        Vectorized inverse lookup: for each (a_target, b_target), find (E, theta) pairs where the modeled (a, b) are close to target values.
        Uses fine interpolation and local minima search for robust multi-target support.

        Parameters:
            a_target (array-like): Target major axis values (um).
            b_target (array-like): Target minor axis values (um).
            error_len (float): Tolerance in (a, b) space for local minima filtering.
            tol_energy (float): Tolerance in energy axis for local minima filtering.
            tol_theta (float): Tolerance in theta axis for local minima filtering.
            refine_energy (int): Number of points for fine energy grid.
            refine_theta (int): Number of points for fine theta grid.
        Returns:
            lookup (ndarray): Array of (target_idx, E, theta) for each found solution.
        """

        if self.fit_results is None:
            raise ValueError("No fit results available. Please run fit() first.")
        a_mesh    = self.fit_results["a_mesh"]
        b_mesh    = self.fit_results["b_mesh"]
        E_mesh    = self.fit_results["E_mesh"]
        theta_mesh = self.fit_results["theta_mesh"]

        a_target = np.atleast_1d(a_target)
        b_target = np.atleast_1d(b_target)
        if a_target.shape != b_target.shape:
            raise ValueError("a_target and b_target must have the same shape")
        if error_len is None:
            error_len = 0.01 * np.sqrt(a_target**2 + b_target**2)
        else:
            error_len = np.broadcast_to(error_len, a_target.shape)

        # Fine grids
        E_grid_fine    = np.linspace(E_mesh[:, 0].min(),    E_mesh[:, 0].max(),    refine_energy)
        theta_grid_fine = np.linspace(theta_mesh[0, :].min(), theta_mesh[0, :].max(), refine_theta)

        # Interpolate a_mesh and b_mesh onto refined grid using cv2
        a_fine = cv2.resize(a_mesh, (len(theta_grid_fine), len(E_grid_fine)), interpolation=cv2.INTER_LINEAR)
        b_fine = cv2.resize(b_mesh, (len(theta_grid_fine), len(E_grid_fine)), interpolation=cv2.INTER_LINEAR)
        # a_fine, b_fine shape: (refine_energy, refine_theta)

        # Vectorized distance maps in (a, b) space for each target pair
        a_fine_exp = a_fine[None, :, :]  # (1, refine_energy, refine_theta)
        b_fine_exp = b_fine[None, :, :]  # (1, refine_energy, refine_theta)
        a_target_exp = a_target[:, None, None]  # (n_targets, 1, 1)
        b_target_exp = b_target[:, None, None]  # (n_targets, 1, 1)
        error_maps = np.sqrt((a_fine_exp - a_target_exp)**2 + (b_fine_exp - b_target_exp)**2)  # (n_targets, refine_energy, refine_theta)

        # Minimum filter to find local minima within tolerance for each target (vectorized)
        dE    = E_grid_fine[1]    - E_grid_fine[0]
        dth   = theta_grid_fine[1] - theta_grid_fine[0]
        size_energy = max(1, int(tol_energy / dE)*2)
        size_theta  = max(1, int(tol_theta  / dth)*2)
        size = (size_energy, size_theta)  # Minimum filter size in (energy, theta) space

        error_maps_filtered = np.empty_like(error_maps)
        for i in range(error_maps.shape[0]):
            error_maps_filtered[i] = minimum_filter(error_maps[i], size=size, mode='reflect')

        # Find mask of local minima within tolerance for all targets at once
        mask = (error_maps_filtered == error_maps) & (error_maps < error_len[:, None, None])  # (n_targets, refine_energy, refine_theta)

        lookup = np.zeros((mask.sum(), 4), dtype=np.float64)  # shape (n_points, 3) for (target_idx, energy, theta, distance)
        pts = np.where(mask)  # Get indices of local minima

        lookup[:, 0] = pts[0]  # Target index
        lookup[:, 1] = E_grid_fine[pts[1]]  # Energy value
        lookup[:, 2] = theta_grid_fine[pts[2]]  # Theta value
        lookup[:, 3] = error_maps[pts]

        return lookup

    def provide_results(self, results):
        self.fit_results = results
