# -*- coding: utf-8 -*-
# @file getDataFromGeant4.py
# @brief Utility functions to extract data from Geant4 simulation output files.
# @date 2025-06-24
# @author Wei Lin
# @license MIT License

"""
getDataFromGeant4.py
"""

# %% ----------------------------------------------------------------------
# ---> Imports
# -------------------------------------------------------------------------
import os
import numpy as np
# %% ----------------------------------------------------------------------
# ---> Get data from Geant4 output files
# -------------------------------------------------------------------------
def get_data_from_Geant4_output(filepath, column_index):
    """
    Extracts a specific column of data from a Geant4 output file.
    
    Parameters:
    filepath (str): Path to the Geant4 output file.
    column_index (int): Index of the column to extract (0-based).
    
    Returns:
    np.ndarray: The extracted column data as a NumPy array.
    """
    data = np.genfromtxt(filepath, delimiter=',')
    if data.ndim == 1:
        return data[column_index]
    else:
        return data[:, column_index]
# %% ----------------------------------------------------------------------
# ---> Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage
    example_filepath = fr"{os.path.join(os.path.dirname(__file__))}/examples/Geant4_output_example.csv"
    example_column_index = 1  # Change this to the desired column index
    example_data = get_data_from_Geant4_output(example_filepath, example_column_index)
    print("Extracted data from column", example_column_index, ":\n", example_data)

# %%
