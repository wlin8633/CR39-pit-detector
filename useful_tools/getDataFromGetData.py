# -*- coding: utf-8 -*-
# @file getDataFromGetData.py
# @brief Utility functions to extract data from GetData output files.
# @date 2025-06-24
# @author Wei Lin
# @license MIT License

"""
getDataFromGetData.py
"""

# %% ----------------------------------------------------------------------
# ---> Imports
# -------------------------------------------------------------------------
import os
import numpy as np
# %% ----------------------------------------------------------------------
# ---> Get data from Geant4 output files
# -------------------------------------------------------------------------
def get_data_from_GetData(filename, linest="Line", lineend="\n", dtype=np.float32):
    """
    Extracts data from a GetData output file.
    This function reads a file formatted by GetData, extracting lines that start with
    a specified string (default "Line") and ending with a specified string (default "\n").
    It collects the data into a list of NumPy arrays, converting them to the specified data type (default np.float32).
    The function handles multiple lines of data, returning a single array if only one line is found,
    or a list of arrays if multiple lines are present.
    
    Parameters:
        filename (str): Path to the GetData output file.
        linest (str, optional): String that indicates the start of a data line. Defaults to "Line".
        lineend (str, optional): String that indicates the end of a data line. Defaults to "\n".
        dtype (type, optional): Data type to which the extracted data should be converted. Defaults to np.float32.
        
    Returns:
        np.ndarray or list of np.ndarray: Extracted data as a NumPy array or a list of arrays if multiple lines are found.
    """
    with open(filename) as f:
        data = []; LineNum = 0
        is_readData = False
        for line in f.readlines():
            if linest in line and is_readData == False:
                is_readData = True
                LineNum += 1
                data.append([])
                continue
            elif line == lineend and is_readData == True:
                is_readData = False
                data[LineNum-1] = np.array(data[LineNum-1], dtype=dtype)
                continue
                
            if is_readData:
                data[LineNum-1].append( line.split("   ") )
                
        data[LineNum-1] = np.array(data[LineNum-1], dtype=dtype)  
    return data[0] if LineNum == 1 else data
# %% ----------------------------------------------------------------------
# ---> Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage
    example_filepath = fr"{os.path.join(os.path.dirname(__file__))}/examples/GetData_example.txt"
    example_data = get_data_from_GetData(example_filepath)
    print("Extracted data:\n", example_data)
# %%
