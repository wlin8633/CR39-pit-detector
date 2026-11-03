"""Small parsers for Geant4 and GetData text exports."""

from pathlib import Path
from typing import Sequence, Union

import numpy as np


def get_data_from_Geant4_output(filepath, column_index: Union[int, Sequence[int]]):
    """Read selected zero-based columns from a comma-separated Geant4 output."""
    data = np.genfromtxt(Path(filepath).resolve(), delimiter=",")
    if data.ndim == 1:
        return data[column_index]
    return data[:, column_index]


def get_data_from_GetData(
    filename, linest="Line", lineend="\n", dtype=np.float32
):
    """Read one or more data blocks from a GetData text export."""
    blocks = []
    current = None
    with Path(filename).resolve().open(encoding="utf-8") as stream:
        for line in stream:
            if linest in line and current is None:
                current = []
                continue
            if line == lineend and current is not None:
                blocks.append(np.array(current, dtype=dtype))
                current = None
                continue
            if current is not None:
                current.append(line.split("   "))

    if current is not None:
        blocks.append(np.array(current, dtype=dtype))
    if not blocks:
        raise ValueError(f"No data blocks beginning with {linest!r} were found")
    return blocks[0] if len(blocks) == 1 else blocks
