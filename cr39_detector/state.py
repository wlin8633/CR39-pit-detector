"""Versioned, non-pickle state files for the CR-39 detector GUI."""

from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import pickle
from typing import Any, Dict, Mapping, Optional, Tuple, Union
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

import numpy as np

from .paths import absolute_path


STATE_SCHEMA_VERSION = 1
STATE_METADATA_NAME = "state.json"
STATE_ARRAYS_NAME = "arrays.npz"


class StateFormatError(ValueError):
    """Raised when a ``.cr39state`` file is missing or malformed."""


def save_state_bundle(
    path: Union[os.PathLike, str],
    settings: Mapping[str, Any],
    arrays: Optional[Mapping[str, np.ndarray]] = None,
) -> Path:
    """Atomically save JSON settings and optional NumPy arrays to a ZIP bundle."""

    target = absolute_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")

    safe_arrays = {
        name: value
        for name, value in (arrays or {}).items()
        if isinstance(value, np.ndarray)
    }
    metadata = {
        "schema_version": STATE_SCHEMA_VERSION,
        "settings": dict(settings),
        "array_names": sorted(safe_arrays),
    }

    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(
                STATE_METADATA_NAME,
                json.dumps(metadata, indent=2, sort_keys=True),
            )
            if safe_arrays:
                buffer = BytesIO()
                np.savez_compressed(buffer, **safe_arrays)
                archive.writestr(STATE_ARRAYS_NAME, buffer.getvalue())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()

    return target


def load_state_bundle(
    path: Union[os.PathLike, str],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Load a ``.cr39state`` bundle without enabling NumPy object loading."""

    source = absolute_path(path)
    try:
        with ZipFile(source, "r") as archive:
            try:
                metadata = json.loads(archive.read(STATE_METADATA_NAME))
            except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise StateFormatError("Missing or invalid state.json") from exc

            version = metadata.get("schema_version")
            if version != STATE_SCHEMA_VERSION:
                raise StateFormatError(
                    f"Unsupported state schema {version!r}; expected {STATE_SCHEMA_VERSION}."
                )
            settings = metadata.get("settings")
            if not isinstance(settings, dict):
                raise StateFormatError("The state settings must be a JSON object.")

            arrays: Dict[str, np.ndarray] = {}
            if STATE_ARRAYS_NAME in archive.namelist():
                with np.load(
                    BytesIO(archive.read(STATE_ARRAYS_NAME)), allow_pickle=False
                ) as loaded:
                    arrays = {name: loaded[name] for name in loaded.files}
    except (BadZipFile, OSError) as exc:
        raise StateFormatError(f"Could not read state file: {source}") from exc

    return settings, arrays


def load_legacy_pickle(path: Union[os.PathLike, str]) -> Dict[str, Any]:
    """Import a trusted legacy ``.sav`` file.

    Pickle can execute code while loading. Callers must only use this function
    for files they created themselves or otherwise trust.
    """

    source = absolute_path(path)
    with source.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, dict):
        raise StateFormatError("The legacy state must contain a dictionary.")
    return data
