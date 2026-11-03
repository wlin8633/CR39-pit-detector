"""Central absolute paths used by the application."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_PATH = (PROJECT_ROOT / "example.bmp").resolve()
DETECTION_WORK_DIR = (PROJECT_ROOT / "savFiles_pitDetection").resolve()
ETCHING_DATA_DIR = (PROJECT_ROOT / "savFiles_etchingModel").resolve()

_runtime_parent = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share"))
RUNTIME_DIR = (_runtime_parent / "CR39PitDetector").resolve()
DEFAULT_AUTOSAVE_PATH = (RUNTIME_DIR / "autosave.cr39state").resolve()


def absolute_path(value: Union[str, os.PathLike], *, base: Path = PROJECT_ROOT) -> Path:
    """Resolve a user or saved path to an absolute path without requiring it to exist."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def safe_generated_directory(value: Union[str, os.PathLike]) -> Path:
    """Resolve a generated-output directory and reject dangerously broad targets."""
    path = absolute_path(value)
    forbidden = {
        Path(path.anchor).resolve(),
        Path.home().resolve(),
        PROJECT_ROOT,
    }
    if path in forbidden:
        raise ValueError(f"Refusing to use broad generated-output directory: {path}")
    return path
