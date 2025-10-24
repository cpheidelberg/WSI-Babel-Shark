# -*- coding: utf-8 -*-
"""Utility helpers for filesystem I/O, timestamps, image discovery, and YAML config loading.

This module offers:
  `ensure_dir`: create a directory tree (idempotent).
  `_timestamp`: generate a run folder-friendly timestamp string.
  `_atomic_write_csv`: atomically write a CSV file.
  `_atomic_write_excel`: atomically write an Excel file with multiple sheets.
  `_find_images`: recursively collect image paths by extension filter.
  `_load_yaml_config`: load a YAML configuration into a Python dict.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import pandas as pd
import yaml

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
"""Set[str]: Allowed image filename extensions used by `_find_images`."""

def ensure_dir(path: str) -> str:
    """Create `path` directory (and parents) if missing; return the same path.

    Args:
        path: Directory path to create.

    Returns:
        The input `path`, unchanged.
    """
    os.makedirs(path, exist_ok=True)
    return path

def _timestamp() -> str:
    """Return current local time formatted as 'YYYY-MM-DD_HH-MM-SS'.

    Useful for timestamped run/output directories.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def _atomic_write_csv(df: pd.DataFrame, out_path: str) -> None:
    """Atomically write a DataFrame to CSV (UTF-8, no index).

    Writes to `out_path + ".tmp"` first, then replaces the destination.

    Args:
        df: DataFrame to serialize.
        out_path: Destination CSV path.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, out_path)

def _atomic_write_excel(dfs: Dict[str, pd.DataFrame], out_path: str) -> None:
    """Atomically write multiple DataFrames as an Excel workbook.

    Each key in `dfs` becomes a sheet name; values are written without index.

    Args:
        dfs: Mapping of sheet_name → DataFrame.
        out_path: Destination XLSX path.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path + ".tmp"
    with pd.ExcelWriter(tmp) as writer:
        for sheet, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
    os.replace(tmp, out_path)

def _find_images(input_dir: str) -> List[str]:
    """Recursively find image files under `input_dir` filtered by `IMG_EXTS`.

    Args:
        input_dir: Root directory to search.

    Returns:
        Sorted list of file paths with allowed image extensions.
    """
    files: List[str] = []
    for root, _, fnames in os.walk(input_dir):
        for fn in fnames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                files.append(os.path.join(root, fn))
    return sorted(files)

def _load_yaml_config(path: str) -> Dict:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration as a dict (empty dict if the file is empty).
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
