# -*- coding: utf-8 -*-
"""ROI structures and JSON loader.

This module defines:
  `ROIBox`: a simple rectangle container with name and float coordinates.
  `load_rois_from_json`: a tolerant loader that accepts either a list of ROI
  dicts or a dict mapping names → coordinate dicts. Coordinates may be
  provided as (left, top, right, bottom) or as (x, y, width, height).
"""

import json
from typing import Dict, List, Optional, Tuple, Union

class ROIBox:
    """Axis-aligned rectangular region-of-interest with a name.

    Attributes:
        name: Label/name of the ROI.
        left: Left (x_min) coordinate.
        top: Top (y_min) coordinate.
        right: Right (x_max) coordinate.
        bottom: Bottom (y_max) coordinate.
    """
    def __init__(self, name: str, left: float, top: float, right: float, bottom: float) -> None:
        """Initialize a new ROIBox.

        Args:
            name: ROI label.
            left: Left coordinate.
            top: Top coordinate.
            right: Right coordinate.
            bottom: Bottom coordinate.
        """
        self.name = name
        self.left = float(left)
        self.top = float(top)
        self.right = float(right)
        self.bottom = float(bottom)

def load_rois_from_json(json_path: str) -> List["ROIBox"]:
    """Load ROI definitions from a JSON file into a list of ROIBox.

    Accepted JSON formats:
      1) List of ROI dicts:
            [{"name": "A", "left": 0, "top": 0, "right": 10, "bottom": 5}, ...]
         or using x/y/width/height:
            [{"name": "A", "x": 0, "y": 0, "width": 10, "height": 5}, ...]
      2) Dict mapping ROI name → coordinate dict:
            {"A": {"left": 0, "top": 0, "right": 10, "bottom": 5}, ...}
         or with x/y/width/height keys.

    Args:
        json_path: Path to the JSON file.

    Returns:
        List of ROIBox instances parsed from the file.

    Raises:
        ValueError: If format is unsupported or coordinates are missing.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def to_box(item_or_name: Union[str, Dict[str, float]],
               d: Optional[Dict[str, float]] = None) -> ROIBox:
        """Convert one JSON entry (list- or dict-style) into an ROIBox.

        Args:
            item_or_name: Either an ROI dict (list-style) or the ROI name (dict-style).
            d: When provided (dict-style), the coordinate dict for the given name.

        Returns:
            ROIBox constructed from the provided entry.

        Raises:
            ValueError: If required fields ('name' or coordinates) are missing.
        """
        if d is None:
            d = item_or_name  # type: ignore[assignment]
            name = d.get("name") or d.get("label")  # type: ignore[union-attr]
        else:
            name = item_or_name  # type: ignore[assignment]
        if name is None:
            raise ValueError(f"ROI missing 'name' in {json_path}")
        if all(k in d for k in ("left", "top", "right", "bottom")):  # type: ignore[arg-type]
            L, T, R, B = d["left"], d["top"], d["right"], d["bottom"]  # type: ignore[index]
        elif all(k in d for k in ("x", "y", "width", "height")):  # type: ignore[arg-type]
            L, T = d["x"], d["y"]  # type: ignore[index]
            R, B = L + d["width"], T + d["height"]  # type: ignore[index]
        else:
            raise ValueError(f"ROI '{name}' missing coordinates in {json_path}")
        return ROIBox(str(name), float(L), float(T), float(R), float(B))

    rois: List[ROIBox] = []
    if isinstance(data, list):
        for it in data:
            rois.append(to_box(it))
    elif isinstance(data, dict):
        for nm, d in data.items():
            rois.append(to_box(nm, d))
    else:
        raise ValueError(f"Unsupported ROI JSON format: {type(data)}")
    return rois
