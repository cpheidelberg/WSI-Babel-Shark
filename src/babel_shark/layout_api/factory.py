# -*- coding: utf-8 -*-
"""
factory.py — Model loading and adapter factory.

This module provides a unified `load_model` entry point that instantiates
a classifier adapter according to a configuration dictionary.

Supported backends:
    • "resnet-open-set" — ResNet-based open-set ROI layout classifier.

Example configuration:
    cfg = {
        "backend": "resnet-open-set",
        "model_dir": "./model",
        "params": {"alpha": 0.6, "beta": 0.2, "tau": 0.92}
    }

Returns a `Classifier`-compliant object for downstream ROI extraction modules.
"""

from pathlib import Path
from typing import Dict
from .base import Classifier
from .resnet_open_set import ResNetOpenSetAdapter


def load_model(cfg: Dict) -> Classifier:
    """
    cfg example:
    {
      "backend": "resnet-open-set",
      "model_dir": "/mnt/g/Babble-shark-v.2.1/model",
      "params": {"crop_left_percent": 1.0, "alpha": 0.60, ...}  # optional
    }
    """
    backend = cfg.get("backend", "resnet-open-set")
    model_dir = Path(cfg["model_dir"])
    params = cfg.get("params", {})

    if backend == "resnet-open-set":
        return ResNetOpenSetAdapter(model_dir, params)

    raise ValueError(f"Unknown backend: {backend}")
