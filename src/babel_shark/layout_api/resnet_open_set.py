# -*- coding: utf-8 -*-
"""
resnet_open_set.py — ResNet-based Open-Set layout classifier adapter.

This adapter bridges the external `layout.label_layout_classifier`
implementation with the internal `Classifier` protocol interface.

Responsibilities:
    • Load prototype index (.npz) and threshold configuration (.json).
    • Initialize the underlying layout classifier (`LayoutClassifier`).
    • Merge runtime parameters with stored thresholds.
    • Expose a unified `predict_image` method returning a structured `Prediction`.

Designed to serve as the default backend for ROI layout selection and
group prediction tasks within the Babble-WSI-Babel-Shark architecture.
"""

from pathlib import Path
import json
from typing import Optional, Dict
from PIL import Image
from .types import Prediction
from .base import Classifier

# Import your existing classifier (single-file)
from layout.label_layout_classifier import PrototypeIndex, LayoutClassifier as _LLC


class ResNetOpenSetAdapter(Classifier):
    def __init__(self, model_dir: Path, params: Optional[Dict] = None):
        self.model_dir = Path(model_dir)
        self.idx_path = self.model_dir / "index.npz"
        self.thr_path = self.model_dir / "thresholds.json"

        self._idx = PrototypeIndex.load(self.idx_path)
        self._clf = _LLC(self._idx)

        self._thr = {}
        if self.thr_path.exists():
            self._thr = json.loads(self.thr_path.read_text(encoding="utf-8"))

        self._params = dict(
            crop_left_percent=1.0,
            alpha=0.60, beta=0.20, gamma=0.20,
            tau=0.92, margin=0.015, tau_high=0.98,
            fast_accept=0.955, top3_gap=0.03,
            per_class_tau=self._thr.get("tau"),
            per_class_margin=self._thr.get("margin"),
        )
        if params:
            for k, v in params.items():
                if k in self._params:
                    self._params[k] = v

    def predict_image(self, img: Image.Image) -> Prediction:
        p = self._params
        dec = self._clf.predict(
            img,
            crop_left_percent=p["crop_left_percent"],
            alpha=p["alpha"], beta=p["beta"], gamma=p["gamma"],
            tau=p["tau"], margin=p["margin"], tau_high=p["tau_high"],
            fast_accept=p["fast_accept"], top3_gap=p["top3_gap"],
            per_class_tau=p["per_class_tau"], per_class_margin=p["per_class_margin"],
        )
        return Prediction(
            label=dec.label,
            score=float(dec.score),
            top3=[(g, float(s)) for g, s in dec.top_scores],
            reason=dec.reason
        )
