# -*- coding: utf-8 -*-
"""
Open-set Universal Layout Classifier

Summary
- Fuses three similarities: ResNet-18 embedding, average-image vector, and a hand-crafted
  layout descriptor (HSV/edges/FFT/ink). Open-set decision with per-class thresholds.

Provides
- build_index(), learn_thresholds(), validate_open_set(), predict_one(), predict_batch().

Quick usage
- Build:    python this_file.py build-index --data_dir train --save_path model/index.npz
- Learn:    python this_file.py learn-thresholds --data_dir val --index_path model/index.npz --out_json model/thresholds.json
- Predict:  python this_file.py predict-batch --in_dir imgs --index_path model/index.npz --out_csv results/preds.csv

Notes
- Optional analytics (confusion matrix/report/plots) require pandas, scikit-learn, matplotlib, seaborn.
- See README.md for details on thresholds, artifacts, and recommended configs.
"""

from __future__ import annotations

# ===========================
# Standard library imports
# ===========================
import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ===========================
# Third-party imports
# ===========================
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

# ===========================
# Constants & Defaults
# ===========================
VERSION = "1.0.0"

SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# Decision/fusion defaults centralized here (single source of truth)
DEF_ALPHA, DEF_BETA, DEF_GAMMA = 0.60, 0.20, 0.20
DEF_TAU, DEF_MARGIN, DEF_TAU_HIGH = 0.92, 0.015, 0.98
DEF_FAST_ACCEPT, DEF_TOP3_GAP = 0.955, 0.03
DEF_CROP_LEFT = 1.0

# Logger
LOGGER = logging.getLogger("layout")

# ---------------------------
# Utils
# ---------------------------

def list_images(folder: Path) -> List[Path]:
    """Recursively list image files with supported extensions."""
    out: List[Path] = []
    if not folder.exists():
        return out
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            out.append(p)
    return out


def list_labels(folder: Path) -> List[str]:
    """Return direct child folders that contain at least one image file as labels."""
    labels: List[str] = []
    if not folder.exists():
        return labels
    for sub in sorted([d for d in folder.iterdir() if d.is_dir()]):
        if any(x.suffix.lower() in SUPPORTED_EXT for x in sub.glob("*")):
            labels.append(sub.name)
    return labels


def load_image(path: Path, mode: Optional[str] = None) -> Image.Image:
    """Load image and (optionally) convert mode ('RGB' or 'L')."""
    im = Image.open(path)
    if mode:
        im = im.convert(mode)
    return im


def crop_left(img: Image.Image, left_percent: float) -> Image.Image:
    """Crop left-most percent of the image (0<left_percent<=1). If >=1.0, returns original."""
    if left_percent >= 1.0:
        return img
    w, h = img.size
    lw = int(max(1, round(w * left_percent)))
    return img.crop((0, 0, lw, h))


def pil_to_unit_gray(img: Image.Image) -> np.ndarray:
    """Convert PIL image to float32 grayscale array in [0,1]."""
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    return g


def l2_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2 normalize a vector with numerical stability."""
    n = np.linalg.norm(v) + eps
    return v / n


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# ---------------------------
# CNN embedder
# ---------------------------
class ResNet18Embed(nn.Module):
    """ResNet-18 feature extractor (512-d) with ImageNet normalization."""
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # remove FC
        self.out_dim = 512
        self.tf = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def forward(self, img: Image.Image, device: torch.device) -> np.ndarray:
        """Return L2-normalized 512-d ResNet18 embedding."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        x = self.tf(img).unsqueeze(0).to(device)
        f = self.backbone(x).flatten(1)[0].cpu().numpy().astype(np.float32)
        return l2_normalize(f)

# ---------------------------
# Layout features (HSV + edge profiles + FFT lowfreq + ink density)
# ---------------------------
class LayoutFeature:
    """Hand-crafted layout descriptor capturing edges, HSV stats, low-freq FFT and ink density."""
    def __init__(self, gray_size: int = 128, proj_bins: int = 16, edge_grid: int = 6, hue_bins: int = 12):
        self.gray_size = gray_size
        self.proj_bins = proj_bins
        self.edge_grid = edge_grid
        self.hue_bins = hue_bins

    def _resize_gray(self, img: Image.Image) -> np.ndarray:
        g = img.convert("L").resize((self.gray_size, self.gray_size), Image.BILINEAR)
        return np.asarray(g, dtype=np.float32) / 255.0

    def _edges_simple(self, g: np.ndarray) -> np.ndarray:
        """Simple edge magnitude via finite differences."""
        gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
        gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
        e = gx + gy
        return e / (e.max() + 1e-8)

    def _downsample_1d(self, v: np.ndarray, bins: int) -> np.ndarray:
        """Uniformly resample a 1D profile to `bins`."""
        n = v.shape[0]
        x_old = np.linspace(0, 1, n, endpoint=False)
        x_new = np.linspace(0, 1, bins, endpoint=False)
        return np.interp(x_new, x_old, v)

    def _fft_lowfreq(self, g: np.ndarray, k: int = 8) -> np.ndarray:
        """Take kxk low-frequency block from log-magnitude of 2D FFT."""
        F = np.fft.fft2(g)
        M = np.log1p(np.abs(F))
        return M[:k, :k].astype(np.float32).reshape(-1)

    def _hsv_stats(self, img: Image.Image) -> np.ndarray:
        """HSV stats (means, stds) + hue histogram."""
        hsv = img.convert("HSV")
        arr = np.asarray(hsv, dtype=np.float32) / 255.0
        H, S, V = arr[..., 0], arr[..., 1], arr[..., 2]
        hist, _ = np.histogram(H, bins=self.hue_bins, range=(0.0, 1.0), density=True)
        feat = [H.mean(), S.mean(), V.mean(), H.std(), S.std(), V.std()]
        return np.concatenate([np.array(feat, dtype=np.float32), hist.astype(np.float32)], axis=0)

    def _edge_grid_density(self, E: np.ndarray, grid: int) -> np.ndarray:
        """Average edge magnitude per cell on a grid."""
        h, w = E.shape
        cell_h = h // grid
        cell_w = w // grid
        vals = []
        for iy in range(grid):
            for ix in range(grid):
                y0, y1 = iy*cell_h, (iy+1)*cell_h
                x0, x1 = ix*cell_w, (ix+1)*cell_w
                vals.append(float(E[y0:y1, x0:x1].mean()))
        return np.array(vals, dtype=np.float32)

    def _ink_density_grid(self, g: np.ndarray, gy: int = 2, gx: int = 3, thr: float = 0.3) -> np.ndarray:
        """Fraction of 'ink' (dark pixels) per grid cell."""
        h, w = g.shape
        cell_h = h // gy
        cell_w = w // gx
        mask = (g < thr).astype(np.float32)
        vals = []
        for iy in range(gy):
            for ix in range(gx):
                y0, y1 = iy*cell_h, (iy+1)*cell_h
                x0, x1 = ix*cell_w, (ix+1)*cell_w
                vals.append(float(mask[y0:y1, x0:x1].mean()))
        return np.array(vals, dtype=np.float32)

    def extract(self, img: Image.Image) -> np.ndarray:
        """Build final layout descriptor and L2-normalize it."""
        g = self._resize_gray(img)
        E = self._edges_simple(g)
        proj_y = E.sum(axis=1) / (E.shape[1] + 1e-8)
        proj_x = E.sum(axis=0) / (E.shape[0] + 1e-8)
        proj_y_ds = self._downsample_1d(proj_y, self.proj_bins)
        proj_x_ds = self._downsample_1d(proj_x, self.proj_bins)
        edge_grid = self._edge_grid_density(E, self.edge_grid)
        fft_low = self._fft_lowfreq(g, k=8)
        hsv_feat = self._hsv_stats(img)
        ink = self._ink_density_grid(g, gy=2, gx=3, thr=0.30)
        feat = np.concatenate([hsv_feat, proj_y_ds, proj_x_ds, edge_grid, fft_low, ink], axis=0).astype(np.float32)
        return l2_normalize(feat)

# ---------------------------
# Index structures
# ---------------------------
@dataclass
class ProtoEntry:
    label: str
    mean_embed: np.ndarray
    avg_vec: np.ndarray
    mean_layout: np.ndarray
    count: int

class PrototypeIndex:
    """Holds a single prototype (mean) per label for 3 modalities: CNN, average-image, layout."""
    def __init__(self):
        self.entries: Dict[str, ProtoEntry] = {}
        self.img_size = (128, 128)  # (H, W) for avg image vectorization

    def add_label(self, label: str, embeds: List[np.ndarray], avg_img: Image.Image, layout_vecs: List[np.ndarray]):
        if len(embeds) == 0 or len(layout_vecs) == 0:
            raise ValueError(f"no features for {label}")
        mean_embed = l2_normalize(np.mean(np.stack(embeds, axis=0), axis=0))
        avg_img_small = avg_img.resize(self.img_size[::-1], Image.BILINEAR)
        avg_vec = l2_normalize(pil_to_unit_gray(avg_img_small).reshape(-1))
        mean_layout = l2_normalize(np.mean(np.stack(layout_vecs, axis=0), axis=0))
        self.entries[label] = ProtoEntry(label, mean_embed, avg_vec, mean_layout, len(embeds))

    def save(self, path: Path):
        labels = list(self.entries.keys())
        data = {
            'img_h': self.img_size[0],
            'img_w': self.img_size[1],
            'labels': np.array(labels, dtype=object),
            'mean_embeds': np.stack([self.entries[l].mean_embed for l in labels]),
            'avg_vecs': np.stack([self.entries[l].avg_vec for l in labels]),
            'mean_layouts': np.stack([self.entries[l].mean_layout for l in labels]),
            'counts': np.array([self.entries[l].count for l in labels], dtype=np.int32)
        }
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path: Path) -> "PrototypeIndex":
        z = np.load(path, allow_pickle=True)
        idx = PrototypeIndex()
        idx.img_size = (int(z['img_h']), int(z['img_w']))
        labels = [str(x) for x in z['labels']]
        for i, lab in enumerate(labels):
            idx.entries[lab] = ProtoEntry(
                label=lab,
                mean_embed=z['mean_embeds'][i].astype(np.float32),
                avg_vec=z['avg_vecs'][i].astype(np.float32),
                mean_layout=z['mean_layouts'][i].astype(np.float32),
                count=int(z['counts'][i])
            )
        return idx

# ---------------------------
# Scoring & decision
# ---------------------------
@dataclass
class Decision:
    label: str
    score: float
    top_scores: List[Tuple[str,float]]
    reason: str


def fused_scores(embed: np.ndarray, img_vec: np.ndarray, layout_vec: np.ndarray,
                 index: PrototypeIndex, alpha: float, beta: float, gamma: float) -> List[Tuple[str,float]]:
    """Fuse 3 cosine similarities with weights alpha/beta/gamma and return sorted list."""
    res: List[Tuple[str, float]] = []
    for lab, e in index.entries.items():
        s1 = cosine_sim(embed, e.mean_embed)
        s2 = cosine_sim(img_vec, e.avg_vec)
        s3 = cosine_sim(layout_vec, e.mean_layout)
        res.append((lab, float(alpha*s1 + beta*s2 + gamma*s3)))
    res.sort(key=lambda x: x[1], reverse=True)
    return res


def decide_open_set(scores: List[Tuple[str,float]], tau: float = DEF_TAU, margin: float = DEF_MARGIN,
                    tau_high: float = DEF_TAU_HIGH, fast_accept: float = DEF_FAST_ACCEPT,
                    top3_gap: float = DEF_TOP3_GAP,
                    per_class_tau: Optional[Dict[str,float]] = None,
                    per_class_margin: Optional[Dict[str,float]] = None) -> Decision:
    """Open-set decision logic with per-class thresholds support."""
    best_lab, best_s = scores[0]
    second_s = scores[1][1] if len(scores) > 1 else -1.0
    third_s  = scores[2][1] if len(scores) > 2 else -1.0

    tau_g    = per_class_tau.get(best_lab, tau) if per_class_tau else tau
    margin_g = per_class_margin.get(best_lab, margin) if per_class_margin else margin

    # Reject if best score below class-specific tau
    if best_s < tau_g:
        return Decision("Unknown", best_s, scores[:3], f"best<{tau_g:.3f}")
    # Accept immediately if high enough
    if best_s >= fast_accept:
        return Decision(best_lab, best_s, scores[:3], "fast_accept")
    # Accept if top1 separated from top3
    if (best_s - third_s) >= top3_gap:
        return Decision(best_lab, best_s, scores[:3], "top3_gap")
    # Otherwise, require margin from top2 or else reject (unless extremely high)
    if (best_s - second_s) < margin_g and best_s < tau_high:
        return Decision("Unknown", best_s, scores[:3], f"margin<{margin_g:.3f} & best<{tau_high:.3f}")
    return Decision(best_lab, best_s, scores[:3], "confident")

# ---------------------------
# Classifier wrapper
# ---------------------------
class LayoutClassifier:
    """Convenience wrapper that holds the CNN and layout extractor and does prediction."""
    def __init__(self, index: PrototypeIndex, device: Optional[str] = None):
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        self.index = index
        self.cnn = ResNet18Embed().to(self.device).eval()
        self.layout = LayoutFeature()
        self.img_size = self.index.img_size

    def _img_vec(self, img: Image.Image) -> np.ndarray:
        """Vectorize grayscale-resized image (avg image modality)."""
        small = img.resize(self.img_size[::-1], Image.BILINEAR)
        return l2_normalize(pil_to_unit_gray(small).reshape(-1))

    def predict(self, img: Image.Image, crop_left_percent: float = DEF_CROP_LEFT,
                alpha: float = DEF_ALPHA, beta: float = DEF_BETA, gamma: float = DEF_GAMMA,
                tau: float = DEF_TAU, margin: float = DEF_MARGIN, tau_high: float = DEF_TAU_HIGH,
                fast_accept: float = DEF_FAST_ACCEPT, top3_gap: float = DEF_TOP3_GAP,
                per_class_tau: Optional[Dict[str, float]] = None,
                per_class_margin: Optional[Dict[str, float]] = None) -> Decision:
        """Run all modalities and apply open-set decision."""
        if crop_left_percent < 1.0:
            img = crop_left(img, crop_left_percent)
        emb = self.cnn(img, self.device)
        vec = self._img_vec(img)
        lay = self.layout.extract(img)
        scores = fused_scores(emb, vec, lay, self.index, alpha, beta, gamma)
        return decide_open_set(scores, tau, margin, tau_high, fast_accept, top3_gap,
                               per_class_tau, per_class_margin)

# ---------------------------
# Build index
# ---------------------------

def build_index(data_dir: Path, avg_dir: Optional[Path], save_path: Path, crop_left_percent: float = DEF_CROP_LEFT) -> bool:
    """Build mean prototypes per label and save a compressed index."""
    labels = list_labels(data_dir)
    if not labels:
        LOGGER.error("no labeled subfolders in %s", data_dir)
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = ResNet18Embed().to(device).eval()
    layout = LayoutFeature()
    idx = PrototypeIndex()

    for lab in labels:
        folder = data_dir / lab
        embeds: List[np.ndarray] = []
        layout_vecs: List[np.ndarray] = []
        avg_accum = None
        n = 0
        for p in tqdm(list_images(folder), desc=f"{lab}"):
            try:
                img = load_image(p, mode="RGB")
                if crop_left_percent < 1.0:
                    img = crop_left(img, crop_left_percent)
                emb = cnn(img, device)
                lays = layout.extract(img)
                embeds.append(emb)
                layout_vecs.append(lays)
                arr = pil_to_unit_gray(img)
                arr_small = Image.fromarray((arr*255).astype(np.uint8)).resize(idx.img_size[::-1], Image.BILINEAR)
                arr_small = np.asarray(arr_small, dtype=np.float32)/255.0
                if avg_accum is None:
                    avg_accum = np.zeros_like(arr_small)
                avg_accum += arr_small
                n += 1
            except (OSError, ValueError) as e:
                LOGGER.warning("Skip %s due to error: %s", p, e)
            except Exception as e:
                LOGGER.warning("Unexpected error on %s: %s", p, e)
        if not embeds:
            LOGGER.warning("no images for label %s", lab)
            continue
        # Prefer external average PNG if exists, otherwise use accumulated mean
        avg_img_path = (avg_dir / f"{lab}.png") if avg_dir else None
        if avg_img_path and avg_img_path.exists():
            avg_img = load_image(avg_img_path, mode="L")
        else:
            avg_img = Image.fromarray((np.clip(avg_accum/max(1, n), 0, 1)*255).astype(np.uint8)).convert("L")
        idx.add_label(lab, embeds, avg_img, layout_vecs)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    idx.save(save_path)
    LOGGER.info("Saved index → %s", save_path)
    return True

# ---------------------------
# Learn per-class thresholds
# ---------------------------

def learn_thresholds(data_dir: Path, index_path: Path, crop_left_percent: float = DEF_CROP_LEFT,
                     alpha: float = DEF_ALPHA, beta: float = DEF_BETA, gamma: float = DEF_GAMMA,
                     percentile: float = 0.05, margin_percentile: float = 0.10, out_json: Path = Path("thresholds.json")) -> bool:
    """Estimate class-specific tau and margin from validation set (correct predictions only)."""
    idx = PrototypeIndex.load(index_path)
    clf = LayoutClassifier(idx)

    labels = list_labels(data_dir)
    per_scores: Dict[str, List[float]] = {l: [] for l in labels}
    per_gaps: Dict[str, List[float]] = {l: [] for l in labels}

    for lab in labels:
        folder = data_dir / lab
        for p in tqdm(list_images(folder), desc=f"LEARN {lab}"):
            try:
                img = load_image(p, mode="RGB")
                if crop_left_percent < 1.0:
                    img = crop_left(img, crop_left_percent)
                emb = clf.cnn(img, clf.device)
                vec = clf._img_vec(img)
                lay = clf.layout.extract(img)
                scores = fused_scores(emb, vec, lay, clf.index, alpha, beta, gamma)
                best_lab, best_s = scores[0]
                second_s = scores[1][1] if len(scores) > 1 else -1.0
                if best_lab == lab:
                    per_scores[lab].append(float(best_s))
                    per_gaps[lab].append(float(best_s - second_s))
            except (OSError, ValueError) as e:
                LOGGER.warning("Skip %s due to error: %s", p, e)
            except Exception as e:
                LOGGER.warning("Unexpected error on %s: %s", p, e)

    tau_dict: Dict[str, float] = {}
    margin_dict: Dict[str, float] = {}
    meta: Dict[str, Dict[str, float]] = {}

    for lab in labels:
        s = np.array(per_scores[lab], dtype=np.float32)
        g = np.array(per_gaps[lab], dtype=np.float32)
        if len(s) >= 5:
            tau_g = float(np.quantile(s, percentile))
            mar_g = float(np.quantile(g, max(0.0, margin_percentile)))
            tau_dict[lab] = tau_g
            margin_dict[lab] = mar_g
            meta[lab] = {"n": int(len(s)), "tau": tau_g, "margin": mar_g,
                         "mean": float(s.mean()), "std": float(s.std())}
        else:
            meta[lab] = {"n": int(len(s)), "tau": None, "margin": None}

    out = {"tau": tau_dict, "margin": margin_dict, "meta": meta,
           "percentile": percentile, "margin_percentile": margin_percentile,
           "alpha": alpha, "beta": beta, "gamma": gamma}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    LOGGER.info("Wrote per-class thresholds → %s", out_json)
    return True

# ---------------------------
# Validation (open-set)
# ---------------------------

def validate_open_set(data_dir: Path, unknown_dir: Optional[Path], index_path: Path,
                      crop_left_percent: float = DEF_CROP_LEFT, alpha: float = DEF_ALPHA,
                      beta: float = DEF_BETA, gamma: float = DEF_GAMMA,
                      tau: float = DEF_TAU, margin: float = DEF_MARGIN, tau_high: float = DEF_TAU_HIGH,
                      fast_accept: float = DEF_FAST_ACCEPT, top3_gap: float = DEF_TOP3_GAP,
                      thresholds_json: Optional[Path] = None) -> bool:
    """Validate per-class accuracy (knowns) and Unknown TPR (if unknown_dir provided)."""
    per_tau = per_margin = None
    if thresholds_json is not None and thresholds_json.exists():
        conf = json.loads(thresholds_json.read_text(encoding='utf-8'))
        per_tau = conf.get('tau', None)
        per_margin = conf.get('margin', None)

    idx = PrototypeIndex.load(index_path)
    clf = LayoutClassifier(idx)

    labels = [l for l in list_labels(data_dir) if l != 'unknown']

    total = correct = unknown = 0
    for lab in labels:
        folder = data_dir / lab
        for p in tqdm(list_images(folder), desc=f"VAL {lab}"):
            img = load_image(p, mode="RGB")
            if crop_left_percent < 1.0:
                img = crop_left(img, crop_left_percent)
            dec = clf.predict(img, crop_left_percent=crop_left_percent, alpha=alpha, beta=beta, gamma=gamma,
                              tau=tau, margin=margin, tau_high=tau_high,
                              fast_accept=fast_accept, top3_gap=top3_gap,
                              per_class_tau=per_tau, per_class_margin=per_margin)
            total += 1
            if dec.label == lab:
                correct += 1
            if dec.label == "Unknown":
                unknown += 1
    acc = correct / max(1, total)

    unk_total = unk_detect = 0
    if unknown_dir and Path(unknown_dir).exists():
        files = list_images(unknown_dir)
        unk_total = len(files)
        for p in tqdm(files, desc="VAL Unknown"):
            img = load_image(p, mode="RGB")
            if crop_left_percent < 1.0:
                img = crop_left(img, crop_left_percent)
            dec = clf.predict(img, crop_left_percent=crop_left_percent, alpha=alpha, beta=beta, gamma=gamma,
                              tau=tau, margin=margin, tau_high=tau_high,
                              fast_accept=fast_accept, top3_gap=top3_gap,
                              per_class_tau=per_tau, per_class_margin=per_margin)
            if dec.label == "Unknown":
                unk_detect += 1

    LOGGER.info("Open-Set Validation\n-------------------")
    LOGGER.info("Known  — total: %4d  acc: %.4f", total, acc)
    if unk_total:
        LOGGER.info("Unknown— total: %4d  TPR: %.4f", unk_total, (unk_detect / max(1, unk_total)))
    return True

# ---------------------------
# Predict (one / batch)
# ---------------------------

def get_assets(label: str, avg_dir: Optional[Path], roi_dir: Optional[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (avg_image_path, roi_json_path) if present for the predicted label."""
    a = r = None
    if avg_dir:
        p = avg_dir / f"{label}.png"
        if p.exists():
            a = p
    if roi_dir:
        q = roi_dir / f"{label}.json"
        if q.exists():
            r = q
    return a, r


def predict_one(image_path: Path, index_path: Path, avg_dir: Optional[Path], roi_dir: Optional[Path],
                crop_left_percent: float = DEF_CROP_LEFT, alpha: float = DEF_ALPHA, beta: float = DEF_BETA, gamma: float = DEF_GAMMA,
                tau: float = DEF_TAU, margin: float = DEF_MARGIN, tau_high: float = DEF_TAU_HIGH,
                fast_accept: float = DEF_FAST_ACCEPT, top3_gap: float = DEF_TOP3_GAP,
                thresholds_json: Optional[Path] = None) -> bool:
    """Predict a single image and print a JSON line with decision details."""
    per_tau = per_margin = None
    if thresholds_json is not None and thresholds_json.exists():
        conf = json.loads(thresholds_json.read_text(encoding='utf-8'))
        per_tau = conf.get('tau', None)
        per_margin = conf.get('margin', None)

    idx = PrototypeIndex.load(index_path)
    clf = LayoutClassifier(idx)
    img = load_image(image_path, mode="RGB")
    dec = clf.predict(img, crop_left_percent, alpha, beta, gamma, tau, margin, tau_high, fast_accept, top3_gap, per_tau, per_margin)
    avg_path = roi_path = None
    if dec.label != "Unknown":
        avg_path, roi_path = get_assets(dec.label, avg_dir, roi_dir)
    out = {
        'image': str(image_path),
        'pred_label': dec.label,
        'score': dec.score,
        'reason': dec.reason,
        'top3': [(g, float(s)) for g, s in dec.top_scores],
        'avg_image': str(avg_path) if avg_path else None,
        'roi_json': str(roi_path) if roi_path else None
    }
    # Keep JSON result on stdout for piping
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return True


def predict_batch(in_dir: Path, index_path: Path, avg_dir: Optional[Path], roi_dir: Optional[Path], out_csv: Path,
                  crop_left_percent: float = DEF_CROP_LEFT, alpha: float = DEF_ALPHA, beta: float = DEF_BETA, gamma: float = DEF_GAMMA,
                  tau: float = DEF_TAU, margin: float = DEF_MARGIN, tau_high: float = DEF_TAU_HIGH,
                  fast_accept: float = DEF_FAST_ACCEPT, top3_gap: float = DEF_TOP3_GAP,
                  thresholds_json: Optional[Path] = None) -> bool:
    """
    Batch predict and write a rich CSV with:
      - true_label (folder name) and file_name (basename only)
      - pred_label, score, reason
      - top1/top2/top3 (with scores)
      - boolean matches: match_pred, match_top1, match_top2, match_top3

    Also saves (if optional deps available):
      - ConfusionMatrix.csv
      - confusion_matrix.png (heatmap)
      - classification_report.xlsx (Predictions, ConfusionMatrix, PerClassReport, PerClassAccuracy)
    """
    # Lazy/optional imports for analytics and plotting
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None
    try:
        from sklearn.metrics import confusion_matrix, classification_report  # type: ignore
    except Exception:
        confusion_matrix = classification_report = None  # type: ignore
    try:
        import seaborn as sns  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        sns_ok = True
    except Exception:
        sns_ok = False

    # Load thresholds if provided
    per_tau = per_margin = None
    if thresholds_json is not None and thresholds_json.exists():
        conf = json.loads(thresholds_json.read_text(encoding='utf-8'))
        per_tau = conf.get('tau', None)
        per_margin = conf.get('margin', None)

    # Load index + classifier
    idx = PrototypeIndex.load(index_path)
    clf = LayoutClassifier(idx)

    # Collect predictions
    records: List[Dict[str, object]] = []
    y_true: List[str] = []
    y_pred: List[str] = []

    for p in tqdm(list_images(in_dir), desc="Predict"):
        try:
            img = load_image(p, mode="RGB")
            dec = clf.predict(img, crop_left_percent, alpha, beta, gamma,
                              tau, margin, tau_high, fast_accept, top3_gap,
                              per_tau, per_margin)

            true_label = p.parent.name
            file_name = p.name

            # Top-3 tuple list (pad to length 3)
            top = dec.top_scores + [("", 0.0)]*3
            top1, top2, top3_ = top[0][0], top[1][0], top[2][0]

            # Boolean matches
            match_pred  = (true_label == dec.label)
            match_top1  = (true_label == top1)
            match_top2  = (true_label == top2)
            match_top3  = (true_label == top3_)

            # Optional assets for the predicted label
            avg_path = roi_path = ""
            if dec.label != "Unknown":
                a, r = get_assets(dec.label, avg_dir, roi_dir)
                avg_path = str(a) if a else ""
                roi_path = str(r) if r else ""

            records.append({
                "true_label": true_label,
                "file_name": file_name,
                "pred_label": dec.label,
                "score": f"{dec.score:.4f}",
                "reason": dec.reason,
                "top1": f"{top1}:{top[0][1]:.3f}",
                "top2": f"{top2}:{top[1][1]:.3f}",
                "top3": f"{top3_}:{top[2][1]:.3f}",
                "match_pred": match_pred,
                "match_top1": match_top1,
                "match_top2": match_top2,
                "match_top3": match_top3,
                "avg_image": avg_path,
                "roi_json": roi_path,
            })

            y_true.append(true_label)
            y_pred.append(dec.label)

        except (OSError, ValueError) as e:
            LOGGER.warning("Skip %s due to error: %s", p, e)
            records.append({
                "true_label": p.parent.name,
                "file_name": p.name,
                "pred_label": "ERROR",
                "score": "0.0000",
                "reason": str(e),
                "top1": "", "top2": "", "top3": "",
                "match_pred": False, "match_top1": False,
                "match_top2": False, "match_top3": False,
                "avg_image": "", "roi_json": ""
            })
        except Exception as e:
            LOGGER.warning("Unexpected error on %s: %s", p, e)
            records.append({
                "true_label": p.parent.name,
                "file_name": p.name,
                "pred_label": "ERROR",
                "score": "0.0000",
                "reason": str(e),
                "top1": "", "top2": "", "top3": "",
                "match_pred": False, "match_top1": False,
                "match_top2": False, "match_top3": False,
                "avg_image": "", "roi_json": ""
            })

    # Write CSV with csv.writer (robust to commas/quotes/newlines)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "true_label","file_name","pred_label","score","reason",
        "top1","top2","top3","match_pred","match_top1","match_top2","match_top3","avg_image","roi_json"
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    LOGGER.info("Wrote %d predictions → %s", len(records), out_csv)

    # If pandas & sklearn exist, generate deeper analytics
    if (pd is not None) and (confusion_matrix is not None) and len(y_true) > 0 and len(y_pred) > 0:
        import pandas as pd  # type: ignore  # re-import for type checkers
        df = pd.DataFrame(records)

        # Labels set (sorted) includes whatever appears in y_true and y_pred
        labels = sorted(set(y_true) | set(y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)  # type: ignore
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_path = out_csv.with_name("ConfusionMatrix.csv")
        cm_df.to_csv(cm_path, encoding="utf-8")
        LOGGER.info("Saved confusion matrix → %s", cm_path)

        # Per-class report (precision/recall/f1)
        rep = classification_report(y_true, y_pred, labels=labels, output_dict=True)  # type: ignore
        rep_df = pd.DataFrame(rep).transpose()

        # Per-class accuracy (custom)
        per_class = []
        for lab in labels:
            idxs = [i for i, t in enumerate(y_true) if t == lab]
            tot = len(idxs)
            cor = sum(1 for i in idxs if y_pred[i] == lab)
            acc = (cor / tot) if tot else 0.0
            per_class.append({"label": lab, "n": tot, "acc": acc})
        acc_df = pd.DataFrame(per_class).sort_values("label")

        # Save Excel workbook with multiple sheets
        xl_path = out_csv.with_name("classification_report.xlsx")
        try:
            with pd.ExcelWriter(xl_path) as writer:
                df.to_excel(writer, sheet_name="Predictions", index=False)
                cm_df.to_excel(writer, sheet_name="ConfusionMatrix")
                rep_df.to_excel(writer, sheet_name="PerClassReport")
                acc_df.to_excel(writer, sheet_name="PerClassAccuracy", index=False)
            LOGGER.info("Saved detailed report → %s", xl_path)
        except Exception as e:
            LOGGER.warning("Could not write Excel report: %s", e)

        # Optional heatmaps/plots
        if sns_ok:
            try:
                import matplotlib.pyplot as plt  # type: ignore
                import seaborn as sns  # type: ignore

                plt.figure(figsize=(max(8, 0.5*len(labels)+4), max(6, 0.5*len(labels)+3)))
                sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title("Confusion Matrix")
                fig_path = out_csv.with_name("confusion_matrix.png")
                plt.tight_layout()
                plt.savefig(fig_path, dpi=150)
                plt.close()
                LOGGER.info("Confusion matrix heatmap saved → %s", fig_path)

                # Bar chart: per-class accuracy
                plt.figure(figsize=(max(8, 0.5*len(labels)+4), 5))
                sns.barplot(data=acc_df, x="label", y="acc")
                plt.ylim(0, 1.0)
                plt.xticks(rotation=45, ha="right")
                plt.title("Per-Class Accuracy")
                plt.ylabel("Accuracy")
                bar_path = out_csv.with_name("per_class_accuracy.png")
                plt.tight_layout()
                plt.savefig(bar_path, dpi=150)
                plt.close()
                LOGGER.info("Per-class accuracy plot saved → %s", bar_path)
            except Exception as e:
                LOGGER.warning("Could not draw heatmap/plots: %s", e)
    else:
        # Minimal text summary if analytics libs are not available
        total = len(records)
        correct = sum(1 for r in records if r["match_pred"])
        LOGGER.info("Summary (no pandas/sklearn): total=%d, correct=%d, acc=%.4f",
                    total, correct, (correct / max(1, total)))
    return True

# ---------------------------
# CLI
# ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Layout Classifier (single-file, open-set)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--version", action="version", version=f"layout {VERSION}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # build-index
    b = sub.add_parser("build-index", help="Build prototype index")
    b.add_argument("--data_dir", type=Path, required=True)
    b.add_argument("--avg_dir", type=Path, default=None)
    b.add_argument("--save_path", type=Path, required=True)
    b.add_argument("--crop_left_percent", type=float, default=DEF_CROP_LEFT)

    # learn-thresholds
    lt = sub.add_parser("learn-thresholds", help="Learn per-class thresholds from a validation set")
    lt.add_argument("--data_dir", type=Path, required=True)
    lt.add_argument("--index_path", type=Path, required=True)
    lt.add_argument("--crop_left_percent", type=float, default=DEF_CROP_LEFT)
    lt.add_argument("--alpha", type=float, default=DEF_ALPHA)
    lt.add_argument("--beta", type=float, default=DEF_BETA)
    lt.add_argument("--gamma", type=float, default=DEF_GAMMA)
    lt.add_argument("--percentile", type=float, default=0.05)
    lt.add_argument("--margin_percentile", type=float, default=0.10)
    lt.add_argument("--out_json", type=Path, required=True)

    # validate-open-set
    vo = sub.add_parser("validate-open-set", help="Validate accuracy on knowns and TPR on unknowns")
    vo.add_argument("--data_dir", type=Path, required=True)
    vo.add_argument("--unknown_dir", type=Path, default=None)
    vo.add_argument("--index_path", type=Path, required=True)
    vo.add_argument("--crop_left_percent", type=float, default=DEF_CROP_LEFT)
    vo.add_argument("--alpha", type=float, default=DEF_ALPHA)
    vo.add_argument("--beta", type=float, default=DEF_BETA)
    vo.add_argument("--gamma", type=float, default=DEF_GAMMA)
    vo.add_argument("--tau", type=float, default=DEF_TAU)
    vo.add_argument("--margin", type=float, default=DEF_MARGIN)
    vo.add_argument("--tau_high", type=float, default=DEF_TAU_HIGH)
    vo.add_argument("--fast_accept", type=float, default=DEF_FAST_ACCEPT)
    vo.add_argument("--top3_gap", type=float, default=DEF_TOP3_GAP)
    vo.add_argument("--thresholds_json", type=Path, default=None)

    # predict-one
    po = sub.add_parser("predict-one", help="Predict a single image")
    po.add_argument("--image", type=Path, required=True)
    po.add_argument("--index_path", type=Path, required=True)
    po.add_argument("--avg_dir", type=Path, default=None)
    po.add_argument("--roi_dir", type=Path, default=None)
    po.add_argument("--crop_left_percent", type=float, default=DEF_CROP_LEFT)
    po.add_argument("--alpha", type=float, default=DEF_ALPHA)
    po.add_argument("--beta", type=float, default=DEF_BETA)
    po.add_argument("--gamma", type=float, default=DEF_GAMMA)
    po.add_argument("--tau", type=float, default=DEF_TAU)
    po.add_argument("--margin", type=float, default=DEF_MARGIN)
    po.add_argument("--tau_high", type=float, default=DEF_TAU_HIGH)
    po.add_argument("--fast_accept", type=float, default=DEF_FAST_ACCEPT)
    po.add_argument("--top3_gap", type=float, default=DEF_TOP3_GAP)
    po.add_argument("--thresholds_json", type=Path, default=None)

    # predict-batch
    pb = sub.add_parser("predict-batch", help="Predict a folder of images and write CSV/analytics")
    pb.add_argument("--in_dir", type=Path, required=True)
    pb.add_argument("--index_path", type=Path, required=True)
    pb.add_argument("--avg_dir", type=Path, default=None)
    pb.add_argument("--roi_dir", type=Path, default=None)
    pb.add_argument("--out_csv", type=Path, required=True)
    pb.add_argument("--crop_left_percent", type=float, default=DEF_CROP_LEFT)
    pb.add_argument("--alpha", type=float, default=DEF_ALPHA)
    pb.add_argument("--beta", type=float, default=DEF_BETA)
    pb.add_argument("--gamma", type=float, default=DEF_GAMMA)
    pb.add_argument("--tau", type=float, default=DEF_TAU)
    pb.add_argument("--margin", type=float, default=DEF_MARGIN)
    pb.add_argument("--tau_high", type=float, default=DEF_TAU_HIGH)
    pb.add_argument("--fast_accept", type=float, default=DEF_FAST_ACCEPT)
    pb.add_argument("--top3_gap", type=float, default=DEF_TOP3_GAP)
    pb.add_argument("--thresholds_json", type=Path, default=None)

    args = ap.parse_args()

    # Configure logging (INFO default; allow override with env/CLI in future)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        if args.cmd == "build-index":
            ok = build_index(args.data_dir, args.avg_dir, args.save_path, args.crop_left_percent)
            return 0 if ok else 2
        elif args.cmd == "learn-thresholds":
            ok = learn_thresholds(args.data_dir, args.index_path, args.crop_left_percent,
                                  args.alpha, args.beta, args.gamma,
                                  args.percentile, args.margin_percentile, args.out_json)
            return 0 if ok else 2
        elif args.cmd == "validate-open-set":
            ok = validate_open_set(args.data_dir, args.unknown_dir, args.index_path,
                                   args.crop_left_percent, args.alpha, args.beta, args.gamma,
                                   args.tau, args.margin, args.tau_high, args.fast_accept, args.top3_gap,
                                   args.thresholds_json)
            return 0 if ok else 2
        elif args.cmd == "predict-one":
            ok = predict_one(args.image, args.index_path, args.avg_dir, args.roi_dir,
                             args.crop_left_percent, args.alpha, args.beta, args.gamma,
                             args.tau, args.margin, args.tau_high, args.fast_accept, args.top3_gap,
                             args.thresholds_json)
            return 0 if ok else 2
        elif args.cmd == "predict-batch":
            ok = predict_batch(args.in_dir, args.index_path, args.avg_dir, args.roi_dir, args.out_csv,
                               args.crop_left_percent, args.alpha, args.beta, args.gamma,
                               args.tau, args.margin, args.tau_high, args.fast_accept, args.top3_gap,
                               args.thresholds_json)
            return 0 if ok else 2
        else:
            return 2
    except FileNotFoundError as e:
        LOGGER.error("File or path not found: %s", e)
        return 2
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        return 130
    except Exception as e:
        LOGGER.error("Unhandled error: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
