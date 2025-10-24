# -*- coding: utf-8 -*-
"""
Image preprocessing and feature-based registration utilities for OCR workflows.

This module exposes:
  Global, mutable toggles for preprocessing and registration behavior (overridable via config).
  `preprocess_for_ocr`: lightweight enhancement (RGB convert, optional upscale, CLAHE, Otsu blend, optional sharpen).
  `sift_register_to_ref`: SIFT-based homography registration with ratio test and RANSAC.
  `orb_register_to_ref`: ORB-based homography registration with configurable feature/ratio; delegates to SIFT when requested.

Design goals:
  Do not change external behavior; only add documentation.
  Be resilient to missing OpenCV features (e.g., SIFT not compiled) and noisy inputs.
"""

from typing import Optional
import numpy as np
import cv2

# Registration & preprocess toggles (mutable by config)
REG_METHOD = "ORB"
REG_MAX_FEATURES = 2000
REG_GOOD_RATIO = 0.25
SIFT_NFEATURES = 4000
SIFT_RATIO = 0.75

PRE_CLAHE = False
PRE_CLAHE_CLIP = 2.0
PRE_CLAHE_GRID = 8
PRE_SHARPEN = False
PRE_SHARPEN_STRENGTH = 1.2

def preprocess_for_ocr(img_bgr: np.ndarray, upscale_if_small: bool = True) -> np.ndarray:
    """Prepare a BGR image for OCR by gentle enhancement and binarization blend.

    Pipeline:
      1) Convert BGR→RGB; optional upscale for small inputs (keeps max dim ≤ 800).
      2) Convert to GRAY and (optionally) apply CLAHE with configurable clip/grid.
      3) Normalize to [0..255], Otsu-threshold to get a binary mask.
      4) Blend binarized RGB (0.7) with original RGB (0.3) to preserve edges and tone.
      5) (Optional) Unsharp-like boost via `addWeighted` with `PRE_SHARPEN_STRENGTH`.

    Args:
        img_bgr: Input image in OpenCV BGR format.
        upscale_if_small: If True, upscales when width<600 or height<300 (cap at ~800 px).

    Returns:
        np.ndarray: Enhanced RGB-like image (still 3 channels, dtype preserved by OpenCV).
        If input is None/empty, returns it unchanged.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if upscale_if_small and (w < 600 or h < 300):
        scale = max(1.0, min(2.0, 800.0 / float(max(w, h))))
        if scale > 1.05:
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if PRE_CLAHE:
        try:
            clahe = cv2.createCLAHE(clipLimit=float(PRE_CLAHE_CLIP),
                                    tileGridSize=(int(PRE_CLAHE_GRID), int(PRE_CLAHE_GRID)))
            gray = clahe.apply(gray)
        except Exception:
            pass
    g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    thr = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    out = cv2.addWeighted(cv2.cvtColor(thr, cv2.COLOR_GRAY2RGB), 0.7, rgb, 0.3, 0)
    if PRE_SHARPEN:
        try:
            k = float(PRE_SHARPEN_STRENGTH)
            out = cv2.addWeighted(out, 1.0 + k, out, -k, 0.0)
        except Exception:
            pass
    return out

def sift_register_to_ref(img_bgr: np.ndarray,
                         ref_bgr: np.ndarray,
                         nfeatures: Optional[int] = None,
                         ratio: Optional[float] = None) -> np.ndarray:
    """Register `img_bgr` onto `ref_bgr` using SIFT keypoints and homography.

    Steps:
      - Build SIFT (if available) with `nfeatures` (default `SIFT_NFEATURES`).
      - Detect/compute on GRAY images; perform BF L2 KNN matching (k=2).
      - Lowe's ratio test with `ratio` (default `SIFT_RATIO`); require ≥12 good matches.
      - Estimate homography with RANSAC (reprojThresh=3.0) and warp perspective to ref size.

    Args:
        img_bgr: Source image (BGR) to be registered.
        ref_bgr: Reference image (BGR) defining target geometry.
        nfeatures: Optional override for number of SIFT features.
        ratio: Optional override for Lowe's ratio threshold.

    Returns:
        np.ndarray: Warped image aligned to `ref_bgr` on success; otherwise original `img_bgr`.
        If SIFT is unavailable, returns original `img_bgr`.
    """
    try:
        nfeatures = int(nfeatures if nfeatures is not None else SIFT_NFEATURES)
        ratio = float(ratio if ratio is not None else SIFT_RATIO)
        if not hasattr(cv2, "SIFT_create"):
            return img_bgr
        sift = cv2.SIFT_create(nfeatures=nfeatures)
        g1 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        kp1, des1 = sift.detectAndCompute(g1, None)
        kp2, des2 = sift.detectAndCompute(g2, None)
        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
            return img_bgr
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in knn if n is not None and m.distance < ratio * n.distance]
        if len(good) < 12:
            return img_bgr
        import numpy as np
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        hmat, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        if hmat is None:
            return img_bgr
        hR, wR = ref_bgr.shape[:2]
        return cv2.warpPerspective(img_bgr, hmat, (wR, hR), flags=cv2.INTER_LINEAR)
    except Exception:
        return img_bgr

def orb_register_to_ref(img_bgr: np.ndarray,
                        ref_bgr: np.ndarray,
                        max_features: int = REG_MAX_FEATURES,
                        good_match_ratio: float = REG_GOOD_RATIO) -> np.ndarray:
    """Register `img_bgr` onto `ref_bgr` using ORB keypoints and homography.

    Behavior:
      - If `REG_METHOD` is 'SIFT', delegates to `sift_register_to_ref` using global SIFT params.
      - Otherwise:
        * Detect ORB with `max_features`; compute on GRAY images.
        * BFMatcher (Hamming) with KNN (k=2); apply ratio test using `good_match_ratio`.
        * Require ≥12 good matches; estimate homography (RANSAC, thresh=3.0).
        * Warp perspective to reference frame size.

    Args:
        img_bgr: Source image (BGR) to align.
        ref_bgr: Reference image (BGR) providing target geometry.
        max_features: Maximum ORB features to detect.
        good_match_ratio: Lowe-style ratio threshold for good matches.

    Returns:
        np.ndarray: Warped/registered image on success, else original `img_bgr`.
    """
    from .imaging import REG_METHOD, SIFT_NFEATURES, SIFT_RATIO
    if str(REG_METHOD).upper() == "SIFT":
        return sift_register_to_ref(img_bgr, ref_bgr, nfeatures=SIFT_NFEATURES, ratio=SIFT_RATIO)
    try:
        g1 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=int(max_features))
        kp1, des1 = orb.detectAndCompute(g1, None)
        kp2, des2 = orb.detectAndCompute(g2, None)
        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
            return img_bgr
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if n is not None and m.distance < float(good_match_ratio) * n.distance]
        if len(good) < 12:
            return img_bgr
        import numpy as np
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        hmat, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        if hmat is None:
            return img_bgr
        hR, wR = ref_bgr.shape[:2]
        return cv2.warpPerspective(img_bgr, hmat, (wR, hR), flags=cv2.INTER_LINEAR)
    except Exception:
        return img_bgr
