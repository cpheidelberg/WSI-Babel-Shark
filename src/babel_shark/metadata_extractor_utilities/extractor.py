# -*- coding: utf-8 -*-
"""ROI-based metadata extractor: OCR, parsing, and layout-aware ROI handling.

This module defines `RoiMetadataExtractor`, which:
  - Loads fixed or group-specific ROI layouts (via JSON or a layout model).
  - Crops ROI regions, applies light pre-processing, and runs OCR.
  - Parses LabID/Year/CaseNumber/Pot/BlockID/Section/Stain from OCR text.
  - Optionally registers images to a reference and rescales to reference size.
  - Reconstructs case numbers across groups and builds a DataMatrix token.
"""

import os, re, time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from .utils.io_utils import ensure_dir
from .imaging.imaging import preprocess_for_ocr, orb_register_to_ref
from .ocr.engine import _OcrEngineInline
from .model.rois import ROIBox, load_rois_from_json
from .parsing.text_utils import sanitize_upper, to_year4_str, is_year_in_valid_range, to_case6_str, to_int_str_no_decimal
from .parsing.case_pot_stain import parse_casenumber, parse_pot_block, parse_pot_block_strict, parse_stain, reconstruct_casenumber_from_all_groups

class RoiMetadataExtractor:
    """Coordinate the ROI-based metadata extraction pipeline.

    Responsibilities:
        - Initialize configuration (ROI sources, OCR, registration, debug).
        - Optionally load a layout model to predict group layout.
        - Provide a public `run_on_image` API to extract metadata and ROI words.

    Notes:
        This class maintains compatibility with the existing pipeline and
        preserves behavior (no logic changes).
    """
    def __init__(self, config: Dict) -> None:
        """Create a new extractor with the provided configuration.

        Args:
            config: Dictionary of settings including ROI sources, OCR options,
                registration parameters, debug paths, and layout model config.

        Raises:
            ValueError: If neither a fixed ROI file nor a roiset root is provided.
            ImportError: If `layout_api` cannot be imported when dynamic mode is needed.
        """
        import sys
        from pathlib import Path
        self.cfg = config or {}

        # ROI sources
        self.roi_fixed_file = str(self.cfg.get("ROI_set_file") or "").strip()
        rsel = self.cfg.get("roiset_selector", {}) or {}
        self.model_cfg = rsel.get("layout_model", {}) or {}
        self.roiset_root = (rsel.get("roiset_root") or "").strip()
        self.strict_dynamic = bool(rsel.get("strict_dynamic", True))
        if not self.roi_fixed_file and not self.roiset_root:
            raise ValueError("Either ROI_set_file or roiset_selector.roiset_root must be set.")

        # Stain resources
        import json
        self.stain_dict_path = str(self.cfg.get("stain_list_path") or "").strip()
        self.stain_repl_path = str(self.cfg.get("stain_replace_map_path") or "").strip()
        self.stain_dict: Optional[Union[Dict[str, str], List[str]]] = None
        self.stain_replacements: Optional[Dict[str, str]] = None
        try:
            if self.stain_dict_path and os.path.exists(self.stain_dict_path):
                with open(self.stain_dict_path, "r", encoding="utf-8") as f:
                    self.stain_dict = json.load(f)
        except Exception:
            self.stain_dict = None
        try:
            if self.stain_repl_path and os.path.exists(self.stain_repl_path):
                with open(self.stain_repl_path, "r", encoding="utf-8") as f:
                    self.stain_replacements = json.load(f)
        except Exception:
            self.stain_replacements = None

        # Debug / rotation
        self.debug_parts_root = str(self.cfg.get("debug_parts_root") or "").strip()
        self.debug_save_mode = str(self.cfg.get("debug_save_mode", "hierarchical")).lower()
        if self.debug_save_mode not in {"hierarchical", "flat", "none"}:
            self.debug_save_mode = "hierarchical"
        self.rotate = int(self.cfg.get("wsi_macro_img_rotation", 0))

        # Layout model
        self.model = None
        # Make repo root importable (for layout_api)
        ROOT = Path(__file__).resolve().parents[1]
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        try:
            from layout_api.factory import load_model as load_layout_model  # type: ignore
            from PIL import Image as _PILImage  # type: ignore
            self._PILImage = _PILImage
            if not self.roi_fixed_file:
                self.model = load_layout_model(self.model_cfg)
        except Exception as e:
            raise ImportError(f"layout_api is required. Original error: {e!r}")

        params = (self.model_cfg.get("params") or {})
        self.low_conf_gate = float(params.get("low_conf_gate", 0.90))
        self.candidate_min = float(params.get("candidate_min", 0.91))

        # Classifier strip
        cstrip = self.cfg.get("classifier_strip", {}) or {}
        self.cls_strip_enabled = bool(cstrip.get("enabled", False))
        self.cls_strip_side = str(cstrip.get("side", "left")).lower()
        self.cls_strip_fraction = float(cstrip.get("fraction", 0.32))

        # Group ref image
        self.group_ref_image_pattern = self.cfg.get("group_ref_image_pattern", "{group}.png")
        self.use_group_ref_image_size = bool(self.cfg.get("use_group_ref_image_size", True))
        self.use_group_ref_as_reg = bool(self.cfg.get("use_group_ref_as_registration", False))
        self.resize_to_ref = bool(self.cfg.get("resize_to_ref", False))

        # OCR
        self.langs = self.cfg.get("ocr_langs", ["en"])
        self.min_conf = float(self.cfg.get("ocr_min_conf", 0.25))
        self.read_kwargs = {
            "text_threshold": float(self.cfg.get("ocr_text_threshold", 0.7)),
            "ycenter_ths": float(self.cfg.get("ocr_ycenter_ths", 0.5)),
            "slope_ths": float(self.cfg.get("ocr_slope_ths", 0.1)),
            "paragraph": bool(self.cfg.get("ocr_paragraph", False)),
            "contrast_ths": 0.1,
            "adjust_contrast": 0.5,
            "width_ths": 0.7,
            "decoder": "greedy",
        }
        self.save_ocr_inputs = bool(self.cfg.get("save_ocr_inputs", True))

        # Registration/cache
        self.ref_bgr: Optional[np.ndarray] = None
        self.do_register = False
        self.roi_ref_size: Optional[List[float]] = None
        reg_cfg = (self.cfg.get("registration") or {})
        self.reg_max_features = int(reg_cfg.get("max_features", 2000))
        self.reg_good_ratio = float(reg_cfg.get("good_match_ratio", 0.25))

        if self.roi_fixed_file:
            cand = str(Path(self.roi_fixed_file).with_suffix(".png"))
            if os.path.exists(cand):
                self._set_group_reference_from_png_path(cand)

        self.last_group: Optional[str] = None
        self.last_group_prob: Optional[float] = None
        self.last_group_top3: List[Tuple[str, float]] = []

        self.allowed_labid_letters = set(
            str(x).upper() for x in (self.cfg.get("allowed_labid_letters") or ["E", "R"])
        ) or {"E", "R"}

        # Apply YAML toggles to imaging module-level vars (no logic change)
        from .imaging import imaging as _im
        _im.REG_METHOD = str(reg_cfg.get("method", _im.REG_METHOD)).upper()
        _im.REG_MAX_FEATURES = int(reg_cfg.get("max_features", _im.REG_MAX_FEATURES))
        _im.REG_GOOD_RATIO = float(reg_cfg.get("good_match_ratio", _im.REG_GOOD_RATIO))
        _im.SIFT_NFEATURES = int(reg_cfg.get("sift_nfeatures", _im.SIFT_NFEATURES))
        _im.SIFT_RATIO = float(reg_cfg.get("sift_ratio", _im.SIFT_RATIO))

        pp = (self.cfg.get("ocr_preprocess") or {})
        _im.PRE_CLAHE = bool(pp.get("clahe", _im.PRE_CLAHE))
        _im.PRE_CLAHE_CLIP = float(pp.get("clahe_clip", _im.PRE_CLAHE_CLIP))
        _im.PRE_CLAHE_GRID = int(pp.get("clahe_grid", _im.PRE_CLAHE_GRID))
        _im.PRE_SHARPEN = bool(pp.get("sharpen", _im.PRE_SHARPEN))
        _im.PRE_SHARPEN_STRENGTH = float(pp.get("sharpen_strength", _im.PRE_SHARPEN_STRENGTH))

        self._ocr_engine = _OcrEngineInline(self.langs, self.min_conf, self.read_kwargs)

    # --- model helpers --------------------------------------------------------
    def _predict_group_layout(self, img_bgr: np.ndarray, img_name: Optional[str] = None
                              ) -> Tuple[Optional[str], Optional[float], List[Tuple[str, float]]]:
        """Predict a layout group label (and score, top3) for an image.

        Args:
            img_bgr: Input BGR image.
            img_name: Optional filename (unused; kept for compatibility).

        Returns:
            (label, score, top3) where label/top3 may be None/empty if model missing.
        """
        if self.model is None:
            return None, None, []
        clf_img = self._crop_classifier_strip(img_bgr)
        rgb = cv2.cvtColor(clf_img, cv2.COLOR_BGR2RGB)
        pil = self._PILImage.fromarray(rgb, mode="RGB")
        out = self.model.predict_image(pil)
        if hasattr(out, "label"):
            label = getattr(out, "label")
            score = getattr(out, "score", None)
            top3 = getattr(out, "top3", None)
        elif isinstance(out, dict):
            label = out.get("label")
            score = out.get("score")
            top3 = out.get("top3")
        else:
            label, score, top3 = None, None, None
        try:
            top3_list: List[Tuple[str, float]] = [(t[0], float(t[1])) for t in (top3 or [])]
        except Exception:
            top3_list = []
        return label, (float(score) if score is not None else None), top3_list

    def _load_rois_for_group(self, group: str) -> List["ROIBox"]:
        """Load ROI definitions for a predicted group.

        Args:
            group: Group label used to pick the corresponding JSON file.

        Returns:
            List of ROIBox objects.

        Raises:
            FileNotFoundError: If the group's ROI JSON file is not found and strict mode applies.
        """
        if not group:
            return []
        roi_json = os.path.join(self.roiset_root, f"{group}.json")
        if not os.path.exists(roi_json):
            if self.strict_dynamic:
                raise FileNotFoundError(f"ROI JSON not found for predicted group '{group}': {roi_json}")
            raise FileNotFoundError(f"ROI JSON not found: {roi_json}")
        self._set_group_reference_size(group)
        return load_rois_from_json(roi_json)

    def _crop_classifier_strip(self, img_bgr: np.ndarray) -> np.ndarray:
        """Crop the classifier strip area from the image when enabled.

        Args:
            img_bgr: Source BGR image.

        Returns:
            Cropped view for classification, or original image if strip is disabled.
        """
        if not self.cls_strip_enabled:
            return img_bgr
        h, w = img_bgr.shape[:2]
        import numpy as np
        frac = float(np.clip(self.cls_strip_fraction, 0.05, 0.9))
        if self.cls_strip_side == "left":
            x0, x1 = 0, int(round(w * frac))
        else:
            x0, x1 = int(round(w * (1.0 - frac))), w
        return img_bgr[:, x0:x1]

    def _set_group_reference_from_png_path(self, ref_png: str) -> None:
        """Load a group reference PNG to set target size and/or registration image.

        Args:
            ref_png: Path to the group reference image.

        Notes:
            Honors `use_group_ref_image_size` and `use_group_ref_as_reg` flags.
        """
        from .imaging import imaging as _im
        if not (_im.use_group_ref_image_size if hasattr(_im, 'use_group_ref_image_size') else True) and not (_im.use_group_ref_as_reg if hasattr(_im, 'use_group_ref_as_reg') else False):
            # keep logic minimal: original checked flags in self; here we keep behavior by using self flags
            pass
        if not (self.use_group_ref_image_size or self.use_group_ref_as_reg):
            return
        if not ref_png or not os.path.exists(ref_png):
            return
        img = cv2.imread(ref_png)
        if img is None or img.size == 0:
            return
        h, w = img.shape[:2]
        if self.use_group_ref_image_size:
            self.roi_ref_size = [w, h]
        if self.use_group_ref_as_reg:
            self.ref_bgr = img
            self.do_register = True

    def _set_group_reference_size(self, group: str) -> None:
        """Set reference size/registration image based on the group label.

        Args:
            group: Predicted group name used to resolve the reference PNG.
        """
        if not (self.use_group_ref_image_size or self.use_group_ref_as_reg):
            return
        if self.roiset_root and group:
            ref_png = os.path.join(self.roiset_root, self.group_ref_image_pattern.format(group=group))
            self._set_group_reference_from_png_path(ref_png)

    def _apply_rule_clean(self, raw_text: str, key: str) -> str:
        """Apply replacement patterns and allowed-char filters for a given key.

        Args:
            raw_text: Original OCR text for the ROI.
            key: Logical field key (e.g., 'Stain', 'Section') to select rule set.

        Returns:
            Cleaned string after configured regex replacements and whitelisting.
        """
        if raw_text is None:
            return ""
        t = str(raw_text)
        rules = self.cfg.get("extraction_rules", {})
        cfg = rules.get(key) or rules.get(key.capitalize())
        if not cfg:
            return t
        import re
        repls = cfg.get("replacement_patterns", []) or []
        for pat, sub in repls:
            t = re.sub(pat, sub)
        allow = cfg.get("allowed_chars")
        if allow:
            t = "".join([ch for ch in t if ch in allow])
        return t

    def _crop_roi(self, base_bgr: np.ndarray, roi: "ROIBox", pxl_offset: int) -> Optional[np.ndarray]:
        """Crop an ROI region from a base image, with optional scale and padding.

        Args:
            base_bgr: Full BGR image.
            roi: ROIBox with coordinates in reference or image space.
            pxl_offset: Padding (pixels) to expand the ROI edges.

        Returns:
            Cropped BGR sub-image, or None for invalid bounds.
        """
        h, w = base_bgr.shape[:2]
        l, t, r, b = roi.left, roi.top, roi.right, roi.bottom
        if self.roi_ref_size:
            wref, href = self.roi_ref_size
            sx, sy = w / float(wref), h / float(href)
            l, t, r, b = l * sx, t * sy, r * sx, b * sy
        left = max(0, int(round(l - pxl_offset)))
        right = min(w, int(round(r + pxl_offset)))
        top = max(0, int(round(t - pxl_offset)))
        bottom = min(h, int(round(b + pxl_offset)))
        if right <= left or bottom <= top:
            return None
        return base_bgr[top:bottom, left:right]

    def _ocr_roi(self, roi_img_bgr: np.ndarray, roi_name: str, debug_path: Optional[str] = None
                 ) -> Tuple[List[str], str]:
        """Run OCR over a single ROI crop with light variants as fallback.

        Args:
            roi_img_bgr: ROI BGR image.
            roi_name: Logical ROI name (used for debug file naming).
            debug_path: Optional path for saving OCR inputs.

        Returns:
            (tokens, text) where tokens are unique strings and text is a joined form.
        """
        if roi_img_bgr is None or roi_img_bgr.size == 0:
            return [], ""
        rgb = preprocess_for_ocr(roi_img_bgr, upscale_if_small=True)
        if self.save_ocr_inputs and debug_path:
            try:
                Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    debug_path.replace(".png", ".ocr_input.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                )
            except Exception:
                pass
        tokens, text = self._ocr_engine.run(rgb, allowlist=None)
        if tokens:
            return tokens, text

        base_rgb = cv2.cvtColor(roi_img_bgr, cv2.COLOR_BGR2RGB)

        def _clahe_strong(gray: np.ndarray) -> np.ndarray:
            """Apply stronger CLAHE for challenging text regions (best-effort)."""
            try:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                return clahe.apply(gray)
            except Exception:
                return gray

        variants: List[Tuple[str, np.ndarray]] = [("base", rgb)]
        g = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY)
        g1 = _clahe_strong(g)
        thr = cv2.threshold(g1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        v1 = cv2.addWeighted(cv2.cvtColor(thr, cv2.COLOR_GRAY2RGB), 0.6, base_rgb, 0.4, 0)
        variants.append(("clahe3_otsu", v1))
        variants.append(("invert", 255 - rgb))
        for _, vv in variants:
            toks2, txt2 = self._ocr_engine.run(vv, allowlist=None)
            if toks2:
                return toks2, txt2
        return [], ""

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def run_on_image(self,
                     img_bgr: np.ndarray,
                     img_name: str = "",
                     pxl_offset: int = 8) -> Tuple[Dict[str, str], bool, Dict[str, str]]:
        """Run the full extraction pipeline on a single image.

        The method supports both fixed-ROI and dynamic group modes:
          - Fixed: apply optional registration/resize, load ROIs from a JSON, OCR + parse.
          - Dynamic: predict layout group, evaluate candidates, OCR + parse per group,
                     attempt cross-group case-number reconstruction.

        Args:
            img_bgr: Input image in BGR format.
            img_name: Optional filename for logging/debug artifacts.
            pxl_offset: Padding used when cropping ROIs.

        Returns:
            (parsed, success, roi_words) where:
              - parsed: Dict of extracted fields (LabID, Year, CaseNumber, Pot, BlockID, Section, Stain, DataMatrix).
              - success: True if a valid DataMatrix was built.
              - roi_words: Flattened per-ROI token strings (possibly across groups).
        """
        def _save_crop(crop_bgr: np.ndarray, base_name: str, roi_name: str, group_label: str) -> Optional[str]:
            """Save cropped ROI image to the debug folder, following configured layout."""
            stem = f"{Path(base_name).stem}.{roi_name}.{group_label}.png"
            if not self.debug_parts_root or self.debug_save_mode == "none":
                return None
            if self.debug_save_mode == "hierarchical":
                droot = ensure_dir(os.path.join(self.debug_parts_root, str(roi_name)))
                outp = os.path.join(droot, stem)
            else:  # flat
                droot = ensure_dir(self.debug_parts_root)
                outp = os.path.join(droot, stem)
            try:
                cv2.imwrite(outp, crop_bgr)
                return outp
            except Exception:
                return None

        def run_extraction(base_img: np.ndarray, rois: List["ROIBox"], group_label: Optional[str] = None
                           ) -> Tuple[Dict[str, str], bool, Dict[str, str]]:
            """Extract metadata from a prepared image using the provided ROIs.

            Args:
                base_img: Preprocessed/aligned image to read from.
                rois: List of ROIBox defining regions to OCR.
                group_label: Optional label used for debug naming.

            Returns:
                (parsed_dict, success, roi_words_dict) for this pass.
            """
            roi_words_text: Dict[str, str] = {}

            def ocr_pass(_base_img: np.ndarray) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
                """Perform a single OCR+parse pass over all ROIs; return raw tokens and parsed fields."""
                raw_vals: Dict[str, str] = {}
                ocr_toks: Dict[str, List[str]] = {}
                for roi in rois:
                    crop = self._crop_roi(_base_img, roi, pxl_offset)
                    if crop is None:
                        continue
                    grp = (group_label or self.last_group or "Unknown")
                    saved_path = _save_crop(crop, img_name, roi.name, grp)

                    tokens, text = self._ocr_roi(
                        crop,
                        roi.name,
                        debug_path=(saved_path if (saved_path and self.save_ocr_inputs) else None)
                    )
                    ocr_toks[roi.name] = tokens
                    roi_words_text[f"ROI_{roi.name}_Words"] = " ".join(tokens) if tokens else ""

                    key = (roi.name or "ROI").strip().lower()
                    if key in ("sec", "section"):
                        raw_vals["Section"] = text
                    elif key in ("casenumber", "case", "caseid", "case-number", "case_id"):
                        raw_vals["Casenumber"] = text
                    elif key in ("pot-blockid", "pot_blockid", "potblockid", "pot", "block"):
                        raw_vals["Pot-BlockID"] = text
                    elif "stain" in key:
                        raw_vals["Stain"] = text
                    elif key in ("datamatrix", "dm", "matrix", "matrix2d"):
                        dm_raw = "".join(re.findall(r"[A-Za-z0-9\-_/]+", text))
                        raw_vals["DataMatrixRaw"] = dm_raw

                parsed: Dict[str, str] = {"Section": "500", "Pot": "A", "BlockID": "1"}

                if "Section" in raw_vals:
                    t = sanitize_upper(raw_vals.get("Section"))
                    m = re.search(r"\b([0-9]{1,2})\b", t or "")
                    parsed["Section"] = str(int(m.group(1))) if m else "500"

                if "Casenumber" in raw_vals:
                    parsed.update(parse_casenumber(raw_vals["Casenumber"], allowed_letters=self.allowed_labid_letters))

                if "Pot-BlockID" in raw_vals:
                    p, b = parse_pot_block(raw_vals["Pot-BlockID"])
                    parsed["Pot"], parsed["BlockID"] = p, b

                parsed["Stain"] = parse_stain(
                    raw_vals.get("Stain"),
                    self.stain_dict,
                    self.stain_replacements
                ) if "Stain" in raw_vals else "H&E"

                y4 = to_year4_str(parsed.get("Year")) if parsed.get("Year") else None
                has_five = bool(parsed.get("LabID") and parsed.get("CaseNumber")
                                and parsed.get("Pot") and parsed.get("BlockID") and y4)
                year_ok = is_year_in_valid_range(y4)
                from .parsing.text_utils import build_datamatrix
                dm_built = build_datamatrix(
                    parsed.get("LabID"), y4, parsed.get("CaseNumber"),
                    parsed.get("Pot"), parsed.get("BlockID"), parsed.get("Section"),
                ) if (has_five and year_ok) else None
                if dm_built:
                    parsed["DataMatrix"] = dm_built

                for key_fix in ("BlockID", "Section"):
                    parsed[key_fix] = to_int_str_no_decimal(parsed.get(key_fix)) or "1"
                if parsed.get("CaseNumber"):
                    parsed["CaseNumber"] = to_case6_str(parsed["CaseNumber"]) or parsed["CaseNumber"]
                if parsed.get("Year"):
                    parsed["Year"] = to_year4_str(parsed["Year"]) or parsed["Year"]
                if parsed.get("Pot"):
                    parsed["Pot"] = str(parsed["Pot"]).strip().upper()

                success_local = bool(parsed.get("DataMatrix"))
                return parsed, ocr_toks

            parsed_out, ocr_tokens = ocr_pass(base_img)
            success_out = bool(parsed_out.get("DataMatrix"))
            return parsed_out, success_out, {f"ROI_{k}_Words": " ".join(v) for k, v in ocr_tokens.items()}

        # Fixed ROI mode
        if self.roi_fixed_file:
            base_img = img_bgr.copy()
            """Apply optional registration/resize for fixed ROI mode and run extraction."""
            if self.do_register and self.ref_bgr is not None:
                base_img = orb_register_to_ref(
                    base_img, self.ref_bgr,
                    max_features=self.reg_max_features,
                    good_match_ratio=self.reg_good_ratio
                )
            if self.roi_ref_size and self.resize_to_ref:
                wref, href = self.roi_ref_size
                base_img = cv2.resize(base_img, (int(wref), int(href)), interpolation=cv2.INTER_AREA)

            rois = load_rois_from_json(self.roi_fixed_file)
            parsed, success, roi_words = run_extraction(base_img, rois, group_label=None)
            return parsed, success, roi_words

        # Dynamic group mode
        label, pval, top3 = self._predict_group_layout(img_bgr, img_name)
        self.last_group = label
        self.last_group_prob = pval
        self.last_group_top3 = top3 or []

        candidates: List[Tuple[str, str, float]] = []
        used: set[str] = set()
        if label and label != "Unknown":
            candidates.append((label, "Predicted", float(pval or 0.0)))
            used.add(label)
        top_items: List[Tuple[str, float]] = []
        for it in (self.last_group_top3 or []):
            try:
                g, s = it[0], float(it[1])
            except Exception:
                continue
            top_items.append((g, s))
        for i, (g, s) in enumerate(top_items[:3], start=1):
            if not g or g == "Unknown" or g in used:
                continue
            if s >= self.candidate_min:
                candidates.append((g, f"Top{i}", s))
                used.add(g)

        base_img_dyn = img_bgr.copy()
        if self.do_register and self.ref_bgr is not None:
            base_img_dyn = orb_register_to_ref(
                base_img_dyn, self.ref_bgr,
                max_features=self.reg_max_features,
                good_match_ratio=self.reg_good_ratio
            )
        if self.roi_ref_size and self.resize_to_ref:
            wref, href = self.roi_ref_size
            base_img_dyn = cv2.resize(base_img_dyn, (int(wref), int(href)), interpolation=cv2.INTER_AREA)

        final_parsed: Dict[str, str] = {}
        final_roi_words: Dict[str, str] = {}
        all_groups_roi_words: Dict[str, str] = {}

        for cand_label, tag, score in candidates[:4]:
            """Iterate candidate groups; on success, keep parsed result and aggregate ROI words."""
            try:
                rois = self._load_rois_for_group(cand_label)
            except Exception:
                continue
            parsed_c, success_c, roi_words_c = run_extraction(base_img_dyn, rois, group_label=cand_label)
            for k, v in (roi_words_c or {}).items():
                all_groups_roi_words[f"{cand_label}::{k}"] = v
            if success_c:
                final_parsed = parsed_c
                final_roi_words = all_groups_roi_words
                break

        reconstructed = reconstruct_casenumber_from_all_groups(all_groups_roi_words)
        if reconstructed:
            parsed_corr = parse_casenumber(reconstructed, allowed_letters=self.allowed_labid_letters)
            if parsed_corr:
                old_csn = (final_parsed.get("CaseNumber") or "")
                new_csn = parsed_corr.get("CaseNumber") or ""
                def _nzint(s: str) -> int:
                    """Normalize a zero-padded integer string to int; -1 on error."""
                    try:
                        import re
                        return int(re.sub(r"^0+", "", s or "") or "0")
                    except Exception:
                        return -1
                if (not old_csn) or (len(new_csn) > len(old_csn)) or                    (len(new_csn) == len(old_csn) and _nzint(new_csn) > _nzint(old_csn)):
                    final_parsed.update(parsed_corr)
                    y4 = to_year4_str(final_parsed.get("Year"))
                    pot = final_parsed.get("Pot") or "A"
                    blk = final_parsed.get("BlockID") or "1"
                    sec = final_parsed.get("Section") or "500"
                    from .parsing.text_utils import build_datamatrix
                    dm = build_datamatrix(parsed_corr.get("LabID"), y4, parsed_corr.get("CaseNumber"), pot, blk, sec)
                    if dm:
                        final_parsed["DataMatrix"] = dm

        success_final = bool(final_parsed.get("DataMatrix"))
        return final_parsed, success_final, final_roi_words
