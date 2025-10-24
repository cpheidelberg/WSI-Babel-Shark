# -*- coding: utf-8 -*-
"""Lightweight inline OCR engine wrapper.

This module provides a minimal wrapper around EasyOCR to extract text tokens
from RGB images with configurable language list, confidence threshold, and
reader parameters. If EasyOCR is unavailable, the engine gracefully returns
empty outputs.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import easyocr  # type: ignore
    EASY_AVAILABLE = True
except Exception:
    easyocr = None  # type: ignore
    EASY_AVAILABLE = False

class _OcrEngineInline:
    """Inline OCR engine using EasyOCR with simple configuration.

    Attributes:
        langs: Languages passed to EasyOCR (defaults to ["en"] if empty).
        min_conf: Minimum confidence score to accept recognized tokens.
        read_kwargs: Extra keyword arguments propagated to EasyOCR `readtext`.
        _easy_reader: Lazy-initialized EasyOCR reader instance.
    """
    def __init__(self, langs: List[str], min_conf: float, read_kwargs: Dict[str, object]) -> None:
        """Initialize the OCR engine with languages, confidence, and kwargs.

        Args:
            langs: List of language codes for the OCR model.
            min_conf: Minimum confidence threshold for accepted tokens.
            read_kwargs: Additional parameters for the EasyOCR reader.
        """
        self.langs = list(langs) if langs else ["en"]
        self.min_conf = float(min_conf)
        self.read_kwargs = dict(read_kwargs or {})
        self._easy_reader = None

    def _get_easy_reader(self):
        """Create (on first call) and return the EasyOCR Reader instance."""
        if self._easy_reader is None:
            self._easy_reader = easyocr.Reader(self.langs, gpu=False)  # type: ignore[attr-defined]
        return self._easy_reader

    @staticmethod
    def _dedup(tokens: List[str]) -> List[str]:
        """Deduplicate tokens while preserving order and trimming whitespace.

        Args:
            tokens: List of raw tokens (possibly repeated/empty).

        Returns:
            A new list with unique, non-empty tokens in their first-seen order.
        """
        seen: set[str] = set()
        out: List[str] = []
        for t in tokens:
            k = (t or "").strip()
            if k and k not in seen:
                out.append(k)
                seen.add(k)
        return out

    def run(self, img_rgb: np.ndarray, allowlist: Optional[str] = None) -> Tuple[List[str], str]:
        """Run OCR on an RGB image and return deduplicated tokens and joined text.

        The method uses EasyOCR's `readtext` with parameters from `read_kwargs`.
        Tokens whose confidence is below `min_conf` are filtered out.

        Args:
            img_rgb: Input image in RGB (H, W, 3) numpy array.
            allowlist: Optional whitelist of characters for recognition.

        Returns:
            A tuple `(tokens, text)` where:
              - `tokens` is a list of unique recognized strings (order-preserving).
              - `text` is a single string joining tokens with spaces.
            Returns `([], "")` if EasyOCR is unavailable or on error.
        """
        if (not EASY_AVAILABLE) or img_rgb is None or img_rgb.size == 0:
            return [], ""
        try:
            reader = self._get_easy_reader()
            result = reader.readtext(  # type: ignore[attr-defined]
                img_rgb,
                detail=1,
                allowlist=allowlist,
                paragraph=bool(self.read_kwargs.get("paragraph", False)),
                text_threshold=float(self.read_kwargs.get("text_threshold", 0.7)),
                ycenter_ths=float(self.read_kwargs.get("ycenter_ths", 0.5)),
                slope_ths=float(self.read_kwargs.get("slope_ths", 0.1)),
                contrast_ths=float(self.read_kwargs.get("contrast_ths", 0.1)),
                adjust_contrast=float(self.read_kwargs.get("adjust_contrast", 0.5)),
                width_ths=float(self.read_kwargs.get("width_ths", 0.7)),
                decoder=str(self.read_kwargs.get("decoder", "greedy")),
            )
            tokens: List[str] = []
            for item in result:
                try:
                    _, txt, conf = item
                except Exception:
                    txt = str(item[1]) if len(item) >= 2 else ""
                    conf = float(item[2]) if len(item) >= 3 else None
                if conf is None or float(conf) < self.min_conf:
                    continue
                t = (txt or "").strip()
                if t:
                    tokens.append(t)
        except Exception:
            tokens = []
        tokens = self._dedup(tokens)
        return tokens, (" ".join(tokens) if tokens else "")
