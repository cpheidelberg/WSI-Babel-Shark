# -*- coding: utf-8 -*-
"""PDF report generator with smart image lookup for ROI metadata cards.

This module renders a cards-only PDF from results and words DataFrames using
ReportLab (if available). It includes utilities to auto-detect nearby image
folders and robustly resolve image paths via multiple matching strategies
(direct, case-insensitive, stem-based, and extension substitution).
"""
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except Exception as e:
    REPORTLAB_AVAILABLE = False
    _IMPORT_ERR = e
    A4 = None
    canvas = None
    ImageReader = None

def _wrap_lines(s: str, width: int) -> List[str]:
    """Soft-wrap a string to a list of lines with a minimum width.

    Args:
        s: Input text to wrap.
        width: Target wrap width (minimum 10 enforced).

    Returns:
        List of wrapped lines preserving internal whitespace as much as possible.
    """
    import textwrap as _tw
    return _tw.wrap(s, width=max(10, width), replace_whitespace=False, drop_whitespace=False)

# ---------- smart image matching helpers ----------

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

def _stem(p: str) -> str:
    """Return filename stem (basename without extension) from a path string.

    Args:
        p: File path.

    Returns:
        Stem part of the basename.
    """
    b = os.path.basename(p)
    s, _ = os.path.splitext(b)
    return s

def _samefile_ci(a: str, b: str) -> bool:
    """Case-insensitive basename equality check.

    Args:
        a: First path.
        b: Second path.

    Returns:
        True if basenames are equal ignoring case; otherwise False.
    """
    return os.path.basename(a).casefold() == os.path.basename(b).casefold()

def _collect_files(base_dir: str, recursive: bool) -> List[str]:
    """Collect files under a directory, optionally recursively.

    Args:
        base_dir: Directory to scan.
        recursive: If True, walk subdirectories; else only list direct files.

    Returns:
        List of absolute/relative file paths found.
    """
    files = []
    if not os.path.isdir(base_dir):
        return files
    if not recursive:
        for f in os.listdir(base_dir):
            fp = os.path.join(base_dir, f)
            if os.path.isfile(fp):
                files.append(fp)
        return files
    for root, _, fs in os.walk(base_dir):
        for f in fs:
            files.append(os.path.join(root, f))
    return files

def _find_image_smart(base_dir: str, fname: str, recursive: bool) -> Optional[str]:
    """
    Tries multiple strategies:
    1) exact path, base_dir/fname, base_dir/basename
    2) case-insensitive match by full filename
    3) match by stem (exact, startswith, contains), trying common image extensions
    4) if fname is .svs/.ndpi, try same stem with image exts

    Args:
        base_dir: Root directory to search for images.
        fname: Original filename (possibly non-image).
        recursive: Whether to search recursively under base_dir.

    Returns:
        Path to the best-matched image if found; otherwise None.
    """
    if not fname or not base_dir:
        return None

    # 1) direct candidates
    direct = []
    if os.path.exists(fname):
        return fname
    direct.append(os.path.join(base_dir, fname))
    direct.append(os.path.join(base_dir, os.path.basename(fname)))

    for cand in direct:
        if os.path.exists(cand):
            logging.info("[PDF] image found (direct): %s", cand)
            return cand

    # 2) scan directory
    files = _collect_files(base_dir, recursive)
    base = os.path.basename(fname)
    base_cf = base.casefold()
    base_stem = _stem(base).casefold()

    # case-insensitive full-name match
    for fp in files:
        if os.path.basename(fp).casefold() == base_cf:
            logging.info("[PDF] image found (CI full-name): %s", fp)
            return fp

    # if fname has non-image ext (e.g., .svs/.ndpi), try image exts on same stem
    name_stem, name_ext = os.path.splitext(base)
    if name_ext.lower() in (".svs", ".ndpi", ".tif", ".tiff"):
        for ext in _IMG_EXTS:
            guess = name_stem + ext
            for fp in files:
                if os.path.basename(fp).casefold() == guess.casefold():
                    logging.info("[PDF] image found (stem + common ext): %s", fp)
                    return fp

    # 3) match by stem: exact stem
    for fp in files:
        if _stem(fp).casefold() == base_stem and fp.lower().endswith(_IMG_EXTS):
            logging.info("[PDF] image found (stem match): %s", fp)
            return fp

    # startswith stem
    for fp in files:
        if _stem(fp).casefold().startswith(base_stem) and fp.lower().endswith(_IMG_EXTS):
            logging.info("[PDF] image found (stem startswith): %s", fp)
            return fp

    # contains stem
    for fp in files:
        if base_stem in _stem(fp).casefold() and fp.lower().endswith(_IMG_EXTS):
            logging.info("[PDF] image found (stem contains): %s", fp)
            return fp

    logging.warning("[PDF] image NOT found for: %s (base_dir=%s, recursive=%s)", fname, base_dir, recursive)
    return None

# ---------- autodetect image dir near PDF ----------

def _autodetect_image_dir(pdf_path: str) -> (str, bool):
    """Guess a nearby directory that likely contains images for the PDF.

    The heuristic searches common sibling/child directories such as
    'failed_datamatrix', 'roi_fallback/failed_datamatrix', 'datamatrix_failed',
    or 'images'.

    Args:
        pdf_path: Target PDF path used to find neighboring folders.

    Returns:
        Tuple (image_dir, recursive_default) where image_dir may be empty if not found,
        and recursive_default is True when auto-detection succeeds.
    """
    pdf_dir = os.path.abspath(os.path.dirname(pdf_path) or ".")
    candidates = [
        os.path.join(pdf_dir, "failed_datamatrix"),
        os.path.join(os.path.dirname(pdf_dir), "failed_datamatrix"),
        os.path.join(pdf_dir, "roi_fallback", "failed_datamatrix"),
        os.path.join(pdf_dir, "datamatrix_failed"),
        os.path.join(pdf_dir, "images"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            logging.info("[PDF] Auto-detected image folder: %s", c)
            return c, True  # enable recursive by default when auto-detected
    logging.info("[PDF] No image folder auto-detected near PDF.")
    return "", False

# ---------- main writer (cards-only) ----------

def _write_pdf_report(pdf_path: str,
                      df_results: pd.DataFrame,
                      df_words: pd.DataFrame,
                      title: str = "ROI Metadata Cards") -> None:
    """Render a cards-only PDF containing ROI results and per-ROI word snippets.

    Behavior:
      - Uses ReportLab if available; logs an error and returns otherwise.
      - Resolves an image directory via env vars or auto-detection near `pdf_path`.
      - For each row in `df_results`, draws a left image (best-effort lookup) and
        a right column listing all fields (except 'FileName'), wrapped as needed.
      - Optionally appends "ROI Words" from `df_words` when available.
      - Lays out multiple cards per page with fixed margins/columns.

    Args:
        pdf_path: Destination path for the generated PDF.
        df_results: DataFrame of metadata rows; must include 'FileName' column for image lookup.
        df_words: DataFrame where each row corresponds to a file's OCR words; may be empty/None.
        title: Title text rendered on each page.

    Returns:
        None. Writes the PDF to `pdf_path` and logs progress/outcomes.
    """
    if not REPORTLAB_AVAILABLE:
        logging.error("reportlab not available; cannot write PDF. Import error: %r",
                      _IMPORT_ERR if '_IMPORT_ERR' in globals() else None)
        return

    # Resolve image dir: ENV -> autodetect near PDF
    img_dir = os.environ.get("INPUT_IMAGE_DIR") or ""
    recursive_env = os.environ.get("INPUT_IMAGE_RECURSIVE", "")
    recursive = bool(int(recursive_env)) if recursive_env != "" else False

    if not img_dir:
        img_dir, auto_rec = _autodetect_image_dir(pdf_path)
        if img_dir and recursive_env == "":
            recursive = auto_rec

    logging.info("[PDF] Using image_dir=%s recursive=%s", img_dir or "<none>", recursive)

    # Prepare page
    Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
    page_w, page_h = A4
    margin = 36
    col_gap = 16
    left_col_w  = (page_w - 2*margin - col_gap) * 0.50
    right_col_w = (page_w - 2*margin - col_gap) * 0.50
    img_max_h = 280

    c = canvas.Canvas(pdf_path, pagesize=A4)

    def new_page(is_first: bool):
        """Start a new page and draw the page title."""
        if not is_first:
            c.showPage()
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, page_h - margin - 18, title)

    # Index ROI words by FileName
    words_by_file: Dict[str, Dict] = {}
    if df_words is not None and not df_words.empty and "FileName" in df_words.columns:
        for _, r in df_words.iterrows():
            try:
                words_by_file[str(r["FileName"])] = r.to_dict()
            except Exception:
                pass

    first = True
    new_page(first)
    first = False
    y_cursor = page_h - margin - 26 - 18  # below title

    for _, row in df_results.iterrows():
        # New page if not enough space
        min_block_height = 260
        if y_cursor - min_block_height < margin:
            new_page(first)
            y_cursor = page_h - margin - 26 - 18

        img_x = margin
        img_y_top = y_cursor
        right_x = margin + left_col_w + col_gap
        right_y = img_y_top
        line_h = 12

        # Left: image (smart search)
        c.setFont("Helvetica", 10)
        fname = str(row.get("FileName") or "").strip()
        img_h_drawn = 0

        try:
            img_path = _find_image_smart(img_dir, fname, recursive=recursive) if img_dir else None
            if img_path:
                ir = ImageReader(img_path)
                iw, ih = ir.getSize()
                scale = min(max(0.01, left_col_w / max(1, iw)), max(0.01, img_max_h / max(1, ih)))
                dw, dh = iw * scale, ih * scale
                c.drawImage(ir, img_x, img_y_top - dh, width=dw, height=dh,
                            preserveAspectRatio=True, mask='auto')
                img_h_drawn = dh
                logging.info("[PDF] drew image: %s", img_path)
            else:
                c.setFont("Helvetica-Oblique", 9)
                c.drawString(img_x, img_y_top - 14, f"[image not found: {fname}]")
        except Exception as e:
            c.setFont("Helvetica-Oblique", 9)
            c.drawString(img_x, img_y_top - 14, f"[image error: {fname}]")
            logging.warning("[PDF] image error for %s: %r", fname, e)

        # Right: ALL fields from df_results (except FileName)
        c.setFont("Helvetica", 10)
        for k in [ck for ck in row.index if ck != "FileName"]:
            try:
                val = row[k]
            except Exception:
                continue
            if pd.isna(val):
                continue
            sval = str(val)
            wrap_cols = int(right_col_w / 5.5)
            wrapped = _wrap_lines(f"{k}: {sval}", wrap_cols) or [f"{k}:"]

            needed = line_h * len(wrapped)
            if right_y - needed < margin:
                new_page(False)
                right_y = page_h - margin - 26 - 18

            for i, ln in enumerate(wrapped):
                c.drawString(right_x, right_y - line_h*i, ln)
            right_y -= needed

        # ROI Words block
        words = words_by_file.get(fname) or {}
        roi_keys = [k for k in words.keys()
                    if k != "FileName" and ("ROI_" in k or k.endswith("_Words"))]
        if roi_keys:
            right_y -= 6
            c.setFont("Helvetica-Bold", 10)
            if right_y - line_h < margin:
                new_page(False)
                right_y = page_h - margin - 26 - 18
            c.drawString(right_x, right_y, "ROI Words:")
            right_y -= line_h

            c.setFont("Helvetica", 9)
            for k in roi_keys:
                sval = str(words.get(k) or "")
                if not sval:
                    continue
                wrap_cols = int(right_col_w / 6.0)
                wrapped = _wrap_lines(f"{k}: {sval}", wrap_cols) or [f"{k}:"]

                needed = line_h * len(wrapped)
                if right_y - needed < margin:
                    new_page(False)
                    right_y = page_h - margin - 26 - 18

                for i, ln in enumerate(wrapped):
                    c.drawString(right_x, right_y - line_h*i, ln)
                right_y -= needed

        # move cursor for next card
        y_cursor = min(img_y_top - img_h_drawn - 10, right_y - 16)

    c.save()
    logging.info("[PDF] Wrote PDF (cards-only): %s", pdf_path)
