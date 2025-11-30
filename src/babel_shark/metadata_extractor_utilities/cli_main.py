# -*- coding: utf-8 -*-
"""
CLI entry-point and runner for roi_metadata_extractor.py (Babel-Shark Fallback stratagy)

This module wires together:
  Config loading and validation
  Input image discovery (explicit label dir OR datamatrix-failed dir)
  Running `RoiMetadataExtractor` over images
  Writing CSV/XLSX/PDF outputs atomically
  A small CLI with `run` and `version` subcommands

"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .utils.logging_utils import setup_logging
from .utils.io_utils import (
    _timestamp,
    _find_images,
    _load_yaml_config,
    _atomic_write_csv,
    _atomic_write_excel,
    ensure_dir,
)
from .extractor import RoiMetadataExtractor
from .report.pdf_report import _write_pdf_report
from .parsing.text_utils import to_year4_str, build_datamatrix  # NEW: for rebuilding DataMatrix with updated Section


__MODULE_PROG__ = "ROI-OCR-Based-Metadata-Extractor"


# ------------------------------ Validation & Input picking ------------------------------ #
def _validate_config(cfg: Dict) -> Tuple[bool, List[str]]:
    """Validate minimal required configuration keys.

    Groups (at least one in each group must be provided):

    ROI set (pick one):
        - `ROI_set_file`
        - `roiset_selector.roiset_root`

    Inputs (pick one):
        - Explicit label folders: `staging_dir` / `label_crops_dir`
                                  / `label_root_dir` / `label_image_dir`
                                  / `watch_dir`
        - Datamatrix failed folders: `datamatrix_failed_folder`
                                     or implicit `<output_run_dir>/failed_datamatrix`
        - Legacy: `failed_output_dir`

    Outputs (pick one):
        - `run_output_dir`
        - any of: `output_csv`, `words_csv`, `output_xlsx`, `output_pdf`, `output_run_dir`

    Args:
        cfg: Parsed YAML/JSON configuration as a dictionary.

    Returns:
        (ok, missing_keys) where `ok` is True if valid, otherwise False with a list of missing requirements.
    """
    missing: List[str] = []

    # ROI source
    if not (cfg.get("ROI_set_file") or (cfg.get("roiset_selector") or {}).get("roiset_root")):
        missing.append("ROI_set_file or roiset_selector.roiset_root")

    # Inputs
    has_explicit_label_dir = any(
        bool(cfg.get(k))
        for k in (
            "staging_dir",
            "label_crops_dir",
            "label_root_dir",
            "label_image_dir",
            "watch_dir",
        )
    )
    has_dm_failed_dir = bool(cfg.get("datamatrix_failed_folder") or cfg.get("output_run_dir"))
    has_legacy_failed = bool(cfg.get("failed_output_dir"))

    if not (has_explicit_label_dir or has_dm_failed_dir or has_legacy_failed):
        missing.append(
            "image source missing: provide one of "
            "[staging_dir/label_crops_dir/label_root_dir/label_image_dir/watch_dir] "
            "or [datamatrix_failed_folder or output_run_dir] (or legacy failed_output_dir)"
        )

    # Outputs
    has_explicit_out = any(
        bool(cfg.get(k))
        for k in ("output_csv", "words_csv", "output_xlsx", "output_pdf", "output_run_dir")
    )
    if not (cfg.get("run_output_dir") or has_explicit_out):
        missing.append(
            "run_output_dir (or explicit outputs via output_csv/words_csv/output_xlsx/output_pdf/output_run_dir)"
        )

    return (len(missing) == 0), missing


def _has_images_in_dir(p: str) -> bool:
    """Return True if the directory contains at least one PNG/JPG/JPEG (recursively)."""
    try:
        files = _find_images(p)
        return len(files) > 0
    except Exception:
        return False


def _pick_input_dir(args: argparse.Namespace, cfg: Dict) -> str:
    """Decide which folder contains label images to process.

    Priority:
      1) CLI override: --input_dir
      2) Explicit label dirs in config:
            label_crops_dir / label_root_dir / label_image_dir /
            staging_dir / watch_dir
      3) Datamatrix failed (explicit): datamatrix_failed_folder
      4) Datamatrix failed (implicit): <output_run_dir>/failed_datamatrix
      5) Legacy fallback: failed_output_dir

    Returns:
        The first directory that actually contains images.

    Raises:
        RuntimeError: If none of the candidate directories contain images.
    """
    candidates: List[Path] = []

    # 1) CLI override
    if getattr(args, "input_dir", ""):
        candidates.append(Path(args.input_dir))

    # 2) Explicit label sources
    for k in (
        "label_crops_dir",
        "label_root_dir",
        "label_image_dir",
        "staging_dir",
        "watch_dir",
    ):
        v = cfg.get(k) or ""
        if v:
            candidates.append(Path(v))

    # 3) Explicit DM-failed
    v = cfg.get("datamatrix_failed_folder") or ""
    if v:
        candidates.append(Path(v))

    # 4) Implicit DM-failed inside output_run_dir
    v = cfg.get("output_run_dir") or ""
    if v:
        candidates.append(Path(v) / "failed_datamatrix")

    # 5) Legacy
    v = cfg.get("failed_output_dir") or ""
    if v:
        candidates.append(Path(v))

    # Resolve the first candidate that actually has images
    for c in candidates:
        if _has_images_in_dir(str(c)):
            return str(c)

    raise RuntimeError(
        "No input folder with PNG/JPG images was found. Provide --input_dir or set one of: "
        "label_crops_dir/label_root_dir/label_image_dir/staging_dir/watch_dir "
        "or datamatrix_failed_folder (or use output_run_dir to imply output_run_dir/failed_datamatrix)."
    )


# ------------------------------------------- Core runner ------------------------------------------- #
def _collect_rows_for_results(filename: str, parsed: Dict[str, str], success: bool) -> Dict[str, str]:
    """Build a single results-row dict for the output table.

    Args:
        filename: Original image file path.
        parsed: Parsed field mapping (LabID, Year, CaseNumber, Pot, BlockID, Section, Stain, DataMatrix).
        success: True if extraction succeeded, False otherwise.

    Returns:
        A dictionary suitable for appending to the results DataFrame.
    """
    return {
        "FileName": os.path.basename(filename),
        "LabID": parsed.get("LabID", ""),
        "Year": parsed.get("Year", ""),
        "CaseNumber": parsed.get("CaseNumber", ""),
        "Pot": parsed.get("Pot", ""),
        "BlockID": parsed.get("BlockID", ""),
        "Section": parsed.get("Section", ""),
        "Stain": parsed.get("Stain", ""),
        "DataMatrix": parsed.get("DataMatrix", ""),
        "Status": "Success" if success else "Failed",
    }


def cmd_run(args: argparse.Namespace) -> None:
    """Execute the main 'run' workflow based on CLI arguments.

    Steps:
      - Configure logging and load YAML config.
      - Apply CLI overrides to config.
      - Validate configuration.
      - Resolve output run directory and filenames.
      - Find input images and run `RoiMetadataExtractor` on each.
      - Write CSV/XLSX/PDF outputs (atomically where applicable).

    Args:
        args: Parsed CLI arguments for the 'run' command.

    Raises:
        SystemExit: With code 2 if configuration is invalid.
    """
    setup_logging(args.log_level)
    cfg = _load_yaml_config(args.config)

    # CLI overrides (non-breaking)
    if args.input_dir:
        cfg["staging_dir"] = args.input_dir
    if args.roi_json:
        cfg["ROI_set_file"] = args.roi_json
    if args.pxl_offset is not None:
        cfg["pxl_offset"] = int(args.pxl_offset)
    if args.debug_folder:
        cfg["debug_parts_root"] = args.debug_folder

    ok, miss = _validate_config(cfg)
    if not ok:
        logging.error("Invalid config. Missing keys: %s", ", ".join(miss))
        raise SystemExit(2)

    # Output locations
    out_base = cfg.get("run_output_dir") or "./output"
    # ---- FIX B (stable tag from runner; fallback to new timestamp) ----
    run_tag = cfg.get("run_timestamp") or _timestamp()
    out_run = cfg.get("output_run_dir") or os.path.join(out_base, run_tag)
    out_run = ensure_dir(out_run)

    # Keep CSVs time-stamped (optional), but XLSX/PDF must be fixed within a run
    results_csv = cfg.get("output_csv") or os.path.join(out_run, f"roi_results_{run_tag}.csv")
    words_csv = cfg.get("words_csv") or os.path.join(out_run, f"roi_words_wide_{run_tag}.csv")
    # ---- FIX B (fixed names for XLSX/PDF to avoid proliferation inside same run) ----
    results_xlsx = cfg.get("output_xlsx") or os.path.join(out_run, "roi_results.xlsx")
    results_pdf = cfg.get("output_pdf") or os.path.join(out_run, "roi_results.pdf")

    # Input resolution (supports explicit label dirs or DM-failed)
    try:
        inp_dir = _pick_input_dir(args, cfg)
    except RuntimeError as e:
        # If no images, still materialize empty outputs for robustness
        logging.warning("%s", e)
        inp_dir = cfg.get("staging_dir") or cfg.get("failed_output_dir") \
            or cfg.get("datamatrix_failed_folder") or cfg.get("watch_dir") or "."

    cfg["staging_dir"] = inp_dir  # keep legacy behavior that the extractor reads this key

    logging.info("Input dir      : %s", inp_dir)
    logging.info("Output run dir : %s", out_run)

    extractor = RoiMetadataExtractor(cfg)

    files = _find_images(inp_dir)
    empty_res_cols = [
        "FileName",
        "LabID",
        "Year",
        "CaseNumber",
        "Pot",
        "BlockID",
        "Section",
        "Stain",
        "DataMatrix",
        "Status",
    ]
    df_res = pd.DataFrame(columns=empty_res_cols)
    df_words = pd.DataFrame(columns=["FileName"])

    if not files:
        logging.warning("No input images found in %s", inp_dir)
        _atomic_write_csv(df_res, results_csv)
        _atomic_write_csv(df_words, words_csv)
        try:
            _atomic_write_excel({"ROI_Results": df_res, "ROI_Words_Wide": df_words}, results_xlsx)
        except Exception as e:
            logging.warning("Could not write empty XLSX: %s", e)
        try:
            _write_pdf_report(results_pdf, df_res, df_words, title="ROI Fallback Results (empty)")
        except Exception as e:
            logging.warning("Could not write empty PDF: %s", e)
        return

    rows_results: List[Dict[str, str]] = []
    rows_words: List[Dict[str, str]] = []
    pxl_offset = int(cfg.get("pxl_offset", 8))

    # Section counters for cases where Section stayed at default "500"
    # Key: (LabID, Year, CaseNumber, Pot, BlockID) -> last assigned Section
    section_counters: Dict[Tuple[str, str, str, str, str], int] = {}

    import time
    start_t = time.time()

    for idx, fp in enumerate(files, 1):
        bname = os.path.basename(fp)
        logging.info("[%d/%d] %s", idx, len(files), bname)

        import cv2
        img = cv2.imread(fp)
        if img is None or img.size == 0:
            logging.warning("Cannot read image (cv2.imread returned None): %s", fp)
            rows_results.append(_collect_rows_for_results(fp, {}, False))
            rows_words.append({"FileName": bname})
            continue

        parsed, success, roi_words = extractor.run_on_image(
            img,
            img_name=bname,
            pxl_offset=pxl_offset,
        )

        # Adjust Section when OCR could not read it (default "500")
        sec_val = (parsed.get("Section") or "").strip()
        if sec_val == "500":
            core_key = (
                parsed.get("LabID", "") or "",
                parsed.get("Year", "") or "",
                parsed.get("CaseNumber", "") or "",
                parsed.get("Pot", "") or "",
                parsed.get("BlockID", "") or "",
            )
            if core_key not in section_counters:
                section_counters[core_key] = 500
            else:
                section_counters[core_key] += 1
            new_section = section_counters[core_key]
            parsed["Section"] = str(new_section)

            y4 = to_year4_str(parsed.get("Year"))
            dm_new = build_datamatrix(
                parsed.get("LabID"),
                y4,
                parsed.get("CaseNumber"),
                parsed.get("Pot"),
                parsed.get("BlockID"),
                parsed.get("Section"),
            )
            if dm_new:
                parsed["DataMatrix"] = dm_new
                success = True

        rows_results.append(_collect_rows_for_results(fp, parsed, success))
        ww: Dict[str, str] = {"FileName": bname}
        ww.update(roi_words or {})
        rows_words.append(ww)

    dur = time.time() - start_t
    logging.info("Done in %.2fs. Writing outputs...", dur)

    df_res = pd.DataFrame(rows_results)
    df_words = pd.DataFrame(rows_words)

    _atomic_write_csv(df_res, results_csv)
    _atomic_write_csv(df_words, words_csv)

    try:
        _atomic_write_excel({"ROI_Results": df_res, "ROI_Words_Wide": df_words}, results_xlsx)
        logging.info("Wrote XLSX: %s", results_xlsx)
    except Exception as e:
        logging.warning("Could not write XLSX: %s", e)

    try:
        _write_pdf_report(results_pdf, df_res, df_words, title="ROI Fallback Results")
        logging.info("Wrote PDF: %s", results_pdf)
    except Exception as e:
        logging.warning("Could not write PDF: %s", e)


# ------------------------------------------- CLI ------------------------------------------- #

def _build_cli(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Build CLI for this module."""
    parser = argparse.ArgumentParser(prog=__MODULE_PROG__, description="ROI metadata extractor (fallback).")
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Run extraction over a folder of label images.")
    p_run.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config file.")
    p_run.add_argument("--input-dir", dest="input_dir", type=str, default="", help="Override input folder (images).")
    p_run.add_argument("--roi-json", dest="roi_json", type=str, default="", help="Override ROI JSON path.")
    p_run.add_argument("--pxl-offset", dest="pxl_offset", type=int, default=None, help="Padding pixels for ROI crops.")
    p_run.add_argument("--debug-folder", dest="debug_folder", type=str, default="", help="Where to save debug parts.")
    p_run.add_argument("--log-level", dest="log_level", type=str, default="INFO", help="Logging level.")

    sub.add_parser("version", help="Print version and exit.")

    args = parser.parse_args(argv)
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_cli(argv)

    if args.command == "version":
        setup_logging("INFO")
        logging.info("version: 1.0.0")
        return 0

    if args.command == "run":
        cmd_run(args)
        return 0

    setup_logging("INFO")
    logging.info("No command provided. Use one of: run, version.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
