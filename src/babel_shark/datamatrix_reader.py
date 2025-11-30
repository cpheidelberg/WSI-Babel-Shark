#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datamatrix_reader.py — Babel-Shark DataMatrix Reader

Decodes DataMatrix from pathology label images and extracts
structured slide metadata into an Excel report.

Main features:
    Supports multiple image formats (.png, .jpg, .tif)
    Automatically parses LabID, Year, CaseNumber, Pot, BlockID, Section
    Detects invalid or unreadable barcodes and logs results
    Exports results and logs atomically to Excel and text files
    Command-line interface for run, validate, and version

Usage:
    PYTHONPATH=./src/babel_shark/ python -u src/babel_shark/datamatrix_reader.py run \
        --config ./config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import pandas as pd
import yaml
from PIL import Image
from pylibdmtx.pylibdmtx import decode
from tqdm import tqdm
from urllib.parse import quote


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a simple, consistent format.

    Args:
        level: Logging level name (e.g., "DEBUG", "INFO", "WARNING").
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Data Types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParsedID:
    """Container for parsed slide ID components."""
    LabID: str
    Year: str
    CaseNumber: str
    Pot: str
    BlockID: str
    Section: str


# --------------------------------------------------------------------------- #
# Helpers (text)
# --------------------------------------------------------------------------- #

def digits_only(x: Any) -> str:
    """Return only digits from the input string."""
    return re.sub(r"[^\d]", "", str(x) if x is not None else "")


def to_int_str_no_decimal(x: Any) -> Optional[str]:
    """Normalize numeric-like text by removing trailing .0; return None if not numeric."""
    if x is None:
        return None
    s = str(x).strip()
    m = re.fullmatch(r"\s*([0-9]+)(?:\.0+)?\s*", s)
    if m:
        return str(int(m.group(1)))
    d = digits_only(s)
    return d if d else None


def to_year4_str(y: Any) -> Optional[str]:
    """Convert to 4-digit year; handle 2-digit years by prefixing '20'."""
    s = to_int_str_no_decimal(y)
    if not s:
        return None
    if re.fullmatch(r"[0-9]{4}", s):
        return s
    if re.fullmatch(r"[0-9]{2}", s):
        return f"20{s}"
    return None


def is_year_in_valid_range(y4: Optional[str]) -> bool:
    """Check year is 2020..2030."""
    if not y4 or not re.fullmatch(r"[0-9]{4}", y4):
        return False
    yi = int(y4)
    return 2020 <= yi <= 2030

# --------------------------------------------------------------------------- #
# Helpers (paths/URIs)
# --------------------------------------------------------------------------- #

def get_excel_uri(path: Path) -> str:
    """Return Excel-friendly file:// URI for a local path (empty for non-drive paths)."""
    path = Path(path).resolve()
    if getattr(path, "drive", ""):
        return "file:///" + quote(path.as_posix())
    return ""

# --------------------------------------------------------------------------- #
# Image Decoding
# --------------------------------------------------------------------------- #

def read_matrix_from_image(image) -> Optional[str]:
    """Try decoding a DataMatrix barcode from an image via multiple scales.

    Args:
        image: BGR image (numpy array) from cv2.imread.

    Returns:
        Decoded string if found; otherwise None.
    """
    from cv2 import resize, INTER_AREA, cvtColor, COLOR_BGR2GRAY

    for scale in (0.25, 0.5, 1.0, 2.0):
        resized = resize(image, (0, 0), fx=scale, fy=scale, interpolation=INTER_AREA)
        gray = cvtColor(resized, COLOR_BGR2GRAY)
        pil_image = Image.fromarray(gray)
        decoded = decode(pil_image)
        if decoded:
            return decoded[0].data.decode("utf-8")
    return None


def parse_slide_id(slide_id_str: Any) -> ParsedID:
    """Parse structured slide ID into components (LabID, Year, CaseNumber, Pot, BlockID, Section).

    Args:
        slide_id_str: Full slide ID string.

    Returns:
        ParsedID with empty strings for missing parts.
    """
    if not slide_id_str:
        return ParsedID("", "", "", "", "", "")
    s = str(slide_id_str).split(".")[0]
    pattern = r"^([A-Z]+)(\d{4})(\d{6})S([A-Z0-9]+)-([^\-]+)-([^\-]+)(?:-[^\-]*)?$"
    m = re.match(pattern, s)
    if not m:
        return ParsedID("", "", "", "", "", "")
    return ParsedID(
        LabID=str(m.group(1)),
        Year=str(m.group(2)),
        CaseNumber=str(m.group(3)),
        Pot=str(m.group(4)),
        BlockID=str(m.group(5)),
        Section=str(m.group(6)),
    )


# --------------------------------------------------------------------------- #
# Core Processing
# --------------------------------------------------------------------------- #

def process_all_images(config: Dict[str, Any]) -> Tuple[List[Dict[str, str]], List[str], Dict[str, Path]]:
    """Scan input images, attempt DataMatrix decode, validate/normalize, and collect rows.

    Args:
        config: Loaded YAML configuration.

    Returns:
        results: List of output dict rows (strings only).
        log_lines: List of textual log lines for the log file.
        paths: Dict containing output paths used (run_dir, failed_dir, output_excel, log_file).
    """
    timestamp = config.get("run_timestamp", datetime.now().strftime("%Y-%m-%d_%H-%M"))
    output_base = Path(config["run_output_dir"])
    output_run_dir = Path(config["output_run_dir"]) if "output_run_dir" in config else output_base / timestamp

    failed_output_dir = output_run_dir / "failed_datamatrix"
    output_excel = output_run_dir / f"datamatrix_results_{timestamp}.xlsx"
    log_file = output_run_dir / f"log_datamatrix_{timestamp}.txt"
    input_folder = Path(config["label_crops_dir"])

    output_run_dir.mkdir(parents=True, exist_ok=True)
    failed_output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, str]] = []
    log_lines: List[str] = []

    image_paths: List[Path] = list(input_folder.glob("*.*"))
    for image_path in tqdm(image_paths, desc="Scanning images"):
        file_name = image_path.name
        image = cv2.imread(str(image_path))
        if image is None:
            log_lines.append(f"[Load Error] Could not read {file_name}")
            try:
                shutil.copy(str(image_path), failed_output_dir / file_name)
            except Exception as e:
                log_lines.append(f"[Copy Error] {file_name} :: {e}")
            datamatrix = ""
            parsed = ParsedID("", "", "", "", "", "")
            final_status = "failed"
        else:
            datamatrix = read_matrix_from_image(image)
            parsed = parse_slide_id(datamatrix)

            if datamatrix:
                parsed_empty = not any([parsed.LabID, parsed.Year, parsed.CaseNumber, parsed.Pot, parsed.BlockID, parsed.Section])
                starts_with_v = isinstance(datamatrix, str) and datamatrix.startswith("V")

                year4 = to_year4_str(parsed.Year) if parsed.Year else None
                has_five = all([parsed.LabID, year4, parsed.CaseNumber, parsed.Pot, parsed.BlockID])
                year_ok = is_year_in_valid_range(year4)

                invalid_gate = (not has_five) or (not year_ok) or parsed_empty or starts_with_v
                final_status = "failed" if invalid_gate else "success"

                if invalid_gate:
                    try:
                        shutil.copy2(str(image_path), str(failed_output_dir / file_name))
                        reason = (
                            "Parsed-Empty" if parsed_empty else (
                                "StartsWithV" if starts_with_v else (
                                    "InvalidYear" if (has_five and not year_ok) else "MissingParts"
                                )
                            )
                        )
                        log_lines.append(f"[Copy to FailedFolder] {file_name} :: {reason} :: DM={datamatrix}")
                    except Exception as e:
                        log_lines.append(f"[Copy Error] {file_name} :: {e}")

                # Normalize fields (string/int-as-text normalization)
                year_norm = year4 or ""
                case_norm = to_int_str_no_decimal(parsed.CaseNumber) or ""
                block_norm = to_int_str_no_decimal(parsed.BlockID) or ""
                section_norm = to_int_str_no_decimal(parsed.Section) or ""
                pot_norm = str(parsed.Pot).strip().upper() if parsed.Pot else ""
                lab_norm = str(parsed.LabID).strip().upper() if parsed.LabID else ""

                parsed = ParsedID(
                    LabID=lab_norm,
                    Year=year_norm,
                    CaseNumber=case_norm,
                    Pot=pot_norm,
                    BlockID=block_norm,
                    Section=section_norm,
                )
            else:
                final_status = "failed"
                datamatrix = ""
                log_lines.append(f"[Not Detected] {file_name}")
                try:
                    shutil.copy(str(image_path), failed_output_dir / file_name)
                except Exception as e:
                    log_lines.append(f"[Copy Error] {file_name} :: {e}")

        uri = get_excel_uri(image_path)
        link_text = final_status
        hyperlink = f'=HYPERLINK("{uri}", "{link_text}")' if uri else link_text

        results.append({
            "FileName": str(file_name),
            "DataMatrix": str(datamatrix),
            "LabID": str(parsed.LabID),
            "Year": str(parsed.Year),
            "CaseNumber": str(parsed.CaseNumber),
            "Pot": str(parsed.Pot),
            "BlockID": str(parsed.BlockID),
            "Section": str(parsed.Section),
            "Status": str(final_status),
            "HyperlinkPath": str(hyperlink),
        })

    paths = {
        "run_dir": output_run_dir,
        "failed_dir": failed_output_dir,
        "output_excel": output_excel,
        "log_file": log_file,
    }
    return results, log_lines, paths


# --------------------------------------------------------------------------- #
# Core Processing
# --------------------------------------------------------------------------- #

def _clean_series_intstr(s: pd.Series) -> pd.Series:
    """Map a Series to normalized int-like strings without decimals."""
    return s.astype("string").map(lambda x: to_int_str_no_decimal(x) if x is not None else "")


def save_results_excel_atomic(results: List[Dict[str, str]], out_path: Path) -> None:
    """Write Excel file atomically (content identical to original script)."""
    if not results:
        return

    df = pd.DataFrame(results)

    # dtype/normalization identical to original intent
    for c in ["LabID", "Pot"]:
        if c in df.columns:
            df[c] = df[c].astype("string")
    for c in ["Year", "CaseNumber", "BlockID", "Section"]:
        if c in df.columns:
            df[c] = _clean_series_intstr(df[c])

    df["DataMatrix"] = df["DataMatrix"].astype("string")
    df["Status"] = df["Status"].astype("string")
    df["FileName"] = df["FileName"].astype("string")

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(
        tmp_path,
        engine="xlsxwriter",
        engine_kwargs={"options": {"strings_to_numbers": False, "strings_to_formulas": False}},
    ) as writer:
        df_out = df.drop(columns=["HyperlinkPath", "Status"]).copy()
        df_out.to_excel(writer, index=False, sheet_name="Results")

        workbook = writer.book
        worksheet = writer.sheets["Results"]
        text_fmt = workbook.add_format({"num_format": "@"})

        for col_idx, col_name in enumerate(df_out.columns):
            if col_name != "Status":
                worksheet.set_column(col_idx, col_idx, 22, text_fmt)

        # Deliberately not writing Status/Hyperlink to sheet (same as original)

    os.replace(tmp_path, out_path)


def save_log_atomic(log_lines: List[str], log_path: Path) -> None:
    """Write log text atomically (content identical to original script)."""
    tmp_path = log_path.with_suffix(log_path.suffix + ".tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(f"DataMatrix scan completed at {datetime.now()}\n")
        for line in log_lines:
            f.write(line + "\n")

    os.replace(tmp_path, log_path)


# --------------------------------------------------------------------------- #
# Core Processing
# --------------------------------------------------------------------------- #

def run_with_config(config_path: Path) -> None:
    """Main processing entry: load config, process images, save outputs atomically."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    results, log_lines, paths = process_all_images(config)

    if results:
        save_results_excel_atomic(results, paths["output_excel"])
        LOGGER.info("DataMatrix results saved: %s", paths["output_excel"])

    save_log_atomic(log_lines, paths["log_file"])
    LOGGER.info("[OK] Log saved: %s", paths["log_file"])


def validate_config(config_path: Path) -> int:
    """Minimal schema check for required keys (non-breaking auxiliary command).

    Returns:
        0 if valid; non-zero otherwise.
    """
    required_keys = [
        "run_output_dir",
        "label_crops_dir",
    ]
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        LOGGER.error("Cannot read config: %s", e)
        return 2

    missing = [k for k in required_keys if k not in cfg]
    if missing:
        LOGGER.error("Missing required config keys: %s", ", ".join(missing))
        return 3

    # Optional keys are allowed; no further checks to avoid behavior changes.
    LOGGER.info("Config looks OK.")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_cli_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Extract DataMatrix codes from label images and export to Excel/log."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_run = subparsers.add_parser("run", help="Run the full pipeline.")
    p_run.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")

    p_val = subparsers.add_parser("validate", help="Validate configuration file (non-breaking).")
    p_val.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")

    subparsers.add_parser("version", help="Show version and exit.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    """CLI entry point."""
    parser = build_cli_parser()
    args = parser.parse_args(args=list(argv) if argv is not None else None)

    setup_logging(level=getattr(args, "log_level", "INFO"))

    if args.command == "version":
        print(__VERSION__)  # version output is typically printed; keep minimal & intentional
        return 0

    if args.command == "validate":
        return validate_config(Path(args.config))

    if args.command == "run":
        run_with_config(Path(args.config))
        return 0

    return 1

# --------------------------------------------------------------------------- #
# Script execution guard
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    raise SystemExit(main())
