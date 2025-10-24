#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slide_id_generator.py — Babel-Shark Slide ID Generator

Generates unique SlideIDs and standardized file names by merging
DataMatrix and stain results, then renames WSIs accordingly.

Main features:
    Merges DataMatrix and stain outputs via case matching
    Builds SlideID and CaseID identifiers
    Moves and renames WSIs atomically into structured folders
    Produces metadata CSV for downstream database updates
    Command-line interface for run, validate, and version

Usage:
    PYTHONPATH=./src python -u src/slide_id_generator.py run \
        --config ./config/config.yaml
"""


from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml


# --------------------------------------------------------------------------- #
# Constants 
# --------------------------------------------------------------------------- #

INVALID_FS_CHARS = '<>:"/\\|?*'
FS_TRANS_TABLE = str.maketrans({ch: "_" for ch in INVALID_FS_CHARS})

# --------------------------------------------------------------------------- #
# Logging Setup
# --------------------------------------------------------------------------- #

def setup_logging(level: str = "INFO") -> None:
    """Configure root logger to mimic prior print-style output (no level prefix)."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Data Classes
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ConfigPaths:
    """Container for all path-like configuration entries."""
    output_run_dir: Path
    source_dir: Path
    renamed_dir: Path
    failed_dir: Path
    datamatrix_excel: Path
    stain_excel: Path
    metadata_csv: Path
    timestamp: str


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def sanitize_fs(name: str) -> str:
    """Replace filesystem-invalid chars and trim trailing dots/spaces (Windows-safe)."""
    if name is None:
        return ""
    out = str(name).translate(FS_TRANS_TABLE)
    return out.strip(" .")


def digits_only(x: object) -> str:
    """Extract only digits from any value."""
    return re.sub(r"[^\d]", "", str(x) if x is not None else "")


def to_int_str_no_decimal(x: object) -> Optional[str]:
    """Render a number-like value as an integer string without decimals; return None if empty.

    Supports both '2453.0' and '2453,0'.
    """
    if x is None:
        return None
    s = str(x).strip()
    m = re.fullmatch(r"\s*([0-9]+)(?:[.,]0+)?\s*", s)
    if m:
        return str(int(m.group(1)))
    d = digits_only(s)
    return d if d else None


def to_year4_str(y: object) -> Optional[str]:
    """Map 2-digit years to 20xx; keep 4-digit years as-is; else None."""
    s = to_int_str_no_decimal(y)
    if not s:
        return None
    if re.fullmatch(r"[0-9]{2}", s):
        return f"20{s}"
    if re.fullmatch(r"[0-9]{4}", s):
        return s
    return None


def to_case6_str(c: object) -> Optional[str]:
    """Zero-pad case numbers to 6 digits (handles inputs like '2453.0' correctly)."""
    s = to_int_str_no_decimal(c)
    if not s:
        return None
    return f"{int(s):06d}"


def normalize_stain(s: Optional[str]) -> str:
    """Normalize common stain spellings; default to H&E when missing."""
    if s is None:
        return "H&E"
    t_raw = str(s).strip()
    t = t_raw.upper().replace(" ", "")
    if t in {"HE", "H-E", "H& E", "H &E", "H & E"}:
        return "H&E"
    if t in {"H&E", "H&Ε"}:
        return "H&E"
    return t_raw.strip()


def build_datamatrix(
    lab: object,
    year: object,
    case: object,
    pot: object,
    blockid: object,
    section: object,
) -> Optional[str]:
    """Build DataMatrix using the standard logic.

    Pattern: {LabID}{Year4}{Case6}S{Pot}-{BlockID}-{Section}
    Returns None if any required field is missing.
    """
    lab_s = (str(lab).strip().upper() if lab else "")
    y4 = to_year4_str(year)
    c6 = to_case6_str(case)
    pot_s = (str(pot).strip().upper() if pot else "")
    bid = to_int_str_no_decimal(blockid) or (str(blockid).strip() if blockid is not None else "")
    sec = to_int_str_no_decimal(section) or (str(section).strip() if section is not None else "")
    if not (lab_s and y4 and c6 and pot_s and bid and sec):
        return None
    return f"{lab_s}{y4}{c6}S{pot_s}-{bid}-{sec}"


def prefer_ext(candidates: List[Path]) -> Optional[Path]:
    """Among multiple 'base.*' matches, prefer typical WSI formats first."""
    if not candidates:
        return None
    priority = [".svs", ".ndpi", ".scn", ".mrxs", ".bif", ".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    cand_by_ext = {p.suffix.lower(): p for p in candidates}
    for ext in priority:
        if ext in cand_by_ext:
            return cand_by_ext[ext]
    return candidates[0]


def unique_path(target: Path) -> Path:
    """If a target path exists, append _1, _2, ... until unique."""
    if not target.exists():
        return target
    stem, suf = target.stem, target.suffix
    parent = target.parent
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suf}"
        if not cand.exists():
            return cand
        i += 1


def read_excel_safe(path: Path) -> pd.DataFrame:
    """Read Excel as TEXT to avoid float coercion; return empty DataFrame on failure."""
    try:
        return pd.read_excel(path, dtype=str)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[ERROR] Cannot read Excel: {path} -> {exc}")
        return pd.DataFrame()


def ensure_metadata_schema(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Ensure slide_metadata.csv has required columns (without 'Status')."""
    cols = ["FileName", "SlideID", "CaseID", "NewFileName", "OriginalFileName", "method of extraction"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols].copy()


def prefer_ext_local(candidates: List[Path]) -> Optional[Path]:
    """Prefer best WSI extension among base.* matches (local shadow helper)."""
    if not candidates:
        return None
    priority = [".svs", ".ndpi", ".scn", ".mrxs", ".bif", ".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    cand_by_ext = {p.suffix.lower(): p for p in candidates}
    for ext in priority:
        if ext in cand_by_ext:
            return cand_by_ext[ext]
    return candidates[0]


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write CSV atomically (temp file + os.replace)."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


# --------------------------------------------------------------------------- #
# Core Pipeline
# --------------------------------------------------------------------------- #

def load_config(config_path: Path) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def materialize_paths(cfg: Dict) -> ConfigPaths:
    """Construct and ensure all paths from config without changing original behavior."""
    timestamp = cfg.get("run_timestamp", datetime.now().strftime("%Y-%m-%d_%H-%M"))
    output_run_dir = Path(cfg.get("output_run_dir", Path(cfg["run_output_dir"]) / timestamp))
    output_run_dir.mkdir(parents=True, exist_ok=True)

    source_dir = Path(cfg["staging_dir"])
    renamed_dir = Path(cfg["final_output_dir"])
    failed_dir = Path(cfg["failed_output_dir"])

    for d in (renamed_dir, failed_dir):
        d.mkdir(parents=True, exist_ok=True)

    dm_path = Path(cfg.get("datamatrix_output_excel", output_run_dir / f"datamatrix_results_{timestamp}.xlsx"))
    stain_path = Path(cfg.get("stain_output_excel", output_run_dir / f"stain_results_{timestamp}.xlsx"))
    metadata_csv = Path(cfg["metadata_csv_path"])
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    return ConfigPaths(
        output_run_dir=output_run_dir,
        source_dir=source_dir,
        renamed_dir=renamed_dir,
        failed_dir=failed_dir,
        datamatrix_excel=dm_path,
        stain_excel=stain_path,
        metadata_csv=metadata_csv,
        timestamp=timestamp,
    )


def infer_method(row: pd.Series) -> str:
    """Infer 'method of extraction' based on ExtractionMethod value."""
    em = str(row.get("ExtractionMethod") or "").strip().lower()
    return "ROIbase" if em == "roibase" else "Datamatrix"


def pick_source_and_dest(
    base: str,
    slide_id: str,
    case_id: str,
    source_dir: Path,
    renamed_dir: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Find source WSI in staging and compute destination unique path."""
    matches = list(source_dir.glob(base + ".*"))
    if not matches:
        return None, None
    src_file = prefer_ext_local(matches) or matches[0]
    safe_slideid = sanitize_fs(slide_id)
    dst_folder = renamed_dir / str(case_id)
    dst_folder.mkdir(parents=True, exist_ok=True)
    target = dst_folder / f"{safe_slideid}{src_file.suffix}"
    return src_file, unique_path(target)


def run_pipeline(cfg: Dict) -> None:
    """Execute the original pipeline logic with structural improvements only."""
    paths = materialize_paths(cfg)

    # 1) Load inputs
    df_dm = read_excel_safe(paths.datamatrix_excel)
    df_stain = read_excel_safe(paths.stain_excel)

    need_dm_cols = [
        "FileName",
        "DataMatrix",
        "LabID",
        "Year",
        "CaseNumber",
        "Pot",
        "BlockID",
        "Section",
        "ExtractionMethod",
    ]
    for c in need_dm_cols:
        if c not in df_dm.columns:
            df_dm[c] = None

    df_dm = df_dm.dropna(subset=["FileName", "LabID", "Year", "CaseNumber"]).copy()

    if "Stain" not in df_stain.columns:
        df_stain["Stain"] = None
    df_stain = df_stain[["FileName", "Stain"]].drop_duplicates()

    # 2) Merge stain (LEFT)
    df = pd.merge(df_dm, df_stain, on="FileName", how="left")

    # 3) Normalize & compute derived fields (unchanged logic)
    df["LabID"] = df["LabID"].astype(str).str.strip().str.upper()
    df["Year4"] = df["Year"].apply(to_year4_str).astype("string")
    df["Case6"] = df["CaseNumber"].apply(to_case6_str).astype("string")
    df["Pot"] = df["Pot"].fillna("A").astype(str).str.strip().str.upper().astype("string")
    df["BlockID"] = df["BlockID"].apply(lambda x: to_int_str_no_decimal(x) or "1").astype("string")
    df["Section"] = df["Section"].apply(lambda x: to_int_str_no_decimal(x) or "1").astype("string")
    df["Stain"] = df["Stain"].apply(normalize_stain).astype("string")

    # 4) Rebuild/normalize DataMatrix (keep original if rebuild fails)
    df["DataMatrix_v2"] = df.apply(
        lambda r: build_datamatrix(r["LabID"], r["Year4"], r["Case6"], r["Pot"], r["BlockID"], r["Section"]),
        axis=1,
    )
    df["DataMatrix"] = df["DataMatrix_v2"].where(
        df["DataMatrix_v2"].notna() & (df["DataMatrix_v2"] != ""), df["DataMatrix"]
    )
    df.drop(columns=["DataMatrix_v2"], inplace=True)

    # 5) IDs
    df["CaseID"] = (df["LabID"].fillna("") + df["Year4"].fillna("") + df["Case6"].fillna("")).astype("string")
    df["Stain"] = df["Stain"].fillna("H&E")
    df["SlideID"] = (df["DataMatrix"].astype(str) + "-" + df["Stain"].astype(str)).astype("string")

    # 6) method of extraction
    df["method of extraction"] = df.apply(infer_method, axis=1).astype("string")

    # 7) Original base name (from label image name)
    df["OriginalBase"] = df["FileName"].str.replace(r"\.(png|jpg|jpeg)$", "", regex=True)

    # 8) Build new filename by finding the actual WSI in staging
    moved_files: List[str] = []
    records: List[Dict[str, str]] = []

    for _, row in df.iterrows():
        src_file, dst_file = pick_source_and_dest(
            base=str(row["OriginalBase"]),
            slide_id=str(row["SlideID"]),
            case_id=str(row["CaseID"]),
            source_dir=paths.source_dir,
            renamed_dir=paths.renamed_dir,
        )
        if src_file is None or dst_file is None:
            logger.warning(f"[WARN] Source file not found in staging for base: {row.get('OriginalBase')}")
            continue
        try:
            shutil.move(str(src_file), str(dst_file))
            moved_files.append(src_file.name)
            records.append(
                {
                    "FileName": str(row["FileName"]),
                    "SlideID": sanitize_fs(str(row["SlideID"])),
                    "CaseID": str(row["CaseID"]),
                    "OriginalFileName": src_file.name,
                    "NewFileName": dst_file.name,
                    "method of extraction": str(row.get("method of extraction", "metadata")),
                }
            )
            logger.info(f"[RENAMED] {src_file.name}  ->  {dst_file}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[ERROR] Could not move {src_file} to {dst_file}: {exc}")

    # 9) Move unmatched files to failed/
    for file in paths.source_dir.glob("*"):
        if file.is_file() and (file.name not in moved_files):
            try:
                shutil.move(str(file), str(paths.failed_dir / file.name))
                logger.info(f"[INFO] Moved unmatched file to failed: {paths.failed_dir / file.name}")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[WARN] Could not move unmatched {file.name} to failed: {exc}")

    # 10) Update slide_metadata.csv (append or upsert by SlideID) — WITHOUT 'Status'
    try:
        if paths.metadata_csv.exists():
            meta = pd.read_csv(paths.metadata_csv)
            cols = ["FileName", "SlideID", "CaseID", "NewFileName", "OriginalFileName", "method of extraction"]
            for c in cols:
                if c not in meta.columns:
                    meta[c] = None
            meta = meta[cols]
        else:
            meta = pd.DataFrame(
                columns=["FileName", "SlideID", "CaseID", "NewFileName", "OriginalFileName", "method of extraction"]
            )

        df_run = pd.DataFrame.from_records(
            records,
            columns=["FileName", "SlideID", "CaseID", "OriginalFileName", "NewFileName", "method of extraction"],
        )

        if not meta.empty and "SlideID" in meta.columns:
            new_ids = set(df_run["SlideID"].astype(str))
            meta = meta[~meta["SlideID"].astype(str).isin(new_ids)]

        meta_updated = pd.concat([meta, df_run], ignore_index=True)

        # Atomic write
        atomic_write_csv(meta_updated, paths.metadata_csv)
        logger.info(f"[OK] slide_metadata.csv updated: {paths.metadata_csv}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[WARN] Could not update slide_metadata.csv: {exc}")

    logger.info("[DONE] SlideIDcreator finished.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def cli(argv: Optional[List[str]] = None) -> int:
    """Command-line interface with subcommands (run/validate/version)."""
    parser = argparse.ArgumentParser(description="SlideID Generator (structure-only refactor)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run pipeline with the given config")
    p_run.add_argument("--config", type=str, required=True, help="Path to config YAML")
    p_run.add_argument("--log-level", type=str, default="INFO", help="Logging level (default: INFO)")

    p_val = sub.add_parser("validate", help="Validate config YAML keys and paths")
    p_val.add_argument("--config", type=str, required=True, help="Path to config YAML")
    p_val.add_argument("--log-level", type=str, default="INFO", help="Logging level (default: INFO)")

    sub.add_parser("version", help="Show version")

    args = parser.parse_args(argv)

    if args.command == "version":
        setup_logging("INFO")
        logger.info(VERSION)
        return 0

    if args.command in {"run", "validate"}:
        setup_logging(args.log_level)

        cfg_path = Path(args.config)
        if not cfg_path.exists():
            logger.error(f"[ERROR] Config file not found: {cfg_path}")
            return 2

        try:
            cfg = load_config(cfg_path)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[ERROR] Failed to load config: {exc}")
            return 2

        required_keys = [
            "run_output_dir",
            "staging_dir",
            "final_output_dir",
            "failed_output_dir",
            "metadata_csv_path",
        ]

        missing = [k for k in required_keys if k not in cfg]
        if missing:
            logger.error(f"[ERROR] Missing required config keys: {', '.join(missing)}")
            return 2

        # Probe path existence for folders that must exist or will be created
        try:
            _ = materialize_paths(cfg)  # also ensures dirs as original behavior
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[ERROR] Invalid config values (paths?): {exc}")
            return 2

        if args.command == "validate":
            logger.info("[OK] Config looks valid.")
            return 0

        if args.command == "run":
            run_pipeline(cfg)
            return 0

    # Should not reach here
    return 1


# --------------------------------------------------------------------------- #
# Script execution guard
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    sys.exit(cli())
