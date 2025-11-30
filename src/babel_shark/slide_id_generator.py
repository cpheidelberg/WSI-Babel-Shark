#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slide_id_generator.py — Unified Slide ID builder & Excel registry writer

This script reads upstream Excel outputs (DataMatrix + Stain), merges them,
computes identifiers (CaseID, SlideID, DataMatrix), moves/renames actual WSI
files from a staging folder into a per-case folder structure, and writes:

    1) slide_metadata.xlsx  (sheet: full_merged)  — per-run registry view

CLI
---
Validate:
    PYTHONPATH=./src/babel_shark \
        python -u src/babel_shark/slide_id_generator.py validate \
        --config ./config/config.yaml

Run:
    PYTHONPATH=./src/babel_shark \
        python -u src/babel_shark/slide_id_generator.py run \
        --config ./config/config.yaml --log-level INFO

-----------------
staging_dir: "data/staging"
final_output_dir: "data/renamed"
failed_output_dir: "data/failed"
metadata_excel_path: "data/registry/slide_metadata.xlsx"
datamatrix_output_excel: "data/outputs/datamatrix_results.xlsx"
stain_output_excel:     "data/outputs/stain_results.xlsx"
dry_run: false
timestamp_tag_enabled: true   # if true, append _UTC<ScanTime_UTC> to SlideID-based filename
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml

# ----------------------------------------------------------------------
# OpenSlide DLL path (Windows)
# ----------------------------------------------------------------------
# IMPORTANT:
#   Adjust openslide_dll to the folder that contains libopenslide-0.dll, etc.

openslide = None
openslide_dll: Optional[str] = None


def _setup_openslide(dll_path: Optional[str]) -> None:
    """Initialize OpenSlide DLL directory (Windows) and import openslide."""
    global openslide, openslide_dll

    openslide_dll = dll_path

    if dll_path and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(dll_path)
        except Exception:
            # Non-fatal; if wrong path, OpenSlide import will fail later.
            pass

    try:
        import openslide as _osl  # type: ignore[import]
        openslide = _osl
    except Exception:  # noqa: BLE001
        openslide = None


# --------------------------- Logging ---------------------------

def setup_logging(level: str = "INFO") -> None:
    """Configure root logger (stdout, minimal format)."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


logger = logging.getLogger(__name__)

# --------------------------- Helpers ---------------------------

INVALID_FS_CHARS = '<>:"/\\|?*'
FS_TRANS_TABLE = str.maketrans({ch: "_" for ch in INVALID_FS_CHARS})


def sanitize_fs(name: str) -> str:
    """Sanitize string for filesystem (Windows-safe)."""
    if name is None:
        return ""
    out = str(name).translate(FS_TRANS_TABLE)
    return out.strip(" .")


def digits_only(x: object) -> str:
    return re.sub(r"[^\d]", "", str(x) if x is not None else "")


def to_int_str_no_decimal(x: object) -> Optional[str]:
    """Normalize number-like values to integer strings; handle '2453.0' and '2453,0'."""
    if x is None:
        return None
    s = str(x).strip()
    m = re.fullmatch(r"\s*([0-9]+)(?:[.,]0+)?\s*", s)
    if m:
        return str(int(m.group(1)))
    d = digits_only(s)
    return d if d else None


def to_year4_str(y: object) -> Optional[str]:
    s = to_int_str_no_decimal(y)
    if not s:
        return None
    if re.fullmatch(r"[0-9]{2}", s):
        return f"20{s}"
    if re.fullmatch(r"[0-9]{4}", s):
        return s
    return None


def to_case6_str(c: object) -> Optional[str]:
    s = to_int_str_no_decimal(c)
    if not s:
        return None
    return f"{int(s):06d}"


def normalize_stain(s: Optional[str]) -> str:
    if s is None:
        return "H&E"
    t_raw = str(s).strip()
    t = t_raw.upper().replace(" ", "")
    if t in {"HE", "H-E", "H& E", "H &E", "H & E", "H&E", "H&Ε"}:
        return "H&E"
    return t_raw.strip()


def build_datamatrix(lab, year, case, pot, blockid, section) -> Optional[str]:
    """Build DataMatrix per template: {LabID}{Year4}{Case6}S{Pot}-{BlockID}-{Section}"""
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
    if not candidates:
        return None
    priority = [".svs", ".ndpi", ".scn", ".mrxs", ".bif", ".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    cand_map = {p.suffix.lower(): p for p in candidates}
    for ext in priority:
        if ext in cand_map:
            return cand_map[ext]
    return candidates[0]


def unique_path(target: Path) -> Path:
    if not target.exists():
        return target
    i = 1
    stem, suf = target.stem, target.suffix
    while True:
        cand = target.with_name(f"{stem}_{i}{suf}")
        if not cand.exists():
            return cand
        i += 1


def read_excel_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path, dtype=str)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[WARN] Cannot read Excel: {path} -> {exc}")
        return pd.DataFrame()


def atomic_write_excel(dfs: Dict[str, pd.DataFrame], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with pd.ExcelWriter(tmp, engine="openpyxl") as xw:
        for name, df in dfs.items():
            (df if df is not None else pd.DataFrame()).to_excel(
                xw, sheet_name=name, index=False
            )
    os.replace(tmp, path)


# --------------------------- Timestamp helpers ---------------------------

def _utc_ts_iso_z() -> str:
    """UTC now in ISO-Z without microseconds, e.g., 2025-10-02T12:00:00Z"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _scan_ts_iso_z_from_metadata(path: Path) -> Optional[str]:
    """
    Extract REAL scan time from slide metadata.

    Rules:
        - Aperio (GT450):
            * aperio.Date       = "MM/DD/YYYY"
            * aperio.Time       = "HH:MM:SS"
            * aperio.Time Zone  = "GMT+0100" (or similar)
          We parse local time and convert to UTC.
        - Hamamatsu:
            * hamamatsu.Created = "YYYY/MM/DD"  (preferred)
            * or hamamatsu.Updated = "YYYY/MM/DD"
            * tiff.DateTime     = "YYYY:MM:DD HH:MM:SS"
          Here we treat the timestamp as already UTC (no extra offset logic),
          because timezone is usually not encoded explicitly.

        - NO FALLBACK:
            If metadata keys are missing or parsing fails, return None.
    """
    if openslide is None:
        logger.warning("[WARN] OpenSlide not available; cannot read ScanTime_UTC from metadata.")
        return None

    try:
        slide = openslide.OpenSlide(str(path))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[WARN] Cannot open slide for metadata: {path} -> {exc}")
        return None

    props = dict(slide.properties)
    vendor = props.get("openslide.vendor", "").lower().strip()

    # --- Aperio branch ---
    if vendor == "aperio":
        date_str = props.get("aperio.Date")
        time_str = props.get("aperio.Time")
        tz_str = props.get("aperio.Time Zone")
        if date_str and time_str and tz_str:
            try:
                dt_date = datetime.strptime(date_str, "%m/%d/%Y")
                dt_time = datetime.strptime(time_str, "%H:%M:%S").time()
                # tz_str like "GMT+0100"
                offset = timezone.utc
                m = re.match(r"GMT([+-])(\d{2})(\d{2})", tz_str)
                if m:
                    sign = 1 if m.group(1) == "+" else -1
                    hours = int(m.group(2))
                    minutes = int(m.group(3))
                    delta = timedelta(hours=hours, minutes=minutes)
                    offset = timezone(sign * delta)
                local_dt = datetime(
                    year=dt_date.year,
                    month=dt_date.month,
                    day=dt_date.day,
                    hour=dt_time.hour,
                    minute=dt_time.minute,
                    second=dt_time.second,
                    tzinfo=offset,
                )
                utc_dt = local_dt.astimezone(timezone.utc)
                return utc_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[WARN] Failed to parse Aperio scan time: {exc}")

    # --- Hamamatsu branch ---
    # Prefer hamamatsu.Created / Updated for date, tiff.DateTime for time if available
    if vendor == "hamamatsu" or True:
        date_hama = props.get("hamamatsu.Created") or props.get("hamamatsu.Updated")
        dt_str = props.get("tiff.DateTime")
        # tiff.DateTime example: "2025:08:25 12:18:29"
        if date_hama and dt_str:
            try:
                # we ignore date_hama and use full tiff.DateTime (contains date+time)
                dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                # Treat as UTC directly (scanner time already in UTC)
                dt = dt.replace(tzinfo=timezone.utc)
                return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[WARN] Failed to parse Hamamatsu tiff.DateTime: {exc}")

        if date_hama and not dt_str:
            # Only a date, no time; treat as 00:00:00 UTC
            try:
                dt = datetime.strptime(date_hama, "%Y/%m/%d")
                dt = dt.replace(tzinfo=timezone.utc)
                return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[WARN] Failed to parse Hamamatsu Created/Updated: {exc}")

    # If we reach here, no readable scan time
    return None


# --------------------------- Config ---------------------------

@dataclass(frozen=True)
class ConfigPaths:
    source_dir: Path
    renamed_dir: Path
    failed_dir: Path
    dm_excel: Path
    stain_excel: Path
    meta_xlsx: Path
    dry_run: bool
    ts_enabled: bool


def load_config(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def materialize_paths(cfg: Dict) -> ConfigPaths:
    source_dir = Path(cfg["staging_dir"])
    renamed_dir = Path(cfg["final_output_dir"])
    failed_dir = Path(cfg["failed_output_dir"])
    for d in (renamed_dir, failed_dir):
        d.mkdir(parents=True, exist_ok=True)

    dm_excel = Path(cfg.get("datamatrix_output_excel", "")) if cfg.get("datamatrix_output_excel") else Path()
    stain_excel = Path(cfg.get("stain_output_excel", "")) if cfg.get("stain_output_excel") else Path()

    meta_xlsx = Path(cfg.get("metadata_excel_path", "slide_metadata.xlsx"))
    meta_xlsx.parent.mkdir(parents=True, exist_ok=True)

    dry_run = bool(cfg.get("dry_run", False))
    ts_enabled = bool(cfg.get("timestamp_tag_enabled", False))

    return ConfigPaths(
        source_dir=source_dir,
        renamed_dir=renamed_dir,
        failed_dir=failed_dir,
        dm_excel=dm_excel,
        stain_excel=stain_excel,
        meta_xlsx=meta_xlsx,
        dry_run=dry_run,
        ts_enabled=ts_enabled,
    )


# --------------------------- Pipeline ---------------------------

def merge_inputs(dm_path: Path, stain_path: Path) -> pd.DataFrame:
    """
    Merge upstream Excel files on 'FileName'; keep ALL DM columns,
    but from Stain input keep ONLY the final 'Stain' column.
    """
    df_dm = read_excel_safe(dm_path) if dm_path and dm_path.exists() else pd.DataFrame()
    df_st = read_excel_safe(stain_path) if stain_path and stain_path.exists() else pd.DataFrame()

    # --- Ensure DM skeleton exists & required DM keys (we keep ALL DM columns as-is) ---
    for c in ["FileName", "DataMatrix", "LabID", "Year", "CaseNumber", "Pot", "BlockID", "Section", "ExtractionMethod"]:
        if c not in df_dm.columns:
            df_dm[c] = None

    if "FileName" not in df_dm.columns:
        # If DM is missing but Stain has FileName, bootstrap skeleton to keep pipeline flowing
        if not df_st.empty and "FileName" in df_st.columns:
            df_dm = df_st[["FileName"]].drop_duplicates().copy()
            for c in ["DataMatrix", "LabID", "Year", "CaseNumber", "Pot", "BlockID", "Section", "ExtractionMethod"]:
                df_dm[c] = None
        else:
            df_dm["FileName"] = []

    # --- From Stain: keep ONLY the final Stain per FileName ---
    if not df_st.empty and "FileName" in df_st.columns:
        stain_col = "Stain" if "Stain" in df_st.columns else None
        if stain_col is None:
            df_st_min = pd.DataFrame(columns=["FileName", "Stain"])
        else:
            df_st_min = df_st[["FileName", stain_col]].rename(columns={stain_col: "Stain"})
            df_st_min = df_st_min.dropna(subset=["Stain"])
            df_st_min = df_st_min.drop_duplicates(subset=["FileName"], keep="last")
    else:
        df_st_min = pd.DataFrame(columns=["FileName", "Stain"])

    # LEFT merge: DM (all cols) + ONLY Stain
    df = pd.merge(df_dm, df_st_min, on="FileName", how="left")
    return df


def compute_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CaseID, SlideID; rebuild DataMatrix if needed; normalize stain."""
    df = df.copy()

    df["LabID"] = df["LabID"].astype(str).str.strip().str.upper()
    df["Year4"] = df["Year"].apply(to_year4_str).astype("string")
    df["Case6"] = df["CaseNumber"].apply(to_case6_str).astype("string")
    df["Pot"] = df["Pot"].fillna("A").astype(str).str.strip().str.upper().astype("string")
    df["BlockID"] = df["BlockID"].apply(lambda x: to_int_str_no_decimal(x) or "1").astype("string")
    df["Section"] = df["Section"].apply(lambda x: to_int_str_no_decimal(x) or "1").astype("string")

    # Stain normalization
    df["Stain"] = df["Stain"].apply(normalize_stain).astype("string")

    # Rebuild DataMatrix if missing/invalid
    df["DataMatrix_v2"] = df.apply(
        lambda r: build_datamatrix(r["LabID"], r["Year4"], r["Case6"], r["Pot"], r["BlockID"], r["Section"]),
        axis=1,
    )
    df["DataMatrix"] = df["DataMatrix_v2"].where(
        df["DataMatrix_v2"].notna() & (df["DataMatrix_v2"] != ""), df["DataMatrix"]
    )
    df.drop(columns=["DataMatrix_v2"], inplace=True)

    df["CaseID"] = (df["LabID"].fillna("") + df["Year4"].fillna("") + df["Case6"].fillna("")).astype("string")
    df["Stain"] = df["Stain"].fillna("H&E")

    # IMPORTANT: do NOT generate "nan-H&E" SlideIDs.
    # Only create SlideID when DataMatrix is present and not 'nan'.
    dm_series = df["DataMatrix"]
    dm_str = dm_series.astype(str)
    valid_mask = dm_str.str.strip().ne("") & dm_str.str.strip().str.lower().ne("nan")
    df["SlideID"] = ""
    df.loc[valid_mask, "SlideID"] = (
        dm_str[valid_mask].astype(str) + "-" + df["Stain"][valid_mask].astype(str)
    )

    # Robust base-name extraction for ANY extension (svs, ndpi, tif, ...)
    df["OriginalBase"] = df["FileName"].apply(lambda s: Path(str(s)).stem)
    df["method of extraction"] = df.get("ExtractionMethod", "").astype(str).str.strip().apply(
        lambda em: "ROIbase" if em.lower() == "roibase" else "Datamatrix"
    )
    return df


def pick_source_and_dest(
    base: str,
    slide_id: str,
    case_id: str,
    source_dir: Path,
    renamed_dir: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Find source by base.* and destination path under renamed/case_id/slide_id.ext"""
    base = Path(str(base)).stem
    matches = list(source_dir.glob(base + ".*"))
    if not matches:
        return None, None
    src_file = prefer_ext(matches) or matches[0]

    dst_folder = renamed_dir / str(case_id)
    dst_folder.mkdir(parents=True, exist_ok=True)
    dst_file = unique_path(dst_folder / f"{slide_id}{src_file.suffix}")
    return src_file, dst_file


def run_pipeline(cfg: Dict) -> None:
    """
    SlideID generator pipeline.

    - Reads merged DataMatrix + stain Excel
    - Computes identifiers (CaseID, SlideID, etc.)
    - Renames/moves WSIs into final tree
    - Produces:
        * Excel: slide_metadata.xlsx (sheet: full_merged) for THIS run only
          enriched with ScanTime_UTC, AnalysisTime_UTC, InputPath, OutputPath, status.
    """
    paths = materialize_paths(cfg)

    # 2) Merge DataMatrix and Stain inputs, then compute identifiers
    df_merge = merge_inputs(paths.dm_excel, paths.stain_excel)
    if df_merge.empty:
        logger.warning("[WARN] Merged inputs are empty; nothing to process.")
    df = compute_identifiers(df_merge).fillna("")

    # Analysis time (one per run)
    analysis_ts = _utc_ts_iso_z()
    runtime_value = str(cfg.get("run_timestamp", analysis_ts))

    moved: List[str] = []
    records: List[Dict[str, str]] = []

    # 3) File renaming loop
    for _, row in df.iterrows():
        original_base = str(row.get("OriginalBase", ""))
        slide_id_raw = str(row.get("SlideID") or "").strip()
        case_id = str(row.get("CaseID") or "").strip()

        # Determine source file by base.* regardless of SlideID validity
        matches = list(paths.source_dir.glob(Path(original_base).stem + ".*"))
        if not matches:
            logger.info(f"[MISS] No WSI found for base: {original_base}")
            continue

        src_file = prefer_ext(matches) or matches[0]
        input_path = str(src_file)

        # Extract ScanTime_UTC from metadata (no fallback)
        scan_ts = _scan_ts_iso_z_from_metadata(src_file)
        scan_ts_str = scan_ts or ""

        # If SlideID is missing/invalid, send to failed/ without renaming to 'nan_H&E'
        slideid_valid = bool(slide_id_raw) and not slide_id_raw.lower().startswith("nan-")

        if not slideid_valid or not case_id:
            # Move to failed/
            status = "failed"
            if not paths.dry_run:
                dst_failed = unique_path(paths.failed_dir / src_file.name)
                try:
                    shutil.move(str(src_file), str(dst_failed))
                    moved.append(src_file.name)
                    logger.info(f"[FAILED] {src_file.name} -> {dst_failed.name}")
                    output_path = str(dst_failed)
                    new_name = dst_failed.name
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"[WARN] Could not move failed file {src_file}: {exc}")
                    output_path = ""
                    new_name = ""
            else:
                logger.info(f"[DRYRUN-FAILED] Would move {src_file.name} -> failed/")
                output_path = ""
                new_name = ""

            records.append(
                {
                    "FileName": str(row.get("FileName", "")),
                    "SlideID": slide_id_raw,
                    "CaseID": case_id,
                    "OriginalFileName": src_file.name,
                    "NewFileName": new_name,
                    "InputPath": input_path,
                    "OutputPath": output_path,
                    "ScanTime_UTC": scan_ts_str,
                    "AnalysisTime_UTC": analysis_ts,
                    "method of extraction": str(row.get("method of extraction", "metadata")),
                    "status": status,
                }
            )
            continue

        # SlideID is valid: build final SlideID used for renaming + registry
        ts_suffix = ""
        if paths.ts_enabled and scan_ts_str:
            ts_suffix = f"_UTC{scan_ts_str}"

        effective_slide_id = sanitize_fs(slide_id_raw + ts_suffix)

        # Determine destination path under CaseID tree
        src_file2, dst_file = pick_source_and_dest(
            base=original_base,
            slide_id=effective_slide_id,
            case_id=case_id,
            source_dir=paths.source_dir,
            renamed_dir=paths.renamed_dir,
        )

        # Defensive: src_file2 should equal src_file, but use it if available
        if src_file2 is not None:
            src_file = src_file2

        if src_file is None or dst_file is None:
            logger.info(f"[MISS] No WSI found for base (rename step): {original_base}")
            status = "failed"
            output_path = ""
            new_name = ""
        else:
            if not paths.dry_run:
                try:
                    shutil.move(str(src_file), str(dst_file))
                    moved.append(src_file.name)
                    logger.info(f"[OK] {src_file.name} -> {dst_file.name}")
                    status = "success"
                    output_path = str(dst_file)
                    new_name = dst_file.name
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"[WARN] Could not move {src_file} -> {dst_file}: {exc}")
                    status = "failed"
                    output_path = ""
                    new_name = ""
            else:
                logger.info(f"[DRYRUN] Would move {src_file.name} -> {dst_file.name}")
                status = "success"
                output_path = str(dst_file)
                new_name = dst_file.name

        records.append(
            {
                "FileName": str(row.get("FileName", "")),
                "SlideID": effective_slide_id,
                "CaseID": case_id,
                "OriginalFileName": src_file.name if src_file else "",
                "NewFileName": new_name,
                "InputPath": input_path,
                "OutputPath": output_path,
                "ScanTime_UTC": scan_ts_str,
                "AnalysisTime_UTC": analysis_ts,
                "method of extraction": str(row.get("method of extraction", "metadata")),
                "status": status,
            }
        )

    # 4) Move unmatched source files into failed/ (safety net)
    if not paths.dry_run:
        for file in paths.source_dir.glob("*"):
            if file.is_file() and (file.name not in moved):
                try:
                    shutil.move(str(file), str(paths.failed_dir / file.name))
                    logger.info(f"[INFO] -> failed/: {file.name}")
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"[WARN] Could not move unmatched {file.name}: {exc}")
    else:
        logger.info("[DRYRUN] Skip moving unmatched files to failed/.")

    # 5) Build per-run full_merged DataFrame (Excel view)
    try:
        df_records = pd.DataFrame.from_records(records)
    except Exception:  # noqa: BLE001
        df_records = pd.DataFrame()

    try:
        df_excel = df.copy()

        # Merge record-level info (NewFileName, paths, times, status) by FileName
        if not df_records.empty and "FileName" in df_excel.columns:
            cols_for_merge = [
                "FileName",
                "OriginalFileName",
                "NewFileName",
                "InputPath",
                "OutputPath",
                "ScanTime_UTC",
                "AnalysisTime_UTC",
                "status",
            ]
            cols_for_merge = [c for c in cols_for_merge if c in df_records.columns]
            df_excel = df_excel.merge(
                df_records[cols_for_merge],
                on="FileName",
                how="left",
            )

        # Add RunTime as a constant column for this run
        df_excel["RunTime"] = runtime_value

        # Drop columns that must NOT appear in full_merged
        drop_cols = ["Year4", "Case6", "ExtractionMethod"]
        for col in drop_cols:
            if col in df_excel.columns:
                df_excel = df_excel.drop(columns=[col])

        df_excel = df_excel.fillna("")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[WARN] Could not build full_merged DataFrame: {exc}")
        df_excel = df.copy().fillna("")

    # 6) Write Excel (per-run, single sheet: full_merged)
    try:
        dfs = {"full_merged": df_excel}
        atomic_write_excel(dfs, paths.meta_xlsx)
        logger.info(f"[OK] Excel (full_merged) saved: {paths.meta_xlsx}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[WARN] Could not write Excel: {exc}")

    logger.info("[DONE] slide_id_generator finished.")


# --------------------------- CLI ---------------------------

def cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Slide ID generator & Excel writer")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="Run pipeline with config")
    p_run.add_argument("--config", type=str, required=True, help="Path to config YAML")
    p_run.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    p_val = sub.add_parser("validate", help="Validate config and paths")
    p_val.add_argument("--config", type=str, required=True, help="Path to config YAML")
    p_val.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    sub.add_parser("version", help="Show version")

    args = parser.parse_args(argv)

    if args.command == "version":
        setup_logging("INFO")
        logger.info("slide_id_generator 2.0.0")
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

        # Initialize OpenSlide from config (dll_paths.openslide_dll)
        dll_cfg = (cfg.get("dll_paths") or {})
        dll_path = dll_cfg.get("openslide_dll")
        _setup_openslide(dll_path)

        required = ["staging_dir", "final_output_dir", "failed_output_dir"]
        missing = [k for k in required if k not in cfg]
        if missing:
            logger.error(f"[ERROR] Missing config keys: {', '.join(missing)}")
            return 2

        try:
            _ = materialize_paths(cfg)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[ERROR] Invalid path values: {exc}")
            return 2

        if args.command == "validate":
            logger.info("[OK] Config looks valid.")
            return 0

        if args.command == "run":
            run_pipeline(cfg)
            return 0

    return 1


if __name__ == "__main__":
    sys.exit(cli())
