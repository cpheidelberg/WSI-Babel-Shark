#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runner.py — Babel-Shark Pipeline Orchestrator

Runs the complete Babel-Shark pipeline, connecting all modules for
automated slide metadata extraction from raw WSIs to the final
SQLite database.

Main features
- Executes all stages sequentially:
    1. collect_slides (copy/move to staging)
    2. label_extractor
    3. datamatrix_reader
    4. roi_metadata_extractor
    5. stain_extractor
    6. slide_id_generator
    7. database_manager (sqlite_table_generator)
- Manages I/O paths and timestamps from a single YAML config.
- Tracks run progress and errors through unified logging.
- Supports watch mode (periodic polling of watch_dir).

Usage
-----
    PYTHONPATH=./src/babel_shark python -u src/babel_shark/runner.py run \
        --config ./config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import yaml
from tqdm import tqdm

__module_prog__ = "WSI-Babel-Shark"

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a simple, consistent format."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _attach_file_logger(log_file: Path) -> None:
    """Attach a FileHandler to the root logger (idempotent for a given path)."""
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(getattr(handler, "baseFilename", "")).resolve() == log_file.resolve():
                    return
            except Exception:
                pass

    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(root.level)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(fh)
    logging.info(f"[LOG] File logging enabled: {log_file}")


# --------------------------------------------------------------------------------------
# Atomic I/O Utilities
# --------------------------------------------------------------------------------------


def _atomic_write_bytes(dst: Path, data: bytes) -> None:
    """Write bytes atomically to dst using a temporary file and os.replace."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dst.parent, delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, dst)


def _atomic_write_text(dst: Path, text: str, encoding: str = "utf-8") -> None:
    """Write text atomically to dst."""
    _atomic_write_bytes(dst, text.encode(encoding))


def _atomic_write_yaml(dst: Path, obj: Dict[str, Any]) -> None:
    """Dump YAML atomically to dst."""
    text = yaml.dump(obj, allow_unicode=True)
    _atomic_write_text(dst, text, encoding="utf-8")


def _atomic_write_excel_from_df(dst: Path, df: pd.DataFrame) -> None:
    """Write DataFrame to a single-sheet Excel atomically."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dst.parent, delete=False, suffix=".xlsx") as tmp:
        with pd.ExcelWriter(tmp.name) as writer:
            df.to_excel(writer, index=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, dst)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _default_wsi_exts() -> Tuple[str, ...]:
    """Fallback list of allowed WSI extensions (plus common image types)."""
    return (".svs", ".ndpi", ".tif", ".tiff", ".scn", ".mrxs", ".bif", ".png", ".jpg", ".jpeg")


def safe_table(name: str) -> str:
    """Validate/whitelist a table name to avoid SQL injection via config."""
    name = str(name)
    m = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
    if not m:
        raise ValueError(f"Invalid table name: {name!r}")
    return name


def clear_folder(folder: Path) -> None:
    """Delete all contents of a folder (files + subfolders)."""
    if folder.exists():
        for item in folder.iterdir():
            if item.is_file():
                try:
                    item.unlink(missing_ok=True)
                except Exception:
                    pass
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)


def _list_regular_files(folder: Path) -> List[Path]:
    """List regular files under a folder or return empty list if not exists."""
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.is_file()]


# --------------------------------------------------------------------------------------
# DB Layer
# --------------------------------------------------------------------------------------


def ensure_db(db_path: Path) -> sqlite3.Connection:
    """Open/create SQLite DB with sane PRAGMAs for stability/concurrency."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def ensure_registry_schema(conn: sqlite3.Connection, table: str) -> None:
    """
    Create slide registry schema and indexes if missing.

    NOTE:
    - This schema is aligned conceptually with database_manager/sqlite_table_generator
      but extended to support:
        * scan_time_utc       (real scan time in UTC, from slide metadata)
        * analysis_time_utc   (timestamp of Babel-Shark analysis)
        * input_path          (root of input WSIs)
        * output_path         (root of renamed WSIs)
    - Population of these fields is done downstream (not in runner itself).
    """
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id               TEXT NOT NULL,
            file_name            TEXT NOT NULL,                -- label image name
            slide_id             TEXT,
            case_id              TEXT,
            original_file_name   TEXT,
            new_file_name        TEXT,                         -- UNIQUE across registry
            datamatrix           TEXT,
            labid                TEXT,
            year4                TEXT,
            case6                TEXT,
            pot                  TEXT,
            blockid              TEXT,
            section              TEXT,
            stain                TEXT,
            method_of_extraction TEXT,
            scan_time_utc        TEXT,                         -- REAL scan time (UTC) from metadata
            analysis_time_utc    TEXT,                         -- Babel-Shark run timestamp (UTC/naive ISO)
            input_path           TEXT,                         -- root input folder (watch_dir)
            output_path          TEXT,                         -- root final_output_dir
            status               TEXT,                         -- success / failed
            inserted_at          TEXT DEFAULT (datetime('now'))
        );
        """
    )
    cur.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS ux_{table}_newfile ON {table}(new_file_name);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS ix_{table}_orig ON {table}(original_file_name);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS ix_{table}_datamatrix ON {table}(datamatrix);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS ix_{table}_lab_case ON {table}(labid, case6);")
    conn.commit()


def fetch_seen_original_names_from_db(cfg: Dict[str, Any]) -> Set[str]:
    """Return case-insensitive set of already-seen original file names from DB."""
    db_path = Path(cfg.get("sqlite_db_path", "registry.db"))
    table = safe_table(cfg.get("ops_table", "slide_registry"))

    conn = ensure_db(db_path)
    ensure_registry_schema(conn, table)

    try:
        df = pd.read_sql_query(
            f"SELECT original_file_name AS name FROM {table} WHERE original_file_name IS NOT NULL;",
            conn,
        )
        names = set(df["name"].astype(str).str.strip().str.lower())
        logging.info(f"[INFO] Loaded {len(names)} seen original names from DB.")
        return names
    except Exception as exc:
        logging.info(f"[WARN] DB duplicate-check query failed: {exc}")
        return set()
    finally:
        conn.close()


# --------------------------------------------------------------------------------------
# File System: COPY/MOVE Stage (marking deferred)
# --------------------------------------------------------------------------------------


def copy_new_slides(
    cfg: Dict[str, Any],
    watch_dir: Path,
    staging_dir: Path,
    failed_dir: Path,
    run_id: str,  # kept for future auditing if needed
) -> List[str]:
    """Copy/move new slides from watch folder into staging based on cfg['operation_mode'].

    Notes:
        - This function does NOT write CSV and does NOT touch the DB.
        - It returns the list of original file names for this run.
        - Duplicate detection relies ONLY on the SQLite registry (original_file_name).
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    operation_mode = str(cfg.get("operation_mode", "copy")).lower().strip()
    if operation_mode not in ("copy", "move"):
        logging.info(f"[WARN] Unknown operation_mode='{operation_mode}', defaulting to 'copy'.")
        operation_mode = "copy"
    logging.info(f"[INFO] operation_mode = {operation_mode}")

    cfg_exts = cfg.get("wsi_types", [])
    valid_exts = tuple(cfg_exts) if cfg_exts else _default_wsi_exts()
    logging.info(f"[DEBUG] watch_dir = {watch_dir}")
    logging.info(f"[DEBUG] valid extensions = {valid_exts}")

    if not watch_dir.exists():
        logging.info(f"[ERROR] Watched folder does not exist: {watch_dir}")
        return []

    seen_names = fetch_seen_original_names_from_db(cfg)

    candidates: List[Path] = []
    for root, _, files in os.walk(watch_dir):
        for file in files:
            candidates.append(Path(root) / file)
    logging.info(f"[DEBUG] Found {len(candidates)} file(s) under watch folder.")

    new_files: List[Tuple[Path, str]] = []
    skipped_ext = skipped_seen = skipped_exists_dest = 0

    for src in candidates:
        name = src.name
        if not name.lower().endswith(valid_exts):
            skipped_ext += 1
            continue
        if name.lower() in seen_names:
            skipped_seen += 1
            continue
        dest = staging_dir / name
        if dest.exists():
            skipped_exists_dest += 1
            continue
        new_files.append((src, name))

    logging.info(f"[DEBUG] Candidates with valid ext: {len(candidates) - skipped_ext}")
    logging.info(f"[DEBUG] Skipped (exists in DB): {skipped_seen}")
    logging.info(f"[DEBUG] Skipped (already in staging): {skipped_exists_dest}")
    logging.info(f"[DEBUG] To process: {len(new_files)}")

    if not new_files:
        return []

    action_label = "Moving new slides" if operation_mode == "move" else "Copying new slides"
    for src, file in tqdm(new_files, desc=action_label):
        dest = staging_dir / file
        try:
            if operation_mode == "move":
                shutil.move(str(src), str(dest))
                logging.info(f"[Moved] {src} -> {dest}")
            else:
                shutil.copy2(str(src), str(dest))
                logging.info(f"[Copied] {src} -> {dest}")
        except Exception as exc:
            logging.info(f"[ERROR] {('Move' if operation_mode=='move' else 'Copy')} failed: {file} -> {exc}")
            try:
                failed_dest = failed_dir / file
                if operation_mode == "move":
                    if src.exists():
                        shutil.move(str(src), str(failed_dest))
                    else:
                        if dest.exists():
                            shutil.move(str(dest), str(failed_dest))
                else:
                    if src.exists():
                        shutil.copy2(str(src), str(failed_dest))
                    elif dest.exists():
                        shutil.copy2(str(dest), str(failed_dest))
                logging.info(f"[INFO] Placed problematic file in failed folder: {failed_dest}")
            except Exception as exc2:
                logging.info(f"[ERROR] Also failed to place in failed folder: {file} -> {exc2}")

    return [name for _, name in new_files]


# --------------------------------------------------------------------------------------
# Child Script Runner
# --------------------------------------------------------------------------------------


def run_script(
    step_title: str,
    script_path: str,
    temp_config_path: Path,
    extra_args: Optional[Sequence[str]] = None,
    log_level: str = "INFO",
    supports_log_level: bool = True,
) -> bool:
    """Run a child script with the temp config and stream logs into main log.

    Strategy:
        1) Try subcommand style with --log-level (if requested).
        2) If stderr shows 'unrecognized arguments: --log-level', retry without it.
        3) Propagate stdout/stderr to main logs.
    """
    logging.info(f"\n[STEP] Running {step_title} ...")

    base_cmd = ["python", script_path, "run", "--config", str(temp_config_path)]
    if extra_args:
        base_cmd.extend(extra_args)

    # Attempt 1: with --log-level
    cmd = list(base_cmd)
    if supports_log_level and log_level:
        cmd += ["--log-level", log_level]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        logging.info(result.stdout)
    if result.returncode == 0:
        return True

    stderr_text = (result.stderr or "").strip()
    if "--log-level" in " ".join(cmd) and "unrecognized arguments: --log-level" in stderr_text:
        logging.info(f"[WARN] {step_title}: child script does not accept --log-level; retrying without it.")
        cmd_no_loglevel = list(base_cmd)
        result2 = subprocess.run(cmd_no_loglevel, capture_output=True, text=True)
        if result2.stdout:
            logging.info(result2.stdout)
        if result2.returncode == 0:
            return True
        logging.info(f"[ERROR] {step_title} failed (no --log-level):\n{result2.stderr}")
        return False

    logging.info(f"[ERROR] {step_title} failed:\n{stderr_text}")
    return False


# --------------------------------------------------------------------------------------
# ROI Merge & Fallback Helpers
# --------------------------------------------------------------------------------------


def _ensure_extraction_method_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'ExtractionMethod' exists and is consistent with populated DataMatrix."""
    if df is None or df.empty:
        return df
    if "ExtractionMethod" not in df.columns:
        df["ExtractionMethod"] = None
    if "DataMatrix" in df.columns:
        mask = (
            df["DataMatrix"].notna()
            & (df["DataMatrix"].astype(str).str.strip() != "")
            & df["ExtractionMethod"].isna()
        )
        df.loc[mask, "ExtractionMethod"] = "DataMatrix"
    return df


def _merge_roi_into_datamatrix(dm_excel: Path, roi_csv: Path) -> None:
    """
    Merge ROI fallback results into the main DataMatrix Excel with FULL metadata merging.

    - Fills DataMatrix from ROI where missing in the DM Excel.
    - Fills LabID / Year / CaseNumber / Pot / BlockID / Section only if missing in DM Excel.
    - Cleans ROI values: '', NaN, 'nan', 'none', 'null', 'NA', '[]' -> treated as missing.
    - Marks rows updated from ROI with ExtractionMethod = 'ROIbase'.
    - Explicitly casts involved columns to string dtype to avoid pandas FutureWarning
      about incompatible dtypes on assignment.
    """

    # ---------------------------------------------------------------
    # 1) Load Excel / CSV safely
    # ---------------------------------------------------------------
    if not dm_excel.exists() or not roi_csv.exists():
        return

    try:
        # Do not rely only on dtype=str; we will explicitly cast later
        df_dm = pd.read_excel(dm_excel)
    except Exception as exc:
        logging.info(f"[WARN] Cannot read DataMatrix Excel: {dm_excel} -> {exc}")
        return

    try:
        df_roi = pd.read_csv(roi_csv)
    except Exception as exc:
        logging.info(f"[WARN] Cannot read ROI CSV: {roi_csv} -> {exc}")
        return

    # Ensure FileName exists in both
    if "FileName" not in df_dm.columns or "FileName" not in df_roi.columns:
        logging.info("[WARN] Missing FileName column; cannot merge ROI.")
        return

    # ---------------------------------------------------------------
    # 2) Normalize missing ROI columns
    # ---------------------------------------------------------------
    roi_meta_cols = ["DataMatrix", "LabID", "Year", "CaseNumber", "Pot", "BlockID", "Section"]
    for col in roi_meta_cols:
        if col not in df_roi.columns:
            df_roi[col] = None

    # ---------------------------------------------------------------
    # 3) Ensure ExtractionMethod exists (no crash on .astype)
    # ---------------------------------------------------------------
    # Use the helper if available in this module
    try:
        df_dm = _ensure_extraction_method_col(df_dm)
    except NameError:
        # Fallback: minimal behavior if helper is not present
        if "ExtractionMethod" not in df_dm.columns:
            df_dm["ExtractionMethod"] = None

    # For ROI we do not strictly need ExtractionMethod, but keep it consistent
    if "ExtractionMethod" not in df_roi.columns:
        df_roi["ExtractionMethod"] = None

    # ---------------------------------------------------------------
    # 4) Clean ROI values — treat empty-like tokens as missing
    # ---------------------------------------------------------------
    def _clean_roi_value(x: object) -> Optional[str]:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        s = str(x).strip()
        if s == "" or s.upper() in {"NAN", "NONE", "NULL", "NA", "[]"}:
            return None
        return s

    for col in roi_meta_cols:
        df_roi[col] = df_roi[col].apply(_clean_roi_value)

    # ---------------------------------------------------------------
    # 5) Prepare merge subset and merge ROI into DM
    # ---------------------------------------------------------------
    merge_cols = ["FileName"] + roi_meta_cols
    merge_subset = df_roi[merge_cols].copy()

    df_dm = df_dm.merge(
        merge_subset,
        on="FileName",
        how="left",
        suffixes=("", "_roi"),
    )

    # ---------------------------------------------------------------
    # 6) Cast involved metadata columns to string dtype explicitly
    #    to avoid FutureWarning on assignment
    # ---------------------------------------------------------------
    all_cols_to_cast = ["DataMatrix", "LabID", "Year", "CaseNumber", "Pot", "BlockID", "Section"]

    for col in all_cols_to_cast:
        # Base column in DM
        if col in df_dm.columns:
            df_dm[col] = df_dm[col].astype("string")
        else:
            # Create empty string column if missing
            df_dm[col] = pd.Series([""] * len(df_dm), dtype="string")

        # ROI column after merge
        roi_col = f"{col}_roi"
        if roi_col in df_dm.columns:
            df_dm[roi_col] = df_dm[roi_col].astype("string")

    # Ensure ExtractionMethod is string-type as well
    if "ExtractionMethod" in df_dm.columns:
        df_dm["ExtractionMethod"] = df_dm["ExtractionMethod"].astype("string")
    else:
        df_dm["ExtractionMethod"] = pd.Series([""] * len(df_dm), dtype="string")

    # ---------------------------------------------------------------
    # 7) Fill DataMatrix when empty in DM and non-empty in ROI
    # ---------------------------------------------------------------
    has_dm_roi = "DataMatrix_roi" in df_dm.columns

    if has_dm_roi:
        mask_dm_update = (
            df_dm["DataMatrix"].isna()
            | (df_dm["DataMatrix"].astype(str).str.strip() == "")
        ) & df_dm["DataMatrix_roi"].notna()

        df_dm.loc[mask_dm_update, "DataMatrix"] = df_dm.loc[mask_dm_update, "DataMatrix_roi"]
    else:
        mask_dm_update = pd.Series([False] * len(df_dm), index=df_dm.index)

    # ---------------------------------------------------------------
    # 8) Fill other metadata fields (LabID, Year, CaseNumber, Pot, BlockID, Section)
    # ---------------------------------------------------------------
    meta_update_mask = mask_dm_update.copy()

    for field in ["LabID", "Year", "CaseNumber", "Pot", "BlockID", "Section"]:
        roi_col = f"{field}_roi"
        if roi_col not in df_dm.columns:
            continue

        # Field is considered empty if NaN or empty string after strip
        mask_field = (
            df_dm[field].isna()
            | (df_dm[field].astype(str).str.strip() == "")
        ) & df_dm[roi_col].notna()

        df_dm.loc[mask_field, field] = df_dm.loc[mask_field, roi_col]
        meta_update_mask |= mask_field

    # ---------------------------------------------------------------
    # 9) Mark rows updated by ROI as ROIbase
    # ---------------------------------------------------------------
    df_dm.loc[meta_update_mask, "ExtractionMethod"] = "ROIbase"

    # ---------------------------------------------------------------
    # 10) Drop temporary *_roi columns
    # ---------------------------------------------------------------
    drop_cols = [c for c in df_dm.columns if c.endswith("_roi")]
    df_dm = df_dm.drop(columns=drop_cols, errors="ignore")

    # ---------------------------------------------------------------
    # 11) Write back to Excel
    # ---------------------------------------------------------------
    _atomic_write_excel_from_df(dm_excel, df_dm)
    logging.info(f"[OK] ROI metadata merged into DataMatrix Excel: {dm_excel}")


def _move_unresolved_to_fallback(
    datamatrix_failed_dir: Path,
    roi_csv: Path,
    fallback_failed_dir: Path,
) -> None:
    """
    Move files that remain unresolved after ROI fallback into a separate 'fallback_failed' folder.
    """
    fallback_failed_dir.mkdir(parents=True, exist_ok=True)

    try:
        df_roi = pd.read_csv(roi_csv)
    except Exception as exc:
        logging.info(f"[WARN] Cannot read ROI CSV for fallback move: {roi_csv} -> {exc}")
        return

    unresolved: Set[str] = set()
    if "FileName" in df_roi.columns and "DataMatrix" in df_roi.columns:
        unresolved = set(
            df_roi.loc[
                df_roi["DataMatrix"].isna()
                | (df_roi["DataMatrix"].astype(str).str.strip() == ""),
                "FileName",
            ].astype(str)
        )
    else:
        logging.info("[WARN] ROI CSV missing 'FileName' or 'DataMatrix'; cannot determine unresolved.")
        return

    if not datamatrix_failed_dir.exists():
        return

    for file in datamatrix_failed_dir.iterdir():
        if not file.is_file():
            continue
        if file.name in unresolved:
            dest = fallback_failed_dir / file.name
            try:
                shutil.move(str(file), str(dest))
                logging.info(f"[INFO] Moved unresolved ROI file to fallback_failed: {dest}")
            except Exception as exc:
                logging.info(f"[WARN] Could not move unresolved file {file} -> {exc}")


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config from disk."""
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    return cfg


def _prepare_runtime_paths(cfg: Dict[str, Any], timestamp: str) -> Dict[str, Path]:
    """Prepare all paths derived from config and timestamp."""
    date_str = timestamp.split("_")[0].replace("-", ".")
    output_run_dir = Path(cfg["run_output_dir"]) / timestamp
    label_crops_dir = Path(cfg["label_root_dir"]) / date_str / timestamp
    datamatrix_output_excel = output_run_dir / f"datamatrix_results_{timestamp}.xlsx"
    stain_output_excel = output_run_dir / f"stain_results_{timestamp}.xlsx"
    stain_output_pdf = output_run_dir / f"stain_report_{timestamp}.pdf"

    paths: Dict[str, Path] = {
        "output_run_dir": output_run_dir,
        "label_crops_dir": label_crops_dir,
        "datamatrix_output_excel": datamatrix_output_excel,
        "stain_output_excel": stain_output_excel,
        "stain_output_pdf": stain_output_pdf,
        "allslides_dir": Path(cfg["staging_dir"]),
        "failed_output_dir": Path(cfg["failed_output_dir"]),
        "datamatrix_failed_dir": output_run_dir / "failed_datamatrix",
        "fallback_failed_dir": output_run_dir / "failed_fallback",
    }
    return paths


def _build_runtime_config(cfg: Dict[str, Any], timestamp: str, paths: Dict[str, Path]) -> Dict[str, Any]:
    """Create the temporary runtime config dict passed to child scripts."""
    runtime_config = cfg.copy()
    runtime_config.update(
        {
            "run_timestamp": timestamp,
            "output_run_dir": str(paths["output_run_dir"]),
            "datamatrix_output_excel": str(paths["datamatrix_output_excel"]),
            "stain_output_excel": str(paths["stain_output_excel"]),
            "stain_output_pdf": str(paths["stain_output_pdf"]),
            "label_crops_dir": str(paths["label_crops_dir"]),
            "staging_dir": str(paths["allslides_dir"]),
            "datamatrix_failed_folder": str(paths["datamatrix_failed_dir"]),
            "fallback_failed_folder": str(paths["fallback_failed_dir"]),
        }
    )
    return runtime_config


def run_once(cfg: Dict[str, Any]) -> None:
    """Execute one full pipeline run based on the provided config."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    paths = _prepare_runtime_paths(cfg, timestamp)

    # Step 0: import new slides into staging (marking deferred)
    originals = copy_new_slides(
        cfg=cfg,
        watch_dir=Path(cfg["watch_dir"]),
        staging_dir=paths["allslides_dir"],
        failed_dir=paths["failed_output_dir"],
        run_id=timestamp,
    )

    if not originals:
        logging.info("[INFO] No new slides to process.")
        return

    paths["output_run_dir"].mkdir(parents=True, exist_ok=True)
    log_file = paths["output_run_dir"] / f"pipeline_{timestamp}.log"
    _attach_file_logger(log_file)

    # Ensure the SQLite registry exists before usage
    try:
        db_path = Path(cfg.get("sqlite_db_path", "registry.db"))
        table = safe_table(cfg.get("ops_table", "slide_registry"))
        conn = ensure_db(db_path)
        ensure_registry_schema(conn, table)
        conn.close()
        logging.info(f"[OK] SQLite registry ready: {db_path} (table: {table})")
    except Exception as exc:
        logging.info(f"[WARN] Could not initialize SQLite registry: {exc}")

    # Ensure required folders for later steps
    paths["label_crops_dir"].mkdir(parents=True, exist_ok=True)
    Path(cfg["slide_id_generator"]["final_output_dir"]).mkdir(parents=True, exist_ok=True)
    paths["failed_output_dir"].mkdir(parents=True, exist_ok=True)
    paths["datamatrix_failed_dir"].mkdir(parents=True, exist_ok=True)
    paths["fallback_failed_dir"].mkdir(parents=True, exist_ok=True)

    # Build runtime config for child scripts and write temp YAML atomically
    runtime_config = _build_runtime_config(cfg, timestamp, paths)
    temp_config_path = Path("temp_config.yaml")
    _atomic_write_yaml(temp_config_path, runtime_config)

    log_level = str(cfg.get("log_level", "INFO")).upper()

    # 1) LabelRegionExtractor
    if not run_script(
        "LabelRegionExtractor",
        cfg["scripts"]["label_extractor"],
        temp_config_path,
        log_level=log_level,
        supports_log_level=True,
    ):
        return

    # 2) DataMatrixReader
    if not run_script(
        "DataMatrixReader",
        cfg["scripts"]["datamatrix_reader"],
        temp_config_path,
        log_level=log_level,
        supports_log_level=True,
    ):
        return
    if not paths["datamatrix_output_excel"].exists():
        logging.info("[ERROR] No DataMatrix result file found.")
        return
    logging.info(f"[OK] DataMatrix Excel set to: {paths['datamatrix_output_excel']}")

    # 2.5) ROI fallback (files from this run's failed_datamatrix)
    if paths["datamatrix_failed_dir"].exists():
        files_waiting = [
            p
            for p in paths["datamatrix_failed_dir"].iterdir()
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]
    else:
        files_waiting = []

    if files_waiting:
        logging.info(f"[STEP] ROI fallback: {len(files_waiting)} file(s) in {paths['datamatrix_failed_dir']}")

        roi_csv = paths["output_run_dir"] / f"roi_results_{timestamp}.csv"
        roi_xlsx = paths["output_run_dir"] / "roi_results.xlsx"
        roi_pdf = paths["output_run_dir"] / "roi_results.pdf"
        roi_debug_parts = paths["output_run_dir"] / f"roi_debug_parts_{timestamp}"

        roi_config = runtime_config.copy()
        roi_config.update(
            {
                "failed_output_dir": str(paths["datamatrix_failed_dir"]),
                "output_csv": str(roi_csv),
                "output_xlsx": str(roi_xlsx),
                "output_pdf": str(roi_pdf),
                "debug_parts_root": str(roi_debug_parts),
                "output_run_dir": str(paths["output_run_dir"]),
            }
        )

        temp_config_roi = Path("temp_config_roi.yaml")
        _atomic_write_yaml(temp_config_roi, roi_config)

        # Call ROI CLI
        extra_args = ["--input-dir", str(paths["datamatrix_failed_dir"])]

        if not run_script(
            "ROI Fallback",
            cfg["scripts"]["roi_metadata_extractor"],
            temp_config_roi,
            extra_args=extra_args,
            log_level=log_level,
            supports_log_level=True,
        ):
            logging.info("[WARN] ROI fallback step failed; continuing without it.")
        else:
            _merge_roi_into_datamatrix(paths["datamatrix_output_excel"], roi_csv)
            _move_unresolved_to_fallback(
                paths["datamatrix_failed_dir"],
                roi_csv,
                paths["fallback_failed_dir"],
            )

        temp_config_roi.unlink(missing_ok=True)
    else:
        logging.info(f"[INFO] ROI fallback skipped (no files in {paths['datamatrix_failed_dir']}).")

    # 3) StainDataExtractor
    if not run_script(
        "StainDataExtractor",
        cfg["scripts"]["stain_extractor"],
        temp_config_path,
        log_level=log_level,
        supports_log_level=True,
    ):
        return
    if not paths["stain_output_excel"].exists():
        logging.info("[ERROR] No Stain result file found.")
        return
    logging.info(f"[OK] Stain Excel set to: {paths['stain_output_excel']}")

    # 4) SlideIDcreator
    if not run_script(
        "SlideIDcreator",
        cfg["scripts"]["slide_id_creator"],
        temp_config_path,
        log_level=log_level,
        supports_log_level=True,
    ):
        return

    # 5) SQLiteTableGenerator (optional, now DB-only; no CSV dependency)
    sqlite_script = (cfg.get("scripts", {}) or {}).get("sqlite_table_generator")
    if sqlite_script:
        logging.info(f"[STEP] SQLiteTableGenerator using config: {temp_config_path}")
        if not run_script("SQLiteTableGenerator", sqlite_script, temp_config_path, log_level=log_level):
            logging.info("[WARN] SQLiteTableGenerator failed; continuing pipeline.")

    logging.info("\n[DONE] Pipeline run completed.")
    try:
        temp_config_path.unlink(missing_ok=True)
    except Exception:
        pass


def run(cfg_path: Path) -> None:
    """Entry point for the `run` subcommand (supports watch mode)."""
    cfg = _load_config(cfg_path)
    setup_logging(cfg.get("log_level", "INFO"))

    watch_mode = bool(cfg.get("watch_mode", False))
    if watch_mode:
        interval = int(cfg.get("watch_interval_minutes", 60))
        logging.info(f"[INFO] Watch mode active. Checking every {interval} minutes. Press Ctrl+C to stop.")
        try:
            while True:
                run_once(cfg)
                logging.info(f"[INFO] Waiting {interval} minutes...")
                time.sleep(interval * 60)
        except KeyboardInterrupt:
            logging.info("[STOP] Watch mode interrupted by user.")
    else:
        run_once(cfg)


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------

_REQUIRED_TOP_KEYS = [
    "run_output_dir",
    "label_root_dir",
    "staging_dir",
    "failed_output_dir",
    "scripts",
    "slide_id_generator",
    "watch_dir",
]

_REQUIRED_SCRIPTS = [
    "label_extractor",
    "datamatrix_reader",
    "stain_extractor",
    "slide_id_creator",
    # sqlite_table_generator is optional
]


def validate(cfg_path: Path) -> int:
    """Validate the config file for required keys and paths."""
    setup_logging("INFO")
    try:
        cfg = _load_config(cfg_path)
    except Exception as exc:
        logging.info(f"[ERROR] Cannot load config: {exc}")
        return 1

    ok = True

    for key in _REQUIRED_TOP_KEYS:
        if key not in cfg:
            logging.info(f"[ERROR] Missing required config key: {key}")
            ok = False

    if "scripts" in cfg:
        scripts = cfg["scripts"]
        for s in _REQUIRED_SCRIPTS:
            if s not in scripts:
                logging.info(f"[ERROR] Missing script path for: {s}")
                ok = False
            else:
                sp = Path(scripts[s])
                if not sp.exists():
                    logging.info(f"[ERROR] Script not found: {sp}")
                    ok = False

    if not ok:
        return 2

    logging.info("[OK] Config validation successful.")
    return 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_cli(argv: Optional[List[str]] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=__module_prog__,
        description="WSI-Babel-Shark pipeline runner.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the pipeline using a YAML config.")
    p_run.add_argument("--config", required=True, help="Path to YAML config file.")
    p_run.set_defaults(func=lambda args: run(Path(args.config)))

    p_val = sub.add_parser("validate", help="Validate the YAML config file.")
    p_val.add_argument("--config", required=True, help="Path to YAML config file.")
    p_val.set_defaults(func=lambda args: sys.exit(validate(Path(args.config))))

    p_ver = sub.add_parser("version", help="Show version information.")
    p_ver.set_defaults(func=lambda args: print(f"{__module_prog__} runner 1.1.0", flush=True))

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_cli(argv)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
