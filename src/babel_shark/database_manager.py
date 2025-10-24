#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
database_manager.py — Babel-Shark SQLite Table Manager

Consolidates DataMatrix, stain, and renamed slide information into a
SQLite registry and mirrors the registry as a CSV file.

Main features:
    Normalizes and merges multiple Excel inputs
    Builds DataMatrix and SlideID columns automatically
    Upserts entries into a SQLite registry with indexing
    Generates compact slide_metadata.csv for downstream tools
    Command-line interface for run, validate, and version

Usage:
    PYTHONPATH=./src python -u src/database_manager.py run \
        --config ./config/config.empty.yaml
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger with a simple, consistent format."""
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("sqlite_table_generator")


logger = setup_logging("INFO")

# --------------------------------------------------------------------------- #
# I/O & Normalizers
# --------------------------------------------------------------------------- #

def read_excel_safe(path: Path) -> pd.DataFrame:
    """Read Excel entirely as TEXT to avoid float coercion (e.g., 2453 -> 2453.0)."""
    try:
        return pd.read_excel(path, dtype=str)
    except Exception as e:  # noqa: BLE001 - preserve original behavior
        logger.warning(f"[WARN] Cannot read Excel: {path} -> {e}")
        return pd.DataFrame()


def digits_only(x: object) -> str:
    """Return only digits from any input (kept for fallback paths)."""
    return re.sub(r"[^\d]", "", str(x) if x is not None else "")


def to_int_str_no_decimal(x: object) -> Optional[str]:
    """
    Return integer-like string, dropping zero decimals (supports both '.0' and ',0').
    Examples:
      '2453.0' -> '2453', '2453,0' -> '2453', '  0021 ' -> '21'
    """
    if x is None:
        return None
    s = str(x).strip()
    m = re.fullmatch(r"\s*([0-9]+)(?:[.,]0+)?\s*", s)
    if m:
        return str(int(m.group(1)))
    d = digits_only(s)
    return d or None


def to_year4_str(y: object) -> Optional[str]:
    """Return 4-digit year; map '24' -> '2024'; return None if invalid."""
    s = to_int_str_no_decimal(y)
    if not s:
        return None
    if re.fullmatch(r"[0-9]{2}", s):
        return f"20{s}"
    if re.fullmatch(r"[0-9]{4}", s):
        return s
    return None


def to_case6_str(c: object) -> Optional[str]:
    """Zero-pad case numbers to 6 digits (robust to '2453.0', '2453,0', ' 2453 ')."""
    s = to_int_str_no_decimal(c)
    if not s:
        return None
    return f"{int(s):06d}"


def normalize_stain(s: Optional[str]) -> str:
    """Normalize common H&E variants; otherwise return trimmed raw string."""
    if s is None:
        return "H&E"
    t_raw = str(s).strip()
    t = t_raw.upper().replace(" ", "")
    if t in {"HE", "H-E", "H& E", "H &E", "H & E", "H&E", "H&Ε"}:
        return "H&E"
    return t_raw.strip()


def build_datamatrix(
    lab: object,
    year4: object,
    case6: object,
    pot: object,
    blockid: object,
    section: object,
) -> Optional[str]:
    """
    Build a DataMatrix-like identifier:
        {LabID}{Year4}{Case6}S{Pot}-{BlockID}-{Section}
    Return None if any required field is missing.
    """
    lab_s = (str(lab).strip().upper() if lab else "")
    y4 = to_year4_str(year4)
    c6 = to_case6_str(case6)
    pot_s = (str(pot).strip().upper() if pot else "")
    bid = to_int_str_no_decimal(blockid) or (str(blockid).strip() if blockid is not None else "")
    sec = to_int_str_no_decimal(section) or (str(section).strip() if section is not None else "")
    if not (lab_s and y4 and c6 and pot_s and bid and sec):
        return None
    return f"{lab_s}{y4}{c6}S{pot_s}-{bid}-{sec}"

# --------------------------------------------------------------------------- #
# SQLite Helpers
# --------------------------------------------------------------------------- #

def ensure_db(db_path: Path) -> sqlite3.Connection:
    """Open SQLite connection with sane PRAGMAs for stability and concurrency."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def safe_table(name: str) -> str:
    """
    Whitelist table identifier: letters, digits, underscore; must start with letter/underscore.
    This avoids accidental SQL injection via config.
    """
    m = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(name))
    if not m:
        raise ValueError(f"Invalid table name: {name!r}")
    return name


def ensure_schema(conn: sqlite3.Connection, table: str) -> None:
    """Create registry table and indexes if they don't exist."""
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
            inserted_at          TEXT DEFAULT (datetime('now'))
        );
        """
    )
    cur.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS ux_{table}_newfile ON {table}(new_file_name);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS ix_{table}_datamatrix ON {table}(datamatrix);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS ix_{table}_lab_case ON {table}(labid, case6);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS ix_{table}_orig ON {table}(original_file_name);")
    conn.commit()

# --------------------------------------------------------------------------- #
# FS Indexing
# --------------------------------------------------------------------------- #

PREFERRED_EXTS: List[str] = [
    ".svs", ".ndpi", ".scn", ".mrxs", ".bif", ".tif", ".tiff", ".png", ".jpg", ".jpeg"
]


def index_final_folder(root: Path) -> Tuple[Dict[str, List[Path]], Dict[str, Path]]:
    """
    Scan 'final_output_dir' once and build two indexes:
    - by_stem: map <filename-without-ext> -> list of matching Paths
    - by_name: map <filename-with-ext>   -> Path
    """
    by_stem: Dict[str, List[Path]] = {}
    by_name: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        stem = p.stem
        by_stem.setdefault(stem, []).append(p)
        by_name[p.name] = p
    return by_stem, by_name


def pick_best_match(candidates: List[Path]) -> Optional[Path]:
    """Pick the 'best' file by extension priority to resolve ambiguity."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    ranked = {ext: i for i, ext in enumerate(PREFERRED_EXTS)}
    best = min(candidates, key=lambda p: ranked.get(p.suffix.lower(), 999))
    return best

# --------------------------------------------------------------------------- #
# Core Logic
# --------------------------------------------------------------------------- #

def load_and_merge(dm_xlsx: Path, stain_xlsx: Path) -> pd.DataFrame:
    """
    Load DataMatrix & Stain Excel sheets, left-join by FileName, and normalize fields.
    Returns a DataFrame with cleaned columns ready for DB insertion.
    """
    dm = read_excel_safe(dm_xlsx)
    st = read_excel_safe(stain_xlsx)

    need_dm_cols = [
        "FileName", "DataMatrix", "LabID", "Year", "CaseNumber",
        "Pot", "BlockID", "Section", "ExtractionMethod",
    ]
    for c in need_dm_cols:
        if c not in dm.columns:
            dm[c] = None

    if "Stain" not in st.columns:
        st["Stain"] = None
    st = st[["FileName", "Stain"]].drop_duplicates()

    df = pd.merge(dm, st, on="FileName", how="left")

    # Normalize core fields
    df["LabID"] = df["LabID"].astype(str).str.strip().str.upper()
    df["Year4"] = df["Year"].apply(to_year4_str).astype("string")
    df["Case6"] = df["CaseNumber"].apply(to_case6_str).astype("string")
    df["Pot"] = df["Pot"].fillna("A").astype(str).str.strip().str.upper().astype("string")
    df["BlockID"] = df["BlockID"].apply(lambda x: to_int_str_no_decimal(x) or "1").astype("string")
    df["Section"] = df["Section"].apply(lambda x: to_int_str_no_decimal(x) or "1").astype("string")
    df["Stain"] = df["Stain"].apply(normalize_stain).astype("string")

    # Rebuild DataMatrix if possible
    df["DataMatrix_v2"] = df.apply(
        lambda r: build_datamatrix(
            r["LabID"], r["Year4"], r["Case6"], r["Pot"], r["BlockID"], r["Section"]
        ),
        axis=1,
    )
    df["DataMatrix"] = df["DataMatrix_v2"].where(
        df["DataMatrix_v2"].notna() & (df["DataMatrix_v2"] != ""),
        df["DataMatrix"],
    )
    df.drop(columns=["DataMatrix_v2"], inplace=True)

    # CaseID & SlideID
    df["CaseID"] = (df["LabID"].fillna("") + df["Year4"].fillna("") + df["Case6"].fillna("")).astype("string")
    df["DataMatrix_NoSection"] = df["DataMatrix"].astype(str).str.replace(r"-[^-]+$", "", regex=True)
    df["Stain"] = df["Stain"].fillna("H&E")
    df["SlideID"] = (df["DataMatrix_NoSection"].astype(str) + "-" + df["Stain"].astype(str)).astype("string")

    # Method of extraction
    def _infer_method(x: object) -> str:
        x_s = str(x or "").strip().lower()
        return "ROIbase" if x_s == "roibase" else "Datamatrix"

    df["method_of_extraction"] = df["ExtractionMethod"].apply(_infer_method).astype("string")
    return df


def resolve_new_file_names(df: pd.DataFrame, final_root: Path) -> pd.DataFrame:
    """
    Try to find the actual renamed slide files in 'final_root' by matching SlideID (stem).
    """
    by_stem, _ = index_final_folder(final_root)

    new_names: List[Optional[str]] = []
    for _, r in df.iterrows():
        stem = str(r["SlideID"])
        cand_paths = by_stem.get(stem) or []
        best = pick_best_match(cand_paths)
        new_names.append(best.name if best else None)

    out = df.copy()
    # Use object dtype to avoid pandas NA boolean issues downstream
    out["NewFileName"] = pd.Series(new_names, index=df.index, dtype=object)
    return out


def _s(v: object) -> str:
    """Safe stringifier: handles None, NaN, pd.NA without boolean evaluation."""
    try:
        if v is None:
            return ""
        if isinstance(v, float) and math.isnan(v):
            return ""
        if pd.isna(v):  # type: ignore[arg-type]
            return ""
    except Exception:  # noqa: BLE001 - defensive path
        pass
    return str(v)


def upsert_rows(conn: sqlite3.Connection, table: str, run_id: str, df: pd.DataFrame) -> None:
    """
    Insert rows into registry; ignore duplicates by unique(new_file_name).
    Rows without NewFileName are still inserted (unique constraint allows multiple NULLs).
    """
    cur = conn.cursor()
    sql = f"""
        INSERT OR IGNORE INTO {table} (
            run_id, file_name, slide_id, case_id,
            original_file_name, new_file_name,
            datamatrix, labid, year4, case6, pot, blockid, section, stain,
            method_of_extraction
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    for _, r in df.iterrows():
        vals = (
            _s(run_id),
            _s(r.get("FileName")),
            _s(r.get("SlideID")),
            _s(r.get("CaseID")),
            _s(r.get("OriginalFileName")),
            _s(r.get("NewFileName")),
            _s(r.get("DataMatrix")),
            _s(r.get("LabID")),
            _s(r.get("Year4")),
            _s(r.get("Case6")),
            _s(r.get("Pot")),
            _s(r.get("BlockID")),
            _s(r.get("Section")),
            _s(r.get("Stain")),
            _s(r.get("method_of_extraction")),
        )
        cur.execute(sql, vals)
    conn.commit()


def _atomic_write_bytes(dst: Path, data: bytes) -> None:
    """Atomically write bytes to dst using a temporary sibling file and os.replace()."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)


def mirror_to_csv(conn: sqlite3.Connection, table: str, csv_path: Path) -> None:
    """
    Export a compact CSV view for downstream steps:
      FileName, SlideID, CaseID, NewFileName, OriginalFileName, method of extraction
    (Written atomically to avoid partial writes.)
    """
    sql = f"""
        SELECT
            file_name           AS FileName,
            slide_id            AS SlideID,
            case_id             AS CaseID,
            new_file_name       AS NewFileName,
            original_file_name  AS OriginalFileName,
            method_of_extraction AS "method of extraction"
        FROM {table}
        ORDER BY id ASC;
    """
    df = pd.read_sql_query(sql, conn)

    # Write atomically (keep exact CSV format as before)
    # Using pandas to CSV into memory first, then atomic replace
    from io import StringIO
    buf = StringIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    _atomic_write_bytes(csv_path, buf.getvalue().encode("utf-8"))
    logger.info(f"[OK] CSV mirrored: {csv_path}")

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_run(args: argparse.Namespace) -> None:
    """Run the full pipeline: load, merge, resolve, upsert, mirror CSV."""
    cfg = _load_config(Path(args.config))

    # Resolve config paths with robust fallbacks (preserved from original)
    db_path = Path(cfg.get("sqlite_db_path", "registry.db"))
    final_root = Path(cfg["final_output_dir"])

    dm_xlsx_cfg = cfg.get("datamatrix_output_excel")
    stain_xlsx_cfg = cfg.get("stain_output_excel")
    if not dm_xlsx_cfg or not stain_xlsx_cfg:
        out_dir = Path(cfg.get("output_run_dir", "."))
        ts = cfg.get("run_timestamp", datetime.now().strftime("%Y-%m-%d_%H-%M"))
        if not dm_xlsx_cfg:
            dm_xlsx_cfg = str(out_dir / f"datamatrix_results_{ts}.xlsx")
        if not stain_xlsx_cfg:
            stain_xlsx_cfg = str(out_dir / f"stain_results_{ts}.xlsx")

    dm_xlsx = Path(dm_xlsx_cfg)
    stain_xlsx = Path(stain_xlsx_cfg)

    csv_out = Path(cfg.get("metadata_csv_path", "slide_metadata.csv"))
    table_name = safe_table(cfg.get("ops_table", "slide_registry"))

    # Run identifier (stable across one pipeline run)
    run_id = str(cfg.get("run_timestamp") or datetime.now().strftime("%Y-%m-%d_%H-%M"))

    # Open DB and ensure schema
    conn = ensure_db(db_path)
    ensure_schema(conn, table_name)

    # Load data, normalize, resolve files
    df = load_and_merge(dm_xlsx, stain_xlsx)
    df = resolve_new_file_names(df, final_root)

    # Replace pd.NA/NaN with None (extra safety before DB upsert)
    df = df.where(pd.notna(df), None)

    # Upsert into DB and mirror to CSV
    upsert_rows(conn, table_name, run_id, df)
    mirror_to_csv(conn, table_name, csv_out)

    conn.close()
    logger.info("[DONE] SQLiteTableGenerator completed.")


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate presence/readability of critical inputs referenced by the config."""
    cfg = _load_config(Path(args.config))

    # Collect checks identical to expectations from original code
    checks: List[Tuple[str, bool, str]] = []

    def _exists(p: Optional[str]) -> Tuple[bool, str]:
        if not p:
            return False, "MISSING"
        q = Path(p)
        return q.exists(), str(q)

    db_p = cfg.get("sqlite_db_path", "registry.db")
    run_dir = cfg.get("output_run_dir", ".")
    ts = cfg.get("run_timestamp", datetime.now().strftime("%Y-%m-%d_%H-%M"))
    dm_p = cfg.get("datamatrix_output_excel") or str(Path(run_dir) / f"datamatrix_results_{ts}.xlsx")
    st_p = cfg.get("stain_output_excel") or str(Path(run_dir) / f"stain_results_{ts}.xlsx")
    final_root = cfg.get("final_output_dir")
    csv_out = cfg.get("metadata_csv_path", "slide_metadata.csv")

    for label, path in [
        ("sqlite_db_path", db_p),
        ("datamatrix_output_excel", dm_p),
        ("stain_output_excel", st_p),
        ("final_output_dir", final_root),
        ("metadata_csv_path", csv_out),
    ]:
        ok, note = _exists(path)
        checks.append((label, ok, note))

    for label, ok, note in checks:
        status = "OK" if ok else "NOT FOUND"
        logger.info(f"{label:24s}: {status:10s} {note}")

    missing = [c for c in checks if not c[1]]
    if missing:
        sys.exit(2)


def cmd_version(_: argparse.Namespace) -> None:
    """Print version information."""
    logger.info(__version__)


def build_cli(argv: Optional[Iterable[str]] = None) -> argparse.ArgumentParser:
    """Construct CLI with subparsers: run, validate, version."""
    ap = argparse.ArgumentParser(
        description="Generate and update SQLite slide registry from latest run outputs."
    )
    sp = ap.add_subparsers(dest="command", required=True)

    p_run = sp.add_parser("run", help="Run the registry update pipeline")
    p_run.add_argument("--config", required=True, help="Path to YAML config")
    p_run.set_defaults(func=cmd_run)

    p_val = sp.add_parser("validate", help="Validate inputs referenced by the config")
    p_val.add_argument("--config", required=True, help="Path to YAML config")
    p_val.set_defaults(func=cmd_validate)

    p_ver = sp.add_parser("version", help="Show module version")
    p_ver.set_defaults(func=cmd_version)

    return ap


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point."""
    parser = build_cli(argv)
    args = parser.parse_args(argv)
    args.func(args)

# --------------------------------------------------------------------------- #
# Script execution guard
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
