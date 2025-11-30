#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
database_manager.py — SQLite registry builder/updater for WSI-Babel-Shark

Reads the Excel registry produced by slide_id_generator and
upserts rows into the central SQLite registry defined by:

    - sqlite_db_path
    - ops_table

in the YAML config.

Usage (called from runner.py):

    python -u src/babel_shark/database_manager.py run \
        --config temp_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


# ---------------------------------------------------------------------------
# Config / DB helpers
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config from disk."""
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg


def safe_table(name: str) -> str:
    """Validate a table name to avoid SQL injection via config."""
    name = str(name)
    import re

    m = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
    if not m:
        raise ValueError(f"Invalid table name: {name!r}")
    return name


def ensure_db(db_path: Path) -> sqlite3.Connection:
    """Open/create SQLite DB with stable PRAGMAs."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def ensure_registry_schema(conn: sqlite3.Connection, table: str) -> None:
    """
    Ensure the slide registry schema exists.

    This must stay in sync with runner.py::ensure_registry_schema.
    """
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id               TEXT NOT NULL,
            file_name            TEXT NOT NULL,                -- label image name / base name
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
    cur.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS ux_{table}_newfile ON {table}(new_file_name);"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS ix_{table}_orig ON {table}(original_file_name);"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS ix_{table}_datamatrix ON {table}(datamatrix);"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS ix_{table}_lab_case ON {table}(labid, case6);"
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Normalization helpers (Year4 / Case6)
# ---------------------------------------------------------------------------


def _digits_only(x: Any) -> str:
    import re

    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return re.sub(r"[^\d]", "", str(x))


def to_year4_str(y: Any) -> str:
    """Normalize Year to 4-digit string (e.g. '24' -> '2024')."""
    s = _digits_only(y)
    if not s:
        return ""
    if len(s) == 2:
        return f"20{s}"
    if len(s) == 4:
        return s
    return s  # fallback: keep as-is


def to_case6_str(c: Any) -> str:
    """Normalize case number to 6-digit string."""
    s = _digits_only(c)
    if not s:
        return ""
    try:
        return f"{int(s):06d}"
    except Exception:
        return s


# ---------------------------------------------------------------------------
# Mapping Excel -> Registry rows
# ---------------------------------------------------------------------------


def build_registry_df(df_full: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """
    Map slide_id_generator's Excel (full_merged) into registry columns.

    Expected columns in df_full (best effort):
        - FileName
        - SlideID
        - CaseID
        - OriginalFileName
        - NewFileName
        - DataMatrix
        - LabID
        - Year
        - CaseNumber
        - Pot
        - BlockID
        - Section
        - Stain
        - method of extraction
        - ScanTime_UTC
        - AnalysisTime_UTC
        - InputPath
        - OutputPath
        - status

    Registry columns (excluding id + inserted_at):
        run_id, file_name, slide_id, case_id, original_file_name, new_file_name,
        datamatrix, labid, year4, case6, pot, blockid, section, stain,
        method_of_extraction, scan_time_utc, analysis_time_utc,
        input_path, output_path, status
    """
    # Ensure we work with strings
    df = df_full.copy()

    for col in df.columns:
        df[col] = df[col].astype("string")

    # Compute year4 and case6 from Year / CaseNumber (if present)
    year4_list: List[str] = []
    case6_list: List[str] = []

    for _, row in df.iterrows():
        year4_list.append(to_year4_str(row.get("Year", "")))
        case6_list.append(to_case6_str(row.get("CaseNumber", "")))

    df["year4"] = pd.Series(year4_list, dtype="string")
    df["case6"] = pd.Series(case6_list, dtype="string")

    # Standardize some column names for mapping
    # (we tolerate missing columns by filling with empty strings later)
    colmap = {
        "FileName": "file_name",
        "SlideID": "slide_id",
        "CaseID": "case_id",
        "OriginalFileName": "original_file_name",
        "NewFileName": "new_file_name",
        "DataMatrix": "datamatrix",
        "LabID": "labid",
        "Pot": "pot",
        "BlockID": "blockid",
        "Section": "section",
        "Stain": "stain",
        "ScanTime_UTC": "scan_time_utc",
        "AnalysisTime_UTC": "analysis_time_utc",
        "InputPath": "input_path",
        "OutputPath": "output_path",
        "status": "status",
        "method of extraction": "method_of_extraction",
    }

    # Build an output DataFrame with canonical registry columns
    registry_cols = [
        "run_id",
        "file_name",
        "slide_id",
        "case_id",
        "original_file_name",
        "new_file_name",
        "datamatrix",
        "labid",
        "year4",
        "case6",
        "pot",
        "blockid",
        "section",
        "stain",
        "method_of_extraction",
        "scan_time_utc",
        "analysis_time_utc",
        "input_path",
        "output_path",
        "status",
    ]

    rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        rec: Dict[str, Any] = {}

        rec["run_id"] = str(run_id)

        # Direct mappings via colmap
        for src, dst in colmap.items():
            val = row.get(src, "")
            rec[dst] = "" if val is None or pd.isna(val) else str(val)

        # year4 / case6 from computed columns
        rec["year4"] = "" if pd.isna(row.get("year4", "")) else str(row.get("year4", ""))
        rec["case6"] = "" if pd.isna(row.get("case6", "")) else str(row.get("case6", ""))

        # Ensure all keys exist (fill missing with "")
        for key in registry_cols:
            if key not in rec:
                rec[key] = ""

        rows.append(rec)

    df_reg = pd.DataFrame(rows, columns=registry_cols)
    return df_reg


# ---------------------------------------------------------------------------
# DB Upsert
# ---------------------------------------------------------------------------


def upsert_rows(conn: sqlite3.Connection, table: str, df_reg: pd.DataFrame) -> None:
    """
    Insert/replace registry rows into SQLite.

    Uses INSERT OR REPLACE relying on unique index on new_file_name
    (see ensure_registry_schema).
    """
    if df_reg is None or df_reg.empty:
        logging.info("[INFO] No rows to upsert into registry.")
        return

    table = safe_table(table)

    # Registry columns (must match build_registry_df and ensure_registry_schema)
    cols = [
        "run_id",
        "file_name",
        "slide_id",
        "case_id",
        "original_file_name",
        "new_file_name",
        "datamatrix",
        "labid",
        "year4",
        "case6",
        "pot",
        "blockid",
        "section",
        "stain",
        "method_of_extraction",
        "scan_time_utc",
        "analysis_time_utc",
        "input_path",
        "output_path",
        "status",
    ]

    placeholders = ",".join(["?"] * len(cols))
    col_list = ",".join(cols)

    sql = f"""
    INSERT OR REPLACE INTO {table}
    ({col_list})
    VALUES ({placeholders})
    """

    cur = conn.cursor()

    # Iterate rows and execute
    for _, row in df_reg.iterrows():
        vals = [row.get(c, "") for c in cols]
        cur.execute(sql, vals)

    conn.commit()
    logging.info(f"[OK] Upserted {len(df_reg)} row(s) into table '{table}'.")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> None:
    """Handle the 'run' subcommand."""
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    # Logging handled here so runner's --log-level doesn't break us
    log_level = str(getattr(args, "log_level", cfg.get("log_level", "INFO"))).upper()
    setup_logging(log_level)

    db_path = Path(cfg.get("sqlite_db_path", "registry.db"))
    table_name = cfg.get("ops_table", "slide_registry")
    table_name = safe_table(table_name)

    # Run ID: use run_timestamp from temp_config.yaml if present
    run_id = str(cfg.get("run_timestamp", datetime.utcnow().isoformat()))

    # Excel with full_merged sheet
    meta_path = Path(cfg.get("metadata_excel_path", "slide_metadata.xlsx"))

    if not meta_path.exists():
        logging.info(f"[ERROR] Metadata Excel not found: {meta_path}")
        return

    try:
        df_full = pd.read_excel(meta_path, sheet_name="full_merged")
    except Exception as exc:
        logging.info(f"[ERROR] Cannot read sheet 'full_merged' from {meta_path}: {exc}")
        return

    # Build registry DataFrame
    df_reg = build_registry_df(df_full, run_id)

    # Connect and upsert
    conn = ensure_db(db_path)
    ensure_registry_schema(conn, table_name)
    upsert_rows(conn, table_name, df_reg)
    conn.close()

    logging.info("[DONE] SQLite registry updated.")


def build_cli(argv: Any = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="WSI-Babel-Shark-SQLite",
        description="SQLite registry manager for WSI-Babel-Shark.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Build/update SQLite registry from Excel.")
    p_run.add_argument("--config", required=True, help="Path to YAML config file.")
    # Accept --log-level to be compatible with runner.run_script
    p_run.add_argument("--log-level", default="INFO", help="Logging level.")
    p_run.set_defaults(func=cmd_run)

    p_ver = sub.add_parser("version", help="Show version.")
    p_ver.set_defaults(func=lambda args: print("WSI-Babel-Shark SQLite manager 1.0.0", flush=True))

    return parser


def main(argv: Any = None) -> None:
    parser = build_cli(argv)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
