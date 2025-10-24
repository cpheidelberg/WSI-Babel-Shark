#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Babel-Shark Unified CLI (thin wrapper, no logic change)

This command-line interface provides a single entry point for all modules
of the WSI-Babel-Shark pipeline (collect, label, datamatrix, extract, stain, etc.).
It only delegates commands to existing module scripts without altering their logic.

Usage examples:
    PYTHONPATH=./src/babel_shark python -m cli collect run --config ./config/config.yaml
    PYTHONPATH=./src/babel_shark python -m cli label run --config ./config/config.yaml
    PYTHONPATH=./src/babel_shark python -m cli dm run --config ./config/config.yaml
    PYTHONPATH=./src/babel_shark python -m cli extract run --config ./config/config.yaml
    PYTHONPATH=./src/babel_shark python -m cli stain run --config ./config/config.yaml
    PYTHONPATH=./src/babel_shark python -m cli slideid run --config ./config/config.yaml
    PYTHONPATH=./src/babel_shark python -m cli db run --config ./config/config.yaml
    PYTHONPATH=./src/babel_shark python -m cli pipeline run --config ./config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple


# --------------------------------------------------------------------------- #
# GLOBAL CONSTANTS                               #
# --------------------------------------------------------------------------- #

HERE = Path(__file__).resolve().parent

# Map subcommands to their corresponding module scripts.
MODULE_MAP = {
    "collect": "collect_slides.py",
    "label": "label_extractor.py",
    "dm": "datamatrix_reader.py",
    "extract": "roi_metadata_extractor.py",
    "stain": "stain_extractor.py",
    "slideid": "slide_id_generator.py",
    "db": "database_manager.py",
    "pipeline": "runner.py",
}

# Supported actions each module can perform
VALID_ACTIONS = ("run", "validate", "version")


# --------------------------------------------------------------------------- #
# UTILITY FUNCTIONS                              #
# --------------------------------------------------------------------------- #

def _setup_logging(verbosity: int) -> None:
    """
    Configure global logging with simple message-only formatting.
    """
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])


def _resolve_script(subcmd: str) -> Path:
    """
    Resolve the absolute path of the underlying Python script
    corresponding to the given subcommand.
    """
    fname = MODULE_MAP.get(subcmd)
    if not fname:
        raise SystemExit(f"[ERROR] Unknown subcommand: {subcmd}")
    path = (HERE / fname).resolve()
    if not path.exists():
        raise SystemExit(f"[ERROR] Module not found: {path}")
    return path


def _build_cmd(pyfile: Path, action: str, passthrough: Sequence[str]) -> List[str]:
    """
    Construct the subprocess command that calls the target script
    with the given action and all passthrough arguments.
    """
    if action not in VALID_ACTIONS:
        raise SystemExit(f"[ERROR] Invalid action '{action}'. Expected one of {VALID_ACTIONS}.")
    return [sys.executable, "-u", str(pyfile), action, *passthrough]


def _run_child(cmd: Sequence[str]) -> int:
    """
    Execute a child process and stream stdout/stderr live.
    Returns the subprocess exit code.
    """
    logging.debug(f"[DEBUG] exec: {' '.join(map(str, cmd))}")
    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        try:
            proc.terminate()
        finally:
            return 130
    return int(proc.returncode or 0)


def _split_known_unknown(argv: Sequence[str], stop_at_double_dash: bool = True) -> Tuple[List[str], List[str]]:
    """
    Split command-line arguments into known vs. passthrough parts.

    Example:
        cli extract run --config cfg.yaml -- --log-level DEBUG
    Everything after '--' is passed directly to the underlying script.
    """
    if stop_at_double_dash and "--" in argv:
        idx = list(argv).index("--")
        return list(argv[:idx]), list(argv[idx + 1:])
    return list(argv), []


# --------------------------------------------------------------------------- #
# CLI LOGIC                                  #
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> int:
    """
    Main entry point for the unified CLI.
    Parses top-level arguments and delegates execution to submodules.
    """
    argv = sys.argv[1:] if argv is None else list(argv)

    parser = argparse.ArgumentParser(
        prog="babel-shark-cli",
        description="Unified command-line interface for WSI-Babel-Shark pipeline.",
        add_help=True,
    )
    parser.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v, -vv).")
    parser.add_argument("--version", action="version", version="babel-shark-cli 0.1.0")

    sub = parser.add_subparsers(dest="subcmd", required=True)

    # Common template: each subcommand supports actions {run, validate, version}
    def add_stage(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("action", choices=VALID_ACTIONS, help="Action to perform (run/validate/version).")
        p.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed through to the target script.")
        return p

    add_stage("collect",  "Collect slides from watch folder into staging_dir.")
    add_stage("label",    "Extract/crop label or macro images from WSIs/PNGs.")
    add_stage("dm",       "Read and decode DataMatrix barcodes from label images.")
    add_stage("extract",  "ROI-based metadata extraction (delegates to metadata_extractor_utilities.cli_main).")
    add_stage("stain",    "Detect or classify stains (optionally with ROI fallback).")
    add_stage("slideid",  "Generate SlideIDs and rename WSIs.")
    add_stage("db",       "Update SQLite registry or slide_metadata.csv.")
    add_stage("pipeline", "Run the full pipeline orchestrator via runner.py.")

    # Split known vs passthrough arguments to support '--' forwarding
    known, unknown = _split_known_unknown(argv)
    args = parser.parse_args(known)
    _setup_logging(args.verbose)

    # Merge passthrough arguments (before and after '--')
    passthrough = args.args + unknown

    # Locate the underlying Python script
    pyfile = _resolve_script(args.subcmd)

    # Construct and execute the subprocess command
    cmd = _build_cmd(pyfile, args.action, passthrough)

    # Ensure unbuffered output for immediate log streaming
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    return _run_child(cmd)


# --------------------------------------------------------------------------- #
# ENTRY POINT                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    raise SystemExit(main())
