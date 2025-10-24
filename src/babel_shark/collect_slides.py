#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_slides.py — Babel-Shark WSI Collector

Collects whole-slide image (WSI) files from a watch folder and stores them
into a defined source directory by copy or move operation. The behavior and
valid extensions are fully configurable through a YAML file.

Main features:
    Supports multiple WSI extensions (e.g., .svs, .ndpi, .mrxs)
    Configurable copy/move mode via YAML key 'operation_mode'
    Ensures atomic file operations (no partial copies)
    Simple logging-based progress reporting
    Command-line interface for run, validate, and version

Usage:
    # Run collection
    PYTHONPATH=./src python -u src/collect_slides.py run \
        --config ./config/config.yaml

    # Validate configuration
    PYTHONPATH=./src python -u src/collect_slides.py validate \
        --config ./config/config.yaml

    # Show version
    PYTHONPATH=./src python -u src/collect_slides.py version
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import yaml

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure logger to emit message-only lines (no timestamps/levels).

    Args:
        level: Logging level name (e.g., "INFO", "DEBUG", "ERROR").

    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger("collect_slides")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Avoid duplicate handlers on repeated runs.
    logger.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: Union[str, Path]) -> Dict:
    """Load YAML configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(Path(path), "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def validate_config(conf: Dict) -> List[str]:
    """Validate presence of required configuration keys.

    Args:
        conf: Configuration dictionary.

    Returns:
        A list of human-readable issues; empty list means OK.
    """
    required_keys = ["watch_dir", "staging_dir", "wsi_types"]
    issues: List[str] = []
    for key in required_keys:
        if key not in conf:
            issues.append(f"Missing required config key: {key}")
    # operation_mode is optional; default is handled in collect_slides().
    return issues


# --------------------------------------------------------------------------- #
# Atomic file operations
# --------------------------------------------------------------------------- #
def _temp_in(dest_dir: Path, final_name: str) -> Path:
    """Create a unique temporary file path inside the destination directory.

    Args:
        dest_dir: Destination directory.
        final_name: Final file name (used for temp name readability).

    Returns:
        Path to a unique temporary file inside dest_dir.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    prefix = f".{final_name}.tmp."
    fd, tmp_path = tempfile.mkstemp(prefix=prefix, dir=dest_dir)
    os.close(fd)
    return Path(tmp_path)


def atomic_copy(src: Path, dest: Path) -> None:
    """Atomically copy a file to destination using os.replace().

    Args:
        src: Source file path.
        dest: Destination file path.
    """
    tmp = _temp_in(dest.parent, dest.name)
    shutil.copy2(src, tmp)
    os.replace(tmp, dest)


def atomic_move(src: Path, dest: Path) -> None:
    """Atomically move a file to destination, handling cross-device moves.

    Args:
        src: Source file path.
        dest: Destination file path.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Fast path: same filesystem; replace is atomic and removes src.
        os.replace(src, dest)
    except OSError:
        # Cross-device: copy2 -> replace -> remove original.
        tmp = _temp_in(dest.parent, dest.name)
        shutil.copy2(src, tmp)
        os.replace(tmp, dest)
        try:
            os.remove(src)
        except FileNotFoundError:
            # If already removed, ignore.
            pass


# --------------------------------------------------------------------------- #
# Core logic 
# --------------------------------------------------------------------------- #
def collect_slides(conf: Dict, logger: logging.Logger) -> None:
    """Collect WSI files by copying/moving from source to destination.

    Behavior:
      - Valid extensions are taken from conf["wsi_types"] and matched via str.endswith.
      - Operation mode is conf.get("operation_mode", "copy").lower() in {'copy','move'}.
      - Destination directory is created if it doesn't exist.
      
    Args:
        conf: Configuration dict.
        logger: Logger configured for message-only output.
    """
    source_dir = Path(conf["watch_dir"])
    dest_dir = Path(conf["staging_dir"])
    dest_dir.mkdir(parents=True, exist_ok=True)

    valid_exts: Tuple[str, ...] = tuple(conf["wsi_types"])
    operation_mode: str = conf.get("operation_mode", "copy").lower()

    moved_or_copied = 0

    for root, _, files in os.walk(source_dir):
        for file in files:
            if not file.lower().endswith(valid_exts):
                continue

            src = Path(root) / file
            dest = dest_dir / file

            try:
                if operation_mode == "move":
                    atomic_move(src, dest)
                    # Keep exact legacy message text:
                    logger.info(f"[Moved] {src} TO {dest}")
                else:
                    atomic_copy(src, dest)
                    # Keep exact legacy message text:
                    logger.info(f"[Copied] {src} TO {dest}")
                moved_or_copied += 1
            except Exception as exc:  # keep broad handling as legacy
                logger.info(f"[ERROR] Failed to {operation_mode} {src}: {exc}")

    # Preserve legacy summary wording:
    logger.info(f"[SUMMARY] {moved_or_copied} files {operation_mode}d.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser with subcommands and legacy support.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Collect and copy/move new WSI files."
    )

    # Legacy compatibility: allow top-level --config without subcommand.
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to the YAML configuration file. (legacy compatibility)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (e.g., INFO, DEBUG, ERROR).",
    )

    subparsers = parser.add_subparsers(dest="command")

    p_run = subparsers.add_parser("run", help="Run slide collection.")
    p_run.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )

    p_val = subparsers.add_parser("validate", help="Validate configuration file.")
    p_val.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )

    subparsers.add_parser("version", help="Show version and exit.")

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Entry point with legacy-compatible behavior.

    Args:
        argv: Optional argument vector. Defaults to sys.argv[1:].

    Returns:
        Process exit code: 0 on success, non-zero on error.
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logger = setup_logging(level=getattr(args, "log_level", "INFO"))

    # Legacy path: no subcommand but --config provided.
    if args.command is None:
        if args.config:
            config_path = Path(args.config).resolve()
            conf = load_config(config_path)

            # Banner preserved exactly:
            logger.info("=" * 50)
            logger.info(" Babel-Shark WSI Data Collection")
            logger.info("=" * 50)

            collect_slides(conf, logger)
            logger.info("Data collection completed successfully.")
            return 0

        parser.print_help()
        return 2

    # Subcommands
    if args.command == "run":
        config_path = Path(args.config).resolve()
        conf = load_config(config_path)

        # Banner preserved exactly:
        logger.info("=" * 50)
        logger.info(" Babel-Shark WSI Data Collection")
        logger.info("=" * 50)

        collect_slides(conf, logger)
        logger.info("Data collection completed successfully.")
        return 0

    if args.command == "validate":
        config_path = Path(args.config).resolve()
        conf = load_config(config_path)
        issues = validate_config(conf)
        if issues:
            for msg in issues:
                logger.info(f"[CONFIG] {msg}")
            return 1
        logger.info("[CONFIG] OK")
        return 0

    if args.command == "version":
        logger.info(__version__)
        return 0

    parser.print_help()
    return 2


# --------------------------------------------------------------------------- #
# Script execution guard
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    raise SystemExit(main())
