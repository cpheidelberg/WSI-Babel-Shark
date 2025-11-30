#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
label_extractor.py — Babel-Shark Label Region Extractor

Extracts label or macro regions from WSI and PNG files, preparing
cropped label images for downstream OCR and barcode decoding.

Main features:
    Detects label and macro associated images in WSIs
    Crops, rotates, and saves label regions atomically as PNG
    Logs unsupported or failed cases to a text file
    Compatible with OpenSlide and Pillow
    Command-line interface for run, validate, and version

Usage:
    PYTHONPATH=./src/babel_shark/ python -u src/babel_shark/label_extractor.py run \
        --config ./config/config.yaml
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import os
import platform
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from PIL import Image

# --------------------------------------------------------------------------- #
# Logging Helpers
# --------------------------------------------------------------------------- #

def setup_logging(level: str = "INFO") -> None:
    """Configure root logging to emulate previous console prints.

    The formatter prints only the message to keep external console output
    visually consistent with prior print(...) usage.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

LOGGER = logging.getLogger("LabelRegionExtractor")
# --------------------------------------------------------------------------- #
# Data Classes
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Paths:
    """Bundle of key filesystem paths derived from the config."""
    input_dir: Path
    output_dir: Path
    output_run_dir: Path
    log_file_path: Path


# --------------------------------------------------------------------------- #
# Utils
# --------------------------------------------------------------------------- #

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration from the provided path."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_windows_dlls(openslide_dll_path: Optional[str]) -> None:
    """Ensure OpenSlide DLLs are available on Windows (no-op elsewhere)."""
    if platform.system() == "Windows" and hasattr(os, "add_dll_directory") and openslide_dll_path:
        try:
            os.environ["PATH"] = str(openslide_dll_path) + os.pathsep + os.environ.get("PATH", "")
            with os.add_dll_directory(openslide_dll_path):
                ctypes.cdll.LoadLibrary(str(Path(openslide_dll_path) / "libopenslide-1.dll"))
        except Exception as exc:
            LOGGER.warning("[WARNING] Could not load OpenSlide DLLs: %s", exc)


def atomic_save_pil_image(img: Image.Image, final_path: Path) -> None:
    """Save a PIL image atomically to final_path using os.replace().

    We pass the image format explicitly because the temporary file ends with
    '.tmp' and Pillow cannot infer the format from that extension.
    """
    final_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep original filename + ".tmp" (e.g., "x.png.tmp") so we can os.replace later
    tmp_path = final_path.with_name(final_path.name + ".tmp")

    # Infer PIL format from the final extension; fallback to PNG (our output is PNG)
    # Example: ".png" -> "PNG"
    ext_to_fmt = Image.registered_extensions()
    pil_fmt = ext_to_fmt.get(final_path.suffix.lower(), "PNG")

    # Explicitly pass format, otherwise Pillow tries to infer from '.tmp' and fails
    img.save(tmp_path, format=pil_fmt)

    # Atomic replace to the final target path
    os.replace(tmp_path, final_path)


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #

# --- Delayed import of openslide (Windows-safe) ------------------------------
# We import openslide only AFTER DLL preparation has been attempted.
openslide = None  # will be set by _load_openslide_after_dll_prep()

def _load_openslide_after_dll_prep():
    """Import openslide only after DLL paths are prepared (Windows-safe)."""
    try:
        import openslide as _openslide  # type: ignore
        return _openslide
    except Exception as exc:  # ImportError or other
        setup_logging("INFO")
        LOGGER.error("[FATAL] Cannot import openslide. Make sure it's installed: %s", exc)
        sys.exit(1)


class LabelExtractor:
    """Extract label or macro regions from WSI and PNG files."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        log_file_path: Path,
        label_crop_ratio: float = 0.3,
        rotation_degrees: int = -90,
        macro_tag: str = "macro",
    ) -> None:
        """Initialize extractor with I/O paths and options."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.log_file_path = log_file_path
        self.label_crop_ratio = float(label_crop_ratio)
        self.rotation_degrees = int(rotation_degrees)
        self.macro_tag = (macro_tag or "macro").lower()

    def log_failure(self, message: str) -> None:
        """Record failures to console and append to log file."""
        LOGGER.error("[FAIL] %s", message)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file_path.open("a", encoding="utf-8") as log:
            log.write(f"{datetime.now().isoformat()}  {message}\n")

    @staticmethod
    def is_supported_wsi(filepath: Path) -> bool:
        """Return True if file can be opened by OpenSlide."""
        try:
            slide = openslide.OpenSlide(str(filepath))  # uses delayed-imported module
            slide.close()
            return True
        except Exception:
            return False

    @staticmethod
    def generate_output_name(file_path: Path) -> str:
        """Return standardized output filename with .png extension."""
        return file_path.stem + ".png"

    def extract_label(self) -> None:
        """Process input directory and extract label/macro regions.

        Notes:
            - PNG files are cropped/rotated to simulate label region.
            - WSI files: prefer 'label' associated image; otherwise crop 'macro'.
            - Unsupported or failed files are logged.
        """
        processed = 0
        if not self.input_dir.exists():
            self.log_failure(f"Input folder not found - {self.input_dir}")
            LOGGER.info("[SUMMARY] Total processed successfully: %d", processed)
            return

        for file in self.input_dir.iterdir():
            if not file.is_file():
                continue

            filename = file.name
            suffix = file.suffix.lower()

            # ----------------------------- PNG path ----------------------------
            if suffix == ".png":
                try:
                    img = Image.open(file)
                    width, height = img.size
                    crop_w = int(width * self.label_crop_ratio)
                    cropped = img.crop((0, 0, crop_w, height))
                    rotated = cropped.rotate(self.rotation_degrees, expand=True)
                    output_path = self.output_dir / self.generate_output_name(file)
                    atomic_save_pil_image(rotated, output_path)
                    processed += 1
                except Exception as exc:
                    self.log_failure(f"PNG error - {filename}: {exc}")
                continue

            # ----------------------------- WSI path ----------------------------
            if not self.is_supported_wsi(file):
                self.log_failure(f"Unsupported WSI - {filename}")
                continue

            try:
                slide = openslide.OpenSlide(str(file))

                # 1) Try 'label' associated image
                if "label" in slide.associated_images:
                    label_img = slide.associated_images["label"]
                    output_path = self.output_dir / self.generate_output_name(file)
                    atomic_save_pil_image(label_img, output_path)
                    processed += 1
                    continue

                # 2) Try 'macro' (case-insensitive)
                macro_key = next((k for k in slide.associated_images if k.lower() == self.macro_tag), None)
                if macro_key:
                    macro = slide.associated_images[macro_key]
                    width, height = macro.size
                    crop_w = int(width * self.label_crop_ratio)
                    cropped = macro.crop((0, 0, crop_w, height))
                    rotated = cropped.rotate(self.rotation_degrees, expand=True)
                    output_path = self.output_dir / self.generate_output_name(file)
                    atomic_save_pil_image(rotated, output_path)
                    processed += 1
                else:
                    self.log_failure(f"No label or macro found in WSI - {filename}")

            except Exception as exc:
                self.log_failure(f"WSI error - {filename}: {exc}")

        LOGGER.info("[SUMMARY] Total processed successfully: %d", processed)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_paths_from_config(config: Dict[str, Any]) -> Paths:
    """Construct key paths from config, creating required folders."""
    timestamp = config.get("run_timestamp") or datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_base = Path(config["run_output_dir"])
    output_run_dir = Path(config.get("output_run_dir", output_base / timestamp))
    output_run_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = output_run_dir / f"label_failed_{timestamp}.log"
    input_dir = Path(config["staging_dir"])
    output_dir = Path(config["label_crops_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    return Paths(
        input_dir=input_dir,
        output_dir=output_dir,
        output_run_dir=output_run_dir,
        log_file_path=log_file_path,
    )


def run_extraction(config_path: Path, openslide_dll: Optional[str], log_level: str) -> None:
    """Entry point for `run` subcommand (original behavior)."""
    setup_logging(log_level)
    config = load_config(config_path)

    paths = build_paths_from_config(config)

    # Read options with same defaults as original
    label_crop_ratio = float(config.get("label_crop_ratio", 0.3))
    macro_tag = str(config.get("wsi_macro_img_tag", "macro")).lower()
    label_rotation_degrees = int(config.get("label_rotation_degrees", -90))

    # Windows DLL setup (same logic)
    prepare_windows_dlls(openslide_dll or (config.get("dll_paths", {}) or {}).get("openslide_dll", None))

    # Import openslide now (AFTER DLL prep)
    global openslide
    openslide = _load_openslide_after_dll_prep()

    extractor = LabelExtractor(
        input_dir=paths.input_dir,
        output_dir=paths.output_dir,
        log_file_path=paths.log_file_path,
        label_crop_ratio=label_crop_ratio,
        rotation_degrees=label_rotation_degrees,
        macro_tag=macro_tag,
    )
    extractor.extract_label()


def validate_config(config_path: Path, openslide_dll: Optional[str], log_level: str) -> int:
    """Lightweight validation of config and environment (no behavior change for `run`)."""
    setup_logging(log_level)
    try:
        config = load_config(config_path)
    except Exception as exc:
        LOGGER.error("[FAIL] Cannot read config: %s", exc)
        return 1

    required_keys = ["run_output_dir", "staging_dir", "label_crops_dir"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        LOGGER.error("[FAIL] Missing required config keys: %s", ", ".join(missing))
        return 2

    # Check folders existence where applicable
    src = Path(config["staging_dir"])
    if not src.exists():
        LOGGER.warning("[WARNING] Source folder does not exist yet: %s", src)

    # Attempt DLL prep if specified
    try:
        prepare_windows_dlls(openslide_dll or (config.get("dll_paths", {}) or {}).get("openslide_dll", None))
    except Exception as exc:
        LOGGER.warning("[WARNING] DLL preparation issue: %s", exc)

    LOGGER.info("Config validation passed.")
    return 0

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Create top-level argparse parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Extract label regions from WSI/PNG slides (structured CLI)."
    )
    parser.set_defaults(func=None)

    # Common options
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = subparsers.add_parser("run", help="Run extraction (original behavior).")
    p_run.add_argument("--config", type=Path, required=True, help="Path to YAML configuration file.")
    p_run.add_argument("--openslide_dll", type=str, required=False,
                       help="Path to OpenSlide DLL directory (Windows only).")
    p_run.set_defaults(func=lambda ns: run_extraction(ns.config, ns.openslide_dll, ns.log_level))

    # validate
    p_val = subparsers.add_parser("validate", help="Validate config and environment.")
    p_val.add_argument("--config", type=Path, required=True, help="Path to YAML configuration file.")
    p_val.add_argument("--openslide_dll", type=str, required=False,
                       help="Path to OpenSlide DLL directory (Windows only).")
    p_val.set_defaults(func=lambda ns: sys.exit(validate_config(ns.config, ns.openslide_dll, ns.log_level)))

    # version
    p_ver = subparsers.add_parser("version", help="Show module version.")
    p_ver.set_defaults(func=lambda ns: (setup_logging(ns.log_level), LOGGER.info(__VERSION__)))

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry-point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Dispatch
    if hasattr(args, "func") and args.func is not None:
        args.func(args) if callable(args.func) else None
    else:
        parser.print_help()

# --------------------------------------------------------------------------- #
# Script execution guard
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
