"""
CLI-level tests for the Babel-Shark pipeline runner.

These tests focus on *lightweight* checks of the runner's command-line
interface, without invoking the full WSI processing pipeline:

1. `python -m babel_shark.runner -h`
   - Verifies that the help/usage screen can be displayed
     and the module is importable as a CLI entry point.

2. `python -m babel_shark.runner validate --config <yaml>`
   - Uses a minimal but structurally valid YAML configuration
     to exercise the built-in config validation logic.
   - Confirms that required top-level keys and script paths
     are recognized and that a valid configuration returns
     an exit code of 0.

3. A negative validation scenario:
   - A deliberately incomplete config (missing keys) should
     produce a non-zero exit code and informative error messages.

These tests are designed to be fast and deterministic and should not
launch any of the heavy child scripts or long-running pipeline steps.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helper: create a minimal, valid runner configuration on disk
# ---------------------------------------------------------------------------

def _write_minimal_valid_config(tmp_path: Path) -> Path:
    """
    Create a minimal YAML config that satisfies runner.validate() requirements.

    The runner expects the following top-level keys:
      - run_output_dir
      - label_root_dir
      - staging_dir
      - failed_output_dir
      - scripts
      - slide_id_generator
      - watch_dir

    In addition, `scripts` must contain paths for:
      - label_extractor
      - datamatrix_reader
      - stain_extractor
      - slide_id_creator

    For this test:
      - All directories and script files are created under tmp_path.
      - Script files are tiny Python stubs; only existence is checked
        by validate(), not executability.
    """
    base = tmp_path

    run_output_dir = base / "run_output"
    label_root_dir = base / "labels"
    staging_dir = base / "staging"
    failed_output_dir = base / "failed"
    watch_dir = base / "watch"
    final_output_dir = base / "final"

    # Create directory structure
    run_output_dir.mkdir(parents=True, exist_ok=True)
    label_root_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    failed_output_dir.mkdir(parents=True, exist_ok=True)
    watch_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Create stub scripts
    scripts_dir = base / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    label_stub = scripts_dir / "label_extractor_stub.py"
    datamatrix_stub = scripts_dir / "datamatrix_reader_stub.py"
    stain_stub = scripts_dir / "stain_extractor_stub.py"
    slideid_stub = scripts_dir / "slide_id_creator_stub.py"

    for stub in (label_stub, datamatrix_stub, stain_stub, slideid_stub):
        stub.write_text("# stub script for runner.validate()\n", encoding="utf-8")

    cfg_path = base / "runner_config.yaml"

    cfg_text = textwrap.dedent(
        f"""
        run_output_dir: '{run_output_dir.as_posix()}'
        label_root_dir: '{label_root_dir.as_posix()}'
        staging_dir: '{staging_dir.as_posix()}'
        failed_output_dir: '{failed_output_dir.as_posix()}'
        watch_dir: '{watch_dir.as_posix()}'

        slide_id_generator:
          final_output_dir: '{final_output_dir.as_posix()}'

        scripts:
          label_extractor: '{label_stub.as_posix()}'
          datamatrix_reader: '{datamatrix_stub.as_posix()}'
          stain_extractor: '{stain_stub.as_posix()}'
          slide_id_creator: '{slideid_stub.as_posix()}'
        """
    ).strip()

    cfg_path.write_text(cfg_text + "\n", encoding="utf-8")
    return cfg_path


# ---------------------------------------------------------------------------
# Test 1: basic help screen should be available
# ---------------------------------------------------------------------------

def test_runner_help_succeeds():
    """
    `python -m babel_shark.runner -h` should succeed and print a usage/help screen.

    We accept both exit code 0 (normal) and 2 (argparse "help" behavior),
    but we require that some usage text appears on stdout.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "babel_shark.runner", "-h"],
        capture_output=True,
        text=True,
    )

    assert proc.returncode in (0, 2)
    stdout_lower = (proc.stdout or "").lower()
    assert "usage" in stdout_lower or "pipeline runner" in stdout_lower


# ---------------------------------------------------------------------------
# Test 2: validate should accept a minimal valid configuration
# ---------------------------------------------------------------------------

def test_runner_validate_accepts_minimal_config(tmp_path):
    """
    Given a syntactically and structurally valid runner config,
    the `validate` subcommand should return exit code 0 and emit
    a confirmation message.
    """
    cfg = _write_minimal_valid_config(tmp_path)

    proc = subprocess.run(
        [sys.executable, "-m", "babel_shark.runner", "validate", "--config", str(cfg)],
        capture_output=True,
        text=True,
    )

    # validate() uses 0 for success, 1/2 for various error conditions.
    assert proc.returncode == 0

    # The validation function logs a success message on stdout.
    stdout = proc.stdout or ""
    assert "[OK] Config validation successful." in stdout


# ---------------------------------------------------------------------------
# Test 3: validate should fail on incomplete configuration
# ---------------------------------------------------------------------------

def test_runner_validate_fails_on_missing_keys(tmp_path):
    """
    A deliberately incomplete config (missing required top-level keys)
    should cause the `validate` subcommand to return a non-zero exit code
    and print informative error messages.

    Here we only provide run_output_dir and omit scripts/label_root_dir/etc.
    """
    bad_cfg = tmp_path / "bad_runner_config.yaml"
    bad_cfg.write_text("run_output_dir: './somewhere'\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "babel_shark.runner", "validate", "--config", str(bad_cfg)],
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0

    stdout = proc.stdout or ""
    # At least one missing-key message should appear.
    assert "Missing required config key:" in stdout

