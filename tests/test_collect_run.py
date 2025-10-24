"""
Lightweight functional test for `collect_slides`:
- Creates temporary watch/stage directories
- Writes a tiny fake .svs file into watch
- Generates a minimal YAML config on-the-fly
- Runs `validate` then `run` and asserts the file appears in staging
"""

import subprocess
import sys
import time
from pathlib import Path


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Helper to run a subprocess with text I/O and a reasonable timeout."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=90)


def test_collect_validate_and_run(tmp_path: Path) -> None:
    # 1) Prepare temp watch/stage folders
    watch = tmp_path / "watch"
    stage = tmp_path / "stage"
    watch.mkdir()
    stage.mkdir()

    # 2) Put a tiny sentinel WSI file in watch
    (watch / "sample.svs").write_bytes(b"x")

    # 3) Create a minimal YAML config (adjust keys if your CLI expects different names)
    cfg = tmp_path / "collect.yaml"
    cfg.write_text(
        "\n".join(
            [
                f'watch_dir: "{watch.as_posix()}"',
                f'staging_dir: "{stage.as_posix()}"',
                "wsi_types: ['.svs']",
                'operation_mode: "copy"',  # or "move" if supported
            ]
        )
    )

    # 4) Run `validate` subcommand
    validate_cmd = [
        sys.executable,
        "-m",
        "babel_shark.collect_slides",
        "validate",
        "--config",
        str(cfg),
    ]
    p_val = _run(validate_cmd)
    assert (
        p_val.returncode == 0
    ), f"validate failed\nSTDOUT:\n{p_val.stdout}\nSTDERR:\n{p_val.stderr}"

    # 5) Run `run` subcommand (single pass expected to copy one file)
    run_cmd = [
        sys.executable,
        "-m",
        "babel_shark.collect_slides",
        "run",
        "--config",
        str(cfg),
    ]
    p_run = _run(run_cmd)
    # Some CLIs might continue watching; here we assert a clean return code.
    assert (
        p_run.returncode == 0
    ), f"run failed\nSTDOUT:\n{p_run.stdout}\nSTDERR:\n{p_run.stderr}"

    # 6) Give a brief moment if the tool processes asynchronously, then assert staged file exists
    time.sleep(0.2)
    staged = list(stage.glob("*.svs"))
    assert staged, "No .svs file appeared in staging after `run`"
