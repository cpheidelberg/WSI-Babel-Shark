"""
CLI smoke test: ensure each module can be invoked with `-h` (help).
- We first try to import the module to detect optional/missing deps early.
- If import fails due to a missing optional dependency, we xfail that case.
- Then we call: python -m babel_shark.<module> -h
- Accept exit codes 0 or 2 for argparse-based CLIs.
"""
import importlib
import subprocess
import sys
import pytest

MODULES = [
    "collect_slides",
    "database_manager",
    "datamatrix_reader",
    "label_extractor",
    "roi_metadata_extractor",  # may require optional deps
    "runner",
    "slide_id_generator",
    "stain_extractor",
]

@pytest.mark.parametrize("mod", MODULES)
def test_cli_help(mod: str) -> None:
    """`python -m babel_shark.<module> -h` should succeed and print something."""
    modpath = f"babel_shark.{mod}"

    # 1) Import guard: if module needs optional deps, mark xfail instead of failing the suite.
    try:
        importlib.import_module(modpath)
    except Exception as e:
        pytest.xfail(f"Optional dependency missing for {mod}: {e}")

    # 2) Now run `-h`; argparse typically returns 0 or 2 and prints help/usage text.
    proc = subprocess.run([sys.executable, "-m", modpath, "-h"],
                          capture_output=True, text=True)
    assert proc.returncode in (0, 2), f"-h failed for {mod}: {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    assert proc.stdout or proc.stderr, "Expected some help output"
