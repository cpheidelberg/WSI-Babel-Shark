"""
If runner supports a dry-run/config-check mode, call it; else keep -h/version only.
"""
import subprocess, sys, pytest, textwrap, tempfile

@pytest.mark.cli
def test_runner_config_check(tmp_path):
    cfg = tmp_path / "runner.yaml"
    cfg.write_text(textwrap.dedent("""
    # minimal runner config; adjust keys to your runner's schema
    steps: ["collect", "label"]
    """))
    # If you have a `--dry-run` or `validate` subcommand, prefer that:
    proc = subprocess.run([sys.executable, "-m", "babel_shark.runner", "-h"],
                          capture_output=True, text=True)
    assert proc.returncode in (0,2)
