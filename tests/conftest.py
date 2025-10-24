"""
Ensure 'src/' is importable during tests without requiring an installed package.
This keeps tests simple and fast in CI and locally.
"""
import sys
from pathlib import Path

# Compute the absolute path to the project's 'src' directory
SRC = Path(__file__).resolve().parents[1] / "src"

# Prepend 'src' to sys.path if it's not already present
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
