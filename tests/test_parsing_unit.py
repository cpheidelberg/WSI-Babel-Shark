"""
Unit tests for parsing and heuristic logic used in metadata_extractor_utilities.

We focus on small, deterministic functions without any file I/O:
- Case number parsing (LabID, CaseNumber, Year)
- Pot / block parsing
- Stain parsing
- Basic text normalization helpers (uppercase, year, case number)
"""

import pytest

from babel_shark.metadata_extractor_utilities.parsing.case_pot_stain import (
    parse_casenumber,
    parse_pot_block,
    parse_stain,
)

from babel_shark.metadata_extractor_utilities.parsing.text_utils import (
    sanitize_upper,
    to_year4_str,
    to_case6_str,
)


# ---------------------------------------------------------------------------
# Case number parsing
# ---------------------------------------------------------------------------

def test_parse_casenumber_basic():
    """
    parse_casenumber() should extract LabID, CaseNumber, and Year
    from a simple OCR-like snippet.

    Example:
        "Case: e 123/24" → LabID="E", CaseNumber="000123", Year="2024"
    """
    text = "Case: e 123/24"
    result = parse_casenumber(text)

    # We expect a full triplet
    assert isinstance(result, dict), "parse_casenumber should return a dict."
    assert result, "parse_casenumber returned an empty dict."

    assert result["LabID"] == "E"
    assert result["CaseNumber"] == "000123"
    assert result["Year"] == "2024"


def test_parse_casenumber_without_lab_defaults_to_E():
    """
    When no LabID letter is present in the pattern, parse_casenumber()
    should still parse the case and default LabID to 'E'.
    """
    text = "012345/27"
    result = parse_casenumber(text)

    assert result["LabID"] == "E"
    assert result["CaseNumber"] == "012345"
    assert result["Year"] == "2027"


# ---------------------------------------------------------------------------
# Pot / block parsing
# ---------------------------------------------------------------------------

def test_parse_pot_block_simple():
    """
    parse_pot_block() should extract pot letter and block number from
    a simple mixed string.
    """
    text = "Pot: B2"
    pot, block = parse_pot_block(text)

    assert pot == "B"
    assert block == "2"


def test_parse_pot_block_defaults_on_empty():
    """
    Empty or whitespace-only input should fall back to ('A', '1').
    """
    pot, block = parse_pot_block("   ")
    assert pot == "A"
    assert block == "1"


# ---------------------------------------------------------------------------
# Stain parsing
# ---------------------------------------------------------------------------

STAIN_DICT = {
    "h&e": "H&E",
    "pas": "PAS",
    "giemsa": "GIEMSA",
}

# For these tests we do not need a complex replacement table;
# the parser’s own heuristics and STAIN_DICT are sufficient.
STAIN_REPL = {}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Stain: H&E", "H&E"),        # explicit H&E should normalize to 'H&E'
        ("染色: PAS", "PAS"),          # mixed language + PAS
        ("Giemsa stain", "GIEMSA"),   # free text with 'Giemsa'
    ],
)
def test_parse_stain_simple(text: str, expected: str):
    """
    parse_stain() should map various noisy inputs to canonical stain labels.
    """
    stain = parse_stain(text, STAIN_DICT, STAIN_REPL, fuzzy_threshold=None)
    assert stain == expected


# ---------------------------------------------------------------------------
# Text utils: normalization and formatting
# ---------------------------------------------------------------------------

def test_sanitize_upper_normalizes_text():
    """
    sanitize_upper() should:
      - Uppercase
      - Normalize backslash and hyphen to '/'
      - Collapse whitespace
      - Replace O→0, I/L→1
    """
    raw = "  e 123\\45 -Oil "
    out = sanitize_upper(raw)

    # Manually computed expectation based on the implementation:
    #   1) to upper: "  E 123\45 -OIL "
    #   2) "\" -> "/", "-" -> "/": "  E 123/45 /OIL "
    #   3) collapse spaces: "E 123/45 /OIL"
    #   4) O→0, I→1, L→1: "E 123/45 /011"
    assert out == "E 123/45 /011"


def test_to_year4_str_and_to_case6_str():
    """
    to_year4_str() should normalize 2-digit and 4-digit years.
    to_case6_str() should zero-pad case numbers to 6 characters.
    """
    # Year normalization
    assert to_year4_str("24") == "2024"
    assert to_year4_str(27) == "2027"
    assert to_year4_str("2025") == "2025"
    assert to_year4_str("abc") is None

    # Case number padding
    assert to_case6_str(123) == "000123"
    assert to_case6_str("45") == "000045"
    assert to_case6_str("001234") == "001234"
    assert to_case6_str("x") is None
