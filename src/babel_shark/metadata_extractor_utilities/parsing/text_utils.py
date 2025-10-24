# -*- coding: utf-8 -*-
"""Text normalization and formatting helpers for label parsing.

This module provides small utilities for:
  - Uppercasing/sanitizing noisy OCR text
  - Digit-only extraction and integer string coercion
  - Year normalization to 4 digits with range checks
  - Case number zero-padding to width 6
  - Building a compact DataMatrix string from structured fields
"""

import re
from typing import Optional

def sanitize_upper(text: Optional[str]) -> str:
    """Uppercase and normalize a text snippet for downstream parsing.

    Operations:
      - None → "".
      - Cast to str, convert to UPPER.
      - Replace backslash and hyphen with forward slash.
      - Collapse consecutive whitespace to single space and strip ends.
      - Heuristically fix common OCR confusions: O→0, I/L→1.

    Args:
        text: Raw input string or None.

    Returns:
        A normalized uppercase string safe for regex parsing.
    """
    if text is None:
        return ""
    t = str(text).upper()
    t = t.replace("\\", "/").replace("-", "/")
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace("O", "0").replace("I", "1").replace("L", "1")
    return t

def only_digits_str(x: object) -> str:
    """Return a string of digits extracted from the input.

    Args:
        x: Any object; converted to string before processing.

    Returns:
        String containing only [0-9] characters (may be empty).
    """
    return re.sub(r"[^\d]", "", str(x) if x is not None else "")

def to_int_str_no_decimal(x: object) -> Optional[str]:
    """Convert numeric-like input to an integer string without decimals.

    Accepts forms like "123", "123.0", "0012.000" and returns "123", "12".
    If not strictly integer-ish, falls back to extracting digits; returns None
    when no digits are present.

    Args:
        x: Input value to coerce.

    Returns:
        Integer string without leading zeros (unless zero itself), or None.
    """
    if x is None:
        return None
    s = str(x).strip()
    m = re.fullmatch(r"\s*([0-9]+)(?:\.0+)?\s*", s)
    return str(int(m.group(1))) if m else (only_digits_str(s) or None)

def to_year4_str(y: object) -> Optional[str]:
    """Normalize a year to 4-digit string.

    Rules:
      - "24" → "2024"
      - "2027" → "2027"
      - Anything else → None

    Args:
        y: Year-like input.

    Returns:
        4-digit year as string, or None if not parseable.
    """
    s = to_int_str_no_decimal(y)
    if not s:
        return None
    if re.fullmatch(r"[0-9]{2}", s):
        return f"20{s}"
    if re.fullmatch(r"[0-9]{4}", s):
        return s
    return None

def is_year_in_valid_range(y4: Optional[str]) -> bool:
    """Check if a 4-digit year string is within the accepted range.

    Current accepted inclusive range: 2020–2030.

    Args:
        y4: 4-digit year string.

    Returns:
        True if valid and in-range; False otherwise.
    """
    if not y4 or not re.fullmatch(r"[0-9]{4}", y4):
        return False
    yi = int(y4)
    return 2020 <= yi <= 2030

def to_case6_str(c: object) -> Optional[str]:
    """Zero-pad a case number to width 6.

    Examples:
      - 123 → "000123"
      - "45" → "000045"

    Args:
        c: Case number-like input.

    Returns:
        6-character zero-padded string, or None if not parseable.
    """
    s = to_int_str_no_decimal(c)
    if not s:
        return None
    return f"{int(s):06d}"

def build_datamatrix(lab: object,
                     year: object,
                     case: object,
                     pot: object,
                     blockid: object,
                     section: object) -> Optional[str]:
    """Compose a compact DataMatrix string from label fields.

    Format:
        "<LAB><YYYY><CASE6>S<POT>-<BLOCKID>-<SECTION>"

    Notes:
      - `LAB` and `POT` are uppercased strings.
      - `YYYY` from `to_year4_str(year)`.
      - `CASE6` from `to_case6_str(case)`.
      - `BLOCKID` and `SECTION` coerced via `to_int_str_no_decimal`, falling back to
        stripped strings when present but not pure digits.
      - Returns None if any required piece is missing/empty.

    Args:
        lab: Laboratory ID (string-like).
        year: Year (2- or 4-digit).
        case: Case number (will be zero-padded to 6).
        pot: Pot letter/label.
        blockid: Block identifier.
        section: Section identifier.

    Returns:
        Assembled DataMatrix string or None if incomplete.
    """
    lab_s = (str(lab).strip().upper() if lab else "")
    from .text_utils import to_year4_str as _to_year4_str, to_case6_str as _to_case6_str, to_int_str_no_decimal as _to_int_str
    y4 = _to_year4_str(year)
    c6 = _to_case6_str(case)
    pot_s = (str(pot).strip().upper() if pot else "")
    bid = _to_int_str(blockid) or (str(blockid).strip() if blockid is not None else "")
    sec = _to_int_str(section) or (str(section).strip() if section is not None else "")
    if not (lab_s and y4 and c6 and pot_s and bid and sec):
        return None
    return f"{lab_s}{y4}{c6}S{pot_s}-{bid}-{sec}"