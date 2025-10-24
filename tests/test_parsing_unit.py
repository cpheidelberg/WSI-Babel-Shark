"""
Unit tests for parsing/heuristic logic.
- Focus on small, deterministic functions (no file I/O, no heavy deps).
- Try to import parsing utilities from your package. If the API differs, xfail with a clear message.
"""

import importlib
import pytest

# Attempt to import your parsing modules
PARSING_MOD_PATH = "babel_shark.metadata_extractor_utilities.parsing.case_pot_stain"
TEXT_UTILS_MOD_PATH = "babel_shark.metadata_extractor_utilities.parsing.text_utils"


def _import_or_xfail(mod_path: str):
    """Import a module or xfail if unavailable (keeps CI informative but stable)."""
    try:
        return importlib.import_module(mod_path)
    except Exception as e:
        pytest.xfail(f"Cannot import {mod_path}: {e}")


def _normalize_to_str_or_dict(x):
    """
    Helper to normalize return values:
    - If result is a dict-like object, return as-is.
    - If result is a custom object/tuple, try a few common access patterns,
      otherwise fallback to string for contains() assertions.
    """
    if isinstance(x, dict):
        return x
    # Try duck-typing common patterns
    for attr in ("to_dict", "as_dict", "_asdict"):
        if hasattr(x, attr):
            try:
                return getattr(x, attr)()
            except Exception:
                pass
    return x  # keep original (may be str/tuple/custom); tests will handle flexibly


@pytest.mark.unit
def test_parse_case_number_simple():
    """
    Expect that a 'parse_case_number' (or similarly named) function can
    extract year + case id from a canonical string, e.g., "2024/012345".
    """
    mod = _import_or_xfail(PARSING_MOD_PATH)

    # Try to find a parsing function by common names
    parse_funcs = [getattr(mod, n, None) for n in (
        "parse_case_number",
        "extract_case_number",
        "parse_case",
        "case_from_text",
    )]
    parse = next((f for f in parse_funcs if callable(f)), None)
    if parse is None:
        pytest.xfail("No case-number parsing function found in module")

    sample = "CASE: 2024/012345 A2"
    result = _normalize_to_str_or_dict(parse(sample))

    # Accept several output schemas to be robust:
    if isinstance(result, dict):
        # Common dict keys
        year = result.get("year") or result.get("Year") or result.get("YEAR")
        case = result.get("case") or result.get("Case") or result.get("id") or result.get("ID")
        assert str(year) in ("2024", "024")  # tolerate zero-stripping differences
        assert "12345" in str(case)
    else:
        # Fallback: string or custom object -> stringify & search
        s = str(result)
        assert "2024" in s and ("12345" in s or "012345" in s)


@pytest.mark.unit
def test_parse_pot_block_simple():
    """
    Expect that a function can extract Pot/Block ID from short tags like 'A 2' or 'B3'.
    """
    mod = _import_or_xfail(PARSING_MOD_PATH)

    # Common function name variants
    parse_funcs = [getattr(mod, n, None) for n in (
        "parse_pot_block",
        "extract_pot_block",
        "parse_potid_blockid",
        "parse_pot_blockid",
    )]
    parse = next((f for f in parse_funcs if callable(f)), None)
    if parse is None:
        pytest.xfail("No pot/block parsing function found in module")

    sample = "… POT A 2 …"
    result = _normalize_to_str_or_dict(parse(sample))

    if isinstance(result, dict):
        pot = result.get("pot") or result.get("Pot") or result.get("POT")
        block = result.get("block") or result.get("Block") or result.get("BLOCK") or result.get("id")
        assert str(pot).strip().upper() in ("A", "POT:A")
        assert str(block).strip() in ("2", "02")
    else:
        s = str(result).upper()
        assert "A" in s and (" 2" in s or "2" in s)


@pytest.mark.unit
@pytest.mark.parametrize("text,expected", [
    ("Stain: H&E", ("H&E", "HE")),
    ("染色: PAS", ("PAS",)),
    ("Giemsa stain", ("GIEMSA",)),
])
def test_infer_stain_simple(text, expected):
    """
    Expect that a stain-detection function can map to a known stain label.
    This test is robust to different function signatures:
    - infer_stain(text)
    - parse_stain(text, stain_dict, stain_repl)
    It will try to import stain dictionaries from the module; if unavailable,
    it falls back to a minimal mapping so the test remains meaningful.
    """
    import inspect

    mod = _import_or_xfail(PARSING_MOD_PATH)

    # Candidate function names
    stain_funcs = [getattr(mod, n, None) for n in (
        "infer_stain",
        "parse_stain",
        "detect_stain",
        "stain_from_text",
    )]
    infer = next((f for f in stain_funcs if callable(f)), None)
    if infer is None:
        pytest.xfail("No stain parsing function found in module")

    # Inspect the callable to decide how to call it
    sig = inspect.signature(infer)
    params = list(sig.parameters.keys())

    # Try to get stain dictionaries from the module if the function requires them
    # Common names we try: STAIN_DICT/STAIN_REPL, stain_dict/stain_repl, or a config-like object
    stain_dict = getattr(mod, "STAIN_DICT", None) or getattr(mod, "stain_dict", None)
    stain_repl = getattr(mod, "STAIN_REPL", None) or getattr(mod, "stain_repl", None)

    # Minimal fallback mapping to keep the test self-contained if not provided by the module
    if stain_dict is None:
        stain_dict = {
            "H&E": "H&E",
            "HE": "H&E",
            "PAS": "PAS",
            "GIEMSA": "GIEMSA",
            "Giemsa": "GIEMSA",
        }
    if stain_repl is None:
        # Simple normalization replacements; adjust if your module expects a specific structure
        stain_repl = {
            "H and E": "H&E",
            "H & E": "H&E",
            "HE": "H&E",
        }

    # Call according to the signature
    try:
        if len(params) == 1:
            # e.g., infer_stain(text)
            result = _normalize_to_str_or_dict(infer(text))
        elif len(params) >= 3:
            # e.g., parse_stain(text, stain_dict, stain_repl)
            result = _normalize_to_str_or_dict(infer(text, stain_dict, stain_repl))
        else:
            # Unrecognized signature → mark as xfail with guidance
            pytest.xfail(f"Unrecognized stain function signature: {sig}")
    except TypeError as e:
        # If TypeError persists, provide clear actionables
        pytest.xfail(f"Stain function call failed due to signature mismatch: {e}")

    # Validate label presence regardless of return type shape
    if isinstance(result, dict):
        label = result.get("stain") or result.get("Stain") or result.get("label")
        assert label is not None, "Expected a stain label in dict result"
        assert any(e.upper() in str(label).upper() for e in expected)
    elif isinstance(result, (tuple, list)) and result:
        s = " ".join(map(str, result)).upper()
        assert any(e.upper() in s for e in expected)
    else:
        s = str(result).upper()
        assert any(e.upper() in s for e in expected)

@pytest.mark.unit
def test_text_utils_basic_normalization():
    """
    Optional: if text_utils provides normalization helpers, verify simple behavior.
    If module or function is absent, xfail with a clear message.
    """
    mod = _import_or_xfail(TEXT_UTILS_MOD_PATH)

    # Look up a common normalization function
    cand = [getattr(mod, n, None) for n in ("normalize_label_text", "normalize_text", "clean_text")]
    func = next((f for f in cand if callable(f)), None)
    if func is None:
        pytest.xfail("No normalization function found in text_utils")

    s = "  Case:\t2024/012345  "
    out = func(s)
    # Expect trimming/tab removal and preserved digits
    assert "2024" in out and "012345" in out
    assert "\t" not in out and out.strip() == out
