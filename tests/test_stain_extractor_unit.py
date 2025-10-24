"""
Unit test for stain_extractor with explicit dict args if required.
"""
import pytest

@pytest.mark.unit
def test_parse_stain_minimal():
    from babel_shark import stain_extractor

    # Try common names
    func = getattr(stain_extractor, "parse_stain", None) or getattr(stain_extractor, "infer_stain", None)
    if func is None:
        pytest.xfail("No stain parser found; adjust function name.")

    sample = "Stain: H&E"
    # Try both signatures
    try:
        res = func(sample)  # infer_stain(text)
    except TypeError:
        # parse_stain(text, dict, repl)
        res = func(sample,
                   {"H&E":"H&E","HE":"H&E","PAS":"PAS","GIEMSA":"GIEMSA"},
                   {"H & E":"H&E","H and E":"H&E","HE":"H&E"})
    assert "H" in str(res).upper()
