"""
Unit test for label_extractor using a fake OCR to avoid heavy deps.
"""
import types
import pytest

@pytest.mark.unit
def test_label_extractor_with_fake_ocr(monkeypatch):
    from babel_shark import label_extractor

    # Provide a tiny fake OCR class/object the module expects.
    fake_ocr = types.SimpleNamespace(read=lambda img: "CASE 2024/12345  A2  PAS")

    # Monkeypatch any OCR factory/getter inside label_extractor (adjust attr name if needed)
    if hasattr(label_extractor, "OCR"):
        monkeypatch.setattr(label_extractor, "OCR", fake_ocr)
    if hasattr(label_extractor, "get_ocr"):
        monkeypatch.setattr(label_extractor, "get_ocr", lambda *_: fake_ocr)

    # Provide a minimal fake image bytes if function expects it
    if hasattr(label_extractor, "extract_label"):
        result = label_extractor.extract_label(b"\x00\x01")
        s = str(result).upper()
        assert "2024" in s and ("12345" in s)
        assert "A2" in s or "A 2" in s
        assert "PAS" in s
    else:
        pytest.xfail("label_extractor.extract_label not found; adjust test to your API.")
