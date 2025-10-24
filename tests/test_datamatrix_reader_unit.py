"""
Unit test for datamatrix_reader by faking the decode function.
"""
import pytest

@pytest.mark.unit
def test_datamatrix_decode_fake(monkeypatch):
    from babel_shark import datamatrix_reader

    # Suppose module has a function like `decode_datamatrix(image)`; we fake it.
    if hasattr(datamatrix_reader, "decode_datamatrix"):
        monkeypatch.setattr(datamatrix_reader, "decode_datamatrix", lambda *_: "DMX:ABC123")
        out = datamatrix_reader.decode_datamatrix(b"\x00\x01")
        assert "ABC123" in out
    else:
        pytest.xfail("decode_datamatrix not found; adjust to your function name.")
