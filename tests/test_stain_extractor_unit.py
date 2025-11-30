"""
Unit tests for core stain parsing logic in babel_shark.stain_extractor.

We do NOT call OCR or ROI or the full pipeline here.
Instead, we test the pure helper functions that implement the stain
normalization and composite detection rules:
    - tokenize_raw_words
    - apply_replacements
    - map_to_canonical
    - detect_plus_pair
    - detect_two_valid_stains
    - find_exact_stain
    - is_exact_he
"""

from babel_shark.stain_extractor import (
    tokenize_raw_words,
    apply_replacements,
    map_to_canonical,
    detect_plus_pair,
    detect_two_valid_stains,
    find_exact_stain,
    is_exact_he,
)

# Small canonical dictionary similar to what would be loaded from stain_list.json
STAIN_DICT = {
    "h&e": "H&E",
    "pas": "PAS",
    "giemsa": "GIEMSA",
}

# Replacement map similar to stain_error_replacements.json
REPLACEMENTS = {
    "H & E": "H&E",
    "H and E": "H&E",
    "HE": "H&E",
}


def test_tokenize_raw_words_splits_and_cleans():
    """
    tokenize_raw_words should:
      - strip whitespace
      - split on spaces
      - keep '*' parts separately
    """
    raw = ["  Stain:  H&E*PAS  "]
    tokens = tokenize_raw_words(raw)

    # We mainly care that we get multiple cleaned tokens and no empty strings.
    assert len(tokens) >= 2
    assert all(t.strip() for t in tokens)


def test_apply_replacements_normalizes_he_variants():
    """
    apply_replacements should map different H&E spellings to canonical 'H&E'
    using a case-insensitive lookup.
    """
    tokens = ["H", "&", "E", "H & E", "H and E", "he"]
    fixed = apply_replacements(tokens, REPLACEMENTS)

    # Compound / verbose forms should be normalized to 'H&E'
    assert "H&E" in fixed

    # 'he' is also in the replacement map (via 'HE') and should become 'H&E',
    # so it must not appear as raw 'he' anymore.
    assert all(t.lower() != "he" for t in fixed)


def test_map_to_canonical_uses_dict_case_insensitive():
    """
    map_to_canonical should map tokens to canonical stain names using STAIN_DICT
    in a case-insensitive fashion, and also special-case 'HE' / 'HE+' → 'HE'.
    """
    assert map_to_canonical("H&E", STAIN_DICT) == "H&E"
    assert map_to_canonical("h&e", STAIN_DICT) == "H&E"
    assert map_to_canonical("PAS", STAIN_DICT) == "PAS"
    assert map_to_canonical("pas", STAIN_DICT) == "PAS"
    # Special-case HE / HE+ → 'HE' regardless of dictionary contents
    assert map_to_canonical("HE", STAIN_DICT) == "HE"
    assert map_to_canonical("HE+", STAIN_DICT) == "HE"


def test_detect_plus_pair_composite_stain():
    """
    detect_plus_pair should detect composite stains of the form A + B
    and return 'A&B' in canonical form.
    """
    tokens = ["H&E", "+", "PAS"]
    canon, src = detect_plus_pair(tokens, STAIN_DICT)
    assert canon == "H&E&PAS"
    assert src == "H&E+PAS"


def test_detect_two_valid_stains_without_plus():
    """
    detect_two_valid_stains should detect when two different valid stains appear
    in the sequence even without an explicit '+' token and return 'A&B'.
    """
    tokens = ["H&E", "PAS", "something-else"]
    canon, src = detect_two_valid_stains(tokens, STAIN_DICT)
    assert canon == "H&E&PAS"
    assert src == "H&E+PAS"


def test_find_exact_stain_prefers_real_match_and_defaults_to_he():
    """
    find_exact_stain should:
      - return a directly matched stain if any token matches
      - otherwise default to 'H&E' with an empty matched token
    """
    # Case 1: explicit PAS present
    stain, matched = find_exact_stain(["xx", "PAS", "yy"], STAIN_DICT)
    assert stain == "PAS"
    assert matched.upper() == "PAS"

    # Case 2: nothing matches → default to 'H&E'
    stain2, matched2 = find_exact_stain(["xx", "yy"], STAIN_DICT)
    assert stain2 == "H&E"
    assert matched2 == ""


def test_is_exact_he_recognizes_variants():
    """
    is_exact_he should recognize exact 'H&E' (ignoring case/whitespace),
    and should NOT treat other variants like 'he' or 'H & E' as H&E.
    """
    # Expected True: exact H&E with case/whitespace variations
    assert is_exact_he("H&E")
    assert is_exact_he("h&e")
    assert is_exact_he(" H&E ")

    # Expected False: these variants are handled earlier by replacements,
    # but is_exact_he itself does not normalize them.
    assert not is_exact_he("he")
    assert not is_exact_he("HE")
    assert not is_exact_he("HE+")
    assert not is_exact_he("H & E")
    assert not is_exact_he("H and E")
    assert not is_exact_he("PAS")
    assert not is_exact_he("")
    assert not is_exact_he(None)

