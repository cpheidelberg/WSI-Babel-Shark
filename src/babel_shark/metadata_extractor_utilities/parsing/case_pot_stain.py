# -*- coding: utf-8 -*-
"""
Utilities for parsing case numbers, pot/block identifiers, and stain labels from
noisy OCR text in pathology slide workflows.

"""

import re
from typing import Dict, List, Optional, Tuple, Union
from .text_utils import sanitize_upper, to_year4_str, is_year_in_valid_range, to_int_str_no_decimal, to_case6_str

_CASE_NUM_CANON = re.compile(r"([A-Z])?\s*[0-9]{1,6}\s*/\s*[0-9]{2}\b")


def _infer_missing_slash_for_casenumber(s: str) -> str:
    """Infer a missing slash in a case number pattern.

    Heuristic rules:
      - If canonical pattern (e.g., '123/45' or 'E 123/45') already exists, return as-is.
      - Otherwise try to convert digit runs like '123145' → '123/45' when the inferred year is valid.
      - Preserves original text when inference is not confident.

    Args:
        s: Raw string possibly containing a case number.

    Returns:
        The input string with a best-effort slash insertion when appropriate; otherwise the original string.
    """
    if _CASE_NUM_CANON.search(s):
        return s
    pat = re.compile(r"(?P<prefix>\d{1,6})(?P<sep>[102])(?P<yy>\d{2})\b")

    def _repl(m: re.Match) -> str:
        prefix = m.group("prefix")
        yy = m.group("yy")
        y4 = to_year4_str(yy)
        if not is_year_in_valid_range(y4):
            return m.group(0)
        return f"{prefix}/{yy}"

    s2 = pat.sub(_repl, s)
    if _CASE_NUM_CANON.search(s2):
        return s2

    m = re.search(r"\d{6,}", s)
    if not m:
        return s
    digits = m.group(0)
    yy = digits[-2:]
    y4 = to_year4_str(yy)
    if not is_year_in_valid_range(y4):
        return s
    if len(digits) >= 3 and digits[-3] in {"1", "0", "2"}:
        prefix = digits[:-3]
        if 1 <= len(prefix) <= 6:
            return s[:m.start()] + f"{prefix}/{yy}" + s[m.end():]
    return s


def parse_casenumber(
    text: str, allowed_letters: Optional[Union[List[str], set]] = None
) -> Dict[str, str]:
    """Parse a case number triplet (LabID, CaseNumber, Year) from text.

    Strategy:
      - Uppercase & sanitize input, infer missing slash, then match '<Lab?> <digits>/<yy>'.
      - Choose the "best" match by (longest number, allowed letter, numeric value).
      - Default allowed Lab letters are {'E','R'} unless overridden.

    Args:
        text: Raw label text segment containing a case-like token.
        allowed_letters: Optional allowed single-letter LabIDs (case-insensitive).

    Returns:
        A dict with keys {'LabID','CaseNumber','Year'} or empty dict on failure.
    """
    if text is None:
        return {}
    t0 = sanitize_upper(text)
    t = _infer_missing_slash_for_casenumber(t0)
    pattern = r"([A-Z])?\s*([0-9]{1,6})(?=\s*/)\s*/\s*([0-9]{2})"
    matches = list(re.finditer(pattern, t))
    if not matches:
        return {}
    default_allowed = {"E", "R"}
    if allowed_letters is None:
        allowed = default_allowed
    else:
        allowed = {str(x).upper() for x in allowed_letters if isinstance(x, str)} or default_allowed

    best: Optional[Tuple[Tuple[int, int, int], str, str, int]] = None
    for m in matches:
        lab_ch = (m.group(1) or "").upper()
        num_str = m.group(2)
        yy = int(m.group(3))
        key = (len(num_str), 1 if lab_ch in allowed else 0, int(num_str))
        if (best is None) or (key > best[0]):
            best = (key, lab_ch, num_str, 2000 + yy)

    if best is None:
        return {}
    _, lab_ch, num_str, year4 = best
    lab = lab_ch if lab_ch in allowed else "E"
    return {"LabID": lab, "CaseNumber": f"{int(num_str):06d}", "Year": str(year4)}


def _extract_block_number_robust(s: str) -> Optional[str]:
    """Extract a plausible block number from noisy text.

    Rules:
      - Strip control/punct chars, keep digits, normalize spaces.
      - Prefer numbers in [1..12] when present; else fall back to last number.
      - Handles compact forms like '0123' → tokens [0,123] → returns in-range/latest.

    Args:
        s: Raw input possibly containing a block number.

    Returns:
        Extracted block number as string, or None if nothing plausible is found.
    """
    s0 = re.sub(r"[\u200e\u200f\u202a-\u202e]", " ", s)
    s0 = re.sub(r"[{}\[\]()<>::;,.~`'\"|\\/&+*%@#^_-]", " ", s0)
    s0 = re.sub(r"[^\d\s]", " ", s0)
    s0 = re.sub(r"\s+", " ", s0).strip()
    if not s0:
        return None
    nums = [int(x) for x in re.findall(r"\d{1,4}", s0)]
    if not nums:
        compact = s0.replace(" ", "")
        if re.fullmatch(r"0\d{2,3}", compact):
            no0 = compact[1:]
            if len(no0) >= 2:
                nums = [int(no0[0]), int(no0[1:])]
    if not nums:
        return None
    in_range = [n for n in nums if 1 <= n <= 12]
    return str(in_range[-1]) if in_range else str(nums[-1])


def __extract_pot_letter_strict(s: str) -> Optional[str]:
    """Extract a single-letter pot ID with strict constraints.

    Logic:
      - Clean brackets/joins, collapse spaces, drop lowercase.
      - Accept 'A', or patterns like 'A 2' / 'A2' (A returns).
      - Fallback to regex for a single A–Z possibly followed by 1–3 digits.

    Args:
        s: Pre-cleaned string segment.

    Returns:
        Uppercase pot letter if found; otherwise None.
    """
    s = re.sub(r"[{}\[\]<>]", "", s)
    s = re.sub(r"[\u200c\u200f\u200e]+", "", s)
    s = re.sub(r"[\\/_-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s2 = re.sub(r"[a-z]", "", s)
    tokens = s2.split()
    for i, tok in enumerate(tokens):
        if re.fullmatch(r"[A-Z]", tok):
            if i + 1 < len(tokens) and re.fullmatch(r"\d{1,3}", tokens[i + 1]):
                return tok
            return tok
        if re.fullmatch(r"[A-Z]\d{1,3}", tok):
            return tok[0]
    m = re.search(r"\b([A-Z])\s*\d{0,3}\b", s2)
    if m:
        if len(m.group(0).strip()) == 1 or re.fullmatch(r"[A-Z]\s*\d{1,3}", m.group(0)):
            return m.group(1)
    return None


def parse_pot_block_strict(text: str) -> Optional[Tuple[str, str]]:
    """Strict pot/block parser with SP/SS guard and defaults.

    Steps:
      - Reject pure 'SP'/'SS' (case-insensitive).
      - Normalize delimiters/whitespace; drop lowercase; then extract pot letter and block number.
      - If only block is present, default pot='A'; if only pot, default block='1'.

    Args:
        text: Raw pot/block string.

    Returns:
        (pot, block) on success; None if parsing fails.
    """
    if text is None:
        return None
    s = str(text)
    if re.fullmatch(r"(?i)\s*(SP|SS)\s*", s):
        return None
    s_clean = re.sub(r"[{}\[\]<>]", "", s)
    s_clean = re.sub(r"[\u200c\u200f\u200e]+", "", s_clean)
    s_clean = re.sub(r"[\\/_-]+", " ", s_clean)
    s_clean = re.sub(r"\s+", " ", s_clean).strip()
    s_clean = re.sub(r"[a-z]", "", s_clean)
    pot = __extract_pot_letter_strict(s_clean)
    block = _extract_block_number_robust(s_clean)
    if pot is None and block is not None:
        pot = "A"
    if pot is not None and block is None:
        block = "1"
    if pot is None or block is None:
        return None
    return pot, block


def parse_pot_block(text: str) -> Tuple[str, str]:
    """Lenient pot/block parser with sensible defaults.

    Behavior:
      - Empty/whitespace-only → ('A','1').
      - Else try strict parser; if it fails, return ('A','1').

    Args:
        text: Raw pot/block string.

    Returns:
        Tuple of (pot, block) strings.
    """
    if text is None or not str(text).strip():
        return "A", "1"
    res = parse_pot_block_strict(text)
    return res if res is not None else ("A", "1")


def stain_lookup(stain_dict: Union[Dict[str, str], List[str], None]) -> Dict[str, str]:
    """Build a case-insensitive lookup table for stain names/aliases.

    For dict inputs, both keys and values map to the canonical value.
    For list inputs, each item maps to itself (uppercased key).

    Args:
        stain_dict: Canonical mapping (dict) or a flat list of canonical names.

    Returns:
        A dict mapping UPPERCASE keys to canonical stain names.
    """
    lut: Dict[str, str] = {}
    if isinstance(stain_dict, dict):
        for k, v in stain_dict.items():
            if isinstance(k, str) and isinstance(v, str):
                lut[k.upper()] = v
                lut[v.upper()] = v
    elif isinstance(stain_dict, list):
        for v in stain_dict:
            if isinstance(v, str):
                lut[v.upper()] = v
    return lut


# ---------- Fuzzy matching helpers for stains (IMPROVED) ----------

def _fuzzy_best_key(
    token: str,
    candidates: List[str],
    threshold: int
) -> Optional[str]:
    """
    Return the best-matching candidate (case-insensitive) if score >= threshold.
    Uses RapidFuzz when available with a scorer chosen based on single- vs multi-word;
    falls back to difflib if RapidFuzz is not installed.
    """
    tok = (token or "").strip()
    if not tok or not candidates:
        return None

    # Prefer RapidFuzz if available
    try:
        from rapidfuzz import process, fuzz  # type: ignore

        def _is_single_word(s: str) -> bool:
            # treat hyphenated as single "word" for stain lexicon purposes
            return " " not in s.strip()

        single = _is_single_word(tok)
        any_multi = any(not _is_single_word(c) for c in candidates)

        # For single-word vs single-word, ratio/WRatio works best; for multi, token_set_ratio is robust.
        scorer = fuzz.ratio if (single and not any_multi) else fuzz.token_set_ratio

        best = process.extractOne(tok, candidates, scorer=scorer, score_cutoff=threshold)
        if best:
            return best[0]
        return None
    except Exception:
        # Fallback: difflib (0..1) → approximate to 0..100 via cutoff mapping
        import difflib
        match = difflib.get_close_matches(
            tok.lower(),
            [c.lower() for c in candidates],
            n=1,
            cutoff=max(0.0, min(1.0, threshold / 100.0)),
        )
        if not match:
            return None
        low2orig = {c.lower(): c for c in candidates}
        return low2orig.get(match[0])


def _stain_lut_keys(stain_dict: Union[Dict[str, str], List[str], None]) -> List[str]:
    """
    Flatten possible keys/aliases for stains.
    For dict: include both keys and values; for list: include the list itself.
    """
    if isinstance(stain_dict, dict):
        out: List[str] = []
        for k, v in stain_dict.items():
            if isinstance(k, str):
                out.append(k)
            if isinstance(v, str):
                out.append(v)
        # de-duplicate preserving order
        seen: set = set()
        uniq: List[str] = []
        for x in out:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq
    elif isinstance(stain_dict, list):
        return [str(x) for x in stain_dict if isinstance(x, str)]
    return []


# ---------- Stain preprocessing helpers ----------

def _normalize_stain_raw(raw_text: str) -> str:
    """
    Normalize raw stain text before tokenization:
      - Remove parentheses along with their content.
      - Remove commas placed between letters and glue letters back together.
      - Normalize whitespace.
      - Keep hyphenated tokens intact (handled later by the tokenizer).
    """
    s = str(raw_text)

    # 1) drop any (...) including the parentheses themselves
    s = re.sub(r"\([^)]*\)", " ", s)

    # 2) remove commas between letters and glue letters again (iterate until stable)
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"([A-Za-z])\s*,\s*([A-Za-z])", r"\1\2", s)

    # 3) collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_stain(
    raw_text: str,
    stain_dict: Union[Dict[str, str], List[str], None],
    stain_repl: Optional[Dict[str, str]],
    fuzzy_threshold: Optional[int] = None,  # optional; when None → no fuzzy path
) -> str:
    """Parse a canonical stain name (or pair 'X&Y') from raw text.

    Pipeline:
      - Normalize, tokenize (keeping '+' and '-'), and apply direct replacements.
      - Handle explicit 'H & E'/'HE' early.
      - Try exact combo 'X + Y', then two singles, then quick LUT.
      - If `fuzzy_threshold` is set, attempt fuzzy match (single or combo) against LUT keys.
      - Fallback is 'H&E'.

    Args:
        raw_text: Raw stain text from OCR.
        stain_dict: Canonical mapping (dict) or canonical list used as LUT.
        stain_repl: Optional direct token replacements (case-insensitive keys).
        fuzzy_threshold: If provided (0–100), enable fuzzy matching with this cutoff.

    Returns:
        Canonical stain string (e.g., 'HE', 'PAS', 'HE&PAS'); default 'H&E' on failure.
    """
    if raw_text is None:
        return "H&E"

    # --- normalization (kept logic) ---
    s_norm = _normalize_stain_raw(raw_text)

    # tokenization: keep '+' and '-' inside tokens
    toks = [w for w in re.split(r"[^A-Za-z0-9\+\-]+", s_norm) if w]
    if not toks:
        return "H&E"

    joined = " ".join(toks)
    if re.search(r"\bH\s*&\s*E\b", joined, flags=re.I):
        return "HE"

    repl_upper = {k.upper(): v for k, v in (stain_repl or {}).items()}
    toks2 = [repl_upper.get(w.upper(), w) for w in toks]

    def _split_tokens_on_plus(tokens: List[str]) -> List[str]:
        out: List[str] = []
        for t in tokens:
            if isinstance(t, str) and "+" in t and len(t) > 1:
                parts = t.split("+")
                for i, p in enumerate(parts):
                    p = p.strip()
                    if p:
                        out.append(p)
                    if i < len(parts) - 1:
                        out.append("+")
            else:
                out.append(t)
        return out

    def _map_to_canonical(token: str) -> Optional[str]:
        if not token:
            return None
        t = str(token).strip()
        if not t:
            return None
        if t.upper() in ("HE", "HE+"):
            return "HE"
        if isinstance(stain_dict, dict):
            return stain_dict.get(t.lower()) or stain_dict.get(t) or stain_dict.get(t.strip().lower())
        elif isinstance(stain_dict, list):
            lut = {str(x).lower(): x for x in stain_dict}
            return lut.get(t.lower())
        return None

    seq = _split_tokens_on_plus(toks2)

    # exact combo: X + Y
    for i in range(len(seq) - 2):
        left, mid, right = seq[i], seq[i + 1], seq[i + 2]
        if mid != "+" or left == "+" or right == "+":
            continue
        left_can = _map_to_canonical(left)
        right_can = _map_to_canonical(right)
        if left_can and right_can:
            return f"{left_can}&{right_can}"

    # exact singles
    seen: List[str] = []
    for t in toks2:
        can = _map_to_canonical(t)
        if can and (can not in seen):
            seen.append(can)
        if len(seen) >= 2:
            return f"{seen[0]}&{seen[1]}"

    if re.search(r"\bHE\+?\b", joined, flags=re.I):
        return "HE"

    # quick exact LUT
    lut = stain_lookup(stain_dict)
    for w in toks2:
        cand = lut.get(w.strip().upper())
        if cand is not None:
            return cand

    # ---------- FUZZY FALLBACK (requested via fuzzy_threshold) ----------
    if isinstance(fuzzy_threshold, int) and fuzzy_threshold >= 0:
        candidates = _stain_lut_keys(stain_dict)
        if candidates:
            # try fuzzy combo X + Y first
            for i in range(len(seq) - 2):
                left, mid, right = seq[i], seq[i + 1], seq[i + 2]
                if mid != "+" or left == "+" or right == "+":
                    continue
                left_best = _fuzzy_best_key(left, candidates, fuzzy_threshold)
                right_best = _fuzzy_best_key(right, candidates, fuzzy_threshold)
                if left_best and right_best:
                    left_can = _map_to_canonical(left_best) or left_best
                    right_can = _map_to_canonical(right_best) or right_best
                    return f"{left_can}&{right_can}"

            # then fuzzy singles (collect up to 2)
            seen_fuzzy: List[str] = []
            for t in toks2:
                best = _fuzzy_best_key(t, candidates, fuzzy_threshold)
                if not best:
                    continue
                mapped = _map_to_canonical(best) or best
                if mapped not in seen_fuzzy:
                    seen_fuzzy.append(mapped)
                if len(seen_fuzzy) >= 2:
                    return f"{seen_fuzzy[0]}&{seen_fuzzy[1]}"
            if seen_fuzzy:
                return seen_fuzzy[0]

    # fallback unchanged
    return "H&E"


def reconstruct_casenumber_from_all_groups(
    words_by_group: Dict[str, str]
) -> Optional[str]:
    """Reconstruct a 'case/yy' token from multiple OCR groups.

    Method:
      - Search groups whose keys contain 'case'.
      - Collect canonical 'A123/45' hits and raw long digit runs (4–8).
      - Prefer longest-number slash hit; align any long number ending with same 'yy' and trim a trailing '1' when another same-yy without '...1' exists.

    Args:
        words_by_group: Mapping from group-name → raw OCR text.

    Returns:
        Reconstructed 'prefix/yy' string if resolvable; otherwise None.
    """
    if not words_by_group:
        return None
    case_keys = [k for k in words_by_group.keys() if "case" in k.lower()]
    if not case_keys:
        return None
    slash_hits: List[Tuple[str, str, str]] = []
    long_nums: List[str] = []
    tok_pat = re.compile(r"[A-Za-z0-9/]+")
    for k in case_keys:
        text = words_by_group.get(k) or ""
        for tok in tok_pat.findall(text):
            m = re.fullmatch(r"([A-Za-z])?(\d{1,6})/(\d{2})", tok)
            if m:
                lab = (m.group(1) or "").upper()
                num = m.group(2)
                yy = m.group(3)
                slash_hits.append((lab, num, yy))
                continue
            m2 = re.fullmatch(r"\d{4,8}", tok)
            if m2:
                long_nums.append(m2.group(0))
    if not slash_hits or not long_nums:
        return None
    slash_hits.sort(key=lambda t: len(t[1]), reverse=True)
    for lab, num, yy in slash_hits:
        any_sameyy_no1 = any((h[2] == yy) and (not h[1].endswith("1")) for h in slash_hits)
        for s in long_nums:
            if s[-2:] != yy:
                continue
            prefix = s[:-2]
            if prefix.endswith("1") and any_sameyy_no1:
                pref2 = prefix[:-1]
                if 1 <= len(pref2) <= 6:
                    return f"{pref2}/{yy}"
            if 1 <= len(prefix) <= 6:
                return f"{prefix}/{yy}"
    return None
