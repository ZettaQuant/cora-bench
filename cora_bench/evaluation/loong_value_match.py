"""Deterministic value and entity matcher for LOONG Spotlight answers."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_SCALE = [
    (r"\bbillions?\b", Decimal(10) ** 9),
    (r"\bmillions?\b", Decimal(10) ** 6),
    (r"\bthousands?\b", Decimal(10) ** 3),
]
_CORP = {
    "inc",
    "co",
    "corp",
    "corporation",
    "ltd",
    "llc",
    "plc",
    "the",
    "group",
    "holdings",
    "company",
    "incorporated",
    "and",
    "of",
}
_NUM = re.compile(r"\(?-?\$?\s*\d[\d,]*(?:\.\d+)?\s*\)?%?")


def _parse(tok: str) -> Decimal | None:
    neg = ("(" in tok and ")" in tok) or tok.strip().startswith("-")
    s = re.sub(r"[(),$%\s]", "", tok)
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        v = Decimal(m.group())
    except InvalidOperation:
        return None
    return -abs(v) if neg else v


def _scale_of(text: str) -> Decimal:
    for pat, s in _SCALE:
        if re.search(pat, text, re.I):
            return s
    return Decimal(1)


def _dp(gold_num: str) -> int:
    m = re.search(r"\.(\d+)", gold_num)
    return len(m.group(1)) if m else 0


def gold_targets(gold: str) -> tuple[list[Decimal], int]:
    """Numeric targets and decimal places for a gold value.

    '$10,135 in thousands' yields both 10135 and 10135000; either counts as a match.
    """
    base = _parse(gold)
    if base is None:
        return [], 0
    scale = _scale_of(gold)
    targets = [base] if scale == 1 else [base, base * scale]
    return targets, _dp(gold)


def _answer_values(answer: str) -> list[Decimal]:
    out: list[Decimal] = []
    global_scale = _scale_of(answer)
    for m in _NUM.finditer(answer):
        v = _parse(m.group())
        if v is None:
            continue
        out.append(v)
        local = _scale_of(answer[m.end() : m.end() + 20])  # unit right after the number
        if local != 1:
            out.append(v * local)
        if global_scale != 1:
            out.append(v * global_scale)
    return out


def _num_match(gold: str, answer: str) -> bool:
    targets, _dp_unused = gold_targets(gold)
    if not targets:
        return False
    # Ignore sign, since filings write negatives as (x), -x, or plain x. The magnitude must
    # match exactly, with no tolerance.
    tset = {abs(t) for t in targets}
    return any(abs(c) in tset for c in _answer_values(answer))


def _entity_match(gold: str, answer: str) -> bool:
    norm = lambda s: re.sub(r"[^a-z0-9 ]", " ", s.lower())
    toks = [t for t in norm(gold).split() if t and t not in _CORP]
    words = set(norm(answer).split())
    return bool(toks) and all(t in words for t in toks)


def spotlight_value_match(gold: str, answer: str) -> bool:
    if not answer or not gold:
        return False
    return (
        _num_match(gold, answer) if any(c.isdigit() for c in gold) else _entity_match(gold, answer)
    )
