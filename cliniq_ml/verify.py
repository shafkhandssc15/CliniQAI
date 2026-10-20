"""Cross-check two independent extractions of the same report (e.g. local OCR vs a cloud VLM).

A decimal-point slip is the classic silent OCR failure: bounds and identities catch ~2/3 of them, the rest
are only visible when a second reader disagrees. Use this to decide which fields a physician must verify
*before* analysis runs.
"""
from __future__ import annotations

import math

from .engine import normalize_input


def cross_check_extractions(a: dict, b: dict, rel_tol: float = 0.08) -> list[dict]:
    """Compare two {analyte: value} dicts. Returns one record per disagreeing/one-sided field."""
    va, _ = normalize_input(a)
    vb, _ = normalize_input(b)
    out = []
    for k in sorted(set(va) | set(vb)):
        if k in ("age", "sex"):
            continue
        x, y = va.get(k), vb.get(k)
        if x is None or y is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(y, float) and math.isnan(y)):
            if (x is None) != (y is None):
                out.append({"field": k, "reader_a": x, "reader_b": y, "issue": "only one reader extracted this field"})
            continue
        if x == 0 or y == 0:
            continue
        ratio = max(abs(x), abs(y)) / min(abs(x), abs(y))
        if abs(x - y) <= rel_tol * max(abs(x), abs(y)):
            continue
        issue = "possible decimal-point slip (values differ ~10x)" if 8.0 <= ratio <= 12.5 else "readers disagree"
        out.append({"field": k, "reader_a": x, "reader_b": y, "issue": issue})
    return out
