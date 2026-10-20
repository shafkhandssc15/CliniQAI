"""Optional LLM narrative layer (e.g. via Roar AI). The model's decisions are the source of truth; the LLM only
phrases them. `llm_call` is any callable (prompt: str) -> str, so no provider/API details are baked in here."""
from __future__ import annotations

import re
from typing import Callable

from .schema import LABEL_DISPLAY, LABEL_KEYWORDS

SYSTEM_RULES = (
    "You are drafting a short clinical-support summary for a registered physician in Sri Lanka. "
    "Use ONLY the structured findings provided. Do not add diagnoses, causes, drug names or doses that are not "
    "in the input. State uncertainty exactly as given (UNCERTAIN / NOT_ASSESSABLE are not negatives). "
    "Mention every critical alert and every field flagged for verification. End with: "
    "'Decision support only - requires physician review and sign-off.'"
)


def build_prompt(result: dict) -> str:
    lines = [SYSTEM_RULES, "", f"STATUS: {result['status']}"]
    for key, title in [("findings", "POSITIVE"), ("uncertain", "UNCERTAIN"), ("not_assessable", "NOT ASSESSABLE (panel not provided)")]:
        if result.get(key):
            lines.append(f"{title}: " + "; ".join(result[key]))
    if result.get("critical_alerts"):
        lines.append("CRITICAL: " + "; ".join(f"{a['analyte']} {a['value']} ({a['limit']})" for a in result["critical_alerts"]))
    if result.get("pattern_alerts"):
        lines.append("PATTERN ALERTS: " + " | ".join(result["pattern_alerts"]))
    if result.get("verify_fields"):
        lines.append("VERIFY SOURCE VALUES: " + ", ".join(v["field"] for v in result["verify_fields"]))
    for c in result.get("conditions", {}).values():
        if c["decision"] in ("POSITIVE", "UNCERTAIN") and c["drivers"]:
            lines.append(f"{c['name']} - main drivers: " + "; ".join(f"{d['feature']} {d['value']}" for d in c["drivers"]))
    if result.get("suggested_next_tests"):
        lines.append("SUGGESTED NEXT TESTS: " + " | ".join(result["suggested_next_tests"]))
    return "\n".join(lines)


def generate_narrative(result: dict, llm_call: Callable[[str], str]) -> dict:
    """Returns {'text', 'ok', 'problems'}. Falls back to a deterministic summary if the LLM output breaks the rules."""
    text = llm_call(build_prompt(result)).strip()
    said_ok = set(result.get("findings", []) + result.get("uncertain", []))
    cannot = set(result.get("not_assessable", []))
    hedges = ("not assess", "cannot", "can't", "not provided", "unable", "not available", "not been", "no result", "not tested")
    problems = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        low = sent.lower()
        for lab, name in LABEL_DISPLAY.items():
            if not any(k in low for k in LABEL_KEYWORDS[lab]) or name in said_ok:
                continue
            if name in cannot and any(h in low for h in hedges):
                continue
            problems.append(f"asserts or mentions {name} without support from the model output")
    for a in result.get("critical_alerts", []):
        if a["analyte"].lower() not in text.lower():
            problems.append(f"omits critical alert for {a['analyte']}")
    if "physician" not in text.lower():
        problems.append("missing physician-review statement")
    if problems:
        return {"text": deterministic_summary(result), "ok": False, "problems": problems}
    return {"text": text, "ok": True, "problems": []}


def deterministic_summary(result: dict) -> str:
    parts = [f"Status: {result['status'].replace('_', ' ').lower()}."]
    if result.get("critical_alerts"):
        parts.append("Critical values: " + "; ".join(f"{a['analyte']} {a['value']:g}" for a in result["critical_alerts"]) + ".")
    if result.get("findings"):
        parts.append("Model findings: " + ", ".join(result["findings"]) + ".")
    if result.get("uncertain"):
        parts.append("Uncertain: " + ", ".join(result["uncertain"]) + ".")
    if result.get("not_assessable"):
        parts.append("Not assessable from the provided results: " + ", ".join(result["not_assessable"]) + ".")
    parts.append("Decision support only - requires physician review and sign-off.")
    return " ".join(parts)
