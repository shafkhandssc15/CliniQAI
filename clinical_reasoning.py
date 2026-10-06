"""
Clinical safety and reasoning layer for CliniQ AI.

Converts raw model votes + extracted labs into an uncertainty-aware
impression, finding-tied follow-up, and context-aware narrative.
Never presents "Healthy" as a definitive diagnosis.
"""

from __future__ import annotations

from typing import Any, Optional

CONFIDENCE_DEFINITIVE_MIN = 70.0

# Canonical disease labels used for ensemble voting
CANON_LABELS = (
    "Anemia",
    "Infection / Inflammation",
    "Leukopenia",
    "Diabetes",
    "Prediabetes Risk",
    "Hypoglycemia",
    "Chronic Kidney Disease",
    "Hypothyroidism",
    "Hyperthyroidism",
    "Hypercholesterolemia",
    "Thrombocytopenia",
    "Thrombocytosis",
    "No major abnormalities detected",
)

_HEALTHY_TOKENS = (
    "healthy",
    "no abnormal",
    "no major abnormal",
    "within normal",
    "normal study",
    "unremarkable",
    "all values are within",
    "none. all values",
)

_DISEASE_MAP = (
    ("chronic kidney", "Chronic Kidney Disease"),
    ("ckd", "Chronic Kidney Disease"),
    ("impaired renal", "Chronic Kidney Disease"),
    ("hypothyroid", "Hypothyroidism"),
    ("hyperthyroid", "Hyperthyroidism"),
    ("thyrotoxic", "Hyperthyroidism"),
    ("hypercholesterol", "Hypercholesterolemia"),
    ("dyslipid", "Hypercholesterolemia"),
    ("prediabetes", "Prediabetes Risk"),
    ("diabetes", "Diabetes"),
    ("hypoglyc", "Hypoglycemia"),
    ("anemia", "Anemia"),
    ("anaemia", "Anemia"),
    ("leukopenia", "Leukopenia"),
    ("thrombocytopenia", "Thrombocytopenia"),
    ("thrombocytosis", "Thrombocytosis"),
    ("infection", "Infection / Inflammation"),
    ("inflammation", "Infection / Inflammation"),
    ("elevated wbc", "Infection / Inflammation"),
)


def canonicalize_diagnosis(raw: Any) -> str:
    """Map heterogeneous model outputs onto a single vote label."""
    if raw is None:
        return "Unknown"
    if isinstance(raw, list):
        raw = raw[0] if raw else "Unknown"
    text = str(raw).strip()
    if not text or text.lower() in {"n/a", "offline", "unknown", "none"}:
        return "Unknown"

    # Drop trailing confidence, take first clause
    text = text.split("(")[0].strip()
    first = text.split(",")[0].strip()
    low = first.lower()

    if any(tok in low for tok in _HEALTHY_TOKENS):
        return "No major abnormalities detected"

    for needle, label in _DISEASE_MAP:
        if needle in low:
            return label

    # Keep a short readable label rather than dumping the whole sentence
    return first[:80] if first else "Unknown"


def _num(val) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def evaluate_clinical_findings(
    values: dict,
    extracted_flags: dict,
    extra_labs: dict | None = None,
    patient: dict | None = None,
) -> list[dict]:
    """
    Rule-based flags from extracted (not imputed) labs.
    Each finding: code, severity (info|mild|moderate|severe), title, detail, differentials.
    """
    extra_labs = extra_labs or {}
    patient = patient or {}
    findings: list[dict] = []
    gender = (patient.get("gender") or values.get("Gender") or "").strip().title()
    age = _num(patient.get("age")) if patient.get("age") not in (None, "", "—") else _num(values.get("Age"))

    def extracted(key: str):
        if extra_labs.get(key) is not None:
            return extra_labs[key]
        if extracted_flags.get(key) and values.get(key) is not None:
            return values[key]
        return None

    hb = _num(extracted("Hemoglobin"))
    if hb is not None:
        hb_cut = 13.0 if gender == "Male" else 12.0
        if gender not in ("Male", "Female"):
            hb_cut = 12.0
        if hb < 8:
            findings.append(_finding(
                "anemia", "severe",
                "Severe anaemia",
                f"Haemoglobin {hb} g/dL is markedly below the {gender or 'adult'} reference threshold (~{hb_cut} g/dL).",
                ["Iron deficiency", "Blood loss", "Haemolysis", "Bone marrow disorder"],
            ))
        elif hb < hb_cut:
            findings.append(_finding(
                "anemia", "moderate",
                "Anaemia",
                f"Haemoglobin {hb} g/dL is below the expected range for a {gender or 'adult'} patient (cut-off ~{hb_cut} g/dL).",
                ["Iron deficiency", "B12/folate deficiency", "Chronic disease", "Occult bleeding"],
            ))

    wbc = _num(extracted("WBC"))
    if wbc is not None:
        if wbc > 11.0:
            findings.append(_finding(
                "leukocytosis", "moderate",
                "Leukocytosis",
                f"WBC {wbc} ×10³/µL is above the usual adult range (4.5–11.0).",
                ["Infection", "Inflammation", "Steroid effect", "Haematological disorder"],
            ))
        elif wbc < 4.0:
            findings.append(_finding(
                "leukopenia", "moderate",
                "Leukopenia",
                f"WBC {wbc} ×10³/µL is below the usual adult range.",
                ["Viral illness", "Drug effect", "Bone marrow suppression"],
            ))

    plt = _num(extracted("Platelets"))
    if plt is not None:
        if plt < 100:
            findings.append(_finding(
                "thrombocytopenia", "moderate",
                "Thrombocytopenia",
                f"Platelet count {int(plt)} ×10³/µL is below the reference range (150–450).",
                ["Immune thrombocytopenia", "Drug effect", "Hypersplenism", "EDTA clumping (pseudothrombocytopenia)"],
            ))
        elif plt < 150:
            findings.append(_finding(
                "thrombocytopenia", "mild",
                "Mild thrombocytopenia",
                f"Platelet count {int(plt)} ×10³/µL is below the lower reference limit of 150 ×10³/µL.",
                ["EDTA clumping", "Recent viral illness", "Medication effect"],
            ))
        elif plt <= 165:
            findings.append(_finding(
                "borderline_platelets", "info",
                "Borderline-low platelets",
                f"Platelet count {int(plt)} ×10³/µL sits at the lower edge of the adult reference range (150–450). "
                "This is not thrombocytopenia, but it should not be ignored.",
                ["Normal variant", "Early downward trend", "Pre-analytical variation"],
            ))
        elif plt > 450:
            findings.append(_finding(
                "thrombocytosis", "mild",
                "Thrombocytosis",
                f"Platelet count {int(plt)} ×10³/µL is above 450 ×10³/µL.",
                ["Reactive (infection/iron deficiency)", "Myeloproliferative neoplasm"],
            ))

    mono_pct = _num(extracted("Monocytes_Pct") or extra_labs.get("Monocytes_Pct"))
    mono_abs = _num(extra_labs.get("Monocytes_Abs"))
    if mono_pct is not None and mono_pct > 10:
        sev = "moderate" if mono_pct >= 15 else "mild"
        abs_bit = f" Absolute monocyte count {mono_abs} ×10³/µL." if mono_abs is not None else ""
        age_bit = ""
        if age is not None:
            if age >= 60:
                age_bit = " In older adults, persistent monocytosis warrants exclusion of chronic inflammation or a myeloproliferative process."
            elif age < 18:
                age_bit = " In children and adolescents this is often a recovery-phase or reactive change."
            else:
                age_bit = " In adults this is most often reactive (recovery from infection, inflammation, or stress)."
        findings.append(_finding(
            "monocytosis", sev,
            "Mild monocytosis" if sev == "mild" else "Monocytosis",
            f"Monocytes {mono_pct}% exceed the usual differential cut-off of 10%.{abs_bit}{age_bit}",
            [
                "Recovery from infection",
                "Chronic inflammation",
                "Tissue stress / corticosteroid effect",
                "Less commonly: chronic myelomonocytic leukaemia if persistent and unexplained",
            ],
        ))

    fbs = _num(extracted("Fasting_Blood_Sugar"))
    if fbs is not None:
        if fbs >= 126:
            findings.append(_finding(
                "diabetes", "moderate",
                "Elevated fasting glucose",
                f"Fasting glucose {int(fbs)} mg/dL meets the laboratory threshold often used for diabetes (≥126 mg/dL) — confirm on a second sample / HbA1c.",
                ["Type 2 diabetes", "Stress hyperglycaemia", "Non-fasting sample"],
            ))
        elif fbs >= 100:
            findings.append(_finding(
                "prediabetes", "mild",
                "Impaired fasting glucose",
                f"Fasting glucose {int(fbs)} mg/dL is in the prediabetes range (100–125 mg/dL).",
                ["Prediabetes", "Non-fasting sample"],
            ))
        elif fbs < 60:
            findings.append(_finding(
                "hypoglycemia", "moderate",
                "Hypoglycaemia",
                f"Glucose {int(fbs)} mg/dL is low.",
                ["Medication-related", "Prolonged fasting", "Endocrine cause"],
            ))

    crea = _num(extracted("Creatinine"))
    if crea is not None and crea > 1.3:
        findings.append(_finding(
            "ckd", "moderate",
            "Elevated creatinine",
            f"Creatinine {crea} mg/dL is above the typical adult upper limit (~1.3 mg/dL). Interpret with muscle mass, hydration, and eGFR.",
            ["Acute kidney injury", "Chronic kidney disease", "Pre-renal azotaemia"],
        ))

    tsh = _num(extracted("TSH"))
    if tsh is not None:
        if tsh > 4.5:
            findings.append(_finding(
                "hypothyroid", "mild" if tsh < 10 else "moderate",
                "Elevated TSH",
                f"TSH {tsh} mIU/L is above the usual range (≈0.4–4.5). Confirm with free T4.",
                ["Primary hypothyroidism", "Non-thyroidal illness", "Recovery phase"],
            ))
        elif tsh < 0.3:
            findings.append(_finding(
                "hyperthyroid", "moderate",
                "Suppressed TSH",
                f"TSH {tsh} mIU/L is suppressed. Confirm with free T4/T3.",
                ["Hyperthyroidism", "Exogenous thyroxine", "Non-thyroidal illness"],
            ))

    chol = _num(extracted("Cholesterol"))
    if chol is not None and chol >= 200:
        findings.append(_finding(
            "cholesterol", "mild" if chol < 240 else "moderate",
            "Elevated total cholesterol",
            f"Total cholesterol {int(chol)} mg/dL is above the desirable range (<200 mg/dL).",
            ["Primary dyslipidaemia", "Hypothyroidism", "Diet / metabolic syndrome"],
        ))

    notes = (patient.get("clinical_notes") or "").strip()
    if notes:
        findings.append(_finding(
            "clinical_notes", "info",
            "Clinical notes considered",
            f"Attending notes: “{notes}”. Laboratory flags should be interpreted against this history, not in isolation.",
            [],
        ))

    return findings


def _finding(code, severity, title, detail, differentials) -> dict:
    return {
        "code": code,
        "severity": severity,
        "title": title,
        "detail": detail,
        "differentials": differentials,
    }


def build_impression(ensemble: dict, findings: list[dict]) -> dict:
    """
    Uncertainty-aware clinical impression.
    Never returns the label 'Healthy'. Low confidence is never a final diagnosis.
    """
    raw_canon = canonicalize_diagnosis(ensemble.get("primary_diagnosis"))
    conf = float(ensemble.get("confidence") or 0)
    band = ensemble.get("confidence_band") or "INCONCLUSIVE"

    lab_findings = [f for f in findings if f["code"] != "clinical_notes"]
    major = [f for f in lab_findings if f["severity"] in ("moderate", "severe")]
    minor = [f for f in lab_findings if f["severity"] in ("mild", "info")]

    definitive = conf >= CONFIDENCE_DEFINITIVE_MIN and band in ("HIGH", "MODERATE")
    disease_like = raw_canon not in (
        "No major abnormalities detected",
        "Unknown",
        "Analysis Inconclusive",
    )

    caution = (
        "This is decision support, not a final clinical diagnosis. "
        "All findings must be verified by the attending physician."
    )

    if not definitive:
        if lab_findings:
            titles = "; ".join(f["title"] for f in lab_findings if f["code"] != "clinical_notes")
            impression = (
                f"No definitive diagnosis — {titles}. Clinical correlation required."
            )
        else:
            impression = (
                "No significant abnormalities detected on the extracted panel. "
                "Clinical correlation required."
            )
        headline = "Clinical correlation required"
        is_final = False
    elif major and disease_like:
        impression = raw_canon
        extra = "; ".join(f["title"] for f in minor if f["code"] != "clinical_notes")
        if extra:
            impression = f"{raw_canon} — also note: {extra}"
        headline = "Provisional impression"
        is_final = True
    elif lab_findings:
        titles = "; ".join(f["title"] for f in lab_findings)
        impression = (
            f"No major disease pattern identified. Minor variations noted: {titles}. "
            "Clinical correlation required."
        )
        headline = "Minor variations noted"
        is_final = False
    else:
        impression = "No significant abnormalities detected. Clinical correlation required."
        headline = "No major abnormalities detected"
        is_final = False

    # Hard safety: never surface the word Healthy
    if "healthy" in impression.lower():
        impression = impression.replace("Healthy", "No major abnormalities detected")
        impression = impression.replace("healthy", "no major abnormalities detected")

    return {
        "impression": impression,
        "headline": headline,
        "is_definitive": is_final,
        "avoided_healthy_label": True,
        "caution": caution,
        "confidence_gate": CONFIDENCE_DEFINITIVE_MIN,
        "canonical_vote": raw_canon,
    }


def build_narrative(
    impression: dict,
    findings: list[dict],
    patient: dict,
    vlm_narrative: str,
) -> str:
    parts = []
    age = patient.get("age") or "unspecified age"
    gender = patient.get("gender") or "unspecified sex"
    name = patient.get("name")
    who = f"{name}, {age}y {gender}" if name and name not in ("Anonymous", "Anonymous Patient") else f"{age}y {gender} patient"

    parts.append(f"Context: {who}.")
    notes = (patient.get("clinical_notes") or "").strip()
    if notes:
        parts.append(f"Clinical notes provided: {notes}.")

    lab_findings = [f for f in findings if f["code"] != "clinical_notes"]
    if lab_findings:
        parts.append("Rule-based review of extracted values:")
        for f in lab_findings:
            parts.append(f"• {f['title']}: {f['detail']}")
            if f.get("differentials"):
                parts.append("  Differential considerations: " + "; ".join(f["differentials"]) + ".")
    else:
        parts.append(
            "Extracted core biomarkers that were present on the report fall within usual adult reference intervals. "
            "Absence of a flag is not proof of health — unextracted parameters and clinical examination still matter."
        )

    if not impression["is_definitive"]:
        parts.append(
            f"Ensemble agreement is below the {int(CONFIDENCE_DEFINITIVE_MIN)}% threshold used for a definitive label, "
            "so no disease name is issued as a final conclusion."
        )

    parts.append(impression["caution"])

    # Keep useful VLM prose after the structured reasoning, if it is not just "healthy"
    vlm = (vlm_narrative or "").strip()
    if vlm and "healthy" not in vlm.lower()[:80]:
        parts.append("Model narrative (for the clinician to corroborate): " + vlm)

    return " ".join(parts) if False else "\n".join(parts)


def build_recommendations(findings: list[dict], patient: dict) -> list[str]:
    recs: list[str] = []
    codes = {f["code"] for f in findings}
    age = _num(patient.get("age"))

    if "monocytosis" in codes:
        recs.append(
            "Mild monocytosis: correlate with recent infection, inflammation, recovery phase, or corticosteroids — do not label as disease on a single differential."
        )
        recs.append(
            "Repeat FBC with white-cell differential in 4–8 weeks to distinguish a transient reactive change from persistent monocytosis."
        )
        recs.append(
            "If monocytosis persists without an explanation, add CRP/ESR and review for occult inflammation; consider haematology input if progressive or accompanied by cytopenias."
        )

    if "borderline_platelets" in codes:
        recs.append(
            "Platelets are at the lower edge of normal: document any bleeding/bruising, review antiplatelet/NSAID use, and repeat the count (citrate tube if EDTA clumping is suspected)."
        )
    if "thrombocytopenia" in codes:
        recs.append(
            "Confirm thrombocytopenia on a repeat sample; exclude platelet clumping on the blood film before investigating further."
        )
        recs.append(
            "Take a bleeding history and review drugs that affect platelets; urgent haematology review if count <50 ×10³/µL or if there is bleeding."
        )

    if "anemia" in codes:
        recs.extend([
            "Iron studies (ferritin, TIBC, serum iron) and a blood film to classify the anaemia.",
            "B12 and folate if the film is macrocytic; consider GI work-up if iron deficiency is confirmed.",
        ])
    if "leukocytosis" in codes:
        recs.append("Correlate leukocytosis with fever, focal symptoms, and medications; repeat FBC if the patient is well and no source is obvious.")
    if "leukopenia" in codes:
        recs.append("Repeat FBC; review marrow-suppressive drugs and recent viral illness.")
    if "diabetes" in codes:
        recs.extend([
            "Confirm with HbA1c and/or a second fasting glucose before diagnosing diabetes.",
            "Once confirmed: urine ACR, lipids, BP, and retinopathy screening.",
        ])
    if "prediabetes" in codes:
        recs.append("Repeat fasting glucose or HbA1c; lifestyle counselling targeted to the impaired fasting glucose, not generic annual screening.")
    if "ckd" in codes:
        recs.extend([
            "Calculate eGFR, check electrolytes, and send urine ACR; review nephrotoxic drugs.",
            "Repeat creatinine to distinguish acute from chronic change.",
        ])
    if "hypothyroid" in codes:
        recs.append("Add free T4 (± TPO antibodies) before starting thyroxine; repeat TSH after any dose change.")
    if "hyperthyroid" in codes:
        recs.append("Add free T4/T3; consider TRAb and an ECG if thyrotoxicosis is confirmed.")
    if "cholesterol" in codes:
        recs.append("Obtain a full fasting lipid profile (LDL/HDL/triglycerides) and estimate cardiovascular risk rather than acting on total cholesterol alone.")

    recs.append(
        "Correlate every flag with history and examination — laboratory decision support does not replace the attending clinician."
    )

    if not recs or set(codes) <= {"clinical_notes"}:
        if age is not None and age >= 50:
            recs.insert(
                0,
                "No major laboratory disease pattern detected: schedule follow-up according to this patient's age and residual flags (not a generic annual template), and recheck any borderline values sooner.",
            )
        else:
            recs.insert(
                0,
                "No major laboratory disease pattern detected: recheck any borderline or unextracted parameters against clinical context rather than defaulting to routine annual screening alone.",
            )

    # Deduplicate while preserving order
    seen = set()
    out = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out[:8]


def collect_differentials(findings: list[dict]) -> list[str]:
    seen = set()
    out = []
    for f in findings:
        for d in f.get("differentials") or []:
            if d not in seen:
                seen.add(d)
                out.append(d)
    return out
