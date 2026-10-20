"""Single source of truth for analytes, units, plausibility bounds, labels and safety limits."""
from __future__ import annotations

# Raw inputs the model understands (units in comments). `sex`: 1 = male, 0 = female.
RAW_FEATURES = [
    "age", "sex",
    "hb",          # g/dL
    "rbc",         # x10^6/uL
    "hct",         # %
    "mcv",         # fL
    "mch",         # pg
    "mchc",        # g/dL
    "wbc",         # x10^3/uL
    "neut_pct",    # %
    "lymph_pct",   # %
    "platelets",   # x10^3/uL
    "fbs",         # mg/dL
    "hba1c",       # %
    "total_chol",  # mg/dL
    "ldl",         # mg/dL
    "hdl",         # mg/dL
    "tg",          # mg/dL
    "creatinine",  # mg/dL
    "urea",        # mg/dL
    "alt",         # U/L
    "ast",         # U/L
    "tsh",         # mIU/L
    "ft4",         # ng/dL
]

LAB_ANALYTES = [f for f in RAW_FEATURES if f not in ("age", "sex")]

DERIVED_FEATURES = [
    "egfr", "non_hdl", "tc_hdl", "tg_hdl", "nlr", "anc", "mentzer",
    "hb_gap", "bun_cr", "alt_ast", "log_tsh",
]
MODEL_FEATURES = RAW_FEATURES + DERIVED_FEATURES

LABELS = [
    "anemia", "ckd", "diabetes", "hypercholesterolemia",
    "hypothyroidism", "hyperthyroidism", "infection",
]

LABEL_DISPLAY = {
    "anemia": "Anemia",
    "ckd": "Chronic kidney disease (reduced eGFR)",
    "diabetes": "Diabetes mellitus",
    "hypercholesterolemia": "Hypercholesterolemia",
    "hypothyroidism": "Hypothyroidism",
    "hyperthyroidism": "Hyperthyroidism",
    "infection": "Acute infection / inflammatory response",
}

# Words that indicate a condition is being mentioned in free text (used to police LLM narratives).
LABEL_KEYWORDS = {
    "anemia": ["anemia", "anaemia"], "ckd": ["kidney", "ckd"], "diabetes": ["diabet"],
    "hypercholesterolemia": ["cholesterol"], "hypothyroidism": ["hypothyroid"],
    "hyperthyroidism": ["hyperthyroid"], "infection": ["infection", "infective", "sepsis"],
}

# A condition can only be assessed if at least one input from each group is present.
REQUIRED_INPUTS = {
    "anemia": ["hb"],
    "ckd": ["creatinine"],
    "diabetes": ["fbs", "hba1c"],           # either one
    "hypercholesterolemia": ["total_chol", "ldl"],  # either one
    "hypothyroidism": ["tsh"],
    "hyperthyroidism": ["tsh"],
    "infection": ["wbc"],
}

# If a condition is uncertain, which missing test would most help resolve it?
NEXT_TEST_FOR = {
    "anemia": [("mcv", "Peripheral blood film, serum ferritin / B12 / folate"), ("hb", "Haemoglobin / full blood count")],
    "ckd": [("urea", "Serum urea"), ("creatinine", "Repeat serum creatinine in 1-2 weeks; urine albumin:creatinine ratio")],
    "diabetes": [("hba1c", "HbA1c"), ("fbs", "Repeat fasting plasma glucose")],
    "hypercholesterolemia": [("ldl", "Full fasting lipid profile (LDL, HDL, triglycerides)"), ("total_chol", "Total cholesterol")],
    "hypothyroidism": [("ft4", "Free T4 (+ anti-TPO if TSH raised)"), ("tsh", "TSH")],
    "hyperthyroidism": [("ft4", "Free T4 / free T3 (+ TRAb if TSH suppressed)"), ("tsh", "TSH")],
    "infection": [("neut_pct", "White cell differential, CRP / procalcitonin"), ("wbc", "Full blood count")],
}

# Hard physiological bounds. A value outside is treated as an extraction/unit error, never as data.
BOUNDS = {
    "age": (0, 120), "hb": (2.0, 25.0), "rbc": (1.0, 9.0), "hct": (8.0, 70.0),
    "mcv": (50.0, 140.0), "mch": (12.0, 45.0), "mchc": (20.0, 42.0),
    "wbc": (0.3, 150.0), "neut_pct": (0.0, 100.0), "lymph_pct": (0.0, 100.0),
    "platelets": (3.0, 1500.0), "fbs": (20.0, 800.0), "hba1c": (3.0, 18.0),
    "total_chol": (60.0, 600.0), "ldl": (10.0, 500.0), "hdl": (5.0, 150.0),
    "tg": (20.0, 3000.0), "creatinine": (0.2, 20.0), "urea": (3.0, 400.0),
    "alt": (3.0, 3000.0), "ast": (3.0, 3000.0), "tsh": (0.001, 150.0), "ft4": (0.1, 8.0),
}

# Conventional laboratory critical-alert limits (adult). These drive an escalation flag that is
# independent of the ML model. They must be confirmed against the lab's own SOP / a clinician.
CRITICAL_LIMITS = {
    "hb": (7.0, 20.0),
    "wbc": (2.0, 30.0),
    "platelets": (20.0, 1000.0),
    "fbs": (45.0, 450.0),
    "creatinine": (None, 5.0),
    "urea": (None, 200.0),
    "tsh": (0.01, 50.0),
    "hba1c": (None, 14.0),
}

# Free-text names -> canonical key (lower-cased, punctuation stripped before lookup).
ALIASES = {
    "hb": ["hb", "hgb", "haemoglobin", "hemoglobin"],
    "rbc": ["rbc", "red blood cell count", "red cell count", "total rbc count", "rbc count"],
    "hct": ["hct", "pcv", "haematocrit", "hematocrit", "packed cell volume"],
    "mcv": ["mcv"], "mch": ["mch"], "mchc": ["mchc"],
    "wbc": ["wbc", "total wbc count", "white blood cell count", "total white cell count", "twbc", "wbc count"],
    "neut_pct": ["neutrophils", "neutrophil", "neutrophils %", "neut", "neut pct", "neutrophil %"],
    "lymph_pct": ["lymphocytes", "lymphocyte", "lymphocytes %", "lymph", "lymph pct", "lymphocyte %"],
    "platelets": ["platelets", "platelet count", "plt", "platelet"],
    "fbs": ["fbs", "fasting blood sugar", "fasting glucose", "fasting plasma glucose", "fpg", "blood glucose fasting", "glucose fasting"],
    "hba1c": ["hba1c", "glycated haemoglobin", "glycosylated hemoglobin", "a1c", "hb a1c"],
    "total_chol": ["total cholesterol", "cholesterol total", "cholesterol", "tc", "total chol", "chol"],
    "ldl": ["ldl", "ldl cholesterol", "ldl c", "ldl-c"],
    "hdl": ["hdl", "hdl cholesterol", "hdl c", "hdl-c"],
    "tg": ["triglycerides", "triglyceride", "tg", "tgl"],
    "creatinine": ["creatinine", "serum creatinine", "s creatinine", "creat"],
    "urea": ["urea", "blood urea", "serum urea"],
    "alt": ["alt", "sgpt", "alanine aminotransferase"],
    "ast": ["ast", "sgot", "aspartate aminotransferase"],
    "tsh": ["tsh", "thyroid stimulating hormone"],
    "ft4": ["ft4", "free t4", "t4 free", "free thyroxine"],
    "age": ["age"], "sex": ["sex", "gender"],
}

FEATURE_TEXT = {
    "hb": "haemoglobin", "rbc": "red cell count", "hct": "haematocrit", "mcv": "MCV",
    "mch": "MCH", "mchc": "MCHC", "wbc": "white cell count", "neut_pct": "neutrophil %",
    "lymph_pct": "lymphocyte %", "platelets": "platelet count", "fbs": "fasting glucose",
    "hba1c": "HbA1c", "total_chol": "total cholesterol", "ldl": "LDL", "hdl": "HDL",
    "tg": "triglycerides", "creatinine": "creatinine", "urea": "urea", "alt": "ALT", "ast": "AST",
    "tsh": "TSH", "ft4": "free T4", "age": "age", "sex": "sex", "egfr": "eGFR",
    "non_hdl": "non-HDL cholesterol", "tc_hdl": "TC/HDL ratio", "tg_hdl": "TG/HDL ratio",
    "nlr": "neutrophil/lymphocyte ratio", "anc": "absolute neutrophil count",
    "mentzer": "Mentzer index (MCV/RBC)", "hb_gap": "haemoglobin vs sex-specific limit",
    "bun_cr": "urea/creatinine ratio", "alt_ast": "ALT/AST ratio", "log_tsh": "TSH (log)",
}
