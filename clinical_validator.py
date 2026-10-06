"""
clinical_validator.py — Clinical Guardrails, Diagnosis Prioritization Engine, 
Severity Classification, Consensus Arbitration, Disagreement Penalty, and Action Plans.

Provides:
1. Multi-Tier Biomarker Severity Classification (Ionized Calcium, GGT, A/G Ratio, Vitamin D, etc.)
2. Disease Flagging Logic with Strict Multi-Marker Requirements (CKD, Liver Disease, Protein Deficiency)
3. Model-Specific Calibration & Weighting (Gemini 40%, NIM 20%, XGBoost 25%, Rules 15%)
4. Disagreement Penalty & Dynamic Confidence Scaling (Caps at 60–70% on model conflict)
5. Structured Clinical Action Plans and Follow-Up Protocols
"""

import math
from typing import Dict, List, Tuple, Optional, Any

# ── TEST TYPE AWARENESS & CAPABILITIES ──────────────────────────────────────────
TEST_TYPE_ALLOWED_BIOMARKERS = {
    "Full Blood Count": {"Hemoglobin", "WBC", "Platelets", "RBC", "PCV", "MCV", "MCH", "MCHC", "Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils", "Basophils", "ESR", "Ferritin"},
    "FBC": {"Hemoglobin", "WBC", "Platelets", "RBC", "PCV", "MCV", "MCH", "MCHC", "Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils", "Basophils", "ESR", "Ferritin"},
    "Complete Blood Count": {"Hemoglobin", "WBC", "Platelets", "RBC", "PCV", "MCV", "MCH", "MCHC", "Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils", "Basophils", "ESR", "Ferritin"},
    "CBC": {"Hemoglobin", "WBC", "Platelets", "RBC", "PCV", "MCV", "MCH", "MCHC", "Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils", "Basophils", "ESR", "Ferritin"},
    "Lipid Profile": {"Cholesterol", "Triglycerides", "HDL", "LDL", "VLDL", "Total_Cholesterol"},
    "Lipid Panel": {"Cholesterol", "Triglycerides", "HDL", "LDL", "VLDL", "Total_Cholesterol"},
    "Renal Function": {"Creatinine", "BUN", "Urea", "eGFR", "Uric_Acid", "Sodium", "Potassium"},
    "Kidney Profile": {"Creatinine", "BUN", "Urea", "eGFR", "Uric_Acid", "Sodium", "Potassium"},
    "Liver Function Tests": {"ALT", "AST", "SGPT", "SGOT", "Bilirubin", "ALP", "Albumin", "Total_Protein", "GGT", "AG_Ratio"},
    "LFT": {"ALT", "AST", "SGPT", "SGOT", "Bilirubin", "ALP", "Albumin", "Total_Protein", "GGT", "AG_Ratio"},
    "Glycemic Profile": {"Fasting_Blood_Sugar", "FBS", "PPBS", "HbA1c", "Random_Blood_Sugar"},
    "Diabetes Panel": {"Fasting_Blood_Sugar", "FBS", "PPBS", "HbA1c", "Random_Blood_Sugar"},
    "Thyroid Profile": {"TSH", "Free_T4", "Free_T3", "Total_T4", "Total_T3"},
    "TFT": {"TSH", "Free_T4", "Free_T3", "Total_T4", "Total_T3"},
    "Vitamin Panel": {"Vitamin_D", "Vitamin_B12", "Calcium", "Ionized_Calcium", "Ferritin", "Folate"},
    "Stool Antigen": {"H_Pylori_Antigen", "Occult_Blood", "Stool_Routine"},
    "Comprehensive Metabolic Panel": None,
    "Mixed Diagnostic Panel": None
}


# ── 1. MULTI-TIER BIOMARKER SEVERITY CLASSIFICATION ───────────────────────────
def classify_biomarker_severity(marker_name: str, val: Any, gender: str = "Male", values_dict: Dict[str, Any] = None) -> Optional[dict]:
    """
    Classifies a single biomarker into calibrated clinical severity tiers:
    Critical | Severe | Moderate | Mild | Normal
    """
    if val is None or not isinstance(val, (int, float)):
        return None

    is_female = (gender or "").lower().startswith("f")

    # 1. Ionized Calcium (mg/dL)
    if marker_name in ("Ionized_Calcium", "Ion_Ca"):
        if val >= 4.6:
            return {"tier": "Normal", "label": "Normal Ionized Calcium (≥ 4.6 mg/dL)", "color": "#10b981", "is_abnormal": False, "condition": None}
        elif val >= 4.3:
            return {"tier": "Mild", "label": "Borderline hypocalcemia (4.3–4.5 mg/dL) — monitor and correlate clinically.", "color": "#f59e0b", "is_abnormal": True, "condition": "Borderline Hypocalcemia (Ionized Ca)"}
        elif val >= 4.0:
            return {"tier": "Moderate", "label": "Hypocalcemia (4.0–4.2 mg/dL) — requires clinical correlation.", "color": "#ef4444", "is_abnormal": True, "condition": "Hypocalcemia"}
        else:
            return {"tier": "Severe", "label": "Significant hypocalcemia (< 4.0 mg/dL) — urgent clinical review.", "color": "#dc2626", "is_abnormal": True, "condition": "Severe Hypocalcemia"}

    # 2. Total Serum Calcium (mg/dL)
    if marker_name in ("Calcium", "Serum_Calcium"):
        if val < 7.0:
            return {"tier": "Critical", "label": "Severe hypocalcemia (< 7.0 mg/dL) — urgent clinical review.", "color": "#dc2626", "is_abnormal": True, "condition": "Severe Hypocalcaemia"}
        elif val < 8.0:
            return {"tier": "Moderate", "label": "Hypocalcemia (7.0–7.9 mg/dL) — requires clinical correlation.", "color": "#ef4444", "is_abnormal": True, "condition": "Hypocalcaemia"}
        elif val < 8.5:
            return {"tier": "Mild", "label": "Borderline hypocalcemia (8.0–8.4 mg/dL) — monitor and correlate clinically.", "color": "#f59e0b", "is_abnormal": True, "condition": "Borderline Hypocalcaemia"}
        elif val <= 10.2:
            return {"tier": "Normal", "label": "Normal Serum Calcium (8.5–10.2 mg/dL)", "color": "#10b981", "is_abnormal": False, "condition": None}
        else:
            return {"tier": "Severe", "label": "Hypercalcaemia (> 10.2 mg/dL)", "color": "#dc2626", "is_abnormal": True, "condition": "Hypercalcaemia"}

    # 3. Gamma GT (GGT) (U/L)
    if marker_name in ("GGT", "Gamma_GT", "GGTP"):
        if val <= 50.0:
            return {"tier": "Normal", "label": "Normal GGT (≤ 50 U/L)", "color": "#10b981", "is_abnormal": False, "condition": None}
        elif val <= 75.0:  # Up to 1.5x upper limit
            return {"tier": "Mild", "label": "Mild GGT elevation — possible hepatic enzyme stress; correlate with other liver markers and history.", "color": "#f59e0b", "is_abnormal": True, "condition": "Mild GGT Elevation"}
        elif val <= 150.0: # 1.5 - 3x upper limit
            return {"tier": "Moderate", "label": "Moderate GGT elevation (1.5–3× upper limit) — correlate with clinical history.", "color": "#f97316", "is_abnormal": True, "condition": "Moderate GGT Elevation"}
        else:              # > 3x upper limit
            return {"tier": "Severe", "label": "Marked GGT elevation (> 3× upper limit) — hepatic review indicated.", "color": "#dc2626", "is_abnormal": True, "condition": "Marked GGT Elevation"}

    # 4. A/G Ratio
    if marker_name in ("AG_Ratio", "A_G_Ratio"):
        alb = (values_dict or {}).get("Albumin")
        tp = (values_dict or {}).get("Total_Protein")
        if val < 1.2:
            if alb is not None and alb >= 3.5:
                return {"tier": "Mild", "label": "Low A/G ratio — may reflect relative globulin increase; no clear evidence of protein deficiency.", "color": "#f59e0b", "is_abnormal": True, "condition": "Low A/G Ratio (Non-specific)"}
            elif alb is not None and tp is not None and alb < 3.5 and tp < 6.0:
                return {"tier": "Moderate", "label": "Low A/G ratio with Hypoalbuminaemia — Protein Deficiency pattern.", "color": "#ef4444", "is_abnormal": True, "condition": "Protein Deficiency"}
            else:
                return {"tier": "Mild", "label": "A/G ratio is low; pattern is non-specific and not diagnostic of protein deficiency alone.", "color": "#f59e0b", "is_abnormal": True, "condition": "Low A/G Ratio"}
        elif val <= 2.5:
            return {"tier": "Normal", "label": "Normal A/G Ratio (1.2–2.5)", "color": "#10b981", "is_abnormal": False, "condition": None}
        else:
            return {"tier": "Mild", "label": "High A/G Ratio", "color": "#3b82f6", "is_abnormal": False, "condition": None}

    # 5. Vitamin D (25-OH) (ng/mL)
    if marker_name in ("Vitamin_D", "Vit_D", "25_OH_Vitamin_D"):
        if val < 10.0:
            return {"tier": "Critical", "label": "Severe Vitamin D Deficiency (< 10 ng/mL)", "color": "#dc2626", "is_abnormal": True, "condition": "Severe Vitamin D Deficiency"}
        elif val < 20.0:
            return {"tier": "Moderate", "label": "Vitamin D Deficiency (10–20 ng/mL)", "color": "#ef4444", "is_abnormal": True, "condition": "Vitamin D Deficiency"}
        elif val < 30.0:
            return {"tier": "Mild", "label": "Vitamin D Insufficiency (20–30 ng/mL)", "color": "#f59e0b", "is_abnormal": True, "condition": "Vitamin D Insufficiency"}
        elif val <= 100.0:
            return {"tier": "Normal", "label": "Sufficient / Optimal (30–100 ng/mL)", "color": "#10b981", "is_abnormal": False, "condition": None}
        else:
            return {"tier": "Severe", "label": "Elevated / Potential Toxicity (> 100 ng/mL)", "color": "#f97316", "is_abnormal": True, "condition": "Hypervitaminosis D"}

    # 6. Hemoglobin (g/dL)
    if marker_name in ("Hemoglobin", "Hb", "Hgb"):
        cutoff = 12.0 if is_female else 13.0
        if val < 7.0:
            return {"tier": "Critical", "label": "Critical / Severe Anaemia (Hb < 7.0 g/dL)", "color": "#dc2626", "is_abnormal": True, "condition": "Severe Anaemia"}
        elif val < 10.0:
            return {"tier": "Moderate", "label": "Moderate Anaemia (Hb 7.0–9.9 g/dL)", "color": "#ef4444", "is_abnormal": True, "condition": "Anemia"}
        elif val < cutoff:
            return {"tier": "Mild", "label": f"Mild Anaemia (Hb 10.0–{cutoff - 0.1:.1f} g/dL)", "color": "#f59e0b", "is_abnormal": True, "condition": "Mild Anemia"}
        elif val <= (15.5 if is_female else 17.5):
            return {"tier": "Normal", "label": f"Normal Haemoglobin (≥ {cutoff} g/dL)", "color": "#10b981", "is_abnormal": False, "condition": None}
        else:
            return {"tier": "Moderate", "label": "Polycythaemia / High Hb", "color": "#f97316", "is_abnormal": True, "condition": "Polycythaemia"}

    # 7. Fasting Blood Sugar (mg/dL)
    if marker_name in ("Fasting_Blood_Sugar", "FBS", "Glucose"):
        if val >= 200:
            return {"tier": "Severe", "label": "Severe Hyperglycaemia (FBS ≥ 200 mg/dL)", "color": "#dc2626", "is_abnormal": True, "condition": "Diabetes Mellitus"}
        elif val >= 126:
            return {"tier": "Moderate", "label": "Diagnostic Diabetes Mellitus (FBS ≥ 126 mg/dL)", "color": "#ef4444", "is_abnormal": True, "condition": "Diabetes Mellitus"}
        elif val >= 100:
            return {"tier": "Mild", "label": "Impaired Fasting Glucose / Prediabetes (100–125 mg/dL)", "color": "#f59e0b", "is_abnormal": True, "condition": "Impaired Fasting Glucose (Prediabetes)"}
        elif val >= 70:
            return {"tier": "Normal", "label": "Normal Fasting Glucose (70–99 mg/dL)", "color": "#10b981", "is_abnormal": False, "condition": None}
        else:
            return {"tier": "Critical", "label": "Hypoglycaemia Risk (FBS < 70 mg/dL)", "color": "#dc2626", "is_abnormal": True, "condition": "Hypoglycaemia"}

    # 8. Creatinine (mg/dL)
    if marker_name in ("Creatinine", "Creat"):
        limit = 1.1 if is_female else 1.3
        if val > 2.5:
            return {"tier": "Critical", "label": "Severe Renal Impairment (Creatinine > 2.5 mg/dL)", "color": "#dc2626", "is_abnormal": True, "condition": "Impaired Renal Function"}
        elif val > limit:
            return {"tier": "Moderate", "label": f"Elevated Creatinine ({val} mg/dL, > {limit})", "color": "#ef4444", "is_abnormal": True, "condition": "Impaired Renal Function"}
        elif val >= (limit - 0.15):
            return {"tier": "Mild", "label": "High-Normal Creatinine (Borderline)", "color": "#f59e0b", "is_abnormal": False, "condition": None}
        elif val >= 0.5:
            return {"tier": "Normal", "label": "Normal Renal Function", "color": "#10b981", "is_abnormal": False, "condition": None}
        else:
            return {"tier": "Mild", "label": "Low Serum Creatinine", "color": "#3b82f6", "is_abnormal": False, "condition": None}

    # 9. TSH (mIU/L)
    if marker_name in ("TSH", "Thyroid_Stimulating_Hormone"):
        if val > 10.0:
            return {"tier": "Severe", "label": "Overt Hypothyroidism (TSH > 10 mIU/L)", "color": "#dc2626", "is_abnormal": True, "condition": "Hypothyroidism"}
        elif val > 4.5:
            return {"tier": "Moderate", "label": "Subclinical Hypothyroidism (TSH 4.5–10 mIU/L)", "color": "#ef4444", "is_abnormal": True, "condition": "Hypothyroidism"}
        elif val < 0.1:
            return {"tier": "Severe", "label": "Overt Hyperthyroidism (TSH < 0.1 mIU/L)", "color": "#dc2626", "is_abnormal": True, "condition": "Hyperthyroidism"}
        elif val < 0.4:
            return {"tier": "Moderate", "label": "Subclinical Hyperthyroidism (TSH 0.1–0.4 mIU/L)", "color": "#f59e0b", "is_abnormal": True, "condition": "Hyperthyroidism"}
        else:
            return {"tier": "Normal", "label": "Euthyroid (TSH 0.4–4.5 mIU/L)", "color": "#10b981", "is_abnormal": False, "condition": None}

    # 10. Platelets (x10^3 / uL)
    if marker_name in ("Platelets", "PLT", "Plt"):
        if val < 50:
            return {"tier": "Critical", "label": "Critical Thrombocytopenia (PLT < 50k, High Bleed Risk)", "color": "#dc2626", "is_abnormal": True, "condition": "Thrombocytopenia"}
        elif val < 100:
            return {"tier": "Moderate", "label": "Moderate Thrombocytopenia (PLT 50k–99k)", "color": "#ef4444", "is_abnormal": True, "condition": "Thrombocytopenia"}
        elif val < 150:
            return {"tier": "Mild", "label": "Mild Thrombocytopenia (PLT 100k–149k)", "color": "#f59e0b", "is_abnormal": True, "condition": "Thrombocytopenia"}
        elif val <= 450:
            return {"tier": "Normal", "label": "Normal Platelet Count", "color": "#10b981", "is_abnormal": False, "condition": None}
        else:
            return {"tier": "Moderate", "label": "Thrombocytosis (PLT > 450k)", "color": "#f97316", "is_abnormal": True, "condition": "Thrombocytosis"}

    # 11. WBC (x10^3 / uL)
    if marker_name in ("WBC", "White_Blood_Cells"):
        if val > 20.0:
            return {"tier": "Critical", "label": "Critical Leukocytosis (WBC > 20k)", "color": "#dc2626", "is_abnormal": True, "condition": "Possible Infection / Leukocytosis"}
        elif val > 11.0:
            return {"tier": "Moderate", "label": "Leukocytosis / Active Inflammation (WBC > 11k)", "color": "#ef4444", "is_abnormal": True, "condition": "Possible Infection / Leukocytosis"}
        elif val < 3.5:
            return {"tier": "Moderate", "label": "Leukopenia (WBC < 3.5k)", "color": "#f97316", "is_abnormal": True, "condition": "Leukopenia"}
        elif val <= 11.0:
            return {"tier": "Normal", "label": "Normal WBC Count", "color": "#10b981", "is_abnormal": False, "condition": None}

    # 12. Cholesterol (mg/dL)
    if marker_name in ("Cholesterol", "Total_Cholesterol", "Chol"):
        if val >= 240:
            return {"tier": "Severe", "label": "Severe Hypercholesterolaemia (≥ 240 mg/dL)", "color": "#dc2626", "is_abnormal": True, "condition": "Hypercholesterolaemia"}
        elif val >= 200:
            return {"tier": "Mild", "label": "Borderline Elevated Cholesterol (200–239 mg/dL)", "color": "#f59e0b", "is_abnormal": True, "condition": "Hypercholesterolaemia"}
        else:
            return {"tier": "Normal", "label": "Desirable Cholesterol (< 200 mg/dL)", "color": "#10b981", "is_abnormal": False, "condition": None}

    return None


def classify_all_biomarkers(values: Dict[str, Any], extracted_flags: Dict[str, bool], gender: str = "Male") -> List[dict]:
    """
    Classifies all verified extracted biomarkers into structured severity records.
    """
    severities = []
    for k, val in values.items():
        if val is not None and extracted_flags.get(k, False):
            info = classify_biomarker_severity(k, val, gender, values)
            if info:
                severities.append({
                    "biomarker": k.replace("_", " "),
                    "raw_key": k,
                    "value": val,
                    **info
                })
    return severities


# ── 2. CLINICAL ACTION PLANS GENERATOR ─────────────────────────────────────────
def generate_clinical_action_plans(severities: List[dict]) -> List[dict]:
    """
    Generates actionable, step-by-step clinical management protocols.
    """
    plans = []
    seen = set()

    for s in severities:
        cond = s.get("condition")
        if not cond or cond in seen:
            continue
        seen.add(cond)
        val = s.get("value")
        tier = s.get("tier")

        # Hypocalcemia (Ionized / Total Calcium)
        if "Hypocalcemia" in cond or "Hypocalcaemia" in cond:
            if tier == "Mild":
                plans.append({
                    "condition": "Borderline Hypocalcemia Monitoring Plan",
                    "severity": s.get("label"),
                    "color": s.get("color"),
                    "steps": [
                        "Clinical Assessment: Correlate with symptoms of neuromuscular irritability (perioral numbness, paresthesia, muscle cramps).",
                        "Dietary Calcium Support: Increase dietary calcium intake (dairy products, fortified plant milks, leafy greens).",
                        "Laboratory Follow-up: Repeat serum Ionized Calcium, Total Calcium, and Albumin in 2–4 weeks."
                    ]
                })
            else:
                plans.append({
                    "condition": "Hypocalcemia Clinical Protocol",
                    "severity": s.get("label"),
                    "color": s.get("color"),
                    "steps": [
                        "Diagnostic Workup: Measure Parathyroid Hormone (PTH), 25-OH Vitamin D, Serum Magnesium, and Phosphate.",
                        "Calcium Replacement: Oral Calcium Carbonate (500–1000 mg elemental calcium daily in divided doses) with active Vitamin D if indicated.",
                        "Clinical Review: Check ECG for QTc prolongation if symptomatic.",
                        "Follow-up: Repeat Calcium and Magnesium in 1–2 weeks."
                    ]
                })

        # GGT Elevation
        elif "GGT" in cond:
            if tier == "Mild":
                plans.append({
                    "condition": "Mild GGT Enzyme Follow-up",
                    "severity": s.get("label"),
                    "color": s.get("color"),
                    "steps": [
                        "History Correlation: Review alcohol consumption, over-the-counter supplements, and hepatotoxic medications.",
                        "Observation: Isolated mild GGT is non-specific; no invasive investigation required without other enzyme abnormalities.",
                        "Repeat Panel: Recheck comprehensive Liver Function Panel (ALT/AST, Bilirubin, GGT) in 4–8 weeks."
                    ]
                })
            else:
                plans.append({
                    "condition": "Hepatic Enzyme Investigation Protocol",
                    "severity": s.get("label"),
                    "color": s.get("color"),
                    "steps": [
                        "Comprehensive Liver Workup: Order viral hepatitis serology (HBsAg, Anti-HCV), abdominal ultrasound, and lipid panel.",
                        "Lifestyle Intervention: Weight optimization, avoidance of hepatotoxic substances and alcohol.",
                        "Clinical Follow-up: Repeat LFTs in 4 weeks; consider gastroenterology referral if enzymes remain persistently > 2× upper limit."
                    ]
                })

        # Vitamin D Deficiency / Insufficiency
        elif "Vitamin D" in cond:
            if tier in ("Critical", "Severe", "Moderate"):
                plans.append({
                    "condition": "Vitamin D Deficiency Replenishment Protocol",
                    "severity": s.get("label"),
                    "color": s.get("color"),
                    "steps": [
                        "Pharmacotherapy: Cholecalciferol (Vitamin D3) 50,000 IU orally once weekly for 8 weeks (or 4,000 IU daily).",
                        "Maintenance: Transition to 1,000–2,000 IU daily once sufficiency (> 30 ng/mL) is achieved.",
                        "Phototherapy / Sunlight: 15–20 minutes daily morning sun exposure (between 9:00 AM – 11:00 AM) with arms and legs unshielded.",
                        "Follow-up: Repeat 25-OH Vitamin D serum level and total calcium in 8–12 weeks."
                    ]
                })
            else:
                plans.append({
                    "condition": "Vitamin D Insufficiency Guidance",
                    "severity": s.get("label"),
                    "color": s.get("color"),
                    "steps": [
                        "Supplementation: Cholecalciferol 1,000–2,000 IU orally daily with a fat-containing meal.",
                        "Sunlight: Daily morning outdoor exposure for 15 minutes.",
                        "Follow-up: Recheck 25-OH Vitamin D in 12 weeks."
                    ]
                })

        # Anaemia
        elif "Anaemia" in cond or "Anemia" in cond:
            plans.append({
                "condition": "Anaemia Investigation & Replenishment Plan",
                "severity": s.get("label"),
                "color": s.get("color"),
                "steps": [
                    "Iron Replenishment: Oral Ferrous Fumarate / Sulfate 200mg daily taken with Vitamin C to maximize intestinal absorption.",
                    "Diagnostic Studies: Complete Iron panel (Serum Ferritin, TIBC, Iron) and Peripheral Blood Smear for RBC morphology.",
                    "Occult Blood Loss Screening: Perform stool occult blood test (FOBT) and investigate GI/menstrual blood loss.",
                    "Monitoring: Repeat Full Blood Count (Hb/RBC/MCV) in 4 weeks."
                ]
            })

        # Diabetes / Prediabetes
        elif "Diabetes" in cond or "Prediabetes" in cond or "Glucose" in cond:
            plans.append({
                "condition": "Glycaemic Optimization Protocol",
                "severity": s.get("label"),
                "color": s.get("color"),
                "steps": [
                    "Diagnostic Confirmation: Order venous HbA1c to establish 3-month glycaemic load.",
                    "Medical Nutrition Therapy: Structured low-glycaemic index meal planning, eliminating refined carbohydrates.",
                    "Physical Activity: Minimum 150 minutes/week of moderate-intensity aerobic exercise.",
                    "Follow-up: Repeat Fasting Glucose and HbA1c in 12 weeks."
                ]
            })

        # Thyroid Dysfunction
        elif "Hypothyroidism" in cond or "Hyperthyroidism" in cond or "Thyroid" in cond:
            plans.append({
                "condition": "Thyroid Function Action Plan",
                "severity": s.get("label"),
                "color": s.get("color"),
                "steps": [
                    "Confirmatory Panel: Repeat TSH together with Free T4 and Free T3 in 6–8 weeks.",
                    "Autoimmune Screening: Measure Anti-Thyroperoxidase (Anti-TPO) antibodies.",
                    "Management: Consider Levothyroxine titration if symptomatic or if TSH remains elevated above 10 mIU/L."
                ]
            })

        # Renal Impairment
        elif "Renal" in cond or "Kidney" in cond:
            plans.append({
                "condition": "Renal Protection & Staging Protocol",
                "severity": s.get("label"),
                "color": s.get("color"),
                "steps": [
                    "Renal Biomarker Workup: Quantify urine Albumin-to-Creatinine Ratio (uACR).",
                    "Nephroprotection: Strictly avoid nephrotoxic agents (NSAIDs e.g. diclofenac, ibuprofen).",
                    "Monitoring: Repeat Serum Creatinine, eGFR, and Electrolytes in 4–8 weeks."
                ]
            })

    return plans


# ── 3. BIOMARKER-DIAGNOSIS HARD VALIDATION RULES ──────────────────────────────
def validate_condition_against_biomarkers(
    condition: str,
    parsed_values: Dict[str, Any],
    extracted_flags: Dict[str, bool],
    gender: str = "Male",
    age: int = 45
) -> Tuple[bool, Optional[str]]:
    """
    Evaluates whether a predicted condition is medically justified by extracted biomarkers.
    Returns (is_valid, rejection_reason).
    """
    c_low = condition.lower().strip()
    hb = parsed_values.get("Hemoglobin")
    fbs = parsed_values.get("Fasting_Blood_Sugar")
    tsh = parsed_values.get("TSH")
    creat = parsed_values.get("Creatinine")
    chol = parsed_values.get("Cholesterol")
    plt = parsed_values.get("Platelets")
    wbc = parsed_values.get("WBC")
    vit_d = parsed_values.get("Vitamin_D")
    ion_ca = parsed_values.get("Ionized_Calcium")
    tot_ca = parsed_values.get("Calcium")
    alb = parsed_values.get("Albumin")
    tp = parsed_values.get("Total_Protein")
    alt = parsed_values.get("ALT")
    ast = parsed_values.get("AST")
    alp = parsed_values.get("ALP")
    bili = parsed_values.get("Bilirubin")
    ggt = parsed_values.get("GGT")
    egfr_val = parsed_values.get("eGFR")

    # 1. Chronic Kidney Disease / Renal Impairment
    if "chronic kidney disease" in c_low or "ckd" in c_low or "renal" in c_low or "kidney" in c_low:
        limit = 1.1 if gender == "Female" else 1.3
        if creat is not None and creat <= limit and (egfr_val is None or egfr_val >= 60):
            return False, "No evidence of chronic kidney disease based on current renal markers."
        return True, None

    # 2. Liver or Bile Duct Disease (Strict multi-marker requirement)
    if "liver" in c_low or "bile duct" in c_low or "hepatic" in c_low:
        abnormal_liver = 0
        if alt is not None and alt > 55: abnormal_liver += 1
        if ast is not None and ast > 45: abnormal_liver += 1
        if bili is not None and bili > 1.2: abnormal_liver += 1
        if alp is not None and alp > 140: abnormal_liver += 1
        if ggt is not None and ggt > 150: abnormal_liver += 1  # Marked GGT
        if abnormal_liver < 2:
            return False, "Isolated mild liver variation alone is not diagnostic of liver or bile duct disease."
        return True, None

    # 3. Protein Deficiency
    if "protein deficiency" in c_low or "hypoproteinemia" in c_low:
        if alb is not None and alb >= 3.5:
            return False, "Albumin is normal (≥ 3.5 g/dL); low A/G ratio alone is not diagnostic of protein deficiency."
        if tp is not None and tp >= 6.0:
            return False, "Total Protein is within normal limits; protein deficiency ruled out."
        return True, None

    # 4. Hypocalcaemia / Hypocalcemia
    if "hypocalcemia" in c_low or "hypocalcaemia" in c_low:
        if ion_ca is not None and ion_ca >= 4.6:
            return False, f"Ionized Calcium ({ion_ca} mg/dL) is normal (≥ 4.6 mg/dL). Hypocalcemia ruled out."
        if tot_ca is not None and tot_ca >= 8.5:
            return False, f"Serum Calcium ({tot_ca} mg/dL) is normal (≥ 8.5 mg/dL). Hypocalcemia ruled out."
        return True, None

    # 5. Anaemia
    if "anemia" in c_low or "anaemia" in c_low:
        threshold = 12.0 if gender == "Female" else 13.0
        if hb is not None and hb >= threshold:
            return False, f"Hemoglobin ({hb} g/dL) is normal (≥ {threshold} g/dL). Anemia ruled out."
        return True, None

    # 6. Diabetes Mellitus / Hyperglycemia
    if "diabetes" in c_low or "hyperglycemia" in c_low:
        if fbs is not None and fbs < 100:
            return False, f"Fasting Glucose ({fbs} mg/dL) is normal (< 100 mg/dL). Diabetes ruled out."
        return True, None

    # 7. Prediabetes / Impaired Fasting Glucose
    if "prediabetes" in c_low or "impaired fasting" in c_low:
        if fbs is not None and (fbs < 100 or fbs >= 126):
            return False, f"FBS ({fbs} mg/dL) is not in prediabetes range (100–125 mg/dL)."
        return True, None

    # 8. Hypothyroidism
    if "hypothyroidism" in c_low or "hypothyroid" in c_low:
        if tsh is not None and tsh <= 4.5:
            return False, f"TSH ({tsh} mIU/L) is not elevated (≤ 4.5 mIU/L). Hypothyroidism ruled out."
        return True, None

    # 9. Hypercholesterolaemia / Dyslipidemia
    if "cholesterol" in c_low or "dyslipidemia" in c_low:
        if chol is not None and chol < 200:
            return False, f"Total Cholesterol ({chol} mg/dL) is desirable (< 200 mg/dL). Hypercholesterolemia ruled out."
        return True, None

    # 10. Thrombocytopenia
    if "thrombocytopenia" in c_low:
        if plt is not None and plt >= 150:
            return False, f"Platelet count ({plt} ×10³/µL) is normal (≥ 150). Thrombocytopenia ruled out."
        return True, None

    # 11. Vitamin D Deficiency
    if "vitamin d" in c_low:
        if vit_d is not None and vit_d >= 30.0:
            return False, f"Vitamin D ({vit_d} ng/mL) is optimal (≥ 30 ng/mL). Deficiency ruled out."
        return True, None

    return True, None


# ── 4. CALIBRATED ENSEMBLE CONSENSUS WITH DISAGREEMENT PENALTY ────────────────
def compute_validated_ensemble(
    m1_rules: dict,
    m2_ml: dict,
    m3_nvidia: dict,
    m4_gemini: dict,
    report_type: str,
    parsed_values: dict,
    extracted_flags: dict,
    gender: str = "Male",
    age: int = 45
) -> dict:
    """
    Arbitrates consensus across 4 models with:
    1. Calibrated Model Weights: Gemini (40%), NIM (20%), XGBoost (25%), Rules (15%)
    2. Strict Disease Flagging Rules (≥ 2 related biomarkers for multi-system disease)
    3. Model Disagreement Penalty (caps confidence at 60–70% when models conflict)
    4. Nuanced Primary Findings vs Strong Disease Labels
    5. Actionable Follow-up Plans
    """
    WEIGHTS = {
        "model_4_gemini": 0.40,
        "model_3_nvidia": 0.20,  # Lowered from 0.35 due to over-sensitivity
        "model_2_ml":     0.25,  # Increased from 0.20 as anchor baseline
        "model_1_rules":  0.15,  # Increased from 0.05 as final clinical arbiter
    }

    # 1. Classify all verified extracted biomarkers into clinical severities
    biomarker_severities = classify_all_biomarkers(parsed_values, extracted_flags, gender)
    
    # Identify high/moderate severity pathologies from ground-truth lab measurements
    lab_pathologies = []
    mild_findings = []
    has_critical = False
    has_severe = False
    has_moderate = False

    for s in biomarker_severities:
        if s.get("is_abnormal") and s.get("condition"):
            tier = s.get("tier")
            if tier in ("Critical", "Severe", "Moderate"):
                lab_pathologies.append(s["condition"])
                if tier == "Critical": has_critical = True
                elif tier == "Severe": has_severe = True
                elif tier == "Moderate": has_moderate = True
            elif tier == "Mild":
                mild_findings.append(s.get("label") or s["condition"])

    # 2. Evaluate model outputs and sanitize against ground-truth
    model_inputs = [
        ("model_4_gemini", m4_gemini),
        ("model_3_nvidia", m3_nvidia),
        ("model_2_ml", m2_ml),
        ("model_1_rules", m1_rules),
    ]

    all_stripped_reasons = []
    validated_model_votes = {}
    active_weight = 0.0
    condition_weights = {}
    arbitration_notes = []

    # Model severity scores for disagreement variance:
    # 0 = Normal / Healthy, 1 = Mild Abnormality, 2 = Disease
    model_severity_scores = {}

    for key, model_data in model_inputs:
        if not model_data or model_data.get("error") or model_data.get("diagnosis") in ("Offline", "N/A"):
            continue
            
        diag = model_data.get("diagnosis", "")
        if "(" in diag:
            diag = diag.split("(")[0].strip()
            
        raw_list = [p.strip() for p in diag.split(",") if p.strip()]
        
        # Validate against true biomarkers
        valid_conditions = []
        for cond in raw_list:
            if not cond or cond.lower() in ("n/a", "none", "unknown", "offline", "[]"):
                continue
            is_valid, reject_reason = validate_condition_against_biomarkers(
                cond, parsed_values, extracted_flags, gender, age
            )
            if is_valid:
                valid_conditions.append(cond.strip())
            else:
                all_stripped_reasons.append(f"[{key}] Overruled '{cond}': {reject_reason}")

        # Post-processing calibration for NIM (Model 3)
        # If NIM predicts strong disease but XGBoost + Rules say Healthy/Normal, downgrade NIM
        if key == "model_3_nvidia" and not lab_pathologies:
            if m2_ml.get("diagnosis") in ("Healthy", "Normal") and m1_rules.get("diagnosis") in ("Healthy", "Normal", "No significant abnormalities detected"):
                downgraded_conditions = [c for c in valid_conditions if "disease" not in c.lower() and "chronic" not in c.lower()]
                if len(downgraded_conditions) < len(valid_conditions):
                    arbitration_notes.append("Post-Processing Calibration: Downgraded over-sensitive NVIDIA NIM disease prediction because local ML and Rules confirmed normal physiology.")
                valid_conditions = downgraded_conditions

        w = WEIGHTS[key]
        active_weight += w
        
        validated_model_votes[key] = {
            "raw_diagnosis": diag,
            "validated_conditions": valid_conditions,
            "weight": w
        }
        
        # Score model output for disagreement variance
        if any("severe" in c.lower() or "disease" in c.lower() or "overt" in c.lower() for c in valid_conditions):
            model_severity_scores[key] = 2
        elif valid_conditions and not all(c in ["Healthy", "Normal", "No major abnormalities detected"] for c in valid_conditions):
            model_severity_scores[key] = 1
        else:
            model_severity_scores[key] = 0

        for c in valid_conditions:
            condition_weights[c] = condition_weights.get(c, 0.0) + w

    # 3. ── DIAGNOSIS PRIORITIZATION ENGINE & ARBITRATION ──
    # If ground-truth laboratory biomarkers show pathology (e.g. Vitamin D deficiency, Anemia, Diabetes),
    # HARD OVERRIDE any conservative "Healthy" / "Normal" votes!
    if lab_pathologies or mild_findings:
        for normal_term in ["Healthy", "Normal", "No major abnormalities detected", "No significant abnormalities detected"]:
            if normal_term in condition_weights:
                del condition_weights[normal_term]
                target_names = lab_pathologies if lab_pathologies else mild_findings
                arbitration_notes.append(f"Model Disagreement Arbitrated: Overruled '{normal_term}' because laboratory values confirm active findings ({', '.join(target_names[:2])}).")

        for lp in lab_pathologies:
            condition_weights[lp] = max(condition_weights.get(lp, 0.0), active_weight * 0.75)
        for mf in mild_findings:
            if not lab_pathologies:
                condition_weights[mf] = max(condition_weights.get(mf, 0.0), active_weight * 0.65)

    # 4. Handle all-normal case
    if not lab_pathologies and not mild_findings and (not condition_weights or all(c in ["Healthy", "Normal", "No major abnormalities detected"] for c in condition_weights)):
        return {
            "primary_diagnosis": "No major abnormalities detected",
            "confidence": 95.0,
            "confidence_band": "HIGH",
            "vote_breakdown": validated_model_votes,
            "all_votes": {"No major abnormalities detected": 100.0},
            "safety_flags": [],
            "biomarker_severities": biomarker_severities,
            "action_plans": [],
            "arbitration_notes": arbitration_notes,
            "stripped_hallucinations": all_stripped_reasons
        }

    # 5. ── DISAGREEMENT PENALTY CALCULATION ──
    # Compute variance / spread across models (0, 1, 2)
    scores = list(model_severity_scores.values())
    disagreement_detected = False
    disagreement_factor = 0.0

    if len(scores) >= 2:
        mean_score = sum(scores) / len(scores)
        variance = sum((x - mean_score) ** 2 for x in scores) / len(scores)
        
        # High disagreement: e.g. one model predicts disease (2) while others predict healthy (0)
        if variance >= 0.5 or (max(scores) == 2 and min(scores) == 0):
            disagreement_detected = True
            disagreement_factor = min(0.35, round(variance * 0.25, 2))
            arbitration_notes.append("Ensemble Disagreement Penalty Applied: Model outputs show partial discordance; confidence calibrated to decision-support range.")

    # Compute base confidence
    if has_critical or has_severe:
        base_conf = 95.0
    elif has_moderate:
        base_conf = 88.0
    else:
        avg_w = sum(condition_weights.values()) / (len(condition_weights) * max(active_weight, 0.01))
        base_conf = min(92.0, max(55.0, avg_w * 100.0))

    # Apply disagreement penalty
    if disagreement_detected:
        confidence_pct = round(min(68.0, base_conf * (1.0 - disagreement_factor)), 1)
        band = "MODERATE"
    else:
        confidence_pct = round(base_conf, 1)
        band = "HIGH" if confidence_pct >= 80.0 else "MODERATE" if confidence_pct >= 65.0 else "INCONCLUSIVE"

    # Build final prioritized primary diagnosis string
    prioritized_conditions = []
    seen_lower = set()

    for lp in lab_pathologies:
        if lp.lower() not in seen_lower:
            seen_lower.add(lp.lower())
            prioritized_conditions.append(lp)

    for mf in mild_findings:
        if not lab_pathologies and mf.lower() not in seen_lower:
            seen_lower.add(mf.lower())
            prioritized_conditions.append(mf)

    for c in condition_weights:
        if c.lower() not in seen_lower and c not in ["Healthy", "Normal", "No major abnormalities detected", "No significant abnormalities detected"]:
            seen_lower.add(c.lower())
            prioritized_conditions.append(c)

    primary_diagnosis = ", ".join(prioritized_conditions[:3]) if prioritized_conditions else "Clinical Correlation Required"

    # Generate clinical action plans
    action_plans = generate_clinical_action_plans(biomarker_severities)

    return {
        "primary_diagnosis": primary_diagnosis,
        "confidence": confidence_pct,
        "confidence_band": band,
        "vote_breakdown": validated_model_votes,
        "all_votes": {k: round((v / max(active_weight, 0.01)) * 100.0, 1) for k, v in condition_weights.items()},
        "safety_flags": [] if confidence_pct >= 70.0 else ["Moderate confidence — models show partial disagreement; findings require clinical correlation."],
        "biomarker_severities": biomarker_severities,
        "action_plans": action_plans,
        "arbitration_notes": arbitration_notes,
        "stripped_hallucinations": list(dict.fromkeys(all_stripped_reasons))
    }
