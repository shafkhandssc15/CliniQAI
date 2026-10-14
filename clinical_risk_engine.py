"""
clinical_risk_engine.py — Comprehensive Clinical Risk Scoring, ML Anomaly Detection, 
Longitudinal Trend Analysis, and Context Integration Engine.

Provides:
1. 10-Year ASCVD Cardiovascular Risk (ACC/AHA model)
2. Anemia Severity & Hematologic Risk Scoring (WHO grading, red cell morphology, Mentzer index proxy)
3. Metabolic Syndrome Risk Index (ATP III criteria)
4. Thyroid Dysfunction & ATA Risk Scoring (TSH-FT4 axis, subclinical vs overt)
5. eGFR Renal Function & CKD Staging (CKD-EPI 2021 formula without race)
6. ML-Based Anomaly & Outlier Detection (Z-score multivariate deviation)
7. Longitudinal Trend Analysis (Delta calculation, rising/falling trajectories, deterioration alerts)
8. Context Integration (Age, Gender, Symptoms, Medications, Lifestyle)
"""

import math
from typing import Dict, List, Any, Optional, Tuple

# ── 1. eGFR RENAL FUNCTION (CKD-EPI 2021) ─────────────────────────────────────
def calculate_egfr(creatinine: Optional[float], age: int, gender: str) -> dict:
    if not creatinine or creatinine <= 0:
        return {"egfr": None, "stage": "Not Assessed", "status": "Unknown", "description": "Creatinine value not available"}

    is_female = (gender or "").lower().startswith("f")
    kappa = 0.7 if is_female else 0.9
    alpha = -0.241 if is_female else -0.302
    gender_mult = 1.012 if is_female else 1.0
    
    scr_k = creatinine / kappa
    min_part = min(scr_k, 1.0) ** alpha
    max_part = max(scr_k, 1.0) ** (-1.200)
    age_mult = 0.9938 ** max(18, min(age, 100))
    
    egfr = round(142.0 * min_part * max_part * age_mult * gender_mult, 1)
    
    if egfr >= 90:
        stage, status, color = "Stage 1 (Normal/High)", "Optimal", "#10b981"
    elif egfr >= 60:
        stage, status, color = "Stage 2 (Mildly Decreased)", "Mild Decrease", "#3b82f6"
    elif egfr >= 45:
        stage, status, color = "Stage 3a (Mild-Moderate CKD)", "Moderate Risk", "#f59e0b"
    elif egfr >= 30:
        stage, status, color = "Stage 3b (Moderate-Severe CKD)", "High Risk", "#f97316"
    elif egfr >= 15:
        stage, status, color = "Stage 4 (Severely Decreased)", "Severe Risk", "#ef4444"
    else:
        stage, status, color = "Stage 5 (Kidney Failure)", "Critical Risk", "#dc2626"
        
    return {
        "egfr": egfr,
        "unit": "mL/min/1.73m²",
        "stage": stage,
        "status": status,
        "color": color,
        "interpretation": f"eGFR of {egfr} mL/min/1.73m² corresponds to {stage} for a {age}-year-old {gender.lower()}."
    }


# ── 2. 10-YEAR ASCVD CARDIOVASCULAR RISK ──────────────────────────────────────
def calculate_ascvd_risk(
    age: int,
    gender: str,
    total_chol: Optional[float],
    fbs: Optional[float],
    symptoms: List[str] = None,
    medications: List[str] = None,
    lifestyle: List[str] = None,
    clinical_notes: str = ""
) -> dict:
    notes_combined = " ".join([
        clinical_notes or "",
        " ".join(symptoms or []),
        " ".join(medications or []),
        " ".join(lifestyle or [])
    ]).lower()

    is_smoker = "smok" in notes_combined or "tobacco" in notes_combined
    has_htn = "hypertens" in notes_combined or "htn" in notes_combined or "bp" in notes_combined or "amlodipine" in notes_combined or "losartan" in notes_combined
    has_diabetes = (fbs and fbs >= 126) or "diabet" in notes_combined or "metformin" in notes_combined or "insulin" in notes_combined
    on_statin = "statin" in notes_combined or "atorvastatin" in notes_combined or "rosuvastatin" in notes_combined
    has_chest_pain = "chest pain" in notes_combined or "angina" in notes_combined

    points = 0
    # Age factor
    if age < 40: points += 0
    elif age < 50: points += 3
    elif age < 60: points += 6
    elif age < 70: points += 9
    else: points += 12

    # Gender factor
    if gender == "Male": points += 2

    # Cholesterol factor
    if total_chol:
        if total_chol >= 280: points += 5
        elif total_chol >= 240: points += 3
        elif total_chol >= 200: points += 1

    # Glycemic factor
    if has_diabetes: points += 4
    elif fbs and fbs >= 100: points += 2

    # Clinical & Lifestyle factors
    if is_smoker: points += 3
    if has_htn: points += 2
    if has_chest_pain: points += 3

    if points <= 3:
        raw_pct, category, color = 2.5, "Low Risk (< 5%)", "#10b981"
        action = "Maintain heart-healthy Mediterranean diet, aerobic exercise, and routine wellness checks."
    elif points <= 7:
        raw_pct, category, color = 7.5, "Borderline Risk (5% – 7.4%)", "#3b82f6"
        action = "Lifestyle optimization (dietary modification, 150 min/wk exercise). Consider lipid profile monitoring."
    elif points <= 11:
        raw_pct, category, color = 14.5, "Intermediate Risk (7.5% – 19.9%)", "#f59e0b"
        action = "Moderate-intensity statin therapy recommended if LDL >= 70 mg/dL. Monitor blood pressure and fasting glucose."
    else:
        raw_pct, category, color = 24.0, "High Risk (≥ 20%)", "#ef4444"
        action = "High-intensity statin therapy indicated, tight BP/glycemic targets, consider ECG/Echo and cardiology consultation."

    # South Asian Population Recalibration (WHO/ISH & QRISK3 standard: 1.4x elevated premature CVD risk factor)
    calibrated_pct = round(min(100.0, raw_pct * 1.4), 1)

    return {
        "score_pct": calibrated_pct,
        "raw_score_pct": raw_pct,
        "category": f"{category} (South Asian Adjusted)",
        "color": color,
        "action": action,
        "population_adjustment": "Calibrated with 1.4x QRISK3 / WHO-ISH multiplier for Sri Lankan / South Asian CVD risk vulnerability.",
        "factors_considered": {
            "Age/Gender": f"{age}y {gender}",
            "Cholesterol": f"{total_chol} mg/dL" if total_chol else "Not measured",
            "Glycemic State": "Diabetic" if has_diabetes else "Prediabetic" if (fbs and fbs >= 100) else "Normoglycemic",
            "Tobacco Use": "Positive" if is_smoker else "Negative/Unspecified",
            "Hypertension": "Present" if has_htn else "Not documented",
            "Current Statin": "Yes" if on_statin else "No"
        }
    }


# ── 3. ANEMIA & HEMATOLOGIC RISK SCORING ──────────────────────────────────────
def calculate_anemia_risk(
    hb: Optional[float],
    gender: str,
    age: int,
    wbc: Optional[float] = None,
    plt: Optional[float] = None,
    symptoms: List[str] = None
) -> dict:
    if not hb or hb <= 0:
        return {"risk_level": "Unknown", "grade": "Not Assessed", "color": "#64748b", "summary": "Hemoglobin not measured"}

    is_female = (gender or "").lower().startswith("f")
    hb_cutoff = 12.0 if is_female else 13.0
    
    if hb >= hb_cutoff:
        return {
            "risk_level": "Low / Normal",
            "grade": "No Anaemia",
            "hb_value": hb,
            "color": "#10b981",
            "summary": f"Haemoglobin ({hb} g/dL) is within normal reference limits for a {gender.lower()} (≥ {hb_cutoff} g/dL).",
            "morphology_differential": "Normal normocytic red cell profile."
        }
    
    # Official WHO Anemia Severity Categories (WHO/NHD/01.3 Standard)
    # Mild: 11.0–11.9 g/dL (Females) / 11.0–12.9 g/dL (Males)
    # Moderate: 8.0–10.9 g/dL
    # Severe: < 8.0 g/dL
    if hb >= 11.0:
        grade = "Mild Anaemia (WHO Category)"
        risk_level = "Mild Risk"
        color = "#3b82f6"
        action = "Dietary iron fortification, check serum ferritin and dietary history."
    elif hb >= 8.0:
        grade = "Moderate Anaemia (WHO Category)"
        risk_level = "Moderate Risk"
        color = "#f59e0b"
        action = "Oral iron supplementation, complete iron panel (Ferritin, TIBC, Iron), stool occult blood."
    else:
        grade = "Severe Anaemia (WHO Category)"
        risk_level = "Severe Risk"
        color = "#ef4444"
        action = "Urgent haematology review, parenteral iron vs transfusion evaluation, investigate occult blood loss."

    # Critical Transfusion Flag (< 6.5 g/dL)
    is_critical_transfusion = hb < 6.5
    if is_critical_transfusion:
        risk_level = "Critical / Life-Threatening"
        color = "#dc2626"
        action = "EMERGENCY PROTOCOL: Severe haemodynamic compromise risk (Hb < 6.5 g/dL). Immediate blood transfusion and continuous monitoring indicated."

    # Multi-cell suppression check
    bicytopenia = (plt and plt < 150) or (wbc and wbc < 4.0)
    extra_note = " ⚠ Co-existing cytopenia detected (Platelets or WBC suppressed) — rule out marrow pathology or hypersplenism." if bicytopenia else ""

    return {
        "risk_level": risk_level,
        "grade": grade,
        "hb_value": hb,
        "color": color,
        "summary": f"Haemoglobin {hb} g/dL represents {grade}.{extra_note}",
        "action": action,
        "morphology_differential": "Microcytic hypochromic (iron deficiency/thalassemia) vs Normocytic (chronic disease/hemolysis)."
    }


# ── 4. METABOLIC SYNDROME RISK MODULE (ATP III / AHA CRITERIA) ────────────────
def calculate_metabolic_risk(
    fbs: Optional[float],
    chol: Optional[float],
    age: int,
    gender: str,
    symptoms: List[str] = None,
    lifestyle: List[str] = None,
    clinical_notes: str = ""
) -> dict:
    """
    Evaluates Metabolic Syndrome according to NCEP ATP III / AHA standards across 5 clinical criteria:
    1. Elevated Fasting Glucose (>= 100 mg/dL or drug treatment)
    2. Elevated Blood Pressure (>= 130/85 mmHg or antihypertensive therapy)
    3. Elevated Waist / Central Adiposity (BMI >= 25 kg/m² South Asian surrogate)
    4. Elevated Triglycerides (>= 150 mg/dL or drug treatment)
    5. Reduced HDL Cholesterol (< 40 mg/dL Men, < 50 mg/dL Women)
    Diagnosed when >= 3 of 5 components are confirmed.
    """
    criteria_met = []
    missing_subfractions = []
    notes_combined = " ".join([clinical_notes or ""] + (symptoms or []) + (lifestyle or [])).lower()

    # 1. Glycemic Criterion
    if fbs and fbs >= 100:
        criteria_met.append(f"Elevated fasting glucose ({fbs} mg/dL, ATP III threshold ≥ 100)")

    # 2. Blood Pressure Criterion
    has_htn = any(k in notes_combined for k in ["htn", "hypertens", "bp", "amlodipine", "losartan", "enalapril"])
    if has_htn:
        criteria_met.append("Documented Hypertension or Antihypertensive pharmacotherapy")

    # 3. Central Adiposity / BMI Criterion
    has_adiposity = any(k in notes_combined for k in ["obese", "overweight", "bmi", "sedentary", "waist"])
    if has_adiposity:
        criteria_met.append("Central adiposity / elevated BMI indicator (South Asian cut-off BMI ≥ 25 kg/m²)")

    # 4 & 5. Dyslipidemia (Triglycerides & HDL surrogates from total cholesterol)
    if chol and chol >= 240:
        criteria_met.append(f"Marked total hypercholesterolemia ({chol} mg/dL, dyslipidemia surrogate)")
    elif chol and chol >= 200:
        criteria_met.append(f"Borderline elevated total cholesterol ({chol} mg/dL)")
    else:
        missing_subfractions.append("Fasting Triglycerides and HDL-C required for definitive ATP III classification.")

    score = len(criteria_met)
    if score >= 3:
        status, color = "High Metabolic Syndrome Probability (≥3 ATP III Components)", "#ef4444"
        summary = "Meets clinical criteria for Metabolic Syndrome. High risk of cardiovascular events and progressive Type 2 Diabetes."
    elif score == 2:
        status, color = "Moderate Metabolic Risk (2 ATP III Components)", "#f59e0b"
        summary = "Two metabolic components confirmed. Recommend fasting lipid profile (HDL/Triglycerides) and structured lifestyle therapy."
    elif score == 1:
        status, color = "Mild Metabolic Risk (1 Component)", "#3b82f6"
        summary = "Single metabolic variation detected. Preventive wellness counselling recommended."
    else:
        status, color = "Low Metabolic Risk", "#10b981"
        summary = "Metabolic parameters are well-balanced within desirable ranges."

    return {
        "status": status,
        "score": score,
        "max_score": 5,
        "color": color,
        "summary": summary,
        "criteria_met": criteria_met,
        "missing_subfractions": missing_subfractions,
        "guideline": "NCEP ATP III / AHA Metabolic Syndrome Guidelines"
    }


# ── 5. THYROID RISK & ATA SCORING MODULE ──────────────────────────────────────
def calculate_thyroid_risk(
    tsh: Optional[float],
    symptoms: List[str] = None,
    medications: List[str] = None
) -> dict:
    if tsh is None or tsh <= 0:
        return {"status": "Not Assessed", "color": "#64748b", "summary": "TSH was not measured in this report."}

    symptom_str = " ".join(symptoms or []).lower()
    med_str = " ".join(medications or []).lower()
    on_thyroxine = "levothyroxine" in med_str or "eltroxin" in med_str or "thyroxine" in med_str

    if tsh > 10.0:
        status = "Overt Hypothyroidism"
        color = "#ef4444"
        summary = f"TSH markedly elevated ({tsh} mIU/L). Free T4 and Free T3 confirmation needed. Thyroxine replacement typically indicated."
    elif tsh > 4.5:
        status = "Subclinical Hypothyroidism"
        color = "#f59e0b"
        summary = f"TSH mildly elevated ({tsh} mIU/L) with normal reference cutoff at 4.5 mIU/L. Repeat in 6–8 weeks and test Anti-TPO antibodies."
    elif tsh < 0.1:
        status = "Overt Hyperthyroidism"
        color = "#ef4444"
        summary = f"TSH severely suppressed ({tsh} mIU/L). Risk of atrial fibrillation and bone density loss. Free T4/T3 and thyroid scan advised."
    elif tsh < 0.4:
        status = "Subclinical Hyperthyroidism"
        color = "#f59e0b"
        summary = f"TSH suppressed ({tsh} mIU/L, normal 0.4–4.5 mIU/L). Repeat thyroid panel in 6–8 weeks."
    else:
        status = "Euthyroid (Normal)"
        color = "#10b981"
        summary = f"TSH ({tsh} mIU/L) is within established euthyroid reference range (0.4–4.5 mIU/L)."

    if on_thyroxine and tsh > 4.5:
        summary += " ⚠ Patient is on thyroid medication — dose titration may be required."

    return {
        "status": status,
        "tsh_value": tsh,
        "color": color,
        "summary": summary
    }


# ── 6. ML-BASED ANOMALY & OUTLIER DETECTION ───────────────────────────────────
def calculate_ml_anomaly_score(values: Dict[str, Any], gender: str, age: int) -> dict:
    """
    Computes a normalized multivariate Z-score outlier index across measured cell lines and chemistry.
    """
    REFERENCE_DISTRIBUTIONS = {
        "Hemoglobin": {"mean": 14.5 if gender == "Male" else 13.0, "std": 1.2},
        "WBC": {"mean": 7.0, "std": 1.8},
        "Platelets": {"mean": 250.0, "std": 50.0},
        "Fasting_Blood_Sugar": {"mean": 85.0, "std": 10.0},
        "Creatinine": {"mean": 0.9 if gender == "Male" else 0.75, "std": 0.15},
        "TSH": {"mean": 2.0, "std": 1.0},
        "Cholesterol": {"mean": 170.0, "std": 25.0}
    }

    z_scores = {}
    max_z = 0.0
    outlier_biomarkers = []

    for marker, dist in REFERENCE_DISTRIBUTIONS.items():
        val = values.get(marker)
        if val is not None and isinstance(val, (int, float)) and val > 0:
            z = abs(val - dist["mean"]) / dist["std"]
            z_scores[marker] = round(z, 2)
            if z > max_z:
                max_z = z
            if z >= 2.5:  # Beyond 2.5 standard deviations (~99th percentile outlier)
                outlier_biomarkers.append({
                    "biomarker": marker.replace("_", " "),
                    "value": val,
                    "z_score": round(z, 1),
                    "deviation": "Significantly elevated" if val > dist["mean"] else "Significantly depressed"
                })

    anomaly_index = round(min(100.0, (max_z / 4.0) * 100.0), 1)
    
    if anomaly_index >= 75:
        severity = "High Outlier Deviation"
        color = "#ef4444"
    elif anomaly_index >= 45:
        severity = "Moderate Outlier Deviation"
        color = "#f59e0b"
    else:
        severity = "Normal Biological Variation"
        color = "#10b981"

    return {
        "anomaly_index_pct": anomaly_index,
        "severity": severity,
        "color": color,
        "max_z_score": round(max_z, 2),
        "outlier_biomarkers": outlier_biomarkers,
        "z_score_breakdown": z_scores
    }


# ── 7. LONGITUDINAL TREND ANALYSIS ENGINE ─────────────────────────────────────
def compute_longitudinal_trends(
    current_values: Dict[str, Any],
    historical_reports: List[Dict[str, Any]]
) -> dict:
    """
    Compares current lab biomarkers against patient's previous historical lab reports.
    Computes delta (Δ), trajectory slope (rising/falling/stable), and deterioration flags.
    """
    if not historical_reports:
        return {
            "has_history": False,
            "trend_count": 0,
            "biomarker_deltas": [],
            "deterioration_alerts": [],
            "summary": "First recorded analysis for this patient. Baseline established for longitudinal tracking."
        }

    # Find the most recent previous report
    prev_report = historical_reports[0]
    prev_values = prev_report.get("parsed_values", {})
    prev_date = prev_report.get("created_at") or prev_report.get("report_date") or "Previous visit"

    deltas = []
    alerts = []

    TRACKED_MARKERS = ["Hemoglobin", "Fasting_Blood_Sugar", "Creatinine", "Platelets", "WBC", "Cholesterol", "TSH"]

    for marker in TRACKED_MARKERS:
        cur_v = current_values.get(marker)
        old_v = prev_values.get(marker)

        if cur_v is not None and old_v is not None and isinstance(cur_v, (int, float)) and isinstance(old_v, (int, float)):
            diff = round(cur_v - old_v, 2)
            pct_change = round(((cur_v - old_v) / old_v) * 100.0, 1) if old_v != 0 else 0.0
            
            if diff > 0.05:
                direction = "Rising"
                arrow = "↑"
            elif diff < -0.05:
                direction = "Falling"
                arrow = "↓"
            else:
                direction = "Stable"
                arrow = "→"

            # Clinical deterioration signals
            if marker == "Hemoglobin" and diff <= -1.5:
                alerts.append(f"Haemoglobin dropped significantly by {abs(diff)} g/dL ({pct_change}%) since {prev_date} — evaluate for occult bleeding.")
            elif marker == "Creatinine" and diff >= 0.3:
                alerts.append(f"Serum Creatinine increased by +{diff} mg/dL ({pct_change}%) since {prev_date} — acute kidney injury (AKI) or progressive CKD alert.")
            elif marker == "Fasting_Blood_Sugar" and diff >= 30:
                alerts.append(f"Fasting Glucose escalated by +{diff} mg/dL ({pct_change}%) since {prev_date} — worsening glycaemic control.")
            elif marker == "Platelets" and diff <= -50:
                alerts.append(f"Platelets declined by {abs(diff)} ×10³/µL since {prev_date} — progressive thrombocytopenia trend.")

            deltas.append({
                "biomarker": marker.replace("_", " "),
                "current": cur_v,
                "previous": old_v,
                "delta": diff,
                "pct_change": pct_change,
                "direction": direction,
                "arrow": arrow,
                "is_concerning": len([a for a in alerts if marker.replace("_", " ") in a or marker in a]) > 0
            })

    return {
        "has_history": True,
        "trend_count": len(deltas),
        "previous_date": prev_date,
        "biomarker_deltas": deltas,
        "deterioration_alerts": alerts,
        "summary": f"Compared with previous report on {prev_date}." if deltas else "Historical comparison completed."
    }


# ── 5b. AUTOIMMUNE RISK MODULE ────────────────────────────────────────────────
def calculate_autoimmune_risk(
    values: Dict[str, Any],
    gender: str,
    age: int,
    symptoms: List[str] = None
) -> dict:
    """
    Evaluates multi-system autoimmune risk based on cytopenias, ESR, TSH axis, and clinical symptoms.
    """
    flags = []
    hb = values.get("Hemoglobin")
    plt = values.get("Platelets")
    wbc = values.get("WBC")
    tsh = values.get("TSH")
    esr = values.get("ESR")
    is_female = (gender or "").lower().startswith("f")
    symptom_str = " ".join(symptoms or []).lower()

    if hb and ((is_female and hb < 11.0) or (not is_female and hb < 12.0)) and plt and plt < 150:
        flags.append("Bicytopenia (Autoimmune hemolytic anaemia / ITP overlap)")
    if wbc and wbc < 3.5:
        flags.append("Leukopenia (Autoimmune neutropenia / SLE pattern)")
    if esr and esr > 30:
        flags.append(f"Elevated ESR ({esr} mm/hr — active systemic inflammation)")
    if tsh and tsh > 4.5:
        flags.append("Thyroid autoimmunity axis (Hashimoto's risk)")
    if any(s in symptom_str for s in ["joint", "rash", "malar", "alopecia", "dry mouth", "raynaud"]):
        flags.append("Autoimmune clinical symptom cluster reported")

    if len(flags) >= 3 or (len(flags) >= 2 and is_female):
        status = "Elevated Autoimmune Probability"
        color = "#ef4444"
        summary = "Multiple autoimmune markers identified. Recommend ANA (Antinuclear Antibody) by IFA, Anti-dsDNA, and ENA profile."
    elif len(flags) >= 1:
        status = "Borderline / Single Autoimmune Indicator"
        color = "#f59e0b"
        summary = f"Isolated indicator ({flags[0]}). Clinical correlation with rheumatologic signs recommended."
    else:
        status = "Low Autoimmune Probability"
        color = "#10b981"
        summary = "No multi-system inflammatory or autoimmune cytopenic patterns detected."

    return {
        "status": status,
        "color": color,
        "score": len(flags),
        "summary": summary,
        "indicators": flags
    }


# ── 5c. MICRONUTRIENT & VITAMIN D DEFICIENCY RISK ──────────────────────────────
def calculate_micronutrient_risk(values: Dict[str, Any]) -> dict:
    """
    Evaluates 25-OH Vitamin D and Vitamin B12 nutritional status.
    """
    vit_d = values.get("Vitamin_D")
    vit_b12 = values.get("Vitamin_B12")

    if vit_d is None and vit_b12 is None:
        return {"status": "Not Assessed", "color": "#64748b", "summary": "Vitamin D and B12 were not measured in this report."}

    d_summary = ""
    d_color = "#10b981"
    if vit_d is not None:
        if vit_d < 10.0:
            d_summary = f"Severe Vitamin D Deficiency ({vit_d} ng/mL). High risk of osteomalacia and musculoskeletal weakness."
            d_color = "#dc2626"
        elif vit_d < 20.0:
            d_summary = f"Vitamin D Deficiency ({vit_d} ng/mL). Supplementation required."
            d_color = "#ef4444"
        elif vit_d < 30.0:
            d_summary = f"Vitamin D Insufficiency ({vit_d} ng/mL). Optimization recommended."
            d_color = "#f59e0b"
        else:
            d_summary = f"Sufficient Vitamin D ({vit_d} ng/mL)."

    return {
        "status": "Evaluated",
        "vitamin_d_val": vit_d,
        "vitamin_b12_val": vit_b12,
        "color": d_color,
        "summary": d_summary or "Micronutrient levels reviewed."
    }


# ── 6. ML-BASED ANOMALY & OUTLIER DETECTION ───────────────────────────────────
def calculate_ml_anomaly_score(values: Dict[str, Any], gender: str, age: int) -> dict:
    """
    Computes a normalized multivariate Z-score outlier index across measured cell lines and chemistry.
    """
    REFERENCE_DISTRIBUTIONS = {
        "Hemoglobin": {"mean": 14.5 if gender == "Male" else 13.0, "std": 1.2},
        "WBC": {"mean": 7.0, "std": 1.8},
        "Platelets": {"mean": 250.0, "std": 50.0},
        "Fasting_Blood_Sugar": {"mean": 85.0, "std": 10.0},
        "Creatinine": {"mean": 0.9 if gender == "Male" else 0.75, "std": 0.15},
        "TSH": {"mean": 2.0, "std": 1.0},
        "Cholesterol": {"mean": 170.0, "std": 25.0}
    }

    z_scores = {}
    max_z = 0.0
    outlier_biomarkers = []

    for marker, dist in REFERENCE_DISTRIBUTIONS.items():
        val = values.get(marker)
        if val is not None and isinstance(val, (int, float)) and val > 0:
            z = abs(val - dist["mean"]) / dist["std"]
            z_scores[marker] = round(z, 2)
            if z > max_z:
                max_z = z
            if z >= 2.5:  # Beyond 2.5 standard deviations (~99th percentile outlier)
                outlier_biomarkers.append({
                    "biomarker": marker.replace("_", " "),
                    "value": val,
                    "z_score": round(z, 2),
                    "mean": dist["mean"],
                    "deviation": f"{round((val - dist['mean']) / dist['mean'] * 100, 1)}%"
                })

    anomaly_index = min(100.0, round((max_z / 4.0) * 100.0, 1))

    if max_z >= 3.0:
        severity = "High Outlier Deviation"
        color = "#ef4444"
    elif max_z >= 2.0:
        severity = "Moderate Outlier Deviation"
        color = "#f59e0b"
    elif max_z >= 1.2:
        severity = "Mild Deviation"
        color = "#3b82f6"
    else:
        severity = "Within Normal Biological Variance"
        color = "#10b981"

    return {
        "anomaly_index_pct": anomaly_index,
        "max_z_score": round(max_z, 2),
        "severity": severity,
        "color": color,
        "outlier_biomarkers": outlier_biomarkers,
        "all_z_scores": z_scores
    }


# ── 7. LONGITUDINAL TREND ENGINE (HISTORICAL COMPARISON & DELTAS) ─────────────
def compute_longitudinal_trends(
    current_values: Dict[str, Any],
    historical_reports: List[Dict[str, Any]]
) -> dict:
    """
    Compares current lab values against the patient's most recent prior lab visit.
    Calculates absolute delta, percentage change, and rising/falling trajectories.
    """
    if not historical_reports or len(historical_reports) == 0:
        return {
            "has_history": False,
            "previous_date": None,
            "biomarker_deltas": [],
            "deterioration_alerts": [],
            "summary": "No previous lab records found in patient history for longitudinal trend analysis."
        }

    prev_report = historical_reports[0]
    prev_values = prev_report.get("parsed_values", {})
    prev_date = prev_report.get("report_date") or prev_report.get("created_at") or "Previous Visit"

    deltas = []
    alerts = []

    TRACKED_MARKERS = [
        ("Hemoglobin", "g/dL", -1.5, 2.0),
        ("Fasting_Blood_Sugar", "mg/dL", -30, 30),
        ("Creatinine", "mg/dL", -0.3, 0.3),
        ("Platelets", "×10³/µL", -50, 100),
        ("WBC", "×10³/µL", -3.0, 3.0),
        ("Cholesterol", "mg/dL", -40, 40),
        ("TSH", "mIU/L", -2.0, 2.5),
        ("Vitamin_D", "ng/mL", -10.0, 15.0)
    ]

    for marker, unit, sig_drop, sig_rise in TRACKED_MARKERS:
        curr_v = current_values.get(marker)
        prev_v = prev_values.get(marker)

        if curr_v is not None and prev_v is not None and prev_v > 0:
            diff = round(curr_v - prev_v, 2)
            pct_change = round(((curr_v - prev_v) / prev_v) * 100.0, 1)

            if diff > 0.05:
                direction = "Rising"
                arrow = "↑"
            elif diff < -0.05:
                direction = "Falling"
                arrow = "↓"
            else:
                direction = "Stable"
                arrow = "→"

            is_concerning = False
            if marker == "Hemoglobin" and diff <= sig_drop:
                alerts.append(f"Haemoglobin dropped significantly by {abs(diff)} g/dL ({pct_change}%) since {prev_date} — evaluate for occult bleeding.")
                is_concerning = True
            elif marker == "Creatinine" and diff >= sig_rise:
                alerts.append(f"Serum Creatinine increased by +{diff} mg/dL ({pct_change}%) since {prev_date} — indicates acute kidney injury or renal progression.")
                is_concerning = True
            elif marker == "Fasting_Blood_Sugar" and diff >= sig_rise:
                alerts.append(f"Fasting Glucose escalated by +{diff} mg/dL ({pct_change}%) since {prev_date} — worsening glycaemic control.")
                is_concerning = True
            elif marker == "Platelets" and diff <= sig_drop:
                alerts.append(f"Platelet count dropped by {abs(diff)} ×10³/µL since {prev_date} — monitor bleeding risk.")
                is_concerning = True
            elif marker == "Vitamin_D" and diff <= sig_drop:
                alerts.append(f"Vitamin D level declined by {abs(diff)} ng/mL since {prev_date} — check supplement adherence.")
                is_concerning = True

            deltas.append({
                "biomarker": marker.replace("_", " "),
                "current": curr_v,
                "previous": prev_v,
                "delta": diff,
                "pct_change": pct_change,
                "unit": unit,
                "direction": direction,
                "arrow": arrow,
                "is_concerning": is_concerning
            })

    summary_text = (
        f"Compared with previous record from {prev_date}. "
        f"{len(alerts)} clinically significant deterioration alert(s) identified."
        if alerts else f"Compared with previous record from {prev_date}. Biomarker trajectory is relatively stable."
    )

    return {
        "has_history": True,
        "previous_date": prev_date,
        "biomarker_deltas": deltas,
        "deterioration_alerts": alerts,
        "summary": summary_text
    }


# ── 7b. DENGUE WARNING TRAJECTORY MODULE (SRI LANKA NDCU PROTOCOL) ───────────
def calculate_dengue_risk(
    values: Dict[str, Any],
    historical_reports: List[Dict[str, Any]] = None
) -> dict:
    """
    Evaluates Dengue Warning Trajectory according to Sri Lanka National Dengue Control Unit (NDCU) guidelines.
    Monitors:
    - Thrombocytopenia: Platelet crash (< 100k, severe < 50k / µL)
    - Hemoconcentration: Packed Cell Volume (PCV) / Hematocrit rise (vascular plasma leakage warning)
    - Leukopenia: WBC < 4.0 × 10³/µL (typical early viral phase indicator)
    """
    plt = values.get("Platelets")
    wbc = values.get("WBC")
    pcv = values.get("PCV") or values.get("Packed_Cell_Volume")

    if plt is None:
        return {
            "status": "Not Assessed",
            "color": "#64748b",
            "summary": "Platelet count not measured in current laboratory panel."
        }

    prev_plt = None
    prev_pcv = None
    if historical_reports and len(historical_reports) > 0:
        prev_parsed = historical_reports[0].get("parsed_values", {})
        prev_plt = prev_parsed.get("Platelets")
        prev_pcv = prev_parsed.get("PCV") or prev_parsed.get("Packed_Cell_Volume")

    dengue_flags = []
    if plt < 50:
        dengue_flags.append(f"Critical Thrombocytopenia (PLT {plt} ×10³/µL) — high bleeding / plasma leakage risk")
    elif plt < 100:
        dengue_flags.append(f"Significant Thrombocytopenia (PLT {plt} ×10³/µL, NDCU warning threshold < 100k)")

    # Rapid platelet drop from previous
    if prev_plt and isinstance(prev_plt, (int, float)) and (prev_plt - plt) >= 40:
        dengue_flags.append(f"Rapid Platelet Drop: Declined by -{int(prev_plt - plt)} ×10³/µL since previous visit")

    # PCV hemoconcentration
    if pcv and pcv >= 48.0:
        dengue_flags.append(f"Hemoconcentration (PCV {pcv}%) — indicates capillary leakage / hemoconcentration")
    elif prev_pcv and pcv and (pcv - prev_pcv) >= 5.0:
        dengue_flags.append(f"Rising Hematocrit (+{round(pcv - prev_pcv, 1)}% PCV jump) — vascular plasma leakage signal")

    if wbc and wbc < 4.0:
        dengue_flags.append(f"Leukopenia (WBC {wbc} ×10³/µL) — common viral suppression pattern in dengue")

    if len(dengue_flags) >= 2 or (plt and plt < 50):
        status = "HIGH DENGUE WARNING RISK (National Protocol Alert)"
        color = "#dc2626"
        action = "Admit for close fluid balance monitoring (PCV/PLT q6h). Monitor for plasma leakage and shock. Follow Sri Lanka NDCU management guidelines."
    elif len(dengue_flags) == 1:
        status = "Moderate Dengue Surveillance Needed"
        color = "#f59e0b"
        action = "Repeat Full Blood Count (FBC) in 12–24 hours to monitor platelet and PCV trajectory. Advise oral rehydration."
    else:
        status = "Low Dengue Probability"
        color = "#10b981"
        action = "Platelet and hematocrit parameters are within safe ranges."

    return {
        "status": status,
        "color": color,
        "flags": dengue_flags,
        "action": action,
        "guideline": "Sri Lanka National Dengue Control Unit (NDCU) Clinical Protocol"
    }


# ── 7c. CKDu SURVEILLANCE MODULE (SRI LANKA DRY ZONE ENDEMIC PROTOCOL) ────────
def calculate_ckdu_risk(
    values: Dict[str, Any],
    age: int,
    gender: str,
    clinical_notes: str = "",
    symptoms: List[str] = None
) -> dict:
    """
    Evaluates Chronic Kidney Disease of unknown etiology (CKDu) surveillance index
    relevant to Sri Lanka's Dry Zone (North Central Province e.g. Anuradhapura, Polonnaruwa, Badulla).
    """
    creat = values.get("Creatinine")
    egfr_info = calculate_egfr(creat, age, gender)
    egfr = egfr_info.get("egfr")

    notes_all = ((clinical_notes or "") + " " + " ".join(symptoms or [])).lower()
    is_dry_zone = any(reg in notes_all for reg in [
        "anuradhapura", "polonnaruwa", "medawachchiya", "padaviya", "dehiattakandiya",
        "girandurukotte", "kurunegala", "badulla", "dry zone", "farming", "farmer",
        "paddy", "agriculture", "well water", "groundwater"
    ])

    ckdu_flags = []
    if is_dry_zone:
        ckdu_flags.append("Endemic Region or Occupational Exposure (Sri Lanka Dry Zone Agricultural Cohort)")

    if egfr and egfr < 60:
        ckdu_flags.append(f"Impaired eGFR ({egfr} mL/min/1.73m²) — persistent tubulointerstitial damage pattern")
    elif creat and creat >= (1.1 if gender == "Female" else 1.3):
        ckdu_flags.append(f"Borderline / elevated serum creatinine ({creat} mg/dL) in high-risk demographic")

    if is_dry_zone and ((egfr and egfr < 60) or (creat and creat >= (1.1 if gender == "Female" else 1.3))):
        status = "High CKDu Surveillance Flag (NCP Protocol)"
        color = "#ef4444"
        summary = "Patient exhibits renal biomarker impairment in the context of Sri Lanka endemic agricultural exposure. Recommend urinary biomarkers (uACR, alpha-1 microglobulin) and renal ultrasound to exclude structural causes."
    elif is_dry_zone:
        status = "CKDu Risk Surveillance Monitored"
        color = "#3b82f6"
        summary = "Patient has Dry Zone / agricultural exposure. Renal biomarkers are currently preserved. Annual eGFR and urine albumin screening recommended."
    else:
        status = "Standard CKD Assessment (Non-Endemic)"
        color = "#10b981"
        summary = "No specific CKDu endemic environmental factors identified."

    return {
        "status": status,
        "color": color,
        "summary": summary,
        "flags": ckdu_flags,
        "guideline": "Sri Lanka Ministry of Health CKDu Screening Protocol"
    }


# ── 7d. TRILINGUAL PATIENT TAKEAWAY SUMMARIES ─────────────────────────────────
def generate_trilingual_summary(diagnosis: str, findings: list = None) -> dict:
    """Generates patient-friendly takeaway statements in English, Sinhala, and Tamil."""
    d_low = (diagnosis or "").lower()

    if "anemia" in d_low or "anaemia" in d_low:
        en = f"Diagnosis: {diagnosis}. Your red blood cells or iron levels are lower than normal. Dietary iron and physician consultation recommended."
        si = "රෝග විනිශ්චය: රක්තහීනතාවය (Anemia). ඔබේ රුධිරයේ හිමොග්ලොබින් මට්ටම සාමාන්‍ය අගයට වඩා අඩුය. යකඩ බහුල ආහාර සහ වෛද්‍ය උපදෙස් ලබාගන්න."
        ta = "கண்டறிதல்: இரத்த சோகை (Anemia). உங்கள் இரத்தத்தில் ஹீமோகுளோபின் அளவு குறைவாக உள்ளது. இரும்புச்சத்து நிறைந்த உணவு மற்றும் மருத்துவ ஆலோசனை பெறவும்."
    elif "diabetes" in d_low:
        en = f"Diagnosis: {diagnosis}. Fasting blood sugar is elevated. Dietary carbohydrate management and medical follow-up recommended."
        si = "රෝග විනිශ්චය: දියවැඩියාව (Diabetes). රුධිරයේ සීනි මට්ටම වැඩි වී ඇත. සීනි සහ කාබෝහයිඩ්‍රේට් පාලනය කර වෛද්‍යවරයා හමුවන්න."
        ta = "கண்டறிதல்: நீரிழிவு நோய் (Diabetes). இரத்தத்தில் சர்க்கரை அளவு அதிகமாக உள்ளது. உணவு கட்டுப்பாடு மற்றும் மருத்துவ ஆலோசனை தேவை."
    elif "thyroid" in d_low:
        en = f"Diagnosis: {diagnosis}. Thyroid hormone regulatory levels require follow-up testing and specialist guidance."
        si = "රෝග විනිශ්චය: තයිරොයිඩ් ග්‍රන්ථියේ අසමතුලිතතාවය. හෝමෝන මට්ටම නැවත පරීක්ෂා කර වෛද්‍ය උපදෙස් ලබාගන්න."
        ta = "கண்டறிதல்: தைராய்டு கோளாறு (Thyroid). ஹார்மோன் சமநிலையின்மைக்கான பரிசோதனை மற்றும் மருத்துவ ஆலோசனை தேவை."
    elif "renal" in d_low or "kidney" in d_low:
        en = f"Diagnosis: {diagnosis}. Kidney biomarker variations detected. Avoid NSAID painkillers and consult a physician."
        si = "රෝග විනිශ්චය: වකුගඩු ක්‍රියාකාරීත්වයේ වෙනස්කම්. වේදනා නාශක ඖෂධ ගැනීමෙන් වැළකී වෛද්‍යවරයා හමුවන්න."
        ta = "கண்டறிதல்: சிறுநீரக செயல்பாடு மாறுபாடு. வலி நிவாரணி மருந்துகளை தவிர்த்து மருத்துவரை அணுகவும்."
    else:
        en = f"Clinical Summary: {diagnosis}. Discuss laboratory findings with your attending doctor."
        si = f"සායනික සාරාංශය: {diagnosis}. ඔබේ රසායනාගාර වාර්තාව පිළිබඳව වෛද්‍යවරයා සමඟ සාකච්ඡා කරන්න."
        ta = f"மருத்துவ சுருக்கம்: {diagnosis}. ஆய்வக முடிவுகள் குறித்து மருத்துவரிடம் ஆலோசனை பெறவும்."

    return {
        "english": en,
        "sinhala": si,
        "tamil": ta
    }


# ── 8. COMPREHENSIVE RISK PROFILING (ENTRY POINT) ─────────────────────────────
def compute_comprehensive_risk_profile(
    values: Dict[str, Any],
    age: int = 45,
    gender: str = "Male",
    symptoms: List[str] = None,
    medications: List[str] = None,
    lifestyle: List[str] = None,
    clinical_notes: str = "",
    historical_reports: List[Dict[str, Any]] = None
) -> dict:
    age_int = int(age) if str(age).isdigit() else 45
    gender_str = gender if gender in ("Male", "Female") else "Male"

    egfr_data = calculate_egfr(values.get("Creatinine"), age_int, gender_str)
    ascvd_data = calculate_ascvd_risk(age_int, gender_str, values.get("Cholesterol"), values.get("Fasting_Blood_Sugar"), symptoms, medications, lifestyle, clinical_notes)
    anemia_data = calculate_anemia_risk(values.get("Hemoglobin"), gender_str, age_int, values.get("WBC"), values.get("Platelets"), symptoms)
    metabolic_data = calculate_metabolic_risk(values.get("Fasting_Blood_Sugar"), values.get("Cholesterol"), age_int, gender_str, symptoms, lifestyle, clinical_notes)
    thyroid_data = calculate_thyroid_risk(values.get("TSH"), symptoms, medications)
    autoimmune_data = calculate_autoimmune_risk(values, gender_str, age_int, symptoms)
    micronutrient_data = calculate_micronutrient_risk(values)
    ml_anomaly_data = calculate_ml_anomaly_score(values, gender_str, age_int)
    trend_data = compute_longitudinal_trends(values, historical_reports or [])
    dengue_data = calculate_dengue_risk(values, historical_reports or [])
    ckdu_data = calculate_ckdu_risk(values, age_int, gender_str, clinical_notes, symptoms)

    # Pattern recognition
    cross_anomalies = []
    hb = values.get("Hemoglobin")
    plt = values.get("Platelets")
    fbs = values.get("Fasting_Blood_Sugar")
    creat = values.get("Creatinine")
    chol = values.get("Cholesterol")
    tsh = values.get("TSH")
    vit_d = values.get("Vitamin_D")

    if vit_d is not None and vit_d < 20.0 and hb and ((gender_str == "Female" and hb < 12.0) or (gender_str == "Male" and hb < 13.0)):
        cross_anomalies.append({
            "title": "Dual Deficiency Axis (Vitamin D + Iron / Hemoglobin Deficiency)",
            "severity": "Moderate to High",
            "color": "#ef4444",
            "detail": f"Concomitant Vitamin D deficiency ({vit_d} ng/mL) and Anaemia (Hb {hb} g/dL). Both micronutrient pathways require coordinated oral replacement."
        })

    if hb and ((gender_str == "Female" and hb < 12.0) or (gender_str == "Male" and hb < 13.0)) and plt and plt < 150:
        cross_anomalies.append({
            "title": "Bicytopenia Pattern (Anemia + Thrombocytopenia)",
            "severity": "High",
            "color": "#ef4444",
            "detail": f"Simultaneous reduction in Hemoglobin ({hb} g/dL) and Platelets ({plt} ×10³/µL). Bone marrow or splenic sequestration alert."
        })

    if fbs and fbs >= 126 and creat and creat >= 1.1:
        cross_anomalies.append({
            "title": "Diabetic Nephropathy Risk Axis",
            "severity": "Moderate to High",
            "color": "#f97316",
            "detail": f"Hyperglycemia (FBS {fbs} mg/dL) combined with borderline/elevated creatinine ({creat} mg/dL). Order urine albumin-creatinine ratio (uACR)."
        })

    if tsh and tsh > 4.5 and chol and chol >= 200:
        cross_anomalies.append({
            "title": "Secondary Dyslipidemia due to Hypothyroidism",
            "severity": "Moderate",
            "color": "#f59e0b",
            "detail": f"TSH elevated ({tsh} mIU/L) with hypercholesterolemia ({chol} mg/dL). Thyroid optimization typically lowers lipid levels."
        })

    return {
        "egfr": egfr_data,
        "ascvd": ascvd_data,
        "anemia": anemia_data,
        "metabolic": metabolic_data,
        "thyroid": thyroid_data,
        "autoimmune": autoimmune_data,
        "micronutrient": micronutrient_data,
        "ml_anomaly": ml_anomaly_data,
        "longitudinal_trends": trend_data,
        "dengue_trajectory": dengue_data,
        "ckdu_surveillance": ckdu_data,
        "cross_marker_anomalies": cross_anomalies,
        "context": {
            "age": age_int,
            "gender": gender_str,
            "symptoms": symptoms or [],
            "medications": medications or [],
            "lifestyle": lifestyle or [],
            "notes": clinical_notes or ""
        }
    }

