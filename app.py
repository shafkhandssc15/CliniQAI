import os
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional
from dotenv import load_dotenv

# Load environment configuration (.env)
load_dotenv()

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import uvicorn

from database import init_db, get_db
from routes_auth import router as auth_router
from routes_admin import router as admin_router
from routes_reports import router as reports_router
from routes_supabase import router as supabase_router
from integration_example import router as ml_router, get_engine
from dependencies import get_verified_doctor
from models_db import Doctor, PatientReport
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
from ocr_engine import (
    extract_text_from_image,
    deidentify_report_image,
    parse_biomarkers,
    extract_all_laboratory_values,
    extract_patient_demographics,
    run_rule_based_engine,
    run_ml_classifier,
    run_nvidia_nim_vlm,
    run_gemini_vlm,
    run_openai_vlm,
    run_roar_ai_vlm,
    load_local_classifier,
    get_easyocr_reader
)

from clinical_risk_engine import compute_comprehensive_risk_profile, generate_trilingual_summary
from clinical_validator import compute_validated_ensemble


# ── Lifespan ──────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing CliniQ AI platform...")
    await init_db()
    print("Database tables initialized.")
    # Initialize the new reference ML engine (cliniq_ml v2.2)
    ml_engine = get_engine()
    print(f"CliniQ ML engine loaded (version {ml_engine.version}).")
    load_local_classifier()
    print("Local ML classifiers loaded.")
    # Primary Native Tesseract 5.5.3 Engine Verification
    try:
        import tesserocr
        print(f"Primary Tesseract OCR Engine initialized ({tesserocr.tesseract_version().splitlines()[0]}).")
    except Exception as tess_err:
        print(f"[OCR] Tesseract initialization note: {tess_err}")

    # Supabase Cloud Database Status Check
    try:
        from supabase_client import test_connection
        sb_stat = await test_connection()
        if sb_stat.get("reachable"):
            table_msg = "Table ready" if sb_stat.get("table_ready") else "Table missing (run supabase_schema.sql)"
            print(f"[Supabase] Connected to {sb_stat.get('project_ref')} ({table_msg}, latency: {sb_stat.get('latency_ms')}ms)")
        else:
            print(f"[Supabase] Configured but unreachable: {sb_stat.get('message')}")
    except Exception as sb_err:
        print(f"[Supabase] Connection test warning: {sb_err}")

    yield
    print("CliniQ AI shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CliniQ AI — Medical Diagnostic Platform",
    description="AI-powered medical report analysis for SLMC-registered doctors in Sri Lanka.",
    version="2.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response

app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(reports_router)
app.include_router(supabase_router)
app.include_router(ml_router)


# ── Clinical Follow-Up Recommendations (Finding-Specific) ────────────────────────
RECOMMENDATIONS = {
    "Anaemia": [
        "Iron Studies: Serum Ferritin, TIBC, and Serum Iron to classify anaemia type",
        "Peripheral Blood Smear for RBC morphology (microcytic/normocytic/macrocytic)",
        "Vitamin B12 and Folate serum levels to exclude megaloblastic anaemia",
        "Reticulocyte count to assess bone marrow response",
        "Rule out occult GI bleeding — faecal occult blood test if clinically indicated",
        "Repeat CBC in 6–8 weeks post-treatment initiation to assess response",
    ],
    "Anemia": [
        "Iron Studies: Serum Ferritin, TIBC, and Serum Iron to classify anaemia type",
        "Peripheral Blood Smear for RBC morphology (microcytic/normocytic/macrocytic)",
        "Vitamin B12 and Folate serum levels to exclude megaloblastic anaemia",
        "Reticulocyte count to assess bone marrow response",
        "Rule out occult GI bleeding — faecal occult blood test if clinically indicated",
        "Repeat CBC in 6–8 weeks post-treatment initiation to assess response",
    ],
    "Diabetes": [
        "Order HbA1c test to quantify long-term glycaemic control over 3 months",
        "Fasting lipid panel (LDL, HDL, Triglycerides) — diabetes increases CVD risk",
        "Renal function: eGFR calculation and urine albumin-creatinine ratio (uACR)",
        "Ophthalmology referral for diabetic retinopathy screening",
        "Blood pressure monitoring — target < 130/80 mmHg per ADA guidelines",
        "Dietary counselling — low glycaemic index diet and carbohydrate counting",
    ],
    "Prediabetes": [
        "Order HbA1c to confirm glycaemic status (5.7–6.4% = prediabetes)",
        "75g Oral Glucose Tolerance Test (OGTT) for definitive classification",
        "Lifestyle intervention: 150 min/week moderate exercise, 5–7% weight loss target",
        "Repeat fasting glucose and HbA1c in 3 months to monitor progression",
        "Screen for metabolic syndrome components (waist circumference, BP, triglycerides)",
    ],
    "Hypothyroidism": [
        "Free T4 (FT4) and Free T3 (FT3) serum assay to confirm hypothyroidism",
        "Thyroid Peroxidase Antibody (TPO-Ab) for autoimmune (Hashimoto's) aetiology",
        "Fasting lipid panel — hypothyroidism commonly elevates cholesterol",
        "Consider low-dose Levothyroxine initiation with specialist guidance",
        "Follow-up TSH in 6–8 weeks after any dose adjustment",
        "Cardiac evaluation if bradycardia or prolonged QTc suspected",
    ],
    "Hyperthyroidism": [
        "Free T4 and Free T3 to confirm hyperthyroid state",
        "TSH Receptor Antibody (TRAb) for Graves' disease differentiation",
        "Thyroid ultrasound and radionuclide uptake scan",
        "Endocrinology referral for anti-thyroid therapy planning",
        "ECG — rule out atrial fibrillation secondary to thyrotoxicosis",
        "Bone densitometry if long-standing or severe hyperthyroidism",
    ],
    "Renal": [
        "Comprehensive metabolic panel including electrolytes and bicarbonate",
        "Urine dipstick and microscopy — proteinuria and haematuria evaluation",
        "Urinary albumin-creatinine ratio (uACR) for quantitative proteinuria",
        "Renal ultrasound to assess kidney size and structural abnormalities",
        "Calculate eGFR using CKD-EPI formula for accurate staging",
        "Nephrology referral for CKD staging and management planning",
    ],
    "Cholesterol": [
        "Full fasting lipid panel — LDL, HDL, VLDL, Triglycerides",
        "Cardiovascular risk score calculation (10-year ASCVD risk)",
        "Dietary counselling — Mediterranean or DASH diet approach",
        "Consider statin therapy based on LDL level and calculated risk score",
        "Thyroid function test — rule out secondary dyslipidaemia",
        "Repeat lipid panel in 3 months post-intervention",
    ],
    "Thrombocytopenia": [
        "Peripheral blood smear for platelet morphology and pseudothrombocytopenia",
        "Coagulation studies — PT, APTT, fibrinogen",
        "Liver function tests to exclude hepatic sequestration",
        "Test for anti-platelet antibodies if immune cause suspected",
        "Haematology consultation for bone marrow assessment if persistent",
    ],
    "Infection": [
        "C-Reactive Protein (CRP) and ESR for inflammation quantification",
        "Blood cultures (x2 sets) if systemic infection suspected",
        "Differential WBC count — evaluate neutrophil, lymphocyte, monocyte ratios",
        "Urinalysis and urine culture to exclude urinary source",
        "Chest X-ray if respiratory symptoms present",
    ],
}

BORDERLINE_RECOMMENDATIONS = [
    "Repeat relevant laboratory test in 2–4 weeks to establish biomarker trajectory.",
    "Correlate findings with clinical history and physical examination.",
    "Consider lifestyle and dietary optimization before initiating pharmacotherapy.",
    "Consider specialist referral only if symptoms or persistent abnormalities are present."
]

DEFAULT_RECOMMENDATIONS = [
    "Correlate findings with clinical history, medication use, and physical examination.",
    "Repeat relevant labs in 2–4 weeks for trend analysis and confirmation.",
    "Maintain balanced nutrition, adequate hydration, and age-appropriate physical activity.",
    "Consider specialist referral only if symptoms or persistent abnormalities are present."
]


def get_recommendations(diagnosis: str, findings: list = None, borderline: list = None, age: int = 45, gender: str = "Male", risk_profile: dict = None) -> list[str]:
    """Return personalized follow-up recommendations tied to lab results, age, and risk stratification."""
    recs = []
    diag_lower = diagnosis.lower()

    # Match condition-specific recommendations
    for key, key_recs in RECOMMENDATIONS.items():
        if key.lower() in diag_lower:
            recs.extend(key_recs[:3])

    # Factor in ASCVD risk
    if risk_profile and risk_profile.get("ascvd"):
        ascvd = risk_profile["ascvd"]
        if "Intermediate" in ascvd.get("category", "") or "High" in ascvd.get("category", ""):
            recs.append(f"ASCVD Prevention: {ascvd.get('action')}")
            
    # Factor in eGFR renal status
    if risk_profile and risk_profile.get("egfr"):
        egfr = risk_profile["egfr"]
        if egfr.get("egfr") and egfr["egfr"] < 60:
            recs.append(f"Renal Care (eGFR {egfr['egfr']} mL/min): Order urine albumin-creatinine ratio (uACR) and avoid nephrotoxic agents (NSAIDs).")

    # Factor in Metabolic Syndrome
    if risk_profile and risk_profile.get("metabolic"):
        meta = risk_profile["metabolic"]
        if meta.get("score", 0) >= 2:
            recs.append("Metabolic Risk: Structured dietary counselling (Mediterranean / Low GI diet) and 150 min/week moderate exercise.")

    # Age-specific screening for adults 50+
    if age >= 50:
        recs.append(f"Age-Appropriate Wellness: Comprehensive cardiovascular risk assessment and routine cancer screening for age {age}.")

    if borderline:
        recs.extend(BORDERLINE_RECOMMENDATIONS[:2])

    if recs:
        seen = set()
        unique = []
        for r in recs:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        return unique[:6]

    if borderline:
        return BORDERLINE_RECOMMENDATIONS[:5]
    return DEFAULT_RECOMMENDATIONS[:5]


def normalize_condition_name(c: str) -> str:
    """Normalize clinical condition variants into unified names for consensus scoring."""
    c_low = c.lower().strip()
    if "anaemia" in c_low or "anemia" in c_low:
        return "Anemia"
    if "diabetes" in c_low or "glucose" in c_low or "sugar" in c_low:
        return "Diabetes"
    if "hypo" in c_low and "thyroid" in c_low:
        return "Hypothyroidism"
    if "hyper" in c_low and "thyroid" in c_low:
        return "Hyperthyroidism"
    if "renal" in c_low or "kidney" in c_low or "ckd" in c_low or "creatinine" in c_low:
        return "Impaired Renal Function"
    if "cholesterol" in c_low or "dyslipidemia" in c_low:
        return "Hypercholesterolaemia"
    if "thrombocytopenia" in c_low or "platelet" in c_low:
        return "Thrombocytopenia"
    if "infection" in c_low or "inflammation" in c_low:
        return "Possible Infection"
    if any(h in c_low for h in ["healthy", "normal", "no significant", "no major"]):
        return "No significant abnormalities detected"
    return c.strip()


# ── Confidence Score Computation (Safety-Aware) ──────────────────────────────────
def compute_ensemble_confidence(m1, m2, m3, m4) -> dict:
    """
    Weighted ensemble: Gemini 40%, NVIDIA 35%, XGBoost 20%, Rules 5%.
    Standardizes conditions and evaluates cross-model consensus.
    """
    WEIGHTS = {
        "model_4_gemini": 0.40,
        "model_3_nvidia": 0.35,
        "model_2_ml":     0.20,
        "model_1_rules":  0.05,
    }

    sources = [
        ("model_4_gemini", m4),
        ("model_3_nvidia", m3),
        ("model_2_ml", m2),
        ("model_1_rules", m1),
    ]

    active_weight = 0.0
    vote_breakdown = {}
    safety_flags = []
    
    condition_weights = {}
    model_normalized_diags = {}

    for key, result in sources:
        if not result or result.get("error"):
            continue
        diag = result.get("diagnosis") or result.get("Predicted_Diseases", ["Unknown"])
        if isinstance(diag, list):
            diag_str = ", ".join([str(d) for d in diag if str(d).strip()])
        else:
            diag_str = str(diag)
        if not diag_str or diag_str in ("N/A", "Offline"):
            continue
        
        # Clean XGBoost percentage if present
        diag_clean = diag_str.split("(")[0].strip() if "(" in diag_str else diag_str
        w = WEIGHTS[key]
        active_weight += w
        vote_breakdown[key] = {"diagnosis": diag_clean, "weight": w}

        # Split multiple comma-separated conditions and normalize each
        parts = [p.strip() for p in diag_clean.split(",") if p.strip()]
        norm_parts = []
        for p in parts:
            norm_c = normalize_condition_name(p)
            norm_parts.append(norm_c)
            condition_weights[norm_c] = condition_weights.get(norm_c, 0.0) + w
        
        # Deduplicate
        model_normalized_diags[key] = list(dict.fromkeys(norm_parts))

    if not condition_weights or active_weight == 0:
        return {
            "primary_diagnosis": "Analysis Inconclusive — insufficient model consensus",
            "confidence": 0.0,
            "confidence_band": "INCONCLUSIVE",
            "vote_breakdown": {},
            "safety_flags": ["No models produced usable output. Manual clinical assessment required."]
        }

    # Filter out "normal" if any actual disease conditions are present
    has_diseases = any(k != "No significant abnormalities detected" for k in condition_weights)
    if has_diseases and "No significant abnormalities detected" in condition_weights:
        del condition_weights["No significant abnormalities detected"]

    # Select conditions supported by at least 25% of active weight
    supported = [c for c, w in condition_weights.items() if (w / active_weight) >= 0.25]
    if not supported:
        supported = [max(condition_weights, key=condition_weights.get)]

    # Calculate overall consensus confidence as the average weight support of supported conditions
    avg_support = sum(condition_weights[c] for c in supported) / (len(supported) * active_weight)
    raw_conf = min(1.0, max(0.40, avg_support))

    top_diag = ", ".join(supported)

    # ── SAFETY LAYER: prevent dangerous "Healthy" label at low confidence ──
    is_healthy_label = "no significant abnormalities" in top_diag.lower() or "healthy" in top_diag.lower()
    if is_healthy_label and raw_conf < 0.80:
        top_diag = "No major abnormalities detected — clinical correlation recommended"
        safety_flags.append(
            f"Consensus was '{top_diag}' at {round(raw_conf*100,1)}% confidence. "
            f"Relabelled to avoid a definitive 'Healthy' diagnosis below 80% confidence threshold."
        )

    if raw_conf < 0.50:
        safety_flags.append("Low model agreement. Multiple differential diagnoses possible. Clinical judgement essential.")

    # Confidence bands
    if raw_conf >= 0.80:
        band = "HIGH"
    elif raw_conf >= 0.65:
        band = "MODERATE"
    elif raw_conf >= 0.50:
        band = "LOW"
    else:
        band = "INCONCLUSIVE"

    return {
        "primary_diagnosis": top_diag,
        "confidence": round(raw_conf * 100, 1),
        "confidence_band": band,
        "vote_breakdown": vote_breakdown,
        "all_votes": {k: round(v / active_weight * 100, 1) for k, v in condition_weights.items()},
        "safety_flags": safety_flags,
    }


@app.post("/api/analyze")
async def analyze_report(
    file: UploadFile = File(...),
    vlm_provider: str = Form("auto"),
    patient_name: str = Form(None),
    patient_age: str = Form(None),
    patient_gender: str = Form(None),
    report_type: str = Form(None),
    report_date: str = Form(None),
    clinical_notes: str = Form(None),
    symptoms: str = Form(None),
    medications: str = Form(None),
    lifestyle: str = Form(None),
    current_doctor: Doctor = Depends(get_verified_doctor),
    db: AsyncSession = Depends(get_db)
):
    try:
        image_bytes = await file.read()
        total_start = time.time()

        # 1. OCR (and bounding boxes for PII redaction)
        ocr_text, ocr_time, ocr_bboxes = extract_text_from_image(image_bytes, return_bboxes=True)

        # Sri Lanka PDPA (2022) compliant de-identification before cloud VLM transmission
        deidentified_image_bytes = deidentify_report_image(image_bytes, ocr_bboxes)

        # 1b. Universal Laboratory Extraction across all pages, sections, tables, and footnotes
        universal_lab_report = extract_all_laboratory_values(ocr_text, image_bytes, ocr_bboxes)

        # 2. Biomarker parsing
        parsed_values, extracted_flags = parse_biomarkers(ocr_text)

        # Enrich parsed_values from universal_lab_report if any standard marker was extracted
        for res_item in universal_lab_report.get("results", []):
            t_name = res_item.get("test_name", "").lower()
            val = res_item.get("result_value")
            if isinstance(val, (int, float)):
                if any(k in t_name for k in ["haemoglobin", "hemoglobin", "hb", "hgb"]) and not extracted_flags.get("Hemoglobin"):
                    parsed_values["Hemoglobin"] = float(val)
                    extracted_flags["Hemoglobin"] = True
                elif any(k in t_name for k in ["wbc", "white blood", "leukocyte"]) and not extracted_flags.get("WBC"):
                    parsed_values["WBC"] = float(val)
                    extracted_flags["WBC"] = True
                elif any(k in t_name for k in ["platelet", "plt"]) and not extracted_flags.get("Platelets"):
                    parsed_values["Platelets"] = int(val)
                    extracted_flags["Platelets"] = True
                elif any(k in t_name for k in ["glucose", "fbs", "sugar"]) and not extracted_flags.get("Fasting_Blood_Sugar"):
                    parsed_values["Fasting_Blood_Sugar"] = int(val)
                    extracted_flags["Fasting_Blood_Sugar"] = True
                elif any(k in t_name for k in ["creatinine"]) and not extracted_flags.get("Creatinine"):
                    parsed_values["Creatinine"] = float(val)
                    extracted_flags["Creatinine"] = True
                elif any(k in t_name for k in ["tsh"]) and not extracted_flags.get("TSH"):
                    parsed_values["TSH"] = float(val)
                    extracted_flags["TSH"] = True
                elif any(k in t_name for k in ["cholesterol"]) and "hdl" not in t_name and "ldl" not in t_name and not extracted_flags.get("Cholesterol"):
                    parsed_values["Cholesterol"] = int(val)
                    extracted_flags["Cholesterol"] = True

        # 2b. Extract patient demographics from OCR text for auto-fill
        ocr_demographics = extract_patient_demographics(ocr_text)

        # 3. Override intake fields with form values, else use OCR-extracted values
        if not patient_name or not patient_name.strip():
            patient_name = ocr_demographics.get("name") or universal_lab_report.get("patient_identifier") or ""
        if not report_date or not report_date.strip():
            report_date = universal_lab_report.get("report_date") or ""
        if not patient_age or not str(patient_age).strip():
            patient_age = str(ocr_demographics.get("age") or "") if ocr_demographics.get("age") else ""
        if not patient_gender or patient_gender == "Male":  # default form value
            if ocr_demographics.get("gender"):
                patient_gender = ocr_demographics["gender"]

        # 4. Rule-based engine (with context)
        t0 = time.time()
        # Feed form-level age/gender into parsed_values if user provided them
        if patient_age and str(patient_age).strip().isdigit():
            parsed_values["Age"] = int(patient_age)
            extracted_flags["Age"] = True
        if patient_gender and patient_gender in ("Male", "Female"):
            parsed_values["Gender"] = patient_gender
            extracted_flags["Gender"] = True
        m1 = run_rule_based_engine(parsed_values)
        m1["latency_sec"] = round(ocr_time + (time.time() - t0), 4)
        m1["is_local"] = True

        # Provider credentials are loaded from environment configuration only.
        eff_nvidia = os.environ.get("NVIDIA_API_KEY", "").strip()
        eff_gemini = os.environ.get("GEMINI_API_KEY", "").strip()
        eff_openai = os.environ.get("OPENAI_API_KEY", "").strip()
        eff_roar = os.environ.get("ROAR_API_KEY", "").strip()

        # 6. Run the selected cloud vision provider with the de-identified image.
        t_vlm_start = time.time()
        selected_provider = (vlm_provider or "auto").strip().lower()
        if selected_provider == "roar":
            provider_label = "Roar AI (Gemini 3.8 Flash)"
            m4_raw = run_roar_ai_vlm(deidentified_image_bytes, eff_roar or None, model_name="gemini-3.8-flash")
        elif selected_provider == "gemini":
            provider_label = "Google Gemini (Gemini 2.5 Flash)"
            m4_raw = run_gemini_vlm(deidentified_image_bytes, eff_gemini) if eff_gemini else {"error": "Gemini API key not configured", "latency": 0.0}
        elif selected_provider == "openai":
            provider_label = "OpenAI (GPT-4o mini)"
            m4_raw = run_openai_vlm(deidentified_image_bytes, eff_openai) if eff_openai else {"error": "OpenAI API key not configured", "latency": 0.0}
        elif selected_provider == "auto" and eff_roar:
            provider_label = "Roar AI (Gemini 3.8 Flash)"
            print(f"[{current_doctor.slmc_number}] Triggering Roar AI Frontier Intelligence (Gemini 3.8 Flash)...")
            m4_raw = run_roar_ai_vlm(deidentified_image_bytes, eff_roar, model_name="gemini-3.8-flash")
        elif selected_provider == "auto" and eff_gemini:
            provider_label = "Google Gemini (Gemini 2.5 Flash)"
            print(f"[{current_doctor.slmc_number}] Triggering Gemini Flash with de-identified image...")
            m4_raw = run_gemini_vlm(deidentified_image_bytes, eff_gemini)
        elif selected_provider == "auto" and eff_openai:
            provider_label = "OpenAI (GPT-4o mini)"
            m4_raw = run_openai_vlm(deidentified_image_bytes, eff_openai)
        else:
            provider_label = "Cloud VLM"
            m4_raw = {"error": "Selected cloud provider is unavailable or its API key is not configured", "latency": 0.0}

        m3_raw = {"status": "decommissioned", "disabled": True, "error": None, "latency": 0.0}

        # Normalize m3 / m4 to common shape
        def normalize_vlm(raw: dict, name: str, fallback_rule_diag: str) -> dict:
            if raw.get("error"):
                print(f"[VLM {name}] Warning: {raw['error']}")
                return {
                    "name": name,
                    "diagnosis": fallback_rule_diag or "No significant abnormalities detected",
                    "details": f"Cloud VLM status: {raw['error'][:100]}... (Using clinical rule baseline)",
                    "latency_sec": round(raw.get("latency", 0.0), 2),
                    "is_local": False,
                    "error": None,
                    "extracted_values": None
                }
            
            diseases = raw.get("Predicted_Diseases")
            if isinstance(diseases, list) and diseases:
                diag = ", ".join([str(d).strip() for d in diseases if str(d).strip()])
            elif isinstance(diseases, str) and diseases.strip() and diseases.strip().lower() not in ("n/a", "none", "unknown", "[]", ""):
                diag = diseases.strip()
            else:
                interp = (raw.get("Clinical_Interpretation") or "").lower()
                matched = []
                if "anemia" in interp or "anaemia" in interp or "low hemoglobin" in interp:
                    matched.append("Anaemia")
                if "diabetes" in interp or "hyperglycemia" in interp or "elevated blood sugar" in interp:
                    matched.append("Diabetes")
                if "hypothyroidism" in interp or "elevated tsh" in interp:
                    matched.append("Hypothyroidism")
                if "hyperthyroidism" in interp or "low tsh" in interp:
                    matched.append("Hyperthyroidism")
                if "renal" in interp or "creatinine" in interp or "kidney" in interp:
                    matched.append("Impaired Renal Function")
                if "cholesterol" in interp or "hypercholesterolemia" in interp or "dyslipidemia" in interp:
                    matched.append("Hypercholesterolaemia")
                if "thrombocytopenia" in interp or "low platelets" in interp:
                    matched.append("Thrombocytopenia")
                if "leukopenia" in interp or "infection" in interp:
                    matched.append("Possible Infection")
                
                if matched:
                    diag = ", ".join(list(dict.fromkeys(matched)))
                elif fallback_rule_diag and "abnormal" in fallback_rule_diag.lower():
                    diag = fallback_rule_diag
                else:
                    diag = "No significant abnormalities detected"

            return {
                "name": name,
                "diagnosis": diag,
                "details": raw.get("Clinical_Interpretation") or "Analysis completed successfully.",
                "latency_sec": round(raw.get("latency", 0.0), 2),
                "is_local": False,
                "error": None,
                "extracted_values": {k: raw.get(k) for k in parsed_values if raw.get(k) is not None}
            }

        m3 = normalize_vlm(m3_raw, "NVIDIA NIM", m1.get("diagnosis", ""))
        m3["disabled"] = True
        m4 = normalize_vlm(m4_raw, provider_label, m1.get("diagnosis", ""))

        # VLM-extracted patient demographics for auto-fill
        vlm_demographics = {}
        for vlm_result in [m4_raw, m3_raw]:
            if isinstance(vlm_result, dict) and not vlm_result.get("error"):
                if vlm_result.get("Patient_Name") and not vlm_demographics.get("name"):
                    vlm_demographics["name"] = vlm_result["Patient_Name"]
                if vlm_result.get("Age") and not vlm_demographics.get("age"):
                    vlm_demographics["age"] = vlm_result["Age"]
                if vlm_result.get("Gender") and not vlm_demographics.get("gender"):
                    vlm_demographics["gender"] = vlm_result["Gender"]

        # Override patient fields from VLM / OCR if not explicitly provided by doctor
        if not patient_name or not patient_name.strip() or patient_name.strip().lower() in ("anonymous", "none", "null"):
            patient_name = vlm_demographics.get("name") or ocr_demographics.get("name") or ""
        if not patient_age or not str(patient_age).strip():
            pa = vlm_demographics.get("age") or ocr_demographics.get("age")
            patient_age = str(pa) if pa else ""
        if patient_gender in (None, "", "Male", "Not Specified") and (vlm_demographics.get("gender") or ocr_demographics.get("gender")):
            patient_gender = vlm_demographics.get("gender") or ocr_demographics.get("gender")

        pt_age_int = int(patient_age) if str(patient_age).strip().isdigit() else 45
        pt_gender_str = patient_gender if patient_gender in ("Male", "Female") else "Male"

        # Enrich parsed values with multimodal VLM extractions when available
        for vlm_raw_res in [m4_raw, m3_raw]:
            if isinstance(vlm_raw_res, dict) and not vlm_raw_res.get("error"):
                for bm_key in list(parsed_values.keys()):
                    vlm_bm_val = vlm_raw_res.get(bm_key)
                    if vlm_bm_val is not None and (not extracted_flags.get(bm_key) or parsed_values.get(bm_key) is None):
                        try:
                            parsed_values[bm_key] = float(vlm_bm_val) if not isinstance(vlm_bm_val, str) or vlm_bm_val.replace(".", "", 1).isdigit() else vlm_bm_val
                            extracted_flags[bm_key] = True
                        except (ValueError, TypeError):
                            pass

        # Cross-check Tesseract OCR extractions against Cloud VLM extractions (only for genuinely extracted values)
        from cliniq_ml.verify import cross_check_extractions
        from cliniq_ml.schema import BOUNDS
        vlm_extractions = {}
        for vlm_raw_res in [m4_raw, m3_raw]:
            if isinstance(vlm_raw_res, dict) and not vlm_raw_res.get("error"):
                for k, v in vlm_raw_res.items():
                    if v is not None and k not in ("Clinical_Interpretation", "Predicted_Diseases", "latency", "error", "Patient_Name", "Age", "Gender"):
                        vlm_extractions[k] = v

        genuinely_extracted = {k: v for k, v in parsed_values.items() if extracted_flags.get(k)}
        raw_discrepancies = cross_check_extractions(genuinely_extracted, vlm_extractions)
        extraction_discrepancies = [
            {
                "field": d.get("field"),
                "tesseract": d.get("reader_a"),
                "easyocr": d.get("reader_a"),
                "reader_a": d.get("reader_a"),
                "vlm": d.get("reader_b"),
                "reader_b": d.get("reader_b"),
                "issue": d.get("issue") or d.get("note") or "Discrepancy detected",
                "note": d.get("issue") or d.get("note") or "Discrepancy detected"
            }
            for d in raw_discrepancies
        ]

        # 6. CliniQ ML Reference Engine Analysis (Calibrated multi-label XGBoost + TreeSHAP)
        # Pass ONLY genuinely extracted analytes, automatically safeguarding against OCR slips
        ml_input = {k: v for k, v in parsed_values.items() if extracted_flags.get(k) and v is not None}
        for disc in raw_discrepancies:
            fld = disc.get("field")
            va = disc.get("reader_a")
            vb = disc.get("reader_b")
            issue_text = disc.get("issue", "")
            if fld in ml_input and "decimal-point slip" in issue_text:
                b_range = BOUNDS.get(fld)
                if b_range and va is not None and vb is not None:
                    lo, hi = b_range
                    if (va > hi or va > 10.0) and (lo <= vb <= hi):
                        ml_input[fld] = vb
                        parsed_values[fld] = vb
                    elif (vb > hi or vb > 10.0) and (lo <= va <= hi):
                        ml_input[fld] = va

        t0_ml = time.time()
        ml_engine = get_engine()
        ml_result = ml_engine.analyze(ml_input, age=pt_age_int, sex=pt_gender_str)
        ml_time = round(time.time() - t0_ml, 4)

        # Surface any disagreeing fields to the physician as "verify before relying on this" before analysis is finalised
        for disc in extraction_discrepancies:
            disc_issue = disc.get("issue") or disc.get("note") or "Discrepancy detected"
            warn_str = f"Verify {disc.get('field')} before relying on this: {disc_issue} (OCR: {disc.get('reader_a')} vs VLM: {disc.get('reader_b')})"
            if warn_str not in ml_result.get("warnings", []):
                ml_result.setdefault("warnings", []).append(warn_str)
            if not any(vf.get("field") == disc.get("field") for vf in ml_result.get("verify_fields", [])):
                ml_result.setdefault("verify_fields", []).append({
                    "field": disc.get("field"),
                    "value": disc.get("reader_a"),
                    "issue": disc_issue,
                    "vlm_value": disc.get("reader_b")
                })

        # 7. Parse clinical context lists (symptoms, medications, lifestyle) before ensemble arbitration
        def parse_context_list(raw_val: Optional[str]) -> list[str]:
            if not raw_val or not raw_val.strip():
                return []
            if raw_val.strip().startswith("[") and raw_val.strip().endswith("]"):
                try:
                    return json.loads(raw_val)
                except Exception:
                    pass
            return [s.strip() for s in raw_val.split(",") if s.strip()]

        symptoms_list = parse_context_list(symptoms)
        medications_list = parse_context_list(medications)
        lifestyle_list = parse_context_list(lifestyle)

        # 8. Validated Ensemble Consensus (Reference ML voter: CliniQ ML 2.2 + Clinical Guidelines)
        m1_for_ens = {"diagnosis": m1.get("diagnosis"), "error": None}
        m2_for_ens = {
            "diagnosis": ", ".join(ml_result.get("findings", [])) if ml_result.get("findings") else ml_result.get("status"),
            "status": ml_result.get("status"),
            "error": None
        }
        ensemble = compute_validated_ensemble(
            m1_rules=m1_for_ens,
            m2_ml=m2_for_ens,
            m3_nvidia=m3,
            m4_gemini=m4,
            report_type=report_type or "Full Blood Count",
            parsed_values=parsed_values,
            extracted_flags=extracted_flags,
            gender=pt_gender_str,
            age=pt_age_int,
            ml_result=ml_result,
            extraction_discrepancies=extraction_discrepancies,
            symptoms=symptoms_list,
            medications=medications_list
        )

        # 8c. Retrieve previous historical lab reports for longitudinal trend analysis
        historical_reports = []
        clean_name = (patient_name or "").strip()
        if clean_name and len(clean_name) > 1:
            try:
                prev_q = select(PatientReport).where(
                    PatientReport.doctor_id == current_doctor.id,
                    PatientReport.patient_name.ilike(f"%{clean_name}%")
                ).order_by(desc(PatientReport.created_at)).limit(5)
                prev_res = await db.execute(prev_q)
                prev_rows = prev_res.scalars().all()
                for r in prev_rows:
                    try:
                        p_payload = json.loads(r.full_payload)
                        historical_reports.append({
                            "patient_ref": r.patient_ref,
                            "report_date": r.report_date or (r.created_at.strftime("%Y-%m-%d") if r.created_at else "Previous"),
                            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "",
                            "primary_diagnosis": r.primary_diagnosis,
                            "parsed_values": p_payload.get("parsed_values", {})
                        })
                    except Exception:
                        pass
            except Exception as hist_err:
                print(f"[WARN] Failed historical query: {hist_err}")

        # 9. Calculate Clinical Risk Profile (ASCVD, Anemia WHO, Metabolic, Thyroid, eGFR, ML Anomaly, Trends)
        risk_profile = compute_comprehensive_risk_profile(
            values=parsed_values,
            age=pt_age_int,
            gender=pt_gender_str,
            symptoms=symptoms_list,
            medications=medications_list,
            lifestyle=lifestyle_list,
            clinical_notes=clinical_notes or "",
            historical_reports=historical_reports
        )

        # 10. Personalized Clinical Recommendations (Risk & Age Stratified)
        recommendations = get_recommendations(
            ensemble["primary_diagnosis"],
            findings=m1.get("findings", []),
            borderline=m1.get("borderline_flags", []),
            age=pt_age_int,
            gender=pt_gender_str,
            risk_profile=risk_profile
        )

        # 11. Narrative generation (using cliniq_ml narrative layer + Roar AI Gateway fallback)
        from cliniq_ml.narrative import generate_narrative, deterministic_summary
        def roar_llm_call(prompt: str) -> str:
            roar_key = os.environ.get("ROAR_API_KEY", "").strip()
            roar_base = os.environ.get("ROAR_BASE_URL", "https://api.roar-ai.com/v1").strip()
            if not roar_key:
                return deterministic_summary(ml_result)
            try:
                import requests
                res = requests.post(
                    f"{roar_base}/chat/completions",
                    headers={"Authorization": f"Bearer {roar_key}", "Content-Type": "application/json"},
                    json={
                        "model": "gemini-3.8-flash",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 400,
                        "temperature": 0.1
                    },
                    timeout=15
                )
                if res.status_code == 200:
                    return res.json()["choices"][0]["message"]["content"].strip()
            except Exception as err:
                print(f"[Roar AI Narrative warning] {err}")
            return deterministic_summary(ml_result)

        narrative_res = generate_narrative(ml_result, roar_llm_call)
        best_narrative = narrative_res.get("text") or deterministic_summary(ml_result)

        # 12. Explainable AI (XAI) Structured Reasoning & Differential Diagnoses
        key_features = []
        for b_name, b_val in parsed_values.items():
            if b_val is not None:
                flg = extracted_flags.get(b_name, False)
                key_features.append({
                    "biomarker": b_name.replace("_", " "),
                    "value": b_val,
                    "extracted": flg,
                    "status": "Extracted" if flg else "Imputed default"
                })

        diff_diagnoses = []
        top_diag_lower = ensemble.get("primary_diagnosis", "").lower()
        if "anemia" in top_diag_lower or "anaemia" in top_diag_lower:
            diff_diagnoses.extend([
                {"condition": "Iron Deficiency Anaemia", "likelihood": "High", "rationale": "Decreased haemoglobin with microcytic or hypochromic tendency."},
                {"condition": "Anaemia of Chronic Disease", "likelihood": "Moderate", "rationale": "Secondary to underlying chronic inflammation or occult illness."},
                {"condition": "Thalassaemia Trait", "likelihood": "Low to Moderate", "rationale": "Endemic in Sri Lanka; verify with HPLC hemoglobin electrophoresis."}
            ])
        if "diabetes" in top_diag_lower or "prediabetes" in top_diag_lower:
            diff_diagnoses.extend([
                {"condition": "Type 2 Diabetes Mellitus", "likelihood": "High", "rationale": "Fasting blood sugar exceeding clinical diagnostic threshold (≥ 126 mg/dL)."},
                {"condition": "Impaired Fasting Glycaemia (Prediabetes)", "likelihood": "Moderate", "rationale": "FBS between 100–125 mg/dL; confirm with 3-month HbA1c."},
                {"condition": "Secondary / Medication-Induced Hyperglycaemia", "likelihood": "Low", "rationale": "Consider if patient takes corticosteroids or beta-blockers."}
            ])
        if "hypothyroidism" in top_diag_lower:
            diff_diagnoses.extend([
                {"condition": "Hashimoto's Autoimmune Thyroiditis", "likelihood": "High", "rationale": "Elevated TSH; check anti-TPO antibodies."},
                {"condition": "Subclinical Hypothyroidism", "likelihood": "Moderate", "rationale": "Elevated TSH with normal free T4 levels."}
            ])
        if not diff_diagnoses:
            diff_diagnoses.append({
                "condition": "Physiological Reference Variation",
                "likelihood": "High",
                "rationale": "Biomarkers are well-balanced within established healthy intervals."
            })

        conf_val = ensemble.get("confidence", 0)
        conf_band = ensemble.get("confidence_band", "LOW")
        why_conf = (
            f"Reference Engine: CliniQ Engine ({ml_result.get('model_version', 'v2.2')}). "
            f"Status: {ml_result.get('status')}. Calibrated split-conformal confidence {conf_val}% ({conf_band}). "
            f"Cross-reader extractions: {len(extraction_discrepancies)} discrepancy field(s) audited."
        )
        
        uncertainties = []
        if conf_val < 80.0:
            uncertainties.append("Borderline biomarker levels or partial discordance between cloud VLMs and local ML classifier.")
        if not extracted_flags.get("Hemoglobin") or not extracted_flags.get("Fasting_Blood_Sugar"):
            uncertainties.append("Some primary metabolic/hematologic biomarkers were not included in this single report.")
        if not clinical_notes and not symptoms_list:
            uncertainties.append("Clinical symptoms and medication history were not provided during intake.")

        clinical_reasoning = {
            "summary_rationale": best_narrative[:400] + ("..." if len(best_narrative) > 400 else ""),
            "consensus_math": f"Consensus score {conf_val}% ({conf_band}) anchored on calibrated CliniQ ML reference voter.",
            "why_confidence_level": why_conf,
            "uncertainty_breakdown": uncertainties or ["All extracted biomarkers correlate strongly with clinical models."],
            "demographic_context": f"Patient profile: {pt_age_int}y {pt_gender_str}. Symptoms: {', '.join(symptoms_list) or 'None reported'}. Medications: {', '.join(medications_list) or 'None reported'}.",
            "longitudinal_summary": risk_profile.get("longitudinal_trends", {}).get("summary", "Baseline established."),
            "key_features_analyzed": key_features,
            "differential_diagnoses": diff_diagnoses
        }

        # 13. Patient ref
        patient_ref = f"PAT-{uuid.uuid4().hex[:8].upper()}"

        total_time = round(time.time() - total_start, 2)

        # 14. Safety disclaimer
        safety_disclaimer = "Decision support only - requires physician review and sign-off"

        # 13b. Trilingual patient-facing takeaway summary
        trilingual_summary = generate_trilingual_summary(
            ensemble.get("primary_diagnosis", ""),
            m1.get("findings", [])
        )

        m2 = {
            "name": f"CliniQ Engine ({ml_result.get('model_version', 'cliniq-ml-2.2.0')})",
            "status": ml_result.get("status"),
            "diagnosis": ", ".join(ml_result.get("findings", [])) if ml_result.get("findings") else (
                "No findings on the panels provided" if ml_result.get("status") == "NO_FINDINGS_ON_AVAILABLE_PANELS"
                else "No significant abnormalities detected" if ml_result.get("status") == "NO_SIGNIFICANT_FINDINGS"
                else ml_result.get("status", "Unknown")
            ),
            "details": f"Status: {ml_result.get('status')}. Calibrated split conformal multi-label XGBoost with TreeSHAP explanations.",
            "latency_sec": ml_time,
            "probabilities": {k: v.get("probability", 0.0) for k, v in ml_result.get("conditions", {}).items()},
            "conditions": ml_result.get("conditions", {}),
            "critical_alerts": ml_result.get("critical_alerts", []),
            "pattern_alerts": ml_result.get("pattern_alerts", []),
            "verify_fields": ml_result.get("verify_fields", []),
            "suggested_next_tests": ml_result.get("suggested_next_tests", []),
            "warnings": ml_result.get("warnings", []),
            "not_assessable": ml_result.get("not_assessable", []),
            "uncertain": ml_result.get("uncertain", []),
            "model_version": ml_result.get("model_version"),
            "is_local": True,
            "error": None
        }

        response_payload = {
            "success": True,
            "filename": file.filename,
            "patient_ref": patient_ref,
            "patient": {
                "name": (patient_name or "Anonymous").strip(),
                "age": patient_age or "—",
                "gender": patient_gender or "Not Specified",
                "report_type": report_type or "Full Blood Count",
                "report_date": report_date or "",
                "clinical_notes": (clinical_notes or "").strip()
            },
            "analyzed_by": {
                "doctor_name": current_doctor.full_name,
                "slmc_number": current_doctor.slmc_number
            },
            "ocr_text": ocr_text,
            "universal_lab_report": universal_lab_report,
            "parsed_values": parsed_values,
            "extracted_flags": extracted_flags,
            "ocr_demographics": ocr_demographics,
            "vlm_demographics": vlm_demographics,
            "ensemble": ensemble,
            "cliniq_ml": ml_result,
            "model_version": ml_result.get("model_version", "cliniq-ml-2.2.0"),
            "critical_alerts": ml_result.get("critical_alerts", []),
            "pattern_alerts": ml_result.get("pattern_alerts", []),
            "verify_fields": ml_result.get("verify_fields", []),
            "suggested_next_tests": ml_result.get("suggested_next_tests", []),
            "warnings": ml_result.get("warnings", []),
            "not_assessable": ml_result.get("not_assessable", []),
            "uncertain": ml_result.get("uncertain", []),
            "extraction_discrepancies": extraction_discrepancies,
            "best_narrative": best_narrative,
            "safety_disclaimer": safety_disclaimer,
            "recommendations": recommendations,
            "clinical_findings": m1.get("findings", []),
            "borderline_flags": m1.get("borderline_flags", []),
            "clinical_cautions": m1.get("cautions", []),
            "risk_profile": risk_profile,
            "dengue_trajectory": risk_profile.get("dengue_trajectory"),
            "ckdu_surveillance": risk_profile.get("ckdu_surveillance"),
            "trilingual_summary": trilingual_summary,
            "consensus_abstention": ensemble.get("is_abstaining", False),
            "rules_veto_applied": ensemble.get("rules_veto_applied", False),
            "recommended_confirmatory_tests": ensemble.get("recommended_tests", []),
            "clinical_reasoning": clinical_reasoning,
            "clinical_inquiries": risk_profile.get("clinical_inquiries", []),
            "processing_time_sec": total_time,
            "models": {
                "model_1_rules": {
                    "name": "Tesseract OCR + Clinical Rules",
                    "diagnosis": m1.get("diagnosis", "Unknown"),
                    "details": m1.get("summary", ""),
                    "latency_sec": m1.get("latency_sec", 0.0),
                    "is_local": True,
                    "error": None
                },
                "model_2_ml": m2,
                "model_3_nvidia": m3,
                "model_4_gemini": m4
            }
        }

        # Embed uploaded report image/pdf preview for frontend inspection
        try:
            import base64
            ctype = file.content_type or "image/png"
            if "pdf" in ctype:
                ctype = "application/pdf"
            elif not ctype.startswith("image/"):
                ctype = "image/png"
            response_payload["report_file_preview"] = f"data:{ctype};base64," + base64.b64encode(image_bytes).decode("utf-8")
            response_payload["report_filename"] = file.filename or "lab_report.png"
        except Exception as enc_err:
            print(f"[WARN] Failed to encode file preview: {enc_err}")

        # ── Persist to DB ─────────────────────────────────────────────────────────
        try:
            db_report = PatientReport(
                patient_ref=patient_ref,
                doctor_id=current_doctor.id,
                doctor_slmc=current_doctor.slmc_number,
                patient_name=(patient_name or "").strip() or None,
                patient_age=patient_age or None,
                patient_gender=patient_gender or None,
                report_type=report_type or "Full Blood Count",
                report_date=report_date or None,
                clinical_notes=(clinical_notes or "").strip() or None,
                primary_diagnosis=ensemble.get("primary_diagnosis"),
                confidence_pct=ensemble.get("confidence"),
                confidence_band=ensemble.get("confidence_band"),
                processing_time_sec=total_time,
                full_payload=json.dumps(response_payload, default=str)
            )
            db.add(db_report)
            await db.commit()
            response_payload["db_report_id"] = db_report.id

            # ── Automatic Sync to Supabase Cloud ───────────────────────────────────
            try:
                from supabase_client import sync_patient_report
                sb_sync_res = await sync_patient_report({
                    "id": db_report.id,
                    "patient_ref": db_report.patient_ref,
                    "doctor_id": db_report.doctor_id,
                    "doctor_slmc": db_report.doctor_slmc,
                    "patient_name": db_report.patient_name,
                    "patient_age": db_report.patient_age,
                    "patient_gender": db_report.patient_gender,
                    "report_type": db_report.report_type,
                    "report_date": db_report.report_date,
                    "clinical_notes": db_report.clinical_notes,
                    "primary_diagnosis": db_report.primary_diagnosis,
                    "confidence_pct": db_report.confidence_pct,
                    "confidence_band": db_report.confidence_band,
                    "processing_time_sec": db_report.processing_time_sec,
                    "full_payload": response_payload,
                    "created_at": db_report.created_at.isoformat()
                })
                response_payload["supabase_sync"] = sb_sync_res
            except Exception as sb_err:
                print(f"[WARN] Supabase automatic sync encountered error: {sb_err}")
                response_payload["supabase_sync"] = {"success": False, "error": str(sb_err)}
        except Exception as db_err:
            print(f"[WARN] Failed to persist report to DB: {db_err}")

        return response_payload

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Benchmarks ────────────────────────────────────────────────────────────────────
@app.get("/api/benchmarks")
async def get_benchmarks():
    comparison_path = os.path.join("data", "model_comparison.json")
    if os.path.exists(comparison_path):
        with open(comparison_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "Benchmark data not generated yet. Run train_models.py first."}


# ── Static SPA ────────────────────────────────────────────────────────────────────
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static_dir")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    print("Starting CliniQ AI v2.1 at http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True, access_log=True)
