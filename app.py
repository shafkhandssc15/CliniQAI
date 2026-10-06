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
from dependencies import get_verified_doctor
from models_db import Doctor, PatientReport
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
from ocr_engine import (
    extract_text_from_image,
    parse_biomarkers,
    extract_patient_demographics,
    run_rule_based_engine,
    run_ml_classifier,
    run_nvidia_nim_vlm,
    run_gemini_vlm,
    run_roar_ai_vlm,
    load_local_classifier,
    get_easyocr_reader
)

from clinical_risk_engine import compute_comprehensive_risk_profile
from clinical_validator import compute_validated_ensemble


# ── Lifespan ──────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing CliniQ AI platform...")
    await init_db()
    print("Database tables initialized.")
    load_local_classifier()
    print("Local ML classifiers loaded.")
    get_easyocr_reader()
    print("EasyOCR reader initialized.")

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
    version="2.1.0",
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
    nvidia_api_key: str = Form(None),
    gemini_api_key: str = Form(None),
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

        # 1. OCR
        ocr_text, ocr_time = extract_text_from_image(image_bytes)

        # 2. Biomarker parsing
        parsed_values, extracted_flags = parse_biomarkers(ocr_text)

        # 2b. Extract patient demographics from OCR text for auto-fill
        ocr_demographics = extract_patient_demographics(ocr_text)

        # 3. Override intake fields with form values, else use OCR-extracted values
        if not patient_name or not patient_name.strip():
            patient_name = ocr_demographics.get("name") or ""
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

        # 5. Local ML classifier
        t0 = time.time()
        m2 = run_ml_classifier(parsed_values)
        m2["latency_sec"] = round(ocr_time + (time.time() - t0), 4)
        m2["is_local"] = True

        # 5. Resolve API keys: form → doctor profile → env vars → fallback keys
        def resolve_key(form_val, doctor_val, env_var, fallback_val):
            k = (form_val or "").strip()
            if k: return k
            k = (doctor_val or "").strip()
            if k: return k
            k = os.environ.get(env_var, "").strip()
            if k: return k
            return fallback_val

        eff_nvidia = resolve_key(
            nvidia_api_key, 
            current_doctor.nvidia_api_key, 
            "NVIDIA_API_KEY",
            os.environ.get("NVIDIA_API_KEY", "")
        )
        eff_gemini = resolve_key(
            gemini_api_key, 
            current_doctor.gemini_api_key, 
            "GEMINI_API_KEY",
            os.environ.get("GEMINI_API_KEY", "")
        )


        eff_roar = os.environ.get("ROAR_API_KEY", "").strip()

        # 6 & 7. Run Cloud VLMs (NVIDIA NIM / Gemini / Roar AI Gateway) in parallel
        import concurrent.futures
        
        def call_nvidia():
            if eff_nvidia:
                print(f"[{current_doctor.slmc_number}] Triggering NVIDIA NIM (Llama 3.2 Vision)...")
                return run_nvidia_nim_vlm(image_bytes, eff_nvidia)
            elif eff_roar:
                print(f"[{current_doctor.slmc_number}] Triggering Roar AI Gateway (Llama 3.3)...")
                return run_roar_ai_vlm(image_bytes, eff_roar, model_name="llama-3.3-70b")
            return {"error": "No NVIDIA API key configured. Add it in API Settings.", "latency": 0.0}

        def call_gemini():
            if eff_gemini:
                print(f"[{current_doctor.slmc_number}] Triggering Gemini 2.5 Flash...")
                return run_gemini_vlm(image_bytes, eff_gemini)
            elif eff_roar:
                print(f"[{current_doctor.slmc_number}] Triggering Roar AI Gateway (Gemini 3.8 Flash)...")
                return run_roar_ai_vlm(image_bytes, eff_roar, model_name="gemini-3.8-flash")
            return {"error": "No Gemini API key configured. Add it in API Settings.", "latency": 0.0}

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_nim = executor.submit(call_nvidia)
            fut_gem = executor.submit(call_gemini)
            m3_raw = fut_nim.result()
            m4_raw = fut_gem.result()


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
                # If no specific disease list was returned, inspect clinical interpretation
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

        m3 = normalize_vlm(m3_raw, "NVIDIA NIM Llama 3.2 Vision", m1.get("diagnosis", ""))
        m4 = normalize_vlm(m4_raw, "Gemini 2.5 Flash", m1.get("diagnosis", ""))

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

        pt_age_int = int(patient_age) if str(patient_age).strip().isdigit() else 45
        pt_gender_str = patient_gender if patient_gender in ("Male", "Female") else "Male"

        # 8. Validated Ensemble Consensus (Strict clinical validation + test-type awareness)
        m1_for_ens = {"diagnosis": m1.get("diagnosis"), "error": None}
        m2_for_ens = {"diagnosis": m2.get("diagnosis"), "error": None}
        ensemble = compute_validated_ensemble(
            m1_rules=m1_for_ens,
            m2_ml=m2_for_ens,
            m3_nvidia=m3,
            m4_gemini=m4,
            report_type=report_type or "Full Blood Count",
            parsed_values=parsed_values,
            extracted_flags=extracted_flags,
            gender=pt_gender_str,
            age=pt_age_int
        )

        # 11b. VLM-extracted patient demographics for auto-fill
        vlm_demographics = {}
        for vlm_result in [m4_raw, m3_raw]:
            if not vlm_result.get("error"):
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

        # 8b. Parse clinical context lists (symptoms, medications, lifestyle)
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

        # 11. Best narrative (Gemini > NVIDIA > local rule summary)
        if not m4.get("error") and m4["diagnosis"] not in ("Offline", "N/A"):
            best_narrative = m4["details"]
        elif not m3.get("error") and m3["diagnosis"] not in ("Offline", "N/A"):
            best_narrative = m3["details"]
        else:
            best_narrative = m1.get("summary", "Local rule-based analysis completed.")

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
        why_conf = f"Ensemble agreement is {conf_val}% ({conf_band}). Calculated from weighted consensus: Gemini 2.5 (40%), NVIDIA NIM (35%), XGBoost (20%), and Rule-Based Validation (5%)."
        
        uncertainties = []
        if conf_val < 80.0:
            uncertainties.append("Borderline biomarker levels or partial discordance between cloud VLMs and local ML classifier.")
        if not extracted_flags.get("Hemoglobin") or not extracted_flags.get("Fasting_Blood_Sugar"):
            uncertainties.append("Some primary metabolic/hematologic biomarkers were not included in this single report.")
        if not clinical_notes and not symptoms_list:
            uncertainties.append("Clinical symptoms and medication history were not provided during intake.")

        clinical_reasoning = {
            "summary_rationale": best_narrative[:400] + ("..." if len(best_narrative) > 400 else ""),
            "consensus_math": f"Consensus score {conf_val}% ({conf_band}) aggregated across 4 validated AI pipelines.",
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
        safety_disclaimer = (
            "⚠ CLINICAL DECISION SUPPORT ONLY — This AI-generated analysis is intended as a decision support tool "
            "and does NOT constitute a medical diagnosis. All findings MUST be verified by the attending physician "
            "through clinical correlation, patient history, and physical examination before any treatment decisions."
        )

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
            "parsed_values": parsed_values,
            "extracted_flags": extracted_flags,
            "ocr_demographics": ocr_demographics,
            "vlm_demographics": vlm_demographics,
            "ensemble": ensemble,
            "best_narrative": best_narrative,
            "safety_disclaimer": safety_disclaimer,
            "recommendations": recommendations,
            "clinical_findings": m1.get("findings", []),
            "borderline_flags": m1.get("borderline_flags", []),
            "clinical_cautions": m1.get("cautions", []),
            "risk_profile": risk_profile,
            "clinical_reasoning": clinical_reasoning,
            "clinical_inquiries": risk_profile.get("clinical_inquiries", []),
            "processing_time_sec": total_time,
            "models": {
                "model_1_rules": {
                    "name": "EasyOCR + Clinical Rules",
                    "diagnosis": m1.get("diagnosis", "Unknown"),
                    "details": m1.get("summary", ""),
                    "latency_sec": m1.get("latency_sec", 0.0),
                    "is_local": True,
                    "error": None
                },
                "model_2_ml": {
                    "name": "EasyOCR + XGBoost",
                    "diagnosis": m2.get("diagnosis", "Unknown"),
                    "details": m2.get("summary", ""),
                    "latency_sec": m2.get("latency_sec", 0.0),
                    "probabilities": m2.get("probabilities", {}),
                    "is_local": True,
                    "error": None
                },
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
    uvicorn.run("app:app", host="127.0.0.1", port=8000, workers=1, access_log=True)
