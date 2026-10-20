"""tests/test_accuracy_enhancements.py — Verification of clinical accuracy improvements."""
import os
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from cliniq_ml.real_data_loader import derive_clinical_labels, load_real_clinical_dataset
from cliniq_ml.engine import CliniQEngine
from clinical_validator import compute_validated_ensemble


def test_real_clinical_label_derivation():
    """Verify that ADA, WHO, and KDIGO clinical guidelines derive accurate disease labels."""
    df = pd.DataFrame([
        # Patient 1: Diabetic (FBS >= 126) + Anemic (Hb 10 < 13 for male)
        {"sex": 1, "hb": 10.0, "fbs": 140.0, "creatinine": 0.9, "total_chol": 180.0, "tsh": 2.0, "wbc": 6.5},
        # Patient 2: Normal
        {"sex": 0, "hb": 13.5, "fbs": 85.0, "creatinine": 0.8, "total_chol": 170.0, "tsh": 1.5, "wbc": 5.5},
        # Patient 3: CKD (Creatinine 2.5) + Hypothyroidism (TSH 8.5)
        {"sex": 1, "hb": 14.0, "fbs": 90.0, "creatinine": 2.5, "total_chol": 190.0, "tsh": 8.5, "wbc": 7.0},
    ])
    labels = derive_clinical_labels(df)
    
    assert labels.loc[0, "diabetes"] == 1
    assert labels.loc[0, "anemia"] == 1
    assert labels.loc[1, "diabetes"] == 0
    assert labels.loc[1, "anemia"] == 0
    assert labels.loc[2, "ckd"] == 1
    assert labels.loc[2, "hypothyroidism"] == 1


def test_clinical_context_medications_recognition():
    """Verify that a patient taking Metformin is flagged with Diabetes even if blood sugar is normal."""
    ml_result = {
        "status": "NO_SIGNIFICANT_FINDINGS",
        "findings": [],
        "uncertain": [],
        "not_assessable": [],
        "critical_alerts": [],
        "conditions": {}
    }
    
    parsed_values = {
        "Fasting_Blood_Sugar": 92.0,  # Normal glucose
        "Hemoglobin": 14.2,
        "Creatinine": 0.9
    }
    extracted_flags = {"Fasting_Blood_Sugar": True, "Hemoglobin": True, "Creatinine": True}

    res = compute_validated_ensemble(
        m1_rules={"diagnosis": "Normal", "error": None},
        m2_ml={"diagnosis": "Normal", "error": None},
        m3_nvidia={"diagnosis": "Normal", "error": None},
        m4_gemini={"diagnosis": "Normal", "error": None},
        report_type="Full Blood Count",
        parsed_values=parsed_values,
        extracted_flags=extracted_flags,
        gender="Male",
        age=52,
        ml_result=ml_result,
        medications=["Metformin 500mg daily", "Atorvastatin 20mg"]
    )

    # Should detect both pharmacologically controlled diabetes and dyslipidemia
    assert "Diabetes Mellitus (Controlled on Pharmacotherapy)" in res["primary_diagnosis"]
    assert "Dyslipidemia (Managed on Statin Therapy)" in res["primary_diagnosis"]
    assert res["status"] == "ABNORMAL"


def test_guideline_primary_arbiter_override():
    """Verify that ADA/WHO guidelines override when ML returns no findings despite diagnostic thresholds."""
    ml_result = {
        "status": "NO_SIGNIFICANT_FINDINGS",
        "findings": [],
        "uncertain": [],
        "not_assessable": [],
        "critical_alerts": [],
        "conditions": {}
    }
    
    parsed_values = {
        "Fasting_Blood_Sugar": 138.0,  # Exceeds ADA threshold of 126
        "Hemoglobin": 9.5,            # Exceeds WHO Anemia threshold (<12 for female)
        "Creatinine": 0.8
    }
    extracted_flags = {"Fasting_Blood_Sugar": True, "Hemoglobin": True, "Creatinine": True}

    res = compute_validated_ensemble(
        m1_rules={"diagnosis": "Normal", "error": None},
        m2_ml={"diagnosis": "Normal", "error": None},
        m3_nvidia={"diagnosis": "Normal", "error": None},
        m4_gemini={"diagnosis": "Normal", "error": None},
        report_type="Full Blood Count",
        parsed_values=parsed_values,
        extracted_flags=extracted_flags,
        gender="Female",
        age=45,
        ml_result=ml_result
    )

    assert "Diabetes Mellitus" in res["primary_diagnosis"]
    assert "Anaemia" in res["primary_diagnosis"]
    assert res["status"] == "ABNORMAL"


def test_recalibrate_site_with_custom_alphas(tmp_path):
    """Verify that site recalibration accepts custom alpha_pos and alpha_neg budgets."""
    engine = CliniQEngine("artifacts")
    
    # Create minimal synthetic site dataset
    from cliniq_ml.synth import generate
    site_data = generate(100, 1, seed=42)
    from cliniq_ml.schema import RAW_FEATURES, LABELS
    
    out_dir = tmp_path / "custom_site"
    rep = engine.recalibrate_site(
        site_data[RAW_FEATURES],
        site_data[LABELS],
        out_dir=out_dir,
        min_cases=50,
        min_pos=5,
        alpha_pos=0.02,  # 98% sensitivity guarantee
        alpha_neg=0.15   # 85% specificity guarantee
    )
    
    assert out_dir.exists()
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["site_recalibrated"] is True
    assert meta["alpha_pos"] == 0.02
    assert meta["alpha_neg"] == 0.15
