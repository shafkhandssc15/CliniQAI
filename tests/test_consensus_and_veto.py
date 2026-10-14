import pytest
from clinical_validator import compute_validated_ensemble, get_recommended_confirmatory_tests


def test_rules_engine_hard_clinical_veto():
    # Severe life-threatening thrombocytopenia (PLT = 35k)
    # Even if ML or VLM predicts "Normal" or "Healthy", Rules Engine must execute a hard veto!
    m1_rules = {"diagnosis": "Thrombocytopenia", "summary": "Platelets 35k"}
    m2_ml = {"diagnosis": "Healthy", "summary": "Normal profile"}
    m3_nvidia = {"diagnosis": "Normal", "summary": "Within limits"}
    m4_gemini = {"diagnosis": "Healthy", "summary": "No findings"}

    parsed_values = {"Platelets": 35}
    extracted_flags = {"Platelets": True}

    res = compute_validated_ensemble(
        m1_rules=m1_rules,
        m2_ml=m2_ml,
        m3_nvidia=m3_nvidia,
        m4_gemini=m4_gemini,
        report_type="Full Blood Count",
        parsed_values=parsed_values,
        extracted_flags=extracted_flags
    )

    assert res["rules_veto_applied"] is True
    assert "Thrombocytopenia" in res["primary_diagnosis"]
    assert res["confidence"] == 99.0
    assert any("CRITICAL" in note or "VETO" in note for note in res["arbitration_notes"])


def test_healthy_guardrail_purged_on_active_pathology():
    # Patient has elevated Fasting Blood Sugar (185 mg/dL)
    # Even if some models vote "Healthy", "Healthy" must be overruled by true lab values
    m1_rules = {"diagnosis": "Diabetes Mellitus", "summary": "FBS 185"}
    m2_ml = {"diagnosis": "Healthy", "summary": "Mild"}
    m3_nvidia = {"diagnosis": "Diabetes Mellitus", "summary": "FBS elevated"}
    m4_gemini = {"diagnosis": "Normal", "summary": "Appears normal"}

    parsed_values = {"Fasting_Blood_Sugar": 185}
    extracted_flags = {"Fasting_Blood_Sugar": True}

    res = compute_validated_ensemble(
        m1_rules=m1_rules,
        m2_ml=m2_ml,
        m3_nvidia=m3_nvidia,
        m4_gemini=m4_gemini,
        report_type="Fasting Blood Sugar",
        parsed_values=parsed_values,
        extracted_flags=extracted_flags
    )

    # "Healthy" and "Normal" must not be in all_votes or primary diagnosis
    assert "Healthy" not in res["all_votes"]
    assert "Normal" not in res["all_votes"]
    assert "Diabetes" in res["primary_diagnosis"]
    assert any("Overruled" in note for note in res["arbitration_notes"])


def test_consensus_disagreement_penalty_and_abstention():
    # Mild borderline finding where models clash sharply:
    # Model 4 says Disease, Model 2 says Normal, without critical lab cutoff
    m1_rules = {"diagnosis": "No significant abnormalities detected", "summary": "Borderline FBS"}
    m2_ml = {"diagnosis": "Normal", "summary": "Normal range"}
    m3_nvidia = {"diagnosis": "Chronic Kidney Disease", "summary": "High risk"}
    m4_gemini = {"diagnosis": "Severe Metabolic Disease", "summary": "High risk"}

    parsed_values = {"Fasting_Blood_Sugar": 105}  # Only mild borderline
    extracted_flags = {"Fasting_Blood_Sugar": True}

    res = compute_validated_ensemble(
        m1_rules=m1_rules,
        m2_ml=m2_ml,
        m3_nvidia=m3_nvidia,
        m4_gemini=m4_gemini,
        report_type="Biochemistry",
        parsed_values=parsed_values,
        extracted_flags=extracted_flags
    )

    # Disagreement penalty must cap confidence in decision-support band
    assert res["confidence"] <= 70.0
    assert any("Disagreement Penalty" in note for note in res["arbitration_notes"])


def test_recommended_confirmatory_tests_generation():
    # Borderline FBS and low Platelets should suggest HbA1c/OGTT and serial FBC / Dengue testing
    parsed_values = {"Fasting_Blood_Sugar": 115, "Platelets": 110}
    extracted_flags = {"Fasting_Blood_Sugar": True, "Platelets": True}
    severities = []

    tests = get_recommended_confirmatory_tests(parsed_values, extracted_flags, severities)
    assert len(tests) >= 2
    assert any("HbA1c" in t or "OGTT" in t for t in tests)
    assert any("FBC" in t or "Platelet" in t or "Dengue" in t for t in tests)
