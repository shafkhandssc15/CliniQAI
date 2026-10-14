import pytest
from clinical_risk_engine import (
    calculate_anemia_risk,
    calculate_metabolic_risk,
    calculate_ascvd_risk,
    calculate_dengue_risk,
    calculate_ckdu_risk,
    generate_trilingual_summary
)


def test_who_anemia_guidelines():
    # WHO Severe Anaemia (< 8.0 g/dL)
    sev = calculate_anemia_risk(hb=7.5, gender="Male", age=45)
    assert sev["risk_level"] == "Severe Risk"
    assert "Severe Anaemia" in sev["grade"]
    assert sev["color"] == "#ef4444"

    # WHO Critical Transfusion Alert (< 6.5 g/dL)
    crit = calculate_anemia_risk(hb=6.2, gender="Female", age=30)
    assert crit["risk_level"] == "Critical / Life-Threatening"
    assert "EMERGENCY PROTOCOL" in crit["action"]

    # WHO Moderate Anaemia (8.0 - 10.9 g/dL)
    mod = calculate_anemia_risk(hb=9.5, gender="Female", age=28)
    assert mod["risk_level"] == "Moderate Risk"
    assert "Moderate Anaemia" in mod["grade"]
    assert mod["color"] == "#f59e0b"

    # WHO Mild Anaemia (11.0 - 11.9 g/dL for women)
    mild_f = calculate_anemia_risk(hb=11.4, gender="Female", age=35)
    assert "Mild Anaemia" in mild_f["grade"]

    # Normal Hb
    norm = calculate_anemia_risk(hb=14.5, gender="Male", age=40)
    assert norm["grade"] == "No Anaemia"


def test_atp_iii_metabolic_syndrome_5_criteria():
    # Patient with high FBS, elevated BP, and high BMI
    res = calculate_metabolic_risk(
        fbs=135,
        chol=240,
        age=52,
        gender="Male",
        symptoms=["Hypertension", "Fatigue"],
        lifestyle=["BMI 29.5", "Sedentary"],
        clinical_notes="BP 145/92 mmHg"
    )
    assert res["score"] >= 3
    assert "High Metabolic Syndrome" in res["status"]
    assert "NCEP ATP III" in res["guideline"]


def test_south_asian_ascvd_risk_calibration():
    # Baseline 55yo male with diabetes & borderline cholesterol
    res = calculate_ascvd_risk(
        age=55,
        gender="Male",
        total_chol=230,
        fbs=140,
        lifestyle=["Smoker", "South Asian descent"]
    )
    assert "South Asian Adjusted" in res["category"]
    assert "QRISK3" in res["population_adjustment"]
    assert res["score_pct"] > 0


def test_sri_lankan_dengue_trajectory():
    # Critical phase: Platelets < 50k and elevated PCV / hematocrit
    values = {"Platelets": 42, "PCV": 49.5, "WBC": 3.1}
    prev_report = [{"parsed_values": {"Platelets": 95, "PCV": 41.0}, "created_at": "2026-10-12"}]
    
    dengue = calculate_dengue_risk(values, historical_reports=prev_report)
    assert "HIGH DENGUE WARNING RISK" in dengue["status"]
    assert dengue["color"] == "#dc2626"
    assert any("Hemoconcentration" in f or "Platelet" in f for f in dengue["flags"])
    assert "NDCU" in dengue["guideline"]


def test_sri_lankan_ckdu_surveillance():
    # High risk farmer from Anuradhapura / NCP with elevated creatinine
    values = {"Creatinine": 1.6}
    ckdu = calculate_ckdu_risk(
        values=values,
        age=48,
        gender="Male",
        clinical_notes="Farming paddy in Anuradhapura, Medawachchiya",
        symptoms=["Back pain", "fatigue"]
    )
    assert "High CKDu Surveillance Flag" in ckdu["status"]
    assert ckdu["color"] == "#ef4444"
    assert "Ministry of Health CKDu" in ckdu["guideline"]


def test_trilingual_patient_summaries():
    # Test diabetes takeaway
    res = generate_trilingual_summary("Diabetes Mellitus")
    assert "Diabetes" in res["english"]
    assert "දියවැඩියාව" in res["sinhala"]
    assert "நீரிழிவு" in res["tamil"]

    # Test anemia takeaway
    res_anemia = generate_trilingual_summary("Microcytic Anemia")
    assert "Anemia" in res_anemia["english"]
    assert "රක්තහීනතාවය" in res_anemia["sinhala"]
    assert "இரத்த சோகை" in res_anemia["tamil"]
