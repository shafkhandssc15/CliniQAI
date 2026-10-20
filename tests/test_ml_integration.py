import os
import sys
import json
import asyncio
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cliniq_ml.engine import CliniQEngine
from cliniq_ml.verify import cross_check_extractions
from clinical_validator import compute_validated_ensemble
from database import AsyncSessionLocal, init_db
from models_db import Doctor, PatientReport, ConditionFeedback
from sqlalchemy import select


@pytest.fixture(scope="module")
def ml_engine():
    artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts"
    return CliniQEngine(str(artifacts_dir))


def test_not_assessable_panels_never_render_healthy(ml_engine):
    """
    When only a partial panel is provided (e.g. only CBC analytes),
    conditions without required analytes (e.g. diabetes, CKD) are NOT_ASSESSABLE.
    The engine status must be NO_FINDINGS_ON_AVAILABLE_PANELS (never NO_SIGNIFICANT_FINDINGS),
    and consensus arbitration must NEVER render the patient as normal/healthy.
    """
    analytes = {
        "Hemoglobin": 14.2,
        "WBC": 6.8,
        "Platelets": 240,
        "Hematocrit": 42.0
    }
    res = ml_engine.analyze(analytes, age=35, sex=1)

    assert res["status"] == "NO_FINDINGS_ON_AVAILABLE_PANELS"
    assert any("diabetes" in x.lower() for x in res["not_assessable"])
    assert any("kidney" in x.lower() or "ckd" in x.lower() for x in res["not_assessable"])
    assert res["conditions"]["diabetes"]["decision"] == "NOT_ASSESSABLE"
    assert res["conditions"]["ckd"]["decision"] == "NOT_ASSESSABLE"

    # Consensus integration check
    ensemble = compute_validated_ensemble(
        m1_rules={"diagnosis": "Normal", "summary": "CBC normal"},
        m2_ml={"diagnosis": "Normal", "summary": "Normal"},
        m3_nvidia={"diagnosis": "Normal", "summary": "Normal"},
        m4_gemini={"diagnosis": "Normal", "summary": "Normal"},
        report_type="Full Blood Count",
        parsed_values=analytes,
        extracted_flags={},
        ml_result=res
    )

    assert ensemble["status"] == "NO_FINDINGS_ON_AVAILABLE_PANELS"
    assert "Healthy" not in ensemble["primary_diagnosis"]
    assert "No findings on the panels provided" in ensemble["primary_diagnosis"]


def test_critical_values_always_escalate(ml_engine):
    """
    Critical values (e.g. severe thrombocytopenia, life-threatening Hb < 6.0)
    must trigger critical_alerts and status == CRITICAL_ESCALATION,
    regardless of any other voter outputs.
    """
    analytes = {
        "Hemoglobin": 5.4,  # Critical < 6.0 g/dL
        "Platelets": 12,    # Critical < 20 x10^3/uL
        "WBC": 4.5
    }
    res = ml_engine.analyze(analytes, age=40, sex=0)

    assert res["status"] == "CRITICAL_ESCALATION"
    assert len(res["critical_alerts"]) >= 1

    # Consensus integration check: critical alerts must escalate prominently
    ensemble = compute_validated_ensemble(
        m1_rules={"diagnosis": "Severe Anaemia", "summary": "Hb critical"},
        m2_ml={"diagnosis": "Healthy", "summary": "Normal"},
        m3_nvidia={"diagnosis": "Healthy", "summary": "Normal"},
        m4_gemini={"diagnosis": "Healthy", "summary": "Normal"},
        report_type="Full Blood Count",
        parsed_values=analytes,
        extracted_flags={},
        ml_result=res
    )

    assert ensemble["critical_escalation"] is True
    assert "CRITICAL" in ensemble["primary_diagnosis"]
    assert any("CRITICAL" in note for note in ensemble["arbitration_notes"])


def test_pediatric_out_of_scope(ml_engine):
    """
    Patients under 18 years old must return OUT_OF_SCOPE because
    adult reference ranges and calibrations are invalid for paediatrics.
    """
    analytes = {
        "Hemoglobin": 12.0,
        "WBC": 7.0,
        "Platelets": 200
    }
    res = ml_engine.analyze(analytes, age=12, sex=1)

    assert res["status"] == "OUT_OF_SCOPE"
    assert "paediatric" in res["reason"].lower() or "adults" in res["reason"].lower()

    # Consensus arbitration handles out-of-scope gracefully
    ensemble = compute_validated_ensemble(
        m1_rules={"diagnosis": "Normal", "summary": "Normal"},
        m2_ml={"diagnosis": "Normal", "summary": "Normal"},
        m3_nvidia={"diagnosis": "Normal", "summary": "Normal"},
        m4_gemini={"diagnosis": "Normal", "summary": "Normal"},
        report_type="Full Blood Count",
        parsed_values=analytes,
        extracted_flags={},
        ml_result=res
    )

    assert ensemble["status"] == "OUT_OF_SCOPE"
    assert "out_of_scope" in ensemble["primary_diagnosis"].lower() or "paediatric" in ensemble["primary_diagnosis"].lower()


def test_decimal_slip_flagged_by_cross_check():
    """
    A decimal slip between EasyOCR and VLM readers (e.g. 1.4 vs 14.0 for Hb, or 11.0 vs 1.1 for Creatinine)
    must be detected and flagged in cross_check_extractions for physician verification.
    """
    easyocr_values = {
        "hb": 1.4,        # Decimal slip (1.4 vs 14.0)
        "platelets": 250,
        "creatinine": 1.1
    }
    vlm_values = {
        "hb": 14.0,
        "platelets": 250,
        "creatinine": 11.0  # Decimal slip (11.0 vs 1.1)
    }

    discrepancies = cross_check_extractions(easyocr_values, vlm_values)
    assert len(discrepancies) >= 2

    flagged_fields = {d["field"] for d in discrepancies}
    assert "hb" in flagged_fields
    assert "creatinine" in flagged_fields

    for d in discrepancies:
        assert d["reader_a"] != d["reader_b"]
        assert "slip" in d["issue"].lower() or "disagree" in d["issue"].lower()


def test_signoff_feedback_row_written(ml_engine):
    """
    On physician sign-off (confirm / modify / reject), the per-condition labels
    must be stored in the ConditionFeedback table for future recalibration auditing.
    """
    async def _run():
        await init_db()

        async with AsyncSessionLocal() as db:
            import uuid
            res_doc = await db.execute(select(Doctor).where(Doctor.email == "kamal@hospital.lk"))
            doctor = res_doc.scalar_one_or_none()
            if not doctor:
                doctor = Doctor(
                    slmc_number=f"SLMC-{uuid.uuid4().hex[:6]}",
                    full_name="Dr. Test Reviewer",
                    email=f"test_{uuid.uuid4().hex[:6]}@cliniq.lk",
                    phone="+94771122334",
                    nic="199012345679",
                    password_hash="fakehash",
                    status="VERIFIED"
                )
                db.add(doctor)
                await db.commit()
                await db.refresh(doctor)

            # Run engine to obtain real conditions
            analytes = {"Hemoglobin": 8.5, "Platelets": 210, "Fasting_Blood_Sugar": 160}
            ml_res = ml_engine.analyze(analytes, age=55, sex=1)

            payload_json = json.dumps({"cliniq_ml": ml_res, "model_version": ml_res.get("model_version", "2.2.0")})

            pref = f"REF-{uuid.uuid4().hex[:6].upper()}"
            # Create test patient report
            report = PatientReport(
                patient_ref=pref,
                patient_name="Feedback Test Patient",
                patient_age="55",
                patient_gender="Male",
                report_type="Comprehensive",
                doctor_id=doctor.id,
                doctor_slmc=doctor.slmc_number,
                primary_diagnosis="Abnormal Findings",
                confidence_pct=95.0,
                full_payload=payload_json
            )
            db.add(report)
            await db.commit()
            await db.refresh(report)

            # Execute sign-off loop (similar to routes_reports.py review handler)
            action = "confirm"
            notes = "Physician verified all active findings."

            from cliniq_ml.schema import LABELS
            conditions = ml_res.get("conditions", {})

            for cond_key in LABELS:
                cond_info = conditions.get(cond_key, {})
                model_dec = cond_info.get("decision", "NOT_ASSESSABLE")
                model_prob = cond_info.get("probability")
                final_label = 1 if model_dec == "POSITIVE" else (0 if model_dec == "NEGATIVE" else None)

                fb = ConditionFeedback(
                    report_id=report.id,
                    patient_ref=report.patient_ref,
                    doctor_id=doctor.id,
                    condition_name=cond_key,
                    model_decision=model_dec,
                    model_probability=model_prob,
                    physician_decision=action.upper(),
                    final_label=final_label,
                    review_notes=notes
                )
                db.add(fb)

            report.review_action = action
            report.review_notes = notes
            await db.commit()

            # Query back ConditionFeedback table
            stmt = select(ConditionFeedback).where(ConditionFeedback.report_id == report.id)
            result = await db.execute(stmt)
            fb_rows = result.scalars().all()

            assert len(fb_rows) == len(LABELS)
            cond_names = {r.condition_name for r in fb_rows}
            assert "anemia" in cond_names
            assert "diabetes" in cond_names
            assert all(r.physician_decision == "CONFIRM" for r in fb_rows)

    asyncio.run(_run())
