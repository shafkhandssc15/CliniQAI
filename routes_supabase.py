"""
routes_supabase.py — Supabase API routes for status checking, on-demand synchronization,
and schema inspection for CliniQ AI.
"""

import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies import get_verified_doctor
from models_db import Doctor, PatientReport
from supabase_client import (
    test_connection,
    sync_patient_report,
    sync_batch_patient_reports,
    get_supabase_config
)


router = APIRouter(prefix="/api/supabase", tags=["supabase"])


@router.get("/status")
async def get_supabase_status(
    current_doctor: Doctor = Depends(get_verified_doctor)
):
    """Check connectivity to the configured Supabase project and check table readiness."""
    status = await test_connection()
    url, key = get_supabase_config()
    # Mask key for privacy
    masked_key = (key[:8] + "..." + key[-6:]) if len(key) > 14 else "configured"
    status["masked_key"] = masked_key
    return status


@router.get("/schema", response_class=PlainTextResponse)
async def get_schema_sql():
    """Returns the SQL migration script to execute in Supabase SQL Editor."""
    schema_path = os.path.join(os.path.dirname(__file__), "supabase_schema.sql")
    if os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            return f.read()
    return "-- supabase_schema.sql not found."


@router.post("/sync-all")
async def sync_all_reports(
    db: AsyncSession = Depends(get_db),
    current_doctor: Doctor = Depends(get_verified_doctor)
):
    """
    Synchronizes all local reports belonging to the current doctor into Supabase.
    Admin doctors will sync all reports in the system.
    """
    # Check connection first
    conn_info = await test_connection()
    if not conn_info.get("reachable"):
        raise HTTPException(
            status_code=502,
            detail=f"Cannot reach Supabase: {conn_info.get('message') or conn_info.get('error')}"
        )
    if not conn_info.get("table_ready"):
        raise HTTPException(
            status_code=400,
            detail="The 'patient_reports' table is not created in Supabase yet. Please run supabase_schema.sql in your Supabase SQL editor first."
        )

    # Fetch local reports
    if current_doctor.role == "ADMIN":
        q = select(PatientReport).order_by(desc(PatientReport.created_at))
    else:
        q = select(PatientReport).where(
            PatientReport.doctor_id == current_doctor.id
        ).order_by(desc(PatientReport.created_at))

    result = await db.execute(q)
    reports = result.scalars().all()

    reports_list = []
    for r in reports:
        reports_list.append({
            "id": r.id,
            "patient_ref": r.patient_ref,
            "doctor_id": r.doctor_id,
            "doctor_slmc": r.doctor_slmc,
            "patient_name": r.patient_name,
            "patient_age": r.patient_age,
            "patient_gender": r.patient_gender,
            "report_type": r.report_type,
            "report_date": r.report_date,
            "clinical_notes": r.clinical_notes,
            "primary_diagnosis": r.primary_diagnosis,
            "confidence_pct": r.confidence_pct,
            "confidence_band": r.confidence_band,
            "review_action": r.review_action,
            "review_notes": r.review_notes,
            "reviewed_at": r.reviewed_at.isoformat() if r.reviewed_at else None,
            "full_payload": r.full_payload,
            "processing_time_sec": r.processing_time_sec,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })

    res = await sync_batch_patient_reports(reports_list)
    synced = res.get("synced", 0)
    failed = len(reports_list) - synced if not res.get("success") else 0

    return {
        "success": res.get("success", False),
        "total": len(reports_list),
        "synced": synced,
        "failed": failed,
        "errors": res.get("errors", []),
        "message": f"Successfully synced {synced} of {len(reports_list)} reports to Supabase Cloud."
    }

