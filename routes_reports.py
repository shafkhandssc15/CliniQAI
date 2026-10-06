"""
routes_reports.py — Patient report CRUD for CliniQ AI.
Endpoints:
  GET  /api/reports          — list current doctor's reports (paginated)
  GET  /api/reports/{ref}    — get single report by patient_ref
  PATCH /api/reports/{ref}/review — doctor sign-off
"""

import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies import get_verified_doctor
from models_db import Doctor, PatientReport

router = APIRouter(prefix="/api/reports", tags=["reports"])


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def report_to_dict(r: PatientReport) -> dict:
    return {
        "id":                r.id,
        "patient_ref":       r.patient_ref,
        "patient_name":      r.patient_name,
        "patient_age":       r.patient_age,
        "patient_gender":    r.patient_gender,
        "report_type":       r.report_type,
        "report_date":       r.report_date,
        "clinical_notes":    r.clinical_notes,
        "primary_diagnosis": r.primary_diagnosis,
        "confidence_pct":    r.confidence_pct,
        "confidence_band":   r.confidence_band,
        "review_action":     r.review_action,
        "review_notes":      r.review_notes,
        "reviewed_at":       r.reviewed_at.isoformat() + "Z" if r.reviewed_at else None,
        "processing_time_sec": r.processing_time_sec,
        "created_at":        r.created_at.isoformat() + "Z",
        "has_full_payload":  bool(r.full_payload and r.full_payload != "{}"),
    }


# ── List reports ──────────────────────────────────────────────────────────────────
@router.get("")
async def list_reports(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    diagnosis: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
    current_doctor: Doctor = Depends(get_verified_doctor)
):
    q = select(PatientReport).where(
        PatientReport.doctor_id == current_doctor.id
    )
    if diagnosis:
        q = q.where(PatientReport.primary_diagnosis.ilike(f"%{diagnosis}%"))

    q = q.order_by(desc(PatientReport.created_at)).offset((page - 1) * limit).limit(limit)
    result = await db.execute(q)
    reports = result.scalars().all()

    # Total count
    count_q = select(PatientReport).where(PatientReport.doctor_id == current_doctor.id)
    count_result = await db.execute(count_q)
    total = len(count_result.scalars().all())

    return {
        "reports": [report_to_dict(r) for r in reports],
        "total": total,
        "page": page,
        "limit": limit,
        "pages": max(1, -(-total // limit))  # ceiling division
    }


# ── Get single report with full payload ──────────────────────────────────────────
@router.get("/{patient_ref}")
async def get_report(
    patient_ref: str,
    db: AsyncSession = Depends(get_db),
    current_doctor: Doctor = Depends(get_verified_doctor)
):
    result = await db.execute(
        select(PatientReport).where(
            PatientReport.patient_ref == patient_ref,
            PatientReport.doctor_id == current_doctor.id
        )
    )
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")

    data = report_to_dict(report)
    # Include full payload for recall
    try:
        data["full_payload"] = json.loads(report.full_payload)
    except Exception:
        data["full_payload"] = {}
    return data


# ── Doctor sign-off / update review ──────────────────────────────────────────────
class ReviewBody(BaseModel):
    action: str       # confirm | modify | reject
    notes: Optional[str] = None


@router.patch("/{patient_ref}/review")
async def update_review(
    patient_ref: str,
    body: ReviewBody,
    db: AsyncSession = Depends(get_db),
    current_doctor: Doctor = Depends(get_verified_doctor)
):
    result = await db.execute(
        select(PatientReport).where(
            PatientReport.patient_ref == patient_ref,
            PatientReport.doctor_id == current_doctor.id
        )
    )
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")

    valid_actions = {"confirm", "modify", "reject"}
    if body.action not in valid_actions:
        raise HTTPException(status_code=422, detail=f"action must be one of {valid_actions}")

    report.review_action = body.action
    report.review_notes  = (body.notes or "").strip() or None
    report.reviewed_at   = utcnow()
    await db.commit()

    # Automatic Supabase sync for doctor review
    sb_review_res = None
    try:
        from supabase_client import sync_report_review
        sb_review_res = await sync_report_review(
            patient_ref=patient_ref,
            review_action=report.review_action,
            review_notes=report.review_notes,
            reviewed_at=report.reviewed_at
        )
    except Exception as sb_err:
        print(f"[WARN] Supabase review sync error: {sb_err}")

    return {
        "success": True,
        "patient_ref": patient_ref,
        "review_action": report.review_action,
        "reviewed_at": report.reviewed_at.isoformat(),
        "supabase_sync": sb_review_res
    }
