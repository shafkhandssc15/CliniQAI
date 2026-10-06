"""
routes_admin.py — Admin-only endpoints for doctor registration management.
"""

import json
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import get_db
from models_db import Doctor, AuditLog
from dependencies import get_admin_doctor

router = APIRouter(prefix="/api/admin", tags=["Administration"])


class RejectRequest(BaseModel):
    reason: str = "Application did not meet verification requirements."


@router.get("/registrations")
async def list_pending_registrations(
    status: str = "PENDING_ADMIN_APPROVAL",
    admin: Doctor = Depends(get_admin_doctor),
    db: AsyncSession = Depends(get_db)
):
    """List all doctors filtered by status. Default: pending approval queue."""
    result = await db.execute(
        select(Doctor).where(Doctor.status == status).order_by(Doctor.created_at.desc())
    )
    doctors = result.scalars().all()
    return {
        "status_filter": status,
        "count": len(doctors),
        "registrations": [
            {
                "id": d.id,
                "full_name": d.full_name,
                "slmc_number": d.slmc_number,
                "email": d.email,
                "phone": d.phone,
                "nic": d.nic,
                "status": d.status,
                "created_at": d.created_at.isoformat() if d.created_at else None
            }
            for d in doctors
        ]
    }


@router.post("/registrations/{doctor_id}/approve")
async def approve_registration(
    doctor_id: str,
    admin: Doctor = Depends(get_admin_doctor),
    db: AsyncSession = Depends(get_db)
):
    """Approve a pending doctor registration."""
    result = await db.execute(select(Doctor).where(Doctor.id == doctor_id))
    doctor = result.scalar_one_or_none()

    if not doctor:
        raise HTTPException(404, detail="Doctor not found.")
    if doctor.status not in ("PENDING_ADMIN_APPROVAL", "PENDING_EMAIL_VERIFICATION"):
        raise HTTPException(409, detail=f"Cannot approve a doctor with status '{doctor.status}'.")

    doctor.status = "VERIFIED"
    doctor.approved_by = admin.id
    db.add(AuditLog(
        event_type="DOCTOR_APPROVED",
        actor_id=admin.id,
        target_id=doctor.id,
        target_type="doctor",
        details=json.dumps({"approved_slmc": doctor.slmc_number})
    ))
    await db.commit()

    return {
        "success": True,
        "message": f"Dr. {doctor.full_name} ({doctor.slmc_number}) has been approved and can now log in.",
        "doctor_id": doctor.id
    }


@router.post("/registrations/{doctor_id}/reject")
async def reject_registration(
    doctor_id: str,
    body: RejectRequest,
    admin: Doctor = Depends(get_admin_doctor),
    db: AsyncSession = Depends(get_db)
):
    """Reject a pending doctor registration with a reason."""
    result = await db.execute(select(Doctor).where(Doctor.id == doctor_id))
    doctor = result.scalar_one_or_none()

    if not doctor:
        raise HTTPException(404, detail="Doctor not found.")

    doctor.status = "REJECTED"
    doctor.rejection_reason = body.reason.strip()
    db.add(AuditLog(
        event_type="DOCTOR_REJECTED",
        actor_id=admin.id,
        target_id=doctor.id,
        target_type="doctor",
        details=json.dumps({"reason": body.reason})
    ))
    await db.commit()

    return {
        "success": True,
        "message": f"Registration for Dr. {doctor.full_name} rejected.",
        "doctor_id": doctor.id
    }


@router.post("/doctors/{doctor_id}/suspend")
async def suspend_doctor(
    doctor_id: str,
    admin: Doctor = Depends(get_admin_doctor),
    db: AsyncSession = Depends(get_db)
):
    """Suspend an active doctor account."""
    result = await db.execute(select(Doctor).where(Doctor.id == doctor_id))
    doctor = result.scalar_one_or_none()

    if not doctor:
        raise HTTPException(404, detail="Doctor not found.")
    if doctor.id == admin.id:
        raise HTTPException(400, detail="Administrators cannot suspend themselves.")

    doctor.is_suspended = True
    await db.commit()
    return {"success": True, "message": f"Dr. {doctor.full_name} account suspended."}


@router.post("/seed-admin")
async def seed_admin(db: AsyncSession = Depends(get_db)):
    """
    Creates the default admin account if none exists.
    Default credentials: admin@cliniq.lk / Admin@123
    """
    from auth import hash_password
    result = await db.execute(select(Doctor).where(Doctor.role == "ADMIN"))
    if result.scalar_one_or_none():
        return {"message": "Admin account already exists."}

    admin = Doctor(
        full_name="Platform Administrator",
        slmc_number="SLMC-00001",
        email="admin@cliniq.lk",
        phone="+94771234567",
        nic="199012345678",
        password_hash=hash_password("Admin@123"),
        email_verified=True,
        status="VERIFIED",
        role="ADMIN"
    )
    db.add(admin)
    await db.commit()

    return {
        "success": True,
        "message": "Admin account created.",
        "email": "admin@cliniq.lk",
        "password": "Admin@123",
        "warning": "Change this password immediately after first login."
    }

