"""
routes_auth.py — Doctor registration, login, logout, email verification, profile.
"""

import os
import json
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import get_db
from models_db import Doctor, AuthSession, AuditLog
from auth import (
    hash_password, verify_password,
    create_access_token, create_email_verify_token,
    validate_slmc, validate_nic, validate_and_normalize_phone
)
from dependencies import get_current_doctor, get_verified_doctor

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# ── Schemas ─────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    full_name: str
    slmc_number: str
    email: EmailStr
    phone: str
    nic: str
    password: str
    confirm_password: str

    @field_validator("full_name")
    @classmethod
    def name_not_empty(cls, v):
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Full name must be at least 3 characters.")
        if len(v) > 150:
            raise ValueError("Full name must be at most 150 characters.")
        return v

    @field_validator("password")
    @classmethod
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UpdateKeysRequest(BaseModel):
    gemini_api_key: str = ""
    nvidia_api_key: str = ""


# ── Helpers ──────────────────────────────────────────────────────────────────────

async def log_event(db: AsyncSession, event_type: str, actor_id: str = None,
                    target_id: str = None, target_type: str = None,
                    ip: str = None, details: dict = None):
    log = AuditLog(
        event_type=event_type,
        actor_id=actor_id,
        target_id=target_id,
        target_type=target_type,
        ip_address=ip,
        details=json.dumps(details) if details else None
    )
    db.add(log)
    await db.commit()


# ── Routes ────────────────────────────────────────────────────────────────────────

@router.post("/register", status_code=201)
async def register(body: RegisterRequest, request: Request, db: AsyncSession = Depends(get_db)):
    """Register a new doctor. Sends email verification (simulated in dev)."""

    # Password match
    if body.password != body.confirm_password:
        raise HTTPException(422, detail="Passwords do not match.")

    # SLMC validation
    slmc_valid, slmc_norm = validate_slmc(body.slmc_number)
    if not slmc_valid:
        raise HTTPException(422, detail={
            "code": "REG_001",
            "message": "Invalid SLMC number format. Expected: SLMC-XXXXX (4–6 digits)."
        })

    # NIC validation
    nic_result = validate_nic(body.nic)
    if not nic_result["valid"]:
        raise HTTPException(422, detail={"code": "REG_002", "message": nic_result["error"]})

    # Phone validation
    phone_valid, phone_norm = validate_and_normalize_phone(body.phone)
    if not phone_valid:
        raise HTTPException(422, detail={
            "code": "REG_006",
            "message": "Invalid phone number. Use Sri Lankan format: 07XXXXXXXX or +94XXXXXXXXX."
        })

    # Uniqueness checks
    existing = await db.execute(select(Doctor).where(Doctor.slmc_number == slmc_norm))
    if existing.scalar_one_or_none():
        raise HTTPException(409, detail={"code": "REG_003", "message": "This SLMC number is already registered."})

    existing = await db.execute(select(Doctor).where(Doctor.email == body.email.lower()))
    if existing.scalar_one_or_none():
        raise HTTPException(409, detail={"code": "REG_004", "message": "This email address is already registered."})

    existing = await db.execute(select(Doctor).where(Doctor.nic == nic_result["normalized"]))
    if existing.scalar_one_or_none():
        raise HTTPException(409, detail={"code": "REG_005", "message": "This NIC is already associated with an account."})

    # Create doctor record with environment or empty API keys (customizable in settings)
    DEFAULT_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
    DEFAULT_NVIDIA_KEY = os.environ.get("NVIDIA_API_KEY", "")

    verify_token = create_email_verify_token()
    doctor = Doctor(
        full_name=body.full_name.strip(),
        slmc_number=slmc_norm,
        email=body.email.lower(),
        phone=phone_norm,
        nic=nic_result["normalized"],
        password_hash=hash_password(body.password),
        email_verify_token=verify_token,
        status="PENDING_EMAIL_VERIFICATION",
        gemini_api_key=DEFAULT_GEMINI_KEY,
        nvidia_api_key=DEFAULT_NVIDIA_KEY
    )
    db.add(doctor)
    await db.commit()
    await db.refresh(doctor)

    await log_event(db, "REGISTER", actor_id=doctor.id, target_id=doctor.id,
                    target_type="doctor", ip=request.client.host,
                    details={"slmc": slmc_norm, "email": doctor.email})

    doctor.email_verified = True
    doctor.email_verify_token = None
    doctor.status = "PENDING_ADMIN_APPROVAL"
    await db.commit()

    # Sync doctor to Supabase Cloud
    try:
        from supabase_client import sync_doctor
        await sync_doctor({
            "id": doctor.id,
            "full_name": doctor.full_name,
            "slmc_number": doctor.slmc_number,
            "email": doctor.email,
            "email_verified": doctor.email_verified,
            "phone": doctor.phone,
            "nic": doctor.nic,
            "role": doctor.role,
            "status": doctor.status,
            "created_at": doctor.created_at.isoformat() if doctor.created_at else None
        })
    except Exception as sb_err:
        print(f"[WARN] Supabase doctor registration sync error: {sb_err}")

    return {
        "success": True,
        "message": "Registration submitted. Your account is pending administrator approval. You will be notified by email once approved.",
        "doctor_id": doctor.id
    }



@router.get("/verify-email/{token}")
async def verify_email(token: str, db: AsyncSession = Depends(get_db)):
    """Verify email via token link."""
    result = await db.execute(select(Doctor).where(Doctor.email_verify_token == token))
    doctor = result.scalar_one_or_none()
    if not doctor:
        raise HTTPException(400, detail="Invalid or expired email verification link.")

    doctor.email_verified = True
    doctor.email_verify_token = None
    doctor.status = "PENDING_ADMIN_APPROVAL"
    await db.commit()

    return HTMLResponse("""
    <html><body style="font-family:sans-serif;text-align:center;padding:60px">
    <h2 style="color:#10b981">Email verified!</h2>
    <p>Your account is now pending administrator approval.</p>
    <a href="/" style="color:#06b6d4">Return to CliniQ AI</a>
    </body></html>
    """)


@router.post("/login")
async def login(body: LoginRequest, request: Request, db: AsyncSession = Depends(get_db)):
    """Authenticate doctor and return JWT access token."""
    result = await db.execute(select(Doctor).where(Doctor.email == body.email.lower()))
    doctor = result.scalar_one_or_none()

    if not doctor or not verify_password(body.password, doctor.password_hash):
        await log_event(db, "LOGIN_FAILED", ip=request.client.host,
                        details={"email": body.email})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    if doctor.is_suspended:
        raise HTTPException(403, detail="Your account has been suspended. Please contact the administrator.")

    if doctor.status == "PENDING_EMAIL_VERIFICATION":
        raise HTTPException(403, detail="Please verify your email address before logging in.")

    if doctor.status == "PENDING_ADMIN_APPROVAL":
        raise HTTPException(403, detail={
            "code": "AUTH_003",
            "message": "Your account is pending administrator approval. You will be notified once approved."
        })

    if doctor.status == "REJECTED":
        raise HTTPException(403, detail=f"Your application was not approved. Reason: {doctor.rejection_reason or 'Not specified.'}")

    # Create JWT
    token, jti = create_access_token(doctor.id, doctor.role, doctor.status)

    # Store session for revocation tracking
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=60 * 8)
    session = AuthSession(
        doctor_id=doctor.id,
        token_jti=jti,
        expires_at=expires_at.replace(tzinfo=None)
    )
    db.add(session)

    # Update last login
    doctor.last_login_at = datetime.now(timezone.utc).replace(tzinfo=None)
    await db.commit()

    await log_event(db, "LOGIN_SUCCESS", actor_id=doctor.id, ip=request.client.host)

    return {
        "access_token": token,
        "token_type": "bearer",
        "doctor": {
            "id": doctor.id,
            "full_name": doctor.full_name,
            "slmc_number": doctor.slmc_number,
            "email": doctor.email,
            "role": doctor.role,
            "status": doctor.status
        }
    }


@router.post("/logout")
async def logout(request: Request, doctor: Doctor = Depends(get_current_doctor),
                 db: AsyncSession = Depends(get_db)):
    """Invalidate the current JWT session."""
    # Extract JTI from the current token and revoke it
    from fastapi.security import HTTPBearer
    from auth import decode_token

    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "")
    payload = decode_token(token)
    if payload:
        jti = payload.get("jti")
        result = await db.execute(select(AuthSession).where(AuthSession.token_jti == jti))
        session = result.scalar_one_or_none()
        if session:
            session.is_revoked = True
            await db.commit()

    await log_event(db, "LOGOUT", actor_id=doctor.id)
    return {"success": True, "message": "Logged out successfully."}


@router.get("/me")
async def get_me(doctor: Doctor = Depends(get_current_doctor)):
    """Return the current authenticated doctor's profile."""
    DEFAULT_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
    DEFAULT_NVIDIA_KEY = os.environ.get("NVIDIA_API_KEY", "")
    
    return {
        "id": doctor.id,
        "full_name": doctor.full_name,
        "slmc_number": doctor.slmc_number,
        "email": doctor.email,
        "phone": doctor.phone,
        "nic": doctor.nic,
        "role": doctor.role,
        "status": doctor.status,
        "gemini_api_key": doctor.gemini_api_key or DEFAULT_GEMINI_KEY,
        "nvidia_api_key": doctor.nvidia_api_key or DEFAULT_NVIDIA_KEY,
        "has_gemini_key": bool(doctor.gemini_api_key or DEFAULT_GEMINI_KEY),
        "has_nvidia_key": bool(doctor.nvidia_api_key or DEFAULT_NVIDIA_KEY)
    }


@router.post("/update-keys")
async def update_api_keys(body: UpdateKeysRequest,
                          doctor: Doctor = Depends(get_verified_doctor),
                          db: AsyncSession = Depends(get_db)):
    """Store or update the doctor's NVIDIA and Gemini API keys."""
    doctor.gemini_api_key = body.gemini_api_key.strip() or None
    doctor.nvidia_api_key = body.nvidia_api_key.strip() or None
    await db.commit()
    return {"success": True, "message": "API keys updated successfully."}
