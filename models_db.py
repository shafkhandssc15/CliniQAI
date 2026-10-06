"""
models_db.py — SQLAlchemy ORM models for CliniQ AI.
Tables: Doctor, AuthSession, AuditLog, PatientReport
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    String, Boolean, DateTime, Text, Integer, Float, func
)
from sqlalchemy.orm import Mapped, mapped_column
from database import Base


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Doctor(Base):
    __tablename__ = "doctors"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    full_name: Mapped[str] = mapped_column(String(150), nullable=False)
    slmc_number: Mapped[str] = mapped_column(String(20), nullable=False, unique=True, index=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    email_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    phone: Mapped[str] = mapped_column(String(20), nullable=False)
    nic: Mapped[str] = mapped_column(String(12), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    mfa_secret: Mapped[str | None] = mapped_column(Text, nullable=True)
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Status flow:
    # PENDING_EMAIL_VERIFICATION -> PENDING_ADMIN_APPROVAL -> VERIFIED | REJECTED | SUSPENDED
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="PENDING_EMAIL_VERIFICATION")
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="DOCTOR")  # DOCTOR | ADMIN
    is_suspended: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    email_verify_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    approved_by: Mapped[str | None] = mapped_column(String(36), nullable=True)

    # API keys (stored per-doctor, encrypted in production — plaintext for dev)
    gemini_api_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    nvidia_api_key: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    doctor_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    token_jti: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)  # JWT ID (blacklist)
    is_revoked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    actor_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    target_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    target_type: Mapped[str | None] = mapped_column(String(30), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON string
    occurred_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)


class PatientReport(Base):
    """Permanent cross-session storage of every AI diagnostic analysis."""
    __tablename__ = "patient_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_ref: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    # Doctor who ran the analysis
    doctor_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    doctor_slmc: Mapped[str] = mapped_column(String(20), nullable=False)

    # Patient demographics
    patient_name: Mapped[str | None] = mapped_column(String(150), nullable=True)
    patient_age: Mapped[str | None] = mapped_column(String(10), nullable=True)
    patient_gender: Mapped[str | None] = mapped_column(String(20), nullable=True)
    report_type: Mapped[str | None] = mapped_column(String(60), nullable=True)
    report_date: Mapped[str | None] = mapped_column(String(20), nullable=True)
    clinical_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Ensemble AI result
    primary_diagnosis: Mapped[str | None] = mapped_column(String(120), nullable=True)
    confidence_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_band: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Doctor review/sign-off (updated via PATCH)
    review_action: Mapped[str | None] = mapped_column(String(20), nullable=True)
    review_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Full analysis JSON for complete recall
    full_payload: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    processing_time_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
