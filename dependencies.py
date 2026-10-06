"""
dependencies.py — FastAPI dependency injection for authenticated routes.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import get_db
from models_db import Doctor, AuthSession
from auth import decode_token

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_doctor(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db)
) -> Doctor:
    """
    Validates JWT Bearer token.
    Returns the authenticated Doctor ORM object.
    Raises 401 if token is missing/invalid/revoked.
    Raises 403 if account is not VERIFIED.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Please log in.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if credentials is None:
        raise credentials_exception

    payload = decode_token(credentials.credentials)
    if payload is None:
        raise credentials_exception

    doctor_id: str = payload.get("sub")
    jti: str = payload.get("jti")

    if not doctor_id or not jti:
        raise credentials_exception

    # Check token is not blacklisted
    result = await db.execute(
        select(AuthSession).where(
            AuthSession.token_jti == jti,
            AuthSession.is_revoked == True
        )
    )
    if result.scalar_one_or_none():
        raise credentials_exception

    # Fetch doctor
    result = await db.execute(select(Doctor).where(Doctor.id == doctor_id))
    doctor = result.scalar_one_or_none()

    if doctor is None:
        raise credentials_exception

    if doctor.is_suspended:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been suspended. Please contact the administrator."
        )

    return doctor


async def get_verified_doctor(doctor: Doctor = Depends(get_current_doctor)) -> Doctor:
    """Requires VERIFIED status. Pending doctors are rejected."""
    if doctor.status != "VERIFIED":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Your account status is '{doctor.status}'. Only verified doctors can access this feature."
        )
    return doctor


async def get_admin_doctor(doctor: Doctor = Depends(get_current_doctor)) -> Doctor:
    """Requires ADMIN role."""
    if doctor.role != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required."
        )
    return doctor
