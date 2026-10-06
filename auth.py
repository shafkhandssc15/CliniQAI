"""
auth.py — Password hashing, JWT creation/verification, SLMC & NIC validation.
"""

import os
import re
import uuid
import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

# ── Config ─────────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("CLINIQ_SECRET_KEY", "cliniq-dev-secret-key-change-in-production-32chars!")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8  # 8-hour sessions for clinic workflow

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Password ────────────────────────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    return pwd_context.hash(plain[:72])


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain[:72], hashed)


# ── JWT ─────────────────────────────────────────────────────────────────────────
def create_access_token(doctor_id: str, role: str, status: str) -> tuple[str, str]:
    """Returns (token_string, jti)."""
    jti = str(uuid.uuid4())
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": doctor_id,
        "role": role,
        "status": status,
        "jti": jti,
        "exp": expire,
        "iat": datetime.now(timezone.utc)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token, jti


def decode_token(token: str) -> Optional[dict]:
    """Decode and verify JWT. Returns payload dict or None on failure."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


# ── Email verification token ────────────────────────────────────────────────────
def create_email_verify_token() -> str:
    return uuid.uuid4().hex + uuid.uuid4().hex  # 64-char hex token


# ── SLMC validation ─────────────────────────────────────────────────────────────
SLMC_PATTERN = re.compile(r"^SLMC[-\/]\d{4,6}$", re.IGNORECASE)


def validate_slmc(slmc: str) -> tuple[bool, str]:
    """Returns (is_valid, normalized_form)."""
    slmc = slmc.strip().upper()
    if SLMC_PATTERN.match(slmc):
        # Normalize to SLMC-XXXXX
        normalized = re.sub(r"SLMC[\/]", "SLMC-", slmc)
        return True, normalized
    return False, slmc


# ── NIC validation ──────────────────────────────────────────────────────────────
OLD_NIC_PATTERN = re.compile(r"^(\d{9})[VXvx]$")
NEW_NIC_PATTERN = re.compile(r"^(\d{12})$")


def validate_nic(nic: str) -> dict:
    """
    Validates Sri Lankan NIC.
    Returns {"valid": bool, "format": "old"|"new", "error": str|None, "normalized": str}.
    """
    nic = nic.strip().upper()

    if m := OLD_NIC_PATTERN.match(nic):
        digits = m.group(1)
        year = 1900 + int(digits[:2])
        doy = int(digits[2:5])
        fmt = "old"
    elif m := NEW_NIC_PATTERN.match(nic):
        digits = m.group(1)
        year = int(digits[:4])
        doy = int(digits[4:7])
        fmt = "new"
    else:
        return {"valid": False, "error": "Invalid NIC format. Use 9-digit+V/X or 12-digit format.", "normalized": nic}

    actual_doy = doy - 500 if doy > 500 else doy
    if not (1 <= actual_doy <= 366):
        return {"valid": False, "error": "NIC contains an invalid day-of-year value.", "normalized": nic}

    current_year = datetime.now().year
    if not (1900 <= year <= current_year):
        return {"valid": False, "error": f"NIC birth year {year} is implausible.", "normalized": nic}

    return {"valid": True, "format": fmt, "error": None, "normalized": nic}


# ── Phone normalization ─────────────────────────────────────────────────────────
PHONE_PATTERN = re.compile(r"^(\+94|0)[0-9]{9}$")


def validate_and_normalize_phone(phone: str) -> tuple[bool, str]:
    phone = phone.strip().replace(" ", "").replace("-", "")
    if not PHONE_PATTERN.match(phone):
        return False, phone
    if phone.startswith("0"):
        phone = "+94" + phone[1:]
    return True, phone
