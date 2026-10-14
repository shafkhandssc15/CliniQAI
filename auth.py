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

import hashlib
import time

# ── Config ─────────────────────────────────────────────────────────────────────
DEFAULT_DEV_SECRET = "cliniq-dev-secret-key-change-in-production-32chars!"
SECRET_KEY = os.environ.get("CLINIQ_SECRET_KEY", DEFAULT_DEV_SECRET)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8  # 8-hour sessions for clinic workflow

# Security guard: refuse to boot in production with default development secret
ENVIRONMENT = os.environ.get("ENVIRONMENT", os.environ.get("CLINIQ_ENV", "development")).lower()
if ENVIRONMENT in ("production", "prod") and (SECRET_KEY == DEFAULT_DEV_SECRET or len(SECRET_KEY) < 32):
    raise RuntimeError(
        "CRITICAL SECURITY CONFIGURATION ERROR: Running in production mode with default or weak CLINIQ_SECRET_KEY. "
        "You must configure a strong, random 32+ character key in environment variables."
    )

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Tamper-Evident Audit Chain ─────────────────────────────────────────────────
def compute_audit_hash(
    prev_hash: Optional[str],
    event_type: str,
    actor_id: Optional[str],
    target_id: Optional[str],
    occurred_at_iso: str,
    details_str: Optional[str]
) -> str:
    """Computes a cryptographic SHA-256 hash linking each audit entry to the previous row."""
    payload = f"{prev_hash or 'GENESIS'}|{event_type}|{actor_id or ''}|{target_id or ''}|{occurred_at_iso}|{details_str or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── In-Memory Login Rate Limiter (Brute-Force Protection) ─────────────────────────
_LOGIN_ATTEMPTS: dict[str, list[float]] = {}
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_WINDOW_SECONDS = 900  # 15 minutes

def is_login_rate_limited(identifier: str) -> tuple[bool, int]:
    """Checks if IP or email has exceeded max failed login attempts."""
    now = time.time()
    history = _LOGIN_ATTEMPTS.get(identifier, [])
    # Keep only attempts within window
    active_attempts = [t for t in history if (now - t) < LOCKOUT_WINDOW_SECONDS]
    _LOGIN_ATTEMPTS[identifier] = active_attempts
    if len(active_attempts) >= MAX_LOGIN_ATTEMPTS:
        retry_after = int(LOCKOUT_WINDOW_SECONDS - (now - active_attempts[0]))
        return True, max(1, retry_after)
    return False, 0

def record_login_failure(identifier: str):
    now = time.time()
    if identifier not in _LOGIN_ATTEMPTS:
        _LOGIN_ATTEMPTS[identifier] = []
    _LOGIN_ATTEMPTS[identifier].append(now)

def record_login_success(identifier: str):
    if identifier in _LOGIN_ATTEMPTS:
        del _LOGIN_ATTEMPTS[identifier]



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
