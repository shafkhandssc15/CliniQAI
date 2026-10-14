import pytest
import os
import io
from PIL import Image
from auth import compute_audit_hash, is_login_rate_limited, record_login_failure, record_login_success
from ocr_engine import deidentify_report_image


def test_audit_hash_chaining():
    # Initial genesis record
    hash_0 = compute_audit_hash(None, "AUTH_LOGIN", "DOC-123", "doctor", "2026-10-14T10:00:00", "Success")
    assert len(hash_0) == 64
    
    # Second chained record
    hash_1 = compute_audit_hash(hash_0, "REPORT_ANALYZE", "DOC-123", "report_42", "2026-10-14T10:05:00", "Success")
    assert len(hash_1) == 64
    assert hash_1 != hash_0

    # Tampering test: altering the action in row 1 changes its hash, breaking row 2's prev_hash check
    tampered_hash_0 = compute_audit_hash(None, "AUTH_TAMPERED", "DOC-123", "doctor", "2026-10-14T10:00:00", "Success")
    assert tampered_hash_0 != hash_0

    # Row 2 re-evaluated with tampered prior hash does not match original row 2 hash
    recomputed_hash_1 = compute_audit_hash(tampered_hash_0, "REPORT_ANALYZE", "DOC-123", "report_42", "2026-10-14T10:05:00", "Success")
    assert recomputed_hash_1 != hash_1


def test_login_rate_limiting():
    test_slmc = "SLMC-TEST-99999"
    # Clean baseline
    record_login_success(test_slmc)
    is_limited, _ = is_login_rate_limited(test_slmc)
    assert is_limited is False

    # 4 failures should not lock out
    for _ in range(4):
        record_login_failure(test_slmc)
    is_limited, _ = is_login_rate_limited(test_slmc)
    assert is_limited is False

    # 5th failure triggers lockout
    record_login_failure(test_slmc)
    is_limited, retry_after = is_login_rate_limited(test_slmc)
    assert is_limited is True
    assert retry_after > 0

    # Successful login resets the lockout counter
    record_login_success(test_slmc)
    is_limited, _ = is_login_rate_limited(test_slmc)
    assert is_limited is False


def test_deidentify_report_image_redaction():
    # Create a small synthetic image containing patient info
    img = Image.new("RGB", (600, 200), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    # Synthetic OCR results with PII and non-PII
    mock_ocr_results = [
        ([[10, 10], [120, 10], [120, 30], [10, 30]], "Patient Name: Kamal Perera", 0.98),
        ([[10, 40], [150, 40], [150, 60], [10, 60]], "NIC: 198512345678", 0.95),
        ([[10, 70], [130, 70], [130, 90], [10, 90]], "Phone: 0771234567", 0.96),
        ([[10, 100], [180, 100], [180, 120], [10, 120]], "Hemoglobin: 13.5 g/dL", 0.99),
    ]

    redacted_bytes = deidentify_report_image(raw_bytes, ocr_results=mock_ocr_results)
    assert redacted_bytes is not None
    assert len(redacted_bytes) > 0

    # Ensure output is a valid readable JPEG image
    redacted_img = Image.open(io.BytesIO(redacted_bytes))
    assert redacted_img.size == (600, 200)


def test_production_jwt_secret_guard():
    # When ENVIRONMENT=production and SECRET_KEY is default, verify safeguard behavior
    from auth import SECRET_KEY
    if os.environ.get("ENVIRONMENT", "").lower() in ("production", "prod"):
        assert SECRET_KEY != "cliniq-ai-super-secret-jwt-key-2025-change-in-production!"
