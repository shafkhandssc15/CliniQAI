"""
supabase_client.py — Automatic Supabase Cloud Database Integration for CliniQ AI.

Provides resilient, asynchronous syncing of:
  - Patient diagnostic reports (created or updated)
  - Doctor review sign-offs
  - Doctor profiles
  - Audit events

Uses direct HTTP PostgREST API with publishable/anon or secret keys,
bypassing SDK version incompatibilities and guaranteeing full async compatibility.
"""

import os
import json
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import httpx
from dotenv import load_dotenv

# Load .env file
load_dotenv()

DEFAULT_SUPABASE_URL = "https://gdxjzvbbgurmjqrxgqjh.supabase.co"
DEFAULT_SUPABASE_KEY = "sb_publishable_7mRE0EaPvSpW-4deUma7qw__yltyWtp"



def get_supabase_config():
    """Retrieve cleaned Supabase URL and Key from environment or defaults."""
    raw_url = os.environ.get("SUPABASE_URL", DEFAULT_SUPABASE_URL).strip()
    # Normalize URL: strip trailing /rest/v1 or trailing slashes
    if raw_url.endswith("/rest/v1/"):
        raw_url = raw_url[:-9]
    elif raw_url.endswith("/rest/v1"):
        raw_url = raw_url[:-8]
    raw_url = raw_url.rstrip("/")

    key = os.environ.get("SUPABASE_KEY", DEFAULT_SUPABASE_KEY).strip()
    return raw_url, key


def get_headers(prefer: Optional[str] = None) -> Dict[str, str]:
    """Build standard Supabase PostgREST authentication and control headers."""
    _, key = get_supabase_config()
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def get_rest_url(table: str = "") -> str:
    """Get full PostgREST endpoint for a table."""
    base_url, _ = get_supabase_config()
    if table:
        return f"{base_url}/rest/v1/{table.lstrip('/')}"
    return f"{base_url}/rest/v1"


async def test_connection() -> Dict[str, Any]:
    """
    Test connectivity to Supabase project and check whether public.patient_reports exists.
    """
    base_url, key = get_supabase_config()
    if not base_url or not key:
        return {
            "configured": False,
            "reachable": False,
            "table_ready": False,
            "message": "Supabase URL or Key is not configured."
        }

    # Extract project ref from URL if possible
    project_ref = base_url.replace("https://", "").replace(".supabase.co", "").split("/")[0]

    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            # Query patient_reports table with limit 1
            url = get_rest_url("patient_reports") + "?select=id&limit=1"
            res = await client.get(url, headers=get_headers())
            latency_ms = round((time.time() - t0) * 1000)

            if res.status_code == 200:
                return {
                    "configured": True,
                    "reachable": True,
                    "table_ready": True,
                    "project_ref": project_ref,
                    "url": base_url,
                    "latency_ms": latency_ms,
                    "message": "Supabase connected and patient_reports table is ready."
                }
            elif res.status_code == 404 and "PGRST205" in res.text:
                return {
                    "configured": True,
                    "reachable": True,
                    "table_ready": False,
                    "project_ref": project_ref,
                    "url": base_url,
                    "latency_ms": latency_ms,
                    "message": "Connected to Supabase, but 'patient_reports' table is not created yet. Please execute supabase_schema.sql in Supabase SQL Editor."
                }
            else:
                return {
                    "configured": True,
                    "reachable": True,
                    "table_ready": False,
                    "project_ref": project_ref,
                    "url": base_url,
                    "status_code": res.status_code,
                    "latency_ms": latency_ms,
                    "message": f"Supabase responded with code {res.status_code}: {res.text[:200]}"
                }
    except Exception as exc:
        return {
            "configured": True,
            "reachable": False,
            "table_ready": False,
            "project_ref": project_ref,
            "url": base_url,
            "error": str(exc),
            "message": f"Could not reach Supabase endpoint: {exc}"
        }


async def sync_patient_report(report_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Automatically upserts a patient diagnostic report to Supabase 'patient_reports' table.
    """
    base_url, key = get_supabase_config()
    if not base_url or not key:
        return {"success": False, "error": "Supabase credentials missing."}

    # Ensure full_payload is JSON object or dict
    payload_val = report_data.get("full_payload")
    if isinstance(payload_val, str):
        try:
            payload_val = json.loads(payload_val)
        except Exception:
            payload_val = {"raw": payload_val}
    elif not isinstance(payload_val, dict):
        payload_val = {}

    row = {
        "id": str(report_data.get("id")),
        "patient_ref": report_data.get("patient_ref"),
        "doctor_id": str(report_data.get("doctor_id") or ""),
        "doctor_slmc": str(report_data.get("doctor_slmc") or ""),
        "patient_name": report_data.get("patient_name") or None,
        "patient_age": str(report_data.get("patient_age")) if report_data.get("patient_age") is not None else None,
        "patient_gender": report_data.get("patient_gender") or None,
        "report_type": report_data.get("report_type") or "Full Blood Count",
        "report_date": report_data.get("report_date") or None,
        "clinical_notes": report_data.get("clinical_notes") or None,
        "primary_diagnosis": report_data.get("primary_diagnosis") or None,
        "confidence_pct": float(report_data["confidence_pct"]) if report_data.get("confidence_pct") is not None else None,
        "confidence_band": report_data.get("confidence_band") or None,
        "review_action": report_data.get("review_action") or None,
        "review_notes": report_data.get("review_notes") or None,
        "reviewed_at": report_data.get("reviewed_at") or None,
        "full_payload": payload_val,
        "processing_time_sec": float(report_data["processing_time_sec"]) if report_data.get("processing_time_sec") is not None else None,
        "created_at": report_data.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

    url = get_rest_url("patient_reports")
    # PostgREST merge duplicates (upsert on primary key)
    headers = get_headers(prefer="resolution=merge-duplicates,return=representation")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(url, headers=headers, json=row)
            if res.status_code in (200, 201):
                return {"success": True, "action": "synced", "report_id": row["id"]}
            elif res.status_code == 404 and "PGRST205" in res.text:
                print("[SUPABASE SYNC] Table 'patient_reports' does not exist yet in Supabase. Run supabase_schema.sql.")
                return {"success": False, "table_missing": True, "error": "Table 'patient_reports' not created yet."}
            else:
                print(f"[SUPABASE SYNC ERROR] HTTP {res.status_code}: {res.text}")
                return {"success": False, "status_code": res.status_code, "error": res.text}
    except Exception as exc:
        print(f"[SUPABASE SYNC EXCEPTION] {exc}")
        return {"success": False, "error": str(exc)}


async def sync_batch_patient_reports(reports_list: list[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Upserts multiple patient diagnostic reports to Supabase 'patient_reports' table in a single atomic HTTP request.
    """
    if not reports_list:
        return {"success": True, "synced": 0, "errors": []}

    base_url, key = get_supabase_config()
    if not base_url or not key:
        return {"success": False, "error": "Supabase credentials missing."}

    rows = []
    for report_data in reports_list:
        payload_val = report_data.get("full_payload")
        if isinstance(payload_val, str):
            try:
                payload_val = json.loads(payload_val)
            except Exception:
                payload_val = {"raw": payload_val}
        elif not isinstance(payload_val, dict):
            payload_val = {}

        rows.append({
            "id": str(report_data.get("id")),
            "patient_ref": report_data.get("patient_ref"),
            "doctor_id": str(report_data.get("doctor_id") or ""),
            "doctor_slmc": str(report_data.get("doctor_slmc") or ""),
            "patient_name": report_data.get("patient_name") or None,
            "patient_age": str(report_data.get("patient_age")) if report_data.get("patient_age") is not None else None,
            "patient_gender": report_data.get("patient_gender") or None,
            "report_type": report_data.get("report_type") or "Full Blood Count",
            "report_date": report_data.get("report_date") or None,
            "clinical_notes": report_data.get("clinical_notes") or None,
            "primary_diagnosis": report_data.get("primary_diagnosis") or None,
            "confidence_pct": float(report_data["confidence_pct"]) if report_data.get("confidence_pct") is not None else None,
            "confidence_band": report_data.get("confidence_band") or None,
            "review_action": report_data.get("review_action") or None,
            "review_notes": report_data.get("review_notes") or None,
            "reviewed_at": report_data.get("reviewed_at") or None,
            "full_payload": payload_val,
            "processing_time_sec": float(report_data["processing_time_sec"]) if report_data.get("processing_time_sec") is not None else None,
            "created_at": report_data.get("created_at") or datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        })

    url = get_rest_url("patient_reports")
    headers = get_headers(prefer="resolution=merge-duplicates,return=minimal")

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            res = await client.post(url, headers=headers, json=rows)
            if res.status_code in (200, 201, 204):
                return {"success": True, "synced": len(rows), "errors": []}
            elif res.status_code == 404 and "PGRST205" in res.text:
                return {"success": False, "synced": 0, "table_missing": True, "errors": ["Table patient_reports does not exist."]}
            else:
                return {"success": False, "synced": 0, "status_code": res.status_code, "errors": [res.text[:300]]}
    except Exception as exc:
        return {"success": False, "synced": 0, "errors": [str(exc)]}



async def sync_report_review(
    patient_ref: str,
    review_action: str,
    review_notes: Optional[str] = None,
    reviewed_at: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Automatically updates a report's doctor review/sign-off status in Supabase.
    """
    base_url, key = get_supabase_config()
    if not base_url or not key:
        return {"success": False, "error": "Supabase credentials missing."}

    review_time_iso = reviewed_at.isoformat() if reviewed_at else datetime.now(timezone.utc).isoformat()
    patch_body = {
        "review_action": review_action,
        "review_notes": (review_notes or "").strip() or None,
        "reviewed_at": review_time_iso,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

    url = f"{get_rest_url('patient_reports')}?patient_ref=eq.{patient_ref}"
    headers = get_headers(prefer="return=representation")

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            res = await client.patch(url, headers=headers, json=patch_body)
            if res.status_code in (200, 204):
                return {"success": True, "patient_ref": patient_ref, "review_action": review_action}
            else:
                return {"success": False, "status_code": res.status_code, "error": res.text}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def sync_doctor(doctor_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optionally upserts a doctor profile in Supabase 'doctors' table.
    """
    base_url, key = get_supabase_config()
    if not base_url or not key:
        return {"success": False, "error": "Supabase credentials missing."}

    row = {
        "id": str(doctor_dict.get("id")),
        "full_name": doctor_dict.get("full_name"),
        "slmc_number": doctor_dict.get("slmc_number"),
        "email": doctor_dict.get("email"),
        "email_verified": bool(doctor_dict.get("email_verified", False)),
        "phone": doctor_dict.get("phone"),
        "nic": doctor_dict.get("nic"),
        "role": doctor_dict.get("role", "DOCTOR"),
        "status": doctor_dict.get("status", "VERIFIED"),
        "created_at": doctor_dict.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "last_login_at": doctor_dict.get("last_login_at")
    }

    url = get_rest_url("doctors")
    headers = get_headers(prefer="resolution=merge-duplicates,return=representation")

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            res = await client.post(url, headers=headers, json=row)
            return {"success": res.status_code in (200, 201), "status_code": res.status_code}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def fetch_doctor_patient_reports(doctor_id: str) -> List[Dict[str, Any]]:
    """
    Fetches patient diagnostic records from Supabase strictly filtered for the specified doctor_id.
    Ensures complete per-doctor data isolation.
    """
    base_url, key = get_supabase_config()
    if not base_url or not key or not doctor_id:
        return []

    url = f"{get_rest_url('patient_reports')}?doctor_id=eq.{doctor_id}&order=created_at.desc"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(url, headers=get_headers())
            if res.status_code == 200:
                return res.json()
            return []
    except Exception as exc:
        print(f"[SUPABASE DOCTOR FETCH ERROR] {exc}")
        return []

