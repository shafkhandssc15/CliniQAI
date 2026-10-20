"""
export_recalibration_data.py — Admin script for site-level model recalibration.

Extracts physician sign-off labels from `condition_feedback` and corresponding raw analytes
from `patient_reports`. Checks if criteria (400+ cases, >=30 positives per condition) are met.
Optionally triggers `CliniQEngine.recalibrate_site()` to output a recalibrated model bundle.

Usage:
    python export_recalibration_data.py --status
    python export_recalibration_data.py --export-csv site_data.csv
    python export_recalibration_data.py --recalibrate --out-dir artifacts_siteA
"""
import sys
import os
import json
import argparse
import asyncio
import pandas as pd
import numpy as np

from database import AsyncSessionLocal
from models_db import PatientReport, ConditionFeedback
from sqlalchemy import select
from cliniq_ml.schema import RAW_FEATURES, LABELS
from cliniq_ml.engine import CliniQEngine


async def fetch_recalibration_dataset():
    """Queries SQLite/Postgres for reports with physician sign-off feedback."""
    async with AsyncSessionLocal() as session:
        # Fetch all feedback records grouped by report_id
        fb_res = await session.execute(select(ConditionFeedback).order_by(ConditionFeedback.created_at))
        feedbacks = fb_res.scalars().all()

        if not feedbacks:
            return pd.DataFrame(), pd.DataFrame()

        # Group feedback by report_id
        report_labels = {}
        for fb in feedbacks:
            if fb.report_id not in report_labels:
                report_labels[fb.report_id] = {}
            if fb.final_label is not None:
                report_labels[fb.report_id][fb.condition_name] = int(fb.final_label)

        # Fetch corresponding reports
        rep_ids = list(report_labels.keys())
        rep_res = await session.execute(select(PatientReport).where(PatientReport.id.in_(rep_ids)))
        reports = {r.id: r for r in rep_res.scalars().all()}

        raw_rows = []
        label_rows = []

        for rep_id, labels in report_labels.items():
            rep = reports.get(rep_id)
            if not rep:
                continue

            try:
                payload = json.loads(rep.full_payload)
            except Exception:
                payload = {}

            parsed = payload.get("parsed_values", {})
            age = float(rep.patient_age) if rep.patient_age and str(rep.patient_age).strip().isdigit() else np.nan
            sex = 1.0 if (rep.patient_gender or "").lower().startswith("m") else 0.0

            # Raw feature row
            row_dict = {"age": age, "sex": sex}
            for col in RAW_FEATURES:
                if col not in ("age", "sex"):
                    row_dict[col] = parsed.get(col, parsed.get(col.replace("_", " "), parsed.get(col.title())))
            raw_rows.append(row_dict)

            # Label row (fill missing condition labels with 0)
            lbl_dict = {col: labels.get(col, 0) for col in LABELS}
            label_rows.append(lbl_dict)

        raw_df = pd.DataFrame(raw_rows)
        labels_df = pd.DataFrame(label_rows)
        return raw_df, labels_df


def audit_readiness(raw_df: pd.DataFrame, labels_df: pd.DataFrame):
    total_cases = len(raw_df)
    print("=" * 65)
    print("CLINIQ AI — SITE RECALIBRATION READINESS AUDIT")
    print("=" * 65)
    print(f"Total Signed-off Cases Available: {total_cases} (Target: >= 400 cases)")
    print("-" * 65)
    print(f"{'Condition':<26} | {'Positives':<10} | {'Status'}")
    print("-" * 65)

    all_ready = total_cases >= 400
    for lab in LABELS:
        pos_count = int(labels_df[lab].sum()) if lab in labels_df else 0
        is_ready = pos_count >= 30
        status_str = "READY (>=30)" if is_ready else f"NEED {30 - pos_count} MORE"
        if not is_ready:
            all_ready = False
        print(f"{lab:<26} | {pos_count:<10} | {status_str}")

    print("-" * 65)
    overall_status = "READY FOR SITE RECALIBRATION" if all_ready else "INSUFFICIENT DATA — ACCUMULATING SIGN-OFFS"
    print(f"Overall Decision: {overall_status}")
    print("=" * 65)
    return all_ready


def main():
    parser = argparse.ArgumentParser(description="Export site labels and trigger model recalibration.")
    parser.add_argument("--status", action="store_true", help="Audit current sample and positive counts.")
    parser.add_argument("--export-csv", type=str, help="Export dataset to specified CSV file prefix.")
    parser.add_argument("--recalibrate", action="store_true", help="Run site recalibration if thresholds are met.")
    parser.add_argument("--out-dir", type=str, default="artifacts_recalibrated", help="Output directory for recalibrated model.")
    parser.add_argument("--min-cases", type=int, default=400, help="Minimum assessable cases per label (default: 400).")
    parser.add_argument("--min-pos", type=int, default=30, help="Minimum positive cases per label (default: 30).")
    parser.add_argument("--alpha-pos", type=float, default=0.05, help="Conformal miss-rate budget for positives (default: 0.05 = 95% sensitivity).")
    parser.add_argument("--alpha-neg", type=float, default=0.10, help="Conformal miss-rate budget for negatives (default: 0.10 = 90% specificity).")
    parser.add_argument("--refit-calibrator", action="store_true", help="Also refit Platt probability calibrator.")
    parser.add_argument("--force", action="store_true", help="Force recalibration even if sample threshold is unmet.")

    args = parser.parse_args()

    raw_df, labels_df = asyncio.run(fetch_recalibration_dataset())

    if raw_df.empty:
        print("[INFO] No signed-off physician feedback records found in database yet.")
        print("Doctors must review and sign off on reports via the workstation to accumulate site labels.")
        return

    is_ready = audit_readiness(raw_df, labels_df)

    if args.export_csv:
        raw_path = f"{args.export_csv}_raw.csv"
        lbl_path = f"{args.export_csv}_labels.csv"
        raw_df.to_csv(raw_path, index=False)
        labels_df.to_csv(lbl_path, index=False)
        print(f"[EXPORT] Saved {len(raw_df)} cases to {raw_path} and {lbl_path}")

    if args.recalibrate:
        if not is_ready and not args.force:
            print("[ABORT] Cannot recalibrate: criteria not met (need >=400 cases and >=30 positives/condition).")
            print("To override for testing/pilot cohorts, supply --force and optionally --min-cases/--min-pos.")
            return

        min_c = args.min_cases if not args.force else min(args.min_cases, len(raw_df))
        min_p = args.min_pos if not args.force else min(args.min_pos, 1)

        print(f"\n[RECALIBRATING] Running site recalibration -> {args.out_dir} (min_cases={min_c}, min_pos={min_p})...")
        engine = CliniQEngine("artifacts")
        rep = engine.recalibrate_site(
            raw_df, labels_df, args.out_dir,
            refit_calibration=args.refit_calibrator,
            min_cases=min_c,
            min_pos=min_p,
            alpha_pos=args.alpha_pos,
            alpha_neg=args.alpha_neg
        )
        print(f"[SUCCESS] Recalibration complete. Output written to {args.out_dir}/")
        print(f"Per-label recalibration status: {json.dumps(rep, indent=2)}")
        print("Note: The platform continues using 'artifacts/' by default. An admin can swap the active directory when ready.")


if __name__ == "__main__":
    main()
