"""real_data_loader.py — Real-world de-identified clinical lab dataset loader.

Supports ingesting open public clinical lab cohorts (such as CDC NHANES, MIMIC-IV lab extracts,
or de-identified local hospital lab CSV exports) and mapping them into the CliniQ ML
multi-label format (RAW_FEATURES + LABELS).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from .schema import LABELS, RAW_FEATURES

# Standard column mapping dictionary for open datasets (NHANES, MIMIC-IV, hospital exports)
STANDARD_COLUMN_MAP = {
    # Demographics
    "age": "age",
    "ridageyr": "age",
    "patient_age": "age",
    "gender": "sex",
    "sex": "sex",
    "riagendr": "sex",      # NHANES 1=Male, 2=Female
    
    # Hematology / FBC
    "hb": "hb",
    "hgb": "hb",
    "hemoglobin": "hb",
    "lbxhgb": "hb",
    "wbc": "wbc",
    "lbxwbcsi": "wbc",
    "white_blood_cells": "wbc",
    "plt": "platelets",
    "platelets": "platelets",
    "lbxpltsi": "platelets",
    "rbc": "rbc",
    "lbxrbcsi": "rbc",
    "hct": "hct",
    "lbxhct": "hct",
    "hematocrit": "hct",
    "mcv": "mcv",
    "lbxmcvsi": "mcv",
    "mch": "mch",
    "lbxmch": "mch",
    "mchc": "mchc",
    "lbxmchc": "mchc",
    "neut_pct": "neut_pct",
    "neutrophils": "neut_pct",
    "lymph_pct": "lymph_pct",
    "lymphocytes": "lymph_pct",

    # Renal / Metabolic
    "fbs": "fbs",
    "fasting_blood_sugar": "fbs",
    "glucose": "fbs",
    "lbxglu": "fbs",
    "hba1c": "hba1c",
    "lbxgh": "hba1c",
    "glycated_hemoglobin": "hba1c",
    "creatinine": "creatinine",
    "creat": "creatinine",
    "lbxscr": "creatinine",
    "urea": "urea",
    "bun": "urea",
    "lbxsn3": "urea",

    # Lipids
    "total_chol": "total_chol",
    "cholesterol": "total_chol",
    "lbxtc": "total_chol",
    "hdl": "hdl",
    "lbdhdd": "hdl",
    "ldl": "ldl",
    "lbdldl": "ldl",
    "tg": "tg",
    "triglycerides": "tg",
    "lbxtr": "tg",

    # Liver & Thyroid
    "alt": "alt",
    "sgpt": "alt",
    "lbxsat": "alt",
    "ast": "ast",
    "sgot": "ast",
    "lbxsat": "ast",
    "tsh": "tsh",
    "lbxtsh": "tsh",
    "ft4": "ft4",
}


def derive_clinical_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Derives clinical ground-truth labels using validated ADA, WHO, and KDIGO guidelines

    when raw reports only contain analyte values without categorical labels.
    """
    labels = pd.DataFrame(index=df.index)

    # Anemia: WHO criteria (Hb < 13 g/dL for males, < 12 g/dL for females)
    male = df.get("sex", pd.Series(1, index=df.index)) == 1
    hb = df.get("hb", pd.Series(np.nan, index=df.index))
    labels["anemia"] = np.where(
        hb.notna(),
        np.where(male, (hb < 13.0).astype(int), (hb < 12.0).astype(int)),
        0
    )

    # CKD: Creatinine > 1.4 mg/dL or eGFR < 60
    creat = df.get("creatinine", pd.Series(np.nan, index=df.index))
    labels["ckd"] = np.where(creat.notna(), (creat >= 1.4).astype(int), 0)

    # Diabetes: ADA criteria (FBS >= 126 mg/dL or HbA1c >= 6.5%)
    fbs = df.get("fbs", pd.Series(np.nan, index=df.index))
    hba1c = df.get("hba1c", pd.Series(np.nan, index=df.index))
    labels["diabetes"] = np.where(
        (fbs.notna() & (fbs >= 126.0)) | (hba1c.notna() & (hba1c >= 6.5)),
        1,
        0
    )

    # Hypercholesterolemia: Total Cholesterol >= 200 mg/dL or LDL >= 130 mg/dL
    tc = df.get("total_chol", pd.Series(np.nan, index=df.index))
    ldl = df.get("ldl", pd.Series(np.nan, index=df.index))
    labels["hypercholesterolemia"] = np.where(
        (tc.notna() & (tc >= 200.0)) | (ldl.notna() & (ldl >= 130.0)),
        1,
        0
    )

    # Hypothyroidism: TSH > 4.5 mIU/L
    tsh = df.get("tsh", pd.Series(np.nan, index=df.index))
    labels["hypothyroidism"] = np.where(tsh.notna() & (tsh > 4.5), 1, 0)

    # Hyperthyroidism: TSH < 0.4 mIU/L
    labels["hyperthyroidism"] = np.where(tsh.notna() & (tsh < 0.4), 1, 0)

    # Infection: WBC > 11.0 ×10³/µL or Neutrophils > 75%
    wbc = df.get("wbc", pd.Series(np.nan, index=df.index))
    neut = df.get("neut_pct", pd.Series(np.nan, index=df.index))
    labels["infection"] = np.where(
        (wbc.notna() & (wbc > 11.0)) | (neut.notna() & (neut > 75.0)),
        1,
        0
    )

    return labels[LABELS]


def load_real_clinical_dataset(file_path: str | Path, assign_site_id: int = 6) -> pd.DataFrame:
    """Loads a real-world CSV/Parquet file, standardizes columns to RAW_FEATURES and LABELS."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found at: {p}")

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    # Lowercase all input column names
    col_map = {col: str(col).strip().lower() for col in df.columns}
    df = df.rename(columns=col_map)

    # Standardize column names using mapping
    rename_dict = {}
    for orig_col in df.columns:
        if orig_col in STANDARD_COLUMN_MAP:
            target = STANDARD_COLUMN_MAP[orig_col]
            if target not in rename_dict.values():
                rename_dict[orig_col] = target

    df = df.rename(columns=rename_dict)

    # Standardize sex/gender: Male=1, Female=0
    if "sex" in df.columns:
        if df["sex"].dtype == object:
            df["sex"] = df["sex"].astype(str).str.strip().str.lower().map({
                "m": 1, "male": 1, "1": 1, "1.0": 1,
                "f": 0, "female": 0, "2": 0, "2.0": 0, "0": 0
            }).fillna(1).astype(int)
        elif set(df["sex"].dropna().unique()).issubset({1, 2}):
            # NHANES coding: 1 = Male, 2 = Female
            df["sex"] = (df["sex"] == 1).astype(int)

    # Ensure all RAW_FEATURES exist
    for f in RAW_FEATURES:
        if f not in df.columns:
            df[f] = np.nan

    # Ensure labels exist; if not present, derive them clinically
    for lab in LABELS:
        if lab not in df.columns:
            derived = derive_clinical_labels(df)
            for l_col in LABELS:
                if l_col not in df.columns:
                    df[l_col] = derived[l_col]
            break

    df["site"] = assign_site_id
    keep_cols = RAW_FEATURES + LABELS + ["site"]
    return df[keep_cols].copy()


def augment_with_real_data(synthetic_df: pd.DataFrame, real_df: pd.DataFrame, real_weight: float = 0.5) -> pd.DataFrame:
    """Blends real de-identified lab data with synthetic data to bridge the sim-to-real domain gap."""
    if real_df.empty:
        return synthetic_df
    
    # Match site numbers appropriately
    combined = pd.concat([synthetic_df, real_df], ignore_index=True)
    return combined
