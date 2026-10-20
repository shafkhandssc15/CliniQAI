"""Feature engineering, plausibility gating and OCR-consistency checks (all vectorised over DataFrames)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import BOUNDS, LAB_ANALYTES, MODEL_FEATURES, RAW_FEATURES


def egfr_ckdepi_2021(creat: pd.Series, age: pd.Series, male: pd.Series) -> pd.Series:
    """CKD-EPI 2021 (race-free). creat in mg/dL."""
    kappa = np.where(male == 1, 0.9, 0.7)
    alpha = np.where(male == 1, -0.302, -0.241)
    ratio = creat / kappa
    out = (142.0 * np.minimum(ratio, 1.0) ** alpha * np.maximum(ratio, 1.0) ** -1.200
           * 0.9938 ** age * np.where(male == 1, 1.0, 1.012))
    return pd.Series(out, index=creat.index).where(creat.notna())


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw analytes -> model matrix. NaN is preserved (XGBoost handles it natively)."""
    X = df.reindex(columns=RAW_FEATURES).astype(float).copy()
    X["egfr"] = egfr_ckdepi_2021(X["creatinine"], X["age"], X["sex"])
    X["non_hdl"] = X["total_chol"] - X["hdl"]
    X["tc_hdl"] = X["total_chol"] / X["hdl"]
    X["tg_hdl"] = X["tg"] / X["hdl"]
    X["anc"] = X["wbc"] * X["neut_pct"] / 100.0
    X["nlr"] = X["neut_pct"] / X["lymph_pct"].clip(lower=1.0)
    X["mentzer"] = X["mcv"] / X["rbc"]
    X["hb_gap"] = X["hb"] - np.where(X["sex"] == 1, 13.0, 12.0)
    X["bun_cr"] = X["urea"] / X["creatinine"]
    X["alt_ast"] = X["alt"] / X["ast"]
    X["log_tsh"] = np.log10(X["tsh"])
    return X[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)


def apply_plausibility(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blank out values outside hard physiological bounds. Returns (clean_df, removed_mask)."""
    out = df.copy()
    removed = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col, (lo, hi) in BOUNDS.items():
        if col in out.columns:
            bad = out[col].notna() & ((out[col] < lo) | (out[col] > hi))
            removed[col] = bad
            out.loc[bad, col] = np.nan
    return out, removed


def consistency_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-field identities that a mis-read OCR value will usually violate.

    Returns a boolean frame with one column per *suspect field*; True means 'verify this value'.
    Each identity implicates the fields that could have caused the violation.
    """
    f = pd.DataFrame(False, index=df.index, columns=LAB_ANALYTES)

    def mark(mask, *cols):
        for c in cols:
            f.loc[mask.fillna(False), c] = True

    # Haematocrit ~ 3 x haemoglobin (+/- 15%)
    hb, hct = df["hb"], df["hct"]
    mark(((hct - 3.0 * hb).abs() > 0.15 * 3.0 * hb), "hb", "hct")
    # MCHC = Hb / Hct x 100 (reported vs recomputed, 12% tolerance)
    mchc_calc = hb / hct * 100.0
    mark(((df["mchc"] - mchc_calc).abs() > 0.12 * mchc_calc), "hb", "hct", "mchc")
    # MCV = Hct / RBC x 10
    mcv_calc = hct / df["rbc"] * 10.0
    mark(((df["mcv"] - mcv_calc).abs() > 0.15 * mcv_calc), "hct", "rbc", "mcv")
    # MCH = Hb / RBC x 10
    mch_calc = hb / df["rbc"] * 10.0
    mark(((df["mch"] - mch_calc).abs() > 0.15 * mch_calc), "hb", "rbc", "mch")
    # Differential can't exceed 100%
    mark((df["neut_pct"] + df["lymph_pct"]) > 102.0, "neut_pct", "lymph_pct")
    # Friedewald: LDL ~= TC - HDL - TG/5 (valid while TG < 400)
    ldl_calc = df["total_chol"] - df["hdl"] - df["tg"] / 5.0
    mark(((df["ldl"] - ldl_calc).abs() > 30.0) & (df["tg"] < 400), "total_chol", "hdl", "tg", "ldl")
    # HbA1c vs fasting glucose: grossly discordant pairs
    est_avg = 28.7 * df["hba1c"] - 46.7
    mark(((df["fbs"] - est_avg).abs() > 120.0), "fbs", "hba1c")
    # Total cholesterol must exceed HDL
    mark(df["hdl"] >= df["total_chol"], "hdl", "total_chol")
    return f
