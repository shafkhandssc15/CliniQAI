import numpy as np
import pandas as pd

from cliniq_ml.engine import normalize_input
from cliniq_ml.features import apply_plausibility, build_features, consistency_flags, egfr_ckdepi_2021
from cliniq_ml.schema import RAW_FEATURES


def row(**kw):
    return pd.DataFrame([{c: kw.get(c, np.nan) for c in RAW_FEATURES}])


def test_egfr_matches_ckdepi_2021_reference():
    e = egfr_ckdepi_2021(pd.Series([1.0]), pd.Series([60.0]), pd.Series([1]))
    assert 84 < e.iloc[0] < 88
    f = egfr_ckdepi_2021(pd.Series([1.0]), pd.Series([60.0]), pd.Series([0]))
    assert 62 < f.iloc[0] < 68


def test_plausibility_blanks_implausible_values_only():
    clean, removed = apply_plausibility(row(age=40, sex=1, hb=135.0, wbc=7.0))
    assert np.isnan(clean.at[0, "hb"]) and removed.at[0, "hb"]
    assert clean.at[0, "wbc"] == 7.0


def test_consistency_flags_catch_hb_hct_mismatch():
    f = consistency_flags(row(age=40, sex=1, hb=9.4, hct=47.0))
    assert f.at[0, "hb"] and f.at[0, "hct"]


def test_consistency_flags_quiet_on_consistent_cbc():
    f = consistency_flags(row(age=40, sex=1, hb=14.0, hct=42.0, rbc=4.8, mcv=87.5, mch=29.2, mchc=33.3))
    assert not f.any(axis=None)


def test_unit_normalisation():
    v, notes = normalize_input({"Fasting Blood Sugar": 5.5, "Serum Creatinine": 106, "Haemoglobin": 135}, age=50, sex="M")
    assert abs(v["fbs"] - 99.1) < 0.5 and abs(v["creatinine"] - 1.2) < 0.01 and abs(v["hb"] - 13.5) < 1e-9
    assert len(notes) == 3


def test_features_keep_nan_not_zero():
    X = build_features(row(age=40, sex=1, hb=14.0))
    assert np.isnan(X.at[0, "egfr"]) and np.isnan(X.at[0, "tc_hdl"])
