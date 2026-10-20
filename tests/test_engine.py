import json

import numpy as np
import pandas as pd

from cliniq_ml.engine import CliniQEngine
from cliniq_ml.narrative import generate_narrative
from cliniq_ml.schema import LABELS, RAW_FEATURES
from cliniq_ml.verify import cross_check_extractions


def test_artifacts_are_pickle_free(trained):
    _, _, d = trained
    names = {p.suffix for p in d.iterdir()}
    assert names == {".json"}


def test_pediatric_is_out_of_scope(trained):
    eng, *_ = trained
    assert eng.analyze({"Hb": 11}, age=9, sex="F")["status"] == "OUT_OF_SCOPE"


def test_missing_age_or_sex_is_refused_not_guessed(trained):
    eng, *_ = trained
    assert eng.analyze({"Hb": 11}, age=None, sex="F")["status"] == "INSUFFICIENT_INPUT"


def test_absent_panel_is_not_assessable_never_negative(trained):
    eng, *_ = trained
    r = eng.analyze({"Hb": 14.0, "WBC": 7.0, "Platelets": 250}, age=35, sex="M")
    assert r["conditions"]["hypothyroidism"]["decision"] == "NOT_ASSESSABLE"
    assert r["conditions"]["diabetes"]["decision"] == "NOT_ASSESSABLE"
    assert r["status"] != "NO_SIGNIFICANT_FINDINGS"


def test_critical_value_always_escalates(trained):
    eng, *_ = trained
    r = eng.analyze({"Hb": 5.2, "WBC": 7.0, "Platelets": 250}, age=35, sex="F")
    assert r["status"] == "CRITICAL_ESCALATION"
    assert any(a["analyte"] == "haemoglobin" for a in r["critical_alerts"])


def test_implausible_value_is_excluded_and_warned(trained):
    eng, *_ = trained
    r = eng.analyze({"Hb": 1350.0, "WBC": 7.0}, age=40, sex="M")  # not rescued by g/L conversion (>30 -> /10 = 135, still implausible)
    assert any("implausible" in w for w in r["warnings"])
    assert r["conditions"]["anemia"]["decision"] == "NOT_ASSESSABLE"


def test_clinical_direction_is_sensible(trained):
    eng, *_ = trained
    lo = eng.analyze({"Hb": 8.5, "MCV": 70}, age=40, sex="F")["conditions"]["anemia"]["probability"]
    hi = eng.analyze({"Hb": 14.0, "MCV": 89}, age=40, sex="F")["conditions"]["anemia"]["probability"]
    assert lo > 0.8 > 0.2 > hi
    c_hi = eng.analyze({"Creatinine": 3.2, "Urea": 90}, age=60, sex="M")["conditions"]["ckd"]["probability"]
    c_lo = eng.analyze({"Creatinine": 0.9, "Urea": 25}, age=60, sex="M")["conditions"]["ckd"]["probability"]
    assert c_hi > 0.8 > 0.2 > c_lo


def test_deterministic(trained):
    eng, *_ = trained
    a = eng.analyze({"Hb": 11.0, "FBS": 140, "Creatinine": 1.4}, age=55, sex="M")
    b = eng.analyze({"Hb": 11.0, "FBS": 140, "Creatinine": 1.4}, age=55, sex="M")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_conformal_coverage_on_heldout(trained):
    """Gross-failure check only: class-conditional coverage near target on held-out data. This fixture is small
    (about 100 positives per label in the conformal set), so the tolerance is wide; train.py reports full-scale numbers."""
    eng, res, _ = trained
    s = res["splits"]["test"]
    pr = eng.predict_frame(s[RAW_FEATURES])
    for lab in ("anemia", "diabetes", "infection"):
        a = pr["decision"][lab] != "NOT_ASSESSABLE"
        pos = (s[lab] == 1) & a
        neg = (s[lab] == 0) & a
        assert (pr["decision"][lab][pos] != "NEGATIVE").mean() >= 0.95 - 0.08
        assert (pr["decision"][lab][neg] != "POSITIVE").mean() >= 0.90 - 0.08


def test_calibration_is_reasonable(trained):
    eng, res, _ = trained
    s = res["splits"]["test"]
    p = eng.predict_frame(s[RAW_FEATURES])["proba"]
    for lab in ("anemia", "diabetes", "ckd"):
        assert abs(p[lab].mean() - s[lab].mean()) < 0.03


def test_site_recalibration_writes_loadable_model(trained, tmp_path):
    eng, res, _ = trained
    ext = res["splits"]["external"]
    rep = eng.recalibrate_site(ext[RAW_FEATURES], ext[LABELS], tmp_path / "site", min_cases=300, min_pos=20)
    assert rep["anemia"]["updated"]
    assert CliniQEngine(tmp_path / "site").meta["site_recalibrated"] is True


def test_cross_check_flags_decimal_slip():
    out = cross_check_extractions({"Hb": 13.5, "Creatinine": 0.9}, {"Hb": 135, "Creatinine": 9.0})
    # Hb 135 is a legitimate g/L reading (converted), creatinine 9.0 vs 0.9 is a real decimal slip
    assert len(out) == 1 and out[0]["field"] == "creatinine" and "decimal" in out[0]["issue"]


def test_narrative_guard_rejects_invented_diagnosis(trained):
    eng, *_ = trained
    r = eng.analyze({"Hb": 8.5, "MCV": 70}, age=40, sex="F")
    assert "Anemia" in r["findings"]
    bad = generate_narrative(r, lambda prompt: "Patient clearly has diabetes. Physician review advised.")
    assert bad["ok"] is False and "Decision support only" in bad["text"]
    good = generate_narrative(r, lambda prompt: "Anemia is flagged by the model. Physician review required.")
    assert good["ok"] is True
