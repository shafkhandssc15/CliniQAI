"""CliniQ multi-label engine: training, calibration, conformal abstention, explanation, inference.

Artifacts are stored as plain JSON + XGBoost JSON (no pickle), so loading a model file cannot execute code.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .features import apply_plausibility, build_features, consistency_flags
from .schema import (ALIASES, BOUNDS, CRITICAL_LIMITS, FEATURE_TEXT, LABEL_DISPLAY, LABELS, MODEL_FEATURES,
                     NEXT_TEST_FOR, RAW_FEATURES, REQUIRED_INPUTS)

MODEL_VERSION = "cliniq-ml-2.2.0"

# Conformal miss-rate budgets: strict for positives (protects sensitivity), looser for negatives.
ALPHA_POS = 0.05
ALPHA_NEG = 0.10

XGB_PARAMS = dict(
    n_estimators=600, learning_rate=0.04, max_depth=4, min_child_weight=3, subsample=0.85,
    colsample_bytree=0.8, reg_lambda=2.0, gamma=0.1, tree_method="hist", n_jobs=2,
    eval_metric="logloss", early_stopping_rounds=40, random_state=11,
)


# ----------------------------------------------------------------------------------------------
# Calibration (JSON-serialisable)
# ----------------------------------------------------------------------------------------------
def _fit_calibrator(p: np.ndarray, y: np.ndarray, force_platt: bool = False) -> dict:
    """Isotonic when there are enough positives, Platt (logistic on logit) otherwise or when forced.

    Site-level refits are small, where isotonic plateaus make the conformal scores tie and unstable, so they use Platt."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    if y.sum() >= 150 and not force_platt:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return {"type": "isotonic", "x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return {"type": "platt", "a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}


def _apply_calibrator(cal: dict, p: np.ndarray) -> np.ndarray:
    if cal["type"] == "isotonic":
        return np.clip(np.interp(p, cal["x"], cal["y"]), 0.0, 1.0)
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1.0 / (1.0 + np.exp(-(cal["a"] * z + cal["b"])))


def _conformal_q(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = min(n, math.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[k - 1])


def conformal_decision(p: np.ndarray, q_pos: float, q_neg: float) -> np.ndarray:
    """Mondrian (class-conditional) split conformal. Returns 'POSITIVE'|'NEGATIVE'|'UNCERTAIN'.

    Label 1 is admitted if (1-p) <= q_pos; label 0 is admitted if p <= q_neg. Under exchangeability the true
    label is inside the set with prob >= 1-alpha *for each class separately*.
    """
    has1 = (1 - p) <= q_pos
    has0 = p <= q_neg
    out = np.full(p.shape, "UNCERTAIN", dtype=object)
    out[has1 & ~has0] = "POSITIVE"
    out[has0 & ~has1] = "NEGATIVE"
    return out


def assessable_mask(clean_raw: pd.DataFrame, lab: str) -> np.ndarray:
    """True where at least one input required to judge `lab` is present (after plausibility gating)."""
    return clean_raw[REQUIRED_INPUTS[lab]].notna().any(axis=1).values


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------
def train(data: pd.DataFrame, out_dir: str | Path, seed: int = 11, verbose: bool = True) -> dict:
    """Train on the internal sites; returns a dict of splits for evaluation."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    internal = data[data["site"] != data["site"].max()].reset_index(drop=True)
    external = data[data["site"] == data["site"].max()].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(internal))
    n = len(idx)
    cuts = np.cumsum([int(n * f) for f in (0.55, 0.10, 0.10, 0.10)])
    parts = np.split(idx, cuts)  # train, val, calib, conformal, test
    names = ["train", "val", "calib", "conf", "test"]
    splits = {k: internal.iloc[v].reset_index(drop=True) for k, v in zip(names, parts)}
    splits["external"] = external

    clean = {k: apply_plausibility(v[RAW_FEATURES])[0] for k, v in splits.items()}
    X = {k: build_features(v) for k, v in clean.items()}

    meta = {"version": MODEL_VERSION, "features": MODEL_FEATURES, "labels": LABELS,
            "alpha_pos": ALPHA_POS, "alpha_neg": ALPHA_NEG, "labels_info": {}}
    for lab in LABELS:
        y = {k: v[lab].values for k, v in splits.items()}
        pos = y["train"].mean()
        params = dict(XGB_PARAMS, random_state=seed, scale_pos_weight=float(np.sqrt((1 - pos) / pos)))
        clf = xgb.XGBClassifier(**params)
        clf.fit(X["train"], y["train"], eval_set=[(X["val"], y["val"])], verbose=False)
        clf.get_booster().save_model(str(out_dir / f"xgb_{lab}.json"))

        raw_cal = clf.predict_proba(X["calib"])[:, 1]
        cal = _fit_calibrator(raw_cal, y["calib"])
        p_conf = _apply_calibrator(cal, clf.predict_proba(X["conf"])[:, 1])
        # Decisions are only ever issued for assessable cases, so the conformal guarantee is calibrated on them.
        a_conf = assessable_mask(clean["conf"], lab)
        s_pos = 1 - p_conf[(y["conf"] == 1) & a_conf]
        s_neg = p_conf[(y["conf"] == 0) & a_conf]
        q_pos, q_neg = _conformal_q(s_pos, ALPHA_POS), _conformal_q(s_neg, ALPHA_NEG)

        # 95%-sensitivity operating point (on assessable calibration data) used as a safety override.
        p_cal = _apply_calibrator(cal, raw_cal)
        a_cal = assessable_mask(clean["calib"], lab)
        t95 = float(np.quantile(p_cal[(y["calib"] == 1) & a_cal], 0.05))
        meta["labels_info"][lab] = dict(calibrator=cal, q_pos=q_pos, q_neg=q_neg, t_sens95=t95,
                                        best_iteration=int(clf.best_iteration), prevalence_train=float(pos))
        if verbose:
            print(f"  {lab:22s} trees={clf.best_iteration:3d} q_pos={q_pos:.3f} q_neg={q_neg:.3f} t95={t95:.3f}")

    # Out-of-distribution envelope: robust per-feature training quantiles.
    tr = clean["train"]
    meta["ood_bounds"] = {c: [float(tr[c].quantile(0.0005)), float(tr[c].quantile(0.9995))]
                          for c in RAW_FEATURES if c not in ("sex",)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return {"splits": splits, "X": X, "meta": meta}


# ----------------------------------------------------------------------------------------------
# Input normalisation (unit detection, aliases)
# ----------------------------------------------------------------------------------------------
def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9% ]+", " ", str(k).lower()).strip()


_ALIAS_LOOKUP = {_norm_key(a): canon for canon, al in ALIASES.items() for a in al}


def normalize_input(raw: dict, age=None, sex=None) -> tuple[dict, list[str]]:
    """Map free-form analyte names to canonical keys and fix obvious unit mix-ups. Returns (values, notes)."""
    notes: list[str] = []
    vals: dict[str, float] = {}
    for k, v in raw.items():
        canon = _ALIAS_LOOKUP.get(_norm_key(k), k if k in RAW_FEATURES else None)
        if canon is None or v is None:
            continue
        if canon == "sex":
            s = str(v).strip().lower()
            vals["sex"] = 1.0 if s in ("m", "male", "1", "පුරුෂ", "ஆண்") else 0.0 if s in ("f", "female", "0", "ස්ත්‍රී", "பெண்") else np.nan
            continue
        try:
            vals[canon] = float(v)
        except (TypeError, ValueError):
            notes.append(f"{k}: could not parse '{v}' as a number - ignored")
    if age is not None:
        vals["age"] = float(age)
    if sex is not None and "sex" not in vals:
        vals["sex"] = 1.0 if str(sex).strip().lower() in ("m", "male", "1") else 0.0

    def conv(key, cond, f, why):
        if key in vals and pd.notna(vals[key]) and cond(vals[key]):
            old = vals[key]
            vals[key] = f(old)
            notes.append(f"{FEATURE_TEXT.get(key, key)}: {old:g} looked like {why}; converted to {vals[key]:.4g}")

    conv("fbs", lambda x: x < 35, lambda x: x * 18.016, "mmol/L")
    conv("total_chol", lambda x: x < 15, lambda x: x * 38.67, "mmol/L")
    conv("ldl", lambda x: x < 12, lambda x: x * 38.67, "mmol/L")
    conv("hdl", lambda x: x < 4, lambda x: x * 38.67, "mmol/L")
    conv("tg", lambda x: x < 12, lambda x: x * 88.57, "mmol/L")
    conv("creatinine", lambda x: x > 20, lambda x: x / 88.4, "umol/L")
    conv("urea", lambda x: x < 3.0, lambda x: x * 6.006, "mmol/L")
    conv("hb", lambda x: x > 30, lambda x: x / 10.0, "g/L")
    conv("platelets", lambda x: x > 2000, lambda x: x / 1000.0, "per uL")
    conv("wbc", lambda x: x > 200, lambda x: x / 1000.0, "per uL")
    conv("hba1c", lambda x: x > 20, lambda x: x * 0.09148 + 2.152, "mmol/mol (IFCC)")
    return vals, notes


# ----------------------------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------------------------
class CliniQEngine:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.model_dir = model_dir
        self.meta = json.loads((model_dir / "meta.json").read_text())
        self.boosters = {}
        for lab in LABELS:
            b = xgb.Booster()
            b.load_model(str(model_dir / f"xgb_{lab}.json"))
            self.boosters[lab] = b
        self.version = self.meta["version"]

    # -- batch API (used by evaluation) -------------------------------------------------------
    def predict_frame(self, raw: pd.DataFrame, gate: bool = True, assess_guard: bool = True) -> dict[str, pd.DataFrame]:
        """Returns {'proba': df, 'decision': df} for a frame of raw analytes.

        gate         - blank physiologically implausible values first (recommended)
        assess_guard - mark a condition NOT_ASSESSABLE when none of its required inputs is present
                       (this is the production path; set False to inspect the bare model)
        """
        r = apply_plausibility(raw[RAW_FEATURES])[0] if gate else raw[RAW_FEATURES]
        X = build_features(r)
        dm = xgb.DMatrix(X, feature_names=MODEL_FEATURES)
        proba, dec = {}, {}
        for lab in LABELS:
            info = self.meta["labels_info"][lab]
            p = _apply_calibrator(info["calibrator"], self.boosters[lab].predict(dm))
            d = conformal_decision(p, info["q_pos"], info["q_neg"])
            d = np.where((d == "NEGATIVE") & (p >= info["t_sens95"]), "UNCERTAIN", d)
            if assess_guard:
                ok = r[REQUIRED_INPUTS[lab]].notna().any(axis=1).values
                d = np.where(ok, d, "NOT_ASSESSABLE")
            proba[lab], dec[lab] = p, d
        return {"proba": pd.DataFrame(proba, index=raw.index), "decision": pd.DataFrame(dec, index=raw.index)}

    # -- site adaptation -----------------------------------------------------------------------
    def recalibrate_site(self, raw: pd.DataFrame, labels: pd.DataFrame, out_dir: str | Path,
                         refit_calibration: bool = False, min_cases: int = 400, min_pos: int = 30) -> dict:
        """Adapt to a new laboratory using locally labelled cases (e.g. physician sign-offs).

        * always: re-derive the conformal thresholds (q_pos / q_neg) and the 95%-sensitivity point on the local
          assessable cases - this restores the coverage guarantee under analyser bias / case-mix shift.
        * refit_calibration=True: also refit the probability calibrator on a disjoint half of the local data
          (needs roughly >= 2000 labelled cases to be stable).
        Boosters are copied unchanged. Labels with too few local positives keep the global thresholds.
        Returns a per-label report; writes a complete model directory to out_dir.
        """
        import shutil
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        clean = apply_plausibility(raw[RAW_FEATURES])[0]
        dm = xgb.DMatrix(build_features(clean), feature_names=MODEL_FEATURES)
        meta = json.loads(json.dumps(self.meta))
        half = len(raw) // 2 if refit_calibration else 0
        rep = {}
        for lab in LABELS:
            y = labels[lab].values
            info = meta["labels_info"][lab]
            raw_p = self.boosters[lab].predict(dm)
            if refit_calibration:
                if y[:half].sum() >= 30:
                    info["calibrator"] = _fit_calibrator(raw_p[:half], y[:half], force_platt=True)
                sl = slice(half, None)
            else:
                sl = slice(None)
            p = _apply_calibrator(info["calibrator"], raw_p)
            a = assessable_mask(clean, lab)[sl]
            ys, ps = y[sl][a], p[sl][a]
            n_pos = int(ys.sum())
            if len(ys) < min_cases or n_pos < min_pos:
                rep[lab] = {"updated": False, "reason": f"{len(ys)} assessable cases / {n_pos} positives below minimum"}
                continue
            info["q_pos"] = _conformal_q(1 - ps[ys == 1], ALPHA_POS)
            info["q_neg"] = _conformal_q(ps[ys == 0], ALPHA_NEG)
            info["t_sens95"] = float(np.quantile(ps[ys == 1], 0.05))
            rep[lab] = {"updated": True, "q_pos": info["q_pos"], "q_neg": info["q_neg"], "assessable_cases": len(ys),
                        "positives": n_pos, "calibrator_refit": bool(refit_calibration)}
        meta["site_recalibrated"] = True
        for lab in LABELS:
            shutil.copy(Path(self.model_dir) / f"xgb_{lab}.json", out_dir / f"xgb_{lab}.json")
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        return rep

    # -- single patient API --------------------------------------------------------------------
    def analyze(self, analytes: dict, age=None, sex=None, top_k: int = 3) -> dict:
        vals, notes = normalize_input(analytes, age, sex)
        warnings = list(notes)
        row = pd.DataFrame([{c: vals.get(c, np.nan) for c in RAW_FEATURES}])

        # scope check
        a, s = row.at[0, "age"], row.at[0, "sex"]
        if pd.isna(a) or pd.isna(s):
            return {"status": "INSUFFICIENT_INPUT", "reason": "age and sex are required", "model_version": self.version}
        if a < 18:
            return {"status": "OUT_OF_SCOPE", "reason": "model is validated for adults (>=18) only; paediatric "
                    "reference ranges differ", "model_version": self.version}

        clean, removed = apply_plausibility(row)
        for c in removed.columns[removed.iloc[0]]:
            warnings.append(f"{FEATURE_TEXT.get(c, c)}: {row.at[0, c]:g} is physiologically implausible - "
                            f"treated as a misread/unit error and EXCLUDED. Verify against the source report.")
        flags = consistency_flags(clean)
        verify = [c for c in flags.columns[flags.iloc[0]]]
        # Critical limits are checked on the cleaned values
        critical = self._critical_flags(clean.iloc[0])
        patterns = self._pattern_alerts(clean.iloc[0])

        X = build_features(clean)
        dm = xgb.DMatrix(X, feature_names=MODEL_FEATURES)

        conditions = {}
        for lab in LABELS:
            info = self.meta["labels_info"][lab]
            raw_p = self.boosters[lab].predict(dm)
            p = float(_apply_calibrator(info["calibrator"], raw_p)[0])
            dec = str(conformal_decision(np.array([p]), info["q_pos"], info["q_neg"])[0])
            if dec == "NEGATIVE" and p >= info["t_sens95"]:
                dec = "UNCERTAIN"
            need = REQUIRED_INPUTS[lab]
            assessable = any(pd.notna(clean.at[0, c]) for c in need)
            if not assessable:
                dec = "NOT_ASSESSABLE"
            conditions[lab] = {
                "name": LABEL_DISPLAY[lab], "decision": dec, "probability": round(p, 4),
                "assessable": assessable,
                "drivers": self._drivers(lab, dm, X, top_k) if assessable else [],
            }

        # data-quality + ood
        ood = self._ood(clean.iloc[0])
        if ood["n_outside"] > 0:
            warnings.append(f"{ood['n_outside']} value(s) lie outside the training envelope "
                            f"({', '.join(ood['features'])}) - treat predictions with extra caution.")
        if verify:
            warnings.append("Cross-field consistency check failed for: "
                            + ", ".join(FEATURE_TEXT.get(c, c) for c in verify) + " - possible OCR misread; verify.")

        findings = [k for k, v in conditions.items() if v["decision"] == "POSITIVE"]
        uncertain = [k for k, v in conditions.items() if v["decision"] == "UNCERTAIN"]
        unassessable = sorted({k for k, v in conditions.items() if v["decision"] == "NOT_ASSESSABLE"})

        if critical:
            status = "CRITICAL_ESCALATION"
        elif findings:
            status = "ABNORMAL"
        elif uncertain or verify or ood["n_outside"]:
            status = "REVIEW_NEEDED"
        elif unassessable:
            status = "NO_FINDINGS_ON_AVAILABLE_PANELS"
        else:
            status = "NO_SIGNIFICANT_FINDINGS"

        return {
            "status": status,
            "findings": [conditions[k]["name"] for k in findings],
            "uncertain": [conditions[k]["name"] for k in uncertain],
            "not_assessable": [LABEL_DISPLAY[k] for k in unassessable],
            "critical_alerts": critical,
            "pattern_alerts": patterns,
            "conditions": conditions,
            "suggested_next_tests": self._next_tests(conditions, clean.iloc[0]),
            "verify_fields": [{"field": c, "value": None if pd.isna(row.at[0, c]) else float(row.at[0, c])} for c in verify],
            "warnings": warnings,
            "ood": ood,
            "egfr": None if pd.isna(X.at[0, "egfr"]) else round(float(X.at[0, "egfr"]), 1),
            "disclaimer": "Decision support only. Requires review and sign-off by a registered physician.",
            "model_version": self.version,
        }

    # -- helpers -------------------------------------------------------------------------------
    @staticmethod
    def _critical_flags(r: pd.Series) -> list[dict]:
        out = []
        for c, (lo, hi) in CRITICAL_LIMITS.items():
            v = r.get(c, np.nan)
            if pd.isna(v):
                continue
            if lo is not None and v < lo:
                out.append({"analyte": FEATURE_TEXT.get(c, c), "value": float(v), "limit": f"< {lo}", "direction": "low"})
            if hi is not None and v > hi:
                out.append({"analyte": FEATURE_TEXT.get(c, c), "value": float(v), "limit": f"> {hi}", "direction": "high"})
        return out

    @staticmethod
    def _pattern_alerts(r: pd.Series) -> list[str]:
        """Rule-based reminders (NOT model output). They prompt clinical review; they are not diagnoses."""
        out = []
        plt, wbc, hct = r.get("platelets", np.nan), r.get("wbc", np.nan), r.get("hct", np.nan)
        if pd.notna(plt) and plt < 100:
            extra = " with leukopenia" if pd.notna(wbc) and wbc < 4.0 else ""
            out.append(f"Thrombocytopenia{extra} (platelets {plt:g} x10^3/uL): in an endemic setting consider dengue - "
                       "serial platelet count and haematocrit, clinical review.")
        if pd.notna(hct) and pd.notna(r.get("sex", np.nan)) and hct > (54 if r["sex"] == 1 else 50):
            out.append(f"Haematocrit {hct:g}% is high: consider haemoconcentration/dehydration or polycythaemia; correlate clinically.")
        if pd.notna(r.get("hb", np.nan)) and r["hb"] < 8.0:
            out.append("Haemoglobin < 8 g/dL (severe anaemia by WHO criteria): same-day clinical assessment.")
        return out

    def _drivers(self, lab: str, dm: xgb.DMatrix, X: pd.DataFrame, k: int) -> list[dict]:
        contrib = self.boosters[lab].predict(dm, pred_contribs=True)[0][:-1]
        order = np.argsort(-np.abs(contrib))[:k]
        out = []
        for i in order:
            f = MODEL_FEATURES[i]
            if abs(contrib[i]) < 0.05:
                continue
            v = X.at[0, f]
            missing = bool(pd.isna(v))
            out.append({"feature": FEATURE_TEXT.get(f, f) + (" (not reported)" if missing else ""),
                        "value": None if missing else round(float(v), 2),
                        "effect": "raises" if contrib[i] > 0 else "lowers", "log_odds_shift": round(float(contrib[i]), 2)})
        return out

    def _ood(self, r: pd.Series) -> dict:
        outside = []
        for c, (lo, hi) in self.meta["ood_bounds"].items():
            v = r.get(c, np.nan)
            if pd.notna(v) and (v < lo or v > hi):
                outside.append(FEATURE_TEXT.get(c, c))
        return {"n_outside": len(outside), "features": outside}

    @staticmethod
    def _next_tests(conditions: dict, r: pd.Series) -> list[str]:
        tests: list[str] = []
        for lab, v in conditions.items():
            if v["decision"] in ("UNCERTAIN", "NOT_ASSESSABLE"):
                for feat, text in NEXT_TEST_FOR[lab]:
                    if pd.isna(r.get(feat, np.nan)):
                        tests.append(f"{text}  [{LABEL_DISPLAY[lab]}]")
                        break
                else:
                    tests.append(f"{NEXT_TEST_FOR[lab][-1][1]}  [{LABEL_DISPLAY[lab]}]")
        return tests
