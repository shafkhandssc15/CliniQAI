"""Train + evaluate the CliniQ multi-label engine.

    python train.py [--n-per-site 5000] [--seed 7]

Writes  artifacts/  (model)  and  reports/metrics.json + reports/evaluation.png.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cliniq_ml.engine import CliniQEngine, train
from cliniq_ml.features import apply_plausibility, consistency_flags
from cliniq_ml.schema import LABELS, RAW_FEATURES
from cliniq_ml.synth import generate, inject_ocr_errors

ROOT = Path(__file__).parent


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def label_metrics(proba: pd.DataFrame, dec: pd.DataFrame, y: pd.DataFrame) -> dict:
    """Ranking/calibration over all rows; decision metrics over ASSESSABLE rows only."""
    out = {}
    for lab in LABELS:
        p, d, t = proba[lab].values, dec[lab].values, y[lab].values
        a = d != "NOT_ASSESSABLE"
        pa, da, ta = p[a], d[a], t[a]
        pos, neg = ta == 1, ta == 0
        decided = da != "UNCERTAIN"
        correct = ((da == "POSITIVE") & pos) | ((da == "NEGATIVE") & neg)
        out[lab] = dict(
            prevalence=float(t.mean()), assessable_rate=float(a.mean()),
            auroc=float(roc_auc_score(ta, pa)), auprc=float(average_precision_score(ta, pa)),
            brier=float(brier_score_loss(ta, pa)), ece=ece(pa, ta),
            pos_coverage=float((da[pos] != "NEGATIVE").mean()),   # target >= 1 - ALPHA_POS
            neg_coverage=float((da[neg] != "POSITIVE").mean()),   # target >= 1 - ALPHA_NEG
            sensitivity=float((da[pos] == "POSITIVE").mean()),
            uncertain_rate=float((~decided).mean()),
            selective_accuracy=float(correct[decided].mean()) if decided.any() else float("nan"),
            positives_called_negative=float((da[pos] == "NEGATIVE").mean()),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-site", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data-path", type=str, default=None, help="Path to real-world de-identified clinical lab CSV/Parquet dataset.")
    ap.add_argument("--blend-synthetic", action="store_true", help="Blend real dataset with synthetic cohort.")
    args = ap.parse_args()

    t0 = time.time()
    if args.data_path:
        from cliniq_ml.real_data_loader import load_real_clinical_dataset, augment_with_real_data
        print(f"Loading real-world clinical dataset from {args.data_path} ...")
        real_data = load_real_clinical_dataset(args.data_path)
        print(f"  Loaded {len(real_data):,} real clinical records.")
        if args.blend_synthetic:
            print("Generating synthetic cohort to blend with real dataset ...")
            syn_data = generate(args.n_per_site, 6, args.seed)
            data = augment_with_real_data(syn_data, real_data)
            print(f"  Blended dataset: {len(data):,} total cases.")
        else:
            data = real_data
    else:
        print("Generating synthetic cohort ...")
        data = generate(args.n_per_site, 6, args.seed)
        print(f"  {len(data):,} patients, 6 labs (lab 5 = shifted external lab, never seen in training)")

    print("Training ...")
    res = train(data, ROOT / "artifacts", seed=11)
    splits, X = res["splits"], res["X"]
    eng = CliniQEngine(ROOT / "artifacts")
    report: dict = {"version": eng.version, "n_total": len(data)}

    # -- baseline: regularised multi-label logistic regression -----------------------------------
    med = X["train"].median()
    prep = lambda df: df.fillna(med).replace([np.inf, -np.inf], 0)
    report["baseline_logreg_auroc_test"] = {}
    for lab in LABELS:
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(prep(X["train"]), splits["train"][lab])
        report["baseline_logreg_auroc_test"][lab] = float(roc_auc_score(splits["test"][lab], lr.predict_proba(prep(X["test"]))[:, 1]))

    # -- internal test + shifted external lab ------------------------------------------------------
    cache = {}
    for name in ("test", "external"):
        s = splits[name]
        cache[name] = eng.predict_frame(s[RAW_FEATURES])
        report[name] = label_metrics(cache[name]["proba"], cache[name]["decision"], s[LABELS])

    # -- site recalibration on the external lab (uses 2500 labelled cases, evaluates on the rest) --
    ext = splits["external"]
    rec_dir = ROOT / "artifacts_site_recalibrated_demo"
    N_LOCAL = 2500
    rep = eng.recalibrate_site(ext.iloc[:N_LOCAL][RAW_FEATURES], ext.iloc[:N_LOCAL][LABELS], rec_dir, refit_calibration=False)
    eng2 = CliniQEngine(rec_dir)
    rest = ext.iloc[N_LOCAL:]
    pr2 = eng2.predict_frame(rest[RAW_FEATURES])
    report["external_after_site_recalibration"] = label_metrics(pr2["proba"], pr2["decision"], rest[LABELS])
    report["external_after_site_recalibration"]["_recalibration_report"] = rep
    prb = eng.predict_frame(rest[RAW_FEATURES])
    report["external_before_recalibration_same_rows"] = label_metrics(prb["proba"], prb["decision"], rest[LABELS])
    import shutil
    shutil.rmtree(rec_dir, ignore_errors=True)

    # -- subgroup AUROC (internal test) ---------------------------------------------------------------
    s = splits["test"]
    pr = cache["test"]
    sub = {}
    for gname, mask in {"female": s["sex"] == 0, "male": s["sex"] == 1, "age<40": s["age"] < 40,
                        "age40-64": (s["age"] >= 40) & (s["age"] < 65), "age>=65": s["age"] >= 65}.items():
        m = mask.values
        sub[gname] = {lab: float(roc_auc_score(s.loc[m, lab], pr["proba"].loc[m, lab])) for lab in LABELS if s.loc[m, lab].nunique() > 1}
    report["subgroup_auroc"] = sub

    # -- missing-panel stress (bare model vs production path with assessability guard) ------------------
    panels = {"thyroid": (["tsh", "ft4"], ["hypothyroidism", "hyperthyroidism"]), "renal": (["creatinine", "urea"], ["ckd"]),
              "glycaemia": (["fbs", "hba1c"], ["diabetes"]), "lipids": (["total_chol", "ldl", "hdl", "tg"], ["hypercholesterolemia"])}
    stress = {}
    for pn, (cols, targets) in panels.items():
        r = s[RAW_FEATURES].copy()
        r[cols] = np.nan
        bare = eng.predict_frame(r, assess_guard=False)["decision"]
        prod = eng.predict_frame(r)["decision"]
        for lab in targets:
            pos = s[lab].values == 1
            stress[f"{pn}_panel_removed/{lab}"] = dict(
                bare_model_positives_called_negative=float((bare[lab].values[pos] == "NEGATIVE").mean()),
                production_positives_called_negative=float((prod[lab].values[pos] == "NEGATIVE").mean()),
                production_not_assessable=float((prod[lab] == "NOT_ASSESSABLE").mean()))
    report["missing_panel_stress"] = stress

    # -- OCR decimal-slip robustness -----------------------------------------------------------------------
    bad, factor = inject_ocr_errors(s[RAW_FEATURES], rate=0.05, seed=3)
    hit = factor != 1.0
    p_raw = eng.predict_frame(bad, gate=False)["proba"]
    p_gate = eng.predict_frame(bad, gate=True)["proba"]
    p_clean = cache["test"]["proba"]
    rob = {"rows_corrupted": int(hit.any(axis=1).sum()),
           "auroc": {lab: dict(clean=float(roc_auc_score(s[lab], p_clean[lab])), corrupted_no_gate=float(roc_auc_score(s[lab], p_raw[lab])),
                               corrupted_with_gate=float(roc_auc_score(s[lab], p_gate[lab]))) for lab in LABELS}}
    cleaned, removed = apply_plausibility(bad)
    flags = consistency_flags(cleaned)
    removed = removed.reindex(columns=hit.columns, fill_value=False)
    flags = flags.reindex(columns=hit.columns, fill_value=False)
    det = {}
    for nm, sel in {"decimal_slip_x10_or_x0.1": (factor == 10.0) | (factor == 0.1), "digit_misread_x2_or_x0.5": (factor == 2.0) | (factor == 0.5)}.items():
        n = int(sel.values.sum())
        b = int((removed & sel).values.sum())
        c = int((flags & sel & ~removed).values.sum())
        det[nm] = dict(n=n, caught_by_bounds=b / n, caught_by_consistency=c / n, total_caught=(b + c) / n)
    det["false_flag_rate_on_clean_rows"] = float(consistency_flags(apply_plausibility(s[RAW_FEATURES])[0]).any(axis=1).mean())
    rob["detection"] = det
    report["ocr_robustness"] = rob

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(report, indent=1))
    make_figure(report, cache, splits)

    # ------------------------------ console summary ------------------------------------------------------
    def show(title, key):
        print(f"\n== {title} ==")
        print(f"{'label':22s} {'prev':>5s} {'AUROC':>6s} {'ECE':>6s} {'posCov':>7s} {'negCov':>7s} {'sens':>5s} {'unc%':>5s} {'selAcc':>6s} {'n/a%':>5s}")
        for lab in LABELS:
            m = report[key][lab]
            print(f"{lab:22s} {m['prevalence']:5.3f} {m['auroc']:6.3f} {m['ece']:6.3f} {m['pos_coverage']:7.3f} {m['neg_coverage']:7.3f} "
                  f"{m['sensitivity']:5.2f} {m['uncertain_rate']*100:5.1f} {m['selective_accuracy']:6.3f} {(1-m['assessable_rate'])*100:5.1f}")
    show("INTERNAL TEST (decision metrics over assessable cases)", "test")
    show("EXTERNAL SHIFTED LAB, no adaptation", "external")
    show("EXTERNAL LAB rows 2500+, BEFORE site recalibration", "external_before_recalibration_same_rows")
    show("EXTERNAL LAB rows 2500+, AFTER site recalibration (2500 local cases)", "external_after_site_recalibration")
    print("\nBaseline LogReg AUROC (test):", {k: round(v, 3) for k, v in report["baseline_logreg_auroc_test"].items()})
    print("\nMissing-panel stress (positives called NEGATIVE):")
    for k, v in stress.items():
        print(f"  {k:42s} bare model {v['bare_model_positives_called_negative']:.3f} | production {v['production_positives_called_negative']:.3f} | not-assessable {v['production_not_assessable']:.2f}")
    print("\nOCR slip detection:")
    for k, v in det.items():
        print(f"  {k}: {v}")
    print(f"\nDone in {time.time()-t0:.0f}s")


def make_figure(report, cache, splits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    s = splits["test"]
    for lab in LABELS:
        p, y = cache["test"]["proba"][lab].values, s[lab].values
        a = cache["test"]["decision"][lab].values != "NOT_ASSESSABLE"
        g = pd.DataFrame({"p": p[a], "y": y[a]})
        g["b"] = pd.qcut(g["p"], 10, duplicates="drop")
        m = g.groupby("b", observed=True).mean()
        ax[0].plot(m["p"], m["y"], marker="o", ms=3, label=lab)
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set(title="Reliability (internal test, assessable cases)", xlabel="predicted probability", ylabel="observed frequency")
    ax[0].legend(fontsize=7)

    x = np.arange(len(LABELS))
    for off, key, nm in [(-0.27, "test", "internal"), (0.0, "external", "shifted lab"), (0.27, "external_after_site_recalibration", "shifted lab + recalibration")]:
        ax[1].bar(x + off, [report[key][l]["pos_coverage"] for l in LABELS], 0.27, label=nm)
    ax[1].axhline(0.95, color="k", ls="--", lw=1)
    ax[1].set_ylim(0.85, 1.0)
    ax[1].set(title="Positive-class coverage (target 0.95)", xticks=x)
    ax[1].set_xticklabels([l[:6] for l in LABELS], fontsize=8)
    ax[1].legend(fontsize=7)

    for off, key, nm in [(-0.27, "test", "internal"), (0.0, "external", "shifted lab"), (0.27, "external_after_site_recalibration", "shifted lab + recalibration")]:
        ax[2].bar(x + off, [report[key][l]["neg_coverage"] for l in LABELS], 0.27, label=nm)
    ax[2].axhline(0.90, color="k", ls="--", lw=1)
    ax[2].set_ylim(0.70, 1.0)
    ax[2].set(title="Negative-class coverage (target 0.90)", xticks=x)
    ax[2].set_xticklabels([l[:6] for l in LABELS], fontsize=8)
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / "evaluation.png", dpi=130)


if __name__ == "__main__":
    main()
