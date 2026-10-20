# CliniQ ML Engine v2.2 - Model Card

**Intended use:** decision support for registered physicians reviewing adult laboratory reports (FBC, glycaemia, lipids, renal, liver, thyroid). Output is never a final diagnosis; every result requires physician sign-off.
**Not for:** patients under 18, pregnancy, acute emergencies, autonomous reporting, or any use without clinician review.

## What it is
Seven independent XGBoost classifiers (anemia, CKD, diabetes, hypercholesterolemia, hypothyroidism, hyperthyroidism, infection) so that comorbidities are predicted together, plus:

| Layer | Purpose |
|---|---|
| Plausibility gate + unit normaliser | values outside physiological bounds are excluded (never guessed); mmol/L, umol/L, g/L slips are converted and logged |
| Cross-field consistency checks | Hb/Hct/MCHC/MCV/MCH identities, differential sum, Friedewald LDL, HbA1c-glucose: flag likely OCR misreads |
| Calibration | isotonic (Platt for rare labels), so 0.7 means about 70% |
| Class-conditional split conformal | each label returns POSITIVE / NEGATIVE / UNCERTAIN; positives are covered at >=95%, negatives at >=90% (under exchangeability) |
| Assessability guard | a condition with none of its required inputs is `NOT_ASSESSABLE`, never "negative" |
| Safety override | critical lab limits escalate regardless of the model; a negative with probability above the 95%-sensitivity point becomes UNCERTAIN |
| Explanations | per-prediction XGBoost TreeSHAP contributions (top 3), missing inputs marked "(not reported)" |
| Next-test suggestions | for UNCERTAIN / NOT_ASSESSABLE conditions, the missing test most likely to resolve it |
| Site recalibration | re-derives conformal thresholds from locally signed-off cases (restores coverage on a new lab) |

Artifacts are plain JSON (no pickle).

## Data - read this first
**All training and evaluation data is synthetic** (`cliniq_ml/synth.py`). It is much harder than the previous generator: labels come from a latent disease process rather than analyte cut-offs, severity is continuous, and it includes comorbidity, mimics (dehydration, stress leukocytosis, pre-diabetes, controlled diabetics, non-thyroidal illness, subclinical thyroid disease, mixed anemia morphology), informative missingness, label noise, and a held-out laboratory with analyser bias and an older case mix.
**None of these numbers are clinical evidence.** They show the engineering works (calibration, coverage, abstention, robustness). Real performance must be measured on de-identified local data before any clinical use.

## Results (synthetic; 30,000 patients, 6 labs; internal test n=3,750; external lab n=5,000, unseen in training)
Decision metrics are over *assessable* cases. Full tables: `reports/metrics.json`, plot: `reports/evaluation.png`.

| Label | AUROC internal | AUROC shifted lab | ECE internal | Positive coverage (target .95) | Negative coverage (target .90) | Uncertain rate |
|---|---|---|---|---|---|---|
| Anemia | 0.941 | 0.911 | 0.008 | 0.959 | 0.905 | 25% |
| CKD | 0.967 | 0.976 | 0.008 | 0.957 | 0.927 | 17% |
| Diabetes | 0.970 | 0.969 | 0.012 | 0.974 | 0.922 | 16% |
| Hypercholesterolemia | 0.919 | 0.873 | 0.015 | 0.959 | 0.937 | 33% |
| Hypothyroidism | 0.990 | 0.970 | 0.008 | 0.986 | 0.980 | 4% |
| Hyperthyroidism | 0.966 | 0.968 | 0.003 | 0.959 | 0.919 | 44% |
| Infection | 0.898 | 0.868 | 0.013 | 0.960 | 0.937 | 67% |

(Thyroid AUROCs are high only because they are computed on cases where TSH was reported; 55% of cases had no TSH and are `NOT_ASSESSABLE`.)

**Honest observations**
- A regularised logistic-regression baseline is within about 0.01 AUROC of XGBoost on most labels (better on diabetes and CKD; XGBoost wins mainly on infection, +0.04). This synthetic task is largely monotone. Do not expect XGBoost to be decisive until it is trained on real data.
- Infection is genuinely hard (viral cases have normal WBC): 67% abstention. That is the correct behaviour, not a defect.
- **Shift hurts negative coverage**: on the shifted lab it fell to 0.74-0.79 for CKD and anemia (positive coverage held at about 0.95+). After conformal-only site recalibration with 2,500 local labelled cases it recovered to 0.89-0.94 on all labels. Positive coverage after recalibration varied between about 0.94 and 1.0 and is noisy with few local positives; refitting the calibrator on small local data made coverage less stable, so it is optional and off by default.
- Hypercholesterolemia has the lowest sensitivity (0.68 at the decision rule) because borderline cases overlap the healthy range by design.
- Removing a whole panel: the bare model called 7-15% of true positives "negative"; the assessability guard reduces that to 0% by refusing to answer.
- OCR decimal slips (5% of rows): AUROC barely moves (<=0.006) because a single wrong field rarely dominates, but only 64% of x10/x0.1 slips were caught by bounds + consistency checks (and 56% of x2/x0.5 misreads). The rest need a second independent reader: use `verify.cross_check_extractions` on EasyOCR vs the VLM output. False flags on clean rows: 0.05%.

## Known limitations
- Reference intervals are not age/sex-specific for most analytes beyond what the model learns.
- Does not read the reference range printed on each report (recommended next step).
- Critical limits and pattern alerts are conventional values; have the lab/clinician confirm them against local SOPs.
- Informative missingness in the synthetic data (tests ordered when disease suspected) may not match real ordering behaviour.
- Single-label ground truth per condition; no severity staging beyond the displayed eGFR.
- Likely to require NMRA review if deployed as software as a medical device.

## Before any real deployment
1. Retrain on de-identified real labelled reports (consented, PDPA-compliant) and report calibration and coverage per site.
2. Run `recalibrate_site` on each new laboratory using physician sign-offs as labels.
3. Prospective silent-mode evaluation against physician decisions.
4. Clinician review of the critical limits, pattern alerts and `NEXT_TEST_FOR` table.
