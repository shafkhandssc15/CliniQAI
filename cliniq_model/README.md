# CliniQ ML Engine v2.2

Calibrated, multi-label, abstaining lab-report model. Replaces the 8-class XGBoost "Engine 2". See `MODEL_CARD.md` for what the numbers do and do not mean.

```bash
pip install -r requirements.txt
python train.py          # ~20 s: generates data, trains, evaluates, writes artifacts/ and reports/
python -m pytest -q tests
```

## Use

```python
from cliniq_ml.engine import CliniQEngine
eng = CliniQEngine("artifacts")
r = eng.analyze({"Haemoglobin": 9.4, "Fasting Blood Sugar": 9.8, "Serum Creatinine": 2.1, "HbA1c": 8.4},
                age=63, sex="M")
r["status"]          # CRITICAL_ESCALATION | ABNORMAL | REVIEW_NEEDED | NO_FINDINGS_ON_AVAILABLE_PANELS | NO_SIGNIFICANT_FINDINGS
r["conditions"]      # per condition: decision, calibrated probability, SHAP drivers
r["not_assessable"]  # panels that were not provided (never reported as healthy)
r["verify_fields"]   # values that failed cross-field checks (probable OCR misreads)
r["suggested_next_tests"]
```

FastAPI wiring: `integration_example.py` (`/api/ml/analyze` + `adapt_for_consensus`).
Optional LLM wording (Roar AI or anything): `cliniq_ml.narrative.generate_narrative(result, llm_call)` where `llm_call` is `prompt -> text`. The output is checked against the model's findings and falls back to a deterministic summary if the LLM invents a condition or drops a critical alert.
Cross-check two readers: `cliniq_ml.verify.cross_check_extractions(easyocr_values, vlm_values)`.

## Adapting to your lab (the important step)
Physician sign-offs are labels. Once you have a few hundred per condition:
```python
eng.recalibrate_site(raw_df, labels_df, "artifacts_labA")   # restores coverage for that lab
```
To train on real data, replace `synth.generate()` in `train.py` with a DataFrame that has the columns in `schema.RAW_FEATURES` plus the seven label columns and a `site` column (the highest site id is held out as the external test lab).

## Layout
`cliniq_ml/schema.py` analytes, units, bounds, critical limits, aliases - `features.py` eGFR, derived features, plausibility, consistency - `synth.py` hard synthetic cohort - `engine.py` training, calibration, conformal, inference - `verify.py`, `narrative.py` - `tests/`
