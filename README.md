# CliniQ AI — Clinical Intelligence & Multimodal Consensus Platform

An advanced medical AI decision-support workstation and clinical consensus command center designed for **SLMC-registered physicians and clinical pathology laboratories in Sri Lanka**.

> **IMPORTANT CLINICAL NOTICE**
> Decision support only — requires review and sign-off by a registered physician. Output is never a final diagnosis. Autonomous reporting is strictly prohibited. Not validated for patients under 18 years of age, acute life-threatening emergencies, or obstetric care.

---

## 🌟 Architecture & Clinical Flow

```
Lab Report Image ──> De-identification ──> EasyOCR + Gemini/NIM Multimodal
                                                    │
                                                    ▼
                                      Extraction Cross-Check
                             (Flags decimal slips & reader discrepancies)
                                                    │
                                                    ▼
                                      CliniQ ML Engine v2.2
                        (Plausibility Gates + TreeSHAP Drivers + Conformal Bounds)
                                                    │
                                                    ▼
                                      Clinical Consensus Core
                       (Reference Voter Arbitration + Emergency Escalation Veto)
                                                    │
                                                    ▼
                                         Physician Sign-Off Station
                                      (Confirm / Modify / Reject)
                                                    │
                                                    ▼
                                 Feedback Loop & Site Recalibration
```

### Multimodal Diagnostic Components
1. **Calibrated Reference Voter (CliniQ ML v2.2)**: 7 independent XGBoost classifiers (`anemia`, `ckd`, `diabetes`, `hypercholesterolemia`, `hypothyroidism`, `hyperthyroidism`, `infection`) with split conformal prediction, isotonic calibration, and TreeSHAP feature explanations.
2. **Deterministic Clinical Rule Engine**: Evidence-based hematological, glycemic, and lipid risk stratification (WHO, KDIGO, ADA, ATP III).
3. **Multimodal VLMs (NVIDIA NIM & Google Gemini)**: Multimodal extraction cross-check and contextual clinical reasoning narrative.
4. **Safety & Veto Guardrails**:
   - **Emergency Critical Alerts**: Life-threatening laboratory threshold breaches (e.g. Hb < 6.0 g/dL, Platelets < 20 × 10³/µL) escalate immediately regardless of model output.
   - **Assessability Guard**: Missing panels return `NOT_ASSESSABLE` and are **never** rendered as normal or healthy.
   - **Healthy Label Gating**: The "healthy" status is issued *only* when all panels are present and status is `NO_SIGNIFICANT_FINDINGS`.

---

## 📊 Honest ML Evaluation (Hard Synthetic Cohort)

> **DATA NOTICE & CLINICAL EVIDENCE DISCLAIMER**
> **All training and evaluation data is synthetic** (`cliniq_ml/synth.py`). It simulates complex continuous disease processes, severe mimics (dehydration, stress leukocytosis, non-thyroidal illness, mixed morphology), informative missingness, and a held-out laboratory with analyser bias.
> **None of these numbers represent clinical trial evidence.** They demonstrate mathematical calibration, conformal coverage guarantees, and abstention behavior. Real-world validation on de-identified local clinical data is mandatory prior to clinical deployment.

### Performance on Hard Synthetic Cohort (N = 30,000 patients, 6 laboratories)
*Internal test set: n = 3,750; External shifted laboratory: n = 5,000 (unseen in training)*

| Condition | AUROC (Internal) | AUROC (Shifted Lab) | ECE (Internal) | Positive Coverage (Target ≥ 0.95) | Negative Coverage (Target ≥ 0.90) | Uncertain / Abstention Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Anemia** | 0.941 | 0.911 | 0.008 | 0.959 | 0.905 | 25% |
| **Chronic Kidney Disease (CKD)** | 0.967 | 0.976 | 0.008 | 0.957 | 0.927 | 17% |
| **Diabetes Mellitus** | 0.970 | 0.969 | 0.012 | 0.974 | 0.922 | 16% |
| **Hypercholesterolemia** | 0.919 | 0.873 | 0.015 | 0.959 | 0.937 | 33% |
| **Hypothyroidism** | 0.990 | 0.970 | 0.008 | 0.986 | 0.980 | 4% |
| **Hyperthyroidism** | 0.966 | 0.968 | 0.003 | 0.959 | 0.919 | 44% |
| **Infection** | 0.898 | 0.868 | 0.013 | 0.960 | 0.937 | 67% |

*(Note: Thyroid AUROCs reflect assessable cases where TSH was reported; 55% of cases had no TSH and were designated `NOT_ASSESSABLE`).*

### Documented Technical Observations & Limitations
1. **Infection Uncertainty (67% Abstention)**: Viral infections often present with normal or fluctuating total WBC. Rather than emitting false negatives or uncalibrated guesses, the engine abstains and triggers clinical review.
2. **Hypercholesterolemia Overlap**: Lower decision sensitivity (0.68) because borderline dyslipidemia distributions intentionally overlap normal healthy ranges.
3. **Shifted Lab Coverage Drop & Recovery**: Unseen analyzer shifts cause negative coverage to drop to 0.74–0.79 for CKD and anemia. Running site recalibration (`python export_recalibration_data.py` and `engine.recalibrate_site`) with local physician sign-off labels restores negative coverage to 0.89–0.94 across all conditions.
4. **Assessability Guard**: Completely eliminates false negatives on missing panels (0% false healthy rate).
5. **OCR Decimal Slips**: Bounds and consistency checks catch ~64% of 10× slips; the remaining slips are caught through `verify.cross_check_extractions` comparing EasyOCR against VLM extractions before analysis runs.

---

## 🚀 Getting Started

### 1. Requirements
* Python 3.10+ (Tested on Python 3.11 & 3.12)
* Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
Verify that all 38 test suites pass:
```bash
python -m pytest -q tests
```

### 3. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Put credentials in the ignored local `.env` file only. Supported provider variables are `ROAR_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, and `NVIDIA_API_KEY`. Configure Supabase and `CLINIQ_SECRET_KEY` there as well. Do not put credential values in the UI, source files, or database.

### 4. Database Initialization
```bash
python -c "import asyncio; from database import init_db; asyncio.run(init_db())"
```

### 5. Start Server
```bash
python app.py
```
Open **`http://127.0.0.1:8000/`** in your browser.

---

## 🩺 Default Verification & Testing Accounts

| Role | Email | Password | Access / Scope |
| :--- | :--- | :--- | :--- |
| **Attending Doctor** | `kamal@hospital.lk` | `Test@1234` | Clinical workstation, AI analysis, physician review sign-off |
| **Platform Administrator** | `admin@cliniq.lk` | `Admin@123` | System audit logs, doctor approvals, recalibration export |

---

## 🔒 Security, Privacy & Compliance (Sri Lanka PDPA)
* **Automatic De-Identification**: Automatically redacts patient names, NIC, and phone numbers before multimodal transmission.
* **Cryptographic Audit Log**: SHA-256 forward-linked hash chain auditing every clinical and diagnostic event.
* **Isolated Doctor RLS**: Supabase PostgreSQL Row Level Security policy enforces tenant data isolation (`doctor_id = auth.uid()::text`).
* **Safe Recalibration Protocol**: Site recalibration data requires 400+ sign-off cases and ≥30 positives per condition. Recalibrated artifacts are never auto-applied to production without human admin review.
