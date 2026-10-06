# CliniQ AI — Premium 3D Clinical Intelligence Platform

An advanced medical AI workstation and clinical consensus command center designed for **SLMC-registered physicians and laboratory specialists in Sri Lanka**.

---

## 🌟 Key Features

* **Signature Clinical Flow**:
  $$\text{Report} \longrightarrow \text{OCR} \longrightarrow \text{4 AI Engines} \longrightarrow \text{Consensus Core} \longrightarrow \text{Clinical Findings} \longrightarrow \text{Physician Approval}$$
* **4 Diagnostic Engines**:
  1. **Deterministic Rule Engine**: Evidence-based hematological and metabolic clinical guidelines.
  2. **Local ML Classifier**: Trained XGBoost / LightGBM models for disease risk stratification.
  3. **NVIDIA NIM Vision**: Cloud VLM reasoning via `meta/llama-3.2-11b-vision-instruct`.
  4. **Google Gemini Multimodal**: Fast multimodal inference with `gemini-2.5-flash`.
* **Central Clinical Consensus Core**: Interactive 3D neural-network canvas showing engine convergence and agreement confidence.
* **Biomarker Evidence Visualization**: Horizontal analyte reference curves with directional depletion/elevation markers.
* **Differential Diagnoses & Risk Stratification**: Probabilistic rankings (e.g. Iron Deficiency Anemia vs. Thalassemia Trait) and ASCVD / WHO classifications.
* **Physician Sign-Off Station**: Mandatory SLMC verification with digital sign-off, confirmation, modification, or rejection workflows.
* **Supabase Cloud Synchronization**: Automatic real-time PostgREST sync of all diagnostic dossiers and doctor reviews to Supabase PostgreSQL.
* **Collapsible 3D Spatial Sidebar**: Smoothly toggles between a 275px workstation navigation rail and a compact 72px icon rail for full-width data analysis.
* **Hospital Letterhead Print Mode**: Dedicated clean black-and-white hospital format for physical medical records.

---

## 🚀 Quick Start

### 1. Requirements
* Python 3.10+
* Dependencies: FastAPI, Uvicorn, EasyOCR, XGBoost, Scikit-Learn, SQLAlchemy, HTTPX, PyJWT, Passlib

### 2. Environment Setup
Copy the example environment configuration:
```bash
cp .env.example .env
```
Fill in your credentials in `.env`:
```ini
SUPABASE_URL=https://gdxjzvbbgurmjqrxgqjh.supabase.co
SUPABASE_KEY=sb_publishable_7mRE0EaPvSpW-4deUma7qw__yltyWtp

```

### 3. Database Migration
Execute `supabase_schema.sql` in your Supabase Dashboard SQL Editor to initialize `patient_reports`, `doctors`, and `audit_log` tables with Row Level Security (RLS).

### 4. Run the Application
```bash
python app.py
```
Open **`http://127.0.0.1:8000/`** in your browser.

---

## 🩺 Verification & Testing Accounts

| Role | Email | Password | Access / Scope |
| :--- | :--- | :--- | :--- |
| **Attending Doctor** | `kamal@hospital.lk` | `Test@1234` | Full clinical workstation, 4-engine consensus, SLMC-00001 sign-off |
| **System Admin** | `admin@cliniq.lk` | `Admin@123` | System audit logs, doctor credential verification (`/admin.html`) |

---

## 🔒 Security & Medical Compliance
* **SLMC Validation**: Form validation ensuring valid Sri Lanka Medical Council registration format.
* **Non-Autonomous AI**: Mandates physician verification and digital sign-off before report finalization.
* **Row-Level Security**: Encrypted JWT authentication with 8-hour clinic session scoping.
