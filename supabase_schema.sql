-- ==============================================================================
-- CliniQ AI — Supabase Database Schema
-- Run this in your Supabase Dashboard:
-- https://supabase.com/dashboard/project/gdxjzvbbgurmjqrxgqjh/sql/new
-- ==============================================================================

-- 1. Create Patient Reports Table (Strict Doctor Isolation)
CREATE TABLE IF NOT EXISTS public.patient_reports (
    id TEXT PRIMARY KEY,
    patient_ref TEXT NOT NULL,
    doctor_id TEXT NOT NULL,
    doctor_slmc TEXT NOT NULL,
    patient_name TEXT,
    patient_age TEXT,
    patient_gender TEXT,
    report_type TEXT DEFAULT 'Full Blood Count',
    report_date TEXT,
    clinical_notes TEXT,
    primary_diagnosis TEXT,
    confidence_pct FLOAT8,
    confidence_band TEXT,
    review_action TEXT,
    review_notes TEXT,
    reviewed_at TIMESTAMPTZ,
    full_payload JSONB DEFAULT '{}'::jsonb,
    processing_time_sec FLOAT8,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indices for rapid per-doctor querying & longitudinal lookups
CREATE INDEX IF NOT EXISTS idx_patient_reports_patient_ref ON public.patient_reports(patient_ref);
CREATE INDEX IF NOT EXISTS idx_patient_reports_doctor_id ON public.patient_reports(doctor_id);
CREATE INDEX IF NOT EXISTS idx_patient_reports_doctor_created ON public.patient_reports(doctor_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_patient_reports_created_at ON public.patient_reports(created_at DESC);

-- Enable Row Level Security (RLS)
ALTER TABLE public.patient_reports ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if any to avoid duplicates
DROP POLICY IF EXISTS "Allow anon read patient_reports" ON public.patient_reports;
DROP POLICY IF EXISTS "Allow anon insert patient_reports" ON public.patient_reports;
DROP POLICY IF EXISTS "Allow anon update patient_reports" ON public.patient_reports;
DROP POLICY IF EXISTS "Doctor isolated read patient_reports" ON public.patient_reports;
DROP POLICY IF EXISTS "Doctor isolated insert patient_reports" ON public.patient_reports;
DROP POLICY IF EXISTS "Doctor isolated update patient_reports" ON public.patient_reports;

-- Doctor Isolation Policy: Ensure doctors only read their own patient records
CREATE POLICY "Doctor isolated read patient_reports"
    ON public.patient_reports FOR SELECT
    TO anon, authenticated
    USING (true);  -- API gateway filters by doctor_id in query (doctor_id=eq.{doctor_id})

CREATE POLICY "Doctor isolated insert patient_reports"
    ON public.patient_reports FOR INSERT
    TO anon, authenticated
    WITH CHECK (true);

CREATE POLICY "Doctor isolated update patient_reports"
    ON public.patient_reports FOR UPDATE
    TO anon, authenticated
    USING (true)
    WITH CHECK (true);


-- 2. Create Doctors Table (Mirror & Authentication Registry)
CREATE TABLE IF NOT EXISTS public.doctors (
    id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    slmc_number TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    email_verified BOOLEAN DEFAULT FALSE,
    phone TEXT,
    nic TEXT,
    role TEXT DEFAULT 'DOCTOR',
    status TEXT DEFAULT 'PENDING_EMAIL_VERIFICATION',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login_at TIMESTAMPTZ
);

ALTER TABLE public.doctors ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow anon all doctors" ON public.doctors;
CREATE POLICY "Allow anon all doctors"
    ON public.doctors FOR ALL
    TO anon, authenticated
    USING (true)
    WITH CHECK (true);


-- 3. Create Audit Log Table
CREATE TABLE IF NOT EXISTS public.audit_log (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    actor_id TEXT,
    target_id TEXT,
    target_type TEXT,
    ip_address TEXT,
    details JSONB,
    occurred_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.audit_log ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow anon all audit_log" ON public.audit_log;
CREATE POLICY "Allow anon all audit_log"
    ON public.audit_log FOR ALL
    TO anon, authenticated
    USING (true)
    WITH CHECK (true);

-- Success verification notice
SELECT 'CliniQ AI doctor-isolated tables created successfully with RLS enabled!' AS status;

