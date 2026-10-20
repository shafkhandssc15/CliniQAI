-- ==============================================================================
-- CliniQ AI — Migration 002: Isolated Doctor RLS & Condition Feedback Ledger
-- Applied to: Supabase PostgreSQL / Local Schema
-- Description:
--   1. Replaces open USING (true) policy with strict per-doctor isolation.
--   2. Adds condition_feedback table for tracking physician sign-off labels.
--   3. Adds RLS isolation policies to condition_feedback.
-- ==============================================================================

-- 1. Create condition_feedback table
CREATE TABLE IF NOT EXISTS public.condition_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL,
    patient_ref VARCHAR(20) NOT NULL,
    doctor_id VARCHAR(36) NOT NULL,
    condition_name VARCHAR(50) NOT NULL,
    model_decision VARCHAR(20) NOT NULL,
    model_probability DOUBLE PRECISION,
    physician_decision VARCHAR(20) NOT NULL,
    final_label INTEGER,
    review_notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_cf_report_id ON public.condition_feedback(report_id);
CREATE INDEX IF NOT EXISTS idx_cf_patient_ref ON public.condition_feedback(patient_ref);
CREATE INDEX IF NOT EXISTS idx_cf_doctor_id ON public.condition_feedback(doctor_id);
CREATE INDEX IF NOT EXISTS idx_cf_condition ON public.condition_feedback(condition_name);

-- 2. Enforce strict Doctor Isolation RLS on patient_reports
-- (Removes the un-isolated USING (true) policy)
ALTER TABLE public.patient_reports ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Doctor isolated read patient_reports" ON public.patient_reports;
DROP POLICY IF EXISTS "Doctor isolated insert patient_reports" ON public.patient_reports;
DROP POLICY IF EXISTS "Doctor isolated update patient_reports" ON public.patient_reports;

-- Strict SELECT: Doctor can only view their own patient dossiers
CREATE POLICY "Doctor isolated read patient_reports"
ON public.patient_reports
FOR SELECT
TO authenticated, anon
USING (
    doctor_id = auth.uid()::text 
    OR doctor_id = (current_setting('request.jwt.claims', true)::jsonb ->> 'sub')
);

-- Strict INSERT: Doctor can only create records under their own doctor_id
CREATE POLICY "Doctor isolated insert patient_reports"
ON public.patient_reports
FOR INSERT
TO authenticated, anon
WITH CHECK (
    doctor_id = auth.uid()::text 
    OR doctor_id = (current_setting('request.jwt.claims', true)::jsonb ->> 'sub')
);

-- Strict UPDATE: Doctor can only update/sign-off their own patient dossiers
CREATE POLICY "Doctor isolated update patient_reports"
ON public.patient_reports
FOR UPDATE
TO authenticated, anon
USING (
    doctor_id = auth.uid()::text 
    OR doctor_id = (current_setting('request.jwt.claims', true)::jsonb ->> 'sub')
);

-- 3. Enforce Doctor Isolation RLS on condition_feedback
ALTER TABLE public.condition_feedback ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Doctor isolated read condition_feedback" ON public.condition_feedback;
DROP POLICY IF EXISTS "Doctor isolated insert condition_feedback" ON public.condition_feedback;

CREATE POLICY "Doctor isolated read condition_feedback"
ON public.condition_feedback
FOR SELECT
TO authenticated, anon
USING (
    doctor_id = auth.uid()::text 
    OR doctor_id = (current_setting('request.jwt.claims', true)::jsonb ->> 'sub')
);

CREATE POLICY "Doctor isolated insert condition_feedback"
ON public.condition_feedback
FOR INSERT
TO authenticated, anon
WITH CHECK (
    doctor_id = auth.uid()::text 
    OR doctor_id = (current_setting('request.jwt.claims', true)::jsonb ->> 'sub')
);
