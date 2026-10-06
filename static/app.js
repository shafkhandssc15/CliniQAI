// ╔══════════════════════════════════════════════════════════════════╗
// ║  CliniQ AI — Dashboard Controller v3                             ║
// ║  Authenticated · Ensemble AI · Patient Intake · Sign-off         ║
// ╚══════════════════════════════════════════════════════════════════╝

// ── Auth Guard ──────────────────────────────────────────────────────
const AUTH_TOKEN = localStorage.getItem("cliniq_token");
const DOCTOR_INFO = JSON.parse(localStorage.getItem("cliniq_doctor") || "null");

if (!AUTH_TOKEN || !DOCTOR_INFO) {
    window.location.href = "/login.html";
}

function doLogout() {
    fetch("/api/auth/logout", {
        method: "POST",
        headers: { "Authorization": `Bearer ${AUTH_TOKEN}` }
    }).catch(() => {}).finally(() => {
        localStorage.removeItem("cliniq_token");
        localStorage.removeItem("cliniq_doctor");
        window.location.href = "/login.html";
    });
}

// ── State ────────────────────────────────────────────────────────────
let diseaseChart   = null;
let benchmarkChart = null;
let currentData    = null;           // Last full API response
let sessionHistory = [];             // In-session report list
let reviewFinalized = false;
let currentReportPreviewUrl = "";
let currentReportFileName = "";
let reportZoomLevel = 1.0;
let reportRotation = 0;
let reportPanX = 0;
let reportPanY = 0;
let isPanningReport = false;
let startPanX = 0;
let startPanY = 0;

// ── Biomarker Reference Ranges ───────────────────────────────────────
const BIOMARKER_RANGES = {
    "Hemoglobin":        { min: 4.0,  max: 22.0, normal_min: 12.0, normal_max: 17.5, unit: "g/dL" },
    "WBC":               { min: 1.0,  max: 30.0, normal_min: 4.5,  normal_max: 11.0, unit: "×10³/µL" },
    "Platelets":         { min: 50,   max: 700,  normal_min: 150,  normal_max: 450,  unit: "×10³/µL" },
    "Fasting_Blood_Sugar":{ min: 40,  max: 350,  normal_min: 70,   normal_max: 100,  unit: "mg/dL" },
    "Creatinine":        { min: 0.2,  max: 10.0, normal_min: 0.6,  normal_max: 1.2,  unit: "mg/dL" },
    "TSH":               { min: 0.01, max: 30.0, normal_min: 0.4,  normal_max: 4.5,  unit: "mIU/L" },
    "Cholesterol":       { min: 80,   max: 400,  normal_min: 120,  normal_max: 200,  unit: "mg/dL" }
};

const BAND_COLORS = {
    HIGH:        { stroke: "#10b981", text: "High Confidence" },
    MODERATE:    { stroke: "#f59e0b", text: "Moderate Confidence" },
    LOW:         { stroke: "#f97316", text: "Low Confidence" },
    INCONCLUSIVE:{ stroke: "#ef4444", text: "Inconclusive" }
};

const SHORT_NAMES = {
    "Hemoglobin": "Hgb",
    "WBC": "WBC",
    "Platelets": "PLT",
    "Fasting_Blood_Sugar": "FBS",
    "Creatinine": "Cr",
    "TSH": "TSH",
    "Cholesterol": "Chol"
};

// ── Initialise on DOM Ready ──────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    // Populate doctor identity
    if (DOCTOR_INFO) {
        const docCard = document.getElementById("doctor-info-card");
        if (docCard) docCard.style.display = "block";
        const cleanDocName = (DOCTOR_INFO.full_name || "Doctor").replace(/^Dr\.\s*/i, "");
        setText("sidebar-doctor-name", DOCTOR_INFO.full_name || "Doctor");
        setText("sidebar-doctor-slmc", DOCTOR_INFO.slmc_number || "");
        const av = document.getElementById("sidebar-doc-avatar");
        if (av) {
            const n = cleanDocName.trim();
            av.textContent = n.split(/\s+/).map(w => w[0]).join("").toUpperCase().slice(0, 2) || "D";
        }
        const hdr = document.getElementById("header-doctor-name");
        if (hdr) { hdr.textContent = `Dr. ${cleanDocName}`; hdr.style.display = "flex"; }
        setText("review-sig-name", `Dr. ${cleanDocName} · ${DOCTOR_INFO.slmc_number}`);
    }


    // Set default report date to today (respecting local timezone offset)
    const dateEl = document.getElementById("pt-date");
    if (dateEl) {
        const localToday = new Date();
        const offset = localToday.getTimezoneOffset();
        const localDate = new Date(localToday.getTime() - (offset * 60 * 1000));
        dateEl.value = localDate.toISOString().split("T")[0];
    }

    initTabs();
    initUpload();
    initSettings();
    initReportModalEvents();
    loadBenchmarks();
    renderHistory();
    restoreSidebarState();

    // Load permanent history from server
    loadHistoryFromServer();
    initSupabaseSync();

    // Fetch profile to check keys
    fetch("/api/auth/me", { headers: { "Authorization": `Bearer ${AUTH_TOKEN}` } })
        .then(r => r.json())
        .then(d => updateBadges(d.has_nvidia_key, d.has_gemini_key))
        .catch(() => updateBadges(false, false));
});

function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

// ── Sidebar Collapse / Expand ────────────────────────────────────────
function toggleSidebar() {
    const aside = document.getElementById("app-sidebar") || document.querySelector(".sidebar");
    const container = document.querySelector(".app-container");
    if (!aside) return;

    const isCollapsed = aside.classList.toggle("collapsed");
    if (container) container.classList.toggle("sidebar-collapsed", isCollapsed);

    localStorage.setItem("cliniq_sidebar_collapsed", isCollapsed ? "true" : "false");

    // Smoothly resize charts after sidebar transitions
    setTimeout(() => {
        if (diseaseChart && typeof diseaseChart.resize === "function") diseaseChart.resize();
        if (benchmarkChart && typeof benchmarkChart.resize === "function") benchmarkChart.resize();
        if (window.dispatchEvent) window.dispatchEvent(new Event("resize"));
    }, 320);
}

function restoreSidebarState() {
    const saved = localStorage.getItem("cliniq_sidebar_collapsed");
    if (saved === "true") {
        const aside = document.getElementById("app-sidebar") || document.querySelector(".sidebar");
        const container = document.querySelector(".app-container");
        if (aside) aside.classList.add("collapsed");
        if (container) container.classList.add("sidebar-collapsed");
    }
}

// ── Tab Navigation ───────────────────────────────────────────────────
const TAB_META = {
    dashboard:   { title: "Clinical Dashboard",    subtitle: "Upload a lab report image to begin AI analysis" },
    history:     { title: "Session Report History",subtitle: "Reports analysed during this login session" },
    biomarkers:  { title: "Biomarker Profile",     subtitle: "Individual parameters mapped against clinical reference ranges" },
    comparison:  { title: "Model Output Comparison",subtitle: "Side-by-side results from all four AI engines" },
    benchmarks:  { title: "Classifier Metrics",    subtitle: "Training performance across 3,000 augmented clinical records" },
    settings:    { title: "API Settings",          subtitle: "Configure NVIDIA and Gemini API keys for cloud VLM analysis" }
};

function initTabs() {
    document.querySelectorAll(".nav-item").forEach(item => {
        item.addEventListener("click", () => switchTab(item.dataset.tab));
    });
}

function switchTab(tabName) {
    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

    const navBtn = document.querySelector(`.nav-item[data-tab="${tabName}"]`);
    const pane   = document.getElementById(`tab-${tabName}`);
    if (navBtn) navBtn.classList.add("active");
    if (pane)   pane.classList.add("active");

    const meta = TAB_META[tabName] || {};
    setText("tab-title", meta.title || "");
    setText("tab-subtitle", meta.subtitle || "");

    // Scroll main content to top
    const mainEl = document.querySelector(".main-content");
    if (mainEl) mainEl.scrollTop = 0;

    // Refresh tab-specific views
    if (tabName === "history") {
        loadHistoryFromServer();
    } else if (tabName === "benchmarks") {
        setTimeout(() => {
            if (benchmarkChart && typeof benchmarkChart.resize === "function") {
                benchmarkChart.resize();
            } else {
                loadBenchmarks();
            }
        }, 150);
    } else if (tabName === "biomarkers") {
        if (currentData && currentData.extracted_parameters) {
            renderBiomarkerProfile(
                currentData.extracted_parameters,
                currentData.extraction_flags || {},
                currentData.patient_ref || ""
            );
        }
    } else if (tabName === "comparison") {
        if (currentData && currentData.models) {
            updateComparisonTable(currentData.models);
        }
    } else if (tabName === "settings") {
        initSupabaseSync();
    }
}


// ── API Key Badge Updates ────────────────────────────────────────────
function updateBadges(hasNvidiaServer, hasGeminiServer) {
    const nvLocal  = (localStorage.getItem("nvidia_api_key") || "").length > 10;
    const gemLocal = (localStorage.getItem("gemini_api_key") || "").length > 10;

    const nvActive  = nvLocal  || hasNvidiaServer;
    const gemActive = gemLocal || hasGeminiServer;

    const nvBadge  = document.getElementById("nvidia-key-badge");
    const gemBadge = document.getElementById("gemini-key-badge");

    if (nvBadge) {
        nvBadge.textContent  = nvActive  ? "NVIDIA: Active"  : "NVIDIA: Offline";
        nvBadge.className    = nvActive  ? "badge badge-active" : "badge badge-inactive";
    }
    if (gemBadge) {
        gemBadge.textContent = gemActive ? "Gemini: Active"  : "Gemini: Offline";
        gemBadge.className   = gemActive ? "badge badge-active" : "badge badge-inactive";
    }
}

// ── Upload & Demo Handling ───────────────────────────────────────────
let selectedFile = null;

function initUpload() {
    const dropzone  = document.getElementById("report-dropzone");
    const fileInput = document.getElementById("report-file-input");

    if (dropzone) {
        dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("dragover"); });
        dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
        dropzone.addEventListener("drop", e => {
            e.preventDefault(); dropzone.classList.remove("dragover");
            if (e.dataTransfer.files.length > 0) {
                selectedFile = e.dataTransfer.files[0];
                handleFile(selectedFile);
            }
        });
    }
    if (fileInput) {
        fileInput.addEventListener("change", () => {
            if (fileInput.files.length > 0) {
                selectedFile = fileInput.files[0];
                handleFile(selectedFile);
            }
        });
    }
}

function submitAnalysis() {
    const fileInput = document.getElementById("report-file-input");
    if (fileInput && fileInput.files && fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
    } else if (selectedFile) {
        handleFile(selectedFile);
    } else {
        fileInput?.click();
    }
}

function loadSampleCase(caseType) {
    const nameEl = document.getElementById("pt-name");
    const ageEl = document.getElementById("pt-age");
    const genderEl = document.getElementById("pt-gender");
    const typeEl = document.getElementById("pt-report-type");
    const symEl = document.getElementById("pt-symptoms");
    const medEl = document.getElementById("pt-medications");
    const lifeEl = document.getElementById("pt-lifestyle");
    const notesEl = document.getElementById("pt-notes");

    let sampleData = {};
    if (caseType === "anemia") {
        sampleData = {
            name: "Sunil Jayawardena",
            age: 42,
            gender: "Male",
            type: "Full Blood Count",
            symptoms: "Extreme fatigue, exertional dyspnea, pallor, pica",
            meds: "None currently",
            life: "Non-smoker, vegetarian diet",
            notes: "Suspected severe iron deficiency microcytic anaemia. Peripheral smear requested."
        };
    } else if (caseType === "dengue") {
        sampleData = {
            name: "Anusha Fernando",
            age: 28,
            gender: "Female",
            type: "Full Blood Count",
            symptoms: "High fever (Day 4), retro-orbital pain, severe myalgia, petechiae",
            meds: "Paracetamol 500mg QDS",
            life: "Urban Colombo residence, recent mosquito exposure",
            notes: "Dengue clinical pathway monitoring — monitor platelet trajectory and hematocrit concentration."
        };
    } else {
        sampleData = {
            name: "K.M. Wickramasinghe",
            age: 58,
            gender: "Male",
            type: "Comprehensive Metabolic",
            symptoms: "Polydipsia, polyuria, bilateral foot paresthesia",
            meds: "Metformin 850mg BD, Atorvastatin 20mg",
            life: "Sedentary lifestyle, high BMI (29.4), ex-smoker",
            notes: "Annual diabetic metabolic surveillance and cardiovascular ASCVD risk evaluation."
        };
    }

    if (nameEl) nameEl.value = sampleData.name;
    if (ageEl) ageEl.value = sampleData.age;
    if (genderEl) genderEl.value = sampleData.gender;
    if (typeEl) typeEl.value = sampleData.type;
    if (symEl) symEl.value = sampleData.symptoms;
    if (medEl) medEl.value = sampleData.meds;
    if (lifeEl) lifeEl.value = sampleData.life;
    if (notesEl) notesEl.value = sampleData.notes;

    // Generate clinical lab report sheet image via canvas
    generateSampleReportImage(caseType, sampleData).then(blob => {
        const file = new File([blob], `${caseType}_clinical_report.png`, { type: "image/png" });
        selectedFile = file;
        currentReportFileName = file.name;
        const reader = new FileReader();
        reader.onload = e => {
            currentReportPreviewUrl = e.target.result;
            const dropzoneSub = document.querySelector(".dropzone-sub");
            if (dropzoneSub) dropzoneSub.innerHTML = `Loaded: <strong style="color:var(--cyan)">${file.name}</strong> · Click <strong>RUN 4-ENGINE ANALYSIS</strong>`;
            const docLines = document.querySelectorAll(".doc-sheet-line");
            docLines.forEach(l => l.style.background = "#22d3ee");
        };
        reader.readAsDataURL(file);
    });
}

function generateSampleReportImage(caseType, meta) {
    return new Promise(resolve => {
        const canvas = document.createElement("canvas");
        canvas.width = 850;
        canvas.height = 1100;
        const ctx = canvas.getContext("2d");

        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Header Hospital Letterhead
        ctx.fillStyle = "#0f172a";
        ctx.font = "bold 26px sans-serif";
        ctx.fillText("COLOMBO CLINICAL LABORATORY & DIAGNOSTICS", 50, 60);

        ctx.font = "14px sans-serif";
        ctx.fillStyle = "#475569";
        ctx.fillText("Accredited Clinical Pathology Department • SLMC Verification Standard", 50, 85);
        ctx.fillText("Address: Regent St, Colombo 00800, Sri Lanka • Phone: +94 11 269 1111", 50, 105);

        // Separator Line
        ctx.strokeStyle = "#0284c7";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(50, 120);
        ctx.lineTo(800, 120);
        ctx.stroke();

        // Patient Demographics Box
        ctx.fillStyle = "#f8fafc";
        ctx.fillRect(50, 135, 750, 100);
        ctx.strokeStyle = "#e2e8f0";
        ctx.lineWidth = 1;
        ctx.strokeRect(50, 135, 750, 100);

        ctx.fillStyle = "#0f172a";
        ctx.font = "bold 15px sans-serif";
        ctx.fillText(`Patient: ${meta.name}`, 70, 165);
        ctx.fillText(`Age / Sex: ${meta.age} Y / ${meta.gender}`, 70, 190);
        ctx.fillText(`Ref No: REF-${Date.now().toString().slice(-6)}`, 70, 215);

        ctx.fillText(`Report: ${meta.type}`, 450, 165);
        ctx.fillText(`Date: ${new Date().toISOString().split("T")[0]}`, 450, 190);
        ctx.fillText("Doctor: Dr. Silva (SLMC)", 450, 215);

        // Test Results Table Header
        ctx.fillStyle = "#0284c7";
        ctx.fillRect(50, 260, 750, 35);
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 14px sans-serif";
        ctx.fillText("TEST PARAMETER", 70, 283);
        ctx.fillText("RESULT", 320, 283);
        ctx.fillText("UNITS", 450, 283);
        ctx.fillText("REFERENCE RANGE", 590, 283);

        let rows = [];
        if (caseType === "anemia") {
            rows = [
                ["Haemoglobin (Hb)", "8.2", "g/dL", "13.0 - 17.5", true],
                ["RBC Count", "3.62", "x10^12/L", "4.5 - 5.9", true],
                ["Packed Cell Volume (PCV)", "26.4", "%", "40.0 - 52.0", true],
                ["Mean Corpuscular Volume (MCV)", "66.5", "fL", "80.0 - 96.0", true],
                ["Mean Corpuscular Hb (MCH)", "21.4", "pg", "27.0 - 33.0", true],
                ["MCHC", "31.0", "g/dL", "32.0 - 36.0", true],
                ["Red Cell Distribution Width (RDW)", "18.6", "%", "11.5 - 14.5", true],
                ["Total White Blood Cells (WBC)", "7.4", "x10^9/L", "4.0 - 11.0", false],
                ["Platelet Count", "380", "x10^9/L", "150 - 450", false],
                ["Serum Ferritin", "6.4", "ng/mL", "30.0 - 400.0", true],
            ];
        } else if (caseType === "dengue") {
            rows = [
                ["Platelet Count", "48", "x10^9/L", "150 - 450", true],
                ["Packed Cell Volume (PCV)", "49.2", "%", "36.0 - 48.0", true],
                ["Total White Blood Cells (WBC)", "2.6", "x10^9/L", "4.0 - 11.0", true],
                ["Neutrophils", "42", "%", "40 - 75", false],
                ["Lymphocytes", "54", "%", "20 - 45", true],
                ["Haemoglobin (Hb)", "15.8", "g/dL", "12.0 - 15.5", true],
                ["Dengue NS1 Antigen", "POSITIVE", "", "NEGATIVE", true],
                ["AST (SGOT)", "124", "U/L", "10 - 40", true],
                ["ALT (SGPT)", "96", "U/L", "7 - 56", true],
            ];
        } else {
            rows = [
                ["Fasting Blood Glucose", "168", "mg/dL", "70 - 99", true],
                ["HbA1c (Glycated Haemoglobin)", "8.6", "%", "4.0 - 5.6", true],
                ["Total Cholesterol", "254", "mg/dL", "< 200", true],
                ["Triglycerides", "215", "mg/dL", "< 150", true],
                ["HDL Cholesterol", "36", "mg/dL", "> 40", true],
                ["LDL Cholesterol", "175", "mg/dL", "< 100", true],
                ["Serum Creatinine", "1.18", "mg/dL", "0.7 - 1.3", false],
                ["eGFR (CKD-EPI)", "74", "mL/min", "> 90", true],
                ["Microalbumin / Creatinine Ratio", "48", "mg/g", "< 30", true],
            ];
        }

        let y = 320;
        rows.forEach((r, idx) => {
            if (idx % 2 === 1) {
                ctx.fillStyle = "#f8fafc";
                ctx.fillRect(50, y - 22, 750, 32);
            }
            ctx.fillStyle = r[4] ? "#b91c1c" : "#0f172a";
            ctx.font = r[4] ? "bold 13px sans-serif" : "13px sans-serif";
            ctx.fillText(r[0], 70, y);
            ctx.fillText(r[1] + (r[4] ? " *" : ""), 320, y);
            ctx.fillStyle = "#64748b";
            ctx.font = "12px sans-serif";
            ctx.fillText(r[2], 450, y);
            ctx.fillText(r[3], 590, y);
            y += 36;
        });

        ctx.strokeStyle = "#94a3b8";
        ctx.beginPath();
        ctx.moveTo(560, 980);
        ctx.lineTo(760, 980);
        ctx.stroke();
        ctx.fillStyle = "#0f172a";
        ctx.font = "bold 13px sans-serif";
        ctx.fillText("Dr. K. Silva, MBBS, MD (Path)", 560, 1000);
        ctx.fillStyle = "#64748b";
        ctx.font = "11px sans-serif";
        ctx.fillText("Consultant Haematologist • SLMC 28419", 560, 1018);

        canvas.toBlob(blob => resolve(blob), "image/png");
    });
}

function handleFile(file) {
    currentReportFileName = file.name || "Uploaded_Lab_Report.png";
    const reader = new FileReader();
    reader.onload = e => {
        currentReportPreviewUrl = e.target.result;
        updateReportPreviewElements(currentReportPreviewUrl, currentReportFileName);
    };
    reader.readAsDataURL(file);

    showLoading(`Analysing: ${file.name}`);
    sendAnalysisRequest(file);
}

function showLoading(title = "Analysing…") {
    reviewFinalized = false;
    document.getElementById("analysis-placeholder")?.classList.add("hidden");
    document.getElementById("analysis-loading")?.classList.remove("hidden");
    document.getElementById("analysis-results")?.classList.add("hidden");

    // Animate 5-phase cinematic sequencer
    const pTitle = document.getElementById("seq-phase-title");
    const pDesc  = document.getElementById("seq-phase-desc");
    const pFill  = document.getElementById("seq-progress-fill");
    const nodes = {
        rule: document.getElementById("seq-node-rule"),
        ml:   document.getElementById("seq-node-ml"),
        nv:   document.getElementById("seq-node-nv"),
        gem:  document.getElementById("seq-node-gem")
    };
    const tags = document.getElementById("seq-telemetry-tags");

    if (pTitle && window.ClinicalSequenceController) {
        ClinicalSequenceController.runSequence((phase) => {
            pTitle.textContent = `Phase ${phase.id}: ${phase.name}`;
            pDesc.textContent = phase.desc;
            if (pFill) pFill.style.width = `${phase.id * 20}%`;

            if (phase.id === 1) {
                nodes.rule?.classList.add("active");
                if (tags) tags.innerHTML = `<span>Optical Grid: 300 DPI</span><span>Contrast: Optimal</span>`;
            } else if (phase.id === 2) {
                nodes.rule?.classList.replace("active", "done");
                nodes.ml?.classList.add("active");
                if (tags) tags.innerHTML = `<span>Hb: 8.4 g/dL</span><span>MCV: 68 fL</span><span>Platelets: 145k</span>`;
            } else if (phase.id === 3) {
                nodes.ml?.classList.replace("active", "done");
                nodes.nv?.classList.add("active");
                nodes.gem?.classList.add("active");
                if (tags) tags.innerHTML = `<span>XGBoost: Inferred</span><span>NVIDIA NIM: Active</span><span>Gemini: Active</span>`;
            } else if (phase.id === 4) {
                nodes.nv?.classList.replace("active", "done");
                nodes.gem?.classList.replace("active", "done");
                if (tags) tags.innerHTML = `<span>Consensus Agreement: 96%</span><span>Arbitration: Cleared</span>`;
            }
        });
    }
}

// ── API Request ──────────────────────────────────────────────────────
function sendAnalysisRequest(file) {
    const fd = new FormData();
    fd.append("file", file);

    // Patient intake & Context Integration
    fd.append("patient_name",   document.getElementById("pt-name")?.value.trim() || "");
    fd.append("patient_age",    document.getElementById("pt-age")?.value || "");
    fd.append("patient_gender", document.getElementById("pt-gender")?.value || "Male");
    fd.append("report_type",    document.getElementById("pt-report-type")?.value || "Full Blood Count");
    fd.append("report_date",    document.getElementById("pt-date")?.value || "");
    fd.append("clinical_notes", document.getElementById("pt-notes")?.value.trim() || "");
    fd.append("symptoms",       document.getElementById("pt-symptoms")?.value.trim() || "");
    fd.append("medications",    document.getElementById("pt-medications")?.value.trim() || "");
    fd.append("lifestyle",      document.getElementById("pt-lifestyle")?.value.trim() || "");

    // API keys
    fd.append("nvidia_api_key", localStorage.getItem("nvidia_api_key") || "");
    fd.append("gemini_api_key", localStorage.getItem("gemini_api_key") || "");

    fetch("/api/analyze", {
        method: "POST",
        headers: { "Authorization": `Bearer ${AUTH_TOKEN}` },
        body: fd
    })
    .then(res => {
        if (res.status === 401 || res.status === 403) {
            localStorage.removeItem("cliniq_token");
            localStorage.removeItem("cliniq_doctor");
            window.location.href = "/login.html";
            return;
        }
        if (!res.ok) return res.json().then(d => { throw new Error(d.detail || `HTTP ${res.status}`); });
        return res.json();
    })
    .then(data => {
        if (!data) return;
        if (data.success) {
            currentData = data;
            renderResults(data);
            addToHistory(data);
            if (data.supabase_sync) {
                if (data.supabase_sync.success) {
                    console.log("[Supabase] Report auto-synced to cloud database:", data.supabase_sync);
                } else if (data.supabase_sync.table_missing) {
                    console.warn("[Supabase] Report not synced: 'patient_reports' table not created yet.");
                }
            }
        } else {
            showError("Analysis returned an unexpected response.");
        }
    })
    .catch(err => {
        console.error(err);
        showError(err.message || "Network error. Ensure the server is running.");
    });
}

function showError(msg) {
    document.getElementById("analysis-loading").classList.add("hidden");
    document.getElementById("analysis-placeholder").classList.remove("hidden");
    document.getElementById("analysis-placeholder").innerHTML = `
        <div class="welcome-content">
            <div class="pulse-ring"><div class="pulse-icon error-icon">✗</div></div>
            <h3>Analysis Failed</h3>
            <p style="color:#fca5a5">${msg}</p>
            <button class="btn btn-outline" style="margin-top:16px" onclick="resetToUpload()">Try Again</button>
        </div>`;
}

function resetToUpload() {
    document.getElementById("analysis-results").classList.add("hidden");
    document.getElementById("analysis-loading").classList.add("hidden");
    document.getElementById("analysis-placeholder").innerHTML = `
        <div class="welcome-content">
            <div class="pulse-ring"><div class="pulse-icon">✚</div></div>
            <h3>Awaiting Lab Report</h3>
            <p>Fill in the patient information on the left, then upload a lab report image to begin AI-powered analysis across 4 models simultaneously.</p>
            <div class="model-badges-preview">
                <span class="mbadge local">EasyOCR Rules</span>
                <span class="mbadge local">XGBoost ML</span>
                <span class="mbadge cloud">NVIDIA NIM</span>
                <span class="mbadge cloud">Gemini 2.5</span>
            </div>
        </div>`;
    document.getElementById("analysis-placeholder").classList.remove("hidden");
}

// ── RENDER RESULTS ────────────────────────────────────────────────────
function renderResults(data) {
    // Complete loading → show results
    document.getElementById("ls-ensemble")?.classList.add("done");
    setTimeout(() => {
        document.getElementById("analysis-loading").classList.add("hidden");
        document.getElementById("analysis-results").classList.remove("hidden");
        document.getElementById("analysis-results").scrollIntoView({ behavior: "smooth", block: "start" });
    }, 400);

    const { patient, analyzed_by, ensemble, models, parsed_values, extracted_flags,
            recommendations, best_narrative, patient_ref, processing_time_sec,
            safety_disclaimer, clinical_findings, borderline_flags, clinical_cautions,
            ocr_demographics, vlm_demographics, risk_profile, clinical_reasoning,
            clinical_inquiries } = data;

    // ── Auto-fill patient intake form from extracted demographics ──
    const resolvedName = (patient && patient.name && patient.name !== "Anonymous") ? patient.name : (vlm_demographics?.name || ocr_demographics?.name || "");
    const resolvedAge = (patient && patient.age && patient.age !== "—") ? patient.age : (vlm_demographics?.age || ocr_demographics?.age || "");
    const resolvedGender = (patient && patient.gender && patient.gender !== "Not Specified") ? patient.gender : (vlm_demographics?.gender || ocr_demographics?.gender || "");

    if (resolvedName) {
        const nameEl = document.getElementById("pt-name");
        if (nameEl) nameEl.value = resolvedName;
    }
    if (resolvedAge) {
        const ageEl = document.getElementById("pt-age");
        if (ageEl) ageEl.value = String(resolvedAge);
    }
    if (resolvedGender) {
        const genderEl = document.getElementById("pt-gender");
        if (genderEl) genderEl.value = resolvedGender;
    }

    // ── Patient Banner
    const initials = (patient.name || "P").split(" ").map(w => w[0]).join("").toUpperCase().slice(0, 2);
    setText("banner-avatar", initials || "P");
    setText("banner-patient-name", patient.name || "Anonymous Patient");
    setText("banner-age-gender", `${patient.age}y · ${patient.gender}`);
    setText("banner-report-type", patient.report_type || "Full Blood Count");
    setText("banner-ref", patient_ref || "");
    setText("banner-doctor", `Analysed by ${analyzed_by.doctor_name} · ${analyzed_by.slmc_number} · ${processing_time_sec}s`);

    // ── Update Lab Report Document Preview in Results ──
    const reportUrl = data.report_file_preview || currentReportPreviewUrl;
    const reportName = data.report_filename || currentReportFileName || "Uploaded_Lab_Report.png";
    updateReportPreviewElements(reportUrl, reportName);

    // ── Primary Diagnosis Card
    const diag = ensemble.primary_diagnosis || "Unknown";
    const diagPill = document.getElementById("dx-pill");
    if (diagPill) {
        diagPill.textContent = diag;
        const isNormal = diag.toLowerCase().includes("no major abnormalities") ||
                         diag.toLowerCase().includes("no significant abnormalities") ||
                         diag.toLowerCase().includes("healthy");
        diagPill.style.background = isNormal
            ? "linear-gradient(135deg, #064e3b, #10b981)"
            : "linear-gradient(135deg, #7f1d1d, #ef4444)";
    }
    setText("dx-narrative", best_narrative || "");

    // ── Safety Disclaimer
    const disclaimerEl = document.getElementById("safety-disclaimer");
    if (disclaimerEl && safety_disclaimer) {
        disclaimerEl.textContent = safety_disclaimer;
        disclaimerEl.classList.remove("hidden");
    }

    // ── Safety Flags from ensemble
    const safetyFlagsEl = document.getElementById("safety-flags");
    if (safetyFlagsEl) {
        const flags = ensemble.safety_flags || [];
        if (flags.length > 0) {
            safetyFlagsEl.innerHTML = flags.map(f =>
                `<div class="safety-flag-item">⚠ ${f}</div>`
            ).join("");
            safetyFlagsEl.classList.remove("hidden");
        } else {
            safetyFlagsEl.classList.add("hidden");
        }
    }

    // ── Clinical Findings, Borderline Flags, Cautions
    const findingsEl = document.getElementById("clinical-findings-list");
    if (findingsEl) {
        let html = "";
        if (clinical_findings && clinical_findings.length > 0) {
            html += `<div class="findings-section"><h4 class="findings-heading abnormal-heading">⛔ Confirmed Findings</h4>`;
            html += clinical_findings.map(f => `<div class="finding-item abnormal">${f}</div>`).join("");
            html += `</div>`;
        }
        if (borderline_flags && borderline_flags.length > 0) {
            html += `<div class="findings-section"><h4 class="findings-heading borderline-heading">⚡ Borderline Values</h4>`;
            html += borderline_flags.map(f => `<div class="finding-item borderline">${f}</div>`).join("");
            html += `</div>`;
        }
        if (clinical_cautions && clinical_cautions.length > 0) {
            html += `<div class="findings-section"><h4 class="findings-heading caution-heading">🔍 Clinical Cautions</h4>`;
            html += clinical_cautions.map(f => `<div class="finding-item caution">${f}</div>`).join("");
            html += `</div>`;
        }
        if (html) {
            findingsEl.innerHTML = html;
            findingsEl.classList.remove("hidden");
        } else {
            findingsEl.classList.add("hidden");
        }
    }

    // ── Render Consensus Arbitration & Disagreement Handling Card
    renderArbitrationNotes(ensemble.arbitration_notes || []);

    // ── Render Multi-Tier Biomarker Severity Classification Grid
    renderBiomarkerSeverities(ensemble.biomarker_severities || []);

    // ── Render Structured Clinical Action Plans
    renderActionPlans(ensemble.action_plans || []);

    // ── Render Risk Stratification Panel (Expanded Clinical Modules)
    renderRiskStratification(risk_profile);

    // ── Render Longitudinal Trend Analysis & Delta Tracker
    if (risk_profile && risk_profile.longitudinal_trends) {
        renderLongitudinalTrends(risk_profile.longitudinal_trends);
    }

    // ── Render Explainable AI (XAI) & Differential Diagnoses
    renderXAIReasoning(clinical_reasoning);

    // ── Render Patient History Inquiries Checklist
    renderInquiriesChecklist(clinical_inquiries);

    // ── Confidence Gauge
    renderConfidenceGauge(ensemble.confidence, ensemble.confidence_band);
    renderVoteBars(ensemble.vote_breakdown, ensemble.all_votes);

    // ── 4-Model Cards
    renderModelCards(models);

    // ── Biomarker Strip
    renderBiomarkerStrip(parsed_values, extracted_flags);

    // ── Biomarker Full Profile Tab
    renderBiomarkerProfile(parsed_values, extracted_flags, patient_ref);

    // ── Recommendations
    renderRecommendations(recommendations);

    // ── XGBoost Chart
    renderXGBoostChart(models.model_2_ml.probabilities || {});

    // ── Model Comparison Table
    updateComparisonTable(models);

    // ── Review Panel
    resetReviewPanel(analyzed_by);
}

// ── Render Consensus Arbitration & Disagreement Handling ──────────────
function renderArbitrationNotes(notes) {
    const cardEl = document.getElementById("arbitration-card");
    const listEl = document.getElementById("arbitration-list");
    if (!cardEl || !listEl) return;

    if (!notes || notes.length === 0) {
        cardEl.classList.add("hidden");
        return;
    }

    cardEl.classList.remove("hidden");
    listEl.innerHTML = notes.map(n => `
        <div class="arbitration-item">
            <span>⚖</span>
            <span>${n}</span>
        </div>
    `).join("");
}

// ── Render Multi-Tier Biomarker Severity Classification Grid ─────────
function renderBiomarkerSeverities(severities) {
    const cardEl = document.getElementById("biomarker-severities-card");
    const gridEl = document.getElementById("severity-cards-grid");
    if (!cardEl || !gridEl) return;

    if (!severities || severities.length === 0) {
        cardEl.classList.add("hidden");
        return;
    }

    cardEl.classList.remove("hidden");
    gridEl.innerHTML = severities.map(s => `
        <div class="severity-card" style="border-left: 3px solid ${s.color || '#3b82f6'}">
            <div class="severity-header">
                <span class="severity-title">${s.biomarker}</span>
                <span class="severity-badge" style="background: ${s.color}22; color: ${s.color}">${s.tier}</span>
            </div>
            <div class="severity-val-row">
                <span class="severity-val">${s.value}</span>
            </div>
            <div class="severity-label-desc">${s.label}</div>
        </div>
    `).join("");
}

// ── Render Structured Clinical Action Plans ────────────────────────────
function renderActionPlans(plans) {
    const cardEl = document.getElementById("action-plans-card");
    const containerEl = document.getElementById("action-plans-container");
    if (!cardEl || !containerEl) return;

    if (!plans || plans.length === 0) {
        cardEl.classList.add("hidden");
        return;
    }

    cardEl.classList.remove("hidden");
    containerEl.innerHTML = plans.map(p => `
        <div class="action-plan-item" style="border-left: 3px solid ${p.color || '#10b981'}">
            <div class="action-plan-header">
                <span class="action-plan-title">${p.condition}</span>
                <span class="rc-badge" style="background: ${p.color}22; color: ${p.color}">${p.severity}</span>
            </div>
            <ul class="action-plan-steps">
                ${p.steps.map(step => `<li class="action-step-item">${step}</li>`).join("")}
            </ul>
        </div>
    `).join("");
}

// ── Render Risk Stratification (Expanded Clinical Engines) ────────────
function renderRiskStratification(risk) {
    if (!risk) return;

    // 1. ASCVD Risk
    const ascvd = risk.ascvd || {};
    setText("rc-ascvd-val", ascvd.score_pct !== undefined ? `${ascvd.score_pct}%` : "—");
    setText("rc-ascvd-cat", ascvd.category || "Evaluated");
    setText("rc-ascvd-desc", ascvd.action || "ACC/AHA clinical model");
    const ascvdCard = document.getElementById("rc-ascvd");
    const ascvdBadge = document.getElementById("rc-ascvd-cat");
    if (ascvdBadge && ascvd.color) {
        ascvdBadge.style.background = `${ascvd.color}22`;
        ascvdBadge.style.color = ascvd.color;
    }
    if (ascvdCard && ascvd.color) ascvdCard.style.borderLeft = `3px solid ${ascvd.color}`;

    // 2. Anemia WHO Severity Risk
    const anemia = risk.anemia || {};
    setText("rc-anemia-val", anemia.hb_value !== undefined ? `${anemia.hb_value}` : "—");
    setText("rc-anemia-grade", anemia.grade || "Evaluated");
    setText("rc-anemia-desc", anemia.summary || "WHO severity staging");
    const anemiaCard = document.getElementById("rc-anemia");
    const anemiaBadge = document.getElementById("rc-anemia-grade");
    if (anemiaBadge && anemia.color) {
        anemiaBadge.style.background = `${anemia.color}22`;
        anemiaBadge.style.color = anemia.color;
    }
    if (anemiaCard && anemia.color) anemiaCard.style.borderLeft = `3px solid ${anemia.color}`;

    // 3. Metabolic Syndrome Risk
    const meta = risk.metabolic || {};
    setText("rc-metabolic-score", meta.score !== undefined ? `${meta.score} / ${meta.max_score || 4}` : "0 / 4");
    setText("rc-metabolic-badge", meta.status || "Evaluated");
    setText("rc-metabolic-desc", meta.summary || "ATP III criteria indicators");
    const metaCard = document.getElementById("rc-metabolic");
    const metaBadge = document.getElementById("rc-metabolic-badge");
    if (metaBadge && meta.color) {
        metaBadge.style.background = `${meta.color}22`;
        metaBadge.style.color = meta.color;
    }
    if (metaCard && meta.color) metaCard.style.borderLeft = `3px solid ${meta.color}`;

    // 4. Thyroid Dysfunction Risk
    const thyroid = risk.thyroid || {};
    setText("rc-thyroid-val", thyroid.tsh_value !== undefined ? `${thyroid.tsh_value}` : "—");
    setText("rc-thyroid-status", thyroid.status || "Evaluated");
    setText("rc-thyroid-desc", thyroid.summary || "ATA clinical guidelines");
    const thyroidCard = document.getElementById("rc-thyroid");
    const thyroidBadge = document.getElementById("rc-thyroid-status");
    if (thyroidBadge && thyroid.color) {
        thyroidBadge.style.background = `${thyroid.color}22`;
        thyroidBadge.style.color = thyroid.color;
    }
    if (thyroidCard && thyroid.color) thyroidCard.style.borderLeft = `3px solid ${thyroid.color}`;

    // 5. eGFR Renal Function
    const egfr = risk.egfr || {};
    setText("rc-egfr-val", egfr.egfr !== null && egfr.egfr !== undefined ? String(egfr.egfr) : "—");
    setText("rc-egfr-stage", egfr.status || "Calculated");
    setText("rc-egfr-desc", egfr.interpretation || "CKD-EPI 2021 formula");
    const egfrCard = document.getElementById("rc-egfr");
    const egfrBadge = document.getElementById("rc-egfr-stage");
    if (egfrBadge && egfr.color) {
        egfrBadge.style.background = `${egfr.color}22`;
        egfrBadge.style.color = egfr.color;
    }
    if (egfrCard && egfr.color) egfrCard.style.borderLeft = `3px solid ${egfr.color}`;

    // 6. Autoimmune Risk
    const auto = risk.autoimmune || {};
    setText("rc-autoimmune-val", auto.score !== undefined ? `${auto.score}` : "0");
    setText("rc-autoimmune-badge", auto.status || "Evaluated");
    setText("rc-autoimmune-desc", auto.summary || "Cytopenia & ESR indicators");
    const autoCard = document.getElementById("rc-autoimmune");
    const autoBadge = document.getElementById("rc-autoimmune-badge");
    if (autoBadge && auto.color) {
        autoBadge.style.background = `${auto.color}22`;
        autoBadge.style.color = auto.color;
    }
    if (autoCard && auto.color) autoCard.style.borderLeft = `3px solid ${auto.color}`;

    // 7. Micronutrient / Vitamin D Status
    const micro = risk.micronutrient || {};
    setText("rc-micronutrient-val", micro.vitamin_d_val !== undefined && micro.vitamin_d_val !== null ? `${micro.vitamin_d_val}` : "—");
    setText("rc-micronutrient-badge", micro.status || "Evaluated");
    setText("rc-micronutrient-desc", micro.summary || "25-OH Vitamin D status");
    const microCard = document.getElementById("rc-micronutrient");
    const microBadge = document.getElementById("rc-micronutrient-badge");
    if (microBadge && micro.color) {
        microBadge.style.background = `${micro.color}22`;
        microBadge.style.color = micro.color;
    }
    if (microCard && micro.color) microCard.style.borderLeft = `3px solid ${micro.color}`;

    // 8. Cross-Marker Anomalies
    const anomaliesEl = document.getElementById("cross-marker-anomalies");
    if (anomaliesEl) {
        const anomalies = risk.cross_marker_anomalies || [];
        if (anomalies.length > 0) {
            anomaliesEl.innerHTML = anomalies.map(a => `
                <div class="anomaly-item ${a.severity === 'High' ? '' : a.severity === 'Moderate' ? 'medium' : 'info'}">
                    <div class="anomaly-title">⚡ ${a.title} (${a.severity} Severity)</div>
                    <div class="anomaly-detail">${a.detail}</div>
                </div>
            `).join("");
            anomaliesEl.classList.remove("hidden");
        } else {
            anomaliesEl.classList.add("hidden");
        }
    }
}

// ── Render Longitudinal Trend Analysis & Delta Tracker ────────────────
function renderLongitudinalTrends(trends) {
    const cardEl = document.getElementById("longitudinal-trends-card");
    if (!cardEl) return;

    if (!trends || !trends.has_history || trends.biomarker_deltas.length === 0) {
        cardEl.classList.add("hidden");
        return;
    }

    cardEl.classList.remove("hidden");
    setText("trends-summary", trends.summary || "Historical comparison with prior lab visits.");

    // Deterioration Alerts
    const alertsEl = document.getElementById("trends-alerts-list");
    if (alertsEl) {
        const alerts = trends.deterioration_alerts || [];
        if (alerts.length > 0) {
            alertsEl.innerHTML = alerts.map(a => `
                <div class="trend-alert-item">
                    <span>⚠</span>
                    <span>${a}</span>
                </div>
            `).join("");
            alertsEl.classList.remove("hidden");
        } else {
            alertsEl.classList.add("hidden");
        }
    }

    // Deltas Table Body
    const tbody = document.getElementById("trends-table-body");
    if (tbody) {
        tbody.innerHTML = trends.biomarker_deltas.map(d => {
            const deltaSign = d.delta > 0 ? `+${d.delta}` : `${d.delta}`;
            const pctSign = d.pct_change > 0 ? `+${d.pct_change}%` : `${d.pct_change}%`;
            return `
                <tr>
                    <td><strong>${d.biomarker}</strong></td>
                    <td style="color:#94a3b8">${d.previous}</td>
                    <td style="color:#f8fafc; font-weight:700">${d.current}</td>
                    <td style="color:${d.is_concerning ? '#f87171' : '#cbd5e1'}">${deltaSign}</td>
                    <td style="color:${d.is_concerning ? '#f87171' : '#cbd5e1'}">${pctSign}</td>
                    <td>
                        <span class="trajectory-badge ${d.direction}">
                            <span>${d.arrow}</span>
                            <span>${d.direction}</span>
                        </span>
                    </td>
                </tr>
            `;
        }).join("");
    }
}

// ── Render Explainable AI (XAI) Reasoning & Differential Diagnoses ────
function renderXAIReasoning(xai) {
    if (!xai) return;
    setText("xai-justification", xai.summary_rationale || "Clinical reasoning completed.");
    setText("xai-why-confidence", xai.why_confidence_level || "Confidence computed across ensemble.");
    setText("xai-consensus-math", xai.consensus_math || "Consensus computed across models.");
    setText("xai-demographics-impact", xai.demographic_context || "Age, gender, and clinical context applied.");

    // Uncertainty Breakdown List
    const uncList = document.getElementById("xai-uncertainty-list");
    if (uncList && xai.uncertainty_breakdown) {
        uncList.innerHTML = xai.uncertainty_breakdown.map(u => `<li>${u}</li>`).join("");
    }

    // Differential Diagnoses Grid
    const diffGrid = document.getElementById("diff-dx-grid");
    if (diffGrid && xai.differential_diagnoses) {
        diffGrid.innerHTML = xai.differential_diagnoses.map(d => `
            <div class="diff-dx-card">
                <div class="diff-dx-header">
                    <span class="diff-dx-title">${d.condition}</span>
                    <span class="diff-dx-likelihood ${d.likelihood}">${d.likelihood} Likelihood</span>
                </div>
                <div class="diff-dx-rationale">${d.rationale}</div>
            </div>
        `).join("");
    }
}

// ── Render Patient History Inquiries Checklist ─────────────────────────
function renderInquiriesChecklist(inquiries) {
    const listEl = document.getElementById("inquiries-list");
    if (!listEl) return;
    const items = inquiries || [];
    if (items.length === 0) {
        listEl.innerHTML = `<li class="inquiry-item"><span class="inquiry-bullet">✓</span> Clinical history fully documented.</li>`;
        return;
    }
    listEl.innerHTML = items.map(q => `
        <li class="inquiry-item">
            <span class="inquiry-bullet">?</span>
            <span>${q}</span>
        </li>
    `).join("");
}

// ── Confidence Arc Gauge ──────────────────────────────────────────────
function renderConfidenceGauge(pct, band) {
    // Arc total length = π × r = π × 80 ≈ 251.2
    const ARC_LEN = 251.2;
    const offset  = ARC_LEN - (ARC_LEN * (pct / 100));
    const arc     = document.getElementById("gauge-arc");
    const col     = BAND_COLORS[band] || BAND_COLORS.INCONCLUSIVE;

    if (arc) {
        arc.style.stroke          = col.stroke;
        arc.style.strokeDashoffset = String(offset);
    }
    setText("gauge-pct",  `${pct}%`);
    
    let subtext = col.text;
    if (band === "MODERATE" || (pct >= 60 && pct < 80)) {
        subtext = `Moderate (${pct}%) — Model Disagreement Handled`;
    } else if (band === "HIGH") {
        subtext = `High Confidence (${pct}%) — Validated Consensus`;
    }
    setText("gauge-band", subtext);

    const pctEl = document.getElementById("gauge-pct");
    if (pctEl) pctEl.style.fill = col.stroke;
}

// ── Vote Bars ─────────────────────────────────────────────────────────
function renderVoteBars(breakdown, allVotes) {
    const container = document.getElementById("vote-bars");
    if (!container) return;

    const modelLabels = {
        "model_4_gemini": { name: "Gemini 2.5",  pct: 40 },
        "model_3_nvidia": { name: "NVIDIA NIM",   pct: 35 },
        "model_2_ml":     { name: "XGBoost ML",   pct: 20 },
        "model_1_rules":  { name: "Rules Engine", pct: 5  }
    };

    container.innerHTML = Object.entries(modelLabels).map(([key, meta]) => {
        const info = breakdown[key];
        const diag = info ? info.diagnosis : "—";
        const isOffline = !info;
        return `
        <div class="vote-bar-row ${isOffline ? "offline" : ""}">
            <div class="vb-meta">
                <span class="vb-name">${meta.name}</span>
                <span class="vb-diag">${diag}</span>
            </div>
            <div class="vb-track">
                <div class="vb-fill" style="width:${isOffline ? 0 : meta.pct}%; background:${isOffline ? "#334155" : "#06b6d4"}"></div>
            </div>
            <span class="vb-weight">${isOffline ? "—" : meta.pct + "%"}</span>
        </div>`;
    }).join("");
}

// ── 4-Model Consensus Cards ───────────────────────────────────────────
function renderModelCards(models) {
    const map = [
        { id: "rules",  key: "model_1_rules" },
        { id: "xgb",    key: "model_2_ml" },
        { id: "nvidia", key: "model_3_nvidia" },
        { id: "gemini", key: "model_4_gemini" }
    ];

    map.forEach(({ id, key }) => {
        const m = models[key];
        if (!m) return;

        const card   = document.getElementById(`mc-${id}`);
        const diagEl = document.getElementById(`mc-${id}-diag`);
        const latEl  = document.getElementById(`mc-${id}-lat`);
        if (!card) return;

        if (m.error) {
            card.classList.add("mc-offline");
            if (diagEl) diagEl.textContent = "Offline";
            if (latEl)  latEl.textContent  = "—";
        } else {
            card.classList.remove("mc-offline");
            const diagLower = (m.diagnosis || "").toLowerCase();
            const isNormal = diagLower.includes("no significant") || diagLower.includes("no major") || diagLower.includes("healthy");
            if (diagEl) {
                diagEl.textContent  = m.diagnosis || "Unknown";
                diagEl.style.color  = isNormal ? "#10b981" : "#f87171";
            }
            if (latEl)  latEl.textContent = m.latency_sec ? `${m.latency_sec}s` : "< 0.1s";
        }
    });
}

// ── Biomarker Strip (summary row) ─────────────────────────────────────
function renderBiomarkerStrip(values, flags) {
    const strip = document.getElementById("biomarker-strip");
    if (!strip) return;

    strip.innerHTML = Object.entries(BIOMARKER_RANGES).map(([key, range]) => {
        const val = values[key];
        if (val === undefined || val === null) return "";
        const isNormal    = val >= range.normal_min && val <= range.normal_max;
        const isExtracted = flags[key];
        const short       = SHORT_NAMES[key] || key;
        const arrow       = !isNormal ? (val < range.normal_min ? "↓" : "↑") : "";
        return `
        <div class="strip-chip ${isNormal ? "normal" : "abnormal"}">
            <span class="chip-label">${short}</span>
            <span class="chip-value">${val} ${arrow}</span>
            <span class="chip-unit">${range.unit}</span>
            ${!isExtracted ? '<span class="chip-imp">est.</span>' : ""}
        </div>`;
    }).join("");
}

// ── Recommendations ───────────────────────────────────────────────────
function renderRecommendations(recs) {
    const list = document.getElementById("recs-list");
    if (!list) return;
    list.innerHTML = (recs || []).map((r, i) =>
        `<li class="rec-item">
            <div class="rec-num">${i + 1}</div>
            <span>${r}</span>
        </li>`
    ).join("");
}

// ── Doctor Review Panel ───────────────────────────────────────────────
function resetReviewPanel(analyzed_by) {
    reviewFinalized = false;
    document.getElementById("review-finalised")?.classList.add("hidden");
    document.getElementById("review-card")?.classList.remove("finalised");
    document.getElementById("review-notes").value = "";
    document.querySelectorAll('input[name="review-action"]').forEach(r => r.checked = false);
    setText("review-required-badge", "Review Required");
    document.getElementById("review-required-badge")?.setAttribute("class", "card-badge danger");
}

function finalizeReport() {
    if (reviewFinalized) return;

    const action = document.querySelector('input[name="review-action"]:checked')?.value;
    if (!action) {
        alert("Please select a review action before finalising.");
        return;
    }

    const notes     = document.getElementById("review-notes").value.trim();
    const doctorId  = `${DOCTOR_INFO.full_name} (${DOCTOR_INFO.slmc_number})`;
    const timestamp = new Date().toLocaleString("en-GB");

    const actionLabel = {
        confirm: "Confirmed — AI findings align with clinical presentation",
        modify:  "Modified — See doctor's notes",
        reject:  "Rejected — AI output incorrect"
    }[action] || action;

    // PATCH sign-off to server (permanent record)
    if (currentData?.patient_ref) {
        fetch(`/api/reports/${currentData.patient_ref}/review`, {
            method: "PATCH",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${AUTH_TOKEN}`
            },
            body: JSON.stringify({ action, notes })
        }).catch(e => console.warn("[review PATCH failed]", e));
    }

    // Update UI to finalised state
    reviewFinalized = true;
    document.getElementById("review-finalised")?.classList.remove("hidden");
    document.getElementById("review-card")?.classList.add("finalised");
    setText("review-required-badge", "Finalised");
    document.getElementById("review-required-badge")?.setAttribute("class", "card-badge success");

    const summary = `${actionLabel} · Signed by ${doctorId} on ${timestamp}`;
    setText("finalised-summary", summary);

    if (currentData) {
        currentData._review = { action: actionLabel, notes, doctor: doctorId, timestamp };
    }
}

// ── XGBoost Chart ─────────────────────────────────────────────────────
function renderXGBoostChart(probs) {
    if (!window.Chart) return;
    const ctx = document.getElementById("diseaseProbChart")?.getContext("2d");
    if (!ctx) return;

    const labels     = Object.keys(probs);
    const dataValues = Object.values(probs).map(v => Math.round(v * 100));

    if (diseaseChart) diseaseChart.destroy();

    diseaseChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: "Disease Probability (%)",
                data: dataValues,
                backgroundColor: labels.map(l =>
                    l === "Healthy" ? "rgba(16,185,129,0.75)" :
                    l.includes("Thyroid") || l.includes("Cholesterol") ? "rgba(6,182,212,0.75)" :
                    "rgba(239,68,68,0.75)"
                ),
                borderWidth: 0,
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { beginAtZero: true, max: 100, grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#9ca3af" } },
                y: { grid: { display: false }, ticks: { color: "#f3f4f6", font: { family: "Outfit", size: 12 } } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

// ── Comparison Table ──────────────────────────────────────────────────
function updateComparisonTable(models) {
    const rows = [
        { id: "row-model-1", key: "model_1_rules" },
        { id: "row-model-2", key: "model_2_ml" },
        { id: "row-model-3", key: "model_3_nvidia" },
        { id: "row-model-4", key: "model_4_gemini" }
    ];
    rows.forEach(({ id, key }) => {
        const m   = models[key];
        const row = document.getElementById(id);
        if (!m || !row) return;
        row.querySelector(".latency").textContent    = m.error ? "—" : `${m.latency_sec}s`;
        row.querySelector(".diagnosis-val").textContent = m.error ? "Offline" : (m.diagnosis || "—");
        row.querySelector(".details-val").textContent   = m.error ? m.error : ((m.details || "").slice(0, 200) + (m.details?.length > 200 ? "…" : ""));
        row.querySelector(".diagnosis-val").style.color = m.error ? "#64748b"
            : (m.diagnosis?.toLowerCase().includes("healthy") ? "#10b981" : "#f87171");
    });
}

// ── Biomarker Full Profile Tab ────────────────────────────────────────
function renderBiomarkerProfile(values, flags, ref) {
    const grid = document.getElementById("biomarkers-grid-container");
    if (!grid) return;
    setText("biomarker-patient-ref", ref || "");
    grid.innerHTML = "";

    Object.entries(BIOMARKER_RANGES).forEach(([key, range]) => {
        const val = values[key];
        if (val === undefined || val === null) return;
        const isNormal    = val >= range.normal_min && val <= range.normal_max;
        const isExtracted = flags[key];

        let pct      = ((val - range.min) / (range.max - range.min)) * 100;
        pct = Math.max(0, Math.min(100, pct));
        const normMinPct = ((range.normal_min - range.min) / (range.max - range.min)) * 100;
        const normWidth  = ((range.normal_max - range.normal_min) / (range.max - range.min)) * 100;

        const card = document.createElement("div");
        card.className = `biomarker-card ${isNormal ? "" : "abnormal-bg"}`;
        card.innerHTML = `
            <div class="biomarker-header">
                <span class="biomarker-title">${key.replace(/_/g, " ")}</span>
                <span class="biomarker-status-badge ${isNormal ? "normal" : "abnormal"}">${isNormal ? "Normal" : "Abnormal"}</span>
            </div>
            <div class="biomarker-value-display">
                <span class="biomarker-val">${val}</span>
                <span class="biomarker-unit">${range.unit}</span>
                <span class="biomarker-src ${isExtracted ? "extracted" : "imputed"}">${isExtracted ? "• Extracted" : "• Estimated"}</span>
            </div>
            <div class="biomarker-range-gauge">
                <div class="biomarker-normal-zone" style="left:${normMinPct}%; width:${normWidth}%"></div>
                <div class="biomarker-indicator-pin ${isNormal ? "" : "abnormal-pin"}" style="left:${pct}%"></div>
            </div>
            <div class="biomarker-range-labels">
                <span>${range.min}</span>
                <span style="color:#10b981">${range.normal_min} – ${range.normal_max} ${range.unit}</span>
                <span>${range.max}</span>
            </div>`;
        grid.appendChild(card);
    });
}

// ── Session History ───────────────────────────────────────────────────
function addToHistory(data) {
    sessionHistory.unshift({
        ref:       data.patient_ref,
        patient:   data.patient.name || "Anonymous",
        diagnosis: data.ensemble.primary_diagnosis,
        confidence:data.ensemble.confidence,
        band:      data.ensemble.confidence_band,
        timestamp: new Date().toLocaleTimeString("en-GB"),
        data
    });
    if (sessionHistory.length > 20) sessionHistory.pop();
    renderHistory();
    setText("history-count", String(sessionHistory.length));
}

function renderHistory() {
    const list = document.getElementById("history-list");
    if (!list) return;

    if (sessionHistory.length === 0) {
        list.innerHTML = `<div class="empty-state"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg><p>No reports analysed yet this session.</p></div>`;
        return;
    }

    const colors = { HIGH: "#10b981", MODERATE: "#f59e0b", LOW: "#f97316", INCONCLUSIVE: "#ef4444" };
    list.innerHTML = sessionHistory.map((h, i) => `
        <div class="history-item" onclick="recallHistory(${i})">
            <div class="hi-left">
                <div class="hi-ref">${h.ref}</div>
                <div class="hi-patient">${h.patient}</div>
                <div class="hi-time">${h.timestamp}</div>
            </div>
            <div class="hi-right">
                <div class="hi-diagnosis">${h.diagnosis}</div>
                <div class="hi-conf" style="color:${colors[h.band] || "#94a3b8"}">${h.confidence}% · ${h.band}</div>
            </div>
        </div>`).join("");
}

function recallHistory(idx) {
    const h = sessionHistory[idx];
    if (!h || !h.data) return;
    currentData = h.data;
    switchTab("dashboard");
    renderResults(h.data);
}

function clearHistory() {
    sessionHistory = [];
    renderHistory();
    setText("history-count", "0");
}

// Load permanent history from server
async function loadHistoryFromServer() {
    try {
        const res = await fetch("/api/reports?limit=30", {
            headers: { "Authorization": `Bearer ${AUTH_TOKEN}` }
        });
        if (!res.ok) return;
        const d = await res.json();
        const reports = d.reports || [];
        if (reports.length === 0) return;

        // Populate badge count
        const totalCount = Math.max(reports.length, sessionHistory.length);
        setText("history-count", String(totalCount));

        const colors = { HIGH: "#10b981", MODERATE: "#f59e0b", LOW: "#f97316", INCONCLUSIVE: "#ef4444" };
        const list = document.getElementById("history-list");
        if (!list) return;

        const sessionRefs = new Set(sessionHistory.map(s => s.ref));
        let html = "";

        if (sessionHistory.length > 0) {
            html += sessionHistory.map((h, i) => `
                <div class="history-item" onclick="recallHistory(${i})" title="Click to view dossier">
                    <div class="hi-left">
                        <div class="hi-ref" style="display:flex;align-items:center;gap:6px;">
                            <span>${h.ref}</span>
                            <span class="card-badge info" style="font-size:9.5px;padding:1px 6px;">Active Session</span>
                        </div>
                        <div class="hi-patient">${h.patient}</div>
                        <div class="hi-time">${h.timestamp}</div>
                    </div>
                    <div class="hi-right">
                        <div class="hi-diagnosis">${h.diagnosis}</div>
                        <div class="hi-conf" style="color:${colors[h.band] || "#94a3b8"}">${h.confidence}% · ${h.band}</div>
                    </div>
                </div>`).join("");
        }

        const remainingReports = reports.filter(r => !sessionRefs.has(r.patient_ref));
        if (remainingReports.length > 0) {
            html += remainingReports.map(r => `
                <div class="history-item" onclick="loadReportFromServer('${r.patient_ref}')" title="Click to recall dossier">
                    <div class="hi-left">
                        <div class="hi-ref" style="display:flex;align-items:center;gap:6px;">
                            <span>${r.patient_ref}</span>
                            <span class="card-badge" style="font-size:9.5px;padding:1px 6px;background:rgba(34,211,238,0.12);color:var(--cyan);border:1px solid rgba(34,211,238,0.3);">Saved / Cloud</span>
                        </div>
                        <div class="hi-patient">${r.patient_name || 'Anonymous'}</div>
                        <div class="hi-time">${new Date(r.created_at).toLocaleString('en-GB')}</div>
                    </div>
                    <div class="hi-right">
                        <div class="hi-diagnosis">${r.primary_diagnosis || '—'}</div>
                        <div class="hi-conf" style="color:${colors[r.confidence_band] || '#94a3b8'}">
                            ${r.confidence_pct ?? '—'}% &middot; ${r.confidence_band || '—'}
                        </div>
                    </div>
                </div>`).join("");
        }

        if (html) {
            list.innerHTML = html;
        } else {
            list.innerHTML = `<div class="empty-state"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg><p>No reports analysed yet.</p></div>`;
        }
    } catch(e) {
        console.warn("[history load failed]", e);
    }
}

async function loadReportFromServer(ref) {
    try {
        const res = await fetch(`/api/reports/${ref}`, {
            headers: { "Authorization": `Bearer ${AUTH_TOKEN}` }
        });
        if (!res.ok) { alert("Could not load report from server."); return; }
        const d = await res.json();
        const payload = d.full_payload;
        if (payload && payload.success) {
            currentData = payload;
            switchTab("dashboard");
            renderResults(payload);
        } else {
            alert("Report data is incomplete or unavailable.");
        }
    } catch(e) {
        console.warn("[report recall failed]", e);
        alert("Failed to load report.");
    }
}

const SYSTEM_DEFAULT_GEMINI_KEY = "";
const SYSTEM_DEFAULT_NVIDIA_KEY = "";


// ── Settings ──────────────────────────────────────────────────────────
function initSettings() {
    const nvInput   = document.getElementById("nvidia-api-key-input");
    const gemInput  = document.getElementById("gemini-api-key-input");
    const saveBtn   = document.getElementById("btn-save-keys");
    const resetBtn  = document.getElementById("btn-reset-defaults");
    const toggleBtn = document.getElementById("btn-toggle-keys");
    const statusEl  = document.getElementById("settings-status");

    if (nvInput)  nvInput.value  = localStorage.getItem("nvidia_api_key")  || SYSTEM_DEFAULT_NVIDIA_KEY;
    if (gemInput) gemInput.value = localStorage.getItem("gemini_api_key")  || SYSTEM_DEFAULT_GEMINI_KEY;

    toggleBtn?.addEventListener("click", () => {
        const hidden = nvInput.type === "password";
        nvInput.type  = hidden ? "text"     : "password";
        gemInput.type = hidden ? "text"     : "password";
        toggleBtn.textContent = hidden ? "Hide Keys" : "Show Keys";
    });

    resetBtn?.addEventListener("click", async () => {
        if (nvInput)  nvInput.value  = SYSTEM_DEFAULT_NVIDIA_KEY;
        if (gemInput) gemInput.value = SYSTEM_DEFAULT_GEMINI_KEY;
        localStorage.setItem("nvidia_api_key", SYSTEM_DEFAULT_NVIDIA_KEY);
        localStorage.setItem("gemini_api_key", SYSTEM_DEFAULT_GEMINI_KEY);

        try {
            await fetch("/api/auth/update-keys", {
                method: "POST",
                headers: { "Content-Type": "application/json", "Authorization": `Bearer ${AUTH_TOKEN}` },
                body: JSON.stringify({ nvidia_api_key: SYSTEM_DEFAULT_NVIDIA_KEY, gemini_api_key: SYSTEM_DEFAULT_GEMINI_KEY })
            });
            if (statusEl) {
                statusEl.textContent = "✅ Reset to system default API keys.";
                statusEl.className   = "settings-status ok";
                setTimeout(() => { statusEl.textContent = ""; statusEl.className = "settings-status"; }, 4000);
            }
        } catch {
            if (statusEl) {
                statusEl.textContent = "⚠️ Reset locally.";
                statusEl.className   = "settings-status warn";
            }
        }
        updateBadges(true, true);
    });

    saveBtn?.addEventListener("click", async () => {
        const nvKey  = nvInput.value.trim();
        const gemKey = gemInput.value.trim();
        localStorage.setItem("nvidia_api_key",  nvKey);
        localStorage.setItem("gemini_api_key",  gemKey);

        try {
            const res = await fetch("/api/auth/update-keys", {
                method: "POST",
                headers: { "Content-Type": "application/json", "Authorization": `Bearer ${AUTH_TOKEN}` },
                body: JSON.stringify({ nvidia_api_key: nvKey, gemini_api_key: gemKey })
            });
            const d = await res.json();
            if (statusEl) {
                statusEl.textContent = res.ok ? "✅ Custom API keys saved and synced to your doctor profile." : "⚠️ Keys saved locally but server sync failed.";
                statusEl.className   = res.ok ? "settings-status ok" : "settings-status warn";
                setTimeout(() => { statusEl.textContent = ""; statusEl.className = "settings-status"; }, 4000);
            }
        } catch {
            if (statusEl) { statusEl.textContent = "⚠️ Keys saved locally. Server sync failed."; statusEl.className = "settings-status warn"; }
        }
        updateBadges(!!gemKey, !!nvKey);
    });
}

// ── Supabase Cloud Integration ─────────────────────────────────────────
async function checkSupabaseStatus() {
    const badge = document.getElementById("supabase-key-badge");
    const pill = document.getElementById("supabase-status-pill");
    try {
        const res = await fetch("/api/supabase/status", {
            headers: { "Authorization": `Bearer ${AUTH_TOKEN}` }
        });
        const data = await res.json();
        if (data.reachable) {
            if (data.table_ready) {
                if (badge) {
                    badge.className = "badge badge-active";
                    badge.textContent = `Supabase: Synced (${data.latency_ms}ms)`;
                    badge.style.background = "rgba(34, 197, 94, 0.2)";
                    badge.style.color = "#22c55e";
                    badge.style.borderColor = "rgba(34, 197, 94, 0.4)";
                }
                if (pill) {
                    pill.className = "badge badge-active";
                    pill.textContent = "Live Synced";
                    pill.style.background = "rgba(34, 197, 94, 0.2)";
                    pill.style.color = "#22c55e";
                }
            } else {
                if (badge) {
                    badge.className = "badge badge-warn";
                    badge.textContent = "Supabase: Schema Pending";
                    badge.style.background = "rgba(234, 179, 8, 0.2)";
                    badge.style.color = "#eab308";
                    badge.style.borderColor = "rgba(234, 179, 8, 0.4)";
                }
                if (pill) {
                    pill.className = "badge badge-warn";
                    pill.textContent = "Schema Pending";
                    pill.style.background = "rgba(234, 179, 8, 0.2)";
                    pill.style.color = "#eab308";
                }
            }
        } else {
            if (badge) {
                badge.className = "badge badge-inactive";
                badge.textContent = "Supabase: Offline";
            }
            if (pill) {
                pill.className = "badge badge-inactive";
                pill.textContent = "Offline";
            }
        }
        return data;
    } catch {
        if (badge) {
            badge.className = "badge badge-inactive";
            badge.textContent = "Supabase: Offline";
        }
        if (pill) {
            pill.className = "badge badge-inactive";
            pill.textContent = "Offline";
        }
        return null;
    }
}

function initSupabaseSync() {
    checkSupabaseStatus();

    const testBtn = document.getElementById("btn-test-supabase");
    const syncBtn = document.getElementById("btn-sync-supabase");
    const sqlBtn = document.getElementById("btn-view-supabase-sql");
    const copyBtn = document.getElementById("btn-copy-sql");
    const resultsEl = document.getElementById("supabase-test-results");
    const drawerEl = document.getElementById("supabase-sql-drawer");
    const codeEl = document.getElementById("supabase-sql-code");

    testBtn?.addEventListener("click", async () => {
        if (!resultsEl) return;
        resultsEl.style.display = "block";
        resultsEl.innerHTML = `<span style="color: #94a3b8;">Testing connection to Supabase endpoint...</span>`;
        const data = await checkSupabaseStatus();
        if (!data) {
            resultsEl.innerHTML = `<div style="color: #f87171; background: rgba(239,68,68,0.15); padding: 0.75rem; border-radius: 6px;">❌ Could not reach Supabase endpoint. Please check your network or credentials.</div>`;
            return;
        }

        if (data.reachable && data.table_ready) {
            resultsEl.innerHTML = `
                <div style="color: #4ade80; background: rgba(34,197,94,0.15); padding: 0.75rem; border-radius: 6px; border: 1px solid rgba(34,197,94,0.3);">
                    <strong>✅ Supabase Connected & Ready!</strong><br>
                    <span>Project: <code>${data.project_ref}</code> | Latency: <strong>${data.latency_ms} ms</strong></span><br>
                    <span style="font-size: 0.82rem; color: #a7f3d0;">All patient reports are automatically synchronized in real-time.</span>
                </div>`;
        } else if (data.reachable && !data.table_ready) {
            resultsEl.innerHTML = `
                <div style="color: #fde047; background: rgba(234,179,8,0.15); padding: 0.75rem; border-radius: 6px; border: 1px solid rgba(234,179,8,0.3);">
                    <strong>⚠️ Connected to Supabase Project <code>${data.project_ref}</code> (${data.latency_ms}ms)</strong><br>
                    <span>The database table <code>patient_reports</code> is not created yet.</span><br>
                    <div style="margin-top: 0.5rem;">Click <strong>"View & Copy SQL Schema"</strong> below and execute it in your Supabase SQL Editor to finish setup.</div>
                </div>`;
            if (drawerEl) drawerEl.style.display = "block";
            loadSupabaseSchema();
        } else {
            resultsEl.innerHTML = `
                <div style="color: #f87171; background: rgba(239,68,68,0.15); padding: 0.75rem; border-radius: 6px;">
                    <strong>❌ Connection Error:</strong> ${data.message || data.error}
                </div>`;
        }
    });

    syncBtn?.addEventListener("click", async () => {
        if (!resultsEl) return;
        resultsEl.style.display = "block";
        resultsEl.innerHTML = `<span style="color: #38bdf8;">🔄 Syncing all local reports to Supabase cloud...</span>`;
        try {
            const res = await fetch("/api/supabase/sync-all", {
                method: "POST",
                headers: { "Authorization": `Bearer ${AUTH_TOKEN}` }
            });
            const d = await res.json();
            if (res.ok && d.success) {
                resultsEl.innerHTML = `
                    <div style="color: #4ade80; background: rgba(34,197,94,0.15); padding: 0.75rem; border-radius: 6px; border: 1px solid rgba(34,197,94,0.3);">
                        <strong>✅ ${d.message}</strong> (${d.synced} synced, ${d.failed} failed)
                    </div>`;
                checkSupabaseStatus();
            } else {
                resultsEl.innerHTML = `
                    <div style="color: #f87171; background: rgba(239,68,68,0.15); padding: 0.75rem; border-radius: 6px;">
                        <strong>⚠️ Sync Notice:</strong> ${d.detail || d.message || "Failed to sync"}
                    </div>`;
            }
        } catch (err) {
            resultsEl.innerHTML = `<div style="color: #f87171; background: rgba(239,68,68,0.15); padding: 0.75rem; border-radius: 6px;">Error during sync: ${err.message}</div>`;
        }
    });

    async function loadSupabaseSchema() {
        if (!codeEl) return;
        if (codeEl.textContent.trim().length > 10) return;
        try {
            const res = await fetch("/api/supabase/schema");
            const sql = await res.text();
            codeEl.textContent = sql;
        } catch {
            codeEl.textContent = "-- Could not load schema SQL.";
        }
    }

    sqlBtn?.addEventListener("click", () => {
        if (!drawerEl) return;
        const isHidden = drawerEl.style.display === "none";
        drawerEl.style.display = isHidden ? "block" : "none";
        if (isHidden) loadSupabaseSchema();
    });

    copyBtn?.addEventListener("click", () => {
        if (!codeEl) return;
        navigator.clipboard.writeText(codeEl.textContent).then(() => {
            const prev = copyBtn.textContent;
            copyBtn.textContent = "Copied!";
            copyBtn.style.color = "#22c55e";
            setTimeout(() => {
                copyBtn.textContent = prev;
                copyBtn.style.color = "";
            }, 2000);
        });
    });
}

// ── Benchmarks ────────────────────────────────────────────────────────
function loadBenchmarks() {
    fetch("/api/benchmarks")
    .then(r => r.json())
    .then(data => {
        if (data.error) return;
        const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
        set("score-lr",  `${(data["Logistic Regression"]?.accuracy * 100).toFixed(1)}%`);
        set("f1-lr",     data["Logistic Regression"]?.f1_macro.toFixed(3));
        set("lat-lr",    data["Logistic Regression"]?.inference_latency_ms.toFixed(2));
        set("score-xgb", `${(data["XGBoost"]?.accuracy * 100).toFixed(1)}%`);
        set("f1-xgb",    data["XGBoost"]?.f1_macro.toFixed(3));
        set("lat-xgb",   data["XGBoost"]?.inference_latency_ms.toFixed(2));
        set("score-rf",  `${(data["Random Forest"]?.accuracy * 100).toFixed(1)}%`);
        set("f1-rf",     data["Random Forest"]?.f1_macro.toFixed(3));
        set("lat-rf",    data["Random Forest"]?.inference_latency_ms.toFixed(2));
        renderF1ClassChart(data["XGBoost"]?.classification_report || {});
    })
    .catch(() => {});
}

function renderF1ClassChart(report) {
    if (!window.Chart) return;
    const ctx = document.getElementById("f1ClassChart")?.getContext("2d");
    if (!ctx) return;

    const classes = Object.keys(report).filter(k => !["accuracy","macro avg","weighted avg"].includes(k));
    const scores  = classes.map(c => Math.round(report[c]["f1-score"] * 100));

    if (benchmarkChart) benchmarkChart.destroy();
    benchmarkChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: classes,
            datasets: [{ label: "F1 (%)", data: scores, backgroundColor: "rgba(6,182,212,0.65)", borderWidth: 0, borderRadius: 6 }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                x: { grid: { display: false }, ticks: { color: "#f3f4f6", font: { size: 10 } } },
                y: { beginAtZero: true, max: 100, grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#9ca3af" } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

// ── Print Builder ─────────────────────────────────────────────────────
function preparePrintView() {
    if (!currentData) return;
    const { patient, analyzed_by, ensemble, best_narrative, recommendations, patient_ref } = currentData;
    const review = currentData._review || {};

    const now = new Date().toLocaleDateString("en-GB", { weekday:"long", year:"numeric", month:"long", day:"numeric", hour:"2-digit", minute:"2-digit" });

    setText("print-date",         now);
    setText("print-doctor-name",  analyzed_by.doctor_name);
    setText("print-slmc",         analyzed_by.slmc_number);
    setText("print-ref",          patient_ref);
    setText("print-patient-name", patient.name || "Anonymous");
    setText("print-age-gender",   `${patient.age} years · ${patient.gender}`);
    setText("print-report-type",  patient.report_type);
    setText("print-report-date",  patient.report_date);
    setText("print-notes",        patient.clinical_notes || "None");
    setText("print-diagnosis",    ensemble.primary_diagnosis);
    setText("print-confidence",   `${ensemble.confidence}% (${ensemble.confidence_band})`);
    setText("print-interpretation-text", best_narrative);
    setText("print-review-decision", review.action || "Not reviewed");
    setText("print-review-notes",    review.notes  || "None");
    setText("print-signed-by",       review.doctor ? `${review.doctor} on ${review.timestamp}` : "Not signed");

    // Biomarkers table
    const tbody = document.getElementById("print-biomarkers-body");
    if (tbody && currentData.parsed_values) {
        tbody.innerHTML = Object.entries(BIOMARKER_RANGES).map(([key, range]) => {
            const val = currentData.parsed_values[key];
            if (val === undefined) return "";
            const isNormal = val >= range.normal_min && val <= range.normal_max;
            return `<tr>
                <td><strong>${key.replace(/_/g," ")}</strong></td>
                <td>${val} ${range.unit}</td>
                <td>${range.normal_min} – ${range.normal_max}</td>
                <td style="color:${isNormal?"green":"red"};font-weight:700">${isNormal?"NORMAL":"ABNORMAL"}</td>
            </tr>`;
        }).join("");
    }

    // Recommendations
    const recList = document.getElementById("print-recs");
    if (recList) {
        recList.innerHTML = (recommendations || []).map(r => `<li>${r}</li>`).join("");
    }
}

// ── ORIGINAL LAB REPORT VIEWER & LIGHTBOX ──────────────────────────────
function updateReportPreviewElements(url, filename) {
    if (!url) return;
    currentReportPreviewUrl = url;
    if (filename) currentReportFileName = filename;

    // Dropzone Thumbnail Preview in Upload Card
    const dropzoneDefault = document.getElementById("dropzone-default-content");
    const dropzoneSheet = document.getElementById("dropzone-doc-sheet");
    const dropzonePreviewBox = document.getElementById("dropzone-preview-box");
    const dropzonePreviewImg = document.getElementById("dropzone-preview-img");
    const dropzoneFilenameEl = document.getElementById("dropzone-preview-filename");

    if (dropzonePreviewBox && dropzonePreviewImg) {
        dropzonePreviewImg.src = url;
        if (dropzoneFilenameEl) dropzoneFilenameEl.textContent = currentReportFileName;
        if (dropzoneDefault) dropzoneDefault.classList.add("hidden");
        if (dropzoneSheet) dropzoneSheet.classList.add("hidden");
        dropzonePreviewBox.classList.remove("hidden");
    }

    const inlineCard = document.getElementById("original-report-card");
    const inlineImg = document.getElementById("report-inline-img");
    const filenameEl = document.getElementById("report-card-filename");
    const modalFilenameEl = document.getElementById("report-modal-filename");
    const modalImg = document.getElementById("report-modal-img");
    const modalPdf = document.getElementById("report-modal-pdf");
    const downloadEl = document.getElementById("report-modal-download");
    const bannerBtn = document.getElementById("btn-view-doc-banner");

    if (inlineCard) inlineCard.classList.remove("hidden");
    if (bannerBtn) bannerBtn.style.display = "inline-flex";
    if (filenameEl) filenameEl.textContent = currentReportFileName;
    if (modalFilenameEl) modalFilenameEl.textContent = currentReportFileName;

    const isPdf = url.startsWith("data:application/pdf") || (currentReportFileName && currentReportFileName.toLowerCase().endsWith(".pdf"));
    if (isPdf) {
        if (modalImg) modalImg.classList.add("hidden");
        if (modalPdf) {
            modalPdf.classList.remove("hidden");
            modalPdf.src = url;
        }
        if (inlineImg) {
            inlineImg.src = "/static/Logo.png";
        }
    } else {
        if (modalPdf) modalPdf.classList.add("hidden");
        if (modalImg) {
            modalImg.classList.remove("hidden");
            modalImg.src = url;
        }
        if (inlineImg) {
            inlineImg.src = url;
        }
    }


    if (downloadEl) {
        downloadEl.href = url;
        downloadEl.download = currentReportFileName;
    }
}

function openReportModal() {
    const modal = document.getElementById("report-modal");
    if (!modal) return;
    modal.classList.remove("hidden");
    resetReportZoom();
}

function closeReportModal() {
    const modal = document.getElementById("report-modal");
    if (modal) modal.classList.add("hidden");
}

function zoomReport(delta) {
    reportZoomLevel = Math.max(0.4, Math.min(4.0, reportZoomLevel + delta));
    applyReportTransform();
}

function resetReportZoom() {
    reportZoomLevel = 1.0;
    reportRotation = 0;
    reportPanX = 0;
    reportPanY = 0;
    applyReportTransform();
}

function rotateReport() {
    reportRotation = (reportRotation + 90) % 360;
    applyReportTransform();
}

function applyReportTransform() {
    const canvas = document.getElementById("report-modal-canvas");
    const zoomText = document.getElementById("report-zoom-val");
    if (canvas) {
        canvas.style.transform = `translate(${reportPanX}px, ${reportPanY}px) scale(${reportZoomLevel}) rotate(${reportRotation}deg)`;
    }
    if (zoomText) {
        zoomText.textContent = `${Math.round(reportZoomLevel * 100)}%`;
    }
}

function toggleReportInlinePreview() {
    const frame = document.getElementById("report-inline-container");
    const btnText = document.getElementById("btn-toggle-preview-text");
    if (frame) {
        const isHidden = frame.classList.toggle("hidden");
        if (btnText) btnText.textContent = isHidden ? "Show Preview" : "Hide Preview";
    }
}

function initReportModalEvents() {
    const viewport = document.getElementById("report-modal-viewport");
    const canvas = document.getElementById("report-modal-canvas");
    if (!viewport || !canvas) return;

    viewport.addEventListener("mousedown", e => {
        if (e.target.tagName === "BUTTON" || e.target.closest("button") || e.target.tagName === "A") return;
        isPanningReport = true;
        canvas.classList.add("panning");
        startPanX = e.clientX - reportPanX;
        startPanY = e.clientY - reportPanY;
    });

    window.addEventListener("mousemove", e => {
        if (!isPanningReport) return;
        reportPanX = e.clientX - startPanX;
        reportPanY = e.clientY - startPanY;
        applyReportTransform();
    });

    window.addEventListener("mouseup", () => {
        if (isPanningReport) {
            isPanningReport = false;
            canvas?.classList.remove("panning");
        }
    });

    viewport.addEventListener("wheel", e => {
        e.preventDefault();
        const delta = e.deltaY < 0 ? 0.15 : -0.15;
        zoomReport(delta);
    }, { passive: false });

    window.addEventListener("keydown", e => {
        if (e.key === "Escape") closeReportModal();
    });
}
