/**
 * clinical_3d.js — CliniQ AI Premium 3D Spatial & Motion Engine
 * 
 * Provides:
 *  1. Background ambient medical spatial particle network (low CPU, high fidelity)
 *  2. 3D Clinical Consensus Core Canvas (interactive rotating 4-engine neural sphere)
 *  3. Cinematic 5-Phase Medical AI Analysis Sequence
 */

class ClinicalSpatialEngine {
    constructor() {
        this.bgCanvas = null;
        this.bgCtx = null;
        this.coreCanvas = null;
        this.coreCtx = null;
        this.bgParticles = [];
        this.coreNodes = [];
        this.coreAngleX = 0;
        this.coreAngleY = 0;
        this.coreTargetX = 0;
        this.coreTargetY = 0;
        this.animFrameId = null;
        this.isHoveringCore = false;
    }

    init() {
        this.setupBackgroundCanvas();
        this.setupNeuralCoreCanvas();
        this.bindEvents();
        this.animate();
    }

    // ── 1. Spatial Background Network ─────────────────────────────────────────────
    setupBackgroundCanvas() {
        let canvas = document.getElementById("clinical-bg-canvas");
        if (!canvas) {
            canvas = document.createElement("canvas");
            canvas.id = "clinical-bg-canvas";
            canvas.style.position = "fixed";
            canvas.style.top = "0";
            canvas.style.left = "0";
            canvas.style.width = "100vw";
            canvas.style.height = "100vh";
            canvas.style.pointerEvents = "none";
            canvas.style.zIndex = "0";
            canvas.style.opacity = "0.75";
            document.body.prepend(canvas);
        }
        this.bgCanvas = canvas;
        this.bgCtx = canvas.getContext("2d");
        this.resizeBg();

        // Initialize particles
        const count = Math.min(45, Math.floor(window.innerWidth / 35));
        this.bgParticles = [];
        for (let i = 0; i < count; i++) {
            this.bgParticles.push({
                x: Math.random() * this.bgCanvas.width,
                y: Math.random() * this.bgCanvas.height,
                vx: (Math.random() - 0.5) * 0.35,
                vy: (Math.random() - 0.5) * 0.35,
                size: Math.random() * 2 + 1,
                alpha: Math.random() * 0.35 + 0.1,
                color: Math.random() > 0.4 ? "34, 211, 238" : "59, 130, 246" // Cyan or Blue
            });
        }
    }

    resizeBg() {
        if (!this.bgCanvas) return;
        this.bgCanvas.width = window.innerWidth;
        this.bgCanvas.height = window.innerHeight;
    }

    // ── 2. 3D Neural Consensus Core ───────────────────────────────────────────────
    setupNeuralCoreCanvas() {
        this.coreCanvas = document.getElementById("neural-core-canvas");
        if (!this.coreCanvas) return;
        this.coreCtx = this.coreCanvas.getContext("2d");
        this.resizeCore();

        // 3D Spherical Coordinates for Neural Nodes
        // 4 Primary Engine Anchor Nodes + Supporting Network Lattice
        this.coreNodes = [];
        const radius = 72;

        // 4 Main Engines mapped to 4 quadrants
        const primaryEngines = [
            { name: "RULES", color: "#22d3ee", phi: Math.PI / 4, theta: Math.PI / 3, isPrimary: true },
            { name: "LOCAL ML", color: "#38bdf8", phi: 3 * Math.PI / 4, theta: 2 * Math.PI / 3, isPrimary: true },
            { name: "NVIDIA", color: "#818cf8", phi: 5 * Math.PI / 4, theta: 4 * Math.PI / 3, isPrimary: true },
            { name: "GEMINI", color: "#34d399", phi: 7 * Math.PI / 4, theta: 5 * Math.PI / 3, isPrimary: true }
        ];

        primaryEngines.forEach(p => {
            this.coreNodes.push({
                x: radius * Math.sin(p.theta) * Math.cos(p.phi),
                y: radius * Math.cos(p.theta),
                z: radius * Math.sin(p.theta) * Math.sin(p.phi),
                label: p.name,
                color: p.color,
                isPrimary: true,
                pulse: 0
            });
        });

        // 20 secondary lattice nodes
        for (let i = 0; i < 20; i++) {
            const theta = Math.acos((Math.random() * 2) - 1);
            const phi = Math.random() * Math.PI * 2;
            const r = radius * (0.85 + Math.random() * 0.3);
            this.coreNodes.push({
                x: r * Math.sin(theta) * Math.cos(phi),
                y: r * Math.cos(theta),
                z: r * Math.sin(theta) * Math.sin(phi),
                isPrimary: false,
                color: "rgba(56, 189, 248, 0.45)"
            });
        }
    }

    resizeCore() {
        if (!this.coreCanvas) return;
        const rect = this.coreCanvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.coreCanvas.width = (rect.width || 280) * dpr;
        this.coreCanvas.height = (rect.height || 220) * dpr;
        if (this.coreCtx) {
            this.coreCtx.scale(dpr, dpr);
        }
    }

    bindEvents() {
        window.addEventListener("resize", () => {
            this.resizeBg();
            this.resizeCore();
        });

        if (this.coreCanvas) {
            this.coreCanvas.addEventListener("mousemove", (e) => {
                const rect = this.coreCanvas.getBoundingClientRect();
                const x = (e.clientX - rect.left) / rect.width - 0.5;
                const y = (e.clientY - rect.top) / rect.height - 0.5;
                this.coreTargetX = y * 0.8;
                this.coreTargetY = x * 1.2;
                this.isHoveringCore = true;
            });

            this.coreCanvas.addEventListener("mouseleave", () => {
                this.isHoveringCore = false;
            });
        }
    }

    // ── Main Animation Loop ───────────────────────────────────────────────────────
    animate() {
        this.drawBackground();
        this.drawNeuralCore();
        this.animFrameId = requestAnimationFrame(() => this.animate());
    }

    drawBackground() {
        if (!this.bgCtx || !this.bgCanvas) return;
        const ctx = this.bgCtx;
        const w = this.bgCanvas.width;
        const h = this.bgCanvas.height;

        ctx.clearRect(0, 0, w, h);

        // Update and draw particles
        for (let i = 0; i < this.bgParticles.length; i++) {
            const p = this.bgParticles[i];
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0) p.x = w;
            if (p.x > w) p.x = 0;
            if (p.y < 0) p.y = h;
            if (p.y > h) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${p.color}, ${p.alpha})`;
            ctx.shadowBlur = 6;
            ctx.shadowColor = `rgba(${p.color}, 0.5)`;
            ctx.fill();
            ctx.shadowBlur = 0;

            // Connect nearby particles with subtle lines
            for (let j = i + 1; j < this.bgParticles.length; j++) {
                const p2 = this.bgParticles[j];
                const dx = p.x - p2.x;
                const dy = p.y - p2.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 130) {
                    const lineAlpha = (1 - dist / 130) * 0.12;
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = `rgba(34, 211, 238, ${lineAlpha})`;
                    ctx.lineWidth = 0.75;
                    ctx.stroke();
                }
            }
        }
    }

    drawNeuralCore() {
        if (!this.coreCtx || !this.coreCanvas) return;
        const ctx = this.coreCtx;
        const rect = this.coreCanvas.getBoundingClientRect();
        const cw = rect.width || 280;
        const ch = rect.height || 220;
        const cx = cw / 2;
        const cy = ch / 2;

        ctx.clearRect(0, 0, cw, ch);

        // Rotation physics
        if (!this.isHoveringCore) {
            this.coreTargetY += 0.007;
            this.coreTargetX = Math.sin(Date.now() * 0.001) * 0.2;
        }
        this.coreAngleX += (this.coreTargetX - this.coreAngleX) * 0.06;
        this.coreAngleY += (this.coreTargetY - this.coreAngleY) * 0.06;

        const sinX = Math.sin(this.coreAngleX);
        const cosX = Math.cos(this.coreAngleX);
        const sinY = Math.sin(this.coreAngleY);
        const cosY = Math.cos(this.coreAngleY);

        // Project 3D nodes
        const projected = this.coreNodes.map(node => {
            // Rotate Y
            let x1 = node.x * cosY + node.z * sinY;
            let z1 = -node.x * sinY + node.z * cosY;
            // Rotate X
            let y2 = node.y * cosX - z1 * sinX;
            let z2 = node.y * sinX + z1 * cosX;

            const scale = 220 / (220 + z2);
            return {
                x2d: cx + x1 * scale,
                y2d: cy + y2 * scale,
                scale: scale,
                z: z2,
                raw: node
            };
        });

        // Sort by Z for realistic depth
        projected.sort((a, b) => a.z - b.z);

        // Draw Central Luminous Core
        const coreGradient = ctx.createRadialGradient(cx, cy, 4, cx, cy, 42);
        coreGradient.addColorStop(0, "rgba(34, 211, 238, 0.45)");
        coreGradient.addColorStop(0.5, "rgba(59, 130, 246, 0.2)");
        coreGradient.addColorStop(1, "rgba(15, 23, 42, 0)");
        ctx.fillStyle = coreGradient;
        ctx.beginPath();
        ctx.arc(cx, cy, 42, 0, Math.PI * 2);
        ctx.fill();

        // Draw Core Orbit Ring
        ctx.beginPath();
        ctx.ellipse(cx, cy, 64, 24, this.coreAngleY * 0.5, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(34, 211, 238, 0.25)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 6]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw connecting lattice beams
        for (let i = 0; i < projected.length; i++) {
            const p1 = projected[i];
            // Connect to center if primary
            if (p1.raw.isPrimary) {
                ctx.beginPath();
                ctx.moveTo(p1.x2d, p1.y2d);
                ctx.lineTo(cx, cy);
                ctx.strokeStyle = `rgba(34, 211, 238, ${0.25 + 0.15 * p1.scale})`;
                ctx.lineWidth = 1.2;
                ctx.stroke();
            }

            for (let j = i + 1; j < projected.length; j++) {
                const p2 = projected[j];
                const dx = p1.x2d - p2.x2d;
                const dy = p1.y2d - p2.y2d;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 55) {
                    ctx.beginPath();
                    ctx.moveTo(p1.x2d, p1.y2d);
                    ctx.lineTo(p2.x2d, p2.y2d);
                    const alpha = (1 - dist / 55) * 0.22 * Math.min(p1.scale, p2.scale);
                    ctx.strokeStyle = `rgba(56, 189, 248, ${alpha})`;
                    ctx.lineWidth = 0.8;
                    ctx.stroke();
                }
            }
        }

        // Draw Nodes
        projected.forEach(p => {
            const isPrimary = p.raw.isPrimary;
            const size = (isPrimary ? 5 : 2.5) * p.scale;
            ctx.beginPath();
            ctx.arc(p.x2d, p.y2d, size, 0, Math.PI * 2);

            if (isPrimary) {
                ctx.fillStyle = p.raw.color;
                ctx.shadowBlur = 10;
                ctx.shadowColor = p.raw.color;
                ctx.fill();
                ctx.shadowBlur = 0;

                // Engine Label
                ctx.fillStyle = "rgba(241, 245, 249, 0.85)";
                ctx.font = "600 9px 'Plus Jakarta Sans', sans-serif";
                ctx.textAlign = "center";
                ctx.fillText(p.raw.label, p.x2d, p.y2d + 14);
            } else {
                ctx.fillStyle = p.raw.color;
                ctx.fill();
            }
        });
    }
}

// ── 3. Cinematic 5-Phase Analysis Runner ─────────────────────────────────────────
class ClinicalSequenceController {
    static async runSequence(onPhaseChange, onComplete) {
        const phases = [
            {
                id: 1,
                name: "DOCUMENT SCAN",
                desc: "Scanning high-resolution morphology & technical grid...",
                duration: 900
            },
            {
                id: 2,
                name: "OCR & BIOMARKER EXTRACTION",
                desc: "Parsing quantitative analyte values (Hb, MCV, Platelets, Glucose)...",
                duration: 950
            },
            {
                id: 3,
                name: "FOUR-ENGINE PARALLEL COMPUTE",
                desc: "Activating Rule Engine, Local XGBoost, NVIDIA NIM, and Gemini 2.5...",
                duration: 1100
            },
            {
                id: 4,
                name: "CONSENSUS CORE SYNTHESIS",
                desc: "Converging multi-engine telemetry into unified clinical consensus...",
                duration: 850
            },
            {
                id: 5,
                name: "DOSSIER GENERATION",
                desc: "Finalizing risk stratifications, evidence tiers, and physician approval card...",
                duration: 600
            }
        ];

        for (const phase of phases) {
            if (onPhaseChange) onPhaseChange(phase);
            await new Promise(r => setTimeout(r, phase.duration));
        }

        if (onComplete) onComplete();
    }
}

// Auto-initialize spatial engine on load
window.clinicalSpatialEngine = new ClinicalSpatialEngine();
window.addEventListener("DOMContentLoaded", () => {
    window.clinicalSpatialEngine.init();
});
