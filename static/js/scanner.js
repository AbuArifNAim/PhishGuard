/* ============================================================
   URL Scanner — verification report + risk indicators
   ============================================================ */

const FEATURE_LABELS = {
    url_length: "URL Length",
    path_length: "Path Length",
    dot_count: "Dot Count",
    subdomain_count: "Subdomains",
    tld_length: "TLD Length",
    has_ip_address: "IP Address",
    has_hyphen_in_domain: "Hyphen",
    https_flag: "HTTPS",
    percentage_numeric_chars: "Numeric %",
    number_of_digits: "Digits",
    token_count: "Tokens",
    url_entropy: "Entropy",
    domain_name_length: "Domain Len",
    suspicious_file_extension: "Suspicious Ext",
    tld_popularity: "Popular TLD",
    query_param_count: "Query Params",
};

const FEATURE_ORDER = [
    "url_length", "path_length", "domain_name_length", "subdomain_count",
    "dot_count", "token_count", "tld_length", "tld_popularity",
    "has_ip_address", "has_hyphen_in_domain", "https_flag",
    "suspicious_file_extension", "percentage_numeric_chars",
    "number_of_digits", "url_entropy", "query_param_count",
];

const BINARY_FEATURES = new Set([
    "has_ip_address",
    "has_hyphen_in_domain",
    "https_flag",
    "suspicious_file_extension",
    "tld_popularity",
]);

const DECIMAL_FEATURES = new Set(["percentage_numeric_chars", "url_entropy"]);

function formatFeature(key, value) {
    if (BINARY_FEATURES.has(key)) {
        return Number(value) ? "Yes" : "No";
    }
    if (DECIMAL_FEATURES.has(key)) {
        return Number(value).toFixed(2);
    }
    return value;
}

const RING_CIRCUMFERENCE = 2 * Math.PI * 80;
let lastResult = null;

document.addEventListener("DOMContentLoaded", () => {
    PG.el("scanForm").addEventListener("submit", (e) => {
        e.preventDefault();
        checkURL();
    });

    PG.el("exportBtn").addEventListener("click", () => window.print());

    PG.el("downloadJsonBtn").addEventListener("click", () => {
        if (!lastResult) return;
        PG.download(`phishguard-report-${lastResult.scan_id || Date.now()}.json`,
            JSON.stringify(lastResult, null, 2));
        PG.toast("Report downloaded");
    });

    PG.el("rescanBtn").addEventListener("click", () => checkURL());

    // Auto-scan when arriving from the topbar / dashboard (?url=...)
    const preset = new URLSearchParams(location.search).get("url");
    if (preset) {
        PG.el("urlInput").value = preset;
        checkURL();
    }
});

async function checkURL() {
    const url = PG.el("urlInput").value.trim();
    const btn = PG.el("checkBtn");
    const loading = PG.el("loading");
    const report = PG.el("report");
    const errorMsg = PG.el("errorMsg");

    report.classList.add("hidden");
    errorMsg.style.display = "none";

    if (!url) {
        errorMsg.textContent = "Please enter a URL to scan.";
        errorMsg.style.display = "block";
        return;
    }

    btn.disabled = true;
    loading.style.display = "block";
    loading.scrollIntoView({ behavior: "smooth", block: "center" });

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url }),
        });
        const data = await res.json();

        if (!res.ok || data.error) {
            throw new Error(data.error || `Scan failed (${res.status})`);
        }

        lastResult = data;
        renderReport(data);
        report.classList.remove("hidden");
        report.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
        errorMsg.textContent = "Error: " + err.message;
        errorMsg.style.display = "block";
    } finally {
        btn.disabled = false;
        loading.style.display = "none";
    }
}

/* ---------- Report rendering ---------- */

function renderReport(data) {
    const band = riskBandFor(data);

    // Header
    PG.el("scanId").textContent = data.scan_id || "—";
    PG.el("scanTime").textContent = data.timestamp
        ? new Date(data.timestamp).toLocaleString()
        : "—";
    PG.el("scanStatus").innerHTML = PG.statusPill(data.status);
    PG.el("reportUrl").textContent = data.url;
    PG.el("reportHost").textContent = PG.hostOf(data.url);

    // Risk ring
    setRing(data.risk_score, band.color);
    PG.el("riskScore").textContent =
        data.risk_score === null || data.risk_score === undefined ? "—" : data.risk_score;

    const badge = PG.el("riskBadge");
    badge.className = "risk-badge " + band.key;
    badge.textContent = band.label;
    PG.el("ringNote").textContent = band.note;

    // Indicators
    const list = PG.el("indicatorList");
    if (data.indicators && data.indicators.length) {
        list.innerHTML = data.indicators.map((ind) => `
            <div class="indicator ${ind.severity}">
                <div class="indicator-top">
                    <span class="indicator-title">${PG.escapeHtml(ind.title)}</span>
                    <span class="sev ${ind.severity}">${ind.severity}</span>
                </div>
                <div class="indicator-desc">${PG.escapeHtml(ind.detail)}</div>
            </div>
        `).join("");
    } else {
        list.innerHTML = `<div class="empty-note">No rule-based risk indicators were triggered for this URL.</div>`;
    }

    // Pipeline
    renderPipeline(data);
    renderLayers(data);

    // Probabilities
    const hasModel = data.phishing_probability !== null && data.phishing_probability !== undefined;
    const phish = hasModel ? data.phishing_probability : 0;
    const legit = hasModel ? data.legitimate_probability : 0;

    PG.el("phishProb").textContent = hasModel ? phish + "%" : "—";
    PG.el("legitProb").textContent = hasModel ? legit + "%" : "—";
    PG.el("phishFill").style.width = phish + "%";
    PG.el("legitFill").style.width = legit + "%";
    PG.el("confidenceText").textContent =
        data.confidence !== null && data.confidence !== undefined
            ? data.confidence + "%"
            : "not available";

    PG.el("rawProbText").textContent =
        data.raw_phishing_probability !== null && data.raw_phishing_probability !== undefined
            ? `${data.raw_phishing_probability}% phishing`
            : "—";

    PG.el("calibratedText").textContent =
        data.phishing_probability !== null && data.phishing_probability !== undefined
            ? `${data.phishing_probability}% phishing`
            : "—";

    PG.el("abstainText").textContent = data.calibration
        ? `${data.calibration.abstain_low}%–${data.calibration.abstain_high}%`
        : "30%–70%";

    // Features
    const grid = PG.el("featuresGrid");
    if (data.features) {
        const ordered = FEATURE_ORDER.filter((k) => k in data.features);
        grid.innerHTML = ordered.map((key) => `
            <div class="feature">
                <span>${PG.escapeHtml(FEATURE_LABELS[key] || key)}</span>
                <b>${PG.escapeHtml(formatFeature(key, data.features[key]))}</b>
            </div>
        `).join("");
    } else {
        grid.innerHTML = `<div class="empty-note">No features could be extracted.</div>`;
    }

    // Pipeline tag
    const total = 4;
    const passed = document.querySelectorAll("#verifySteps .step.pass").length;
    PG.el("pipelineTag").textContent = `${passed} / ${total} stages passed`;
}

function setRing(score, color) {
    const ring = PG.el("riskRing");
    const pct = score === null || score === undefined ? 0 : Math.max(0, Math.min(100, score));

    ring.style.stroke = color;
    ring.style.strokeDashoffset = RING_CIRCUMFERENCE;

    // setTimeout (not rAF) so the animation also runs when the tab is hidden
    setTimeout(() => {
        ring.style.strokeDashoffset = RING_CIRCUMFERENCE - (RING_CIRCUMFERENCE * pct) / 100;
    }, 60);
}

function riskBandFor(data) {
    const score = data.risk_score;
    if (data.status === "Phishing") {
        return {
            key: "high",
            label: "High risk",
            color: "#ef4444",
            note: `Model verdict: Phishing at ${data.confidence ?? "—"}% confidence. Treat this URL as unsafe — do not enter credentials or download files.`,
        };
    }
    if (data.status === "Legitimate") {
        return {
            key: "low",
            label: "Low risk",
            color: "#22c55e",
            note: `Model verdict: Legitimate at ${data.confidence ?? "—"}% confidence. The site resolved and responded normally.`,
        };
    }
    if (data.status === "Unreachable") {
        return {
            key: "medium",
            label: "Unverified",
            color: "#f59e0b",
            note: "The site could not be reached, so the verdict is neither Legitimate nor Phishing. Phishing sites often go offline — review the indicators manually.",
        };
    }
    return {
        key: "neutral",
        label: "Unknown",
        color: "#94a3b8",
        note: "Not enough reliable information for a verdict. Check the pipeline details and risk indicators below.",
    };
}

const STEP_ICONS = {
    pass: '<svg viewBox="0 0 24 24"><path d="M20 6L9 17l-5-5"/></svg>',
    fail: '<svg viewBox="0 0 24 24"><path d="M18 6L6 18M6 6l12 12"/></svg>',
    warn: '<svg viewBox="0 0 24 24"><path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9L2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>',
    skip: '<svg viewBox="0 0 24 24"><path d="M5 12h14"/></svg>',
    info: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></svg>',
};

const STEP_TICKS = { pass: "PASS", fail: "FAIL", warn: "WARN", skip: "SKIP", info: "INFO" };

function renderPipeline(data) {
    const v = data.verification || {};
    const hasModel = data.prediction !== null && data.prediction !== undefined;
    const steps = [];

    // 1 — URL validation
    steps.push({
        state: v.url_valid ? "pass" : "fail",
        title: "URL Validation (Format & Scheme)",
        detail: v.url_detail || (v.url_valid ? "Valid URL" : "Invalid URL"),
    });

    // 2 — Domain verification
    if (v.dns_resolved === null || v.dns_resolved === undefined) {
        steps.push({
            state: "skip",
            title: "Domain Verification (DNS + HTTP Probe)",
            detail: "Skipped — URL validation failed",
        });
    } else if (!v.dns_resolved) {
        steps.push({
            state: "fail",
            title: "Domain Verification (DNS + HTTP Probe)",
            detail: v.domain_detail || "DNS resolution failed",
        });
    } else if (!v.reachable) {
        steps.push({
            state: "fail",
            title: "Domain Verification (DNS + HTTP Probe)",
            detail: "DNS resolved, but the connection failed — " + (v.domain_detail || ""),
        });
    } else {
        steps.push({
            state: "pass",
            title: "Domain Verification (DNS + HTTP Probe)",
            detail: "DNS resolved — " + (v.domain_detail || "reachable"),
        });
    }

    // 3 — ML prediction
    if (hasModel) {
        steps.push({
            state: data.prediction === "Phishing" ? "fail" : "pass",
            title: "Machine Learning Prediction (Random Forest)",
            detail: `Model verdict: ${data.prediction} — confidence ${data.confidence}%`,
        });
    } else {
        steps.push({
            state: "skip",
            title: "Machine Learning Prediction (Random Forest)",
            detail: "Not run — insufficient information",
        });
    }

    // 4 — Final assessment
    const finalState = {
        Phishing: "fail",
        Legitimate: "pass",
        Unreachable: "warn",
        Unknown: "warn",
    }[data.status] || "warn";

    steps.push({
        state: finalState,
        title: "Final Assessment",
        detail: `${data.status} — ${data.status_reason || ""}`,
    });

    PG.el("verifySteps").innerHTML = steps.map((s) => `
        <div class="step ${s.state}">
            <div class="step-dot">${STEP_ICONS[s.state]}</div>
            <div class="step-body">
                <div class="step-title">${PG.escapeHtml(s.title)}
                    <span class="tick">${STEP_TICKS[s.state]}</span>
                </div>
                <div class="step-detail">${PG.escapeHtml(s.detail)}</div>
            </div>
        </div>
    `).join("");
}

function renderLayers(data) {
    const list = PG.el("layerSteps");
    const tag = PG.el("layerTag");
    const layers = data.layers || [];

    if (!layers.length) {
        list.innerHTML = `<div class="empty-note">The verification layers did not run for this scan.</div>`;
        tag.textContent = "not run";
        return;
    }

    const clean = layers.filter((l) => l.state === "pass").length;
    const suspicious = layers.filter((l) => l.state === "fail").length;
    tag.textContent = `${clean} pass · ${suspicious} suspicious`;

    list.innerHTML = layers.map((layer) => {
        const state = STEP_ICONS[layer.state] ? layer.state : "info";
        const influence = layer.influence && layer.influence !== "—"
            ? `<span class="layer-influence">${PG.escapeHtml(layer.influence)}</span>`
            : "";
        return `
            <div class="step ${state}">
                <div class="step-dot">${STEP_ICONS[state]}</div>
                <div class="step-body">
                    <div class="step-title">${PG.escapeHtml(layer.name || layer.key)}
                        <span class="tick">${STEP_TICKS[state]}</span>
                        ${influence}
                    </div>
                    <div class="step-detail">${PG.escapeHtml(layer.detail || "")}</div>
                </div>
            </div>
        `;
    }).join("");
}
