/* ============================================================
   XAI — SHAP local explanation + global feature importance
   ============================================================ */

let historyRecords = [];
let selectedRecord = null;
let shapChartRef = null;
let globalChartRef = null;

const BINARY_FEATURES = new Set([
    "has_ip_address",
    "has_hyphen_in_domain",
    "https_flag",
    "suspicious_file_extension",
    "tld_popularity",
]);
const DECIMAL_FEATURES = new Set(["percentage_numeric_chars", "url_entropy"]);

function formatValue(feature, value) {
    if (BINARY_FEATURES.has(feature)) return Number(value) ? "Yes" : "No";
    if (DECIMAL_FEATURES.has(feature)) return Number(value).toFixed(2);
    return value;
}

document.addEventListener("DOMContentLoaded", async () => {
    PG.el("exportXaiBtn").addEventListener("click", () => {
        if (!selectedRecord) return;
        PG.download(`phishguard-xai-${selectedRecord.id}.json`,
            JSON.stringify(selectedRecord, null, 2));
        PG.toast("Explanation downloaded");
    });

    PG.el("printXaiBtn").addEventListener("click", () => window.print());

    try {
        const [history, analytics] = await Promise.all([
            PG.getJSON("/api/history?limit=25"),
            PG.getJSON("/api/analytics"),
        ]);

        historyRecords = (history.records || []).filter((r) => r.explanation && r.explanation.contributions);

        if (!historyRecords.length) {
            PG.el("emptyState").classList.remove("hidden");
            PG.el("xaiContent").classList.add("hidden");
            PG.el("historySelect").innerHTML = `<option>No explained scans yet</option>`;
            return;
        }

        PG.el("emptyState").classList.add("hidden");
        PG.el("xaiContent").classList.remove("hidden");

        // Render the global chart only after the container is visible,
        // otherwise Chart.js measures a 0×0 box.
        renderGlobal(analytics.feature_importance || []);

        const select = PG.el("historySelect");
        select.innerHTML = historyRecords.map((r, i) => {
            const label = `${PG.truncate(r.url, 46)} · ${r.status}`;
            return `<option value="${i}">${PG.escapeHtml(label)}</option>`;
        }).join("");

        select.addEventListener("change", () => {
            renderExplanation(historyRecords[Number(select.value)]);
        });

        renderExplanation(historyRecords[0]);
    } catch (err) {
        PG.toast("Could not load explanations");
    }
});

function renderExplanation(record) {
    selectedRecord = record;
    const exp = record.explanation;
    const contributions = exp.contributions || [];

    // Hero
    PG.el("xaiUrl").textContent = record.url;
    PG.el("xaiMeta").innerHTML =
        `${PG.statusPill(record.status)} &nbsp;·&nbsp; ${PG.escapeHtml(record.time || "")}` +
        ` &nbsp;·&nbsp; Scan ID ${PG.escapeHtml(record.id || "—")}`;

    // Summary numbers. SHAP explains the *raw* forest output, so use the raw
    // probability here — the calibrated/combined risk is shown on the scanner.
    const base = exp.base_percent;
    const output = record.raw_phishing_probability ?? record.phishing_probability;
    const net = contributions.reduce((sum, c) => sum + c.shap_phishing, 0);

    PG.el("xaiBase").textContent = base.toFixed(2) + "%";
    PG.el("xaiOutput").textContent = output.toFixed(2) + "%";
    PG.el("xaiNet").textContent = (net >= 0 ? "+" : "") + net.toFixed(4);

    PG.el("xaiVerdict").innerHTML = PG.statusPill(record.status);
    PG.el("xaiConfidence").textContent =
        record.confidence !== null && record.confidence !== undefined
            ? record.confidence + "%"
            : "not available";

    // Top factors
    const top = [...contributions]
        .sort((a, b) => Math.abs(b.shap_phishing) - Math.abs(a.shap_phishing))
        .slice(0, 5);

    PG.el("topFactors").innerHTML = top.map((c) => `
        <div class="factor">
            <span class="fname">${PG.escapeHtml(c.label)}</span>
            <span class="fval">= ${PG.escapeHtml(formatValue(c.feature, c.value))}</span>
            <span class="fshap ${c.shap_phishing >= 0 ? "pos" : "neg"}">
                ${c.shap_phishing >= 0 ? "+" : ""}${c.shap_phishing.toFixed(3)}
            </span>
        </div>
    `).join("");

    renderShapChart(contributions);
}

function renderShapChart(contributions) {
    const canvas = PG.el("shapChart");
    if (!canvas) return;
    if (shapChartRef) shapChartRef.destroy();

    const top = [...contributions]
        .sort((a, b) => Math.abs(b.shap_phishing) - Math.abs(a.shap_phishing))
        .slice(0, 10);

    shapChartRef = new Chart(canvas.getContext("2d"), {
        type: "bar",
        data: {
            labels: top.map((c) => c.label),
            datasets: [{
                data: top.map((c) => c.shap_phishing),
                backgroundColor: top.map((c) =>
                    c.shap_phishing >= 0 ? "rgba(239,68,68,0.85)" : "rgba(34,197,94,0.85)"),
                hoverBackgroundColor: top.map((c) =>
                    c.shap_phishing >= 0 ? "#ef4444" : "#22c55e"),
                borderRadius: 4,
                barThickness: 15,
            }],
        },
        options: {
            indexAxis: "y",
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const v = ctx.parsed.x;
                            return `${Math.abs(v).toFixed(4)} toward ${v >= 0 ? "Phishing" : "Legitimate"}`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    grid: { color: "rgba(148,163,184,0.07)" },
                    border: { display: false },
                    ticks: { color: "#64748b" },
                },
                y: {
                    grid: { display: false },
                    border: { display: false },
                    ticks: { color: "#64748b", font: { size: 11 } },
                },
            },
        },
    });
}

function renderGlobal(featureImportance) {
    const canvas = PG.el("globalChart");
    if (!canvas) return;
    if (globalChartRef) globalChartRef.destroy();

    const top = featureImportance.slice(0, 10);

    globalChartRef = new Chart(canvas.getContext("2d"), {
        type: "bar",
        data: {
            labels: top.map((f) => f.label),
            datasets: [{
                data: top.map((f) => f.importance),
                backgroundColor: "rgba(59,130,246,0.75)",
                hoverBackgroundColor: "#60a5fa",
                borderRadius: 4,
                barThickness: 15,
            }],
        },
        options: {
            indexAxis: "y",
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `importance ${ctx.parsed.x.toFixed(4)}`,
                    },
                },
            },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: { color: "rgba(148,163,184,0.07)" },
                    border: { display: false },
                    ticks: { color: "#64748b" },
                },
                y: {
                    grid: { display: false },
                    border: { display: false },
                    ticks: { color: "#64748b", font: { size: 11 } },
                },
            },
        },
    });
}
