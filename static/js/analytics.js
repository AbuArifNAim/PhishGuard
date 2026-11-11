/* ============================================================
   Analytics — model comparison, ablation, confusion, metrics
   ============================================================ */

const PRETTY_MODEL = {
    logistic_regression: "Logistic Regression",
    decision_tree: "Decision Tree",
    random_forest: "Random Forest",
    svm: "SVM",
    xgboost: "XGBoost",
};

const pretty = (name) => PRETTY_MODEL[name] || name;

document.addEventListener("DOMContentLoaded", async () => {
    let data;
    try {
        data = await PG.getJSON("/api/analytics");
    } catch (err) {
        PG.toast("Could not load analytics data");
        return;
    }

    renderKPIs(data);
    renderModelChart(data.models);
    renderTimeChart(data.models);
    renderAblation(data.ablation);
    renderMetricsTable(data.models);
    renderConfusion(data.confusion);
    renderCalibration(data.calibration);
});

function pct(value, digits = 2) {
    return (Number(value) * 100).toFixed(digits) + "%";
}

function renderKPIs(data) {
    const deployed = data.models.find((m) => m.model === "random_forest");
    const best = [...data.models].sort((a, b) => b.roc_auc - a.roc_auc)[0];

    PG.el("kpiDeployedAcc").textContent = deployed ? pct(deployed.accuracy) : "—";
    PG.el("kpiBestAuc").textContent = best ? pct(best.roc_auc) : "—";
    PG.el("kpiBestModel").textContent = best ? pretty(best.model) : "—";
    PG.el("kpiInference").textContent = deployed ? deployed.predict_time.toFixed(3) + " s" : "—";
    PG.el("kpiDataset").textContent = data.dataset_rows
        ? data.dataset_rows.toLocaleString()
        : "—";
}

function renderModelChart(models) {
    const canvas = PG.el("modelChart");
    if (!canvas) return;

    new Chart(canvas.getContext("2d"), {
        type: "bar",
        data: {
            labels: models.map((m) => pretty(m.model)),
            datasets: [
                {
                    label: "Accuracy",
                    data: models.map((m) => m.accuracy),
                    backgroundColor: "rgba(34,211,238,0.8)",
                    hoverBackgroundColor: "#22d3ee",
                    borderRadius: 4,
                    maxBarThickness: 20,
                },
                {
                    label: "F1",
                    data: models.map((m) => m.f1),
                    backgroundColor: "rgba(59,130,246,0.8)",
                    hoverBackgroundColor: "#3b82f6",
                    borderRadius: 4,
                    maxBarThickness: 20,
                },
                {
                    label: "ROC-AUC",
                    data: models.map((m) => m.roc_auc),
                    backgroundColor: "rgba(139,92,246,0.8)",
                    hoverBackgroundColor: "#8b5cf6",
                    borderRadius: 4,
                    maxBarThickness: 20,
                },
            ],
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${pct(ctx.parsed.y, 3)}`,
                    },
                },
            },
            scales: {
                x: {
                    grid: { display: false },
                    border: { display: false },
                    ticks: { color: "#64748b" },
                },
                y: {
                    min: 0.975,
                    max: 1.0,
                    border: { display: false },
                    grid: { color: "rgba(148,163,184,0.07)" },
                    ticks: {
                        color: "#64748b",
                        callback: (v) => (v * 100).toFixed(1) + "%",
                    },
                },
            },
        },
    });
}

function renderTimeChart(models) {
    const canvas = PG.el("timeChart");
    if (!canvas) return;

    const sorted = [...models].sort((a, b) => a.predict_time - b.predict_time);

    new Chart(canvas.getContext("2d"), {
        type: "bar",
        data: {
            labels: sorted.map((m) => pretty(m.model)),
            datasets: [{
                data: sorted.map((m) => m.predict_time),
                backgroundColor: sorted.map((m) =>
                    m.model === "random_forest" ? "rgba(34,211,238,0.9)" : "rgba(245,158,11,0.6)"),
                hoverBackgroundColor: "#f59e0b",
                borderRadius: 4,
                barThickness: 16,
            }],
        },
        options: {
            indexAxis: "y",
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.parsed.x.toFixed(3)} s per prediction`,
                    },
                },
            },
            scales: {
                x: {
                    beginAtZero: true,
                    border: { display: false },
                    grid: { color: "rgba(148,163,184,0.07)" },
                    ticks: {
                        color: "#64748b",
                        callback: (v) => v + "s",
                    },
                },
                y: {
                    grid: { display: false },
                    border: { display: false },
                    ticks: { color: "#64748b" },
                },
            },
        },
    });
}

function renderAblation(ablation) {
    const canvas = PG.el("ablationChart");
    if (!canvas || !ablation.length) return;

    const short = {
        "Model A - All Features": "A · All features",
        "Model B - Path Only": "B · Path only",
        "Model C - Without Path": "C · No path",
        "Model D - Without HTTPS": "D · No HTTPS",
    };

    new Chart(canvas.getContext("2d"), {
        type: "bar",
        data: {
            labels: ablation.map((a) => short[a.name] || a.name),
            datasets: [
                {
                    label: "Accuracy",
                    data: ablation.map((a) => a.accuracy),
                    backgroundColor: "rgba(34,211,238,0.8)",
                    borderRadius: 4,
                    maxBarThickness: 26,
                },
                {
                    label: "Macro F1",
                    data: ablation.map((a) => a.macro_f1),
                    backgroundColor: "rgba(139,92,246,0.8)",
                    borderRadius: 4,
                    maxBarThickness: 26,
                },
            ],
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${pct(ctx.parsed.y, 3)}`,
                    },
                },
            },
            scales: {
                x: {
                    grid: { display: false },
                    border: { display: false },
                    ticks: { color: "#64748b" },
                },
                y: {
                    min: 0.985,
                    max: 1.0,
                    border: { display: false },
                    grid: { color: "rgba(148,163,184,0.07)" },
                    ticks: {
                        color: "#64748b",
                        callback: (v) => (v * 100).toFixed(1) + "%",
                    },
                },
            },
        },
    });
}

function renderMetricsTable(models) {
    const body = PG.el("metricsBody");
    if (!body) return;

    body.innerHTML = models.map((m) => `
        <tr>
            <td style="font-weight:600;color:${m.model === "random_forest" ? "#7ff3ff" : "#e8eefb"}">
                ${PG.escapeHtml(pretty(m.model))}
                ${m.model === "random_forest" ? '<span class="pill legitimate" style="margin-left:8px"><i></i>deployed</span>' : ""}
            </td>
            <td>${pct(m.accuracy)}</td>
            <td>${pct(m.precision)}</td>
            <td>${pct(m.recall)}</td>
            <td>${pct(m.f1)}</td>
            <td>${pct(m.roc_auc)}</td>
            <td class="cell-mono">${m.train_time.toFixed(2)}</td>
            <td class="cell-mono">${m.predict_time.toFixed(3)}</td>
        </tr>
    `).join("");
}

function renderConfusion(confusion) {
    const select = PG.el("confusionSelect");
    if (!select || !confusion.length) return;

    select.innerHTML = confusion.map((c, i) =>
        `<option value="${i}">${PG.escapeHtml(pretty(c.model))}</option>`).join("");

    const draw = (index) => {
        const c = confusion[index];
        // The CSV stores sklearn's tn/fp/fn/tp with *Legitimate* as the positive
        // class. Re-map them so the matrix and metrics read with Phishing as
        // positive (which is what the labels say).
        const phishingCorrect = c.tn;   // actual phishing, predicted phishing
        const phishingMissed = c.fp;    // actual phishing, predicted legitimate
        const falseAlarm = c.fn;        // actual legitimate, predicted phishing
        const legitCorrect = c.tp;      // actual legitimate, predicted legitimate

        const precision = phishingCorrect + falseAlarm
            ? phishingCorrect / (phishingCorrect + falseAlarm) : 0;
        const recall = phishingCorrect + phishingMissed
            ? phishingCorrect / (phishingCorrect + phishingMissed) : 0;

        PG.el("matrixWrap").innerHTML = `
            <div></div>
            <div class="m-head">Pred. Phishing</div>
            <div class="m-head">Pred. Legitimate</div>

            <div class="m-side">Actual<br>Phishing</div>
            <div class="matrix-cell good"><span>True Positive</span><b>${phishingCorrect.toLocaleString()}</b></div>
            <div class="matrix-cell bad"><span>False Negative</span><b>${phishingMissed.toLocaleString()}</b></div>

            <div class="m-side">Actual<br>Legitimate</div>
            <div class="matrix-cell bad"><span>False Positive</span><b>${falseAlarm.toLocaleString()}</b></div>
            <div class="matrix-cell good"><span>True Negative</span><b>${legitCorrect.toLocaleString()}</b></div>
        `;

        PG.el("matrixPrecision").textContent = pct(precision);
        PG.el("matrixRecall").textContent = pct(recall);
        PG.el("matrixFp").textContent = falseAlarm.toLocaleString();
    };

    select.addEventListener("change", () => draw(Number(select.value)));

    // Default to the deployed model
    const deployedIndex = confusion.findIndex((c) => c.model === "random_forest");
    const startIndex = deployedIndex >= 0 ? deployedIndex : 0;
    select.value = String(startIndex);
    draw(startIndex);
}

function renderCalibration(calibration) {
    if (!calibration) return;

    const raw = calibration.raw || {};
    const adjusted = calibration.calibrated || {};
    const score = (value) => Number(value).toFixed(4);

    const set = (id, text) => {
        const node = PG.el(id);
        if (node) node.textContent = text;
    };

    set("calBrier", `${score(raw.brier)} → ${score(adjusted.brier)}`);
    set("calLogloss", `${score(raw.log_loss)} → ${score(adjusted.log_loss)}`);
    set("calEce", `${score(raw.ece_10bin)} → ${score(adjusted.ece_10bin)}`);

    const band = calibration.abstain_band || [0.3, 0.7];
    set(
        "calAbstain",
        `${(band[0] * 100).toFixed(0)}%–${(band[1] * 100).toFixed(0)}% · ` +
        `${pct(adjusted.abstain_rate || 0)} of test rows`
    );
}
