/* ============================================================
   Dashboard — KPIs, activity chart, verdict split, gauge, recent
   ============================================================ */

document.addEventListener("DOMContentLoaded", async () => {
    let stats;
    try {
        stats = await PG.getJSON("/api/stats");
    } catch (err) {
        PG.toast("Could not load dashboard data");
        return;
    }

    renderKPIs(stats);
    renderActivity(stats);
    renderVerdict(stats);
    renderGauge(stats);
    renderRecent(stats);
});

function renderKPIs(s) {
    PG.el("kpiTotal").textContent = s.total_scans.toLocaleString();
    PG.el("kpiPhishing").textContent = s.phishing.toLocaleString();
    PG.el("kpiLegitimate").textContent = s.legitimate.toLocaleString();

    PG.el("kpiTotalDelta").textContent = s.scans_last_7_days;
    PG.el("kpiPhishingPct").textContent =
        s.total_scans ? Math.round((s.phishing / s.total_scans) * 100) + "%" : "0%";
    PG.el("kpiLegitPct").textContent =
        s.total_scans ? Math.round((s.legitimate / s.total_scans) * 100) + "%" : "0%";

    const conf = PG.el("kpiConfidence");
    conf.textContent = s.avg_confidence !== null ? s.avg_confidence + "%" : "—";
}

function barGradient(ctx, from, to) {
    const g = ctx.createLinearGradient(0, 0, 0, 320);
    g.addColorStop(0, from);
    g.addColorStop(1, to);
    return g;
}

function renderActivity(s) {
    const canvas = PG.el("activityChart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: s.activity.labels,
            datasets: [
                {
                    label: "Legitimate",
                    data: s.activity.legitimate,
                    backgroundColor: barGradient(ctx, "#60a5fa", "#2563eb"),
                    hoverBackgroundColor: "#93c5fd",
                    borderRadius: 5,
                    maxBarThickness: 26,
                    stack: "scans",
                },
                {
                    label: "Phishing",
                    data: s.activity.phishing,
                    backgroundColor: barGradient(ctx, "#f87171", "#dc2626"),
                    hoverBackgroundColor: "#fca5a5",
                    borderRadius: 5,
                    maxBarThickness: 26,
                    stack: "scans",
                },
                {
                    label: "Other",
                    data: s.activity.other,
                    backgroundColor: "rgba(100,116,139,0.55)",
                    hoverBackgroundColor: "#94a3b8",
                    borderRadius: 5,
                    maxBarThickness: 26,
                    stack: "scans",
                },
            ],
        },
        options: {
            responsive: true,
            interaction: { mode: "index", intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: { mode: "index" },
            },
            scales: {
                x: {
                    stacked: true,
                    grid: { display: false },
                    border: { display: false },
                    ticks: { color: "#64748b" },
                },
                y: {
                    stacked: true,
                    beginAtZero: true,
                    border: { display: false },
                    ticks: { precision: 0, color: "#64748b" },
                    grid: { color: "rgba(148,163,184,0.07)" },
                },
            },
        },
    });
}

function renderVerdict(s) {
    const canvas = PG.el("verdictChart");
    if (!canvas) return;

    PG.el("verdictTotal").textContent = s.total_scans.toLocaleString();
    PG.el("legendPhishing").textContent = s.phishing;
    PG.el("legendLegitimate").textContent = s.legitimate;
    PG.el("legendOther").textContent = s.unreachable + s.unknown;

    const values = [s.phishing, s.legitimate, s.unreachable + s.unknown];
    const hasData = values.some((v) => v > 0);

    new Chart(canvas.getContext("2d"), {
        type: "doughnut",
        data: {
            labels: ["Phishing", "Legitimate", "Other"],
            datasets: [{
                data: hasData ? values : [1, 1, 1],
                backgroundColor: hasData
                    ? ["#ef4444", "#22c55e", "#f59e0b"]
                    : ["rgba(148,163,184,0.14)", "rgba(148,163,184,0.10)", "rgba(148,163,184,0.07)"],
                borderColor: document.documentElement.classList.contains("light") ? "#ffffff" : "#0d1425",
                borderWidth: 4,
                hoverOffset: 4,
            }],
        },
        options: {
            responsive: true,
            cutout: "74%",
            plugins: {
                legend: { display: false },
                tooltip: { enabled: hasData },
            },
        },
    });
}

function renderGauge(s) {
    const canvas = PG.el("gaugeChart");
    if (!canvas) return;

    const score = s.security_score;
    const hasData = score !== null;
    const value = hasData ? score : 0;
    const color = !hasData ? "#334155" : score >= 80 ? "#22c55e" : score >= 60 ? "#f59e0b" : "#ef4444";

    PG.el("gaugeScore").textContent = hasData ? score : "—";

    const note = PG.el("gaugeNote");
    if (!hasData) {
        note.textContent = "Scan a URL to compute your security score. Higher is safer.";
    } else if (score >= 80) {
        note.textContent = `${score}% of classified scans were legitimate — your recent traffic looks healthy.`;
    } else if (score >= 60) {
        note.textContent = `${score}% of classified scans were legitimate — keep an eye on suspicious links.`;
    } else {
        note.textContent = `Only ${score}% of classified scans were legitimate — a large share of tested URLs were phishing.`;
    }

    new Chart(canvas.getContext("2d"), {
        type: "doughnut",
        data: {
            labels: ["Score", "Remaining"],
            datasets: [{
                data: hasData ? [value, 100 - value] : [100, 0],
                backgroundColor: [color, "rgba(148,163,184,0.10)"],
                borderWidth: 0,
                hoverBackgroundColor: [color, "rgba(148,163,184,0.10)"],
            }],
        },
        options: {
            responsive: true,
            rotation: -90,
            circumference: 180,
            cutout: "76%",
            plugins: {
                legend: { display: false },
                tooltip: { enabled: hasData },
            },
        },
    });
}

function renderRecent(s) {
    const body = PG.el("recentBody");
    if (!body) return;

    if (!s.recent.length) {
        body.innerHTML = `<tr><td colspan="5" class="empty-note">
            No scans yet — <a href="/scanner" style="color:#67e8f9">run your first scan</a> to populate the dashboard.
        </td></tr>`;
        return;
    }

    body.innerHTML = s.recent.map((r) => {
        const risk = r.risk_score === null || r.risk_score === undefined ? "—" : r.risk_score;
        const band = PG.riskBand(r.risk_score);
        return `<tr>
            <td><div class="cell-url" title="${PG.escapeHtml(r.url)}">${PG.escapeHtml(PG.truncate(r.url, 52))}</div>
                <div class="cell-mono" style="font-size:10.5px">${PG.escapeHtml(PG.hostOf(r.url))}</div></td>
            <td>${PG.statusPill(r.status)}</td>
            <td>${r.confidence !== null && r.confidence !== undefined ? r.confidence + "%" : "—"}</td>
            <td><span style="color:${band.color};font-weight:650">${risk}</span></td>
            <td class="cell-mono">${PG.escapeHtml(PG.timeAgo(r.timestamp))}</td>
        </tr>`;
    }).join("");
}
