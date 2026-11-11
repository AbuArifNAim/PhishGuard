/* ============================================================
   PhishGuard AI — shared front-end helpers
   ============================================================ */

const PG = {
    el: (id) => document.getElementById(id),

    escapeHtml(value) {
        return String(value ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    },

    truncate(text, max = 64) {
        const s = String(text ?? "");
        return s.length > max ? s.slice(0, max - 1) + "…" : s;
    },

    hostOf(url) {
        try {
            return new URL(url.includes("://") ? url : "http://" + url).hostname;
        } catch {
            return String(url ?? "");
        }
    },

    statusKey(status) {
        return String(status || "unknown").toLowerCase();
    },

    statusPill(status) {
        const key = PG.statusKey(status);
        const label = status || "Unknown";
        return `<span class="pill ${key}"><i></i>${PG.escapeHtml(label)}</span>`;
    },

    riskBand(score) {
        if (score === null || score === undefined || Number.isNaN(Number(score))) {
            return { key: "neutral", label: "Not available", color: "#94a3b8" };
        }
        const s = Number(score);
        if (s >= 70) return { key: "high", label: "High risk", color: "#ef4444" };
        if (s >= 40) return { key: "medium", label: "Medium risk", color: "#f59e0b" };
        return { key: "low", label: "Low risk", color: "#22c55e" };
    },

    timeAgo(iso) {
        if (!iso) return "";
        const then = new Date(iso);
        if (Number.isNaN(then.getTime())) return "";
        const diff = Math.max(0, Date.now() - then.getTime());
        const mins = Math.floor(diff / 60000);
        if (mins < 1) return "just now";
        if (mins < 60) return `${mins}m ago`;
        const hours = Math.floor(mins / 60);
        if (hours < 24) return `${hours}h ago`;
        const days = Math.floor(hours / 24);
        if (days < 7) return `${days}d ago`;
        return then.toLocaleDateString();
    },

    toast(message) {
        const node = PG.el("toast");
        if (!node) return;
        node.textContent = message;
        node.classList.add("show");
        clearTimeout(PG._toastTimer);
        PG._toastTimer = setTimeout(() => node.classList.remove("show"), 2600);
    },

    async getJSON(url) {
        const res = await fetch(url, { headers: { Accept: "application/json" } });
        if (!res.ok) throw new Error(`Request failed (${res.status})`);
        return res.json();
    },

    download(filename, content, type = "application/json") {
        const blob = new Blob([content], { type });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    },
};

/* ---------- Chart.js global theme ---------- */

if (window.Chart) {
    Chart.defaults.color = "#93a6c6";
    Chart.defaults.font.family = "'Inter','Segoe UI',system-ui,sans-serif";
    Chart.defaults.font.size = 11;
    Chart.defaults.borderColor = "rgba(148,163,184,0.08)";
    Chart.defaults.plugins.legend.display = false;
    Chart.defaults.plugins.tooltip.backgroundColor = "rgba(10,16,30,0.96)";
    Chart.defaults.plugins.tooltip.borderColor = "rgba(148,163,184,0.22)";
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.titleColor = "#e8eefb";
    Chart.defaults.plugins.tooltip.bodyColor = "#93a6c6";
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
    Chart.defaults.plugins.tooltip.displayColors = true;
    Chart.defaults.animation.duration = 750;
    Chart.defaults.maintainAspectRatio = false;
}

/* ---------- Sidebar toggle (mobile) ---------- */

document.addEventListener("DOMContentLoaded", () => {
    const toggle = PG.el("sidebarToggle");
    const sidebar = PG.el("sidebar");
    if (toggle && sidebar) {
        toggle.addEventListener("click", (e) => {
            e.stopPropagation();
            sidebar.classList.toggle("open");
        });
        document.addEventListener("click", (e) => {
            if (sidebar.classList.contains("open") && !sidebar.contains(e.target) && e.target !== toggle) {
                sidebar.classList.remove("open");
            }
        });
    }
});

/* ---------- Theme (day / night) + font size ---------- */

const FONT_MIN = 0.85;
const FONT_MAX = 1.35;
/* One button cycles through these presets: 90% → 100% → 115% → 130% → 90% */
const FONT_STEPS = [0.9, 1.0, 1.15, 1.3];

function uiScale() {
    const raw = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--ui-scale"));
    return Number.isFinite(raw) && raw > 0 ? raw : 1;
}

document.addEventListener("DOMContentLoaded", () => {
    const themeBtn = PG.el("themeToggle");
    const fontBtn = PG.el("fontToggle");

    /* --- day / night --- */
    const syncThemeLabel = () => {
        if (!themeBtn) return;
        const light = document.documentElement.classList.contains("light");
        themeBtn.title = light ? "Switch to night mode" : "Switch to day mode";
        themeBtn.setAttribute("aria-label", themeBtn.title);
    };
    syncThemeLabel();

    themeBtn?.addEventListener("click", () => {
        document.documentElement.classList.toggle("light");
        const light = document.documentElement.classList.contains("light");
        try {
            localStorage.setItem("pg-theme", light ? "light" : "dark");
        } catch (e) {}
        syncThemeLabel();
        PG.toast(light ? "Day mode enabled" : "Night mode enabled");
    });

    /* --- font size (one button cycles through presets) --- */
    const syncFontLabel = (scale) => {
        if (!fontBtn) return;
        const pct = Math.round(scale * 100);
        fontBtn.title = `Font size ${pct}% — click to change`;
        fontBtn.setAttribute("aria-label", `Change font size (currently ${pct}%)`);
        fontBtn.dataset.scale = String(scale);
    };

    const applyScale = (scale) => {
        const clamped = Math.min(FONT_MAX, Math.max(FONT_MIN, Math.round(scale * 100) / 100));
        document.documentElement.style.setProperty("--ui-scale", clamped);
        try {
            localStorage.setItem("pg-font-scale", clamped);
        } catch (e) {}
        syncFontLabel(clamped);
        return clamped;
    };

    let currentScale = applyScale(uiScale());

    fontBtn?.addEventListener("click", () => {
        const next = FONT_STEPS.find((s) => s > currentScale + 0.001) ?? FONT_STEPS[0];
        currentScale = applyScale(next);
        PG.toast(`Font size ${Math.round(currentScale * 100)}%`);
    });
});
