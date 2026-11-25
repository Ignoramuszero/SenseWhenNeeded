"""Generate the three core experiment figures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swn.config import ACTIVITIES, FIGURE_DIR, RESULT_DIR, TIMELINE_DIR  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    }
)

C_ALWAYS = "#9aa5b1"
C_PERIODIC = "#f0a45c"
C_OURS = "#2f7ed8"
C_ACCENT = "#d1495b"
C_SENSOR = "#5b8c5a"
C_IDLE = "#cfe0f3"


def load():
    comp = json.loads((RESULT_DIR / "experiments.json").read_text(encoding="utf-8"))
    train = json.loads((RESULT_DIR / "training_metrics.json").read_text(encoding="utf-8"))
    demo = json.loads((TIMELINE_DIR / f"demo_{comp['demo_session_id']}.json").read_text(encoding="utf-8"))
    return comp, train, demo


def fig_accuracy(comp):
    names = ["always_on", "periodic", "event_driven"]
    labels = ["Always-on\n20 Hz", "Periodic\n3 s / 10 s", "Event-driven\n(ours)"]
    colors = [C_ALWAYS, C_PERIODIC, C_OURS]
    acc = [comp["strategies"][n]["accuracy"] for n in names]
    f1 = [comp["strategies"][n]["macro_f1"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    x = np.arange(3)
    ax.bar(x - 0.19, acc, 0.36, color=colors, label="Accuracy")
    ax.bar(x + 0.19, f1, 0.36, color=colors, alpha=0.45, hatch="//", label="Macro F1")
    for i, (a, f) in enumerate(zip(acc, f1)):
        ax.text(i - 0.19, a + 0.008, f"{a:.3f}", ha="center", fontsize=7.5)
        ax.text(i + 0.19, f + 0.008, f"{f:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("Per-second tracking score")
    ax.set_title("Recognition quality (held-out subjects)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7, loc="lower right", ncol=2)

    # per-class F1
    ax = axes[1]
    w = 0.26
    x = np.arange(len(ACTIVITIES))
    for j, (n, c) in enumerate(zip(names, colors)):
        vals = [comp["strategies"][n]["per_class_f1"][a] for a in ACTIVITIES]
        ax.bar(x + (j - 1) * w, vals, w, color=c, label=labels[j].replace("\n", " "))
    ax.set_xticks(x, [a[:4] for a in ACTIVITIES], fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1", fontsize=9.5)
    ax.legend(frameon=False, fontsize=6.4, loc="lower left")
    fig.tight_layout()
    out = FIGURE_DIR / "fig1_accuracy.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_cost(comp):
    names = ["always_on", "periodic", "event_driven"]
    labels = ["Always-on", "Periodic", "Event-driven"]
    colors = [C_ALWAYS, C_PERIODIC, C_OURS]
    duty = [comp["strategies"][n]["counters"]["duty_cycle"] for n in names]
    samples = [comp["strategies"][n]["counters"]["samples"] / 1000 for n in names]
    energy = [comp["strategies"][n]["counters"]["energy_j_per_hour"] for n in names]
    gps = [comp["strategies"][n]["counters"]["gps_on_s"] for n in names]
    cpu = [comp["strategies"][n]["counters"]["cpu_ms_per_min"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))
    for ax, vals, title, unit, logy in (
        (axes[0], samples, "IMU samples collected", "k samples / h", False),
        (axes[1], energy, "Energy proxy", "J / h", False),
        (axes[2], cpu, "CPU time", "ms / min", False),
    ):
        bars = ax.bar(labels, vals, color=colors, width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:.2f}" if v < 20 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7.5)
        ax.set_title(title, fontsize=9.5)
        ax.set_ylabel(unit)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_ylim(0, max(vals) * 1.22)
    axes[1].text(2, energy[2] * 0.55, f"+{gps[2]:.0f} s/h GPS", ha="center", fontsize=6.5, color="#555")
    fig.tight_layout()
    out = FIGURE_DIR / "fig2_cost.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_timeline(demo):
    ev = demo["event_driven"]
    tl = ev["timeline"]
    t = np.array([r["t"] for r in tl])
    true = np.array([r["true"] for r in tl])
    pred = np.array([r["pred"] for r in tl])
    state = [r["state"] for r in tl]
    gps = np.array([r["gps"] for r in tl])
    prob = np.array([r["event_prob"] if r["event_prob"] is not None else np.nan for r in tl])

    fig, axes = plt.subplots(
        4, 1, figsize=(8.2, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0, 0.7, 0.8]}
    )

    def ribbon(ax, series, title):
        cmap = plt.get_cmap("tab10")
        for k in range(len(ACTIVITIES)):
            m = series == k
            if not m.any():
                continue
            ax.fill_between(t, 0, 1, where=m, step="mid", color=cmap(k), alpha=0.85, linewidth=0)
        ax.set_yticks([])
        ax.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=7.5)
        ax.grid(False)

    ribbon(axes[0], true, "truth")
    ribbon(axes[1], pred, "estimate")

    ax = axes[2]
    active = np.array([s == "active" for s in state])
    idle = np.array([s == "idle" for s in state])
    ax.fill_between(t, 0, 1, where=idle, step="mid", color=C_IDLE)
    ax.fill_between(t, 0, 1, where=active, step="mid", color=C_OURS, alpha=0.85)
    for tr in ev["triggers"]:
        ax.axvline(tr["t"], color=C_ACCENT, lw=0.7, alpha=0.7)
    ax.set_yticks([])
    ax.set_ylabel("state", rotation=0, ha="right", va="center", fontsize=7.5)
    ax.grid(False)
    ax.text(0.01, 0.82, "idle = 1 s burst / 6 s", transform=ax.transAxes, fontsize=6, color="#4a5b6b")
    ax.text(0.99, 0.82, "active = 20 Hz", transform=ax.transAxes, fontsize=6, color="#123c63", ha="right")

    ax = axes[3]
    ax.plot(t, prob, color=C_OURS, lw=0.9, label="P(change)")
    ax.axhline(0.5, color=C_ACCENT, ls="--", lw=0.8, label="operating point")
    for tr in ev["llm_calls"]:
        ax.axvline(tr["t"], color=C_SENSOR, lw=1.0, ls=":", alpha=0.9)
    for tt in ev["transitions"]:
        ax.axvline(tt["t"], color="k", lw=0.6, alpha=0.35)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(change)", fontsize=7.5)
    ax.set_xlabel("time (s)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=2)
    ax.text(0.005, -0.42, "dotted green = LLM call   |   grey = true transition",
            transform=ax.transAxes, fontsize=6, color="#555")

    handles = [plt.Rectangle((0, 0), 1, 1, color=plt.get_cmap("tab10")(k)) for k in range(len(ACTIVITIES))]
    fig.legend(handles, ACTIVITIES, ncol=5, frameon=False, fontsize=7, loc="upper center",
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    out = FIGURE_DIR / "fig3_timeline.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_events(train, comp):
    ev = comp["strategies"]["event_driven"]["events"]
    types = ["static2moving", "moving2moving", "moving2static", "static2static"]
    types = [t for t in types if t in ev["recall_by_type"]]
    rec = [ev["recall_by_type"][t]["recall"] for t in types]
    n = [ev["recall_by_type"][t]["n"] for t in types]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    bars = ax.bar([t.replace("2", "→") for t in types], rec, color=[C_OURS, C_PERIODIC, C_SENSOR, C_ALWAYS])
    for b, r, c in zip(bars, rec, n):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.02, f"{r:.2f}\n(n={c})", ha="center", fontsize=6.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall")
    ax.set_title("Transition recall by type", fontsize=9.5)
    ax.tick_params(axis="x", labelsize=7)

    ax = axes[1]
    table = train["event_detector"]["threshold_tuning"]
    labels_ = [f'rf{r["rf_threshold"]:.1f}\nz={r["rule_z"]:.0f}' for r in table]
    xs = np.arange(len(table))
    ax.plot(xs, [r["precision"] for r in table], "o-", color=C_ACCENT, ms=3, label="precision")
    ax.plot(xs, [r["recall"] for r in table], "s-", color=C_OURS, ms=3, label="recall")
    ax.plot(xs, [r["f1"] for r in table], "^-", color=C_SENSOR, ms=3, label="F1")
    sel = train["event_detector"]["selected"]
    sel_x = next(i for i, r in enumerate(table)
                 if r["rf_threshold"] == sel["rf_threshold"] and r["rule_z"] == sel["rule_z"])
    ax.axvline(sel_x, color="k", ls=":", lw=0.8)
    ax.set_xticks(xs, labels_, fontsize=5.5, rotation=60)
    ax.set_xlabel("RF threshold / rule z")
    ax.set_ylim(0, 1)
    ax.set_title("Threshold sweep (validation)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    out = FIGURE_DIR / "fig4_events.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    comp, train, demo = load()
    outs = [fig_accuracy(comp), fig_cost(comp), fig_timeline(demo), fig_events(train, comp)]
    for o in outs:
        print("wrote", o)


if __name__ == "__main__":
    main()
