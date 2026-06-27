#!/usr/bin/env python3
"""
Generate the Stage 0 report charts (SVG) from the benchmark results JSON.

Reproducible: rerun after new benchmark runs to refresh the figures.
    pip install -r charts/requirements.txt
    python charts/make_charts.py            # writes charts/*.svg

Outputs (referenced from results/REPORT.md):
    charts/degradation_3to4b.svg   - faithful num_gpu degradation, 3-4B models
    charts/degradation_small.svg   - faithful num_gpu degradation, sub-1B tier
    charts/strict_json_passes.svg  - strict-JSON pass count across conditions
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent          # sandbox/
RESULTS = ROOT / "results"
OUT = ROOT / "charts"

# Tidepool categorical colours (fixed per model, never by rank)
STYLE = {
    "qwen2.5:3b":   ("#2a78d6", "-",  3.0),
    "gemma3:4b":    ("#1baf7a", "--", 2.0),
    "llama3.2:3b":  ("#eda100", ":",  2.0),
    "phi3.5":       ("#e34948", "-.", 2.0),
    "qwen2.5:0.5b": ("#2a78d6", "-",  3.0),
    "gemma3:270m":  ("#1baf7a", "--", 2.0),
    "llama3.2:1b":  ("#eda100", ":",  2.0),
}


def load(label: str) -> dict:
    f = RESULTS / label / "results.json"
    if not f.exists():
        return {}
    return {m["model"]: m for m in json.loads(f.read_text())["models"]}


def _tok(d: dict, model: str):
    return (d.get(model) or {}).get("mean_tokens_per_sec")


def _base_axes(ax, title: str, xlabel: str):
    ax.set_title(title, fontsize=13, loc="left", color="#0b0b0b", pad=10)
    ax.set_xlabel(xlabel, fontsize=11, color="#52514e")
    ax.set_ylabel("tokens / sec", fontsize=11, color="#52514e")
    ax.grid(axis="y", color="#e1e0d9", linewidth=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=10)


def line_chart(out, title, xlabel, x_labels, series_labels, data_by_label):
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    x = list(range(len(x_labels)))
    for model in series_labels:
        ys = [_tok(data_by_label[k], model) for k in data_by_label]
        color, dash, lw = STYLE[model]
        ax.plot(x, ys, dash, color=color, linewidth=lw, marker="o",
                markersize=5, label=model)
    _base_axes(ax, title, xlabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=10, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


def degradation_big():
    order = {"Uncapped": load("metal-clean"), "8 GB": load("metal-budget8"),
             "4 GB": load("metal-budget4"), "2 GB": load("metal-budget2")}
    line_chart(
        OUT / "degradation_3to4b.svg",
        "Faithful VRAM-budget degradation (3-4B, num_gpu offload)",
        "simulated VRAM budget",
        list(order.keys()),
        ["qwen2.5:3b", "gemma3:4b", "llama3.2:3b", "phi3.5"],
        order,
    )


def degradation_small():
    order = {"8 GB": load("metal-small-budget8"), "4 GB": load("metal-small-budget4"),
             "2 GB": load("metal-small-budget2"), "1 GB": load("metal-small-budget1"),
             "0.5 GB": load("metal-small-budget05")}
    line_chart(
        OUT / "degradation_small.svg",
        "Faithful VRAM-budget degradation (sub-1B tier)",
        "simulated VRAM budget",
        list(order.keys()),
        ["qwen2.5:0.5b", "gemma3:270m", "llama3.2:1b"],
        order,
    )


def strict_json_passes():
    conds = ["metal-quality-clean", "metal-quality-budget8",
             "metal-quality-budget4", "metal-quality-budget2"]
    loaded = [load(c) for c in conds]
    models = ["qwen2.5:3b", "gemma3:4b", "llama3.2:3b", "phi3.5"]
    counts = []
    for m in models:
        c = sum(1 for d in loaded
                if (d.get(m) or {}).get("quality_summary", {}).get("strict_json_pass"))
        counts.append(c)

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    colors = [STYLE[m][0] for m in models]
    bars = ax.bar(models, counts, color=colors, width=0.6, zorder=3)
    ax.bar_label(bars, labels=[f"{c}/{len(conds)}" for c in counts],
                 padding=4, fontsize=11, color="#0b0b0b")
    ax.set_title("Strict-JSON pass rate across VRAM conditions (higher is better)",
                 fontsize=13, loc="left", color="#0b0b0b", pad=10)
    ax.set_ylabel("conditions passing strict JSON", fontsize=11, color="#52514e")
    ax.set_ylim(0, len(conds))
    ax.set_yticks(range(len(conds) + 1))
    ax.grid(axis="y", color="#e1e0d9", linewidth=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "strict_json_passes.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT / "strict_json_passes.svg").relative_to(ROOT))


def main():
    OUT.mkdir(exist_ok=True)
    degradation_big()
    degradation_small()
    strict_json_passes()


if __name__ == "__main__":
    main()
