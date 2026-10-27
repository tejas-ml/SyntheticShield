from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _ensure_out(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_main_pipeline(out_dir: str | Path) -> Path:
    out = _ensure_out(out_dir)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axis("off")

    boxes = [
        (0.04, 0.35, 0.18, 0.28, "Private dataset\n+ inserted canaries", "Train records, outliers,\nrare groups"),
        (0.30, 0.35, 0.18, 0.28, "Synthetic generator", "No defense / DP-inspired\ntraining / deduplication"),
        (0.56, 0.35, 0.18, 0.28, "Synthetic data", "Generated rows used\nfor downstream tasks"),
        (0.82, 0.35, 0.15, 0.28, "Privacy auditor", "NN leakage, MIA,\ncanary exposure"),
    ]

    for x, y, w, h, title, subtitle in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=1.5,
            edgecolor="black",
            facecolor="white",
        )
        ax.add_patch(patch)
        ax.text(x + w/2, y + h*0.62, title, ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(x + w/2, y + h*0.27, subtitle, ha="center", va="center", fontsize=10)

    for x1, x2 in [(0.22, 0.30), (0.48, 0.56), (0.74, 0.82)]:
        arr = FancyArrowPatch((x1, 0.49), (x2, 0.49), arrowstyle="-|>", mutation_scale=18, linewidth=1.5)
        ax.add_patch(arr)

    ax.text(0.5, 0.83, "SyntheticShield Audit Pipeline", ha="center", fontsize=18, fontweight="bold")
    ax.text(0.5, 0.17, "Outputs: utility score, membership leakage rate, canary exposure by rarity", ha="center", fontsize=12)
    path = out / "main.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _bar_plot(df: pd.DataFrame, value: str, ylabel: str, title: str, path: Path, ylim: tuple[float, float] | None = None):
    summary = df.groupby("defense", as_index=False).agg(mean=(value, "mean"), std=(value, "std"))
    order = [d for d in ["no_defense", "dp_training", "deduplication"] if d in set(summary["defense"])]
    summary["defense"] = pd.Categorical(summary["defense"], categories=order, ordered=True)
    summary = summary.sort_values("defense")
    labels = [s.replace("_", "\n") for s in summary["defense"].astype(str)]

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.bar(labels, summary["mean"], yerr=summary["std"].fillna(0), capsize=5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_result_panels(results_csv: str | Path, canary_csv: str | Path, out_dir: str | Path):
    out = _ensure_out(out_dir)
    results = pd.read_csv(results_csv)
    canaries = pd.read_csv(canary_csv)

    _bar_plot(
        results,
        value="utility_auc",
        ylabel="TSTR utility AUC",
        title="Panel A: Utility of synthetic data",
        path=out / "figure2.png",
        ylim=(0.45, 1.0),
    )
    _bar_plot(
        results,
        value="leakage_rate_at_5pct_fpr",
        ylabel="Leakage rate @ 5% FPR",
        title="Panel B: Membership leakage rate",
        path=out / "figure3.png",
        ylim=(0.0, 1.0),
    )

    grouped = canaries.groupby(["defense", "rarity"], as_index=False).agg(
        exposure=("exposed", "mean"),
        exposure_score=("exposure_score", "mean"),
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for defense, part in grouped.groupby("defense"):
        part = part.sort_values("rarity")
        ax.plot(part["rarity"], part["exposure"], marker="o", label=defense.replace("_", " "))
    ax.set_xlabel("Canary rarity strength")
    ax.set_ylabel("Canary exposure rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Panel C: Canary exposure vs rarity")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.savefig(out / "figure4.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return [out / "figure2.png", out / "figure3.png", out / "figure4.png"]
