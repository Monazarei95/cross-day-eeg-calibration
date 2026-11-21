"""Plot all participants and the locked primary estimate from a completed score run."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_controls(summary, output):
    comparisons = [
        ("raw", "Raw EEG"),
        ("random", "Random projections"),
        ("pca", "Calibration EEG PCA"),
    ]
    fig, ax = plt.subplots(figsize=(7.3, 3.7), layout="constrained")
    ax.axvline(0, color="0.45", lw=1)
    for method, label, marker, color, offset in [
        ("frozen_ssl", "Frozen SSL encoder", "o", "#116487", 0.12),
        ("classical", "Classical decoder", "s", "#aa511e", -0.12),
    ]:
        for i, (name, _) in enumerate(comparisons):
            effect = summary[method]["paired_differences"]["ocular_minus_" + name]
            mean = 100 * effect["mean"]
            lo, hi = 100 * np.array(effect["ci95"])
            ax.errorbar(
                mean,
                2 - i + offset,
                xerr=[[mean - lo], [hi - mean]],
                fmt=marker,
                capsize=3,
                color=color,
                label=label if i == 0 else None,
            )
    ax.set_yticks([2, 1, 0], [label for _, label in comparisons])
    ax.set_ylim(-0.45, 2.55)
    ax.set_xlabel("Eye-calibrated minus comparison accuracy (percentage points)")
    ax.set_title("Day-two balanced accuracy on BCI IV 2a", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--controls", type=Path, help="Output path for calibrated-minus-control contrasts"
    )
    args = parser.parse_args()
    with (args.scores / "participants.csv").open(newline="") as source:
        rows = [r for r in csv.DictReader(source) if r["method"] == "frozen_ssl"]
    summary = json.loads((args.scores / "summary.json").read_text())
    if args.controls is not None:
        plot_controls(summary, args.controls)
    primary = summary["frozen_ssl"]["paired_differences"]["ocular_minus_raw"]
    differences = [100 * (float(r["ocular"]) - float(r["raw"])) for r in rows]
    mean = 100 * primary["mean"]
    lower, upper = np.array(primary["ci95"]) * 100
    fig, ax = plt.subplots(figsize=(6.6, 4.1), layout="constrained")
    y = np.arange(len(rows), 0, -1)
    ax.axvline(0, color="0.5", linewidth=0.9)
    ax.hlines(y, 0, differences, color="#83adc2", linewidth=1.8)
    ax.scatter(differences, y, color="#116487", s=33, zorder=3)
    ax.errorbar(
        mean,
        -0.3,
        xerr=[[mean - lower], [upper - mean]],
        fmt="D",
        capsize=4,
        color="#182f3a",
        markersize=5,
    )
    ax.set_yticks([*y, -0.3], [f"A{int(r['participant']):02d}" for r in rows] + ["Mean (95% CI)"])
    ax.set_xlabel("Change in day-two balanced accuracy (percentage points)")
    ax.set_title(
        "Eye-calibrated versus raw EEG in the frozen SSL decoder", loc="left", fontsize=13
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.margins(y=0.09, x=0.08)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
