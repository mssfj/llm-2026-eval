"""Optional matplotlib figures for completed sweeps (PNG and SVG)."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    args = parser.parse_args()
    rows = read_csv(args.report_dir / "per_seed.csv")
    blocks = sorted({int(r["block_size"]) for r in rows})
    thresholds = sorted({int(r["threshold"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})

    def save(fig, name):
        fig.savefig(args.report_dir / (name + ".png"), dpi=160, bbox_inches="tight")
        fig.savefig(args.report_dir / (name + ".svg"), bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for ax, metric, title, multiplier in zip(
        axes, ("val_accuracy", "total_fires", "counter_abs_mean"),
        ("Validation accuracy (%)", "Total coordinate updates", "Mean absolute counter"), (100, 1, 1)
    ):
        means = np.zeros((len(blocks), len(thresholds)))
        stds = np.zeros_like(means)
        for i, block in enumerate(blocks):
            for j, threshold in enumerate(thresholds):
                values = [float(r[metric]) * multiplier for r in rows
                          if int(r["block_size"]) == block and int(r["threshold"]) == threshold]
                means[i, j] = statistics.mean(values)
                stds[i, j] = statistics.stdev(values) if len(values) > 1 else 0
        im = ax.imshow(means, cmap="Blues", aspect="auto", vmin=0)
        for i in range(len(blocks)):
            for j in range(len(thresholds)):
                ax.text(j, i, f"{means[i,j]:.1f}\n±{stds[i,j]:.1f}", ha="center", va="center",
                        color="white" if means[i,j] > means.max() * 0.6 else "black", fontsize=9)
        ax.set(xticks=range(len(thresholds)), xticklabels=thresholds,
               yticks=range(len(blocks)), yticklabels=blocks, xlabel="Firing threshold", ylabel="Block size", title=title)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f"MNIST TDT-D / 1,000 weights / mean ± sample SD over {len(seeds)} seeds")
    save(fig, "comparison")

    hist = read_csv(args.report_dir / "counter_histograms.csv")
    fig, axes = plt.subplots(len(blocks), len(thresholds), figsize=(14, 8), squeeze=False, layout="constrained")
    for i, block in enumerate(blocks):
        for j, threshold in enumerate(thresholds):
            ax = axes[i, j]
            for seed in seeds:
                members = [r for r in hist if int(r["block_size"]) == block
                           and int(r["threshold"]) == threshold and int(r["seed"]) == seed]
                members.sort(key=lambda r: int(r["counter_value"]))
                ax.step([int(r["counter_value"]) for r in members], [float(r["fraction"]) for r in members],
                        where="mid", label=f"seed {seed}", alpha=0.8)
            ax.axvline(-threshold, color="gray", linestyle=":", linewidth=1)
            ax.axvline(threshold, color="gray", linestyle=":", linewidth=1)
            ax.set(title=f"block {block}, threshold {threshold}", xlabel="Signed counter C", ylabel="Fraction")
            ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Measured edges before reset, pooled over epochs; dotted lines: ±threshold (not saturation)")
    save(fig, "counter_distributions")

    manifest = json.loads((args.report_dir / "manifest.json").read_text())
    run_dir = Path(manifest["output_dir"])
    fig, axes = plt.subplots(len(blocks), len(thresholds), figsize=(14, 8), squeeze=False, layout="constrained")
    for i, block in enumerate(blocks):
        for j, threshold in enumerate(thresholds):
            ax = axes[i, j]
            for seed in seeds:
                data = read_csv(run_dir / f"seed{seed}-block{block}-threshold{threshold}" / "metrics.csv")
                data = [r for r in data if r["val_accuracy"]]
                ax.plot([int(r["step"]) for r in data], [100*float(r["val_accuracy"]) for r in data],
                        label=f"seed {seed}", linewidth=1.3)
            ax.set(title=f"block {block}, threshold {threshold}", xlabel="Accumulation epoch", ylabel="Val accuracy (%)", ylim=(0, 100))
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Validation learning curves; same K, steps and minibatch size in every run")
    save(fig, "learning_curves")


if __name__ == "__main__":
    main()
