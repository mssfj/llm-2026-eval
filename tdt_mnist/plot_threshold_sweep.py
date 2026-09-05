"""Plot a dense firing-threshold sweep without a 32-column panel layout."""
import argparse
import csv
from pathlib import Path

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
    parser.add_argument("--block-size", type=int, default=8)
    args = parser.parse_args()
    rows = [r for r in read_csv(args.report_dir / "per_seed.csv") if int(r["block_size"]) == args.block_size]
    if not rows:
        parser.error("no results for the requested block size")
    thresholds = sorted({int(r["threshold"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for ax, metric, factor, title in zip(
        axes.flat, ("val_accuracy", "val_loss", "total_fires"), (100, 1, 1),
        ("Validation accuracy (%)", "Validation loss", "Total coordinate updates")
    ):
        values = np.array([[float(next(r for r in rows if int(r["threshold"]) == t and int(r["seed"]) == seed)[metric])
                            * factor for t in thresholds] for seed in seeds])
        means = values.mean(axis=0)
        std = values.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(means)
        for seed, ys in zip(seeds, values):
            ax.plot(thresholds, ys, color="gray", linewidth=0.7, alpha=0.4)
        ax.plot(thresholds, means, "o-", markersize=3, color="tab:blue", label="Mean")
        ax.fill_between(thresholds, means-std, means+std, color="tab:blue", alpha=0.15, label="± sample SD")
        ax.set(xlabel="Firing threshold", title=title)
        ax.set_xticks(thresholds[::2])
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].legend(fontsize=8)
    hist = [r for r in read_csv(args.report_dir / "counter_histograms.csv") if int(r["block_size"]) == args.block_size]
    limit = max(abs(int(r["counter_value"])) for r in hist)
    probabilities = np.zeros((2 * limit + 1, len(thresholds)))
    for row in hist:
        probabilities[int(row["counter_value"]) + limit, thresholds.index(int(row["threshold"]))] += float(row["fraction"]) / len(seeds)
    assert np.allclose(probabilities.sum(axis=0), 1)
    ax = axes[1, 1]
    im = ax.imshow(probabilities, origin="lower", aspect="auto", cmap="magma",
                   extent=(0.5, len(thresholds)+0.5, -limit-0.5, limit+0.5))
    ax.set_xticks(np.arange(1, len(thresholds)+1)[::2], thresholds[::2])
    ax.set(xlabel="Firing threshold", ylabel="Signed counter C", title="Counter distribution before reset")
    fig.colorbar(im, ax=ax, label="Mean observed fraction across seeds")
    fig.suptitle(f"MNIST TDT-D / block {args.block_size} / {len(seeds)} seeds / equal measurement budget")
    for extension in ("png", "svg"):
        fig.savefig(args.report_dir / f"threshold_comparison.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
