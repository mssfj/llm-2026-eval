"""Plots for activation precision sweeps; requires the optional plots extra."""
import argparse
import csv
import json
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
    args = parser.parse_args()
    manifest = json.loads((args.report_dir / "manifest.json").read_text())
    aggregate = read_csv(args.report_dir / "aggregate.csv")
    diagnostics = read_csv(args.report_dir / "activation_diagnostics.csv")
    num_params = int(read_csv(args.report_dir / "per_seed.csv")[0]["num_params"])
    precisions = manifest["precisions"]
    seeds = manifest["seeds"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
    colors = plt.get_cmap("tab10").colors
    for index, precision in enumerate(precisions):
        curves = []
        for seed in seeds:
            path = Path(manifest["output_dir"]) / f"{precision}-seed{seed}" / "metrics.csv"
            curves.append([r for r in read_csv(path) if r["val_accuracy"]])
        xs = [int(r["step"]) for r in curves[0]]
        for ax, field, factor, title in ((axes[0, 0], "val_accuracy", 100, "Validation accuracy (%)"),
                                         (axes[0, 1], "val_loss", 1, "Validation loss")):
            values = np.array([[float(r[field])*factor for r in curve] for curve in curves])
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(mean)
            ax.plot(xs, mean, label="W3"+precision.upper(), color=colors[index])
            ax.fill_between(xs, mean-std, mean+std, alpha=0.1, color=colors[index])
            ax.set(xlabel="Accumulation epoch", title=title)
            ax.grid(alpha=0.2)
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].legend(fontsize=8)
    positions = np.arange(len(precisions))
    ordered = [next(r for r in aggregate if r["precision"] == p) for p in precisions]
    mean = [100*float(r["test_accuracy_mean"]) for r in ordered]
    std = [100*float(r["test_accuracy_std"]) for r in ordered]
    axes[1, 0].bar(positions, mean, yerr=std, capsize=4, color=colors[:len(precisions)])
    axes[1, 0].set(xticks=positions, xticklabels=["W3"+p.upper() for p in precisions],
                   ylim=(0, 100), title="Final test accuracy (%)")
    for i, (m, s) in enumerate(zip(mean, std)):
        axes[1, 0].text(i, m+s+2, f"{m:.2f}", ha="center", fontsize=9)
    for layer, offset, label in ((0, -.18, "Pooled input"), (1, .18, "Post-ReLU hidden")):
        means, stds = [], []
        for precision in precisions:
            values = [100*float(r["zero_fraction"]) for r in diagnostics
                      if r["precision"] == precision and r["stage"] == "final" and int(r["layer"]) == layer]
            means.append(np.mean(values))
            stds.append(np.std(values, ddof=1) if len(values) > 1 else 0)
        axes[1, 1].bar(positions+offset, means, width=.36, yerr=stds, capsize=3, label=label)
    axes[1, 1].set(xticks=positions, xticklabels=["W3"+p.upper() for p in precisions],
                   ylim=(0, 100), title="Final activation zero fraction (%)")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(f"{num_params:,} ternary weights / block {manifest['block_size']} / threshold {manifest['threshold']} / {len(seeds)} seeds\n"
                 "Linear-input precision changes; FP32 accumulation. Bands/error bars: sample SD.")
    for extension in ("png", "svg"):
        fig.savefig(args.report_dir / ("activation_comparison."+extension), dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
