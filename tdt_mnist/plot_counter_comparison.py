"""Plot validation trajectories and update counts for the counter ablation."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.report_dir / "manifest.json").read_text())
    results = json.loads((args.report_dir / "summaries.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for method, label, color in (("counter", "Counter (threshold 8)", "tab:blue"),
                                 ("no_counter", "No counter (immediate)", "tab:orange")):
        # Label supports rerunning the comparison with a different threshold.
        if method == "counter":
            label = f"Counter (threshold {manifest['threshold']})"
        series = []
        for seed in manifest["seeds"]:
            path = Path(manifest["output_dir"]) / f"{method}-seed{seed}" / "metrics.csv"
            with path.open() as handle:
                series.append([r for r in csv.DictReader(handle) if r["val_accuracy"]])
        xs = np.array([int(r["train_forward_calls"]) / 1000 for r in series[0]])
        for ax, field, multiplier, title in zip(axes, ("val_accuracy", "val_loss", "total_fires"),
                                              (100, 1, 1), ("Validation accuracy (%)", "Validation loss", "Cumulative updates")):
            ys = np.array([[float(r[field]) * multiplier for r in rows] for rows in series])
            mean = ys.mean(axis=0)
            std = ys.std(axis=0, ddof=1) if len(ys) > 1 else np.zeros_like(mean)
            ax.plot(xs, mean, label=label, color=color)
            ax.fill_between(xs, mean-std, mean+std, color=color, alpha=0.15)
            ax.set(xlabel="Training forward calls (thousands)", title=title)
            ax.grid(alpha=0.2)
    axes[0].set_ylim(0, 100)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"MNIST / {results[0]['num_params']:,} ternary weights / block {manifest['block_size']} / mean ± sample SD over {len(manifest['seeds'])} seeds")
    for suffix in ("png", "svg"):
        fig.savefig(args.report_dir / ("learning_comparison." + suffix), dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
