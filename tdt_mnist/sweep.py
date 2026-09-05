"""Run paired MNIST block/threshold/seed experiments with equal forward budgets."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    groups = []
    metrics = ["initial_val_accuracy", "val_accuracy", "val_loss", "val_loss_drop",
               "test_accuracy", "total_fires", "fire_epoch_fraction", "counter_mean",
               "counter_abs_mean", "counter_abs_max", "counter_saturated_fraction"]
    for block, threshold in sorted({(r["block_size"], r["threshold"]) for r in rows}):
        members = [r for r in rows if (r["block_size"], r["threshold"]) == (block, threshold)]
        row = {"block_size": block, "threshold": threshold, "seeds": len(members),
               "loss_improved_seeds": sum(r["val_loss_drop"] > 0 for r in members),
               "accuracy_improved_seeds": sum(r["val_accuracy"] > r["initial_val_accuracy"] for r in members)}
        for metric in metrics:
            values = [r[metric] for r in members]
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        groups.append(row)
    return groups


def make_report(args, rows, distributions):
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.report_dir / "per_seed.csv", rows)
    write_csv(args.report_dir / "counter_histograms.csv", distributions)
    groups = aggregate(rows)
    write_csv(args.report_dir / "aggregate.csv", groups)
    text = ["# MNIST TDT-D: ブロック・閾値・seedの比較", "",
            f"1,000重み、seed={args.seeds}、訓練{args.train_size}件、検証{args.val_size}件、テスト10,000件。",
            f"全条件でK={args.measurements}、{args.steps}区間、batch={args.batch_size}、最大発火1座標。",
            f"各実験の訓練forwardは{2 * args.steps * args.measurements:,}回。測定例数も同一。",
            "データ分割seedは0で固定。同じseedでは初期重みと独立した訓練バッチ乱数列を共有。",
            "oracle監査は無効化し、評価回数も固定。計算効率の優劣は判定しない。",
            "毎区間の終了時に全カウンタをリセットする同期方式。INT8の容量は±127。",
            "平均±標本標準偏差（信頼区間ではない）。改善seed数は初期値に対する検証損失の低下。", "",
            "| block | 閾値 | 検証精度 % | テスト精度 % | 発火数 | 損失改善seed |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in groups:
        text.append(f"| {r['block_size']} | {r['threshold']} | "
                    f"{100*r['val_accuracy_mean']:.2f} ± {100*r['val_accuracy_std']:.2f} | "
                    f"{100*r['test_accuracy_mean']:.2f} ± {100*r['test_accuracy_std']:.2f} | "
                    f"{r['total_fires_mean']:.1f} ± {r['total_fires_std']:.1f} | "
                    f"{r['loss_improved_seeds']}/{r['seeds']} |")
    text.extend(["", "## カウンタ分布", "",
                 "各区間で一度以上測定した辺の、リセット直前の値を集計。未訪問のゼロは除外。",
                 "符号付き平均は正負が相殺するので、絶対値平均も表示する。最大は全seed・全区間の最大絶対値。",
                 "飽和率は|C|=127の観測割合。閾値への到達率ではない。",
                 "counter_histograms.csvにseed別の完全な符号付きヒストグラムを保存。", "",
                 "| block | 閾値 | 符号付き平均C | 平均abs(C) | 最大abs(C) | 飽和率 % |",
                 "| ---: | ---: | ---: | ---: | ---: | ---: |"])
    for r in groups:
        max_abs = max(v["counter_abs_max"] for v in rows
                      if (v["block_size"], v["threshold"]) == (r["block_size"], r["threshold"]))
        text.append(f"| {r['block_size']} | {r['threshold']} | {r['counter_mean_mean']:.3f} | "
                    f"{r['counter_abs_mean_mean']:.3f} | {max_abs} | {100*r['counter_saturated_fraction_mean']:.3f} |")
    text.extend(["", "全seedの値はper_seed.csv、平均と標準偏差はaggregate.csv。",
                 f"詳細ログ・設定・学習済み重み: `{args.output_dir.resolve()}`", "",
                 "3 seedの結果は今回の小規模モデル・固定蓄積方式に限定され、一般的な収束保証ではない。",
                 "テスト値で条件を選ぶと選択バイアスが生じるため、条件の判断には検証値を用いる。", ""])
    (args.report_dir / "README.md").write_text("\n".join(text))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--blocks", nargs="+", type=int, default=[1, 8, 32])
    p.add_argument("--thresholds", nargs="+", type=int, default=[4, 8, 16, 32])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--measurements", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-size", type=int, default=10000)
    p.add_argument("--val-size", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/grid"))
    p.add_argument("--report-dir", type=Path, default=Path("results/grid"))
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if min(args.steps, args.measurements, args.batch_size, args.workers, args.eval_every) <= 0:
        p.error("steps, measurements, batch-size, workers and eval-every must be positive")
    if not 1 <= args.val_size < 60000 or not 1 <= args.train_size <= 60000 - args.val_size:
        p.error("invalid train-size or val-size")
    if min(args.blocks) < 1 or max(args.blocks) > 1000:
        p.error("blocks must be in [1,1000]")
    if min(args.thresholds) < 1 or max(args.thresholds) > min(args.measurements, 127):
        p.error("thresholds must fit K and INT8 counter capacity")
    for name in ("seeds", "blocks", "thresholds"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            p.error(f"duplicate {name}")
    script = Path(__file__).with_name("train.py").resolve()
    manifest = {key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in vars(args).items() if key not in ("workers", "resume")}
    manifest["train_sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
    manifest["sweep_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if args.resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            p.error("resume requires exactly matching settings and source code")
    elif any(args.output_dir.iterdir()):
        p.error("output-dir is not empty; use --resume for matching experiments")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    # Download once, before parallel workers access the cache.
    from torchvision.datasets import MNIST
    MNIST(args.data_dir, train=True, download=True)
    MNIST(args.data_dir, train=False, download=True)
    tasks = list(itertools.product(args.seeds, args.blocks, args.thresholds))

    def run(task):
        seed, block, threshold = task
        name = f"seed{seed}-block{block}-threshold{threshold}"
        directory = args.output_dir / name
        summary_path = directory / "summary.json"
        if summary_path.exists() and args.resume:
            return task, json.loads(summary_path.read_text())
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f"incomplete run at {directory}; preserve or move it before retrying")
        command = [sys.executable, str(script), "--seed", str(seed), "--data-seed", "0",
                   "--batch-seed", str(seed + 100000), "--block-size", str(block),
                   "--threshold", str(threshold), "--steps", str(args.steps),
                   "--measurements", str(args.measurements), "--batch-size", str(args.batch_size),
                   "--train-size", str(args.train_size), "--val-size", str(args.val_size),
                   "--eval-every", str(args.eval_every), "--oracle-every", "0",
                   "--output-dir", str(directory.resolve()), "--data-dir", str(args.data_dir.resolve()),
                   "--no-download"]
        environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        log_path = args.output_dir / (name + ".log")
        with log_path.open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        if completed.returncode:
            raise RuntimeError(f"{name} failed: see {log_path}")
        return task, json.loads(summary_path.read_text())

    rows, distributions, errors = [], [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run, task): task for task in tasks}
        for future in as_completed(pending):
            try:
                (seed, block, threshold), result = future.result()
                assert result["train_forward_calls"] == 2 * args.measurements * args.steps
                counter = result["counter_distribution"]
                row = {"seed": seed, "block_size": block, "threshold": threshold,
                       "initial_val_accuracy": result["initial_validation"]["accuracy"],
                       "initial_val_loss": result["initial_validation"]["loss"],
                       "val_accuracy": result["final_validation"]["accuracy"],
                       "val_loss": result["final_validation"]["loss"],
                       "val_loss_drop": result["initial_validation"]["loss"] - result["final_validation"]["loss"],
                       "test_accuracy": result["test"]["accuracy"], "total_fires": result["total_fires"],
                       "fire_epoch_fraction": result["fire_epoch_fraction"],
                       "train_forward_calls": result["train_forward_calls"],
                       "total_forward_calls": result["total_forward_calls"],
                       "train_forward_examples": result["train_forward_examples"]}
                for key in ("min", "max", "mean", "abs_max", "abs_mean", "capacity", "count",
                            "saturated_count", "saturated_fraction", "saturation_update_count",
                            "peak_abs_during_accumulation"):
                    row["counter_" + key] = counter[key]
                rows.append(row)
                for value, count in counter["histogram"].items():
                    distributions.append({"seed": seed, "block_size": block, "threshold": threshold,
                                          "counter_value": int(value), "count": count,
                                          "fraction": count / counter["count"]})
                print(f"[{len(rows)}/{len(tasks)}] seed={seed} block={block} threshold={threshold} "
                      f"val={100*row['val_accuracy']:.2f}% fires={row['total_fires']}", flush=True)
            except Exception as error:
                errors.append(str(error))
                print(f"FAILED: {error}", file=sys.stderr, flush=True)
    if errors:
        raise SystemExit("Sweep incomplete:\n" + "\n".join(errors))
    assert len({row["total_forward_calls"] for row in rows}) == 1
    rows.sort(key=lambda r: (r["block_size"], r["threshold"], r["seed"]))
    distributions.sort(key=lambda r: (r["block_size"], r["threshold"], r["seed"], r["counter_value"]))
    make_report(args, rows, distributions)
    (args.report_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Completed {len(rows)} runs. Report: {args.report_dir / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
