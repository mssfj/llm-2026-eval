"""Paired W3A32/A16/A8/A4/A3 experiments with one fixed TDT configuration."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from activation_quantization import PRECISIONS, activation_description
from train import parser as train_parser, validate


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_report(args, results):
    rows, diagnostics = [], []
    for precision in args.precisions:
        for seed in args.seeds:
            r = next(r for p, s, r in results if (p, s) == (precision, seed))
            rows.append({"precision": precision, "seed": seed, "num_params": r["num_params"],
                         "initial_val_loss": r["initial_validation"]["loss"],
                         "initial_val_accuracy": r["initial_validation"]["accuracy"],
                         "val_loss": r["final_validation"]["loss"],
                         "val_accuracy": r["final_validation"]["accuracy"],
                         "test_loss": r["test"]["loss"], "test_accuracy": r["test"]["accuracy"],
                         "total_fires": r["total_fires"],
                         "layer_0_fires": r["layer_update_counts"][0],
                         "layer_1_fires": r["layer_update_counts"][1],
                         "train_forward_calls": r["train_forward_calls"],
                         "total_forward_calls": r["total_forward_calls"],
                         "train_forward_examples": r["train_forward_examples"],
                         "counter_abs_mean": r["counter_distribution"]["abs_mean"],
                         "counter_abs_max": r["counter_distribution"]["abs_max"],
                         "counter_saturated_count": r["counter_distribution"]["saturated_count"]})
            for stage in ("initial", "final"):
                for obs in r[f"{stage}_activation_statistics"]:
                    diagnostics.append({"precision": precision, "seed": seed, "stage": stage,
                                        "layer": obs["layer"], "values": obs["values"],
                                        "zero_fraction": obs["zero_fraction"], "mse": obs["mse"],
                                        "relative_squared_error": obs["relative_squared_error"],
                                        "observed_codes": len(obs["code_histogram"]) if obs["code_histogram"] else None})
    aggregates = []
    for precision in args.precisions:
        members = [r for r in rows if r["precision"] == precision]
        row = {"precision": precision, "seeds": len(members),
               "loss_improved_seeds": sum(r["val_loss"] < r["initial_val_loss"] for r in members),
               "accuracy_improved_seeds": sum(r["val_accuracy"] > r["initial_val_accuracy"] for r in members)}
        for field in ("initial_val_loss", "initial_val_accuracy", "val_loss", "val_accuracy",
                      "test_accuracy", "total_fires", "layer_0_fires", "layer_1_fires", "counter_abs_mean"):
            values = [r[field] for r in members]
            row[field + "_mean"] = statistics.mean(values)
            row[field + "_std"] = statistics.stdev(values) if len(values) > 1 else 0
        aggregates.append(row)
    write_csv(args.report_dir / "per_seed.csv", rows)
    write_csv(args.report_dir / "aggregate.csv", aggregates)
    write_csv(args.report_dir / "activation_diagnostics.csv", diagnostics)
    (args.report_dir / "summaries.json").write_text(json.dumps(
        [{"precision": p, "seed": seed, **r} for p, seed, r in results], indent=2) + "\n")
    lines = ["# W3A32・W3A16・W3A8・W3A4・W3A3の比較", "",
             f"{rows[0]['num_params']:,}パラメータ、block={args.block_size}、発火閾値={args.threshold}、seed={args.seeds}。",
             f"訓練{args.train_size}件・検証{args.val_size}件・テスト10,000件、K={args.measurements}、{args.steps}区間、batch={args.batch_size}。",
             "同じseedで重み初期化・データ・訓練バッチ・摂動乱数・票の丸め乱数を対応させた。",
             "活性化の量子化は決定的で、乱数を消費しない。各精度でスクラッチから学習し、検証値で更新を選別しない。", "",
             "## 精度の定義", "",
             "量子化するのは各線形層への入力（正規化済み画像入力とReLU後の隠れ層出力）。",
             "行列積の累積・中間線形出力・ReLU計算・最終logit・損失はFP32。逆伝播とSTEは使わない。",
             "| 設定 | 活性化の表現 |",
             "| --- | --- |",
             "| A32 | FP32、そのまま |",
             "| A16 | FP16にキャストし、FP32に戻して行列積へ渡す |",
             "| A8 | 符号付き整数コード−127～127（255段階）とFP32スケール |",
             "| A4 | 符号付き整数コード−7～7（15段階）とFP32スケール |",
             "| A3 | コード−1・0・+1の3値とFP32スケール。3ビットではない |", "",
             "整数系は各サンプル・各層のmax(abs(x))/qmaxをスケールとし、最近傍（同距離は偶数）へ丸める。",
             "q = clamp(round(x / scale), -qmax, qmax)、復元値はscale*q。全ゼロ行ではscale=1。",
             "スケールは各候補のforwardごとに再計算する。データに依存する補助スケールもFP32で保持する。",
             "ReLU後は非負なので、A3の隠れ層入力で実際に使われるコードは0と+1の2値。",
             "A8/A4のReLU後も対称符号付き範囲の非負側を使い、符号なし範囲への変更はしていない。",
             "A8/A4/A3はINT8コンテナへ符号化してからFP32へ復元する。sub-byte packingや専用整数GEMMは未使用。",
             "これは活性化表現の精度が学習に与える影響の比較で、演算速度・メモリ帯域の優位性を測るものではない。", "",
             "## 結果", "", "平均±標本標準偏差。", "",
             "| 設定 | 検証loss | 検証精度 % | テスト精度 % | 発火数 | 損失改善seed |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in aggregates:
        lines.append(f"| W3{r['precision'].upper()} | {r['val_loss_mean']:.4f} ± {r['val_loss_std']:.4f} | "
                     f"{100*r['val_accuracy_mean']:.2f} ± {100*r['val_accuracy_std']:.2f} | "
                     f"{100*r['test_accuracy_mean']:.2f} ± {100*r['test_accuracy_std']:.2f} | "
                     f"{r['total_fires_mean']:.1f} ± {r['total_fires_std']:.1f} | {r['loss_improved_seeds']}/{r['seeds']} |")
    lines += ["", "activation_diagnostics.csvに初期・最終検証時の各層のゼロ率と量子化誤差を保存。",
              "誤差は各モデルの各層で量子化直前と復元後を比較した値で、A32モデルとの差ではない。",
              "FP64の誤差集計は読み取り専用の診断で、forwardや重み更新の計算には使わない。",
              "summaries.jsonに整数コードの分布・層別更新数・カウンタ分布を保存。",
              f"各実験の設定・学習曲線・重み: `{args.output_dir.resolve()}`", ""]
    (args.report_dir / "README.md").write_text("\n".join(lines))


def main():
    p = train_parser()
    train_names = [a.dest for a in p._actions if a.dest != "help"]
    p.description = __doc__
    p.add_argument("--precisions", type=str.lower, nargs="+", choices=PRECISIONS, default=list(PRECISIONS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--report-dir", type=Path, default=Path("results/activation-grid"))
    p.add_argument("--resume", action="store_true")
    p.set_defaults(pool_shape=[9, 10], hidden_size=100, expected_params=10000, block_size=32,
                   threshold=8, measurements=64, steps=3000, train_size=10000, val_size=1000,
                   data_seed=0, oracle_every=0, eval_every=500, output_dir=Path("runs/activation-grid"))
    args = p.parse_args()
    validate(args, p)
    if args.workers < 1 or len(args.seeds) != len(set(args.seeds)) or len(args.precisions) != len(set(args.precisions)):
        p.error("workers must be positive; seeds and precisions must be unique")
    if args.hidden_size <= 0 or args.data_seed is None or args.oracle_every != 0 or args.test_size != 0:
        p.error("require a two-layer model, fixed data-seed, no oracle audit, and full test set")
    script = Path(__file__).with_name("train.py").resolve()
    manifest = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ("workers", "resume")}
    sources = [Path(__file__).with_name(name) for name in ("train.py", "activation_quantization.py", "sweep_activations.py")]
    manifest["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest["precision_definitions"] = {precision: activation_description(precision) for precision in args.precisions}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    path = args.report_dir / "manifest.json"
    if args.resume:
        if not path.exists() or json.loads(path.read_text()) != manifest:
            p.error("resume requires matching settings and sources")
    elif any(args.output_dir.iterdir()) or any(args.report_dir.iterdir()):
        p.error("output-dir and report-dir must be empty; use --resume only for matching experiments")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    source_dir = args.report_dir / "sources"
    source_dir.mkdir(exist_ok=True)
    for source in sources:
        shutil.copy2(source, source_dir / source.name)
    from torchvision.datasets import MNIST
    MNIST(args.data_dir, train=True, download=args.download)
    MNIST(args.data_dir, train=False, download=args.download)

    def run(task):
        seed, precision = task
        directory = args.output_dir / f"{precision}-seed{seed}"
        summary = directory / "summary.json"
        if args.resume and summary.exists():
            return precision, seed, json.loads(summary.read_text())
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f"incomplete run at {directory}; move it aside before retrying")
        options = {name: getattr(args, name) for name in train_names}
        options.update(seed=seed, batch_seed=seed+100000, activation_precision=precision,
                       output_dir=directory.resolve(), data_dir=args.data_dir.resolve(), download=False)
        command = [sys.executable, str(script)]
        for name, value in options.items():
            if value is None:
                continue
            option = "--" + name.replace("_", "-")
            if isinstance(value, bool):
                command.append(option if value else "--no-" + name.replace("_", "-"))
            elif isinstance(value, (tuple, list)):
                command.extend([option, *map(str, value)])
            else:
                command.extend([option, str(value)])
        environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        log_path = args.output_dir / f"{precision}-seed{seed}.log"
        with log_path.open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        if result.returncode:
            raise RuntimeError(f"{precision} seed {seed} failed; see {log_path}")
        return precision, seed, json.loads(summary.read_text())

    tasks = list(itertools.product(args.seeds, args.precisions))
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, task) for task in tasks]):
            try:
                precision, seed, summary = future.result()
                results.append((precision, seed, summary))
                print(f"[{len(results)}/{len(tasks)}] W3{precision.upper()} seed={seed} "
                      f"val={100*summary['final_validation']['accuracy']:.2f}% fires={summary['total_fires']}", flush=True)
            except Exception as error:
                errors.append(str(error))
                print(f"FAILED: {error}", flush=True)
    if errors:
        raise SystemExit("Incomplete sweep:\n" + "\n".join(errors))
    assert len({r["train_forward_calls"] for _, _, r in results}) == 1
    assert len({r["total_forward_calls"] for _, _, r in results}) == 1
    assert len({r["num_params"] for _, _, r in results}) == 1
    make_report(args, results)
    print(f"Report: {args.report_dir / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
