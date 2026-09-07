"""A3 ReLU x quantizer factorial experiment at the v5 best 100k setting."""
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

from train import parser, validate

CONDITIONS = list(itertools.product(("relu", "identity"), ("absmax", "mean_threshold")))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(directory, records):
    rows, diagnostics, codes, aggregates = [], [], [], []
    records = sorted(records, key=lambda r: (r["hidden_activation"], r["a3_method"], r["seed"]))
    for r in records:
        tags = {k: r[k] for k in ("hidden_activation", "a3_method", "seed")}
        rows.append({**tags, "initial_val_accuracy": r["initial_validation"]["accuracy"],
                     "initial_val_loss": r["initial_validation"]["loss"],
                     "val_accuracy": r["final_validation"]["accuracy"],
                     "val_loss": r["final_validation"]["loss"],
                     "test_accuracy": r["test"]["accuracy"], "test_loss": r["test"]["loss"],
                     **{k: r[k] for k in ("total_fires", "zero_difference_count", "candidate_pair_count",
                                           "zero_difference_fraction", "train_forward_calls", "elapsed_seconds")},
                     **{f"layer_{i}_fires": n for i, n in enumerate(r["layer_update_counts"])}})
        for stage in ("initial", "final"):
            for obs in r[f"{stage}_activation_statistics"]:
                diagnostics.append({**tags, "stage": stage, **{k: v for k, v in obs.items() if k != "code_histogram"}})
                for code in (-1, 0, 1):
                    count = obs["code_histogram"].get(str(code), 0)
                    codes.append({**tags, "stage": stage, "layer": obs["layer"], "code": code,
                                  "count": count, "fraction": count / obs["values"]})
    for activation, method in CONDITIONS:
        members = [r for r in rows if (r["hidden_activation"], r["a3_method"]) == (activation, method)]
        if not members:
            continue
        row = {"hidden_activation": activation, "a3_method": method, "seeds": len(members)}
        for field in ("val_accuracy", "val_loss", "test_accuracy", "test_loss", "total_fires", "zero_difference_fraction"):
            values = [r[field] for r in members]
            row[field + "_mean"] = statistics.mean(values)
            row[field + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregates.append(row)
    write_csv(directory / "per_seed.csv", rows)
    write_csv(directory / "aggregate.csv", aggregates)
    write_csv(directory / "activation_diagnostics.csv", diagnostics)
    write_csv(directory / "activation_codes.csv", codes)
    (directory / "summaries.json").write_text(json.dumps(records, indent=2) + "\n")
    lines = ["# A3: ReLU × 量子化方式の100k比較", "",
             "90→1000→10、100,000個の3値重み。全条件で入力と隠れ層をA3に量子化。",
             "本実験の既定値はv5最高平均の条件: block=16、発火閾値8、K=64、12,000区間、batch=128、最大1発火。",
             "訓練10,000・検証1,000・テスト10,000、分割seed=0、実験seed=0,1,2、batch seed=seed+100000。",
             "実際の引数と環境・データハッシュはmanifest.json、各runのconfig.jsonに保存。", "",
             "absmax: scale=max(abs(x)); q=round(x/scale)（同距離は偶数）。",
             "mean_threshold: tau=0.5*mean(abs(x)); |x|>tauでq=sign(x)、それ以外0。",
             "復元scaleは選択された非ゼロ集合の平均絶対値。全ゼロ・空集合のscale=1。",
             "いずれもサンプルごと・層ごと・各forwardで決定的に再計算。FP32補助scale、FP32 GEMMとloss。",
             "ReLUなしでは中間線形出力を直接量子化。追加の接続数による除算は行わない。",
             "閾値係数0.5は事前固定し、この実験で探索しない。票の正規化Sとは別のスケール。", "",
             "量子化診断は初期・最終の同じ検証集合で量子化直前と復元後を比較。FP64集計は学習に戻さない。",
             "zero_difference_fractionは全訓練候補対のFP32 loss差が厳密に0の割合。近似ゼロや真の損失同値とは異なる。",
             "同一seedの初期重み、ミニバッチ列、摂動・丸め乱数列を対応させる。状態の変化で候補・S・票は異なる。",
             "各runは最初から学習し、テストは指定区間終了後1回だけ評価。平均±標本標準偏差。", "",
             "| ReLU | 量子化 | seeds | 検証精度 % | テスト精度 % | 検証loss | 損失差ゼロ % |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in aggregates:
        def fmt(k, scale=1):
            return f"{scale*r[k+'_mean']:.3f} ± {scale*r[k+'_std']:.3f}"
        lines.append(f"| {r['hidden_activation']} | {r['a3_method']} | {r['seeds']} | {fmt('val_accuracy',100)} | {fmt('test_accuracy',100)} | {fmt('val_loss')} | {fmt('zero_difference_fraction',100)} |")
    lines += ["", f"完了run数: {len(records)}。全体の完了状態はstatus.json参照。", "",
              "activation_diagnostics.csv: 層別ゼロ率・MSE・相対二乗誤差。",
              "activation_codes.csv: 層別−1/0/+1の個数と割合。",
              "各runのmetrics.csv: 全区間の損失差ゼロ率、検証学習曲線、カウンタ統計。", ""]
    (directory / "README.md").write_text("\n".join(lines))


def main():
    p = parser()
    names = [a.dest for a in p._actions if a.dest != "help"]
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--report-dir", type=Path, default=Path("results/a3-ablation-100k-20260907"))
    p.add_argument("--resume", action="store_true")
    p.set_defaults(activation_precision="a3", pool_shape=[9, 10], hidden_size=1000,
                   expected_params=100000, block_size=16, threshold=8, measurements=64,
                   steps=12000, train_size=10000, val_size=1000, data_seed=0,
                   oracle_every=0, eval_every=500, data_dir=Path("/tmp/tdt-mnist-data"),
                   output_dir=Path("runs/a3-ablation-100k-20260907"))
    args = p.parse_args()
    validate(args, p)
    if args.workers < 1 or len(args.seeds) != len(set(args.seeds)):
        p.error("positive workers and unique seeds required")
    if args.activation_precision != "a3" or args.hidden_size <= 0 or args.data_seed is None or args.oracle_every or args.test_size:
        p.error("require A3, hidden layer, fixed data seed, no oracle, full test set")
    if args.a3_threshold_factor != 0.5:
        p.error("this preregistered experiment fixes threshold factor at 0.5")
    import torch
    import torchvision
    from torchvision.datasets import MNIST
    MNIST(args.data_dir, train=True, download=args.download)
    MNIST(args.data_dir, train=False, download=args.download)
    sources = [Path(__file__).with_name(n).resolve() for n in ("train.py", "activation_quantization.py", "sweep_a3_ablation.py")]
    manifest = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ("workers", "resume")}
    manifest.update(conditions=CONDITIONS, torch_version=str(torch.__version__), torchvision_version=str(torchvision.__version__),
                    source_sha256={s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
                    data_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((args.data_dir / "MNIST/raw").glob("*-ubyte"))})
    manifest = json.loads(json.dumps(manifest))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.report_dir / "manifest.json"
    if args.resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            p.error("resume requires exact matching sources, data and settings")
    elif any(args.output_dir.iterdir()) or any(args.report_dir.iterdir()):
        p.error("output directories must be empty")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (args.report_dir / "sources").mkdir(exist_ok=True)
    for source in sources:
        shutil.copy2(source, args.report_dir / "sources" / source.name)
    tasks = [(a, m, seed) for seed in args.seeds for a, m in CONDITIONS]
    def run(task):
        activation, method, seed = task
        label = f"{activation}-{method}-seed{seed}"
        directory = args.output_dir / label
        summary = directory / "summary.json"
        if args.resume and summary.exists() and (directory / "model.pt").exists():
            return {**json.loads(summary.read_text()), "seed": seed}
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f"incomplete run: {directory}")
        options = {n: getattr(args, n) for n in names}
        options.update(seed=seed, batch_seed=seed+100000, hidden_activation=activation, a3_method=method,
                       output_dir=directory.resolve(), data_dir=args.data_dir.resolve(), download=False)
        command = [sys.executable, str(sources[0])]
        for name, value in options.items():
            if value is None:
                continue
            flag = "--" + name.replace("_", "-")
            if isinstance(value, bool):
                command.append(flag if value else "--no-" + name.replace("_", "-"))
            elif isinstance(value, (tuple, list)):
                command.extend([flag, *map(str, value)])
            else:
                command.extend([flag, str(value)])
        with (args.output_dir / f"{label}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        r = {**json.loads(summary.read_text()), "seed": seed}
        assert r["num_params"] == args.expected_params
        assert r["train_forward_calls"] == 2 * args.steps * args.measurements
        assert r["candidate_pair_count"] == args.steps * args.measurements
        return r
    results, errors = [], []
    def status():
        (args.report_dir / "status.json").write_text(json.dumps({"completed": len(results), "expected": len(tasks),
            "complete": len(results) == len(tasks) and not errors, "errors": errors}, indent=2) + "\n")
    status()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                report(args.report_dir, results)
                print(f"[{len(results)}/{len(tasks)}] {futures[future]} val={result['final_validation']['accuracy']:.4f}", flush=True)
            except Exception as error:
                errors.append(f"{futures[future]}: {error}")
                print(errors[-1], flush=True)
            status()
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
