"""Paired evidence-counter versus immediate single-vote MNIST ablation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import statistics
import time

import torch

from train import (TernaryModel, candidate_pair, epoch, evaluate, load_data, loss,
                   parser as train_parser, validate)


def immediate_action(weights, indices, edges, votes, max_fires):
    """Apply outward votes directly; no edge counters and no accumulation.

    Equal single-vote scores use the existing randomized block order, just as
    the counter variant uses that order to break ties. At most k moves/pair.
    """
    current = weights[indices].long()
    low, high = edges - 1, edges
    connected = (current == low) | (current == high)
    direction = torch.where(current == low, 1, -1)
    eligible = connected & (votes * direction > 0)
    chosen = torch.nonzero(eligible).flatten()[:max_fires]
    proposal = weights.clone()
    targets = torch.where(current == low, high, low).to(torch.int8)
    proposal[indices[chosen]] = targets[chosen]
    return proposal, len(chosen)


@torch.no_grad()
def no_counter_epoch(model, x, y, args, generator, scale, batch_generator):
    """K sequential immediate updates on one block, with epoch-fixed S.

    Model remains unmodified until return, but each candidate pair uses the
    latest working weights. No validation labels or counter state are involved.
    The block and S refresh cadence match the counter baseline exactly.
    """
    indices = torch.randperm(model.num_params, generator=generator, device=model.device)[:args.block_size]
    working = model.weights.clone()
    fires = clipped = nonzero = 0
    differences = []
    for _ in range(args.measurements):
        batch = torch.randint(len(x), (args.batch_size,), generator=batch_generator, device=model.device)
        plus, minus, edges, phi = candidate_pair(working, indices, generator)
        difference = loss(model, x[batch], y[batch], plus) - loss(model, x[batch], y[batch], minus)
        signal = -difference * phi / scale
        # Identical stochastic rounding and RNG consumption to train.accumulate.
        uniform = torch.rand(signal.shape, generator=generator, device=model.device)
        votes = (signal.sign() * (uniform < signal.clamp(-1, 1).abs())).to(torch.int32)
        clipped += int((signal.abs() > 1).sum())
        nonzero += int((votes != 0).sum())
        working, moves = immediate_action(working, indices, edges, votes, args.max_fires)
        fires += moves
        differences.append(float(difference.abs()))
    median = sorted(differences)[len(differences) // 2]
    next_scale = max(args.min_scale, (1 - args.scale_ema) * scale + args.scale_ema * median)
    total_votes = args.measurements * args.block_size
    return working, {"fires": fires, "scale": scale, "clip_rate": clipped / total_votes,
                     "nonzero_vote_rate": nonzero / total_votes}, next_scale


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_one(method, seed, options):
    args = argparse.Namespace(**options)
    args.seed = seed
    args.batch_seed = seed + 100000
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    directory = args.output_dir / f"{method}-seed{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    model = TernaryModel(args.pool_size, args.hidden_size, args.zero_rate, args.gain, args.device, seed)
    generator = torch.Generator(device=model.device).manual_seed(seed + 1)
    batches = torch.Generator(device=model.device).manual_seed(args.batch_seed)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(method=method, num_params=model.num_params, layer_scales=model.scales,
                  effective_threshold=args.threshold if method == "counter" else None,
                  torch_version=torch.__version__, activation_precision="float32",
                  update_cadence="after K pairs" if method == "counter" else "after every pair",
                  block_cadence="every K pairs", normalization_cadence="every K pairs")
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (tx, ty), (vx, vy), (testx, testy) = load_data(args, model.device)
    initial = evaluate(model, vx, vy)
    total_fires = 0
    scale = args.scale
    started = time.perf_counter()
    histogram = {}
    counter_peak = 0
    rows = [{"step": 0, "train_forward_calls": 0, "fires": 0, "total_fires": 0,
             "val_loss": initial["loss"], "val_accuracy": initial["accuracy"],
             "scale": scale, "clip_rate": None, "nonzero_vote_rate": None,
             "net_changed_coordinates": 0, "counter_mean": None, "counter_abs_mean": None,
             "counter_abs_max": None, "counter_saturated_fraction": None}]
    for step in range(1, args.steps + 1):
        if method == "counter":
            proposal, _, stats, next_scale = epoch(model, tx, ty, args, generator, scale, batches)
            for value, count in stats["counter_histogram"].items():
                histogram[value] = histogram.get(value, 0) + count
            counter_peak = max(counter_peak, stats["counter_peak_abs"])
        else:
            proposal, stats, next_scale = no_counter_epoch(model, tx, ty, args, generator, scale, batches)
        changed = int((model.weights != proposal).sum())
        model.weights.copy_(proposal)
        total_fires += stats["fires"]
        row = {"step": step, "train_forward_calls": 2 * step * args.measurements,
               "fires": stats["fires"], "total_fires": total_fires,
               "val_loss": None, "val_accuracy": None, "scale": scale,
               "clip_rate": stats["clip_rate"], "nonzero_vote_rate": stats["nonzero_vote_rate"],
               "net_changed_coordinates": changed,
               "counter_mean": stats.get("counter_mean"), "counter_abs_mean": stats.get("counter_abs_mean"),
               "counter_abs_max": stats.get("counter_abs_max"),
               "counter_saturated_fraction": stats.get("counter_saturated_fraction")}
        scale = next_scale
        if step % args.eval_every == 0 or step == args.steps:
            final = evaluate(model, vx, vy)
            row.update(val_loss=final["loss"], val_accuracy=final["accuracy"])
            print(f"{method} seed={seed} step={step} val={100*final['accuracy']:.2f}% fires={total_fires}", flush=True)
        rows.append(row)
    test = evaluate(model, testx, testy)
    distribution = None
    if histogram:
        count = sum(histogram.values())
        capacity = 2 ** (args.counter_bits - 1) - 1
        distribution = {"count": count, "histogram": histogram,
                        "scope": "measured edges before each epoch reset, pooled across epochs",
                        "mean": sum(int(k)*v for k,v in histogram.items()) / count,
                        "abs_mean": sum(abs(int(k))*v for k,v in histogram.items()) / count,
                        "abs_max": max(abs(int(k)) for k in histogram),
                        "peak_abs": counter_peak, "capacity": capacity,
                        "saturated_count": sum(histogram.get(str(k), 0) for k in (-capacity, capacity))}
    summary = {"method": method, "seed": seed, "num_params": model.num_params,
               "initial_validation": initial, "final_validation": final, "test": test,
               "total_fires": total_fires,
               "net_changed_coordinates_sum": sum(r["net_changed_coordinates"] for r in rows),
               "train_forward_calls": 2 * args.steps * args.measurements,
               "train_forward_examples": 2 * args.steps * args.measurements * args.batch_size,
               "total_forward_calls": model.forward_calls,
               "total_forward_examples": model.forward_examples,
               "counter_distribution": distribution, "elapsed_seconds": time.perf_counter() - started}
    write_csv(directory / "metrics.csv", rows)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save({"weights": model.weights.cpu(), "config": config}, directory / "model.pt")
    return summary


def report(args, results):
    args.report_dir.mkdir(parents=True, exist_ok=True)
    individual = []
    for r in sorted(results, key=lambda r: (r["method"], r["seed"])):
        individual.append({"method": r["method"], "seed": r["seed"],
                           "initial_val_loss": r["initial_validation"]["loss"],
                           "initial_val_accuracy": r["initial_validation"]["accuracy"],
                           "val_loss": r["final_validation"]["loss"], "val_accuracy": r["final_validation"]["accuracy"],
                           "test_loss": r["test"]["loss"], "test_accuracy": r["test"]["accuracy"],
                           "total_fires": r["total_fires"], "net_changed_coordinates_sum": r["net_changed_coordinates_sum"],
                           "train_forward_calls": r["train_forward_calls"], "total_forward_calls": r["total_forward_calls"],
                           "train_forward_examples": r["train_forward_examples"]})
    write_csv(args.report_dir / "per_seed.csv", individual)
    aggregate = []
    for method in ("counter", "no_counter"):
        members = [r for r in individual if r["method"] == method]
        row = {"method": method, "seeds": len(members),
               "loss_improved_seeds": sum(r["val_loss"] < r["initial_val_loss"] for r in members)}
        for metric in ("val_loss", "val_accuracy", "test_accuracy", "total_fires"):
            values = [r[metric] for r in members]
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else 0
        aggregate.append(row)
    write_csv(args.report_dir / "aggregate.csv", aggregate)
    # Pairing is verified, not merely assumed from matching seed labels.
    for seed in args.seeds:
        a, b = [r for r in results if r["seed"] == seed]
        assert a["initial_validation"] == b["initial_validation"]
        assert a["train_forward_calls"] == b["train_forward_calls"]
        assert a["total_forward_calls"] == b["total_forward_calls"]
        assert a["train_forward_examples"] == b["train_forward_examples"]
    paired = []
    for seed in args.seeds:
        a = next(r for r in individual if r["seed"] == seed and r["method"] == "counter")
        b = next(r for r in individual if r["seed"] == seed and r["method"] == "no_counter")
        paired.append({"seed": seed, "counter_minus_no_counter_val_accuracy": a["val_accuracy"] - b["val_accuracy"],
                       "counter_minus_no_counter_val_loss": a["val_loss"] - b["val_loss"]})
    write_csv(args.report_dir / "paired_differences.csv", paired)
    (args.report_dir / "summaries.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# カウンタあり・なしの比較", "",
             f"seed={args.seeds}、block={args.block_size}、counter閾値={args.threshold}、K={args.measurements}、{args.steps}区間。",
             f"各runは{2*args.steps*args.measurements:,}回の訓練forward。batch={args.batch_size}。",
             f"訓練{args.train_size}件・検証{args.val_size}件・テスト10,000件、分割seed={args.data_seed}で固定。",
             "同じseedで初期重み、訓練バッチ列、摂動・確率丸め乱数列を対応させた。活性化はFP32。",
             "カウンタありはK回、重みを固定して票を蓄積し、区間末に閾値を判定して全証拠をリセット。",
             "カウンタなしは辺カウンタを作らず、各候補対の非ゼロ票から直ちに更新。閾値は適用しない。",
             "ブロック選択とS更新は両方式ともK回ごと。各更新機会の最大更新数は同じ。",
             "カウンタなしの更新機会はK倍になるため、証拠蓄積と発火待機を合わせた比較である。",
             "途中更新があるため、カウンタなしでは同一区間に同じ座標を複数回更新し得る。総発火数はその全更新を数える。",
             "正規化Sは同じ初期値とEMA規則を使うが、学習軌跡に応じて値が変わる。検証結果で更新を選別しない。", "",
             "平均±標本標準偏差。", "",
             "| 方式 | 検証loss | 検証精度 % | テスト精度 % | 更新数 | 損失改善seed |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in aggregate:
        lines.append(f"| {r['method']} | {r['val_loss_mean']:.4f} ± {r['val_loss_std']:.4f} | "
                     f"{100*r['val_accuracy_mean']:.2f} ± {100*r['val_accuracy_std']:.2f} | "
                     f"{100*r['test_accuracy_mean']:.2f} ± {100*r['test_accuracy_std']:.2f} | "
                     f"{r['total_fires_mean']:.1f} ± {r['total_fires_std']:.1f} | {r['loss_improved_seeds']}/{r['seeds']} |")
    lines.extend(["", "per_seed.csvに個別結果、paired_differences.csvに同一seedの差、summaries.jsonにカウンタ分布を保存。",
                  "カウンタなしの分布はnull（存在しないため）で、ゼロカウンタとして扱わない。",
                  f"詳細ログと学習済み重み: `{args.output_dir.resolve()}`", ""])
    (args.report_dir / "README.md").write_text("\n".join(lines))


def main():
    p = train_parser()
    p.description = __doc__
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--report-dir", type=Path, default=Path("results/counter-comparison"))
    p.set_defaults(block_size=8, threshold=8, measurements=64, steps=3000, train_size=10000,
                   val_size=1000, data_seed=0, oracle_every=0, eval_every=500,
                   output_dir=Path("runs/counter-comparison"))
    args = p.parse_args()
    validate(args, p)
    if args.workers < 1 or len(set(args.seeds)) != len(args.seeds):
        p.error("workers must be positive and seeds must be unique")
    if args.device != "cpu" or args.oracle_every != 0 or args.test_size != 0:
        p.error("this paired runner uses CPU, no oracle audit, and the full test set")
    if args.data_seed is None:
        p.error("a fixed --data-seed is required")
    if any(path.exists() and any(path.iterdir()) for path in (args.output_dir, args.report_dir)):
        p.error("output-dir and report-dir must be new or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    for name in ("train.py", "compare_counters.py"):
        source = Path(__file__).with_name(name)
        manifest[name + "_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (args.report_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    from torchvision.datasets import MNIST
    MNIST(args.data_dir, train=True, download=args.download)
    MNIST(args.data_dir, train=False, download=args.download)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(run_one, method, seed, vars(args))
                   for seed in args.seeds for method in ("counter", "no_counter")]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Completed {result['method']} seed={result['seed']} ({len(results)}/{len(futures)})", flush=True)
    report(args, results)
    print(f"Report: {args.report_dir / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
