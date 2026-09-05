"""Small, auditable synchronous TDT-D experiment. No autograd or latent weights."""

import argparse
import csv
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F


class TernaryModel:
    """Bias-free linear classifier, or one-hidden-layer ReLU MLP; fixed scales."""

    def __init__(self, pool_size=10, hidden_size=0, zero_rate=1 / 3,
                 gain=1.0, device="cpu", seed=0):
        self.device = torch.device(device)
        inputs = pool_size ** 2
        self.shapes = ([(hidden_size, inputs), (10, hidden_size)]
                       if hidden_size else [(10, inputs)])
        self.num_params = sum(math.prod(shape) for shape in self.shapes)
        self.scales = [gain / math.sqrt(shape[1] * (1 - zero_rate))
                       for shape in self.shapes]
        generator = torch.Generator(device=self.device).manual_seed(seed)
        values = torch.rand(self.num_params, generator=generator, device=self.device)
        self.weights = torch.zeros(self.num_params, dtype=torch.int8, device=self.device)
        self.weights[values < (1 - zero_rate) / 2] = -1
        self.weights[values >= (1 + zero_rate) / 2] = 1
        self.forward_calls = 0
        self.forward_examples = 0

    @torch.no_grad()
    def __call__(self, x, weights=None):
        self.forward_calls += 1
        self.forward_examples += len(x)
        weights = self.weights if weights is None else weights
        offset = 0
        for index, (shape, scale) in enumerate(zip(self.shapes, self.scales)):
            size = math.prod(shape)
            matrix = weights[offset:offset + size].view(shape).to(torch.float32) * scale
            x = F.linear(x, matrix)
            if index < len(self.shapes) - 1:
                x = F.relu(x)
            offset += size
        return x


def loss(model, x, y, weights=None):
    return F.cross_entropy(model(x, weights), y)


def candidate_pair(weights, indices, generator):
    """One incident edge per active coordinate; random orientation, no clipping.

    At zero choose (-1, 0) or (0, 1) uniformly. At boundaries the only
    incident edge is used. Each candidate differs from the current state by
    at most one level per coordinate; the pair never jumps -1 to +1.
    """
    current = weights[indices].long()
    edges = torch.where(current == -1, 0, 1)
    choose = torch.randint(2, current.shape, generator=generator, device=weights.device)
    edges = torch.where(current == 0, choose, edges)
    phi = 2 * torch.randint(2, current.shape, generator=generator, device=weights.device) - 1
    low, high = edges - 1, edges
    plus, minus = weights.clone(), weights.clone()
    plus[indices] = torch.where(phi > 0, high, low).to(torch.int8)
    minus[indices] = torch.where(phi > 0, low, high).to(torch.int8)
    return plus, minus, edges, phi


def accumulate(evidence, indices, edges, signal, generator, leak, capacity):
    """Unbiased stochastic rounding on [-1,1]; clip/saturation counted separately."""
    bounded = signal.clamp(-1, 1)
    uniform = torch.rand(signal.shape, generator=generator, device=signal.device)
    votes = (signal.sign() * (uniform < bounded.abs())).to(torch.int32)
    raw = torch.trunc(evidence[indices, edges].float() * leak).to(torch.int32) + votes
    evidence[indices, edges] = raw.clamp(-capacity, capacity).to(evidence.dtype)
    return int((signal.abs() > 1).sum()), int((raw.abs() > capacity).sum()), int((votes != 0).sum())


def select_actions(weights, evidence, counts, indices, threshold, max_fires, scale):
    """Rank outward threshold crossings by normalized mean supporting evidence."""
    scores = torch.full((len(indices), 2), -torch.inf, device=weights.device)
    targets = torch.zeros((len(indices), 2), dtype=torch.int8, device=weights.device)
    current = weights[indices].long()
    for edge in (0, 1):
        low, high = edge - 1, edge
        connected = (current == low) | (current == high)
        direction = torch.where(current == low, 1, -1)
        support = evidence[indices, edge].float() * direction
        eligible = connected & (support >= threshold)
        score = support * scale / counts[indices, edge].clamp_min(1)
        scores[:, edge] = torch.where(eligible, score, -torch.inf)
        targets[:, edge] = torch.where(current == low, high, low).to(torch.int8)
    best, edge_choice = scores.max(dim=1)
    eligible = torch.nonzero(torch.isfinite(best)).flatten()
    # One edge per coordinate, at most k coordinates; tied scores use coordinate order.
    order = torch.argsort(best[eligible], descending=True, stable=True)
    chosen = eligible[order[:max_fires]]
    result = weights.clone()
    result[indices[chosen]] = targets[chosen, edge_choice[chosen]]
    return result, len(chosen)


COUNTER_FIELDS = ["counter_count", "counter_min", "counter_max", "counter_mean",
                  "counter_abs_max", "counter_abs_mean", "counter_saturated_count",
                  "counter_saturated_fraction", "counter_capacity", "counter_peak_abs",
                  "counter_all_mean", "counter_all_abs_mean"]


def counter_statistics(evidence, counts, capacity):
    """Pre-reset distribution over edges measured at least once in this epoch.

    Unvisited counters are excluded from the primary distribution; all-state
    means are reported separately so block size does not hide activity in zeros.
    """
    values = evidence[counts > 0].to(torch.int32)
    if not values.numel():
        raise ValueError("counter statistics require at least one measured edge")
    unique, frequencies = torch.unique(values, return_counts=True)
    histogram = {str(int(k)): int(v) for k, v in zip(unique.tolist(), frequencies.tolist())}
    return {"counter_count": values.numel(), "counter_min": int(values.min()),
            "counter_max": int(values.max()), "counter_mean": float(values.float().mean()),
            "counter_abs_max": int(values.abs().max()),
            "counter_abs_mean": float(values.abs().float().mean()),
            "counter_saturated_count": int((values.abs() == capacity).sum()),
            "counter_saturated_fraction": float((values.abs() == capacity).float().mean()),
            "counter_capacity": capacity,
            "counter_all_mean": float(evidence.float().mean()),
            "counter_all_abs_mean": float(evidence.float().abs().mean()),
            "counter_histogram": histogram}


@torch.no_grad()
def epoch(model, x, y, args, generator, scale, batch_generator=None):
    """A fixed-weight accumulation epoch; discard all evidence at its end.

    This conservative reset policy prevents stale evidence, even for interacting
    blocks. It deliberately trades sample efficiency for an interpretable trial.
    """
    device = model.device
    dtype = torch.int8 if args.counter_bits == 8 else torch.int16
    capacity = 2 ** (args.counter_bits - 1) - 1
    evidence = torch.zeros((model.num_params, 2), dtype=dtype, device=device)
    counts = torch.zeros((model.num_params, 2), dtype=torch.int32, device=device)
    indices = torch.randperm(model.num_params, generator=generator, device=device)[:args.block_size]
    clipped = saturated = nonzero = 0
    differences = []
    peak_abs = 0
    for _ in range(args.measurements):
        batch = torch.randint(len(x), (args.batch_size,), generator=batch_generator if batch_generator is not None else generator, device=device)
        bx, by = x[batch], y[batch]
        plus, minus, edges, phi = candidate_pair(model.weights, indices, generator)
        difference = loss(model, bx, by, plus) - loss(model, bx, by, minus)
        signal = -difference * phi / scale
        c, s, n = accumulate(evidence, indices, edges, signal, generator, args.leak, capacity)
        clipped += c
        saturated += s
        nonzero += n
        peak_abs = max(peak_abs, int(evidence[indices, edges].to(torch.int32).abs().max()))
        counts[indices, edges] += 1
        differences.append(float(difference.abs()))
    proposal, fires = select_actions(model.weights, evidence, counts, indices,
                                     args.threshold, args.max_fires, scale)
    votes = args.measurements * args.block_size
    stats = {"fires": fires, "clip_rate": clipped / votes,
             "saturation_rate": saturated / votes, "nonzero_vote_rate": nonzero / votes,
             "scale": scale, "counter_peak_abs": peak_abs,
             **counter_statistics(evidence, counts, capacity)}
    # Updating S only AFTER the epoch, whose evidence is discarded.
    median = sorted(differences)[len(differences) // 2]
    next_scale = max(args.min_scale, (1 - args.scale_ema) * scale + args.scale_ema * median)
    return proposal, indices, stats, next_scale


@torch.no_grad()
def oracle_metrics(model, proposal, indices, x, y, tie_tolerance):
    """Independent held-out action audit. Never used to accept/reject updates."""
    base = float(loss(model, x, y))
    correct = 0
    regrets = []
    gains = []
    for index in indices.tolist():
        current = int(model.weights[index])
        changes = {current: 0.0}
        for target in (-1, 0, 1):
            if abs(target - current) != 1:
                continue
            candidate = model.weights.clone()
            candidate[index] = target
            changes[target] = float(loss(model, x, y, candidate)) - base
        best = min(changes.values())
        selected = changes[int(proposal[index])]
        correct += selected <= best + tie_tolerance
        regrets.append(max(0.0, selected - best))
        gains.append(-best)
    delta = float(loss(model, x, y, proposal)) - base
    return {"action_accuracy": correct / len(indices),
            "local_regret": sum(regrets) / len(regrets),
            "oracle_gain": sum(gains) / len(gains), "fire_delta_loss": delta}


@torch.no_grad()
def evaluate(model, x, y, batch_size=1024):
    total_loss = 0.0
    correct = 0
    for start in range(0, len(x), batch_size):
        bx, by = x[start:start + batch_size], y[start:start + batch_size]
        logits = model(bx)
        total_loss += float(F.cross_entropy(logits, by, reduction="sum"))
        correct += int((logits.argmax(1) == by).sum())
    return {"loss": total_loss / len(x), "accuracy": correct / len(x)}


def load_data(args, device):
    from torchvision.datasets import MNIST

    training = MNIST(args.data_dir, train=True, download=args.download)
    testing = MNIST(args.data_dir, train=False, download=args.download)
    order = torch.randperm(len(training), generator=torch.Generator().manual_seed(args.seed if args.data_seed is None else args.data_seed))
    val_ids = order[:args.val_size]
    train_ids = order[args.val_size:]
    if args.train_size:
        train_ids = train_ids[:args.train_size]

    def prepare(dataset, indices):
        images = dataset.data[indices].float().unsqueeze(1) / 255.0
        # Fixed preprocessing; neither pooling nor normalization has trained parameters.
        images = F.adaptive_avg_pool2d(images, (args.pool_size, args.pool_size))
        images = ((images - 0.1307) / 0.3081).flatten(1).to(device)
        return images, dataset.targets[indices].to(device)

    test_ids = torch.arange(args.test_size or len(testing))
    return prepare(training, train_ids), prepare(training, val_ids), prepare(testing, test_ids)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pool-size", type=int, default=10, help="pooled image side; default model has 10*10*10=1000 weights")
    p.add_argument("--hidden-size", type=int, default=0, help="0: linear; positive: bias-free ReLU MLP")
    p.add_argument("--steps", type=int, default=5000, help="number of fixed-state accumulation epochs, not dataset passes")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--block-size", type=int, default=1, help="independently perturbed coordinates per pair")
    p.add_argument("--measurements", type=int, default=16, help="K candidate pairs per fixed-state epoch; each costs 2 forwards")
    p.add_argument("--threshold", type=int, default=4)
    p.add_argument("--max-fires", type=int, default=1)
    p.add_argument("--counter-bits", type=int, choices=(8, 16), default=8)
    p.add_argument("--leak", type=float, default=1.0, help="integer counter leak uses truncation toward zero")
    p.add_argument("--scale", type=float, default=0.02, help="initial vote normalization scale S")
    p.add_argument("--scale-ema", type=float, default=0.1, help="epoch-boundary median-absolute-difference EMA; 0 fixes S")
    p.add_argument("--min-scale", type=float, default=1e-5)
    p.add_argument("--zero-rate", type=float, default=1 / 3)
    p.add_argument("--gain", type=float, default=1.0, help="fixed layer-scale gain")
    p.add_argument("--train-size", type=int, default=0, help="0 uses all training examples outside validation")
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--test-size", type=int, default=0, help="0 uses all 10000 test examples")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--oracle-every", type=int, default=100, help="independent action audit interval; 0 disables")
    p.add_argument("--oracle-size", type=int, default=512)
    p.add_argument("--tie-tolerance", type=float, default=1e-5)
    p.add_argument("--data-seed", type=int, default=None, help="fixed dataset split seed; default uses seed")
    p.add_argument("--batch-seed", type=int, default=None, help="independent batch RNG for paired sweeps; default shares update RNG")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=1, help="small models usually benefit from one CPU thread")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/tdt-1000"))
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    return p


def validate(args, p):
    for name in ("steps", "batch_size", "block_size", "measurements", "threshold", "max_fires",
                 "eval_every", "oracle_size", "threads"):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if not 1 <= args.pool_size <= 28 or args.hidden_size < 0:
        p.error("pool-size must be in [1, 28]; hidden-size must be nonnegative")
    if not 0 <= args.zero_rate < 1 or not 0 < args.leak <= 1 or not 0 <= args.scale_ema <= 1:
        p.error("require 0 <= zero-rate < 1, 0 < leak <= 1, 0 <= scale-ema <= 1")
    if any(not math.isfinite(v) or v <= 0 for v in (args.scale, args.min_scale, args.gain)):
        p.error("scale, min-scale and gain must be finite and positive")
    if not 1 <= args.val_size < 60000 or not 0 <= args.train_size <= 60000 - args.val_size:
        p.error("invalid train-size or val-size for MNIST's 60000 training examples")
    if not 0 <= args.test_size <= 10000 or args.oracle_every < 0:
        p.error("test-size must be in [0, 10000]; oracle-every must be nonnegative")
    if not math.isfinite(args.tie_tolerance) or args.tie_tolerance < 0:
        p.error("tie-tolerance must be finite and nonnegative")
    if args.threshold > min(args.measurements, 2 ** (args.counter_bits - 1) - 1):
        p.error("threshold exceeds counter capacity or votes available before epoch reset")
    count = (args.pool_size ** 2 + 10) * args.hidden_size if args.hidden_size else 10 * args.pool_size ** 2
    if args.block_size > count or args.max_fires > args.block_size:
        p.error("require max-fires <= block-size <= number of parameters")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is not available; use --device cpu")


def main():
    p = parser()
    args = p.parse_args()
    validate(args, p)
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    # Never silently overwrite an earlier experiment.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        p.error("output-dir is not empty; choose a new experiment directory")
    model = TernaryModel(args.pool_size, args.hidden_size, args.zero_rate,
                         args.gain, args.device, args.seed)
    generator = torch.Generator(device=model.device).manual_seed(args.seed + 1)
    batch_generator = (torch.Generator(device=model.device).manual_seed(args.batch_seed)
                       if args.batch_seed is not None else None)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(num_params=model.num_params, shapes=model.shapes, layer_scales=model.scales,
                  activation_precision="float32", weight_storage="int8 ternary",
                  reset_policy="all evidence discarded after every accumulation epoch",
                  torch_version=torch.__version__)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"TDT-D: {model.num_params} ternary weights, shapes={model.shapes}, device={model.device}", flush=True)
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_data(args, model.device)
    started = time.perf_counter()
    initial = evaluate(model, val_x, val_y)
    print(f"initial val loss={initial['loss']:.4f}, accuracy={initial['accuracy']:.2%}", flush=True)
    fields = ["step", "elapsed_seconds", "val_loss", "val_accuracy", "fires", "total_fires",
              "scale", "clip_rate", "saturation_rate", "nonzero_vote_rate", "action_accuracy",
              "local_regret", "oracle_gain", "fire_delta_loss", "train_forward_calls",
              "train_forward_examples", "total_forward_calls", "total_forward_examples"] + COUNTER_FIELDS
    total_fires = 0
    scale = args.scale
    final = initial
    audited_deltas = []
    counter_histogram = {}
    peak_counter_abs = 0
    saturation_updates = 0
    fire_epochs = 0
    with (args.output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"step": 0, "val_loss": initial["loss"], "val_accuracy": initial["accuracy"],
                         "total_fires": 0, "train_forward_calls": 0, "train_forward_examples": 0,
                         "total_forward_calls": model.forward_calls, "total_forward_examples": model.forward_examples})
        for step in range(1, args.steps + 1):
            proposal, indices, stats, scale = epoch(model, train_x, train_y, args, generator, scale, batch_generator)
            for value, count in stats.pop("counter_histogram").items():
                counter_histogram[value] = counter_histogram.get(value, 0) + count
            peak_counter_abs = max(peak_counter_abs, stats["counter_peak_abs"])
            saturation_updates += round(stats["saturation_rate"] * args.measurements * args.block_size)
            fire_epochs += int(stats["fires"] > 0)
            if args.oracle_every and step % args.oracle_every == 0:
                stats.update(oracle_metrics(model, proposal, indices, val_x[:args.oracle_size],
                                            val_y[:args.oracle_size], args.tie_tolerance))
                if stats["fires"]:
                    audited_deltas.append(stats["fire_delta_loss"])
            model.weights.copy_(proposal)  # No validation-based acceptance filter.
            total_fires += stats["fires"]
            train_calls = 2 * args.measurements * step
            row = {"step": step, "elapsed_seconds": time.perf_counter() - started,
                   "total_fires": total_fires, "train_forward_calls": train_calls,
                   "train_forward_examples": train_calls * args.batch_size, **stats}
            if step % args.eval_every == 0 or step == args.steps:
                final = evaluate(model, val_x, val_y)
                row.update(val_loss=final["loss"], val_accuracy=final["accuracy"])
                print(f"step={step:5d} val_loss={final['loss']:.4f} val_acc={final['accuracy']:.2%} "
                      f"fires={total_fires} Cmax={stats['counter_max']} Cmean={stats['counter_mean']:.2f} "
                      f"absCmean={stats['counter_abs_mean']:.2f} "
                      f"sat={stats['counter_saturated_count']}/{stats['counter_count']} "
                      f"capacity=±{stats['counter_capacity']} train_forwards={train_calls}", flush=True)
            row.update(total_forward_calls=model.forward_calls, total_forward_examples=model.forward_examples)
            writer.writerow(row)
            handle.flush()
    # Test is evaluated once, after the predetermined number of training steps.
    test = evaluate(model, test_x, test_y)
    summary = {"num_params": model.num_params, "initial_validation": initial,
               "final_validation": final, "test": test, "total_fires": total_fires,
               "audited_fire_events": len(audited_deltas),
               "audited_p_improve": (sum(d < 0 for d in audited_deltas) / len(audited_deltas)
                                     if audited_deltas else None),
               "audited_mean_delta_loss": (sum(audited_deltas) / len(audited_deltas)
                                            if audited_deltas else None),
               "train_forward_calls": 2 * args.measurements * args.steps,
               "train_forward_examples": 2 * args.measurements * args.steps * args.batch_size,
               "total_forward_calls": model.forward_calls, "total_forward_examples": model.forward_examples,
               "elapsed_seconds": time.perf_counter() - started}
    observations = sum(counter_histogram.values())
    counter_summary = {
        "scope": "measured edges at every epoch end, before reset; pooled across epochs",
        "count": observations, "min": min(map(int, counter_histogram)),
        "max": max(map(int, counter_histogram)),
        "mean": sum(int(k) * v for k, v in counter_histogram.items()) / observations,
        "abs_mean": sum(abs(int(k)) * v for k, v in counter_histogram.items()) / observations,
        "abs_max": max(abs(int(k)) for k in counter_histogram),
        "peak_abs_during_accumulation": peak_counter_abs,
        "capacity": 2 ** (args.counter_bits - 1) - 1,
        "saturation_update_count": saturation_updates,
        "histogram": dict(sorted(counter_histogram.items(), key=lambda item: int(item[0]))),
    }
    capacity = counter_summary["capacity"]
    counter_summary["saturated_count"] = sum(counter_histogram.get(str(k), 0) for k in (-capacity, capacity))
    counter_summary["saturated_fraction"] = counter_summary["saturated_count"] / observations
    summary.update(counter_distribution=counter_summary, fire_epochs=fire_epochs,
                   fire_epoch_fraction=fire_epochs / args.steps)
    with (args.output_dir / "counter_histogram.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["counter_value", "count", "fraction"])
        for value, count in counter_summary["histogram"].items():
            writer.writerow([value, count, count / observations])
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save({"weights": model.weights.cpu(), "config": config}, args.output_dir / "model.pt")
    print(f"test loss={test['loss']:.4f}, accuracy={test['accuracy']:.2%}; saved to {args.output_dir}")


if __name__ == "__main__":
    main()
