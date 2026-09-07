"""Deterministic activation encoding; decoding emulates FP32 accumulation.

A3 denotes three values, not three bits. Signed symmetric integer ranges
use 255 (A8), 15 (A4), or 3 (A3) levels and one FP32 scale per example/row.
"""
import torch


PRECISIONS = ("a32", "a16", "a8", "a4", "a3")
QMAX = {"a8": 127, "a4": 7, "a3": 1}


def encode_activation(x, precision, a3_method="absmax", threshold_factor=0.5):
    if a3_method not in ("absmax", "mean_threshold"):
        raise ValueError("unknown A3 method")
    if precision == "a3" and a3_method == "mean_threshold":
        magnitude = x.abs()
        threshold = threshold_factor * magnitude.mean(dim=-1, keepdim=True)
        selected = magnitude > threshold  # Equality maps to zero.
        count = selected.sum(dim=-1, keepdim=True)
        scale = (magnitude * selected).sum(dim=-1, keepdim=True) / count.clamp_min(1)
        scale = torch.where(count > 0, scale, torch.ones_like(scale))
        codes = (x.sign() * selected).to(torch.int8)
        return codes, scale
    if precision == "a32":
        return x, None
    if precision == "a16":
        return x.to(torch.float16), None
    if precision not in QMAX:
        raise ValueError(f"unknown activation precision: {precision}")
    limit = QMAX[precision]
    maximum = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(maximum > 0, (maximum / limit).clamp_min(torch.finfo(torch.float32).tiny),
                        torch.ones_like(maximum))
    codes = torch.round(x / scale).clamp(-limit, limit).to(torch.int8)
    return codes, scale


def decode_activation(codes, scale):
    values = codes.to(torch.float32)
    return values if scale is None else values * scale


def activation_description(precision, a3_method="absmax", threshold_factor=0.5, hidden_activation="relu"):
    if precision == "a3" and a3_method == "mean_threshold":
        description = activation_description(precision, hidden_activation=hidden_activation)
        description.update(a3_method=a3_method, threshold_factor=threshold_factor,
                           activation_scale="per-example, per-layer mean abs of values strictly above threshold; empty set scale=1",
                           rounding="q=sign(x) if abs(x)>threshold_factor*mean(abs(x)), otherwise 0; reconstruction=scale*q")
        return description
    return {
        "name": "W3" + precision.upper(),
        "quantized_locations": f"normalized pooled input and hidden activation after {hidden_activation}, immediately before each linear layer",
        "accumulator_dtype": "float32", "logits_dtype": "float32", "loss_dtype": "float32",
        "scale_dtype": "float32", "weight_scales": "fixed, not trained",
        "encoding": ("float32" if precision == "a32" else "float16" if precision == "a16"
                     else f"signed integer codes in [-{QMAX[precision]}, {QMAX[precision]}]"),
        "activation_scale": "none" if precision in ("a32", "a16") else "per-example, per-layer absmax / qmax; recomputed on every forward",
        "rounding": "native float cast" if precision in ("a32", "a16") else "nearest, ties to even",
        "storage": "native float tensor" if precision in ("a32", "a16") else "int8 codes with FP32 scale; sub-byte codes are not packed",
        "implementation": "encode then decode for FP32 linear kernels; no STE, no autograd, no integer-kernel speed claim",
        "relu_note": "A3 hidden activations use only codes 0 and +1 because ReLU is nonnegative" if precision == "a3" and hidden_activation == "relu" else None,
    }


class ActivationObserver:
    """Read-only diagnostics, enabled only during initial/final validation."""
    def __init__(self, layers, precision):
        self.precision = precision
        self.rows = [{"count": 0, "zeros": 0, "squared_error": 0.0,
                      "input_squared": 0.0, "code_histogram": {}} for _ in range(layers)]

    def record(self, layer, original, codes, reconstructed):
        row = self.rows[layer]
        row["count"] += original.numel()
        row["zeros"] += int((codes == 0).sum())
        # FP64 here is diagnostic only; these sums never enter the model or optimizer.
        row["squared_error"] += float((original.double() - reconstructed.double()).square().sum())
        row["input_squared"] += float(original.double().square().sum())
        if self.precision in QMAX:
            values, counts = torch.unique(codes, return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist()):
                key = str(value)
                row["code_histogram"][key] = row["code_histogram"].get(key, 0) + count

    def summary(self):
        return [{"layer": i, "values": r["count"],
                 "zero_fraction": r["zeros"] / r["count"],
                 "mse": r["squared_error"] / r["count"],
                 "relative_squared_error": r["squared_error"] / max(r["input_squared"], 1e-300),
                 "code_histogram": dict(sorted(r["code_histogram"].items(), key=lambda kv: int(kv[0])))}
                for i, r in enumerate(self.rows) if r["count"]]
