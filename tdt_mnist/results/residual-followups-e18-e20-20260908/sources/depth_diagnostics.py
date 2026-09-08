"""Read-only forward signal and per-layer weight firing diagnostics."""
import math
import torch

# Closely matched hidden widths, exact 100,000 active weights, 90 inputs / 10 outputs.
DEPTH_WIDTHS = {
    4: [200, 200, 200],
    8: [120, 120, 121, 121, 121, 122, 123],
    16: [79, 82, 80, 82, 80, 82, 81, 81, 81, 81, 81, 82, 81, 82, 80],
}


def layer_events(model, proposal, indices):
    sizes = [math.prod(s) for s in model.shapes]
    ends = torch.tensor(sizes, device=model.device).cumsum(0)
    selected = torch.bincount(torch.bucketize(indices, ends, right=True), minlength=len(sizes)).tolist()
    changed = torch.nonzero(model.weights != proposal).flatten()
    fires = torch.bincount(torch.bucketize(changed, ends, right=True), minlength=len(sizes)).tolist()
    return [{"layer": i, "parameters": size, "selected_coordinates": s,
             "selected_interval": int(s > 0), "fires": f, "fire_interval": int(f > 0)}
            for i, (size, s, f) in enumerate(zip(sizes, selected, fires))]


class SignalObserver:
    def __init__(self):
        self.rows = {}

    @torch.no_grad()
    def record(self, layer, stage, x):
        key = (layer, stage)
        if key not in self.rows:
            self.rows[key] = {"values": 0, "examples": 0, "sum": 0., "squared_sum": 0.,
                "zeros": 0, "negative": 0, "max_abs": 0., "nonfinite": 0,
                "ever_nonzero": torch.zeros(x.shape[-1], dtype=torch.bool, device=x.device)}
        r = self.rows[key]
        values = x.double()
        r['values'] += x.numel()
        r['examples'] += len(x)
        r['sum'] += float(values.sum())
        r['squared_sum'] += float(values.square().sum())
        r['zeros'] += int((x == 0).sum())
        r['negative'] += int((x < 0).sum())
        r['max_abs'] = max(r['max_abs'], float(x.abs().max()))
        r['nonfinite'] += int((~torch.isfinite(x)).sum())
        r['ever_nonzero'] |= (x != 0).any(dim=0)

    def summary(self):
        output = []
        for (layer, stage), r in sorted(self.rows.items()):
            mean = r['sum']/r['values']
            ms = r['squared_sum']/r['values']
            output.append({'layer': layer, 'stage': stage, 'values': r['values'], 'examples': r['examples'],
                'features': r['ever_nonzero'].numel(), 'mean': mean, 'std': math.sqrt(max(0., ms-mean*mean)),
                'rms': math.sqrt(ms), 'zero_fraction': r['zeros']/r['values'],
                'negative_fraction': r['negative']/r['values'], 'max_abs': r['max_abs'],
                'nonfinite_count': r['nonfinite'],
                'dead_features': int((~r['ever_nonzero']).sum()),
                'dead_feature_fraction': float((~r['ever_nonzero']).double().mean())})
        return output
