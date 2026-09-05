"""Audit completed length sweeps, including exact equality of training prefixes."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report_dir', type=Path)
    args = p.parse_args()
    report = args.report_dir
    manifest = json.loads((report / 'manifest.json').read_text())
    root = Path(manifest['output_dir'])
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((report / 'sources' / name).read_bytes()).hexdigest() == digest
    prefixes = {}
    summaries = []
    for length, block, seed in itertools.product(manifest['lengths'], manifest['blocks'], manifest['seeds']):
        run = root / f'steps{length}' / f'seed{seed}-block{block}-threshold{manifest["threshold"]}'
        config = json.loads((run / 'config.json').read_text())
        summary = json.loads((run / 'summary.json').read_text())
        for key, expected in dict(num_params=100000, activation_precision='a32', steps=length,
                                  block_size=block, seed=seed, data_seed=0, batch_seed=100000+seed,
                                  measurements=64, batch_size=128, max_fires=1,
                                  threshold=manifest['threshold'], oracle_every=0,
                                  train_size=10000, val_size=1000, test_size=0,
                                  shapes=[[1000, 90], [10, 1000]]).items():
            assert config[key] == expected, (run, key)
        metrics = list(csv.DictReader((run / 'metrics.csv').open()))
        assert len(metrics) == length + 1
        assert summary['train_forward_calls'] == 128 * length
        assert summary['train_forward_examples'] == 128 * 128 * length
        assert sum(int(r['fires']) for r in metrics[1:]) == summary['total_fires']
        assert sum(summary['layer_update_counts']) == summary['total_fires']
        assert all(v > 0 for v in summary['layer_update_counts'])
        counters = summary['counter_distribution']
        assert sum(counters['histogram'].values()) == counters['count']
        assert sum(int(r['counter_count']) for r in metrics[1:]) == counters['count']
        assert counters['capacity'] == 127 and counters['abs_max'] <= 64
        assert counters['saturation_update_count'] == counters['saturated_count'] == 0
        weights = torch.load(run / 'model.pt', map_location='cpu', weights_only=False)['weights']
        assert weights.dtype == torch.int8 and weights.numel() == 100000
        assert set(weights.unique().tolist()) <= {-1, 0, 1}
        # Timing differs between jobs, but all other logged training quantities must match.
        prefixes[length, block, seed] = [{k: v for k, v in r.items() if k != 'elapsed_seconds'} for r in metrics]
        summaries.append({'steps': length, 'block': block, 'seed': seed,
                          'loss_improved': summary['final_validation']['loss'] < summary['initial_validation']['loss'],
                          'accuracy_improved': summary['final_validation']['accuracy'] > summary['initial_validation']['accuracy']})
    matches = 0
    for block, seed in itertools.product(manifest['blocks'], manifest['seeds']):
        for short, long in itertools.combinations(sorted(manifest['lengths']), 2):
            assert prefixes[short, block, seed] == prefixes[long, block, seed][:short+1], (short, long, block, seed)
            matches += 1
    result = {'runs_checked': len(summaries), 'exact_prefix_comparisons_passed': matches,
              'loss_improved_runs': sum(r['loss_improved'] for r in summaries),
              'accuracy_improved_runs': sum(r['accuracy_improved'] for r in summaries),
              'source_hashes_verified': True, 'conditions_budgets_counters_and_weights_verified': True}
    (report / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
