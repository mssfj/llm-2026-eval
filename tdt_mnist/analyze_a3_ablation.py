"""Audit completed A3 factorial runs and generate diagnostics/figures."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report_dir', type=Path)
    args = p.parse_args()
    root = args.report_dir
    manifest = json.loads((root / 'manifest.json').read_text())
    status = json.loads((root / 'status.json').read_text())
    assert status['complete'] and status['completed'] == 4*len(manifest['seeds'])
    runs = Path(manifest['output_dir'])
    records = json.loads((root / 'summaries.json').read_text())
    assert len(records) == status['completed']
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((root / 'sources' / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest['data_sha256'].items():
        assert hashlib.sha256((Path(manifest['data_dir'])/'MNIST/raw'/name).read_bytes()).hexdigest() == digest
    expected_params = (math.prod(manifest['pool_shape'])+10)*manifest['hidden_size']
    expected_calls = 2*manifest['measurements']*manifest['steps']
    curves = {}
    audit = []
    keys = set()
    import torch
    for record in records:
        a, m, seed = (record[k] for k in ('hidden_activation', 'a3_method', 'seed'))
        assert (a, m, seed) not in keys
        keys.add((a, m, seed))
        label = f'{a}-{m}-seed{seed}'
        directory = runs / label
        config = json.loads((directory / 'config.json').read_text())
        rows = read_csv(directory / 'metrics.csv')
        checkpoint = torch.load(directory / 'model.pt', map_location='cpu', weights_only=False)
        assert json.loads(json.dumps(checkpoint['config'])) == config
        assert checkpoint['weights'].numel() == expected_params
        assert set(checkpoint['weights'].unique().tolist()).issubset({-1, 0, 1})
        assert config['seed'] == seed and config['batch_seed'] == seed+100000
        assert config['hidden_activation'] == a and config['a3_method'] == m
        for field in ('steps', 'measurements', 'block_size', 'threshold', 'max_fires', 'batch_size',
                      'train_size', 'val_size', 'test_size', 'data_seed', 'pool_shape', 'hidden_size',
                      'activation_precision', 'counter_bits', 'gain', 'zero_rate', 'scale', 'scale_ema',
                      'leak', 'a3_threshold_factor', 'oracle_every', 'eval_every', 'device'):
            assert config[field] == manifest[field], (label, field)
        assert record['num_params'] == expected_params == config['num_params']
        assert len(rows) == manifest['steps']+1
        assert [int(r['step']) for r in rows] == list(range(manifest['steps']+1))
        assert record['train_forward_calls'] == expected_calls == int(rows[-1]['train_forward_calls'])
        assert record['candidate_pair_count'] == expected_calls//2
        zero_count = sum(int(r['zero_difference_count']) for r in rows[1:])
        assert zero_count == record['zero_difference_count']
        assert zero_count/(expected_calls//2) == record['zero_difference_fraction']
        assert sum(int(r['fires']) for r in rows[1:]) == record['total_fires']
        assert sum(record['layer_update_counts']) == record['total_fires']
        assert record['final_validation']['accuracy'] == float(rows[-1]['val_accuracy'])
        for r in rows[1:]:
            assert 0 <= int(r['zero_difference_count']) <= manifest['measurements']
            assert int(r['zero_difference_count'])/manifest['measurements'] == float(r['zero_difference_fraction'])
            assert 0 <= int(r['fires']) <= manifest['max_fires']
        for stage in ('initial', 'final'):
            for obs in record[f'{stage}_activation_statistics']:
                width = math.prod(manifest['pool_shape']) if obs['layer'] == 0 else manifest['hidden_size']
                assert obs['values'] == width*manifest['val_size']
                assert sum(obs['code_histogram'].values()) == obs['values']
                assert obs['code_histogram'].get('0', 0)/obs['values'] == obs['zero_fraction']
                assert all(math.isfinite(obs[k]) and obs[k] >= 0 for k in ('mse','relative_squared_error'))
                if a == 'relu' and obs['layer'] == 1:
                    assert obs['code_histogram'].get('-1', 0) == 0
        curves[(a, m, seed)] = rows
        audit.append({'run': label, 'steps': len(rows)-1, 'train_forward_calls': expected_calls,
                      'zero_difference_count': zero_count, 'checkpoint_ternary': True, 'passed': True})
    assert keys == set(itertools.product(('relu','identity'), ('absmax','mean_threshold'), manifest['seeds']))
    assert len({r['total_forward_calls'] for r in records}) == 1
    (root/'verification.json').write_text(json.dumps({'passed': True, 'runs': audit,
        'source_and_data_hashes_verified': True, 'equal_forward_budgets': True}, indent=2)+'\n')
    diags = []
    for a, m, stage, layer in itertools.product(('relu','identity'), ('absmax','mean_threshold'), ('initial','final'), (0,1)):
        members = [r[f'{stage}_activation_statistics'][layer] for r in records if (r['hidden_activation'],r['a3_method']) == (a,m)]
        row = {'hidden_activation':a,'a3_method':m,'stage':stage,'layer':layer,'seeds':len(members)}
        for metric in ('zero_fraction','mse','relative_squared_error'):
            row[metric+'_mean'], row[metric+'_std'] = mean_std([r[metric] for r in members])
        for code in (-1,0,1):
            row[f'code_{code}_fraction_mean'], row[f'code_{code}_fraction_std'] = mean_std([r['code_histogram'].get(str(code),0)/r['values'] for r in members])
        diags.append(row)
    write_csv(root/'activation_aggregate.csv', diags)
    effects = []
    lookup = {(r['hidden_activation'],r['a3_method'],r['seed']):r for r in records}
    for seed in manifest['seeds']:
        for metric, field in (('val_accuracy','accuracy'),('val_loss','loss')):
            v = {(a,m):lookup[(a,m,seed)]['final_validation'][field] for a,m in itertools.product(('relu','identity'),('absmax','mean_threshold'))}
            effects.append({'seed':seed,'metric':metric,
                'remove_relu_at_absmax':v[('identity','absmax')]-v[('relu','absmax')],
                'remove_relu_at_mean_threshold':v[('identity','mean_threshold')]-v[('relu','mean_threshold')],
                'change_quantizer_with_relu':v[('relu','mean_threshold')]-v[('relu','absmax')],
                'change_quantizer_without_relu':v[('identity','mean_threshold')]-v[('identity','absmax')],
                'interaction':v[('identity','mean_threshold')]-v[('relu','mean_threshold')]-v[('identity','absmax')]+v[('relu','absmax')]})
    write_csv(root/'paired_effects.csv', effects)
    # Append final layer diagnostics after the sweep has finished writing its report.
    readme = root/'README.md'
    marker = '\n## 最終検証の層別診断\n'
    lines = [marker, '各値は3seedの平均（seed数の実値はmanifest参照）。相対誤差は量子化誤差の二乗和/量子化前の二乗和。', '',
             '| 活性化 | 量子化 | 層 | MSE | 相対二乗誤差 | −1 % | 0 % | +1 % |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in diags:
        if r['stage'] != 'final':
            continue
        layer_name = '入力' if r['layer'] == 0 else '隠れ'
        lines.append(f"| {r['hidden_activation']} | {r['a3_method']} | {layer_name} | {r['mse_mean']:.6g} | {r['relative_squared_error_mean']:.6f} | "
                     f"{100*r['code_-1_fraction_mean']:.3f} | {100*r['code_0_fraction_mean']:.3f} | {100*r['code_1_fraction_mean']:.3f} |")
    lines += ['', 'activation_aggregate.csvに標本標準偏差を含む全診断を保存。paired_effects.csvは同じseed内の効果差と交互作用。',
              '量子化方式を変えると入力表現と隠れ表現の両方が変わるため、その寄与をこの4条件だけでは分離できない。',
              'ReLUあり/なしでは量子化前の分布自体が異なるため、相対誤差の低さだけで分類精度を説明しない。',
              'comparison.png/svgに学習曲線、500区間ごとの損失差ゼロ率、最終層別誤差、隠れコード分布を保存。',
              'verification.jsonで全runの区間数・forward予算・3値重み・診断件数・ログ集計・ソース/データハッシュを照合済み。', '']
    readme.write_text(readme.read_text().split(marker)[0]+'\n'.join(lines))
    (root/'analysis_manifest.json').write_text(json.dumps({'script':Path(__file__).name,
        'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    conditions=list(itertools.product(('relu','identity'),('absmax','mean_threshold')))
    labels=['ReLU / absmax','ReLU / threshold','No ReLU / absmax','No ReLU / threshold']
    fig, axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for (a,m), label in zip(conditions,labels):
        seed_rows=[curves[(a,m,s)] for s in manifest['seeds']]
        val_rows=[[r for r in rows if r['val_accuracy']] for rows in seed_rows]
        steps=[int(r['step']) for r in val_rows[0]]
        vals=np.array([[100*float(r['val_accuracy']) for r in rows] for rows in val_rows])
        mean=vals.mean(axis=0); std=vals.std(axis=0,ddof=1) if len(vals)>1 else np.zeros_like(mean)
        axes[0,0].plot(steps,mean,label=label)
        axes[0,0].fill_between(steps,mean-std,mean+std,alpha=.15)
        window=500
        ends=list(range(window,manifest['steps']+1,window))
        if not ends or ends[-1]!=manifest['steps']: ends.append(manifest['steps'])
        zero=[]
        start=0
        for end in ends:
            zero.append(statistics.mean([100*sum(int(r['zero_difference_count']) for r in rows[start+1:end+1])/((end-start)*manifest['measurements']) for rows in seed_rows]))
            start=end
        axes[0,1].plot(ends,zero,label=label,marker='.',markersize=4)
    for ax in axes[0]:
        ax.set_xlabel('Accumulation intervals');ax.grid(alpha=.2)
    axes[0,0].set_ylabel('Validation accuracy (%)');axes[0,0].legend(fontsize=8)
    axes[0,1].set_ylabel('Exactly zero loss difference (%)')
    positions=np.arange(4)
    for layer, offset in ((0,-.18),(1,.18)):
        members=[next(r for r in diags if (r['hidden_activation'],r['a3_method'],r['stage'],r['layer'])==(a,m,'final',layer)) for a,m in conditions]
        axes[1,0].bar(positions+offset,[r['relative_squared_error_mean'] for r in members],width=.35,
                      yerr=[r['relative_squared_error_std'] for r in members],label='Input' if layer==0 else 'Hidden',capsize=3)
    axes[1,0].set_ylabel('Final relative squared quantization error');axes[1,0].legend()
    bottom=np.zeros(4)
    for code in (-1,0,1):
        vals=[100*next(r for r in diags if (r['hidden_activation'],r['a3_method'],r['stage'],r['layer'])==(a,m,'final',1))[f'code_{code}_fraction_mean'] for a,m in conditions]
        axes[1,1].bar(positions,vals,bottom=bottom,label=str(code));bottom+=vals
    axes[1,1].set_ylabel('Final hidden code distribution (%)');axes[1,1].legend(title='Code')
    for ax in axes[1]: ax.set_xticks(positions,labels,rotation=15,ha='right')
    fig.suptitle(f"{expected_params:,} TDT A3: ReLU x quantizer ({len(manifest['seeds'])} seeds; mean ± sample SD)")
    for extension in ('png','svg'): fig.savefig(root/f'comparison.{extension}',dpi=180)
    plt.close(fig)
    print(f'Verified {len(records)} runs; diagnostics and plots saved in {root}')


if __name__=='__main__':
    main()
