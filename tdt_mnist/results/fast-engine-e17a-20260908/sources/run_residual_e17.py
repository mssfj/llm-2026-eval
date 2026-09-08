"""Fixed E17 protocol. Preflight first, then nine independent CPU processes."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
from train import parser, load_data, evaluate, epoch, candidate_pair, loss, TernaryModel
from residual_stream import ResidualStreamModel
from depth_diagnostics import SignalObserver, layer_events, DEPTH_WIDTHS
from activation_quantization import ActivationObserver

HERE = Path(__file__).resolve().parent
ROOT = HERE/'results/residual-stream-a8-e17-20260908'
CONDITIONS = {'E17a': ('a8', 'relu'), 'E17b': ('a8', 'identity'), 'E17c': ('a32', 'relu')}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup():
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True)


def config(seed, data):
    a = parser().parse_args([])
    for k, v in dict(pool_shape=[9, 10], steps=12000, measurements=64,
        block_size=16, threshold=8, batch_size=128, max_fires=1,
        counter_bits=8, leak=1., scale=.02, scale_ema=.1, min_scale=1e-5,
        train_size=10000, val_size=1000, test_size=10000, data_seed=0,
        seed=seed, batch_seed=seed+100000, threads=1, eval_every=500,
        oracle_every=0, data_dir=Path(data), download=False,
        loss_diagnostics=True, layer_diagnostics=True).items():
        setattr(a, k, v)
    return a


def observe(m, vx, vy, step):
    m.signal_observer = SignalObserver()
    m.activation_observer = ActivationObserver(18, m.activation_precision)
    val = evaluate(m, vx, vy)
    signals = [{'step': step, 'matrix': m.matrix_names[r['layer']], **r}
               for r in m.signal_observer.summary()]
    activation = []
    for r in m.activation_observer.summary():
        r['code_histogram'] = json.dumps(r['code_histogram'], sort_keys=True)
        activation.append({'step': step, 'matrix': m.matrix_names[r['layer']], **r})
    ratios = []
    for b in range(8):
        rows = {r['stage']: r for r in signals if r['layer'] == 1+2*b}
        stream = rows['stream_before']['rms']
        branch = rows['branch_output']['rms']
        ratios.append(dict(step=step, block=b, stream_rms=stream, branch_rms=branch,
                           branch_stream_rms_ratio=branch/stream if stream else None))
    m.signal_observer = m.activation_observer = None
    return val, signals, activation, ratios


def probes(m, x, y, seed, stage):
    rows = []
    offset = 0
    before = m.weights.clone()
    for layer, shape in enumerate(m.shapes):
        size = math.prod(shape)
        g = torch.Generator().manual_seed(700000+seed*1000+18*20+layer)
        bg = torch.Generator().manual_seed(900000+seed*1000+18*20+layer)
        for pair in range(64):
            indices = torch.randperm(size, generator=g)[:16]+offset
            plus, minus, _, _ = candidate_pair(m.weights, indices, g)
            batch = torch.randint(len(x), (128,), generator=bg)
            lp = loss(m, x[batch], y[batch], plus)
            lm = loss(m, x[batch], y[batch], minus)
            rows.append(dict(stage=stage, layer=layer, matrix=m.matrix_names[layer],
                pair=pair, perturbed_coordinates=16, loss_plus=float(lp),
                loss_minus=float(lm), abs_y=float((lp-lm).abs())))
        offset += size
    assert torch.equal(before, m.weights)
    return rows


def preflight(root, data):
    setup()
    root.mkdir(parents=True, exist_ok=True)
    assert not (root/'preflight.json').exists(), 'Preflight already exists; do not overwrite'
    a = config(0, data)
    a.download = True
    (x, y), (vx, vy), _ = load_data(a, torch.device('cpu'))
    rows = []
    for condition, (precision, activation) in CONDITIONS.items():
        for seed in range(3):
            m = ResidualStreamModel(seed, precision, activation)
            val = evaluate(m, vx, vy)
            rows.append(dict(condition=condition, seed=seed, **val))
    timings = []
    for label, m in [('E17a', ResidualStreamModel()),
                     ('E16_A8', TernaryModel(pool_shape=(9, 10), hidden_sizes=DEPTH_WIDTHS[16],
                         activation_precision='a8', hidden_activation='identity'))]:
        a.download = False
        g = torch.Generator().manual_seed(1)
        bg = torch.Generator().manual_seed(100000)
        start_calls = m.forward_calls
        start = time.perf_counter()
        epoch(m, x, y, a, g, .02, bg)  # Disposable model; no update or test evaluation.
        elapsed = time.perf_counter()-start
        assert m.forward_calls-start_calls == 128
        timings.append(dict(model=label, forwards=128, seconds=elapsed,
                            matrix_macs_per_example=m.num_params))
    write_csv(root/'per_seed/initial_validation.csv', rows)
    dump(root/'preflight.json', dict(initial_validation=rows, timings=timings,
        num_params=100016, matrices=18, test_evaluated=False,
        data_sha256={p.name: sha(p) for p in sorted((Path(data)/'MNIST/raw').glob('*-ubyte'))}))
    print(json.dumps(dict(initial_validation=rows, timings=timings), indent=2), flush=True)


def train_run(condition, seed, root, data):
    setup()
    a = config(seed, data)
    precision, activation = CONDITIONS[condition]
    m = ResidualStreamModel(seed, precision, activation)
    out = root/'per_seed'/f'{condition}-seed{seed}'
    out.mkdir(parents=True, exist_ok=False)
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    cfg.update(condition=condition, architecture='residual_stream', width=76, blocks=8,
        activation_precision=precision, hidden_activation=activation, num_params=m.num_params,
        shapes=m.shapes, layer_scales=m.scales, matrix_names=m.matrix_names,
        rmsnorm_eps=1e-8, rmsnorm_trainable=False, stream_dtype='float32',
        weight_storage='int8 ternary', torch_version=torch.__version__,
        source_sha256=json.loads((root/'manifest.json').read_text())['sources'])
    dump(out/'config.json', cfg)
    (x, y), (vx, vy), (tx, ty) = load_data(a, m.device)
    g = torch.Generator().manual_seed(seed+1)
    bg = torch.Generator().manual_seed(seed+100000)
    initial, signals, activations, ratios = observe(m, vx, vy, 0)
    expected = [r for r in json.loads((root/'preflight.json').read_text())['initial_validation']
                if r['condition'] == condition and r['seed'] == seed][0]
    assert initial == {k: expected[k] for k in ('loss', 'accuracy')}
    probe_rows = probes(m, x, y, seed, 'initial')
    write_csv(out/'probes.csv', probe_rows)
    write_csv(out/'signal.csv', signals)
    write_csv(out/'activation.csv', activations)
    write_csv(out/'rms_ratios.csv', ratios)
    abs_y = np.lib.format.open_memmap(out/'abs_y.npy', mode='w+', dtype='float32', shape=(12000,64))
    totals = [dict(fires=0, selected_intervals=0, fire_intervals=0, selected_coordinates=0) for _ in range(18)]
    scale = .02
    calls = 0
    started = time.perf_counter()
    histogram = {}
    with (out/'metrics.csv').open('w', newline='') as mf, (out/'layer_metrics.csv').open('w', newline='') as lf:
        mw = None
        lw = csv.DictWriter(lf, fieldnames=['step','layer','parameters','selected_coordinates','selected_interval','fires','fire_interval'])
        lw.writeheader()
        for step in range(1, 12001):
            before = m.forward_calls
            proposal, indices, stats, scale = epoch(m, x, y, a, g, scale, bg)
            assert m.forward_calls-before == 128
            calls += 128
            values = np.asarray(stats.pop('abs_y_values'), dtype=np.float32)
            assert np.isfinite(values).all()
            abs_y[step-1] = values
            for k, v in stats.pop('counter_histogram').items():
                histogram[k] = histogram.get(k, 0)+v
            for e in layer_events(m, proposal, indices):
                lw.writerow(dict(step=step, **e))
                t = totals[e['layer']]
                for dst, src in [('fires','fires'), ('selected_intervals','selected_interval'),
                                 ('fire_intervals','fire_interval'), ('selected_coordinates','selected_coordinates')]:
                    t[dst] += e[src]
            m.weights.copy_(proposal)
            row = dict(step=step, elapsed_seconds=time.perf_counter()-started,
                train_forward_calls=calls, abs_y_mean=float(values.astype('float64').mean()),
                val_loss=None, val_accuracy=None, **stats)
            if step % 500 == 0:
                final, sr, ar, rr = observe(m, vx, vy, step)
                signals.extend(sr); activations.extend(ar); ratios.extend(rr)
                row.update(val_loss=final['loss'], val_accuracy=final['accuracy'])
                write_csv(out/'signal.csv', signals)
                write_csv(out/'activation.csv', activations)
                write_csv(out/'rms_ratios.csv', ratios)
                abs_y.flush()
                torch.save(dict(weights=m.weights, step=step, scale=scale, generator=g.get_state(),
                    batch_generator=bg.get_state(), config=cfg), out/'checkpoint.pt')
                dump(out/'progress.json', dict(step=step, validation=final, elapsed_seconds=time.perf_counter()-started))
                print(f'{condition} seed{seed} step={step} validation={final["accuracy"]:.3%}', flush=True)
            if mw is None:
                mw = csv.DictWriter(mf, fieldnames=list(row)); mw.writeheader()
            mw.writerow(row)
            mf.flush(); lf.flush()
    abs_y.flush()
    assert calls == 1536000
    probe_rows.extend(probes(m, x, y, seed, 'final'))
    write_csv(out/'probes.csv', probe_rows)
    # The only test evaluation, after all predetermined training is complete.
    test = evaluate(m, tx, ty)
    torch.save(dict(weights=m.weights, config=cfg), out/'model.pt')
    summary = dict(condition=condition, seed=seed, initial_validation=initial,
        final_validation=final, test=test, train_forward_calls=calls,
        diagnostic_probe_forward_calls=4608, total_forward_calls=m.forward_calls,
        total_forward_examples=m.forward_examples, test_evaluations=1,
        elapsed_seconds=time.perf_counter()-started, layer_totals=totals,
        counter_histogram=histogram, num_params=m.num_params)
    dump(out/'summary.json', summary)
    dump(out/'manifest.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    return condition, seed


def aggregate(root):
    per_seed = []; firing = []; signals = []; activations = []; ratios = []; isolated = []
    for condition in CONDITIONS:
        for seed in range(3):
            out = root/'per_seed'/f'{condition}-seed{seed}'
            s = json.loads((out/'summary.json').read_text())
            cfg = json.loads((out/'config.json').read_text())
            checkpoint = torch.load(out/'model.pt', weights_only=False, map_location='cpu')
            assert checkpoint['weights'].dtype == torch.int8 and checkpoint['weights'].numel() == 100016
            assert set(checkpoint['weights'].unique().tolist()) <= {-1,0,1}
            assert s['train_forward_calls'] == 1536000 and s['test_evaluations'] == 1
            metrics = list(csv.DictReader((out/'metrics.csv').open()))
            assert [int(r['step']) for r in metrics] == list(range(1,12001))
            assert [int(r['step']) for r in metrics if r['val_accuracy']] == list(range(500,12001,500))
            y = np.load(out/'abs_y.npy')
            assert y.shape == (12000,64)
            per_seed.append(dict(condition=condition, seed=seed, test_accuracy_percent=100*s['test']['accuracy'],
                validation_accuracy_percent=100*s['final_validation']['accuracy'],
                mean_abs_y=float(y.astype('float64').mean()), zero_difference_fraction=float((y==0).mean()),
                fires=sum(t['fires'] for t in s['layer_totals'])))
            for layer, t in enumerate(s['layer_totals']):
                firing.append(dict(condition=condition, seed=seed, layer=layer, matrix=cfg['matrix_names'][layer],
                    **t, all_interval_firing_rate=t['fire_intervals']/12000,
                    selected_interval_firing_rate=t['fire_intervals']/t['selected_intervals'] if t['selected_intervals'] else None))
            for name, target in [('signal',signals),('activation',activations),('rms_ratios',ratios)]:
                rows = list(csv.DictReader((out/f'{name}.csv').open()))
                assert len(set(r['step'] for r in rows)) == 25
                target.extend(dict(condition=condition, seed=seed, **r) for r in rows)
            rows = list(csv.DictReader((out/'probes.csv').open()))
            assert len(rows) == 2304
            for stage in ('initial','final'):
                for layer in range(18):
                    vals = [float(r['abs_y']) for r in rows if r['stage']==stage and int(r['layer'])==layer]
                    assert len(vals)==64
                    isolated.append(dict(condition=condition, seed=seed, stage=stage, layer=layer,
                        matrix=cfg['matrix_names'][layer], mean_abs_y=statistics.mean(vals),
                        std_abs_y=statistics.stdev(vals), zero_fraction=sum(v==0 for v in vals)/64))
    ag = []
    for c in CONDITIONS:
        values = [r['test_accuracy_percent'] for r in per_seed if r['condition']==c]
        ag.append(dict(condition=c, test_mean_percent=statistics.mean(values), test_sample_std_percent=statistics.stdev(values)))
    effects = []
    for label, pos, neg in [('ReLU_effect','E17a','E17b'), ('A8_cost','E17c','E17a')]:
        diffs = [next(r['test_accuracy_percent'] for r in per_seed if r['condition']==pos and r['seed']==s)-
                 next(r['test_accuracy_percent'] for r in per_seed if r['condition']==neg and r['seed']==s) for s in range(3)]
        effects.append(dict(effect=label, mean_pp=statistics.mean(diffs), sample_std_pp=statistics.stdev(diffs),
                            seed0_pp=diffs[0], seed1_pp=diffs[1], seed2_pp=diffs[2]))
    passed = ag[0]['test_mean_percent'] >= 90.31 and all(r['test_accuracy_percent']>87.31 for r in per_seed if r['condition']=='E17a')
    warnings = [dict(condition=r['condition'],seed=r['seed'],block=r['block'],ratio=float(r['branch_stream_rms_ratio']))
                for r in ratios if r['step']=='12000' and float(r['branch_stream_rms_ratio'])>.5]
    logits_warnings = [r for r in signals if r['step']=='12000' and r['layer']=='17' and r['stage']=='output' and float(r['rms'])>10]
    for path, rows in [('per_seed/results.csv',per_seed),('aggregate/results.csv',ag),('aggregate/paired_effects.csv',effects),
                       ('firing/matrices.csv',firing),('signal/metrics.csv',signals),('signal/rms_ratios.csv',ratios),
                       ('signal/isolated_candidates.csv',isolated),('activation/metrics.csv',activations)]:
        write_csv(root/path, rows)
    dump(root/'verification.json', dict(passed=True, runs=9, primary_criterion_passed=passed,
        rms_ratio_warnings=warnings, logits_warnings=logits_warnings))
    report = ['# E17 結果', '', '各条件12,000区間×3seed。平均±標本標準偏差。testは最終モデルのみ。', '',
              '| 条件 | test精度 (%) |', '| --- | ---: |']
    report += [f"| {r['condition']} | {r['test_mean_percent']:.4f} ± {r['test_sample_std_percent']:.4f} |" for r in ag]
    report += ['| E16 A32（既存） | 87.31 ± 0.471 |', '| E16 A8（既存） | 87.03 ± 0.104 |',
               '| E14 backprop（参考・異なる学習則と重み数） | 93.89 |', '',
               f'事前登録主判定: {"合格（非線形性が機能した）" if passed else "不合格（事前登録基準に未達）"}。', '']
    report += [f"{r['effect']}: {r['mean_pp']:.4f} ± {r['sample_std_pp']:.4f}ポイント（対応seed差）。" for r in effects]
    report += ['', 'E17a−E17bはReLU追加の効果。E17bにもRMSNorm・動的量子化の非線形性がある。',
        f'最終枝/ストリームRMS比>0.5: {len(warnings)}件。logits RMS>10: {len(logits_warnings)}件。詳細はverification.json。',
        '層別一覧: signal/metrics.csv、signal/rms_ratios.csv、signal/isolated_candidates.csv、firing/matrices.csv。',
        '量子化診断: activation/metrics.csv。全候補・区間・行列の生記録はper_seed配下。']
    (root/'README.md').write_text('\n'.join(report)+'\n')
    dump(root/'status.json',dict(complete=True,completed=9,expected=9))
    dump(root/'artifacts_sha256.json', {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
        if p.is_file() and p.name != 'artifacts_sha256.json'})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['preflight','run','aggregate'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--data', type=Path, default=HERE/'data')
    p.add_argument('--workers', type=int, default=9)
    a = p.parse_args()
    if a.mode == 'preflight':
        preflight(a.root, a.data); return
    if a.mode == 'aggregate':
        aggregate(a.root); return
    assert (a.root/'preflight.json').exists()
    assert not (a.root/'manifest.json').exists(), 'Existing experiment; do not overwrite'
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor','1ad7b12',revision],cwd=HERE,check=True)
    sources = ['train.py','activation_quantization.py','depth_diagnostics.py','residual_stream.py',
               'run_residual_e17.py','test_residual_stream.py','E17_PREREGISTRATION.md']
    source_dir = a.root/'sources'; source_dir.mkdir()
    for n in sources:
        shutil.copy2(HERE/n, source_dir/n)
    dump(a.root/'manifest.json',dict(git_revision=revision, preregistration_commit='1ad7b12',
        sources={n:sha(source_dir/n) for n in sources},conditions=CONDITIONS,seeds=[0,1,2],
        data_sha256=json.loads((a.root/'preflight.json').read_text())['data_sha256']))
    completed=[]; errors=[]
    dump(a.root/'status.json',dict(complete=False,completed=0,expected=9,errors=[]))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(train_run,c,s,a.root,a.data):(c,s) for c in CONDITIONS for s in range(3)}
        for f in as_completed(futures):
            try:
                completed.append(f.result())
            except Exception as e:
                errors.append(dict(run=futures[f],error=repr(e)))
            dump(a.root/'status.json',dict(complete=False,completed=len(completed),expected=9,errors=errors))
    if errors:
        raise RuntimeError(errors)
    aggregate(a.root)


if __name__ == '__main__':
    main()
