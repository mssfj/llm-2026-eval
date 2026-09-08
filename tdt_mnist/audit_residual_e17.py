"""Independent E17 final audit and detailed tables; never evaluates test data."""
import csv
import json
from pathlib import Path
import shutil
import statistics
import sys
import numpy as np
import torch
from run_residual_e17 import (ROOT, HERE, CONDITIONS, config, setup, sha, dump,
                              write_csv, observe, probes)
from train import load_data
from residual_stream import ResidualStreamModel


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def grouped(rows, keys, metrics):
    groups = {}
    for r in rows:
        key = tuple(r[k] for k in keys)
        groups.setdefault(key, []).append(r)
    result = []
    for key, values in sorted(groups.items()):
        row = dict(zip(keys, key))
        row['seeds'] = len(values)
        assert len(values) == 3
        for metric in metrics:
            numbers = [float(r[metric]) for r in values]
            row[metric+'_mean'] = statistics.mean(numbers)
            row[metric+'_sample_std'] = statistics.stdev(numbers)
        result.append(row)
    return result


def main(root):
    setup()
    assert json.loads((root/'status.json').read_text())['complete'], 'Wait for all nine runs'
    manifest = json.loads((root/'manifest.json').read_text())
    old = HERE/'results/depth-precision-16layer-100k-20260907'
    old_manifest = json.loads((old/'manifest.json').read_text())
    assert manifest['data_sha256'] == old_manifest['data_sha256']
    for name, digest in manifest['sources'].items():
        assert sha(root/'sources'/name) == digest
        assert sha(HERE/name) == digest, f'Live source changed: {name}'
    for name in ['train.py', 'activation_quantization.py', 'depth_diagnostics.py']:
        assert manifest['sources'][name] == old_manifest['sources'][name]
    curves = []; windows = []; conditioned = []; audits = []; selections = {}
    for c, (precision, activation) in CONDITIONS.items():
        for seed in range(3):
            out = root/'per_seed'/f'{c}-seed{seed}'
            cfg = json.loads((out/'config.json').read_text())
            s = json.loads((out/'summary.json').read_text())
            for name, digest in json.loads((out/'manifest.json').read_text()).items():
                assert sha(out/name) == digest, (out, name)
            a = config(seed, cfg['data_dir'])
            for key in ['pool_shape','steps','measurements','block_size','threshold','batch_size',
                        'max_fires','counter_bits','leak','scale','scale_ema','min_scale','zero_rate','gain',
                        'train_size','val_size','test_size','data_seed','seed','batch_seed','threads','eval_every','oracle_every']:
                assert cfg[key] == getattr(a, key), (out, key)
            assert cfg['activation_precision']==precision and cfg['hidden_activation']==activation
            assert cfg['rmsnorm_eps']==1e-8 and cfg['rmsnorm_trainable'] is False
            for name, digest in manifest['data_sha256'].items():
                assert sha(Path(a.data_dir)/'MNIST/raw'/name) == digest
            metrics = read(out/'metrics.csv')
            layer_rows = read(out/'layer_metrics.csv')
            assert len(metrics)==12000 and len(layer_rows)==12000*18
            selected = np.zeros((12000,18), dtype=np.int64)
            fires = np.zeros_like(selected)
            for index, r in enumerate(layer_rows):
                step, layer = divmod(index,18)
                assert int(r['step'])==step+1 and int(r['layer'])==layer
                selected[step,layer] = int(r['selected_coordinates'])
                fires[step,layer] = int(r['fires'])
                assert int(r['selected_interval']) == int(selected[step,layer]>0)
                assert int(r['fire_interval']) == int(fires[step,layer]>0)
            assert (selected.sum(1)==16).all() and (fires.sum(1)<=1).all()
            assert (fires<=selected).all()
            if seed in selections:
                assert np.array_equal(selected, selections[seed]), 'Paired selection RNG diverged'
            else:
                selections[seed]=selected
            y = np.load(out/'abs_y.npy').astype('float64')
            assert np.isfinite(y).all() and (y>=0).all()
            scale=.02
            histogram_count=0
            for i,r in enumerate(metrics):
                assert int(r['step']) == i+1
                assert float(r['scale']) == scale
                scale=max(1e-5,.9*scale+.1*sorted(y[i])[32])
                assert int(r['fires']) == int(fires[i].sum())
                assert int(r['train_forward_calls']) == 128*(i+1)
                assert float(r['abs_y_mean']) == float(y[i].mean())
                assert int(r['zero_difference_count']) == int((y[i]==0).sum())
                assert float(r['zero_difference_fraction']) == float((y[i]==0).mean())
                assert int(r['counter_capacity'])==127 and int(r['counter_peak_abs'])<=64
                assert float(r['saturation_rate'])==0
                histogram_count += int(r['counter_count'])
                assert bool(r['val_accuracy']) == ((i+1)%500==0)
                if r['val_accuracy']:
                    curves.append(dict(condition=c,seed=seed,step=i+1,
                        val_accuracy=float(r['val_accuracy']),val_loss=float(r['val_loss'])))
            assert sum(s['counter_histogram'].values())==histogram_count
            assert s['train_forward_calls']==1536000 and s['diagnostic_probe_forward_calls']==4608
            assert s['total_forward_calls']==1536000+4608+25+10
            assert s['total_forward_examples']==1536000*128+4608*128+25000+10000
            for layer,t in enumerate(s['layer_totals']):
                assert t['fires']==int(fires[:,layer].sum())
                assert t['selected_coordinates']==int(selected[:,layer].sum())
                assert t['selected_intervals']==int((selected[:,layer]>0).sum())
                assert t['fire_intervals']==int((fires[:,layer]>0).sum())
                values = y[selected[:,layer]>0].ravel()
                conditioned.append(dict(condition=c,seed=seed,layer=layer,matrix=cfg['matrix_names'][layer],
                    selected_intervals=len(values)//64,candidate_pairs=len(values),
                    mean_abs_y=float(values.mean()),zero_fraction=float((values==0).mean()),
                    scope='mixed-layer block conditioned on selection; not isolated contribution'))
            for start in range(0,12000,500):
                values=y[start:start+500].ravel()
                windows.append(dict(condition=c,seed=seed,start_step=start+1,end_step=start+500,
                    mean_abs_y=float(values.mean()),median_abs_y=float(np.median(values)),
                    p90_abs_y=float(np.quantile(values,.9)),zero_fraction=float((values==0).mean())))
            m = ResidualStreamModel(seed,precision,activation)
            (x,labels),(vx,vy),_=load_data(a,m.device)
            checkpoint=torch.load(out/'model.pt',map_location='cpu',weights_only=False)
            assert checkpoint['config']==cfg
            expected_probes=read(out/'probes.csv')
            assert len(expected_probes)==2304
            replay=[]
            for stage,step in [('initial',0),('final',12000)]:
                if stage=='final':
                    m.weights.copy_(checkpoint['weights'])
                val,signals,activations,ratios=observe(m,vx,vy,step)
                assert val==s[stage+'_validation'], (out,stage,'validation replay')
                for filename,rows in [('signal.csv',signals),('activation.csv',activations),('rms_ratios.csv',ratios)]:
                    expected=[r for r in read(out/filename) if int(r['step'])==step]
                    assert len(rows)==len(expected)
                    for actual,saved in zip(rows,expected):
                        assert {k: '' if v is None else str(v) for k,v in actual.items()} == saved, (out,stage,filename)
                actual_probes=probes(m,x,labels,seed,stage)
                saved_probes=[r for r in expected_probes if r['stage']==stage]
                for actual,saved in zip(actual_probes,saved_probes):
                    assert {k:str(v) for k,v in actual.items()}==saved, (out,stage,'probe replay')
                replay.append(dict(stage=stage,validation=val,probe_pairs=len(actual_probes)))
            curves.append(dict(condition=c,seed=seed,step=0,val_accuracy=s['initial_validation']['accuracy'],
                               val_loss=s['initial_validation']['loss']))
            audits.append(dict(condition=c,seed=seed,passed=True,validation_and_probe_replay=replay,
                audit_forward_calls=m.forward_calls,test_evaluated_during_audit=False))
            print(f'Audited {c} seed{seed}: full logs, initial/final validation and probes match',flush=True)
    write_csv(root/'aggregate/validation_curves.csv',curves)
    write_csv(root/'signal/abs_y_windows.csv',windows)
    write_csv(root/'signal/abs_y_by_selected_matrix.csv',conditioned)
    group_specs=[('firing/matrices.csv',['condition','layer','matrix'],
                  ['fires','selected_intervals','all_interval_firing_rate','selected_interval_firing_rate']),
                 ('signal/rms_ratios.csv',['condition','step','block'],
                  ['stream_rms','branch_rms','branch_stream_rms_ratio']),
                 ('signal/isolated_candidates.csv',['condition','stage','layer','matrix'],['mean_abs_y','zero_fraction']),
                 ('signal/metrics.csv',['condition','step','layer','matrix','stage'],['rms','zero_fraction','nonfinite_count']),
                 ('activation/metrics.csv',['condition','step','layer','matrix'],['relative_squared_error','cosine_mean_valid'])]
    for name,keys,metrics in group_specs:
        path=root/name
        write_csv(path.with_name(path.stem+'_aggregate.csv'),grouped(read(path),keys,metrics))
    historical=[]
    for r in read(HERE/'results/depth-activation-100k-20260907/identity-a32/per_seed.csv'):
        if r['depth']=='16' and r['threshold']=='8':
            historical.append(dict(condition='E16_A32',seed=int(r['seed']),test_accuracy_percent=100*float(r['test_accuracy'])))
    for r in read(old/'per_seed.csv'):
        if r['condition']=='a8':
            historical.append(dict(condition='E16_A8',seed=int(r['seed']),test_accuracy_percent=100*float(r['test_accuracy'])))
    assert len(historical)==6
    write_csv(root/'aggregate/historical_controls.csv',historical)
    current=read(root/'per_seed/results.csv')
    comparison=[]
    for c in CONDITIONS:
        for base in ('E16_A32','E16_A8'):
            diffs=[float(next(r['test_accuracy_percent'] for r in current if r['condition']==c and int(r['seed'])==seed))-
                   next(r['test_accuracy_percent'] for r in historical if r['condition']==base and r['seed']==seed) for seed in range(3)]
            comparison.append(dict(condition=c,control=base,mean_difference_pp=statistics.mean(diffs),
                sample_std_difference_pp=statistics.stdev(diffs),seed0_pp=diffs[0],seed1_pp=diffs[1],seed2_pp=diffs[2]))
    write_csv(root/'aggregate/historical_paired_differences.csv',comparison)
    lines=['# E17 層別一覧（3seed平均）','','全区間発火率と選択時発火率は異なる分母。行列番号は0始まり。', '']
    for c in CONDITIONS:
        lines += [f'## {c}', '', '| block | stream RMS | branch RMS | branch/stream |', '| --- | ---: | ---: | ---: |']
        for r in read(root/'signal/rms_ratios_aggregate.csv'):
            if r['condition']==c and r['step']=='12000':
                lines.append(f"| {r['block']} | {float(r['stream_rms_mean']):.5f} | {float(r['branch_rms_mean']):.5f} | {float(r['branch_stream_rms_ratio_mean']):.5f} |")
        lines += ['', '| 行列 | 全区間発火率 % | 選択時発火率 % | 発火数 | 初期 単独mean abs(y) | 最終 単独mean abs(y) |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        ps=read(root/'signal/isolated_candidates_aggregate.csv')
        for r in sorted(read(root/'firing/matrices_aggregate.csv'),key=lambda r:int(r['layer'])):
            if r['condition']!=c: continue
            candidate=[next(float(t['mean_abs_y_mean']) for t in ps if t['condition']==c and t['layer']==r['layer'] and t['stage']==stage) for stage in ('initial','final')]
            lines.append(f"| {r['matrix']} | {100*float(r['all_interval_firing_rate_mean']):.4f} | {100*float(r['selected_interval_firing_rate_mean']):.4f} | {float(r['fires_mean']):.2f} | {candidate[0]:.7f} | {candidate[1]:.7f} |")
        lines.append('')
    (root/'LAYER_TABLES.md').write_text('\n'.join(lines)+'\n')
    primary_values=[float(r['test_accuracy_percent']) for r in current if r['condition']=='E17a']
    primary=dict(mean_test_percent=statistics.mean(primary_values), fixed_control_percent=87.31,
        mean_improvement_pp=statistics.mean(primary_values)-87.31,
        seed_improvements_pp=[v-87.31 for v in primary_values],
        passed=statistics.mean(primary_values)>=90.31 and all(v>87.31 for v in primary_values))
    dump(root/'aggregate/primary_criterion.json',primary)
    shutil.copy2(Path(__file__),root/'sources'/Path(__file__).name)
    dump(root/'audit.json',dict(passed=True,runs=audits,legacy_source_hashes_match=True,
        mnist_hashes_match=True,paired_selection_rng_matches=True,test_evaluated=False,
        audit_source_sha256=sha(Path(__file__))))
    with (root/'README.md').open('a') as f:
        f.write('\n独立監査: audit.json。全区間ログ・S更新・発火集計・ソース/データハッシュを照合し、初期/最終validationと全層単独プローブを完全再現。監査でtestは評価していない。\n\n層別平均の一覧は[LAYER_TABLES.md](LAYER_TABLES.md)。各CSVの*_aggregate.csvに標本標準偏差を併記。\n')
    with (root/'README.md').open('a') as f:
        f.write('\nE17aの固定対照87.31%への平均差: '+str(primary['mean_improvement_pp'])+'ポイント。各seed差: '+str(primary['seed_improvements_pp'])+'ポイント。\n')
        f.write('\nE16の丸め前平均はA32 87.313333…%、A8 87.03%。主判定は事前登録の87.31%を固定使用。既存seed別の対応差はaggregate/historical_paired_differences.csv。E14は95,274個の連続重み、93.89±0.551%の参考値。\n')
        f.write('\n三値重み100,016個、18行列、FP32ストリーム幅76、8ブロック。学習forwardは1run 1,536,000回、9run合計13,824,000回。初期・最終の層単独診断は1run 4,608 forwardで別計上。\n')
    dump(root/'artifacts_sha256.json',{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
                                      if p.is_file() and p.name!='artifacts_sha256.json'})


if __name__=='__main__':
    main(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT)
