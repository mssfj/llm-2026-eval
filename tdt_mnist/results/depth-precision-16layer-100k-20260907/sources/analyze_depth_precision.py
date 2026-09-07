"""Audit A16/A8/A4 and saved A32; aggregate signals, firing and loss differences."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv,json,hashlib,subprocess,sys,itertools,statistics,shutil,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CONDITIONS=['a32','a16','a8','a4']
LABELS=['A32','A16','A8','A4']
def read(path):
    with path.open() as f:return list(csv.DictReader(f))
def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def describe(values):
    v=np.asarray(values,dtype=np.float64).reshape(-1)
    return {'count':len(v),'mean':float(v.mean()),'rms':float(np.sqrt(np.mean(v*v))),'median':float(np.median(v)),
        'p90':float(np.quantile(v,.9)),'p99':float(np.quantile(v,.99)),'max':float(v.max()),'zero_fraction':float(np.mean(v==0))}
def aggregate(rows,keys,fields):
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    result=[]
    for key,rs in groups.items():
        a=dict(zip(keys,key));a['seeds']=len(rs)
        for f in fields:
            vals=[float(r[f]) for r in rs if r[f] not in [None,'']]
            a[f+'_mean']=statistics.mean(vals) if vals else None
            a[f+'_std']=statistics.stdev(vals) if len(vals)>1 else 0. if vals else None
        result.append(a)
    return result

def main(p):
    manifest=json.loads((p/'manifest.json').read_text());status=json.loads((p/'status.json').read_text())
    assert status['training_complete'] and not status['errors']
    assert not manifest['smoke']
    for name,digest in manifest['sources'].items():assert hashlib.sha256((p/'sources'/name).read_bytes()).hexdigest()==digest
    records=json.loads((p/'summaries.json').read_text());assert len(records)==9
    for seed in [0,1,2]:
        d=Path(manifest['baseline_root'])/f'depth16-threshold8-seed{seed}'
        assert hashlib.sha256((d/'config.json').read_bytes()).hexdigest()==manifest['baseline_config_sha256'][str(seed)]
        records.append({'condition':'a32','seed':seed,'run_directory':str(d),**json.loads((d/'summary.json').read_text())})
    assert {(r['condition'],r['seed']) for r in records}==set(itertools.product(CONDITIONS,[0,1,2]))
    per=[];signals=[];windows=[];targeted=[];fire=[];checks=[];selection={};curves={};all_y=[]
    basecfg=json.loads((Path(manifest['baseline_root'])/'depth16-threshold8-seed0/config.json').read_text())
    for name,digest in manifest['data_sha256'].items():assert hashlib.sha256((Path(basecfg['data_dir'])/'MNIST/raw'/name).read_bytes()).hexdigest()==digest
    fixed=['pool_shape','hidden_sizes','hidden_size','hidden_activation','a3_method','a3_threshold_factor',
        'steps','measurements','threshold','block_size','batch_size','max_fires','train_size','val_size','test_size','data_seed',
        'gain','zero_rate','counter_bits','leak','scale','scale_ema','min_scale','oracle_every','eval_every','device','threads']
    for r in records:
        d=Path(r['run_directory']);cfg=json.loads((d/'config.json').read_text());tags={k:r[k] for k in ['condition','seed']}
        for key in fixed:assert cfg[key]==basecfg[key],(d,key)
        assert cfg.get('a3_improvement','none')=='none' and cfg['activation_precision']==r['condition']
        assert cfg['seed']==r['seed'] and cfg['batch_seed']==r['seed']+100000
        assert r['num_params']==100000 and r['train_forward_calls']==1536000
        rows=read(d/'metrics.csv');assert len(rows)==12001
        y=np.load(d/'abs_y.npy',mmap_mode='r');assert y.shape==(12000,64) and y.dtype==np.float32 and np.isfinite(y).all() and (y>=0).all()
        assert int((y==0).sum())==r['zero_difference_count']
        ys=describe(y);all_y.append({**tags,**ys})
        for k in ['mean','rms','median','p90','p99','max']:assert abs(ys[k]-r['abs_y_statistics'][k])<1e-12
        scales=np.array([float(row['scale']) for row in rows[1:]])
        norm=describe(y/scales[:,None]);all_y[-1].update({'normalized_'+k:v for k,v in norm.items()})
        selected=np.zeros((12000,16),dtype=np.int16);counts=np.zeros(16,dtype=np.int64);selected_intervals=np.zeros(16,dtype=np.int64)
        with (d/'layer_metrics.csv').open() as f:
            n=0
            for row in csv.DictReader(f):
                step=int(row['step'])-1;l=int(row['layer']);sel=int(row['selected_coordinates']);fired=int(row['fires'])
                assert step==n//16 and l==n%16
                selected[step,l]=sel;counts[l]+=fired;selected_intervals[l]+=sel>0;n+=1
                assert int(row['cumulative_fires'])==counts[l]
            assert n==192000
        assert np.all(selected.sum(1)==16)
        assert counts.tolist()==r['layer_update_counts'] and int(counts.sum())==r['total_fires']
        assert selected.sum(0).tolist()==r['layer_selected_coordinates']
        assert selected_intervals.tolist()==r['layer_selected_intervals']
        selection[(r['condition'],r['seed'])]=selected
        for l in range(16):
            mask=selected[:,l]>0
            targeted.append({**tags,'layer':l,'selected_intervals':int(mask.sum()),**describe(y[mask])})
            fire.append({**tags,'layer':l,'fires':int(counts[l]),'all_interval_rate':counts[l]/12000,
                'selected_interval_rate':counts[l]/selected_intervals[l],'selected_coordinates':int(selected[:,l].sum())})
        for i,row in enumerate(rows[1:]):
            assert int(row['step'])==i+1
            assert abs(float(y[i].astype(np.float64).mean())-float(row['abs_y_mean']))<1e-12
            assert int((y[i]==0).sum())==int(row['zero_difference_count'])
        assert sum(int(row['fires']) for row in rows[1:])==r['total_fires']
        for start in range(0,12000,500):
            end=start+500
            windows.append({**tags,'start_step':start+1,'end_step':end,**describe(y[start:end]),
                'normalized_mean':float((y[start:end]/scales[start:end,None]).mean()),
                'clip_rate':statistics.mean(float(row['clip_rate']) for row in rows[start+1:end+1]),
                'nonzero_vote_rate':statistics.mean(float(row['nonzero_vote_rate']) for row in rows[start+1:end+1])})
        sr=read(d/'signal_metrics.csv');assert {int(row['step']) for row in sr}==set(range(0,12001,500))
        for row in sr:
            assert int(row['nonfinite_count'])==0 and all(np.isfinite(float(row[k])) for k in ['rms','mean','std','max_abs'])
            signals.append({**tags,**row})
        outputs=[row for row in sr if row['stage']=='output'];assert len(outputs)==25*16
        curves[(r['condition'],r['seed'])]=[(int(row['step']),float(row['val_accuracy'])) for row in rows if row['val_accuracy']]
        per.append({**tags,'initial_val_accuracy':r['initial_validation']['accuracy'],'initial_val_loss':r['initial_validation']['loss'],
            'val_accuracy':r['final_validation']['accuracy'],'val_loss':r['final_validation']['loss'],
            'test_accuracy':r['test']['accuracy'],'test_loss':r['test']['loss'],'total_fires':r['total_fires'],
            'train_forward_calls':r['train_forward_calls'],'elapsed_seconds':r['elapsed_seconds'],'zero_difference_fraction':ys['zero_fraction'],
            'abs_y_mean':ys['mean'],'abs_y_rms':ys['rms'],'normalized_abs_y_mean':norm['mean']})
        checks.append({**tags,'passed':True,'layer_rows':192000,'signal_rows':len(sr),'abs_y_count':int(y.size),
            'abs_y_sha256':hashlib.sha256((d/'abs_y.npy').read_bytes()).hexdigest()})
    for seed in [0,1,2]:
        for c in CONDITIONS[1:]:assert np.array_equal(selection[(c,seed)],selection[('a32',seed)])
    # Read-only replay provides matching cosine/error diagnostics for the historical controls as well.
    def diagnostic(r):
        dest=p/'diagnostics'/f"{r['condition']}-seed{r['seed']}"
        result=subprocess.run([sys.executable,str(ROOT/'tdt_mnist/diagnose_depth_precision.py'),r['run_directory'],str(dest)],check=True,capture_output=True,text=True)
        return r,dest,json.loads(result.stdout)
    activations=[];replay_signals=[];isolated=[];diag_checks=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for r,dest,check in pool.map(diagnostic,records):
            tags={k:r[k] for k in ['condition','seed']};diag_checks.append(check)
            activations.extend({**tags,**row} for row in read(dest/'activation.csv'))
            replay_signals.extend({**tags,**row} for row in read(dest/'signals.csv'))
            probes=read(dest/'probes.csv');assert len(probes)==2048
            for row in probes:
                assert float(row['abs_y'])==float(np.abs(np.float32(row['loss_plus'])-np.float32(row['loss_minus'])))
            for stage,l in itertools.product(['initial','final'],range(16)):
                vals=[float(row['abs_y']) for row in probes if row['stage']==stage and int(row['layer'])==l]
                assert len(vals)==64;isolated.append({**tags,'stage':stage,'layer':l,**describe(vals)})
    fields=['val_accuracy','val_loss','test_accuracy','test_loss','total_fires','zero_difference_fraction','abs_y_mean','abs_y_rms','normalized_abs_y_mean']
    agg=aggregate(per,['condition'],fields)
    sa=aggregate(signals,['condition','step','layer','stage'],['rms','mean','std','zero_fraction','max_abs'])
    ia=aggregate(isolated,['condition','stage','layer'],['mean','rms','median','p90','p99','zero_fraction'])
    aa=aggregate(activations,['condition','stage','layer'],['mse','relative_squared_error','cosine_mean_valid','cosine_undefined_examples','zero_fraction'])
    wa=aggregate(windows,['condition','start_step','end_step'],['mean','rms','zero_fraction','normalized_mean','clip_rate','nonzero_vote_rate'])
    fa=aggregate(fire,['condition','layer'],['fires','all_interval_rate','selected_interval_rate'])
    pairs=[]
    for c,seed in itertools.product(CONDITIONS[1:],[0,1,2]):
        a=next(r for r in per if r['condition']==c and r['seed']==seed);b=next(r for r in per if r['condition']=='a32' and r['seed']==seed)
        pairs.append({'condition':c,'seed':seed,**{k+'_difference':a[k]-b[k] for k in ['val_accuracy','test_accuracy','val_loss','total_fires']}})
    outputs={'per_seed.csv':per,'aggregate.csv':agg,'signal_metrics.csv':signals,'signal_aggregate.csv':sa,
        'abs_y_per_seed.csv':all_y,'abs_y_windows.csv':windows,'abs_y_windows_aggregate.csv':wa,
        'abs_y_by_perturbed_layer.csv':targeted,'layer_firing.csv':fire,'layer_firing_aggregate.csv':fa,
        'activation_diagnostics.csv':activations,'activation_aggregate.csv':aa,'checkpoint_signal_diagnostics.csv':replay_signals,
        'layer_isolated_abs_y.csv':isolated,'layer_isolated_abs_y_aggregate.csv':ia,'paired_effects.csv':pairs}
    for name,rs in outputs.items():write(p/name,rs)
    plot(p,agg,sa,ia,aa,wa,curves)
    lines=['# 活性化精度比較：16層・100,000重み・ReLUなし','',
        'A16/A8/A4×seed0/1/2の新規9runと、既存A32×3seedを比較。TDTの重みは全条件三値。カウンタ閾値8、12,000区間、block16、K64、batch128、max_fires1。',
        '隠れ幅79,82,80,82,80,82,81,81,81,81,81,82,81,82,80。入力90、出力10、出力を含む16線形層、バイアスなし。',
        '前処理・データ分割・乱数seed・初期重み・TDT設定はA32対照と共通。train10,000 / validation1,000 / test10,000。',
        'A16は入力をFP16へ変換後FP32へ復元。A8/A4は各例・各層のabsmaxでスケールを決め、符号付き整数[-127,127]/[-7,7]へ丸めて復元する（255値/15値）。A32はFP32のまま。',
        '正規化済み画像入力と全隠れ層を線形層直前で量子化。重みスケール・積和・logits・損失はFP32。追加のRMS正規化や残差接続なし。','',
        '| 活性化精度 | validation精度 % | test精度 % | 発火数 | mean abs(y) | 損失差ゼロ率 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for c in CONDITIONS:
        r=next(r for r in agg if r['condition']==c)
        def fmt(k,scale=1):return f"{r[k+'_mean']*scale:.4g} ± {r[k+'_std']*scale:.3g}"
        lines.append(f"| {c.upper()} | {fmt('val_accuracy',100)} | {fmt('test_accuracy',100)} | {fmt('total_fires')} | {fmt('abs_y_mean')} | {fmt('zero_difference_fraction')} |")
    lines+=['','3seedの平均±標本標準偏差。最終12,000区間のモデルを評価し、testで条件選択はしない。paired_effects.csvに同じseedのA32との差。','',
        'signal_metrics.csv：初期と500区間ごとの全16層RMS。outputがh_l（最終層はlogits）、inputが量子化・復元後の線形層入力。',
        'layer_firing.csv：層ごとの総発火数、全区間に対する発火率、選択区間に対する発火率。',
        '各runのabs_y.npyは全768,000候補対のFP32 abs(L(T+)-L(T−))。abs_y_windows.csvには500区間ごとの分布と正規化比|y|/S、クリップ率、非ゼロ票率。',
        'abs_y_by_perturbed_layer.csvは当該層を含む摂動ブロックに条件付けた|y|。複数層を含むブロックの値は重複分類されるため、層単独の寄与ではない。',
        'layer_isolated_abs_y.csv：初期・最終の各層だけに16辺を摂動した64候補対の統計。diagnostics/*/probes.csvに全24,576候補対の生データ。専用乱数で評価し学習には反映しない。',
        'activation_diagnostics.csv：初期・最終の層別量子化MSE、相対二乗誤差、コサイン、整数コード分布（A8/A4）。A16/A32は浮動小数点なので整数コード分布欄は空の辞書。',
        '保存モデルの初期・最終validationを再評価して学習ログと完全一致を確認。全seed・全区間で摂動座標の層別選択数がA32対照と一致することも検証。','']
    (p/'README.md').write_text('\n'.join(lines))
    sources={}
    for name in ['analyze_depth_precision.py','diagnose_depth_precision.py','run_depth_precision.py']:
        f=ROOT/'tdt_mnist'/name;shutil.copy2(f,p/'sources'/name);sources[name]=hashlib.sha256(f.read_bytes()).hexdigest()
    (p/'verification.json').write_text(json.dumps({'passed':True,'new_runs':9,'controls':3,'checks':checks,'diagnostics':diag_checks,
        'new_training_pairs':9*768000,'all_training_pairs_with_controls':12*768000,'independent_probe_pairs':24576,
        'paired_layer_selection_verified':True,'exact_parameter_counts_and_checkpoint_replay_verified':True,'analysis_sources':sources},indent=2))
    print('Verified nine new runs, three controls, all layer signals and isolated probes.',flush=True)

def plot(p,agg,sa,ia,aa,wa,curves):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#666666','#0072B2','#D55E00','#009E73']
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(p/(name+'.'+ext),dpi=170)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for c,label,color in zip(CONDITIONS,LABELS,colors):
        arr=np.array([[v*100 for _,v in curves[(c,s)]] for s in [0,1,2]]);xs=[step for step,_ in curves[(c,0)]]
        axes[0].plot(xs,arr.mean(0),label=label,color=color);axes[0].fill_between(xs,arr.mean(0)-arr.std(0,ddof=1),arr.mean(0)+arr.std(0,ddof=1),color=color,alpha=.12)
    ordered=[next(r for r in agg if r['condition']==c) for c in CONDITIONS]
    axes[1].bar(range(4),[100*r['test_accuracy_mean'] for r in ordered],yerr=[100*r['test_accuracy_std'] for r in ordered],color=colors)
    axes[1].set_xticks(range(4),LABELS,rotation=15);axes[1].set_ylabel('Final test accuracy (%)')
    axes[0].set(xlabel='Training interval',ylabel='Validation accuracy (%)');axes[0].legend();axes[0].grid(alpha=.2)
    save(fig,'accuracy_comparison')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for c,label,color in zip(CONDITIONS,LABELS,colors):
        for step,ls in [('0','--'),('12000','-')]:
            rows=sorted([r for r in sa if r['condition']==c and str(r['step'])==step and r['stage']=='output'],key=lambda r:int(r['layer']))
            axes[0,0].plot(range(1,17),[max(r['rms_mean'],1e-30) for r in rows],ls,label=label+(' initial' if step=='0' else ' final'),color=color)
        rows=sorted([r for r in ia if r['condition']==c and r['stage']=='final'],key=lambda r:int(r['layer']))
        axes[0,1].plot(range(1,17),[max(r['mean_mean'],1e-30) for r in rows],label=label,color=color)
        rows=sorted([r for r in aa if r['condition']==c and r['stage']=='final'],key=lambda r:int(r['layer']))
        axes[1,0].plot(range(1,17),[r['relative_squared_error_mean'] for r in rows],label=label,color=color)
        rows=sorted([r for r in wa if r['condition']==c],key=lambda r:int(r['end_step']))
        axes[1,1].plot([r['end_step'] for r in rows],[max(r['mean_mean'],1e-30) for r in rows],label=label,color=color)
    for ax in [axes[0,0],axes[0,1],axes[1,1]]:ax.set_yscale('log')
    axes[0,0].set(ylabel='RMS(h_l)',xlabel='Layer (last = logits)');axes[0,1].set(ylabel='Isolated-layer mean |y| (final)',xlabel='Perturbed layer')
    axes[1,0].set(ylabel='Relative squared quantization error',xlabel='Quantized linear input (1 = image)');axes[1,1].set(ylabel='Training mean |y|',xlabel='Training interval')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=6)
    save(fig,'layer_diagnostics')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('report',type=Path);main(a.parse_args().report)
