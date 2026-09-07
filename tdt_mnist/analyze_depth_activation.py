"""Audit 108 runs, export h_l RMS and layer-conditioned |y|, and compare methods."""
import argparse,csv,json,hashlib,itertools,statistics,subprocess,sys,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
CONDITIONS=['relu-a3-threshold','identity-a32','identity-a3-threshold']
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def describe(v):
    a=np.asarray(v,dtype=np.float64).reshape(-1)
    return dict(count=len(a),mean=float(a.mean()),rms=float(np.sqrt(np.mean(a*a))),median=float(np.median(a)),
        p90=float(np.quantile(a,.9)),p99=float(np.quantile(a,.99)),max=float(a.max()),zero_fraction=float(np.mean(a==0)))
def aggregate(rows,keys,fields):
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    out=[]
    for key,rs in groups.items():
        a=dict(zip(keys,key));a['seeds']=len(rs)
        for f in fields:
            v=[float(r[f]) for r in rs];a[f+'_mean']=statistics.mean(v);a[f+'_std']=statistics.stdev(v) if len(v)>1 else 0.
        out.append(a)
    return out

def main(root):
    per=[];rms=[];layer_y=[];window_y=[];all_stats=[];records=[];checks=[]
    for condition in CONDITIONS:
        directory=root/condition
        subprocess.run([sys.executable,str(Path(__file__).with_name('audit_depth_activation_condition.py')),str(directory)],check=True)
        per.extend({'condition':condition,**r} for r in read(directory/'per_seed.csv'))
        recs=json.loads((directory/'summaries.json').read_text())
        for r in recs:
            r['condition']=condition;records.append(r)
            run=Path(r['run_directory']);cfg=json.loads((run/'config.json').read_text())
            assert cfg['loss_diagnostics'] and cfg['layer_diagnostics']
            assert cfg['a3_method']==('absmax' if condition=='identity-a32' else 'mean_threshold')
            assert cfg['a3_threshold_factor']==.5
            tags={k:r[k] for k in ['condition','depth','threshold','seed']}
            y=np.load(run/'abs_y.npy',mmap_mode='r');steps=cfg['steps'];k=cfg['measurements'];depth=r['depth']
            assert y.shape==(steps,k) and y.dtype==np.float32 and np.isfinite(y).all() and (y>=0).all()
            assert int((y==0).sum())==r['zero_difference_count']
            stat=describe(y);all_stats.append({**tags,**stat})
            for key in ['mean','rms','median','p90','p99','max']:
                assert abs(stat[key]-r['abs_y_statistics'][key])<1e-12
            metrics=read(run/'metrics.csv')
            selected=np.zeros((steps,depth),dtype=np.int16)
            with (run/'layer_metrics.csv').open() as f:
                for row in csv.DictReader(f):selected[int(row['step'])-1,int(row['layer'])]=int(row['selected_coordinates'])
            assert np.all(selected.sum(1)==cfg['block_size'])
            for i,row in enumerate(metrics[1:]):
                z=y[i].astype(np.float64)
                assert abs(z.mean()-float(row['abs_y_mean']))<1e-12
                assert abs(np.sqrt(np.mean(z*z))-float(row['abs_y_rms']))<1e-12
                assert int((z==0).sum())==int(row['zero_difference_count'])
            np.save(root/condition/(run.name+'-perturbed-layer-counts.npy'),selected)
            for layer in range(depth):
                mask=selected[:,layer]>0
                assert mask.any()
                layer_y.append({**tags,'layer':layer,'selected_intervals':int(mask.sum()),
                    'selected_coordinates':int(selected[:,layer].sum()),**describe(y[mask])})
            for start in range(0,steps,500):
                end=min(start+500,steps);window_y.append({**tags,'start_step':start+1,'end_step':end,**describe(y[start:end])})
            for row in read(run/'signal_metrics.csv'):
                stage=row['stage']
                if stage in ['output','input']:
                    rms.append({**tags,'step':int(row['step']),'layer':int(row['layer']),
                        'quantity':'h_l' if stage=='output' else 'quantized_linear_input',
                        'rms':float(row['rms']),'zero_fraction':float(row['zero_fraction']),
                        'values':int(row['values'])})
            checks.append({**tags,'abs_y_count':y.size,'abs_y_sha256':hashlib.sha256((run/'abs_y.npy').read_bytes()).hexdigest(),
                'paired_layer_counts_verified':True})
    assert len(records)==108
    # Initial states and coordinate-selection opportunities match across activation conditions.
    for depth,threshold,seed in itertools.product([4,8,16],[1,4,8,16],[0,1,2]):
        rs=[r for r in records if (r['depth'],r['threshold'],r['seed'])==(depth,threshold,seed)]
        assert len(rs)==3
        for r in rs[1:]:
            for key in ['layer_selected_coordinates','layer_selected_intervals']:
                assert r[key]==rs[0][key]
    write(root/'per_seed.csv',per);write(root/'rms_h_per_seed.csv',rms)
    global_y=aggregate(all_stats,['condition','depth','threshold'],['mean','rms','median','p90','p99','max','zero_fraction'])
    write(root/'abs_y_aggregate.csv',global_y)
    write(root/'abs_y_per_seed.csv',all_stats);write(root/'abs_y_by_perturbed_layer.csv',layer_y);write(root/'abs_y_windows.csv',window_y)
    agg=aggregate(per,['condition','depth','threshold'],['val_accuracy','test_accuracy','val_loss','total_fires','zero_difference_fraction'])
    ra=aggregate(rms,['condition','depth','threshold','step','layer','quantity'],['rms','zero_fraction'])
    ya=aggregate(layer_y,['condition','depth','threshold','layer'],['mean','rms','median','p90','p99','zero_fraction'])
    wa=aggregate(window_y,['condition','depth','threshold','start_step','end_step'],['mean','rms','median','p90','zero_fraction'])
    write(root/'aggregate.csv',agg);write(root/'rms_h_aggregate.csv',ra);write(root/'abs_y_by_perturbed_layer_aggregate.csv',ya);write(root/'abs_y_windows_aggregate.csv',wa)
    # Independent, layer-isolated diagnostic probes never feed votes or weight updates.
    probe_dir=root/'layer_isolated_probes';probe_dir.mkdir(exist_ok=True)
    def task(r):
        dest=probe_dir/(r['condition']+'-'+Path(r['run_directory']).name+'.csv')
        meta=dest.with_suffix('.json')
        if meta.exists() and dest.exists():
            audit=json.loads(meta.read_text())
            assert audit['checkpoint_sha256']==hashlib.sha256((Path(r['run_directory'])/'model.pt').read_bytes()).hexdigest()
            assert audit['csv_sha256']==hashlib.sha256(dest.read_bytes()).hexdigest()
        else:
            cmd=[sys.executable,str(Path(__file__).with_name('probe_depth_activation.py')),r['run_directory'],str(dest)]
            result=subprocess.run(cmd,check=True,capture_output=True,text=True)
            audit=json.loads(result.stdout);meta.write_text(result.stdout)
        return r,dest,audit
    probes=[];probe_checks=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for r,dest,audit in pool.map(task,records):
            tags={k:r[k] for k in ['condition','depth','threshold','seed']};rows=read(dest)
            assert len(rows)==r['depth']*2*64
            for observation in rows:
                expected=float(np.abs(np.float32(observation['loss_plus'])-np.float32(observation['loss_minus'])))
                assert expected==float(observation['abs_y'])
                assert int(observation['perturbed_coordinates'])==16
            for stage,layer in itertools.product(['initial','final'],range(r['depth'])):
                values=[float(x['abs_y']) for x in rows if x['stage']==stage and int(x['layer'])==layer]
                assert len(values)==64
                probes.append({**tags,'stage':stage,'layer':layer,'layer_parameters':int(np.prod(r['shapes'][layer])),
                    'perturbed_coordinates_per_pair':16,'perturbed_fraction_per_pair':16/int(np.prod(r['shapes'][layer])),**describe(values)})
            probe_checks.append(audit)
    write(root/'layer_isolated_abs_y.csv',probes)
    pa=aggregate(probes,['condition','depth','threshold','stage','layer'],['mean','rms','median','p90','p99','zero_fraction'])
    write(root/'layer_isolated_abs_y_aggregate.csv',pa)
    plot(root,agg,ra,wa,pa)
    lines=['# 100k：活性化方式 × 深さ × カウンタ閾値','',
        '3方式（ReLU+A3閾値分離、ReLUなし+A32、ReLUなし+A3閾値分離）×4/8/16層×閾値1/4/8/16×seed0/1/2、108run。',
        '100,000重み固定、入力9×10、block16、K64、12,000区間、batch128、最大1発火。train10,000 / val1,000 / test10,000、data_seed0。',
        '前回と同じ幅・初期化・gain1・TDT設定。A3閾値は0.5 mean(|x|)、復元スケールは非ゼロ選択値の絶対値平均。入力と隠れ入力を量子化。',
        '各方式4並列で開始し、A3は実行中のrunを完走させた区切りで6並列へ変更。workersは学習乱数・条件に影響しない実行資源の設定で、runtime_workers.jsonに記録。',
        '各層のh_lは活性化後・次層量子化前（最終層はlogits）。RMS=sqrt(全検証例・全特徴の二乗平均)、初期と500区間ごと。量子化復元後の線形層入力RMSも保存。',
        'ReLUなし+A32はバイアスなし線形層の合成であり、前処理後の入力からlogitsまでが線形写像。A3はReLUなしでも量子化が非線形。深さと同時に幅も変化する固定予算比較。','',
        '## |y| の2種類の層別診断','',
        '1. 学習中の全候補対：各runのabs_y.npyはfloat32[12000,64]、行が区間step−1、列が候補対。同じミニバッチの平均FP32交差エントロピー差の絶対値で、Sで正規化する前。',
        '各runのlayer_metrics.csvと*-perturbed-layer-counts.npy[12000,depth]で摂動対象層・辺数と結合できる。同一区間の64候補対は同じ16座標でT+とT−が異なる。',
        'abs_y_by_perturbed_layer.csvは「当該層を含む摂動」の条件付き分布。同じ候補対を複数層に重複計上するため、層単独の因果寄与や加法分解ではない。',
        '2. 独立した層単独摂動：初期・最終モデルの各層で、その層だけから16辺を選び64候補対を評価。保存モデルを変更せず、学習の票にも使わない。',
        '固定train集合の128例ミニバッチ、seedから決定した専用乱数。初期/最終/方式間でミニバッチ・座標・乱数を対応させるが、状態依存の辺選択は異なり得る。',
        '層単独診断の摂動辺数は各層16本で一定だが、層の総重み数が異なるため摂動率は一定ではない。層別集計CSVに重み数と摂動率も記録する。',
        'layer_isolated_probes/*.csvは各候補のL(T+),L(T−),|y|を保存。学習中の候補対数・forward予算とは別枠。',
        'mean/rms/median/p90/p99/zero_fractionを保存。集計CSVには3seedの平均・標本標準偏差を保存。精度図のエラーバーは標本標準偏差で、RMS・|y|の曲線は平均。ゼロ率はFP32差の厳密な0で、表示丸めではない。','',
        '## 最終結果','', '| 条件 | 層 | 閾値 | val % | test % |', '| --- | ---: | ---: | ---: | ---: |']
    for r in agg:
        lines.append(f"| {r['condition']} | {r['depth']} | {r['threshold']} | {100*r['val_accuracy_mean']:.3f} ± {100*r['val_accuracy_std']:.3f} | {100*r['test_accuracy_mean']:.3f} ± {100*r['test_accuracy_std']:.3f} |")
    lines += ['', '## 閾値8：出力RMSと候補差の比較','',
        'v5の基準閾値8で固定した代表比較。すべて3seed平均。RMS(h_D)のDは出力層。|y|の学習欄は全768,000候補対/run、層単独欄は最終モデル各層64候補対/runの平均。',
        '| 条件 | 層 | 初期RMS(h_D) | 最終RMS(h_D) | 学習mean abs(y) | 層1単独mean abs(y) | 出力層単独mean abs(y) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for condition,depth in itertools.product(CONDITIONS,[4,8,16]):
        def find_row(rows,**extra):
            return next(r for r in rows if r['condition']==condition and int(r['depth'])==depth and int(r['threshold'])==8 and all(r[k]==v for k,v in extra.items()))
        initial=find_row(ra,step=0,layer=depth-1,quantity='h_l')['rms_mean']
        final=find_row(ra,step=12000,layer=depth-1,quantity='h_l')['rms_mean']
        training=find_row(global_y)['mean_mean']
        first=find_row(pa,stage='final',layer=0)['mean_mean']
        last=find_row(pa,stage='final',layer=depth-1)['mean_mean']
        lines.append(f'| {condition} | {depth} | {initial:.6g} | {final:.6g} | {training:.6g} | {first:.6g} | {last:.6g} |')
    lines += ['', '全層の時系列はrms_h_per_seed.csv / rms_h_aggregate.csv、学習候補差はabs_y_aggregate.csv / abs_y_windows_aggregate.csv、摂動対象層別はabs_y_by_perturbed_layer_aggregate.csv、層単独診断はlayer_isolated_abs_y_aggregate.csv。']
    lines += ['', '## 解釈上の制限','',
        'CSVのlayerは0始まり、図の層番号は1始まり。h_lは図の番号に従う（CSV layer=0がh_1）。',
        'RMSや|y|の大きさだけでは識別に有用な情報や更新方向の正しさを示さない。精度、損失、層別更新率を併せて評価する。',
        '3seedの平均・標準偏差であり、有意差検定や深層収束の保証ではない。過去ReLU+A32の結果は新3方式と区別し、必要に応じて前回ディレクトリを参照する。',
        'すべてのrunの設定・ソース・データハッシュは各条件のmanifest.json、個別監査はverification.json。',
        'A32は最初にA3専用引数との整合性チェックで開始前に停止したため、a3_method=absmax（A32では不使用）に修正して全36runを実施した。A3の設定は全runでmean_threshold。','']
    (root/'README.md').write_text('\n'.join(lines))
    sources={}
    (root/'sources').mkdir(exist_ok=True)
    for name in ['analyze_depth_activation.py','audit_depth_activation_condition.py','probe_depth_activation.py','run_depth_activation.py','finish_depth_activation.py','watch_depth_probes.py','rebalance_depth_activation.py']:
        f=Path(__file__).with_name(name);shutil.copy2(f,root/'sources'/name);sources[name]=hashlib.sha256(f.read_bytes()).hexdigest()
    (root/'verification.json').write_text(json.dumps({'passed':True,'runs':108,'candidate_pairs':108*768000,
        'checks':checks,'probe_checks':probe_checks,'analysis_sources':sources,'paired_selection_opportunities_verified':True},indent=2))
    print('Audited 108 runs, raw |y|, per-layer RMS and isolated-layer probes',flush=True)

def plot(root,agg,ra,wa,pa):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#0072B2','#D55E00','#009E73']
    labels=['ReLU + A3 threshold','No ReLU + A32','No ReLU + A3 threshold']
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(root/(name+'.'+ext),dpi=170)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,d in zip(axes,[4,8,16]):
        for c,label,color in zip(CONDITIONS,labels,colors):
            rows=sorted([r for r in agg if r['condition']==c and int(r['depth'])==d],key=lambda r:int(r['threshold']))
            ax.errorbar([int(r['threshold']) for r in rows],[100*r['test_accuracy_mean'] for r in rows],yerr=[100*r['test_accuracy_std'] for r in rows],marker='o',label=label,color=color)
        ax.set(title=f'{d} layers',xlabel='Counter threshold',ylabel='Final test accuracy (%)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    save(fig,'accuracy_comparison')
    for threshold in [1,4,8,16]:
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        for col,d in enumerate([4,8,16]):
            for c,label,color in zip(CONDITIONS,labels,colors):
                for step,ls in [(0,'--'),(12000,'-')]:
                    rows=sorted([r for r in ra if r['condition']==c and int(r['depth'])==d and int(r['threshold'])==threshold and int(r['step'])==step and r['quantity']=='h_l'],key=lambda r:int(r['layer']))
                    axes[0,col].plot([int(r['layer'])+1 for r in rows],[max(r['rms_mean'],1e-30) for r in rows],ls,color=color,label=label+(' initial' if step==0 else ' final'))
                rows=sorted([r for r in pa if r['condition']==c and int(r['depth'])==d and int(r['threshold'])==threshold and r['stage']=='final'],key=lambda r:int(r['layer']))
                axes[1,col].plot([int(r['layer'])+1 for r in rows],[max(r['mean_mean'],1e-30) for r in rows],color=color,label=label)
            for ax in axes[:,col]:ax.set_xticks(range(1,d+1));ax.set_yscale('log');ax.set_xlabel('Layer (last = logits)');ax.grid(alpha=.2);ax.legend(fontsize=6)
            axes[0,col].set(title=f'{d} layers, threshold {threshold}',ylabel='RMS(h_l)')
            axes[1,col].set_ylabel('Isolated-layer mean |y| (final)')
        save(fig,f'layer_rms_and_abs_y_threshold{threshold}')
        fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
        for ax,d in zip(axes,[4,8,16]):
            for c,label,color in zip(CONDITIONS,labels,colors):
                rows=sorted([r for r in wa if r['condition']==c and int(r['depth'])==d and int(r['threshold'])==threshold],key=lambda r:int(r['end_step']))
                ax.plot([int(r['end_step']) for r in rows],[max(r['mean_mean'],1e-30) for r in rows],label=label,color=color)
            ax.set_yscale('log');ax.set(title=f'{d} layers, threshold {threshold}',xlabel='Training interval',ylabel='Training mean |y| (500-interval window)');ax.grid(alpha=.2);ax.legend(fontsize=7)
        save(fig,f'abs_y_training_threshold{threshold}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
