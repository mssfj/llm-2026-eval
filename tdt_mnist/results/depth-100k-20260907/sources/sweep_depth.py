"""100k ReLU/A32 depth x threshold sweep, with mandatory layer diagnostics."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from train import parser, validate
from depth_diagnostics import DEPTH_WIDTHS


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def report(root, records):
    records=sorted(records,key=lambda r:(r['depth'],r['threshold'],r['seed']))
    rows=[];layers=[];signals=[]
    for r in records:
        tags={k:r[k] for k in ('depth','threshold','seed')}
        rows.append({**tags,'initial_val_loss':r['initial_validation']['loss'],
            'initial_val_accuracy':r['initial_validation']['accuracy'],
            'val_loss':r['final_validation']['loss'],'val_accuracy':r['final_validation']['accuracy'],
            'test_loss':r['test']['loss'],'test_accuracy':r['test']['accuracy'],
            **{k:r[k] for k in ('total_fires','fire_epoch_fraction','zero_difference_fraction','train_forward_calls','elapsed_seconds')}})
        for layer,shape in enumerate(r['shapes']):
            layers.append({**tags,'layer':layer,'in_width':shape[1],'out_width':shape[0],'parameters':shape[0]*shape[1],
                'fires':r['layer_update_counts'][layer], 'selected_coordinates':r['layer_selected_coordinates'][layer],
                'selected_intervals':r['layer_selected_intervals'][layer],'fire_intervals':r['layer_fire_intervals'][layer],
                'fire_interval_rate':r['layer_fire_interval_rates'][layer],
                'fire_given_selected_interval_rate':r['layer_fire_given_selected_interval_rates'][layer],
                'fires_per_selected_coordinate':r['layer_fires_per_selected_coordinate'][layer],
                'updates_per_parameter':r['layer_updates_per_parameter'][layer]})
        with (Path(r['run_directory'])/'signal_metrics.csv').open() as f:
            for obs in csv.DictReader(f):
                signals.append({**tags,**{k:int(v) if k in ('step','layer','values','examples','features','nonfinite_count','dead_features') else v if k=='stage' else float(v) for k,v in obs.items()}})
    aggregates=[]
    for depth,threshold in sorted({(r['depth'],r['threshold']) for r in rows}):
        members=[r for r in rows if (r['depth'],r['threshold'])==(depth,threshold)]
        row={'depth':depth,'threshold':threshold,'seeds':len(members)}
        for field in ('val_accuracy','val_loss','test_accuracy','test_loss','total_fires','fire_epoch_fraction','zero_difference_fraction'):
            vals=[r[field] for r in members]
            row[field+'_mean']=statistics.mean(vals)
            row[field+'_std']=statistics.stdev(vals) if len(vals)>1 else 0.
        aggregates.append(row)
    write_csv(root/'per_seed.csv',rows);write_csv(root/'aggregate.csv',aggregates)
    write_csv(root/'layer_firing.csv',layers);write_csv(root/'signal_metrics.csv',signals)
    (root/'summaries.json').write_text(json.dumps(records,indent=2)+'\n')
    lines=['# TDT 100k: 深さ × カウンタ閾値','',
        'ReLUあり・A32。層数は出力層を含む線形層の数。隠れ層の後だけにReLUを置く。',
        '既定実験: 深さ4/8/16 × 閾値1/4/8/16 × seed0/1/2。block=16、K=64、12,000区間、batch=128、最大1発火。',
        '訓練10,000、検証1,000、テスト10,000、分割seed=0、batch seed=seed+100000。',
        'v5のgain=1、初期ゼロ率1/3、固定alpha=1/sqrt(fan_in*(1-zero_rate))、INT8カウンタ、全区間リセット、Sの設定を維持。',
        '残差・バイアス・追加正規化・He gain補正は導入しない。全100,000重みが前向き計算に使われる。',
        '同じseedで平坦な初期3値重みと乱数列を対応させる。深さ間では重みの接続先と固定層スケールが異なる。',
        '深さを増やすと幅も狭くなるため、固定パラメータ予算の深さ・幅比較として解釈する。',
        'forward回数は一致するが、層数によるカーネル起動数などは異なり、実時間やFLOPs一致を意味しない。','',
        '| 線形層数 | 全層幅（入力→出力） | パラメータ数 |','| ---: | --- | ---: |']
    for d in sorted({r['depth'] for r in records}): lines.append(f"| {d} | {'→'.join(map(str,[90,*DEPTH_WIDTHS[d],10]))} | 100,000 |")
    lines+=['','## 精度','', '平均±標本標準偏差。検証は初期・500区間ごと・最終、テストは指定区間終了後の1回。',
        '| 層数 | 閾値 | seeds | 検証精度 % | テスト精度 % | 検証loss | 発火数 | 損失差ゼロ % |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in aggregates:
        def fmt(k,scale=1): return f"{r[k+'_mean']*scale:.3f} ± {r[k+'_std']*scale:.3f}"
        lines.append(f"| {r['depth']} | {r['threshold']} | {r['seeds']} | {fmt('val_accuracy',100)} | {fmt('test_accuracy',100)} | {fmt('val_loss')} | {fmt('total_fires')} | {fmt('zero_difference_fraction',100)} |")
    lines+=['','## 層別発火率の定義','',
        '層番号は入力側から0始まり。発火はニューロンの活性化ではなく重み更新を指す。',
        '- fire_interval_rate: その層で1個以上更新した区間数 / 全区間数。',
        '- fire_given_selected_interval_rate: その層で1個以上更新した区間数 / その層がblockに含まれた区間数。',
        '- fires_per_selected_coordinate: 更新回数 / blockに選ばれた座標数の累計（同一座標の再選択を含む）。',
        '- updates_per_parameter: 更新回数 / その層の重み数。確率ではなく反復更新を含む回数比。',
        '分母が0の条件付き率はnull。層の重み数による選択機会の差を明示する。',
        '各runのlayer_metrics.csvに全区間・全層の選択数・発火数・累積率、layer_firing.csvに最終集計を保存。','',
        '## 信号伝搬の診断','',
        'signal_metrics.csv: 検証集合を使い各層のinput/pre_activation/outputのRMS、平均、標準偏差、ゼロ率、負値率、最大絶対値を記録。',
        'dead_feature_fractionは検証集合の全サンプルで0だった特徴の割合で、母集団上の永久的な死滅を意味しない。',
        '診断は読み取り専用のFP64集計で、学習のforward、票や採否には戻さない。追加のforwardは使わず通常の検証で記録。',
        '層別発火があるだけで有用な信号伝搬を証明しない。信号強度、検証loss/精度、損失差ゼロ率と合わせて評価する。',
        'zero_difference_fractionは全訓練候補対のFP32損失差が厳密に0の割合。','',
        f'完了run数: {len(records)}。全体の完了状態はstatus.jsonを参照。実際の設定・ハッシュはmanifest.json。','']
    (root/'README.md').write_text('\n'.join(lines))


def main():
    p=parser();names=[a.dest for a in p._actions if a.dest!='help']
    p.add_argument('--depths',nargs='+',type=int,choices=sorted(DEPTH_WIDTHS),default=[4,8,16])
    p.add_argument('--thresholds',nargs='+',type=int,default=[1,4,8,16])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--report-dir',type=Path,default=Path('tdt_mnist/results/depth-100k-20260907'))
    p.add_argument('--resume',action='store_true')
    p.set_defaults(pool_shape=[9,10],hidden_sizes=DEPTH_WIDTHS[4],hidden_size=0,expected_params=100000,
        hidden_activation='relu',activation_precision='a32',block_size=16,measurements=64,threshold=8,
        steps=12000,train_size=10000,val_size=1000,data_seed=0,oracle_every=0,eval_every=500,
        layer_diagnostics=True,data_dir=Path('/tmp/tdt-mnist-data'),output_dir=Path('tdt_mnist/runs/depth-100k-20260907'))
    args=p.parse_args();validate(args,p)
    if args.workers<1 or any(len(v)!=len(set(v)) for v in (args.depths,args.thresholds,args.seeds)):
        p.error('positive workers and unique conditions required')
    if any(not 1<=v<=min(args.measurements,2**(args.counter_bits-1)-1) for v in args.thresholds): p.error('invalid thresholds')
    if (args.pool_shape!=[9,10] or args.expected_params!=100000 or args.hidden_size or args.activation_precision!='a32'
        or args.hidden_activation!='relu' or not args.layer_diagnostics or args.oracle_every or args.test_size or args.data_seed is None):
        p.error('require 100k, 9x10 input, ReLU/A32, layer diagnostics, fixed data split, no oracle, full test set')
    import torch,torchvision
    from torchvision.datasets import MNIST
    MNIST(args.data_dir,train=True,download=args.download);MNIST(args.data_dir,train=False,download=args.download)
    files=[Path(__file__).with_name(n).resolve() for n in ('train.py','activation_quantization.py','depth_diagnostics.py','sweep_depth.py')]
    manifest={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('workers','resume','hidden_sizes','threshold')}
    manifest.update(hidden_widths={d:DEPTH_WIDTHS[d] for d in args.depths},torch_version=str(torch.__version__),
        torchvision_version=str(torchvision.__version__),source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        data_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((args.data_dir/'MNIST/raw').glob('*-ubyte'))})
    manifest=json.loads(json.dumps(manifest));args.output_dir.mkdir(parents=True,exist_ok=True);args.report_dir.mkdir(parents=True,exist_ok=True)
    path=args.report_dir/'manifest.json'
    if args.resume:
        if not path.exists() or json.loads(path.read_text())!=manifest: p.error('resume requires matching settings, data and sources')
    elif any(args.output_dir.iterdir()) or any(args.report_dir.iterdir()): p.error('output directories must be empty')
    path.write_text(json.dumps(manifest,indent=2)+'\n');(args.report_dir/'sources').mkdir(exist_ok=True)
    for f in files: shutil.copy2(f,args.report_dir/'sources'/f.name)
    tasks=list(itertools.product(args.seeds,args.depths,args.thresholds))
    def run(task):
        seed,depth,threshold=task;label=f'depth{depth}-threshold{threshold}-seed{seed}';directory=args.output_dir/label
        summary=directory/'summary.json'
        if args.resume and summary.exists() and (directory/'model.pt').exists(): r=json.loads(summary.read_text())
        else:
            if directory.exists() and any(directory.iterdir()): raise RuntimeError(f'incomplete run: {directory}')
            options={n:getattr(args,n) for n in names}
            options.update(seed=seed,batch_seed=seed+100000,hidden_sizes=DEPTH_WIDTHS[depth],threshold=threshold,
                output_dir=directory.resolve(),data_dir=args.data_dir.resolve(),download=False)
            command=[sys.executable,str(files[0])]
            for name,value in options.items():
                if value is None: continue
                flag='--'+name.replace('_','-')
                if isinstance(value,bool): command.append(flag if value else '--no-'+name.replace('_','-'))
                elif isinstance(value,(list,tuple)): command.extend([flag,*map(str,value)])
                else: command.extend([flag,str(value)])
            with (args.output_dir/f'{label}.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
            r=json.loads(summary.read_text())
        config=json.loads((directory/'config.json').read_text())
        assert r['num_params']==100000 and len(config['shapes'])==depth
        assert r['train_forward_calls']==2*args.measurements*args.steps
        return {**r,'depth':depth,'threshold':threshold,'seed':seed,'shapes':config['shapes'],'run_directory':str(directory.resolve())}
    results=[];errors=[]
    def status():
        (args.report_dir/'status.json').write_text(json.dumps({'completed':len(results),'expected':len(tasks),
            'complete':len(results)==len(tasks) and not errors,'errors':errors},indent=2)+'\n')
    status()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run,t):t for t in tasks}
        for future in as_completed(futures):
            try:
                r=future.result();results.append(r);report(args.report_dir,results)
                print(f"[{len(results)}/{len(tasks)}] {futures[future]} val={r['final_validation']['accuracy']:.4f} fires={r['total_fires']}",flush=True)
            except Exception as error: errors.append(f'{futures[future]}: {error}');print(errors[-1],flush=True)
            status()
    if errors: raise SystemExit('\n'.join(errors))


if __name__=='__main__': main()
