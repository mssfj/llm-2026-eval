"""FP32 backpropagation followed by frozen-weight layer-wise A3 inference."""
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import argparse,csv,hashlib,json,math,os,statistics,subprocess,sys,time,shutil
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from train import load_data
from activation_quantization import encode_activation,decode_activation,ActivationObserver
from depth_diagnostics import SignalObserver
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'tdt_mnist/results/backprop-a3-inference-16x79-20260907'
BASE=ROOT/'tdt_mnist/runs/depth-activation-100k-20260907/identity-a3-threshold/depth16-threshold8-seed0/config.json'

class FP32MLP(nn.Module):
    def __init__(self,width=79,depth=16,inputs=90):
        super().__init__();dims=[inputs]+[width]*(depth-1)+[10]
        self.layers=nn.ModuleList([nn.Linear(a,b,bias=False) for a,b in zip(dims,dims[1:])])
        for i,layer in enumerate(self.layers):
            nn.init.normal_(layer.weight,mean=0.,std=math.sqrt((2. if i<depth-1 else 1.)/layer.in_features))
    def forward(self,x,quantized=(),signals=None,activations=None):
        if self.training and quantized:raise ValueError('A3 is inference-only; no quantization-aware training')
        for i,layer in enumerate(self.layers):
            if signals is not None:signals.record(i,'pre_quantization',x)
            if i in quantized:
                q,b=encode_activation(x,'a3','mean_threshold',.5);decoded=decode_activation(q,b)
                if activations is not None:activations.record(i,x,q,decoded)
                x=decoded
            if signals is not None:signals.record(i,'input',x)
            x=layer(x)
            if i<len(self.layers)-1:x=F.relu(x)
            if signals is not None:signals.record(i,'output',x)
        return x

def modes():
    return [('fp32',())]+[(f'single_hidden_{i:02}',(i,)) for i in range(1,16)]+[(f'prefix_hidden_{i:02}',tuple(range(1,i+1))) for i in range(1,16)]+[('input_only',(0,)),('all_linear_inputs',tuple(range(16)))]
def digest(model):
    return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read(path):
    with path.open() as f:return list(csv.DictReader(f))
@torch.no_grad()
def evaluate(model,x,y,quantized=(),diagnostics=False):
    model.eval();sig=SignalObserver() if diagnostics else None;act=ActivationObserver(16,'a3') if diagnostics else None
    total=0.;pred=[]
    for start in range(0,len(x),1024):
        z=model(x[start:start+1024],quantized,sig,act)
        if not torch.isfinite(z).all():raise ValueError('nonfinite inference logits')
        total+=float(F.cross_entropy(z,y[start:start+1024],reduction='sum'));pred.append(z.argmax(1))
    pred=torch.cat(pred).numpy();stats={'loss':total/len(x),'accuracy':float(np.mean(pred==y.numpy()))}
    return stats,pred,sig.summary() if sig else [],act.summary() if act else []

def worker(seed,root,smoke=False):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
    run=root/f'seed{seed}';run.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(BASE.read_text());cfg.update(seed=seed,download=False)
    if smoke:cfg.update(train_size=128,val_size=64,test_size=64)
    args=SimpleNamespace(**cfg);args.data_dir=Path(args.data_dir)
    (x,y),(vx,vy),(tx,ty)=load_data(args,torch.device('cpu'))
    model=FP32MLP();assert sum(p.numel() for p in model.parameters())==95274
    before=digest(model);max_epochs=2 if smoke else 100
    config={'seed':seed,'hidden_activation':'relu','hidden_sizes':[79]*15,'depth_including_output':16,'input_width':90,
        'num_params':95274,'bias':False,'weights':'FP32 trainable','training_activations':'FP32','optimizer':'Adam',
        'learning_rate':.001,'batch_size':128,'max_epochs':max_epochs,'early_stopping_patience':20,'minimum_epochs':30,
        'selection':'minimum validation loss','lr_scheduler':'ReduceLROnPlateau validation loss, factor=.5, patience=5, min_lr=1e-5',
        'weight_decay':0.,'data_seed':cfg['data_seed'],'train_size':len(x),'val_size':len(vx),'test_size':len(tx),'pool_shape':cfg['pool_shape'],
        'preprocessing':'shared train.load_data: adaptive pooling 9x10, (image/255-.1307)/.3081',
        'a3_method':'mean_threshold','threshold_factor':.5,'quantization':'inference only, per-example per-layer threshold and reconstruction scale',
        'modes':{name:list(q) for name,q in modes()},'linear_input_indices':'0=image, 1=hidden1 output, ..., 15=hidden15 output; logits never quantized',
        'smoke':smoke,'torch_version':torch.__version__,'threads':1,'initial_weights_sha256':before}
    (run/'config.json').write_text(json.dumps(config,indent=2))
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=5,min_lr=1e-5)
    g=torch.Generator().manual_seed(100000+seed);start=time.monotonic();history=[];best=float('inf');best_epoch=0;stale=0;first_grad=[]
    initial=evaluate(model,vx,vy)[0]
    for epoch in range(1,max_epochs+1):
        model.train();order=torch.randperm(len(x),generator=g);total=0.;correct=0
        for i in range(0,len(x),128):
            ids=order[i:i+128];optimizer.zero_grad(set_to_none=True);z=model(x[ids]);loss=F.cross_entropy(z,y[ids])
            assert torch.isfinite(loss);loss.backward()
            if epoch==1 and i==0:
                first_grad=[float(layer.weight.grad.norm()) for layer in model.layers]
                assert all(math.isfinite(v) and v>0 for v in first_grad)
            optimizer.step();total+=float(loss.detach())*len(ids);correct+=int((z.detach().argmax(1)==y[ids]).sum())
        val=evaluate(model,vx,vy)[0];lr=optimizer.param_groups[0]['lr'];scheduler.step(val['loss'])
        improved=val['loss']<best
        if improved:
            best=val['loss'];best_epoch=epoch;stale=0
            torch.save({'state_dict':model.state_dict(),'config':config,'epoch':epoch,'validation':val},run/'best_model.pt')
        else:stale+=1
        history.append({'epoch':epoch,'train_loss':total/len(x),'train_accuracy':correct/len(x),'val_loss':val['loss'],'val_accuracy':val['accuracy'],
            'lr':lr,'selected_best':improved,'elapsed_seconds':time.monotonic()-start})
        write(run/'training.csv',history)
        print(json.dumps({'seed':seed,**history[-1]}),flush=True)
        if epoch>=30 and stale>=20:break
    checkpoint=torch.load(run/'best_model.pt',weights_only=False);model.load_state_dict(checkpoint['state_dict']);model.eval()
    learned=digest(model);assert learned!=before
    assert all(p.dtype==torch.float32 and torch.isfinite(p).all() for p in model.parameters())
    assert evaluate(model,vx,vy)[0]==checkpoint['validation']
    rows=[];signals=[];activation=[];predictions={};checks=[]
    for split,xx,yy in [('validation',vx,vy),('test',tx,ty)]:
        baseline_pred=None;baseline_accuracy=None
        for name,q in modes():
            stats,pred,sig,act=evaluate(model,xx,yy,q,True);assert digest(model)==learned
            if name=='fp32':baseline_pred=pred;baseline_accuracy=stats['accuracy']
            rows.append({'seed':seed,'split':split,'mode':name,'quantized_count':len(q),'quantized_linear_inputs':','.join(map(str,q)),
                **stats,'accuracy_delta_pp':100*(stats['accuracy']-baseline_accuracy),'prediction_disagreement_fraction':float(np.mean(pred!=baseline_pred)),
                'correct_to_wrong':int(np.sum((baseline_pred==yy.numpy())&(pred!=yy.numpy()))),
                'wrong_to_correct':int(np.sum((baseline_pred!=yy.numpy())&(pred==yy.numpy())))})
            predictions[split+'__'+name]=pred
            for r in sig:
                assert r['nonfinite_count']==0;signals.append({'seed':seed,'split':split,'mode':name,**r})
            assert len(sig)==48
            assert sorted(r['layer'] for r in act)==list(q)
            for r in act:
                codes=r.pop('code_histogram')
                if r['layer']>0:assert int(codes.get('-1',0))==0
                activation.append({'seed':seed,'split':split,'mode':name,**r,
                    **{'code_'+str(k)+'_fraction':int(codes.get(str(k),0))/r['values'] for k in [-1,0,1]}})
            print(json.dumps({'seed':seed,'evaluation':name,'split':split,**stats}),flush=True)
        assert np.array_equal(predictions[split+'__single_hidden_01'],predictions[split+'__prefix_hidden_01'])
        a=next(r for r in rows if r['split']==split and r['mode']=='single_hidden_01')
        b=next(r for r in rows if r['split']==split and r['mode']=='prefix_hidden_01');assert a['loss']==b['loss']
        predictions[split+'__labels']=yy.numpy()
        checks.append({'split':split,'modes':33,'examples':len(xx),'duplicate_mode_identical':True})
    write(run/'inference.csv',rows);write(run/'signal_metrics.csv',signals);write(run/'activation_metrics.csv',activation)
    np.savez_compressed(run/'predictions.npz',**predictions)
    summary={'seed':seed,'passed':True,'epochs':len(history),'best_epoch':best_epoch,'initial_validation':initial,'selected_validation':checkpoint['validation'],
        'num_params':95274,'weights_unchanged_during_inference':True,'weights_sha256':learned,'first_batch_gradient_norm_by_layer':first_grad,
        'elapsed_seconds':time.monotonic()-start,'checks':checks,'evaluation_rows':len(rows),'signal_rows':len(signals),
        'activation_rows':len(activation),'output_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in run.iterdir() if f.is_file()}}
    (run/'summary.json').write_text(json.dumps(summary,indent=2));return summary

def aggregate(root,seeds):
    rows=[];signals=[];activations=[];summaries=[]
    for seed in seeds:
        d=root/f'seed{seed}';s=json.loads((d/'summary.json').read_text());assert s['passed'];summaries.append(s)
        for name,sha in s['output_sha256'].items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==sha
        rows.extend(read(d/'inference.csv'));signals.extend(read(d/'signal_metrics.csv'));activations.extend(read(d/'activation_metrics.csv'))
        with np.load(d/'predictions.npz') as preds:
            for r in read(d/'inference.csv'):
                pred=preds[r['split']+'__'+r['mode']];labels=preds[r['split']+'__labels']
                assert float(r['accuracy'])==float(np.mean(pred==labels))
    def grouped(rs,keys,fields):
        groups={}
        for r in rs:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
        result=[]
        for key,group in groups.items():
            out=dict(zip(keys,key));out['seeds']=len(group)
            assert len(group)==len(seeds)
            for field in fields:
                vals=[float(r[field]) for r in group if r[field] not in ['',None]]
                out[field+'_mean']=statistics.mean(vals) if vals else None
                out[field+'_std']=statistics.stdev(vals) if len(vals)>1 else 0. if vals else None
            result.append(out)
        return result
    agg=grouped(rows,['split','mode'],['accuracy','loss','accuracy_delta_pp','prediction_disagreement_fraction'])
    sa=grouped(signals,['split','mode','layer','stage'],['rms','zero_fraction','mean','std'])
    aa=grouped(activations,['split','mode','layer'],['mse','relative_squared_error','cosine_mean_valid','code_-1_fraction','code_0_fraction','code_1_fraction'])
    for name,data in [('per_seed.csv',rows),('aggregate.csv',agg),('signal_metrics.csv',signals),('signal_aggregate.csv',sa),('activation_metrics.csv',activations),('activation_aggregate.csv',aa)]:write(root/name,data)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    baseline=next(r for r in agg if r['split']=='test' and r['mode']=='fp32')
    for ax,prefix,title in zip(axes[:2],['single_hidden_','prefix_hidden_'],['One hidden activation quantized','Hidden activations quantized cumulatively']):
        rs=[next(r for r in agg if r['split']=='test' and r['mode']==prefix+f'{i:02}') for i in range(1,16)]
        ax.errorbar(range(1,16),[r['accuracy_mean']*100 for r in rs],yerr=[r['accuracy_std']*100 for r in rs],marker='o',markersize=3)
        ax.axhline(baseline['accuracy_mean']*100,color='gray',ls='--',label='FP32');ax.set(title=title,xlabel='Hidden layer',ylabel='Test accuracy (%)');ax.legend();ax.grid(alpha=.2)
    for mode,label in [('fp32','FP32'),('prefix_hidden_15','All hidden A3'),('all_linear_inputs','Input + hidden A3')]:
        rs=sorted([r for r in sa if r['split']=='test' and r['mode']==mode and r['stage']=='input'],key=lambda r:int(r['layer']))
        axes[2].plot(range(1,17),[max(r['rms_mean'],1e-30) for r in rs],label=label)
    axes[2].set(yscale='log',xlabel='Linear layer',ylabel='RMS(actual linear input)',title='Signal propagation');axes[2].legend();axes[2].grid(alpha=.2)
    for ext in ['png','svg']:fig.savefig(root/('comparison.'+ext),dpi=170)
    plt.close(fig)
    lines=['# FP32 backpropagation → 推論時のみ層別A3','',
        'ReLUあり、隠れ幅79固定×15層＋10クラス出力＝16線形層。入力90、バイアスなし、学習可能なFP32重み95,274個。',
        '既存TDTの10万重みモデルは隠れ幅79〜82であり、この幅79固定モデルと厳密に同一の層幅ではない。',
        'データ分割・前処理は既存TDTと共通（通常run：train10,000 / validation1,000 / test10,000、data_seed0、9×10 pooling）。',
        'Adam lr=.001、batch128、最大100epoch。validation損失に基づき学習率を下げ、30epoch以降20epoch改善がなければ終了。最小validation損失のモデルを選択。testはモデル選択に使わない。',
        '初期化は隠れ層std=sqrt(2/fan_in)、出力std=sqrt(1/fan_in)の正規分布。残差・正規化層なし。学習中は重み・活性・演算ともFP32、A3化は保存モデルの推論だけ。',
        'A3は既存の閾値・復元スケール分離型：τ=.5 mean(abs(x))、q=sign(x) 1[abs(x)>τ]、β=mean(abs(x)|selected)、復元βq。各例・各層で再計算。ReLU後は符号コード0,+1のみ。',
        '単独量子化15通り、浅い層から累積量子化15通り、FP32、入力画像のみ、入力＋全隠れ層の計33設定を同じ重みで評価。logitsはFP32のまま。累積15が全隠れ層A3に相当する。','',
        '## 結果','', '| 推論条件 | validation精度 % | test精度 % | test差 pp |','| --- | ---: | ---: | ---: |']
    for mode,label in [('fp32','FP32'),('input_only','入力のみA3'),('prefix_hidden_15','全隠れ層A3'),('all_linear_inputs','入力＋全隠れ層A3')]:
        v=next(r for r in agg if r['split']=='validation' and r['mode']==mode);t=next(r for r in agg if r['split']=='test' and r['mode']==mode)
        lines.append(f"| {label} | {100*v['accuracy_mean']:.3f} ± {100*v['accuracy_std']:.3f} | {100*t['accuracy_mean']:.3f} ± {100*t['accuracy_std']:.3f} | {t['accuracy_delta_pp_mean']:+.3f} |")
    lines+=['','平均±seed間標本標準偏差。各seedのFP32推論からの差を対応付けて集計。','',
        'aggregate.csvに全33設定、per_seed.csvに各seed・データ集合の精度、損失、予測変更率、正解→誤り・誤り→正解の件数。',
        'signal_metrics.csvは全設定・全16層のRMS。outputはReLU後（最終層はlogits）、inputは量子化・復元後に線形層へ入る値、pre_quantizationはその直前。',
        'activation_metrics.csvは量子化した層のMSE、相対二乗誤差、コード分布、コサイン類似度。layerは0始まりの線形層入力番号で、1は第1隠れ層の出力を表す。',
        '各seedのbest_model.pt、training.csv、predictions.npzでモデル選択と全予測を追跡できる。全推論で重みハッシュ不変、保存モデルのvalidation再現、重複設定の完全一致を検証。',
        'これはFP32学習済み表現の量子化感度を測る対照である。TDTとの比較では重み精度、最適化法、学習予算、厳密な層幅が異なり、精度差を単一要因に帰属しない。','']
    (root/'README.md').write_text('\n'.join(lines))
    (root/'verification.json').write_text(json.dumps({'passed':True,'seeds':seeds,'modes_per_seed_per_split':33,'summaries':summaries},indent=2))

def main(args):
    if args.worker is not None:worker(args.worker,args.output,args.smoke);return
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'sources').mkdir()
    hashes={}
    for name in ['backprop_a3_inference.py','activation_quantization.py','depth_diagnostics.py','train.py']:
        src=ROOT/'tdt_mnist'/name;shutil.copy2(src,args.output/'sources'/name);hashes[name]=hashlib.sha256(src.read_bytes()).hexdigest()
    data=Path(json.loads(BASE.read_text())['data_dir'])/'MNIST/raw'
    manifest={'sources':hashes,'baseline_config_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),
        'data_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in data.glob('*ubyte')},'smoke':args.smoke,'seeds':args.seeds}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def launch(seed):
        cmd=[sys.executable,__file__,'--worker',str(seed),'--output',str(args.output)]
        if args.smoke:cmd.append('--smoke')
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (args.output/f'seed{seed}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
        return seed
    completed=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for seed in pool.map(launch,args.seeds):
            completed.append(seed);(args.output/'status.json').write_text(json.dumps({'complete':False,'completed_seeds':completed}))
    aggregate(args.output,args.seeds)
    for name,sha in hashes.items():assert hashlib.sha256((args.output/'sources'/name).read_bytes()).hexdigest()==sha
    (args.output/'status.json').write_text(json.dumps({'complete':True,'completed_seeds':completed,'verified':True}))
    print('FP32 training and all layer-wise A3 inference checks completed.',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--smoke',action='store_true');p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);main(p.parse_args())
