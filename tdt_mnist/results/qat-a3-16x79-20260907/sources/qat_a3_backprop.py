"""A3 QAT from the same FP32 initialization, with exact forward and identity STE."""
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,math,os,shutil,statistics,subprocess,sys,time
import numpy as np
import torch
import torch.nn.functional as F
from backprop_a3_inference import FP32MLP,BASE,ROOT,digest,evaluate,write,read
from train import load_data
from activation_quantization import encode_activation,decode_activation
ALL=tuple(range(16))
DEFAULT=ROOT/'tdt_mnist/results/qat-a3-16x79-20260907'
CONTROL=ROOT/'tdt_mnist/results/backprop-a3-inference-16x79-20260907'

class A3STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        q,b=encode_activation(x,'a3','mean_threshold',.5)
        return decode_activation(q,b)
    @staticmethod
    def backward(ctx,gradient):
        return gradient

class QATMLP(FP32MLP):
    def forward(self,x,quantized=ALL,signals=None,activations=None):
        for i,layer in enumerate(self.layers):
            if signals is not None:signals.record(i,'pre_quantization',x)
            if i in quantized:
                decoded=A3STE.apply(x)
                if activations is not None:
                    q,_=encode_activation(x,'a3','mean_threshold',.5);activations.record(i,x,q,decoded)
                x=decoded
            if signals is not None:signals.record(i,'input',x)
            x=layer(x)
            if i<len(self.layers)-1:x=F.relu(x)
            if signals is not None:signals.record(i,'output',x)
        return x

def worker(seed,root,smoke=False):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
    d=root/f'seed{seed}';d.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(BASE.read_text());cfg.update(seed=seed,download=False)
    if smoke:cfg.update(train_size=128,val_size=64,test_size=64)
    args=SimpleNamespace(**cfg);args.data_dir=Path(args.data_dir)
    (x,y),(vx,vy),(tx,ty)=load_data(args,torch.device('cpu'))
    model=QATMLP();initial_hash=digest(model)
    ctrl=json.loads((CONTROL/f'seed{seed}/config.json').read_text())
    assert initial_hash==ctrl['initial_weights_sha256']
    assert sum(p.numel() for p in model.parameters())==95274
    max_epochs=2 if smoke else 100
    config={**ctrl,'max_epochs':max_epochs,'smoke':smoke,'train_size':len(x),'val_size':len(vx),'test_size':len(tx),
        'training_activations':'A3 quantized/dequantized before every linear layer',
        'initialization':'same random initialization as paired FP32 control, not pretrained',
        'ste':'exact quantized/dequantized forward, identity backward dQ/dx=1; no derivative through threshold or scale',
        'quantized_linear_inputs':list(ALL),'selection':'minimum validation loss with all linear inputs A3',
        'modes':{'qat_a3':list(ALL),'fp32_ablation':[]}}
    (d/'config.json').write_text(json.dumps(config,indent=2))
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=5,min_lr=1e-5)
    g=torch.Generator().manual_seed(100000+seed);start=time.monotonic();history=[];best=float('inf');best_epoch=0;stale=0
    initial=evaluate(model,vx,vy,ALL)[0];gradient_rows=[]
    for epoch in range(1,max_epochs+1):
        model.train();order=torch.randperm(len(x),generator=g);total=0.;correct=0;grad_sums=np.zeros(16);batches=0
        for i in range(0,len(x),128):
            ids=order[i:i+128];optimizer.zero_grad(set_to_none=True);z=model(x[ids]);loss=F.cross_entropy(z,y[ids]);assert torch.isfinite(loss)
            loss.backward();norms=[float(layer.weight.grad.norm()) for layer in model.layers]
            assert all(math.isfinite(v) for v in norms)
            grad_sums+=norms;batches+=1;optimizer.step()
            total+=float(loss.detach())*len(ids);correct+=int((z.detach().argmax(1)==y[ids]).sum())
        val=evaluate(model,vx,vy,ALL)[0];lr=optimizer.param_groups[0]['lr'];scheduler.step(val['loss'])
        if val['loss']<best:
            best=val['loss'];best_epoch=epoch;stale=0
            torch.save({'state_dict':model.state_dict(),'config':config,'epoch':epoch,'validation':val},d/'best_model.pt')
        else:stale+=1
        history.append({'epoch':epoch,'train_loss':total/len(x),'train_accuracy':correct/len(x),'val_loss':val['loss'],'val_accuracy':val['accuracy'],
            'lr':lr,'best_epoch':best_epoch,'elapsed_seconds':time.monotonic()-start})
        gradient_rows.extend({'epoch':epoch,'layer':i,'mean_gradient_norm':v/batches} for i,v in enumerate(grad_sums))
        write(d/'training.csv',history);write(d/'gradient_metrics.csv',gradient_rows)
        print(json.dumps({'seed':seed,**history[-1]}),flush=True)
        if epoch>=30 and stale>=20:break
    ckpt=torch.load(d/'best_model.pt',weights_only=False);model.load_state_dict(ckpt['state_dict']);model.eval();trained_hash=digest(model)
    assert trained_hash!=initial_hash
    assert evaluate(model,vx,vy,ALL)[0]==ckpt['validation']
    assert all(p.dtype==torch.float32 and torch.isfinite(p).all() for p in model.parameters())
    rows=[];signals=[];activations=[];predictions={}
    for split,xx,yy in [('validation',vx,vy),('test',tx,ty)]:
        predictions[split+'__labels']=yy.numpy()
        for mode,q in [('qat_a3',ALL),('fp32_ablation',())]:
            stats,pred,sig,act=evaluate(model,xx,yy,q,True)
            assert digest(model)==trained_hash and all(r['nonfinite_count']==0 for r in sig)
            rows.append({'seed':seed,'split':split,'mode':mode,**stats});signals.extend({'seed':seed,'split':split,'mode':mode,**r} for r in sig)
            predictions[split+'__'+mode]=pred
            assert [r['layer'] for r in act]==list(q)
            for r in act:
                codes=r.pop('code_histogram')
                if r['layer']>0:assert int(codes.get('-1',0))==0
                activations.append({'seed':seed,'split':split,'mode':mode,**r,
                    **{'code_'+str(k)+'_fraction':int(codes.get(str(k),0))/r['values'] for k in [-1,0,1]}})
    # Independent reference uses the original post-training quantizer without STE.
    reference=FP32MLP().eval();reference.load_state_dict(model.state_dict())
    with torch.no_grad():assert torch.equal(model(vx[:64]),reference(vx[:64],ALL))
    write(d/'inference.csv',rows);write(d/'signal_metrics.csv',signals);write(d/'activation_metrics.csv',activations)
    np.savez_compressed(d/'predictions.npz',**predictions)
    summary={'seed':seed,'passed':True,'epochs':len(history),'best_epoch':best_epoch,'initial_validation':initial,
        'selected_validation':ckpt['validation'],'num_params':95274,'initialization_matches_fp32_control':True,
        'exact_a3_forward_matches_reference':True,'weights_unchanged_during_inference':True,'weights_sha256':trained_hash,
        'elapsed_seconds':time.monotonic()-start,'output_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in d.iterdir() if f.is_file()}}
    (d/'summary.json').write_text(json.dumps(summary,indent=2));return summary

def aggregate(root,seeds):
    rows=[];signals=[];acts=[];summaries=[];comparison=[]
    controls=read(CONTROL/'per_seed.csv')
    for seed in seeds:
        d=root/f'seed{seed}';s=json.loads((d/'summary.json').read_text());assert s['passed'];summaries.append(s)
        for name,sha in s['output_sha256'].items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==sha
        rr=read(d/'inference.csv');rows.extend(rr);signals.extend(read(d/'signal_metrics.csv'));acts.extend(read(d/'activation_metrics.csv'))
        with np.load(d/'predictions.npz') as preds:
            for r in rr:assert float(r['accuracy'])==np.mean(preds[r['split']+'__'+r['mode']]==preds[r['split']+'__labels'])
        for split in ['validation','test']:
            for method,source,mode in [('FP32',controls,'fp32'),('PTQ_A3',controls,'all_linear_inputs'),('QAT_A3',rr,'qat_a3')]:
                r=next(r for r in source if int(r['seed'])==seed and r['split']==split and r['mode']==mode)
                comparison.append({'seed':seed,'split':split,'method':method,'accuracy':float(r['accuracy']),'loss':float(r['loss'])})
    agg=[];paired=[]
    for split in ['validation','test']:
        for method in ['FP32','PTQ_A3','QAT_A3']:
            group=[r for r in comparison if r['split']==split and r['method']==method]
            agg.append({'split':split,'method':method,'seeds':len(group),**{f+'_'+kind:(statistics.mean([r[f] for r in group]) if kind=='mean' else statistics.stdev([r[f] for r in group]) if len(group)>1 else 0.) for f in ['accuracy','loss'] for kind in ['mean','std']}})
        for seed in seeds:
            m={r['method']:r for r in comparison if r['split']==split and r['seed']==seed}
            paired.append({'seed':seed,'split':split,'qat_minus_ptq_pp':100*(m['QAT_A3']['accuracy']-m['PTQ_A3']['accuracy']),
                'qat_minus_fp32_pp':100*(m['QAT_A3']['accuracy']-m['FP32']['accuracy'])})
    for name,data in [('per_seed.csv',rows),('comparison_per_seed.csv',comparison),('aggregate.csv',agg),('paired_effects.csv',paired),('signal_metrics.csv',signals),('activation_metrics.csv',acts)]:write(root/name,data)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    ts=[r for r in agg if r['split']=='test'];axes[0].bar([r['method'] for r in ts],[r['accuracy_mean']*100 for r in ts],yerr=[r['accuracy_std']*100 for r in ts]);axes[0].set_ylabel('Test accuracy (%)')
    bp=read(CONTROL/'signal_metrics.csv')
    for source,mode,label in [(bp,'fp32','FP32'),(bp,'all_linear_inputs','PTQ A3'),(signals,'qat_a3','QAT A3')]:
        ys=[statistics.mean(float(r['rms']) for r in source if int(r['seed']) in seeds and r['split']=='test' and r['mode']==mode and r['stage']=='input' and int(r['layer'])==i) for i in range(16)]
        axes[1].plot(range(1,17),ys,label=label)
    axes[1].set(yscale='log',xlabel='Linear layer',ylabel='RMS(actual linear input)');axes[1].legend();axes[1].grid(alpha=.2)
    for ext in ['png','svg']:fig.savefig(root/('comparison.'+ext),dpi=170)
    plt.close(fig)
    lines=['# 16層・幅79：A3 STE付きQAT','',
        'ReLUあり、入力90、隠れ幅79×15層＋10クラス出力、バイアスなし、95,274個のFP32重み。',
        '通常FP32対照と同じseedの乱数初期重みから学習し、初期重みハッシュの完全一致を確認。FP32学習済みモデルからの微調整ではない。',
        '入力・全隠れ層を線形層直前でA3化する。閾値τ=.5 mean(abs(x))、復元値β=mean(abs(x)|selected)。各例・各層で動的計算。ReLU後のコードは0,+1。',
        '順伝播は既存量子化器と完全一致するカスタムautograd関数。逆伝播だけdQ/dx=1とする恒等STEで、閾値・復元スケール経由の微分は行わない。ReLUと線形層の逆伝播は通常どおり。',
        '重み・スケール・積和・logitsはFP32。重みを三値化したTDTとは異なる。',
        'データ・前処理・初期化・Adam・学習率スケジュール・最大100epoch・早期終了規則はFP32対照と共通。validation損失最小のモデルを選ぶ。QATでは選択時もA3。testはモデル選択に使わない。','',
        '| 方法 | validation精度 % | test精度 % |','| --- | ---: | ---: |']
    for method in ['FP32','PTQ_A3','QAT_A3']:
        v=next(r for r in agg if r['method']==method and r['split']=='validation');t=next(r for r in agg if r['method']==method and r['split']=='test')
        lines.append(f"| {method} | {100*v['accuracy_mean']:.3f} ± {100*v['accuracy_std']:.3f} | {100*t['accuracy_mean']:.3f} ± {100*t['accuracy_std']:.3f} |")
    lines+=['','平均±seed間標本標準偏差。PTQ_A3は既存FP32学習後に入力・全隠れ層をA3化した結果。QAT_A3も同じ量子化位置・閾値・復元規則。',
        'paired_effects.csvに各seedのQAT−PTQ、QAT−FP32の差。signal_metrics.csvに全16層のRMS、activation_metrics.csvに量子化誤差・コード分布・コサイン。各seedのgradient_metrics.csvにepochごとの層別平均勾配ノルム。',
        'fp32_ablationはQAT重みの量子化を外した診断であり、通常FP32学習対照とは異なる。',
        '保存モデルのvalidation再現、推論中の重み不変、既存量子化器との順伝播一致、全予測からの精度再集計を検証。','']
    (root/'README.md').write_text('\n'.join(lines));(root/'verification.json').write_text(json.dumps({'passed':True,'seeds':seeds,'summaries':summaries},indent=2))

def main(args):
    if args.worker is not None:worker(args.worker,args.output,args.smoke);return
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'sources').mkdir();hashes={}
    for name in ['qat_a3_backprop.py','backprop_a3_inference.py','activation_quantization.py','depth_diagnostics.py','train.py']:
        src=ROOT/'tdt_mnist'/name;shutil.copy2(src,args.output/'sources'/name);hashes[name]=hashlib.sha256(src.read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps({'sources':hashes,'seeds':args.seeds,'smoke':args.smoke,
        'control_manifest_sha256':hashlib.sha256((CONTROL/'manifest.json').read_bytes()).hexdigest()},indent=2))
    (args.output/'status.json').write_text(json.dumps({'complete':False,'expected_seeds':args.seeds}))
    def launch(seed):
        cmd=[sys.executable,__file__,'--worker',str(seed),'--output',str(args.output)]
        if args.smoke:cmd.append('--smoke')
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (args.output/f'seed{seed}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
        return seed
    with ThreadPoolExecutor(max_workers=3) as pool:completed=list(pool.map(launch,args.seeds))
    aggregate(args.output,args.seeds)
    (args.output/'status.json').write_text(json.dumps({'complete':True,'completed_seeds':completed,'verified':True}))
    print('QAT training, inference and paired comparison verified.',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--smoke',action='store_true');p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);main(p.parse_args())
