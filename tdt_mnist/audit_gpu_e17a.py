"""Audit fresh GPU E17a runs without evaluating test again."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,csv,math,time
from pathlib import Path
import numpy as np
import torch
from run_gpu_e17a import ROOT,OLD,HERE,read,gens,init
from run_residual_e17 import sha,dump,write_csv,config,load_data,observe,probes
from residual_followup_models import ResidualTDT
from gpu_evaluation_engines import GPUEvaluator,schedule
from depth_diagnostics import layer_events

def main():
    assert json.loads((ROOT/'status.json').read_text())['training_complete'];init();os.sched_setaffinity(0,{15})
    started=time.perf_counter();manifest=json.loads((ROOT/'manifest.json').read_text());hash_checks=0
    for n,h in manifest['sources'].items():assert sha(HERE/n)==h and sha(ROOT/'sources'/n)==h;hash_checks+=2
    for n,h in manifest['data'].items():assert sha(HERE/'data/MNIST/raw'/n)==h;hash_checks+=1
    assert sha(OLD/'per_seed/results.csv')==manifest['cpu_reference_sha256']
    args=config(0,HERE/'data');(x,y),(vx,vy),_=load_data(args,torch.device('cpu'))
    audit_rows=[];all_firing=[];curves=[]
    for seed in range(3):
        out=ROOT/'per_seed'/f'seed{seed}'
        for n,h in json.loads((out/'manifest.json').read_text()).items():assert sha(out/n)==h;hash_checks+=1
        summary=json.loads((out/'summary.json').read_text());metrics=read(out/'metrics.csv');fires=read(out/'firing.csv');layers=iter(read(out/'layer_metrics.csv'))
        losses=np.load(out/'candidate_losses.npy',mmap_mode='r');ys=np.load(out/'abs_y.npy',mmap_mode='r');indices=np.load(out/'selected_indices.npy',mmap_mode='r')
        assert losses.shape==(12000,128) and ys.shape==(12000,64) and indices.shape==(12000,16)
        assert np.isfinite(losses).all() and np.array_equal(np.abs(losses[:,::2]-losses[:,1::2]),ys)
        assert len(metrics)==len(fires)==12000 and summary['test_evaluations']==1
        m=ResidualTDT(seed,8,76,'a8');g,bg=gens(seed);scale=.02;fire_total=0;totals=[dict(fires=0,selected_intervals=0,fire_intervals=0,selected_coordinates=0) for _ in range(18)]
        initial,sr,ar,rr=observe(m,vx,vy,0);assert initial==summary['initial_validation'];curves.append(dict(seed=seed,step=0,val_accuracy=initial['accuracy']))
        probes_initial=probes(m,x,y,seed,'initial')
        ev=GPUEvaluator(m,x,y,'gpu_graph');sampled=[];sample_steps={1,500,6000,12000}
        if summary['first_firing_divergence'] is not None:sample_steps.add(summary['first_firing_divergence'])
        for j,(row,fire) in enumerate(zip(metrics,fires)):
            step=j+1;assert int(row['step'])==int(fire['step'])==step and float(row['scale'])==scale
            if step in sample_steps:
                plan=schedule(m,x,args,g,bg);v,_=ev.evaluate(m,plan);assert np.array_equal(v.numpy(),losses[j]),(seed,step,'GPU replay')
                sampled.append(step)
            expected_indices=torch.randperm(m.num_params,generator=g)[:16];assert np.array_equal(expected_indices.numpy(),indices[j])
            # Preserve exact interleaving and shapes of original RNG calls.
            for _ in range(64):
                torch.randint(len(x),(128,),generator=bg)
                torch.randint(2,(16,),generator=g);torch.randint(2,(16,),generator=g);torch.rand((16,),generator=g)
            diffs=ys[j];assert int(row['zero_difference_count'])==int((diffs==0).sum())
            assert abs(float(row['abs_y_mean'])-float(diffs.astype('float64').mean()))<1e-12
            median=float(np.sort(diffs)[32]);scale=max(1e-5,.9*scale+.1*median)
            proposal=m.weights.clone();a=fire['action']
            if a:
                assert ';' not in a;coord,target=map(int,a.split(':'));assert coord in indices[j] and target in [-1,0,1] and abs(target-int(m.weights[coord]))==1
                proposal[coord]=target;fire_total+=1
            assert int(row['fires'])==int(bool(a))
            for event in layer_events(m,proposal,expected_indices):
                saved=next(layers);assert int(saved['step'])==step and all(int(saved[k])==v for k,v in event.items())
                t=totals[event['layer']]
                for dst,src in [('fires','fires'),('selected_intervals','selected_interval'),('fire_intervals','fire_interval'),('selected_coordinates','selected_coordinates')]:t[dst]+=event[src]
            m.weights.copy_(proposal)
            if step%500==0:curves.append(dict(seed=seed,step=step,val_accuracy=float(row['val_accuracy'])))
        try:next(layers);raise AssertionError('extra layer rows')
        except StopIteration:pass
        final_state=torch.load(out/'model.pt',weights_only=False);checkpoint=torch.load(out/'checkpoint.pt',weights_only=False)
        assert torch.equal(m.weights,final_state['weights']) and scale==final_state['scale']==checkpoint['scale']
        assert torch.equal(g.get_state(),checkpoint['generator']) and torch.equal(bg.get_state(),checkpoint['batch_generator'])
        assert totals==summary['layer_totals']
        final,sr,ar,rr=observe(m,vx,vy,12000);assert final==summary['final_validation']
        # Final diagnostics and initial/final CPU probes use historical E17 definitions.
        for filename,computed,keys in [('signal.csv',sr,['layer','stage']),('activation.csv',ar,['layer']),('rms_ratios.csv',rr,['block'])]:
            saved=[r for r in read(out/filename) if int(r['step'])==12000]
            assert len(saved)==len(computed)
            for c in computed:
                r=next(r for r in saved if all(str(c[k])==r[k] for k in keys))
                for k,v in c.items():
                    if isinstance(v,(float,int)):assert float(r[k])==float(v),(seed,filename,k)
                    elif isinstance(v,str):assert r[k]==v
        saved=read(out/'probes.csv');computed=probes_initial+probes(m,x,y,seed,'final');assert len(saved)==len(computed)
        for r,c in zip(saved,computed):
            for k,v in c.items():assert (float(r[k])==float(v) if isinstance(v,(int,float)) else r[k]==v),(seed,'probe',k)
        for layer,t in enumerate(totals):all_firing.append(dict(seed=seed,layer=layer,matrix=m.matrix_names[layer],**t,all_interval_rate=t['fires']/12000,selected_interval_rate=t['fires']/t['selected_intervals'] if t['selected_intervals'] else 0))
        audit_rows.append(dict(seed=seed,passed=True,intervals=12000,candidate_losses_checked=1536000,final_weights_reconstructed=True,rng_final_state_reproduced=True,scale_updates_reproduced=True,initial_final_validation_reproduced=True,final_signals_reproduced=True,probes_reproduced=True,gpu_loss_replay_steps=sampled,fire_total=fire_total,test_replayed=False))
        del ev;torch.cuda.empty_cache();dump(ROOT/'audit_progress.json',dict(completed=len(audit_rows),expected=3));print(f'GPU E17a seed{seed} audit passed',flush=True)
    write_csv(ROOT/'firing/matrices.csv',all_firing);write_csv(ROOT/'aggregate/validation_curves.csv',curves)
    stats=json.loads((ROOT/'report.json').read_text());per=read(ROOT/'per_seed/results.csv')
    assert abs(np.mean([float(r['gpu_trained_test_percent']) for r in per])-stats['gpu_test_mean_percent'])<1e-10
    # Simple scientific figure, generated only after all final results are fixed.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,3.8))
    for seed in range(3):
        rows=[r for r in curves if r['seed']==seed];ax[0].plot([r['step'] for r in rows],[100*r['val_accuracy'] for r in rows],label=f'GPU seed{seed}')
    ax[0].set(xlabel='Interval',ylabel='CPU-evaluated validation accuracy (%)');ax[0].legend()
    ax[1].bar(['CPU E17a','GPU-trained E17a'],[stats['cpu_test_mean_percent'],stats['gpu_test_mean_percent']],yerr=[stats['cpu_test_sample_std_percent'],stats['gpu_test_sample_std_percent']],capsize=5,color=['#65758b','#247f96'])
    ax[1].axhspan(90.337,90.937,color='orange',alpha=.15,label='Preregistered band');ax[1].set(ylim=(min(85,stats['gpu_test_mean_percent']-2),max(94,stats['gpu_test_mean_percent']+2)),ylabel='Final test accuracy (%)');ax[1].legend(fontsize=8)
    fig.tight_layout();(ROOT/'figures').mkdir(exist_ok=True)
    fig.savefig(ROOT/'figures/accuracy.png',dpi=160);fig.savefig(ROOT/'figures/accuracy.svg');plt.close(fig)
    (ROOT/'README.md').write_text((ROOT/'README.md').read_text()+'\n精度比較図：figures/accuracy.png（SVG併存）。独立監査で全区間の候補差・S・乱数系列・発火後重み・層イベント、初期/最終validation、最終診断、全プローブを照合。代表区間のGPU128損失もビット一致で再現し、testは再評価していない。\n')
    dump(ROOT/'audit.json',dict(passed=True,seeds=audit_rows,hash_checks=hash_checks,elapsed_seconds=time.perf_counter()-started,test_replayed=False))
    dump(ROOT/'status.json',dict(complete=True,training_complete=True,completed=3,expected=3,audited=True,accuracy_band_pass=stats['accuracy_band_pass']))
    dump(ROOT/'artifacts_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='artifacts_sha256.json'})
    print('Full GPU E17a audit complete',flush=True)
if __name__=='__main__':main()
