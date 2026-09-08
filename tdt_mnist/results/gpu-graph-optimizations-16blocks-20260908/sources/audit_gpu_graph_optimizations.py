"""Independent CPU replay of every recorded loss/decision and frozen artifact audit."""
import ast,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import train
from allocation_engines import private_function
from benchmark_allocation_engines import make_model,state_generators
from benchmark_gpu_engines import action
from run_residual_e17 import setup,config,load_data,dump,sha
from benchmark_gpu_graph_optimizations import ROOT,HERE,read

def audit():
    started=time.perf_counter();setup();cfg=config(0,HERE/'data');(x,y),_,_=load_data(cfg,torch.device('cpu'))
    manifest=json.loads((ROOT/'manifest.json').read_text());checked=0
    for name,digest in manifest['sources'].items():
        assert sha(ROOT/'sources'/name)==digest and sha(HERE/name)==digest,name;checked+=1
    old=HERE/'results/residual-followups-e18-e20-20260908'
    for seed,digest in manifest['trained_models'].items():assert sha(old/'per_seed'/f'E18a-seed{seed}'/'model.pt')==digest;checked+=1
    for name,digest in manifest['data'].items():assert sha(HERE/'data/MNIST/raw'/name)==digest;checked+=1
    audits=[];diagnostic_differences=[]
    for directory in sorted((ROOT/'benchmarks').iterdir()):
        summary=json.loads((directory/'summary.json').read_text());seed=summary['seed'];m=make_model(seed,'trained');g,bg=state_generators(seed);scale=.02
        intervals=read(directory/'intervals.csv');metrics=read(directory/'metrics.csv');losses=np.load(directory/'candidate_losses.npy');abs_y=np.load(directory/'abs_y.npy');indices=np.load(directory/'selected_indices.npy')
        assert len(intervals)==len(metrics)==100 and losses.shape==(100,128)
        assert np.array_equal(abs_y,np.abs(losses[:,::2]-losses[:,1::2]))
        for step in range(100):
            cursor=0;values=torch.from_numpy(losses[step].copy())
            def loss(model,bx,by,weights=None):
                nonlocal cursor
                result=values[cursor];cursor+=1;return result
            proposal,selected,stats,new_scale=private_function(train.epoch,{'loss':loss})(m,x,y,cfg,g,scale,bg)
            assert cursor==128 and np.array_equal(selected.numpy(),indices[step])
            assert action(m.weights,proposal)==intervals[step]['action'] and new_scale==float(intervals[step]['scale'])
            for k,v in stats.items():
                saved=ast.literal_eval(metrics[step][k])
                if v!=saved:
                    assert k in ['counter_all_mean','counter_all_abs_mean'],(directory,step,k,v,saved)
                    diagnostic_differences.append(dict(engine=summary['engine'],seed=seed,step=step+1,field=k,reference=v,actual=saved))
            m.weights.copy_(proposal);scale=new_scale
        final=torch.load(directory/'final.pt',map_location='cpu',weights_only=True)
        assert torch.equal(m.weights,final['weights']) and scale==final['scale']
        assert torch.equal(g.get_state(),final['generator']) and torch.equal(bg.get_state(),final['batch_generator'])
        assert hashlib.sha256(m.weights.numpy().tobytes()).hexdigest()==summary['final_weights_sha256']
        audits.append(dict(seed=seed,engine=summary['engine'],intervals=100,losses=12800,final_weights_reconstructed=True,scale_rng_actions_reproduced=True))
        print('audit passed',directory.name,flush=True)
    status=json.loads((ROOT/'status.json').read_text());assert len(audits)==status['completed']
    report=dict(passed=True,source_model_data_hash_checks=checked,runs=audits,total_intervals=len(audits)*100,total_candidate_losses=len(audits)*12800,diagnostic_only_difference_count=len(diagnostic_differences),test_evaluated=False,elapsed_seconds=time.perf_counter()-started)
    if diagnostic_differences:dump(ROOT/'audit_diagnostic_differences.json',diagnostic_differences)
    dump(ROOT/'audit.json',report);status['audited']=True;dump(ROOT/'status.json',status)
    with (ROOT/'README.md').open('a') as f:f.write(f'\n\n## 独立監査\n\n全{len(audits)}測定・{len(audits)*100}区間について、保存した128候補損失を元のCPU epochへ渡し、発火・S・両乱数状態・最終重みを再構成して一致を確認。ソース/モデル/データ{checked}ハッシュを照合。診断集計だけの差は{len(diagnostic_differences)}件。test再評価なし。\n')
    dump(ROOT/'artifacts_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='artifacts_sha256.json'})
    print(json.dumps(report),flush=True)
if __name__=='__main__':audit()
