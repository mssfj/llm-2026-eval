"""Audit frozen depth-budget endpoints, then evaluate all predetermined tests."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import csv,json,math,time,statistics,hashlib
from pathlib import Path
import numpy as np
import torch
from run_depth_budget import ROOT,HERE,BUDGETS,ORDER,init,model,gens,outdir,read,check_local
from run_residual_e17 import config,load_data,evaluate,dump,write_csv,sha

def audit():
    init();frozen=json.loads((ROOT/'all_training_frozen.json').read_text())['endpoints'];assert len(frozen)==27
    checks=0
    for p,h in frozen.items():assert sha(ROOT/p)==h;checks+=1
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for p,h in manifest['sources'].items():assert sha(ROOT/'sources'/p)==h and sha(HERE/p)==h;checks+=1
    for p,h in manifest['data'].items():assert sha(HERE/'data/MNIST/raw'/p)==h;checks+=1
    a=config(0,HERE/'data');(x,y),(vx,vy),_=load_data(a,torch.device('cpu'));reports=[]
    for b,seed in ORDER:
        out=outdir(b,seed);m=model(seed,b);w=m.weights.numpy().copy();updates=np.zeros(m.num_params,np.int32);selections=np.zeros(m.num_params,np.int32);s=.02;fires=0
        losses=np.load(out/'candidate_losses.npy',mmap_mode='r');abs_y=np.load(out/'abs_y.npy',mmap_mode='r');selected=np.load(out/'selected_indices.npy',mmap_mode='r');events=np.load(out/'layer_events.npy',mmap_mode='r')
        assert losses.shape==(max(BUDGETS[b]),128) and np.isfinite(losses).all()
        assert np.array_equal(np.abs(losses[:,::2]-losses[:,1::2]),abs_y)
        medians=np.partition(np.asarray(abs_y),32,axis=1)[:,32]
        ends=np.cumsum([math.prod(z) for z in m.shapes]);n=0
        with (out/'metrics.csv').open() as f,(out/'firing.csv').open() as af:
            for metrics,action in zip(csv.DictReader(f),csv.DictReader(af),strict=True):
                n+=1;assert int(metrics['step'])==int(action['step'])==n;assert float(metrics['scale'])==s
                s=max(1e-5,.9*s+.1*float(medians[n-1]));assert float(metrics['next_scale'])==s
                idx=selected[n-1];assert len(np.unique(idx))==16
                selections[idx]+=1
                layer=np.bincount(np.searchsorted(ends,idx,side='right'),minlength=len(m.shapes))
                assert np.array_equal(events[n-1,:,0],layer) and np.array_equal(events[n-1,:,1],layer>0)
                c=int(action['coordinate']);target=int(action['target'])
                assert int(metrics['fires'])==int(c>=0)
                if c>=0:
                    assert c in idx and abs(int(w[c])-target)==1 and target in [-1,0,1]
                    w[c]=target;updates[c]+=1;fires+=1
                    assert events[n-1,:,2].sum()==1 and events[n-1,np.searchsorted(ends,c,side='right'),2]==1
                else:assert events[n-1,:,2].sum()==0
                assert int(metrics['cumulative_fires'])==fires
                if n in BUDGETS[b]:
                    ep=out/'budgets'/str(n);ck=torch.load(ep/'model.pt',map_location='cpu',weights_only=True)
                    assert np.array_equal(w,ck['weights'].numpy()) and ck['scale']==s and ck['total_fires']==fires
                    assert np.array_equal(updates,ck['updates'].numpy()) and np.array_equal(selections,ck['selections'].numpy())
                    summary=json.loads((ep/'training_summary.json').read_text());assert summary['model_sha256']==sha(ep/'model.pt')
                    assert summary['total_fires']==fires and summary['unique_updated_coordinates']==np.count_nonzero(updates)
                    mm=model(seed,b);mm.weights.copy_(ck['weights']);v=evaluate(mm,vx,vy);assert v==summary['validation']
                    g=torch.Generator().set_state(ck['generator']);bg=torch.Generator().set_state(ck['batch_generator'])
                    local=check_local(mm,x,y,a,g,bg,s)
                    assert len(read(ep/'probes.csv'))==len(m.shapes)*64
                    reports.append(dict(blocks=b,seed=seed,steps=n,weights_from_actions=True,scale_from_all_losses=True,coordinate_counts=True,validation_equal=True,checkpoint_local_legacy_graph_checks=local))
        assert n==max(BUDGETS[b])
        assert len(read(out/'validation.csv'))==1+n//500
        print(f'audit blocks{b} seed{seed} all{n} intervals passed',flush=True)
    report=dict(passed=True,hash_checks=checks,endpoint_checks=reports,training_intervals=432000,test_evaluated=False,rng_scope='Saved states used for three local originalGraph/compact comparisons per endpoint; not full from-initial RNG replay.')
    dump(ROOT/'audit.json',report);return frozen

def final_tests(frozen):
    init();a=config(0,HERE/'data');_,_,(tx,ty)=load_data(a,torch.device('cpu'));rows=[]
    for b,seed in ORDER:
        for steps in BUDGETS[b]:
            ep=outdir(b,seed)/'budgets'/str(steps);assert sha(ep/'model.pt')==frozen[str((ep/'model.pt').relative_to(ROOT))]
            dest=ep/'final_test.json'
            if dest.exists():test=json.loads(dest.read_text())
            else:
                m=model(seed,b);m.weights.copy_(torch.load(ep/'model.pt',map_location='cpu',weights_only=True)['weights'])
                result=evaluate(m,tx,ty);test=dict(**result,model_sha256=sha(ep/'model.pt'),test_calls=1,evaluation_device='cpu');dump(dest,test)
            assert test['model_sha256']==sha(ep/'model.pt') and test['test_calls']==1
            summary=json.loads((ep/'training_summary.json').read_text())
            rows.append(dict(blocks=b,seed=seed,steps=steps,num_params=summary['num_params'],test_percent=test['accuracy']*100,test_loss=test['loss'],validation_percent=summary['validation']['accuracy']*100,total_fires=summary['total_fires'],fire_fraction=summary['fire_fraction'],fires_per_parameter=summary['fires_per_parameter'],intervals_per_parameter=summary['intervals_per_parameter'],unique_updated_fraction=summary['unique_updated_fraction'],branch_ratio_exceed=summary['branch_ratio_exceed'],branch_ratio_total=b,logits_rms=summary['logits_rms'],engine_seconds=summary['engine_seconds'],elapsed_seconds=summary['elapsed_seconds']))
    assert len(rows)==27;write_csv(ROOT/'per_seed/results.csv',rows);return rows

def analyze(rows):
    agg=[]
    for b in BUDGETS:
        for steps in BUDGETS[b]:
            rs=[r for r in rows if r['blocks']==b and r['steps']==steps]
            a=dict(blocks=b,steps=steps,num_params=rs[0]['num_params'],seeds=3)
            for field in ['test_percent','validation_percent','total_fires','fire_fraction','fires_per_parameter','unique_updated_fraction','logits_rms','engine_seconds','elapsed_seconds']:
                values=[r[field] for r in rs];a[field+'_mean']=statistics.mean(values);a[field+'_sample_sd']=statistics.stdev(values)
            a['branch_ratio_exceed_count']=sum(r['branch_ratio_exceed'] for r in rs);a['branch_ratio_count']=3*b
            agg.append(a)
    write_csv(ROOT/'aggregate/results.csv',agg)
    comparisons=[]
    def paired(label,left,right):
        dif=[]
        for seed in range(3):
            a=next(r for r in rows if (r['blocks'],r['steps'],r['seed'])==(*left,seed));b=next(r for r in rows if (r['blocks'],r['steps'],r['seed'])==(*right,seed));dif.append(a['test_percent']-b['test_percent'])
        comparisons.append(dict(comparison=label,left_blocks=left[0],left_steps=left[1],right_blocks=right[0],right_steps=right[1],seed0_pp=dif[0],seed1_pp=dif[1],seed2_pp=dif[2],mean_pp=statistics.mean(dif),sample_sd_pp=statistics.stdev(dif)))
    for steps in BUDGETS[8]:paired(f'depth_gap_at_{steps}',(32,steps),(8,steps))
    paired('near_density_32_48000_vs_8_12000',(32,48000),(8,12000));paired('near_density_32_96000_vs_8_24000',(32,96000),(8,24000));paired('32_96000_vs_8_48000',(32,96000),(8,48000))
    for b in [8,32]:paired(f'budget_gain_{b}_48000_vs_12000',(b,48000),(b,12000))
    paired('budget_gain_32_96000_vs_12000',(32,96000),(32,12000));paired('budget_gain_32_96000_vs_48000',(32,96000),(32,48000))
    write_csv(ROOT/'aggregate/paired_comparisons.csv',comparisons)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for b in BUDGETS:
        ar=[r for r in agg if r['blocks']==b];x=[r['steps'] for r in ar];y=[r['test_percent_mean'] for r in ar];sd=[r['test_percent_sample_sd'] for r in ar]
        axes[0].errorbar(x,y,yerr=sd,marker='o',label=f'{b} blocks')
        axes[1].errorbar([r['fires_per_parameter_mean'] for r in ar],y,yerr=sd,marker='o',label=f'{b} blocks')
        val=[read(outdir(b,s)/'validation.csv') for s in range(3)];vs=np.array([[float(r['accuracy'])*100 for r in rr] for rr in val]);vx=[int(r['step']) for r in val[0]]
        axes[2].plot(vx,vs.mean(0),label=f'{b} blocks');axes[2].fill_between(vx,vs.mean(0)-vs.std(0,ddof=1),vs.mean(0)+vs.std(0,ddof=1),alpha=.15)
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    axes[0].set(xlabel='Intervals',ylabel='Final checkpoint test (%)');axes[1].set(xlabel='Actual cumulative fires / parameter',ylabel='Test (%)');axes[2].set(xlabel='Intervals',ylabel='Validation (%)')
    fig.tight_layout();(ROOT/'figures').mkdir(exist_ok=True);fig.savefig(ROOT/'figures/depth_budget.png',dpi=170);fig.savefig(ROOT/'figures/depth_budget.pdf');plt.close(fig)
    lines=['# 深さと更新予算：CUDA Graph CPU整理版','', '8/32残差ブロック・幅76・A8/ReLU、各3 seed。8側6k/12k/24k/48k、32側はさらに96k。6軌跡から27固定checkpointを保存し、全学習凍結と監査の後にCPU推論でtestを各1回評価。途中選択なし。同一軌跡内の予算点は独立runではない。','', '| blocks | 区間 | test平均±標本SD % | 実発火/重み | 更新済み重み割合 | logits RMS | 比>0.5件数 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in agg:lines.append(f'| {r["blocks"]} | {r["steps"]} | {r["test_percent_mean"]:.3f} ± {r["test_percent_sample_sd"]:.3f} | {r["fires_per_parameter_mean"]:.5f} | {r["unique_updated_fraction_mean"]:.5f} | {r["logits_rms_mean"]:.4f} | {r["branch_ratio_exceed_count"]}/{r["branch_ratio_count"]} |')
    lines+=['','| 比較（左−右） | 平均差±標本SD pt |','|---|---:|']
    for r in comparisons:lines.append(f'| {r["comparison"]} | {r["mean_pp"]:+.3f} ± {r["sample_sd_pp"]:.3f} |')
    gap12=next(r['mean_pp'] for r in comparisons if r['comparison']=='depth_gap_at_12000');density=next(r['mean_pp'] for r in comparisons if r['comparison']=='near_density_32_48000_vs_8_12000')
    lines+=['',f'同一12,000区間の深さ差は{gap12:+.3f}pt、深32/48,000対浅8/12,000の近似予算密度比較は{density:+.3f}pt。差の変化は{density-gap12:+.3f}pt。差が縮まるなら更新予算希薄化の寄与と整合し、残る差は試した予算内での追加の困難を示唆するが、二要因の純粋な因果分解や一方のみが主因であることを証明しない。','', '重み数比は377264/100016=3.772倍。4倍区間は深い側の区間/重みを約6.05%多く与える。実発火率、反復更新、選択率も異なり得るため、実発火/重みとユニーク更新率を必ず併読する。区間増加はデータ抽出回数・計算量も増やす。testで都合のよい予算や補間点を選んでいない。','', '旧CPU E17a90.637%、E18c87.553%は背景の参照値。今回の主比較は両深さを同じGPU CPU整理エンジンで学習した値。8ブロック12kは保存済みGPU E17aとのloss/state一致を別途記録。CPUスケール警告は診断であり自動失敗扱いにしない。','', 'per_seed以下に全損失・選択・発火・S・層イベント・500区間ごとvalidation/RMS/量子化・初期と各budgetの層単独候補差・model/RNGを保存。全432,000区間の損失差・S・発火からのendpoint再構成を監査。RNG検証は各checkpointから局所3区間の基準Graph対照であり、全履歴をnaiveで再学習したものではない。']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')
    dump(ROOT/'status.json',dict(complete=True,stage='complete',completed_trajectories=6,endpoint_results=27,audited=True,test_evaluations=27))
    dump(ROOT/'artifacts_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='artifacts_sha256.json'})
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    frozen=audit();rows=final_tests(frozen);analyze(rows)
