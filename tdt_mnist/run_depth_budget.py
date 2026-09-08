"""Fixed depth/budget trajectories with deferred test and automatic final audit."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,json,time,sys,subprocess,shutil,math,statistics,hashlib
from pathlib import Path
import numpy as np
import torch
from gpu_graph_optimizations import evaluator,epoch
from gpu_evaluation_engines import configure_gpu
from residual_followup_models import ResidualTDT
from run_residual_e17 import setup,config,load_data,evaluate,dump,write_csv,sha
from run_residual_followups import observe,probes
HERE=Path(__file__).resolve().parent;ROOT=HERE/'results/depth-budget-cpu-compact-20260908'
BUDGETS={8:[6000,12000,24000,48000],32:[6000,12000,24000,48000,96000]}
ORDER=[(8,0),(32,0),(32,1),(8,1),(8,2),(32,2)]
def init():setup();configure_gpu();os.sched_setaffinity(0,{15})
def gens(s):return torch.Generator().manual_seed(s+1),torch.Generator().manual_seed(s+100000)
def read(p):return list(csv.DictReader(p.open()))
def model(s,b):return ResidualTDT(s,b,76,'a8')
def outdir(b,s):return ROOT/'per_seed'/f'blocks{b}-seed{s}'
def append_csv(p,rows):
    if not rows:return
    p.parent.mkdir(exist_ok=True,parents=True);new=not p.exists()
    with p.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        if new:w.writeheader()
        w.writerows(rows)

def check_local(m,x,y,a,g,bg,scale):
    e0=evaluator(m,x,y,'gpu_graph');e1=evaluator(m,x,y,'cpu_compact')
    refs=[]
    for step in range(3):
        g0=torch.Generator().set_state(g.get_state());b0=torch.Generator().set_state(bg.get_state());t0={};t1={}
        r0,_=epoch(m,x,y,a,g0,scale,b0,e0,'gpu_graph',t0)
        r1,_=epoch(m,x,y,a,g,scale,bg,e1,'cpu_compact',t1)
        assert torch.equal(e0.last_losses,e1.last_losses) and torch.equal(r0[0],r1[0]) and r0[2]==r1[2] and r0[3]==r1[3]
        assert torch.equal(g.get_state(),g0.get_state()) and torch.equal(bg.get_state(),b0.get_state())
        for k in ['votes','counters']:assert torch.equal(torch.stack(t0[k]),torch.stack(t1[k]))
        for k in ['indices','batches','codes']:assert torch.equal(getattr(e0.last_plan,k),getattr(e1.last_plan,k))
        m.weights.copy_(r1[0]);scale=r1[3]
    del e0,e1;torch.cuda.empty_cache()
    return dict(steps=3,losses=384,all_loss_bits_votes_counters_actions_stats_scale_rng_equal=True)

def preflight():
    init();a=config(0,HERE/'data');(x,y),(vx,vy),_=load_data(a,torch.device('cpu'));rows=[];timings=[]
    old17=HERE/'results/residual-stream-a8-e17-20260908';old18=HERE/'results/residual-followups-e18-e20-20260908'
    expected17=json.loads((old17/'preflight.json').read_text())['initial_validation'];expected18=json.loads((old18/'preflight.json').read_text())['initial_validation']
    for b in BUDGETS:
        for seed in range(3):
            m=model(seed,b);v=evaluate(m,vx,vy)
            expected=next(r for r in (expected17 if b==8 else expected18) if r['seed']==seed and r['condition']==('E17a' if b==8 else 'E18c'))
            assert v=={k:expected[k] for k in ['loss','accuracy']}
            assert m.num_params==(100016 if b==8 else 377264) and len(m.shapes)==2*b+2
            for state in ['initial','trained']:
                m=model(seed,b)
                if state=='trained':
                    old=(old17/'per_seed'/f'E17a-seed{seed}'/'model.pt') if b==8 else (old18/'per_seed'/f'E18c-seed{seed}'/'model.pt')
                    m.weights.copy_(torch.load(old,map_location='cpu',weights_only=False)['weights'])
                g,bg=gens(seed);checks=check_local(m,x,y,a,g,bg,.02)
                rows.append(dict(blocks=b,seed=seed,state=state,num_params=m.num_params,matrices=len(m.shapes),initial_validation=v,**checks))
            print(f'preflight {b} blocks seed{seed} passed, initialval={v["accuracy"]:.3%}',flush=True)
        m=model(0,b);g,bg=gens(0);e=evaluator(m,x,y,'cpu_compact');s=.02;t=time.perf_counter()
        for _ in range(100):r,_=epoch(m,x,y,a,g,s,bg,e,'cpu_compact');m.weights.copy_(r[0]);s=r[3]
        sec=(time.perf_counter()-t)/100
        timings.append(dict(blocks=b,intervals=100,seconds_per_interval=sec,estimated_training_seconds_per_seed=sec*max(BUDGETS[b])))
        del e;torch.cuda.empty_cache()
    result=dict(passed=True,checks=rows,timing=timings,estimated_six_trajectory_engine_seconds=sum(r['estimated_training_seconds_per_seed']*3 for r in timings),test_evaluated=False)
    dump(ROOT/'preflight.json',result);print(json.dumps(result['timing']),flush=True)

def checkpoint(m,g,bg,s,step,updates,selections,engine_seconds,wall_seconds,total_fires):
    return dict(weights=m.weights.clone(),generator=g.get_state(),batch_generator=bg.get_state(),scale=s,step=step,updates=torch.from_numpy(updates.copy()),selections=torch.from_numpy(selections.copy()),engine_seconds=engine_seconds,elapsed_seconds=wall_seconds,total_fires=total_fires)

def worker(blocks,seed):
    init();a=config(seed,HERE/'data');a.steps=max(BUDGETS[blocks]);m=model(seed,blocks);out=outdir(blocks,seed);out.mkdir(parents=True,exist_ok=False)
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()};cfg.update(blocks=blocks,width=76,matrices=len(m.shapes),num_params=m.num_params,budgets=BUDGETS[blocks],engine='cpu_compact',candidate_device='cuda',evaluation_device='cpu',defer_all_test=True,shapes=m.shapes,scales=m.scales)
    dump(out/'config.json',cfg);(x,y),(vx,vy),_=load_data(a,m.device);g,bg=gens(seed);s=.02
    initial,signals,activations,ratios=observe(m,vx,vy,0)
    dump(out/'initial_validation.json',initial);append_csv(out/'validation.csv',[dict(step=0,**initial)]);append_csv(out/'signal.csv',signals);append_csv(out/'activation.csv',activations);append_csv(out/'rms_ratios.csv',ratios)
    diagnostic_seconds=0.;t=time.perf_counter();write_csv(out/'probes-initial.csv',probes(m,x,y,seed,'initial'));diagnostic_seconds+=time.perf_counter()-t
    t=time.perf_counter();ev=evaluator(m,x,y,'cpu_compact');setup_seconds=time.perf_counter()-t
    steps=a.steps;layers=len(m.shapes)
    arrays={name:np.lib.format.open_memmap(out/f'{name}.npy',mode='w+',dtype=dtype,shape=shape) for name,dtype,shape in [('candidate_losses','float32',(steps,128)),('abs_y','float32',(steps,64)),('selected_indices','int32',(steps,16)),('layer_events','uint8',(steps,layers,3))]}
    ends=np.cumsum([math.prod(z) for z in m.shapes]);updates=np.zeros(m.num_params,np.int32);selections=np.zeros(m.num_params,np.int32);total_fires=0;engine_seconds=0.;validation_seconds=0.;started=time.perf_counter();endpoint_rows=[];history_mismatch=0
    oldgpu=HERE/'results/gpu-e17a-reproduction-20260908/per_seed'/f'seed{seed}'
    oldloss=np.load(oldgpu/'candidate_losses.npy',mmap_mode='r') if blocks==8 else None
    with (out/'metrics.csv').open('w',newline='') as mf,(out/'firing.csv').open('w',newline='') as ff:
        mw=None;fw=csv.DictWriter(ff,fieldnames=['step','coordinate','target']);fw.writeheader()
        for step in range(1,steps+1):
            t=time.perf_counter();(proposal,indices,stats,new_s),timing=epoch(m,x,y,a,g,s,bg,ev,'cpu_compact');engine_seconds+=time.perf_counter()-t
            idx=indices.numpy();local=(m.weights[indices]!=proposal[indices]).nonzero().flatten();assert len(local)<=1
            coordinate=int(indices[local[0]]) if len(local) else -1;target=int(proposal[coordinate]) if coordinate>=0 else 0
            values=ev.last_losses.numpy();assert np.isfinite(values).all()
            arrays['candidate_losses'][step-1]=values;arrays['abs_y'][step-1]=np.abs(values[::2]-values[1::2]);arrays['selected_indices'][step-1]=idx
            if oldloss is not None and step<=12000:history_mismatch+=int(np.any(values.view(np.int32)!=oldloss[step-1].view(np.int32)))
            selections[idx]+=1;selected_layers=np.bincount(np.searchsorted(ends,idx,side='right'),minlength=layers)
            arrays['layer_events'][step-1,:,0]=selected_layers;arrays['layer_events'][step-1,:,1]=selected_layers>0;arrays['layer_events'][step-1,:,2]=0
            if coordinate>=0:
                updates[coordinate]+=1;total_fires+=1;arrays['layer_events'][step-1,np.searchsorted(ends,coordinate,side='right'),2]=1
            fw.writerow(dict(step=step,coordinate=coordinate,target=target));m.weights.copy_(proposal);s=new_s
            stats.pop('abs_y_values',None);stats['counter_histogram']=json.dumps(stats['counter_histogram'],sort_keys=True)
            row=dict(step=step,elapsed_seconds=time.perf_counter()-started,engine_seconds=engine_seconds,cumulative_fires=total_fires,actual_fires_per_parameter=total_fires/m.num_params,next_scale=s,**timing,**stats)
            if mw is None:mw=csv.DictWriter(mf,fieldnames=list(row));mw.writeheader()
            mw.writerow(row)
            if step%500==0:
                t=time.perf_counter();val,sr,ar,rr=observe(m,vx,vy,step);validation_seconds+=time.perf_counter()-t
                append_csv(out/'validation.csv',[dict(step=step,**val)]);append_csv(out/'signal.csv',sr);append_csv(out/'activation.csv',ar);append_csv(out/'rms_ratios.csv',rr)
                state=checkpoint(m,g,bg,s,step,updates,selections,engine_seconds,time.perf_counter()-started,total_fires)
                torch.save(state,out/'checkpoint.tmp');(out/'checkpoint.tmp').replace(out/'checkpoint.pt')
                mf.flush();ff.flush()
                for array in arrays.values():array.flush()
                dump(out/'progress.json',dict(blocks=blocks,seed=seed,step=step,target=steps,validation=val,cumulative_fires=total_fires,fires_per_parameter=total_fires/m.num_params,elapsed_seconds=time.perf_counter()-started,engine_seconds=engine_seconds,test_evaluated=False))
                print(f'blocks{blocks} seed{seed} step{step}/{steps} val={val["accuracy"]:.3%} fires={total_fires}',flush=True)
                if step in BUDGETS[blocks]:
                    ep=out/'budgets'/str(step);ep.mkdir(parents=True,exist_ok=False);torch.save(state,ep/'model.pt')
                    t=time.perf_counter();write_csv(ep/'probes.csv',probes(m,x,y,seed,f'budget{step}'));diagnostic_seconds+=time.perf_counter()-t
                    totals=arrays['layer_events'][:step].sum(axis=0,dtype=np.int64)
                    write_csv(ep/'firing_by_matrix.csv',[dict(layer=i,matrix=m.matrix_names[i],parameters=math.prod(m.shapes[i]),selected_coordinates=int(totals[i,0]),selected_intervals=int(totals[i,1]),fires=int(totals[i,2]),all_interval_firing_rate=float(totals[i,2]/step),selected_interval_firing_rate=float(totals[i,2]/max(totals[i,1],1))) for i in range(layers)])
                    logits=next(z['rms'] for z in sr if z['matrix']=='W_out' and z['stage']=='output')
                    endpoint=dict(blocks=blocks,seed=seed,steps=step,num_params=m.num_params,validation=val,total_fires=total_fires,fire_fraction=total_fires/step,fires_per_parameter=total_fires/m.num_params,intervals_per_parameter=step/m.num_params,selected_coordinates_per_parameter=16*step/m.num_params,unique_updated_coordinates=int(np.count_nonzero(updates)),unique_updated_fraction=float(np.count_nonzero(updates)/m.num_params),branch_ratio_exceed=sum(z['branch_stream_rms_ratio'] is not None and z['branch_stream_rms_ratio']>.5 for z in rr),branch_ratio_total=blocks,logits_rms=logits,logits_rms_exceed=logits>10,engine_seconds=engine_seconds,elapsed_seconds=state['elapsed_seconds'],model_sha256=sha(ep/'model.pt'),test_evaluated=False)
                    dump(ep/'training_summary.json',endpoint);endpoint_rows.append(endpoint)
                    if blocks==8 and step==12000:
                        hist=torch.load(oldgpu/'model.pt',map_location='cpu',weights_only=False)
                        equal=torch.equal(hist['weights'],m.weights)
                        dump(out/'historical_gpu_12000_check.json',dict(loss_mismatch_intervals=history_mismatch,weights_equal=equal,scale_equal=hist['scale']==s,test_read=False));assert not history_mismatch and equal and hist['scale']==s
    for array in arrays.values():array.flush()
    dump(out/'training_complete.json',dict(blocks=blocks,seed=seed,steps=steps,endpoint_count=len(endpoint_rows),engine_seconds=engine_seconds,training_loop_seconds=time.perf_counter()-started,diagnostic_seconds=diagnostic_seconds,validation_seconds=validation_seconds,gpu_setup_seconds=setup_seconds,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),test_evaluated=False))

def run_all():
    ROOT.mkdir(parents=True,exist_ok=True);assert not (ROOT/'sources').exists(),'Do not overwrite a registered experiment'
    init();names=['run_depth_budget.py','analyze_depth_budget.py','DEPTH_BUDGET_PREREGISTRATION.md','gpu_graph_optimizations.py','gpu_evaluation_engines.py','allocation_engines.py','train.py','residual_stream.py','residual_followup_models.py','run_residual_e17.py','run_residual_followups.py','depth_diagnostics.py','activation_quantization.py']
    (ROOT/'sources').mkdir()
    for name in names:shutil.copy2(HERE/name,ROOT/'sources'/name)
    dump(ROOT/'manifest.json',dict(preregistration_commit='a36cb75',sources={n:sha(ROOT/'sources'/n) for n in names},data={p.name:sha(p) for p in (HERE/'data/MNIST/raw').glob('*-ubyte')}))
    dump(ROOT/'config.json',dict(budgets=BUDGETS,seeds=[0,1,2],trajectories=6,endpoints=27,engine='cpu_compact',cpu_threads=1,cpu_affinity=[15],test_deferred_until_all_training_complete=True))
    dump(ROOT/'runtime_workers.json',dict(order=ORDER,sequential=True,cpu_affinity=[15],processes=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True)))
    dump(ROOT/'environment.json',dict(torch_version=torch.__version__,cuda_version=torch.version.cuda,nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),TF32=False,cublas=os.environ['CUBLAS_WORKSPACE_CONFIG']))
    dump(ROOT/'status.json',dict(complete=False,stage='preflight',completed_trajectories=0,expected_trajectories=6,test_evaluated=False))
    preflight();env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for i,(b,s) in enumerate(ORDER):
        dump(ROOT/'status.json',dict(complete=False,stage='training',completed_trajectories=i,expected_trajectories=6,active=dict(blocks=b,seed=s,target=max(BUDGETS[b])),test_evaluated=False))
        with (ROOT/f'blocks{b}-seed{s}.log').open('w') as f:subprocess.run([sys.executable,__file__,'worker','--blocks',str(b),'--seed',str(s)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    frozen={}
    for b,s in ORDER:
        assert (outdir(b,s)/'training_complete.json').exists()
        for step in BUDGETS[b]:frozen[str((outdir(b,s)/'budgets'/str(step)/'model.pt').relative_to(ROOT))]=sha(outdir(b,s)/'budgets'/str(step)/'model.pt')
    assert len(frozen)==27;dump(ROOT/'all_training_frozen.json',dict(endpoints=frozen,test_evaluated=False))
    dump(ROOT/'status.json',dict(complete=False,stage='audit_and_final_evaluation',completed_trajectories=6,test_evaluated=False))
    subprocess.run([sys.executable,str(HERE/'analyze_depth_budget.py')],env=env,check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker','preflight']);p.add_argument('--blocks',type=int,choices=[8,32],default=8);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
    try:
        if a.mode=='run':run_all()
        elif a.mode=='preflight':preflight()
        else:worker(a.blocks,a.seed)
    except Exception as e:
        if a.mode=='run':dump(ROOT/'failure.json',dict(error=repr(e),stage='orchestrator',test_evaluated=False))
        raise
