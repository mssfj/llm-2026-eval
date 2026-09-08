"""Audited fast-engine preflight, same-host benchmark and E17a reproduction."""
import argparse,csv,json,os,sys,time,resource,math,shutil,subprocess,statistics
from pathlib import Path
import numpy as np
import torch
import train
from residual_stream import ResidualStreamModel
from residual_followup_models import ResidualTDT
from fast_engine import epoch,schedule,candidate_losses
from run_residual_e17 import config,setup,load_data,evaluate,observe,probes,dump,write_csv,sha
from depth_diagnostics import layer_events
HERE=Path(__file__).resolve().parent
ROOT=HERE/'results/fast-engine-e17a-20260908'
OLD=HERE/'results/residual-stream-a8-e17-20260908'

def rss():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):return int(line.split()[1])*1024

def peak():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024

def changed(old,new):
    idx=(old!=new).nonzero().flatten()
    return [(int(i),int(new[i])) for i in idx]

def finalize_run(out):
    dump(out/'manifest.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'})

def benchmark(a):
    setup();cfg=config(0,HERE/'data');(x,y),_,_=load_data(cfg,torch.device('cpu'))
    model=ResidualStreamModel() if a.blocks==8 else ResidualTDT(seed=0,blocks=16,width=76)
    g=torch.Generator().manual_seed(1);bg=torch.Generator().manual_seed(100000)
    baseline=rss();baselinepeak=peak();s=.02;rows=[]
    # Warm up a separate schedule; no update or RNG advancement of measured model.
    if a.engine=='unguarded':candidate_losses(model,x,y,schedule(model,x,cfg,g,bg),guard=False)
    else:epoch(model,x,y,cfg,torch.Generator().set_state(g.get_state()),s,torch.Generator().set_state(bg.get_state()),engine=a.engine)
    started=time.perf_counter()
    for step in range(1,a.intervals+1):
        t=time.perf_counter();trace={}
        if a.engine=='unguarded':
            # Diagnostic fixed-weight workloads, RNG advances identically via naive decision schedule.
            plan=schedule(model,x,cfg,g,bg);losses,trace=candidate_losses(model,x,y,plan,guard=False)
            # Used only as evaluation-only timing, explicitly not an epoch benchmark.
            g.manual_seed(step+1);bg.manual_seed(step+100000)
        else:
            proposal,idx,stats,s=epoch(model,x,y,cfg,g,s,bg,engine=a.engine,trace=trace if a.engine=='fast' else None)
            model.weights.copy_(proposal)
        if step%500==0:dump(ROOT/'benchmark_progress.json',dict(engine=a.engine,blocks=a.blocks,step=step,intervals=a.intervals,elapsed_seconds=time.perf_counter()-started))
        rows.append(dict(step=step,seconds=time.perf_counter()-t,first_matrix=trace.get('first'),guard_fallbacks=trace.get('guard_fallbacks'),cache_tensor_bytes=trace.get('cache_bytes')))
    elapsed=time.perf_counter()-started
    out=ROOT/'benchmark'/f'{a.engine}-{a.blocks}-{a.intervals}';write_csv(out/'intervals.csv',rows)
    # INT8 weights + C8 evidence + INT32 visit counts are the persistent logical state.
    # Evidence/counts actually allocate per epoch and are reset; counted explicitly.
    logical=model.num_params*(1+2+8)
    summary=dict(engine=a.engine,blocks=a.blocks,intervals=a.intervals,threads=1,seconds=elapsed,seconds_per_interval=elapsed/a.intervals,estimated_12000_seconds=elapsed/a.intervals*12000,actual_12000_engine_seconds=sum(r['seconds'] for r in rows) if a.intervals==12000 else None,
        rss_before_workload=baseline,peak_rss_before_workload=baselinepeak,peak_rss=peak(),rss_after_workload=rss(),rss_peak_increment=max(0,peak()-baseline),logical_weights_bytes=model.num_params,logical_counter_bytes=model.num_params*10,logical_total_bytes=logical,
        max_cache_tensor_bytes=max([r['cache_tensor_bytes'] or 0 for r in rows]),guard_fallback_mean=statistics.mean([r['guard_fallbacks'] or 0 for r in rows]),
        timing_scope='evaluation_only_fixed_weights' if a.engine=='unguarded' else 'full_legacy_interval_with_update',cpu_affinity=sorted(os.sched_getaffinity(0)))
    if a.engine=='fast':
        L=len(model.shapes);firsts=[r['first_matrix'] for r in rows]
        summary['requested_ideal_forward_equivalent_mean']=statistics.mean([1+128*(L-l)/L for l in firsts])
        summary['actual_dense_matmul_equivalent_mean']=statistics.mean([64+128*sum(math.prod(z) for z in model.shapes[l+1:])/model.num_params+r['guard_fallbacks'] for l,r in zip(firsts,rows)])
    dump(out/'summary.json',summary);print(json.dumps(summary),flush=True)

def run(a):
    setup();cfg=config(a.seed,HERE/'data');m=ResidualStreamModel(a.seed)
    out=ROOT/'per_seed'/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=False)
    dump(out/'config.json',{**{k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()},'engine':a.engine,'guard_half_integer_distance':1e-4,'source_manifest':json.loads((ROOT/'manifest.json').read_text())})
    (x,y),(vx,vy),(tx,ty)=load_data(cfg,m.device)
    g=torch.Generator().manual_seed(a.seed+1);bg=torch.Generator().manual_seed(a.seed+100000)
    naive=ResidualStreamModel(a.seed);ng=torch.Generator().manual_seed(a.seed+1);nbg=torch.Generator().manual_seed(a.seed+100000)
    s=ns=.02;first_div=None;first_numeric=None;hist_mismatch=[]
    historical=OLD/'per_seed'/f'E17a-seed{a.seed}'
    history=list(csv.DictReader((historical/'metrics.csv').open()));oldabs=np.load(historical/'abs_y.npy',mmap_mode='r')
    oldlayers=iter(csv.DictReader((historical/'layer_metrics.csv').open()))
    initial,signals,activations,ratios=observe(m,vx,vy,0)
    dump(out/'initial_validation.json',initial)
    write_csv(out/'probes.csv',probes(m,x,y,a.seed,'initial'))
    ys=np.lib.format.open_memmap(out/'abs_y.npy',mode='w+',dtype='float32',shape=(12000,64))
    baseline=rss();started=time.perf_counter();engine_seconds=0;reference_seconds=0;fallbacks=0;certified=0;replayed=0
    with (out/'metrics.csv').open('w',newline='') as mf,(out/'firing.csv').open('w',newline='') as ff,(out/'layer_metrics.csv').open('w',newline='') as lf:
        mw=lw=None;fw=csv.DictWriter(ff,fieldnames=['step','coordinate','target','naive_comparison']);fw.writeheader()
        for step in range(1,12001):
            trace={};t=time.perf_counter()
            proposal,indices,stats,new_s=epoch(m,x,y,cfg,g,s,bg,engine=a.engine,trace=trace if a.engine=='fast' else None)
            engine_seconds+=time.perf_counter()-t
            events=layer_events(m,proposal,indices);fire=changed(m.weights,proposal)
            numeric=False;comparison='after_first_divergence'
            if first_div is None:
                if trace.get('guard_fallbacks')==128 and s==ns and torch.equal(m.weights,naive.weights):
                    comparison='all_losses_original_naive';certified+=1
                    naive.weights.copy_(proposal);ng.set_state(g.get_state());nbg.set_state(bg.get_state());ns=new_s
                else:
                    t=time.perf_counter();np_,ni,nstats,ns2=train.epoch(naive,x,y,cfg,ng,ns,nbg);reference_seconds+=time.perf_counter()-t;replayed+=1
                    nfire=changed(naive.weights,np_)
                    assert torch.equal(indices,ni) and torch.equal(g.get_state(),ng.get_state()) and torch.equal(bg.get_state(),nbg.get_state())
                    numeric=stats['abs_y_values']!=nstats['abs_y_values'] or new_s!=ns2
                    if fire!=nfire:
                        first_div=step
                        dump(out/'first_divergence.json',dict(step=step,fast_fire=fire,naive_fire=nfire,fast_scale=s,naive_scale=ns,fast_stats=stats,naive_stats=nstats,fast_losses=trace.get('losses'),reason='identical candidate schedule and shared decision code; inspect numerical differences'))
                    comparison='independent_naive_replay';naive.weights.copy_(np_);ns=ns2
                if numeric and first_numeric is None:first_numeric=step
            old=history[step-1]
            mism=[]
            if first_div is None:
                for key in ['fires','counter_min','counter_max','counter_mean','counter_abs_mean','nonzero_vote_rate','scale']:
                    if float(stats[key])!=float(old[key]):mism.append(key)
                if not np.array_equal(np.array(stats['abs_y_values'],dtype=np.float32),oldabs[step-1]):mism.append('abs_y')
            for e in events:
                olde=next(oldlayers)
                if first_div is None and any(int(e[k])!=int(olde[k]) for k in ['selected_coordinates','fires']):mism.append('layer_events')
                row=dict(step=step,**e)
                if lw is None:lw=csv.DictWriter(lf,fieldnames=list(row));lw.writeheader()
                lw.writerow(row)
            if mism and len(hist_mismatch)<100:hist_mismatch.append(dict(step=step,fields=mism))
            for coord,target in fire:fw.writerow(dict(step=step,coordinate=coord,target=target,naive_comparison=comparison))
            if not fire:fw.writerow(dict(step=step,coordinate='',target='',naive_comparison=comparison))
            values=stats.pop('abs_y_values');ys[step-1]=values;stats['counter_histogram']=json.dumps(stats['counter_histogram'],sort_keys=True)
            s=new_s;m.weights.copy_(proposal);fallbacks+=trace.get('guard_fallbacks',0)
            row=dict(step=step,elapsed_seconds=time.perf_counter()-started,engine_seconds=engine_seconds,first_matrix=trace.get('first'),guard_fallbacks=trace.get('guard_fallbacks'),abs_y_mean=float(np.mean(values)),val_loss=None,val_accuracy=None,**stats)
            if step%500==0:
                val,sr,ar,rr=observe(m,vx,vy,step);signals.extend(sr);activations.extend(ar);ratios.extend(rr)
                row.update(val_loss=val['loss'],val_accuracy=val['accuracy'])
                write_csv(out/'signal.csv',signals);write_csv(out/'activation.csv',activations);write_csv(out/'rms_ratios.csv',ratios)
                torch.save(dict(weights=m.weights,step=step,scale=s,generator=g.get_state(),batch_generator=bg.get_state()),out/'checkpoint.pt')
                dump(out/'progress.json',dict(step=step,validation=val,elapsed_seconds=time.perf_counter()-started,first_firing_divergence=first_div,guard_fallback_fraction=fallbacks/(step*128)))
                ys.flush();print(f'seed{a.seed} step={step} val={val["accuracy"]:.3%} first_divergence={first_div}',flush=True)
            if mw is None:mw=csv.DictWriter(mf,fieldnames=list(row));mw.writeheader()
            mw.writerow(row);mf.flush();ff.flush();lf.flush()
    train_elapsed=time.perf_counter()-started;ys.flush()
    initial_probes=list(csv.DictReader((out/'probes.csv').open()));write_csv(out/'probes.csv',initial_probes+probes(m,x,y,a.seed,'final'))
    torch.save(dict(weights=m.weights,scale=s),out/'model.pt')
    # Single test evaluation, after the frozen 12000 intervals.
    test=evaluate(m,tx,ty)
    oldmodel=torch.load(historical/'model.pt',weights_only=False);oldweights=oldmodel['weights'] if isinstance(oldmodel,dict) else oldmodel
    oldtest=next(r for r in csv.DictReader((OLD/'per_seed/results.csv').open()) if r['condition']=='E17a' and int(r['seed'])==a.seed)
    summary=dict(seed=a.seed,engine=a.engine,test_percent=test['accuracy']*100,naive_test_percent=float(oldtest['test_accuracy_percent']),delta_pp=test['accuracy']*100-float(oldtest['test_accuracy_percent']),first_firing_divergence=first_div,first_numeric_difference=first_numeric,certified_original_loss_intervals=certified,independently_replayed_intervals=replayed,historical_mismatches=hist_mismatch,final_weights_equal_historical=torch.equal(m.weights,oldweights),train_elapsed_seconds=train_elapsed,engine_seconds=engine_seconds,reference_replay_seconds=reference_seconds,guard_fallback_fraction=fallbacks/(12000*128),rss_before_training=baseline,peak_rss=peak(),logical_weights_bytes=m.num_params,logical_counter_bytes=m.num_params*10,test=test)
    dump(out/'summary.json',summary);finalize_run(out);print(f'seed{a.seed} complete; test saved for final report',flush=True)

def analyze():
    rows=[]
    for seed in range(3):
        out=ROOT/'per_seed'/f'seed{seed}'
        for name,h in json.loads((out/'manifest.json').read_text()).items():assert sha(out/name)==h
        s=json.loads((out/'summary.json').read_text());rows.append(s)
        metrics=list(csv.DictReader((out/'metrics.csv').open()));assert len(metrics)==12000
        assert all(int(r['fires'])<=1 for r in metrics)
        assert np.isfinite(np.load(out/'abs_y.npy')).all()
        # Historical numerical differences are reported; trajectory criteria are separate.
    vals=[r['test_percent'] for r in rows];strong=all(r['first_firing_divergence'] is None and r['final_weights_equal_historical'] for r in rows)
    mean=statistics.mean(vals);fallback=90.337<=mean<=90.937
    write_csv(ROOT/'per_seed/results.csv',[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows])
    benches=[json.loads(p.read_text()) for p in sorted((ROOT/'benchmark').glob('*/summary.json'))]
    keys=sorted({k for r in benches for k,v in r.items() if not isinstance(v,list)})
    write_csv(ROOT/'benchmark/results.csv',[{k:r.get(k) for k in keys} for r in benches])
    report=dict(level1=json.loads((ROOT/'level1.json').read_text()),strong_pass=strong,fallback_accuracy_pass=fallback,level2_pass=strong or fallback,test_mean_percent=mean,test_sample_std_percent=statistics.stdev(vals),runs=rows,benchmarks=benches)
    dump(ROOT/'report.json',report)
    lines=['# TDT evaluation engine optimization / E17a reproduction','',f'Test: {mean:.4f} ± {statistics.stdev(vals):.4f}% (sample SD, 3 seeds).',f'Level 1 passed: {report["level1"]["passed"]}; maximum relative error: {report["level1"]["max_relative_error"]:.9g}.',f'Level 2 strongest firing-series criterion: {strong}; accuracy fallback criterion: {fallback}.','','The unguarded low-rank path failed A8 numerical acceptance. The production fast path includes naive re-evaluation near A8 rounding boundaries. This preserves observed results but can be slower; passing equivalence is not evidence of acceleration.','','| seed | fast test % | naive test % | delta pp | first firing divergence | actual train seconds | fallback fraction |','|---|---:|---:|---:|---|---:|---:|']
    for r in rows:lines.append(f'| {r["seed"]} | {r["test_percent"]:.2f} | {r["naive_test_percent"]:.2f} | {r["delta_pp"]:+.2f} | {r["first_firing_divergence"]} | {r["train_elapsed_seconds"]:.2f} | {r["guard_fallback_fraction"]:.6f} |')
    lines+=['','| blocks | naive sec/interval | fast sec/interval | speedup naive/fast | naive peak MiB | fast peak MiB |','|---|---:|---:|---:|---:|---:|']
    for blocks in [8,16]:
        n=next(r for r in benches if r['blocks']==blocks and r['engine']=='naive' and r['intervals']==100);f=next(r for r in benches if r['blocks']==blocks and r['engine']=='fast')
        lines.append(f'| {blocks} | {n["seconds_per_interval"]:.6f} | {f["seconds_per_interval"]:.6f} | {n["seconds_per_interval"]/f["seconds_per_interval"]:.3f} | {n["peak_rss"]/2**20:.2f} | {f["peak_rss"]/2**20:.2f} |')
    lines+=['','RSS includes Python/PyTorch/data and allocator retention. Logical weights/counters and cache tensor estimates are recorded separately in benchmark JSON. Counter state resets each interval. Cache byte estimate sums references and may overcount aliased stream/output tensors; RSS is the process-level measurement.','The requested ideal 1+128*suffix/L assumes one common minibatch. Legacy uses 64 minibatches; the actual batched cache has64 full-minibatch equivalents, and all later perturbed matrices require sparse corrections. See requested_ideal_forward_equivalent_mean and actual_dense_matmul_equivalent_mean.','Actual reproduction runtimes include concurrent worker resource sharing and validation/checkpoints. Benchmark12000-interval estimates are explicitly extrapolated; historical naive total times are not same-load speed comparisons.','Fixed-input numerical tests are not a universal mathematical proof. All comparisons are for the specified machine, model, seed and FP32 implementation.']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')
    dump(ROOT/'status.json',dict(complete=True,completed=3,audited=True))
    dump(ROOT/'artifacts_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='artifacts_sha256.json'})

def orchestrate():
    ROOT.mkdir(exist_ok=True,parents=True)
    assert json.loads((ROOT/'level1.json').read_text())['passed']
    sources=['fast_engine.py','run_fast_engine.py','test_fast_engine.py','FAST_ENGINE_PREREGISTRATION.md','train.py','residual_stream.py','residual_followup_models.py','run_residual_e17.py','activation_quantization.py','depth_diagnostics.py']
    (ROOT/'sources').mkdir(exist_ok=False)
    for name in sources:shutil.copy2(HERE/name,ROOT/'sources'/name)
    dump(ROOT/'manifest.json',dict(sources={n:sha(ROOT/'sources'/n) for n in sources},preregistration_commit='a4a69ae',git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),torch_version=torch.__version__,data_sha256={p.name:sha(p) for p in (HERE/'data/MNIST/raw').glob('*-ubyte')}))
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    # Sequential benchmarks before any concurrent reproduction jobs.
    for blocks in [8,16]:
        for engine in ['naive','fast']:
            subprocess.run([sys.executable,__file__,'benchmark','--blocks',str(blocks),'--engine',engine],env=env,check=True)
    subprocess.run([sys.executable,__file__,'benchmark','--blocks','8','--engine','naive','--intervals','12000'],env=env,check=True)
    dump(ROOT/'status.json',dict(complete=False,completed=0,active_seeds=[0],phase='isolated_seed0'))
    with (ROOT/'seed0.log').open('w') as log:
        subprocess.run([sys.executable,__file__,'worker','--seed','0','--engine','fast'],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    workers=[];events=[]
    for seed in [1,2]:
        log=(ROOT/f'seed{seed}.log').open('w')
        p=subprocess.Popen([sys.executable,__file__,'worker','--seed',str(seed),'--engine','fast'],stdout=log,stderr=subprocess.STDOUT,env=env)
        workers.append((seed,p,log));events.append(dict(seed=seed,pid=p.pid,start_time=time.time()))
    dump(ROOT/'runtime_workers.json',dict(workers=2,seed0_isolated_after_full_naive_benchmark=True,threads_per_worker=1,cpu_affinity=sorted(os.sched_getaffinity(0)),cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),events=events,benchmarks_sequential_before_training=True))
    while workers:
        for item in list(workers):
            seed,p,log=item
            if p.poll() is not None:
                log.close();assert p.returncode==0,(seed,p.returncode)
                workers.remove(item)
        dump(ROOT/'status.json',dict(complete=False,completed=3-len(workers),active_seeds=[s for s,p,l in workers]))
        if workers:time.sleep(5)
    analyze();print('All 3 fast reproduction runs and final audit complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['benchmark','worker','run','analyze']);p.add_argument('--engine',choices=['naive','fast','unguarded'],default='fast');p.add_argument('--blocks',type=int,choices=[8,16],default=8);p.add_argument('--seed',type=int,default=0);p.add_argument('--intervals',type=int,default=100);a=p.parse_args()
    if a.mode=='benchmark':benchmark(a)
    elif a.mode=='worker':run(a)
    elif a.mode=='run':orchestrate()
    else:analyze()
