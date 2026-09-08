"""Preregistered standalone 16-block CUDA Graph ablations; no test use."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','1')
import argparse,csv,json,hashlib,subprocess,sys,time,statistics,shutil,gc,traceback
from pathlib import Path
import numpy as np
import torch
from gpu_graph_optimizations import MODES,evaluator,epoch
from gpu_evaluation_engines import configure_gpu
from benchmark_allocation_engines import make_model,state_generators
from benchmark_gpu_engines import action,rss_key
from run_residual_e17 import setup,config,load_data,dump,write_csv,sha
HERE=Path(__file__).resolve().parent
ROOT=HERE/'results/gpu-graph-optimizations-16blocks-20260908'
EXACT=set(MODES)-{'gpu_graph','fused_graph'}
def read(p):return list(csv.DictReader(p.open()))
def init():setup();configure_gpu();os.sched_setaffinity(0,{15})
def gc_gpu():gc.collect();torch.cuda.empty_cache()
def bitdiff(a,b):return int((a.contiguous().view(torch.int32)!=b.contiguous().view(torch.int32)).sum())

def validate(modes=MODES):
    init();cfg=config(0,HERE/'data');(x,y),_,_=load_data(cfg,torch.device('cpu'))
    summaries=[];lossrows=[];diagnostics=[];failures=[]
    # Fresh evaluator per mode, reused across six states with explicit full reset.
    m=make_model(0,'initial');evs={}
    for mode in modes:
        print(f'validation setup {mode}',flush=True)
        try:evs[mode]=evaluator(m,x,y,mode)
        except Exception:
            if mode!='fused_graph':raise
            failures.append(dict(engine=mode,stage='compile_or_capture',error=traceback.format_exc()));dump(ROOT/'failed_conditions.json',failures)
            gc_gpu()
    assert 'gpu_graph' in evs
    for state in ['initial','trained']:
        for seed in range(3):
            m=make_model(seed,state);g,bg=state_generators(seed);scale=.02
            for ev in evs.values():ev.reset_model(m)
            for step in range(1,4):
                originals=(g.get_state(),bg.get_state());rt={}
                ref,_=epoch(m,x,y,cfg,g,scale,bg,evs['gpu_graph'],'gpu_graph',rt)
                reference=torch.stack(rt['losses']);refplan=evs['gpu_graph'].last_plan
                for mode,ev in evs.items():
                    if mode=='gpu_graph':continue
                    eg=torch.Generator().set_state(originals[0]);ebg=torch.Generator().set_state(originals[1]);tr={}
                    result,_=epoch(m,x,y,cfg,eg,scale,ebg,ev,mode,tr)
                    values=torch.stack(tr['losses']);plan=ev.last_plan
                    schedule_equal=all(torch.equal(getattr(plan,k),getattr(refplan,k)) for k in ['indices','batches','codes'])
                    rng_equal=torch.equal(eg.get_state(),g.get_state()) and torch.equal(ebg.get_state(),bg.get_state())
                    assert schedule_equal and rng_equal,(mode,state,seed,step,'schedule/RNG')
                    assert torch.isfinite(values).all() and torch.isin(result[0],torch.tensor([-1,0,1],dtype=torch.int8)).all()
                    rv=torch.stack(rt['votes']);rc=torch.stack(rt['counters']);tv=torch.stack(tr['votes']);tc=torch.stack(tr['counters'])
                    relative=(values-reference).abs()/reference.abs().clamp_min(1e-30)
                    dkeys=[k for k in ref[2] if result[2][k]!=ref[2][k]]
                    for k in dkeys:diagnostics.append(dict(state=state,seed=seed,step=step,engine=mode,field=k,reference=ref[2][k],actual=result[2][k]))
                    row=dict(state=state,seed=seed,step=step,engine=mode,loss_bit_mismatches=bitdiff(values,reference),max_relative_loss_error=float(relative.max()),relative_error_failures=int((relative>=1e-5).sum()),vote_mismatches=int((tv!=rv).sum()),counter_mismatches=int((tc!=rc).sum()),proposal_equal=torch.equal(result[0],ref[0]),scale_equal=result[3]==ref[3],schedule_equal=schedule_equal,rng_equal=rng_equal,diagnostic_mismatch_fields=len(dkeys))
                    summaries.append(row)
                    for i in range(128):lossrows.append(dict(state=state,seed=seed,step=step,engine=mode,candidate=i,reference_loss=float(reference[i]),loss=float(values[i]),relative_error=float(relative[i])))
                    if row['vote_mismatches'] or row['counter_mismatches']:
                        out=ROOT/'validation'/f'{mode}-{state}-seed{seed}-step{step}-mismatch.npz';out.parent.mkdir(exist_ok=True,parents=True)
                        np.savez_compressed(out,reference_votes=rv.numpy(),votes=tv.numpy(),reference_counters=rc.numpy(),counters=tc.numpy())
                    if mode in EXACT:
                        assert row['loss_bit_mismatches']==row['vote_mismatches']==row['counter_mismatches']==0 and row['proposal_equal'] and row['scale_equal'],row
                for ev in evs.values():ev.accepted(m,ref[0],ref[1])
                m.weights.copy_(ref[0]);scale=ref[3]
            print(f'validation {state} seed{seed} 3 updates done',flush=True)
    write_csv(ROOT/'validation/summary.csv',summaries);write_csv(ROOT/'validation/losses.csv',lossrows)
    if diagnostics:dump(ROOT/'validation/diagnostic_differences.json',diagnostics)
    active=list(evs)
    report=dict(structural_checks_passed=True,exact_expected_passed=True,fused_numerical_passed=(None if 'fused_graph' not in active else all(r['relative_error_failures']==r['vote_mismatches']==r['counter_mismatches']==0 and r['proposal_equal'] for r in summaries if r['engine']=='fused_graph')),active_conditions=active,failed_conditions=failures,loss_comparisons=len(lossrows),diagnostic_difference_count=len(diagnostics),common_reference_states=True,test_evaluated=False)
    dump(ROOT/'validation.json',report);print(json.dumps(report),flush=True)
    del evs;gc_gpu();return report


def worker(args):
    init();cfg=config(args.seed,HERE/'data');(x,y),_,_=load_data(cfg,torch.device('cpu'))
    m=make_model(args.seed,'trained');g,bg=state_generators(args.seed);scale=.02
    t=time.perf_counter();ev=evaluator(m,x,y,args.engine);setup_seconds=time.perf_counter()-t
    t=time.perf_counter()
    for _ in range(3):
        r,_=epoch(m,x,y,cfg,g,scale,bg,ev,args.engine);ev.accepted(m,r[0],r[1]);m.weights.copy_(r[0]);scale=r[3]
    warmup=time.perf_counter()-t
    m=make_model(args.seed,'trained');g,bg=state_generators(args.seed);scale=.02;ev.reset_model(m)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();rss=rss_key('VmRSS')
    reset=False
    try:Path('/proc/self/clear_refs').write_text('5');reset=True
    except OSError:pass
    records=[];allloss=[];allstats=[];selected=[]
    for step in range(1,101):
        t=time.perf_counter();r,timing=epoch(m,x,y,cfg,g,scale,bg,ev,args.engine)
        fired=action(m.weights,r[0]);ev.accepted(m,r[0],r[1]);m.weights.copy_(r[0]);scale=r[3];seconds=time.perf_counter()-t
        records.append(dict(step=step,seconds=seconds,action=fired,scale=scale,fires=r[2]['fires'],**timing))
        allloss.append(ev.last_losses.numpy().copy());selected.append(r[1].numpy().copy());allstats.append(dict(step=step,**r[2]))
    out=ROOT/'benchmarks'/f'seed{args.seed}-{args.engine}';write_csv(out/'intervals.csv',records);write_csv(out/'metrics.csv',allstats)
    values=np.stack(allloss);np.save(out/'candidate_losses.npy',values);np.save(out/'abs_y.npy',np.abs(values[:,::2]-values[:,1::2]));np.save(out/'selected_indices.npy',np.stack(selected))
    torch.save(dict(weights=m.weights,scale=scale,generator=g.get_state(),batch_generator=bg.get_state()),out/'final.pt')
    summary=dict(seed=args.seed,engine=args.engine,intervals=100,seconds=sum(r['seconds'] for r in records),seconds_per_interval=statistics.mean(r['seconds'] for r in records),gpu_workflow_milliseconds_mean=statistics.mean(r['gpu_workflow_milliseconds'] for r in records),schedule_seconds_mean=statistics.mean(r['schedule_seconds'] for r in records),setup_seconds=setup_seconds,warmup_seconds=warmup,final_weights_sha256=hashlib.sha256(m.weights.numpy().tobytes()).hexdigest(),final_scale=scale,generator_sha256=hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest(),batch_generator_sha256=hashlib.sha256(bg.get_state().numpy().tobytes()).hexdigest(),gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),rss_before=rss,peak_rss=rss_key('VmHWM'),rss_peak_reset=reset)
    dump(out/'summary.json',summary);print(f'{args.engine} seed{args.seed}: {summary["seconds_per_interval"]*1000:.3f} ms/interval',flush=True)


def analyze():
    raw=[json.loads(p.read_text()) for p in sorted((ROOT/'benchmarks').glob('*/summary.json'))];val=json.loads((ROOT/'validation.json').read_text())
    modes=val['active_conditions'];assert len(raw)==len(modes)*3
    comparison=[]
    for seed in range(3):
        b=ROOT/'benchmarks'/f'seed{seed}-gpu_graph';bs=json.loads((b/'summary.json').read_text());br=read(b/'intervals.csv');bl=np.load(b/'candidate_losses.npy')
        for mode in modes:
            d=ROOT/'benchmarks'/f'seed{seed}-{mode}';s=json.loads((d/'summary.json').read_text());rs=read(d/'intervals.csv');ls=np.load(d/'candidate_losses.npy')
            diff=[int(a['step']) for a,z in zip(rs,br) if a['action']!=z['action']];numerical=np.flatnonzero(np.any(ls.view(np.int32)!=bl.view(np.int32),axis=1))
            same_rng=all(s[k]==bs[k] for k in ['generator_sha256','batch_generator_sha256']);assert same_rng
            row=dict(seed=seed,engine=mode,first_action_divergence=diff[0] if diff else None,action_difference_intervals=len(diff),first_loss_bit_difference=int(numerical[0]+1) if len(numerical) else None,loss_bit_mismatch_count=int((ls.view(np.int32)!=bl.view(np.int32)).sum()),final_weights_equal=s['final_weights_sha256']==bs['final_weights_sha256'],final_scale_equal=s['final_scale']==bs['final_scale'],rng_equal=same_rng)
            if mode in EXACT:assert not diff and row['loss_bit_mismatch_count']==0 and row['final_weights_equal'] and row['final_scale_equal'],row
            comparison.append(row)
    baseline=statistics.mean(r['seconds_per_interval'] for r in raw if r['engine']=='gpu_graph');agg=[]
    for mode in modes:
        rs=[r for r in raw if r['engine']==mode];times=[r['seconds_per_interval'] for r in rs]
        agg.append(dict(engine=mode,seconds_per_interval_mean=statistics.mean(times),seconds_per_interval_sample_sd=statistics.stdev(times),speedup_vs_original_graph=baseline/statistics.mean(times),schedule_ms_mean=statistics.mean(r['schedule_seconds_mean']*1000 for r in rs),gpu_workflow_ms_mean=statistics.mean(r['gpu_workflow_milliseconds_mean'] for r in rs),setup_seconds_mean=statistics.mean(r['setup_seconds'] for r in rs),gpu_reserved_mib_max=max(r['gpu_peak_reserved_bytes']/2**20 for r in rs),gpu_allocated_mib_max=max(r['gpu_peak_allocated_bytes']/2**20 for r in rs)))
    write_csv(ROOT/'per_seed.csv',raw);write_csv(ROOT/'aggregate.csv',agg);write_csv(ROOT/'trajectory_comparison.csv',comparison)
    lines=['# CUDA Graph追加最適化：16残差ブロック','', 'RTX5090、幅76、34行列、192,432三値重み、A8/ReLU。CPU threads1・affinity15。各条件3 seed×100更新区間、保存済みE18a重みから開始。各変更は新たに測定した元gpu_graphから独立。CPU整理は提案1〜3をまとめた条件。組合せやtest評価は行っていない。','', '| 条件 | ms/区間 平均±標本SD | 基準Graph比 | CPU schedule ms | GPU処理 ms | 予約MiB |','|---|---:|---:|---:|---:|---:|']
    for r in agg:lines.append(f'| {r["engine"]} | {r["seconds_per_interval_mean"]*1000:.3f} ± {r["seconds_per_interval_sample_sd"]*1000:.3f} | {r["speedup_vs_original_graph"]:.3f}倍 | {r["schedule_ms_mean"]:.3f} | {r["gpu_workflow_ms_mean"]:.3f} | {r["gpu_reserved_mib_max"]:.1f} |')
    lines+=['','区間時間はCPUの候補生成・判定、転送、GPU計算、受理通知、CPUモデル更新を含む。初期化・コンパイル・Graph捕捉・3区間ウォームアップ・ログのディスク書込みは除外し別記録。GPU処理のCUDAイベント区間は転送後からloss計算まで。transfer_buffersだけ受理座標更新をGraph内部へ含めており、GPU内訳の境界差を全区間時間と混同しない。','', '## 数値一致性', '',f'固定共通状態での検査：初期/学習済み×3 seed×3連続更新、{val["loss_comparisons"]}候補損失比較。構造・乱数の一致：{val["structural_checks_passed"]}。完全一致を期待したCPU整理・候補再利用・転送整理の合格：{val["exact_expected_passed"]}。融合版の相対損失<1e-5かつ投票/カウンタ/発火一致：{val["fused_numerical_passed"]}。','', '| seed | 条件 | 最初の発火分岐 | 損失ビット不一致数 | 最終重み一致 | 最終S一致 |','|---|---|---:|---:|---|---|']
    for r in comparison:lines.append(f'| {r["seed"]} | {r["engine"]} | {r["first_action_divergence"]} | {r["loss_bit_mismatch_count"]} | {r["final_weights_equal"]} | {r["final_scale_equal"]} |')
    lines+=['','固定状態検査は基準Graphの重みに揃えて比較する。100区間測定は各条件が自分の損失で更新するため、分岐後の損失差にはモデル差も含まれる。診断だけの丸め差はvalidation/diagnostic_differences.jsonに別記する。融合による数値差をCPU整理の完全一致結果と混同しない。','', '## 実装と限界', '', 'cpu_compactは元candidate_pairを16座標配列に適用し、元accumulateに保存した一様乱数を供給、元select_actionsを使う。カウンタ診断は全体座標順に揃え、未選択ゼロの分母を復元する。persistent_candidatesはGPU候補の前回座標を現在baseへ戻し、新候補を直接代入する。transfer_buffersはpinned固定バッファ、受理座標の直接通知、イベント再利用、1回のmetadata H2Dとloss D2H。fused_graphはInductorをfullgraphで使用し、内部自動CUDA Graphを無効にして手動捕捉する。', '', 'GPU予約量はPyTorchプールであり、CUDAコンテキストを含むプロセス全VRAMではない。Graph割当カウンタのみから実際の生存テンソルピークを断定しない。過去の測定との時間差は環境・負荷が異なり得るため、倍率には今回の基準Graphを使う。3 seedの短い動作測定で、最終test精度や普遍的な速度保証は主張しない。すべての失敗・遅い条件も報告する。']
    if val['failed_conditions']:lines+=['',json.dumps(val['failed_conditions'],ensure_ascii=False,indent=2)]
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')
    dump(ROOT/'status.json',dict(complete=True,completed=len(raw),failed_conditions=val['failed_conditions'],test_evaluated=False,audited=False))
    print('\n'.join(lines[:12]),flush=True)


def run_all():
    ROOT.mkdir(parents=True,exist_ok=True);assert not (ROOT/'sources').exists(),'Existing run: do not overwrite'
    init();names=['gpu_graph_optimizations.py','benchmark_gpu_graph_optimizations.py','GPU_GRAPH_OPTIMIZATION_PROTOCOL.md','gpu_evaluation_engines.py','train.py','allocation_engines.py','benchmark_allocation_engines.py','benchmark_gpu_engines.py','run_residual_e17.py','residual_stream.py','residual_followup_models.py','activation_quantization.py']
    if (HERE/'audit_gpu_graph_optimizations.py').exists():names.append('audit_gpu_graph_optimizations.py')
    (ROOT/'sources').mkdir()
    for name in names:shutil.copy2(HERE/name,ROOT/'sources'/name)
    old=HERE/'results/residual-followups-e18-e20-20260908'
    dump(ROOT/'manifest.json',dict(preregistration_commit='5e27fdc',sources={n:sha(ROOT/'sources'/n) for n in names},trained_models={str(s):sha(old/'per_seed'/f'E18a-seed{s}'/'model.pt') for s in range(3)},data={p.name:sha(p) for p in (HERE/'data/MNIST/raw').glob('*-ubyte')}))
    cfg=config(0,HERE/'data');dump(ROOT/'config.json',dict(base_config={k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()},conditions=MODES,seeds=[0,1,2],blocks=16,width=76,weights=192432,measured_intervals=100,cpu_threads=1,cpu_affinity=[15],independent_ablations=True,TF32=False,test_evaluated=False))
    dump(ROOT/'environment.json',dict(torch_version=torch.__version__,torch_cuda=torch.version.cuda,nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],compile_threads=os.environ['TORCHINDUCTOR_COMPILE_THREADS']))
    dump(ROOT/'runtime_workers.json',dict(sequential=True,cpu_affinity=[15],processes=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True),order=[MODES[s:]+MODES[:s] for s in range(3)]))
    val=validate();modes=val['active_conditions'];done=0
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for seed in range(3):
        for mode in modes[seed:]+modes[:seed]:
            dump(ROOT/'status.json',dict(complete=False,completed=done,expected=len(modes)*3,active=dict(seed=seed,engine=mode)))
            with (ROOT/f'seed{seed}-{mode}.log').open('w') as log:
                subprocess.run([sys.executable,__file__,'worker','--engine',mode,'--seed',str(seed)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            s=json.loads((ROOT/'benchmarks'/f'seed{seed}-{mode}'/'summary.json').read_text());print(f'{mode} seed{seed}: {s["seconds_per_interval"]*1000:.3f}ms',flush=True);done+=1
    analyze()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','validate','worker','analyze']);p.add_argument('--engine',choices=MODES,default='gpu_graph');p.add_argument('--seed',type=int,default=0);a=p.parse_args()
    if a.mode=='run':run_all()
    elif a.mode=='validate':validate()
    elif a.mode=='worker':worker(a)
    else:analyze()
