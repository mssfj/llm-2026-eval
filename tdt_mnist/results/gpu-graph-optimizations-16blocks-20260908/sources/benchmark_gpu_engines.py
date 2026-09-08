"""16 residual blocks: CPU-cache vs GPU sequential/batched/Graphs,3x100."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,json,hashlib,subprocess,sys,time,statistics,shutil,gc
from pathlib import Path
import torch
import train
from gpu_evaluation_engines import GPUEvaluator,epoch as gpu_epoch,schedule,configure_gpu
from allocation_engines import epoch as cpu_epoch
from benchmark_allocation_engines import make_model,state_generators
from run_residual_e17 import setup,config,load_data,dump,write_csv,sha
HERE=Path(__file__).resolve().parent
ROOT=HERE/'results/gpu-evaluation-16blocks-20260908'
ENGINES=['cpu_restore_cache','gpu_sequential','gpu_batched','gpu_graph']
GPU_MODES=ENGINES[1:]

def setup_all():setup();configure_gpu()
def read(p):return list(csv.DictReader(p.open()))
def action(before,after):return ';'.join(f'{int(i)}:{int(after[i])}' for i in (before!=after).nonzero().flatten())
def rss_key(key):
    for l in Path('/proc/self/status').read_text().splitlines():
        if l.startswith(key+':'):return int(l.split()[1])*1024

def evaluate_epoch(m,x,y,a,g,s,bg,evaluator=None,trace=None):
    if evaluator is None:return cpu_epoch(m,x,y,a,g,s,bg,engine='restore_cache',trace=trace),dict(gpu_workflow_milliseconds=0.,schedule_seconds=0.)
    return gpu_epoch(m,x,y,a,g,s,bg,evaluator,trace=trace)


def validation():
    setup_all();a=config(0,HERE/'data');(x,y),_,_=load_data(a,torch.device('cpu'))
    rows=[];loss_rows=[];mismatch_rows=[];graph_rows=[];algebra=[]
    for state in ['initial','trained']:
        for seed in range(3):
            m=make_model(seed,state);g,bg=state_generators(seed);ct={}
            ref,_=evaluate_epoch(m,x,y,a,g,.02,bg,trace=ct);cpu=torch.stack(ct['losses']);reference_g=g.get_state();reference_bg=bg.get_state()
            eager=None;grapher=None
            for mode in GPU_MODES:
                ev=GPUEvaluator(m,x,y,mode);eg,ebg=state_generators(seed);trace={}
                r,_=evaluate_epoch(m,x,y,a,eg,.02,ebg,evaluator=ev,trace=trace)
                losses=torch.stack(trace['losses']);rel=(losses-cpu).abs()/cpu.abs().clamp_min(1e-30)
                votes=torch.stack(trace['votes']);cv=torch.stack(ct['votes']);counts=torch.stack(trace['counters']);cc=torch.stack(ct['counters'])
                row=dict(state=state,seed=seed,engine=mode,max_relative_loss_error=float(rel.max()),relative_error_failures=int((rel>=1e-5).sum()),bitwise_loss_mismatches=int((losses.view(torch.int32)!=cpu.view(torch.int32)).sum()),vote_mismatches=int((votes!=cv).sum()),counter_mismatches=int((counts!=cc).sum()),proposal_equal=torch.equal(r[0],ref[0]),scale_equal=r[3]==ref[3],indices_equal=torch.equal(r[1],ref[1]),rng_equal=torch.equal(eg.get_state(),reference_g) and torch.equal(ebg.get_state(),reference_bg))
                assert row['indices_equal'] and row['rng_equal'];rows.append(row)
                for p in range(128):loss_rows.append(dict(state=state,seed=seed,engine=mode,candidate=p,pair=p//2,orientation='plus' if p%2==0 else 'minus',cpu_loss=float(cpu[p]),gpu_loss=float(losses[p]),relative_error=float(rel[p])))
                for pair,coord,edge in (votes!=cv).nonzero().tolist():mismatch_rows.append(dict(state=state,seed=seed,engine=mode,kind='vote',pair=pair,coordinate=int(ref[1][coord]),edge=edge,cpu_value=int(cv[pair,coord,edge]),gpu_value=int(votes[pair,coord,edge])))
                # Record every per-pair counter discrepancy, not only final counts.
                for pair,coord,edge in (counts!=cc).nonzero().tolist():mismatch_rows.append(dict(state=state,seed=seed,engine=mode,kind='counter',pair=pair,coordinate=int(ref[1][coord]),edge=edge,cpu_value=int(cc[pair,coord,edge]),gpu_value=int(counts[pair,coord,edge])))
                if not row['proposal_equal']:mismatch_rows.append(dict(state=state,seed=seed,engine=mode,kind='fire',pair='',coordinate='',edge='',cpu_value=action(m.weights,ref[0]),gpu_value=action(m.weights,r[0])))
                if mode=='gpu_batched':eager=ev
                elif mode=='gpu_graph':grapher=ev
                else:del ev
            # Changed base weights, batches and candidate coordinates must refresh the Graph.
            m.weights.copy_(ref[0]);plan=schedule(m,x,a,g,bg)
            v1,_=eager.evaluate(m,plan);v2,_=grapher.evaluate(m,plan)
            graph_rows.append(dict(state=state,seed=seed,changed_base_coordinates=len((grapher.cpu_weights!=make_model(seed,state).weights).nonzero()),changed_indices=not torch.equal(plan.indices,ref[1]),bitwise_mismatches=int((v1.view(torch.int32)!=v2.view(torch.int32)).sum()),max_relative_error=float(((v1-v2).abs()/v1.abs()).max())))
            assert torch.equal(v1,v2),'Captured graph differs from identical eager batched workflow'
            del eager,grapher,ev;gc.collect();torch.cuda.empty_cache()
    # No-A8 controls at a fixed saved-trained model, same candidates, no training/test.
    m=make_model(0,'trained');m.activation_precision='a32';g,bg=state_generators(0);ct={};ref,_=evaluate_epoch(m,x,y,a,g,.02,bg,trace=ct);cpu=torch.stack(ct['losses'])
    for mode in GPU_MODES:
        ev=GPUEvaluator(m,x,y,mode,precision='a32');eg,ebg=state_generators(0);t={};evaluate_epoch(m,x,y,a,eg,.02,ebg,evaluator=ev,trace=t);v=torch.stack(t['losses'])
        algebra.append(dict(engine=mode,max_relative_loss_error=float(((v-cpu).abs()/cpu.abs()).max())))
        del ev;gc.collect();torch.cuda.empty_cache()
    write_csv(ROOT/'validation/summary.csv',rows);write_csv(ROOT/'validation/losses.csv',loss_rows)
    if mismatch_rows:write_csv(ROOT/'validation/mismatches.csv',mismatch_rows)
    write_csv(ROOT/'validation/graph_input_refresh.csv',graph_rows);write_csv(ROOT/'validation/a32_controls.csv',algebra)
    report=dict(structural_checks_passed=True,numerical_acceptance_passed=all(r['relative_error_failures']==0 and r['vote_mismatches']==0 and r['counter_mismatches']==0 and r['proposal_equal'] for r in rows),max_relative_loss_error=max(r['max_relative_loss_error'] for r in rows),loss_comparisons=len(loss_rows),rng_all_equal=all(r['rng_equal'] for r in rows),graph_bitwise_equal_all_refresh_cases=all(r['bitwise_mismatches']==0 for r in graph_rows),test_evaluated=False)
    dump(ROOT/'validation.json',report);print(json.dumps(report),flush=True)


def worker(a):
    setup_all();os.sched_setaffinity(0,{a.cpu});cfg=config(a.seed,HERE/'data');(x,y),_,_=load_data(cfg,torch.device('cpu'))
    m=make_model(a.seed,'trained');g,bg=state_generators(a.seed);s=.02
    setup_started=time.perf_counter();ev=None if a.engine=='cpu_restore_cache' else GPUEvaluator(m,x,y,a.engine)
    setup_seconds=time.perf_counter()-setup_started
    warm_started=time.perf_counter()
    for _ in range(3):
        r,_=evaluate_epoch(m,x,y,cfg,g,s,bg,evaluator=ev);m.weights.copy_(r[0]);s=r[3]
    warm_seconds=time.perf_counter()-warm_started
    m=make_model(a.seed,'trained');g,bg=state_generators(a.seed);s=.02
    if ev is not None:
        ev.sync_weights(m);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    before_rss=rss_key('VmRSS');reset=False
    try:Path('/proc/self/clear_refs').write_text('5');reset=True
    except OSError:pass
    rows=[];started=time.perf_counter()
    for step in range(1,101):
        t=time.perf_counter();r,timing=evaluate_epoch(m,x,y,cfg,g,s,bg,evaluator=ev);fires=action(m.weights,r[0]);m.weights.copy_(r[0]);s=r[3]
        rows.append(dict(step=step,seconds=time.perf_counter()-t,action=fires,scale=s,fires=r[2]['fires'],**timing))
        if step%25==0:dump(ROOT/'worker_progress.json',dict(engine=a.engine,seed=a.seed,step=step,intervals=100))
    total=time.perf_counter()-started
    out=ROOT/'benchmarks'/f'seed{a.seed}-{a.engine}';write_csv(out/'intervals.csv',rows)
    summary=dict(seed=a.seed,engine=a.engine,blocks=16,width=76,matrices=34,weights=m.num_params,intervals=100,seconds=total,seconds_per_interval=total/100,setup_seconds=setup_seconds,warmup_seconds=warm_seconds,gpu_workflow_milliseconds_mean=statistics.mean(r['gpu_workflow_milliseconds'] for r in rows),schedule_seconds_mean=statistics.mean(r['schedule_seconds'] for r in rows),gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if ev else 0,gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved() if ev else 0,rss_before=before_rss,peak_rss=rss_key('VmHWM'),rss_peak_reset=reset,cpu_affinity=sorted(os.sched_getaffinity(0)),cpu_threads=1,final_weights_sha256=hashlib.sha256(m.weights.numpy().tobytes()).hexdigest(),final_scale=s,generator_sha256=hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest(),batch_generator_sha256=hashlib.sha256(bg.get_state().numpy().tobytes()).hexdigest())
    dump(out/'summary.json',summary);print(f'{a.engine} seed{a.seed} {total/100:.6f} sec/interval',flush=True)


def analyze():
    raw=[json.loads(p.read_text()) for p in sorted((ROOT/'benchmarks').glob('*/summary.json'))];assert len(raw)==12
    divergence=[]
    for seed in range(3):
        n=next(r for r in raw if r['seed']==seed and r['engine']=='cpu_restore_cache');ni=read(ROOT/'benchmarks'/f'seed{seed}-cpu_restore_cache/intervals.csv')
        for mode in GPU_MODES:
            r=next(r for r in raw if r['seed']==seed and r['engine']==mode);ri=read(ROOT/'benchmarks'/f'seed{seed}-{mode}/intervals.csv');assert len(ri)==len(ni)==100
            assert all(r[k]==n[k] for k in ['generator_sha256','batch_generator_sha256'])
            dif=[int(a['step']) for a,b in zip(ni,ri) if a['action']!=b['action']]
            divergence.append(dict(seed=seed,engine=mode,first_observed_action_difference=dif[0] if dif else None,action_difference_intervals=len(dif),final_weights_equal=r['final_weights_sha256']==n['final_weights_sha256'],rng_equal=True))
    write_csv(ROOT/'first_divergence.csv',divergence)
    agg=[];base=statistics.mean(r['seconds_per_interval'] for r in raw if r['engine']=='cpu_restore_cache')
    for mode in ENGINES:
        rs=[r for r in raw if r['engine']==mode];ts=[r['seconds_per_interval'] for r in rs]
        agg.append(dict(engine=mode,seconds_per_interval_mean=statistics.mean(ts),seconds_per_interval_sample_std=statistics.stdev(ts),speedup_vs_cpu_cache=base/statistics.mean(ts),gpu_workflow_milliseconds_mean=statistics.mean(r['gpu_workflow_milliseconds_mean'] for r in rs),gpu_peak_allocated_mib_max=max(r['gpu_peak_allocated_bytes']/2**20 for r in rs),gpu_peak_reserved_mib_max=max(r['gpu_peak_reserved_bytes']/2**20 for r in rs)))
    write_csv(ROOT/'aggregate.csv',agg);write_csv(ROOT/'per_seed.csv',[{**r,'cpu_affinity':json.dumps(r['cpu_affinity'])} for r in raw])
    val=read(ROOT/'validation/summary.csv');validation=json.loads((ROOT/'validation.json').read_text())
    lines=['# GPUによるTDT候補評価：16残差ブロック','', 'RTX5090、CPU threads=1。幅76・34行列・192,432三値重み、A8＋ReLU。先の単独最適化と同じ3 seed×100区間。CPU復元キャッシュ、GPU逐次、GPU候補並列、GPU並列＋CUDA Graphsの4条件を各seedの保存済みE18a重みから測定。test評価なし。','', '| 条件 | 秒/区間 平均±標本SD | CPUキャッシュ比 | GPU処理区間 ms | GPU最大割当 MiB |','|---|---:|---:|---:|---:|']
    for r in agg:lines.append(f'| {r["engine"]} | {r["seconds_per_interval_mean"]:.6f} ± {r["seconds_per_interval_sample_std"]:.6f} | {r["speedup_vs_cpu_cache"]:.3f}倍 | {r["gpu_workflow_milliseconds_mean"]:.3f} | {r["gpu_peak_allocated_mib_max"]:.1f} |')
    lines+=['', '時間は候補スケジュールのCPU生成、GPUへのメタデータ転送・受理重み更新、128候補評価、損失一括転送、元のCPU判定、重み更新を含む。GPU処理区間はCUDAイベントの経過で、CPUカーネル投入による空白時間も含む。初期転送・CUDA Graph構築・3区間ウォームアップは除外し、seed別setup_seconds/warmup_secondsへ別記。','', 'GPU上では三値コードと復元FP32重みを常駐させ、候補の有効行列をGPU上で構成。低ランク出力補正・CPU再評価による数値補完は使わない。候補別の128例を維持し、GPU逐次は128forward、候補並列は[candidate128,sample128,feature]のBMM。CUDA Graphsは同じ並列処理を捕捉する。','', '## 数値一致性','',f'CPU基準の最大相対損失誤差：{validation["max_relative_loss_error"]:.9g}。相対誤差<1e-5かつ投票・カウンタ・発火完全一致の基準：{validation["numerical_acceptance_passed"]}。数値基準を満たさない条件の高速化を、既存CPU結果をそのまま再現する高速化とは解釈しない。','', '| 条件 | 最大相対誤差 | 誤差基準超過候補 | 投票不一致 | カウンタ不一致（全時点） | 発火不一致ケース |','|---|---:|---:|---:|---:|---:|']
    for mode in GPU_MODES:
        rs=[r for r in val if r['engine']==mode]
        lines.append(f'| {mode} | {max(float(r["max_relative_loss_error"]) for r in rs):.9g} | {sum(int(r["relative_error_failures"]) for r in rs)} | {sum(int(r["vote_mismatches"]) for r in rs)} | {sum(int(r["counter_mismatches"]) for r in rs)} | {sum(r["proposal_equal"]!="True" for r in rs)} |')
    lines+=['', '固定重み検査は初期/学習済み×3 seed×128候補。候補・ミニバッチ・両CPU乱数系列は全条件一致。グラフ入力の更新（重み・バッチ・候補座標）後もeager並列とGraphの128損失は全6ケースでビット一致した。A32対照はvalidation/a32_controls.csv。GPU行列積とCPU行列積のFP32順序差、それがA8丸めを通じて増幅する可能性を区別するための診断であり、任意入力での同一性の証明ではない。','', '| seed | GPU条件 | CPUから最初に発火が異なった区間 | 100区間中の発火相違数 | 最終重み一致 |','|---|---|---:|---:|---|']
    for r in divergence:lines.append(f'| {r["seed"]} | {r["engine"]} | {r["first_observed_action_difference"]} | {r["action_difference_intervals"]} | {r["final_weights_equal"]} |')
    lines+=['', '各条件は自分の損失・発火結果で100区間を更新するため、分岐後は重み状態も異なる。数値変化による学習結果の違いは100区間だけでは判断できず、test精度の再現を主張しない。','', 'TF32無効・FP32・決定的アルゴリズム、CUBLAS_WORKSPACE_CONFIG=:4096:8。CPUの確率的投票・C8・閾値・タイブレーク・S更新は元のepochコードをprivate globalsで再利用。先行するCPU再現ジョブとの資源共有あり。CPU affinity15に固定し、条件は逐次実行・seedごとに順序を循環。GPU起動時の他プロセスと環境情報はruntime_workers.json/environment.json。','', '全候補損失、投票/カウンタ/発火の不一致内容、100区間ログ、CPU/GPUメモリ、設定、実行時ソース、SHA-256を保存した。速い条件だけを選んだ追加学習・test評価・精度条件探索は行っていない。']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')
    dump(ROOT/'status.json',dict(complete=True,completed=12,numerical_acceptance_passed=validation['numerical_acceptance_passed'],test_evaluated=False))
    dump(ROOT/'artifacts_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='artifacts_sha256.json'})
    print('\n'.join(lines[:12]),flush=True)


def run_all(a):
    ROOT.mkdir(exist_ok=True,parents=True);assert not (ROOT/'sources').exists()
    names=['gpu_evaluation_engines.py','benchmark_gpu_engines.py','allocation_engines.py','benchmark_allocation_engines.py','GPU_ENGINE_BENCHMARK_PROTOCOL.md','train.py','residual_followup_models.py','residual_stream.py','activation_quantization.py','run_residual_e17.py']
    (ROOT/'sources').mkdir()
    for name in names:shutil.copy2(HERE/name,ROOT/'sources'/name)
    old=HERE/'results/residual-followups-e18-e20-20260908'
    dump(ROOT/'manifest.json',dict(sources={n:sha(ROOT/'sources'/n) for n in names},preregistration_commit='981d584',trained_models={str(s):sha(old/'per_seed'/f'E18a-seed{s}'/'model.pt') for s in range(3)},data={p.name:sha(p) for p in (HERE/'data/MNIST/raw').glob('*-ubyte')}))
    cfg=config(0,HERE/'data');dump(ROOT/'config.json',dict(legacy_config={k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()},measured_intervals=100,seeds=[0,1,2],conditions=ENGINES,blocks=16,width=76,matrices=34,weights=192432,cpu_threads=1,cpu_affinity=[a.cpu],fp32_precision='ieee',TF32=False,test_evaluated=False))
    setup_all();props=torch.cuda.get_device_properties(0)
    dump(ROOT/'environment.json',dict(torch_version=torch.__version__,cuda_version=torch.version.cuda,gpu_name=props.name,gpu_memory=props.total_memory,compute_capability=[props.major,props.minor],nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG']))
    context=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True)
    dump(ROOT/'runtime_workers.json',dict(sequential=True,cpu_affinity=[a.cpu],existing_jobs=[l for l in context.splitlines() if 'run_fast_engine.py' in l or 'finish_fast_engine_report.py' in l],order=[ENGINES[s:]+ENGINES[:s] for s in range(3)]))
    validation()
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8');done=0
    for seed in range(3):
        for mode in ENGINES[seed:]+ENGINES[:seed]:
            dump(ROOT/'status.json',dict(complete=False,completed=done,expected=12,active=dict(seed=seed,engine=mode)))
            subprocess.run([sys.executable,__file__,'worker','--engine',mode,'--seed',str(seed),'--cpu',str(a.cpu)],env=env,check=True);done+=1
    analyze()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','validate','worker','analyze']);p.add_argument('--engine',choices=ENGINES,default=ENGINES[0]);p.add_argument('--seed',type=int,default=0);p.add_argument('--cpu',type=int,default=15);a=p.parse_args()
    if a.mode=='run':run_all(a)
    elif a.mode=='validate':validation()
    elif a.mode=='worker':worker(a)
    else:analyze()
