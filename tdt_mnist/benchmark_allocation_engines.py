"""Separate restoration/copy ablations, only E18a's16 residual blocks."""
import argparse,csv,json,os,sys,subprocess,time,resource,statistics,hashlib,shutil
from pathlib import Path
import torch
from allocation_engines import epoch
from residual_followup_models import ResidualTDT
from run_residual_e17 import config,setup,load_data,dump,write_csv,sha
HERE=Path(__file__).resolve().parent
ROOT=HERE/'results/allocation-ablations-16blocks-20260908'
OLD=HERE/'results/residual-followups-e18-e20-20260908'
ENGINES=['naive','restore_cache','candidate_buffers']

def make_model(seed,state):
    m=ResidualTDT(seed=seed,blocks=16,width=76)
    if state=='trained':
        p=OLD/'per_seed'/f'E18a-seed{seed}'/'model.pt'
        w=torch.load(p,weights_only=False);m.weights.copy_(w['weights'] if isinstance(w,dict) else w)
    return m

def state_generators(seed):return torch.Generator().manual_seed(seed+1),torch.Generator().manual_seed(seed+100000)

def bits(t):return t.view(torch.int32) if t.dtype==torch.float32 else t

def validation():
    setup();a=config(0,HERE/'data');(x,y),_,_=load_data(a,torch.device('cpu'))
    rows=[]
    for state in ['initial','trained']:
        for seed in range(3):
            models={e:make_model(seed,state) for e in ENGINES};gs={e:state_generators(seed) for e in ENGINES};scales={e:.02 for e in ENGINES}
            for step in range(1,4):
                results={};traces={}
                for e in ENGINES:
                    trace={};before=models[e].weights.clone();calls=models[e].forward_calls
                    result=epoch(models[e],x,y,a,*gs[e][:1],scales[e],gs[e][1],engine=e,trace=trace)
                    assert torch.equal(before,models[e].weights),e
                    assert models[e].forward_calls-calls==128,e
                    results[e]=result;traces[e]=trace
                ref=results['naive'];nt=traces['naive']
                for e in ENGINES[1:]:
                    r=results[e];t=traces[e]
                    nl=torch.stack(nt['losses']);el=torch.stack(t['losses'])
                    row=dict(state=state,seed=seed,step=step,engine=e,loss_count=128,bitwise_loss_mismatches=int((bits(nl)!=bits(el)).sum()),max_relative_loss_error=float(((nl-el).abs()/nl.abs()).max()),vote_mismatches=int((torch.stack(nt['votes'])!=torch.stack(t['votes'])).sum()),counter_mismatches=int((torch.stack(nt['counters'])!=torch.stack(t['counters'])).sum()),proposal_equal=torch.equal(ref[0],r[0]),indices_equal=torch.equal(ref[1],r[1]),stats_equal=ref[2]==r[2],scale_equal=ref[3]==r[3],rng_equal=all(torch.equal(g.get_state(),n.get_state()) for g,n in zip(gs[e],gs['naive'])))
                    rows.append(row)
                for e in ENGINES:models[e].weights.copy_(results[e][0]);scales[e]=results[e][3]
    write_csv(ROOT/'validation.csv',rows)
    passed=all(r['bitwise_loss_mismatches']==r['vote_mismatches']==r['counter_mismatches']==0 and all(r[k] for k in ['proposal_equal','indices_equal','stats_equal','scale_equal','rng_equal']) for r in rows)
    dump(ROOT/'validation.json',dict(passed=passed,rows=len(rows),loss_comparisons=sum(r['loss_count'] for r in rows),max_relative_loss_error=max(r['max_relative_loss_error'] for r in rows),blocks=16,linear_matrices=34,weights=192432,seeds=[0,1,2],states=['initial','trained'],intervals_per_state_seed=3,test_evaluated=False))
    assert passed,rows
    print('Exact equality validation passed',flush=True)

def worker(args):
    setup()
    if args.cpu is not None:os.sched_setaffinity(0,{args.cpu})
    a=config(args.seed,HERE/'data');(x,y),_,_=load_data(a,torch.device('cpu'))
    m=make_model(args.seed,'trained');g,bg=state_generators(args.seed);s=.02
    # Warmup then reconstruct exactly the same starting state and RNG.
    for _ in range(3):
        w,_,_,s=epoch(m,x,y,a,g,s,bg,engine=args.engine);m.weights.copy_(w)
    m=make_model(args.seed,'trained');g,bg=state_generators(args.seed);s=.02
    def rss():
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    baseline=rss();peak_reset=False
    try:Path('/proc/self/clear_refs').write_text('5');peak_reset=True
    except OSError:pass
    rows=[];started=time.perf_counter()
    for step in range(1,101):
        t=time.perf_counter();w,idx,stats,s=epoch(m,x,y,a,g,s,bg,engine=args.engine);m.weights.copy_(w)
        rows.append(dict(step=step,seconds=time.perf_counter()-t,fires=stats['fires'],scale=s))
    elapsed=time.perf_counter()-started
    out=ROOT/'benchmarks'/f'seed{args.seed}-{args.engine}';write_csv(out/'intervals.csv',rows)
    digest=hashlib.sha256(m.weights.numpy().tobytes()).hexdigest()
    # VmHWM is scoped to post-warmup when clear_refs5 is supported.
    status=Path('/proc/self/status').read_text().splitlines();hwm=next(int(l.split()[1])*1024 for l in status if l.startswith('VmHWM:'))
    dump(out/'summary.json',dict(seed=args.seed,engine=args.engine,blocks=16,linear_matrices=34,width=76,weights=m.num_params,intervals=100,seconds=elapsed,seconds_per_interval=elapsed/100,interval_seconds_sum=sum(r['seconds'] for r in rows),threads=1,cpu_affinity=sorted(os.sched_getaffinity(0)),rss_baseline=baseline,peak_rss=hwm,peak_rss_reset_after_warmup=peak_reset,rss_peak_increment=max(0,hwm-baseline),final_weights_sha256=digest,final_scale=s,generator_sha256=hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest(),batch_generator_sha256=hashlib.sha256(bg.get_state().numpy().tobytes()).hexdigest()))
    print(f'seed{args.seed} {args.engine}: {elapsed/100:.6f} sec/interval',flush=True)

def analyze():
    raw=[json.loads(p.read_text()) for p in sorted((ROOT/'benchmarks').glob('*/summary.json'))]
    assert len(raw)==9
    for seed in range(3):
        rows=[r for r in raw if r['seed']==seed]
        for key in ['final_weights_sha256','final_scale','generator_sha256','batch_generator_sha256']:assert len({str(r[key]) for r in rows})==1,(seed,key)
    aggregate=[];paired=[]
    for e in ENGINES:
        rows=[r for r in raw if r['engine']==e];times=[r['seconds_per_interval'] for r in rows]
        aggregate.append(dict(engine=e,seconds_per_interval_mean=statistics.mean(times),seconds_per_interval_sample_std=statistics.stdev(times),seconds_100_intervals_mean=statistics.mean(r['seconds'] for r in rows),peak_rss_mib_mean=statistics.mean(r['peak_rss']/2**20 for r in rows)))
        for r in rows:
            n=next(n for n in raw if n['seed']==r['seed'] and n['engine']=='naive')
            paired.append(dict(seed=r['seed'],engine=e,naive_seconds_per_interval=n['seconds_per_interval'],seconds_per_interval=r['seconds_per_interval'],speedup=n['seconds_per_interval']/r['seconds_per_interval'],time_reduction_percent=100*(1-r['seconds_per_interval']/n['seconds_per_interval'])))
    write_csv(ROOT/'per_seed.csv',[{**r,'cpu_affinity':json.dumps(r['cpu_affinity'])} for r in raw]);write_csv(ROOT/'aggregate.csv',aggregate);write_csv(ROOT/'paired_speedups.csv',paired)
    baseline=aggregate[0]['seconds_per_interval_mean']
    lines=['# 復元キャッシュと候補コピー削減の単独比較','', '対象はE18a相当の16残差ブロック・幅76・34行列・192,432三値重み。各条件CPU threads=1、A8＋ReLU。3 seedの保存済み学習済み重みから各100区間、計9測定。test評価・条件探索なし。','', '| 条件 | 秒/区間：平均 ± 標本SD | naive比速度 | 時間削減 |','|---|---:|---:|---:|']
    labels={'naive':'naive','restore_cache':'復元済み重みの再利用のみ','candidate_buffers':'候補全体コピー削減のみ'}
    for r in aggregate:
        t=r['seconds_per_interval_mean'];lines.append(f'| {labels[r["engine"]]} | {t:.6f} ± {r["seconds_per_interval_sample_std"]:.6f} | {baseline/t:.3f}倍 | {100*(1-t/baseline):+.2f}% |')
    lines+=['', '速度倍率は3 seedの平均区間時間同士の比。seed別対応比はpaired_speedups.csv。標準偏差にはseedによる状態の違いと測定揺らぎが含まれる。','', '復元キャッシュ版：区間開始時にFP32行列を復元し、候補ごとに選択された16座標を元と同じ三値コード×スケールで上書きする。候補の全体コピーはそのまま。復元行列は一時バッファであり、潜在学習重みではない。','', 'コピー削減版：区間開始時だけplus/minus用INT8バッファを2個コピーし、64候補対で再利用する。候補の乱数処理は元のcandidate_pairを利用し、ASTで全体cloneの2箇所だけをバッファ参照に置換する。各forwardでの全FP32復元はそのまま。','', '行列積、A8、損失、ミニバッチ、候補乱数、カウンタ、発火の計算順序は変更していない。低ランク補正・候補のバッチ化・naive再評価による補完は使用しない。両最適化を組み合わせた条件は測定していない。','', '一致性：初期/学習済み×3 seed×3区間、各最適化で2,304候補損失（計4,608比較）のFP32ビット一致、全投票・全時点カウンタ・発火・S・乱数系列を確認。加えて100区間性能測定後も全3条件で重み・S・両乱数状態が完全一致した。','', '全測定は逐次実行し、seedごとに条件順を循環（naive→cache→copy、cache→copy→naive、copy→naive→cache）。CPU affinityと並行中の既存再現ジョブをruntime_workers.jsonに記録。既存ジョブを停止・変更せず測定したため、共有資源による揺らぎがあり、専有マシンの速度保証ではない。','', 'RSSはウォームアップ後にVmHWMリセットを試み、成功の有無を記録。データ・Python/PyTorchとアロケータ保持を含む。区間数を増やした収束実験や新しいtest精度の評価は行わない。']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')
    dump(ROOT/'status.json',dict(complete=True,benchmarks=9,exact_validation_passed=True,test_evaluated=False))
    dump(ROOT/'artifacts_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='artifacts_sha256.json'})
    print('\n'.join(lines[:11]),flush=True)

def main(a):
    ROOT.mkdir(exist_ok=True,parents=True)
    if a.mode=='validate':return validation()
    if a.mode=='worker':return worker(a)
    if a.mode=='analyze':return analyze()
    assert not (ROOT/'sources').exists(),'Do not overwrite an existing experiment'
    names=['allocation_engines.py','benchmark_allocation_engines.py','train.py','residual_followup_models.py','residual_stream.py','activation_quantization.py','run_residual_e17.py']
    (ROOT/'sources').mkdir()
    for name in names:shutil.copy2(HERE/name,ROOT/'sources'/name)
    dump(ROOT/'manifest.json',dict(sources={n:sha(ROOT/'sources'/n) for n in names},blocks=16,matrices=34,width=76,weights=192432,seeds=[0,1,2],conditions=ENGINES,intervals=100,trained_models={str(s):sha(OLD/'per_seed'/f'E18a-seed{s}'/'model.pt') for s in range(3)},data={p.name:sha(p) for p in (HERE/'data/MNIST/raw').glob('*-ubyte')}))
    validation()
    context=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True)
    dump(ROOT/'runtime_workers.json',dict(sequential_benchmarks=True,threads_per_worker=1,cpu=a.cpu,concurrent_existing_jobs=[l for l in context.splitlines() if 'run_fast_engine.py' in l or 'finish_fast_engine_report.py' in l],order=[ENGINES[s:]+ENGINES[:s] for s in range(3)]))
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');done=0
    for seed in range(3):
        for e in ENGINES[seed:]+ENGINES[:seed]:
            cmd=[sys.executable,__file__,'worker','--engine',e,'--seed',str(seed)]
            if a.cpu is not None:cmd+=['--cpu',str(a.cpu)]
            dump(ROOT/'status.json',dict(complete=False,completed=done,expected=9,active=dict(seed=seed,engine=e)))
            subprocess.run(cmd,check=True,env=env);done+=1
    analyze()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','validate','worker','analyze']);p.add_argument('--engine',choices=ENGINES,default='naive');p.add_argument('--seed',type=int,default=0);p.add_argument('--cpu',type=int,default=None);main(p.parse_args())
