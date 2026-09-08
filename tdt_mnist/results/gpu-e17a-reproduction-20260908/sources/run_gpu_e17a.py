"""Three fresh full E17a runs with frozen GPU Graph candidate evaluation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,json,time,sys,subprocess,shutil,math,statistics,hashlib
from pathlib import Path
import numpy as np
import torch
import train
from gpu_evaluation_engines import GPUEvaluator,epoch as gpu_epoch,configure_gpu,schedule
from residual_followup_models import ResidualTDT
from residual_stream import ResidualStreamModel
from run_residual_e17 import config,setup,load_data,observe,probes,evaluate,dump,write_csv,sha
from depth_diagnostics import layer_events
HERE=Path(__file__).resolve().parent;ROOT=HERE/'results/gpu-e17a-reproduction-20260908';OLD=HERE/'results/residual-stream-a8-e17-20260908'

class RecordingEvaluator(GPUEvaluator):
    def evaluate(self,model,plan):
        result=super().evaluate(model,plan);self.last_losses=result[0];return result

def gens(seed):return torch.Generator().manual_seed(seed+1),torch.Generator().manual_seed(seed+100000)
def action(before,after):return ';'.join(f'{int(i)}:{int(after[i])}' for i in (before!=after).nonzero().flatten())
def read(p):return list(csv.DictReader(p.open()))
def init():setup();configure_gpu()

def preflight():
    init();a=config(0,HERE/'data');(x,y),(vx,vy),_=load_data(a,torch.device('cpu'));rows=[]
    expected=json.loads((OLD/'preflight.json').read_text())['initial_validation']
    for seed in range(3):
        m=ResidualTDT(seed,8,76,'a8');legacy=ResidualStreamModel(seed)
        assert m.num_params==100016 and len(m.shapes)==18
        assert torch.equal(m.weights,legacy.weights) and torch.equal(m(x[:128]),legacy(x[:128]))
        val=evaluate(m,vx,vy);old=next(r for r in expected if r['condition']=='E17a' and r['seed']==seed)
        assert val=={k:old[k] for k in ['loss','accuracy']}
        ev=GPUEvaluator(m,x,y,'gpu_graph');eager=GPUEvaluator(m,x,y,'gpu_batched');g,bg=gens(seed);plan=schedule(m,x,a,g,bg)
        v1,_=ev.evaluate(m,plan);v2,_=eager.evaluate(m,plan);assert torch.equal(v1,v2)
        rows.append(dict(seed=seed,initial_validation=val,initial_weights_equal=True,initial_forward_bitwise_equal=True,graph_eager_losses_bitwise_equal=True,num_params=m.num_params))
        del ev,eager;torch.cuda.empty_cache()
    dump(ROOT/'preflight.json',dict(passed=True,seeds=rows,test_evaluated=False));print('E17a 8-block preflight passed for all3 seeds',flush=True)


def worker(seed):
    init();os.sched_setaffinity(0,{15});a=config(seed,HERE/'data');m=ResidualTDT(seed,8,76,'a8')
    out=ROOT/'per_seed'/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    cfg.update(condition='GPU_E17a',engine='gpu_graph',blocks=8,width=76,num_params=100016,matrices=18,shapes=m.shapes,layer_scales=m.scales,activation_precision='a8',hidden_activation='relu',evaluation_device='cpu',candidate_device='cuda',source_manifest=json.loads((ROOT/'manifest.json').read_text()))
    dump(out/'config.json',cfg)
    (x,y),(vx,vy),(tx,ty)=load_data(a,m.device);g,bg=gens(seed);s=.02
    initial,signals,activation,ratios=observe(m,vx,vy,0)
    expected=next(r for r in json.loads((ROOT/'preflight.json').read_text())['seeds'] if r['seed']==seed)['initial_validation'];assert initial==expected
    dump(out/'initial_validation.json',initial)
    diagnostic_start=time.perf_counter();probe_rows=probes(m,x,y,seed,'initial');initial_probe_seconds=time.perf_counter()-diagnostic_start
    write_csv(out/'probes.csv',probe_rows)
    t=time.perf_counter();ev=RecordingEvaluator(m,x,y,'gpu_graph');setup_seconds=time.perf_counter()-t
    naive=ResidualStreamModel(seed);ng,nbg=gens(seed);ns=.02;first_numeric=None;first_div=None;reference_seconds=0;replayed=0
    historical=OLD/'per_seed'/f'E17a-seed{seed}';oldmetrics=read(historical/'metrics.csv');oldabs=np.load(historical/'abs_y.npy',mmap_mode='r')
    losses=np.lib.format.open_memmap(out/'candidate_losses.npy',mode='w+',dtype='float32',shape=(12000,128))
    abs_y=np.lib.format.open_memmap(out/'abs_y.npy',mode='w+',dtype='float32',shape=(12000,64))
    selected=np.lib.format.open_memmap(out/'selected_indices.npy',mode='w+',dtype='int32',shape=(12000,16))
    totals=[dict(fires=0,selected_intervals=0,fire_intervals=0,selected_coordinates=0) for _ in range(18)];hist={}
    started=time.perf_counter();engine_seconds=0.;gpu_milliseconds=0.;validation_seconds=0.
    with (out/'metrics.csv').open('w',newline='') as mf,(out/'layer_metrics.csv').open('w',newline='') as lf,(out/'firing.csv').open('w',newline='') as ff:
        mw=lw=None;fw=csv.DictWriter(ff,fieldnames=['step','action','naive_comparison']);fw.writeheader()
        for step in range(1,12001):
            t=time.perf_counter();(proposal,indices,stats,new_s),timing=gpu_epoch(m,x,y,a,g,s,bg,ev)
            engine_seconds+=time.perf_counter()-t;gpu_milliseconds+=timing['gpu_workflow_milliseconds']
            vals=ev.last_losses.numpy();losses[step-1]=vals;selected[step-1]=indices.numpy();values=np.asarray(stats['abs_y_values'],dtype=np.float32);abs_y[step-1]=values
            assert np.isfinite(vals).all() and np.array_equal(np.abs(vals[::2]-vals[1::2]),values)
            fire=action(m.weights,proposal);comparison='after_first_divergence'
            if first_div is None:
                t=time.perf_counter();np_,ni,nstats,new_ns=train.epoch(naive,x,y,a,ng,ns,nbg);reference_seconds+=time.perf_counter()-t;replayed+=1
                assert torch.equal(indices,ni) and torch.equal(g.get_state(),ng.get_state()) and torch.equal(bg.get_state(),nbg.get_state())
                # Verify reference replay against immutable historical CPU log before using it.
                assert np.array_equal(np.asarray(nstats['abs_y_values'],dtype=np.float32),oldabs[step-1])
                for k in ['scale','fires','counter_min','counter_max','counter_mean','nonzero_vote_rate']:assert float(nstats[k])==float(oldmetrics[step-1][k]),(seed,step,k)
                numeric=stats['abs_y_values']!=nstats['abs_y_values'] or new_s!=new_ns
                if numeric and first_numeric is None:first_numeric=step
                nfire=action(naive.weights,np_)
                if fire!=nfire:
                    first_div=step;dump(out/'first_divergence.json',dict(step=step,gpu_action=fire,cpu_action=nfire,gpu_scale=s,cpu_scale=ns,gpu_stats=stats,cpu_stats=nstats,naive_historical_log_verified=True))
                naive.weights.copy_(np_);ns=new_ns;comparison='independent_naive_replay'
            fw.writerow(dict(step=step,action=fire,naive_comparison=comparison))
            for e in layer_events(m,proposal,indices):
                row=dict(step=step,**e)
                if lw is None:lw=csv.DictWriter(lf,fieldnames=list(row));lw.writeheader()
                lw.writerow(row);t=totals[e['layer']]
                for dst,src in [('fires','fires'),('selected_intervals','selected_interval'),('fire_intervals','fire_interval'),('selected_coordinates','selected_coordinates')]:t[dst]+=e[src]
            for k,v in stats.pop('counter_histogram').items():hist[k]=hist.get(k,0)+v
            stats.pop('abs_y_values');m.weights.copy_(proposal);s=new_s
            row=dict(step=step,elapsed_seconds=time.perf_counter()-started,engine_seconds=engine_seconds,train_candidate_forward_equivalents=step*128,abs_y_mean=float(values.astype('float64').mean()),val_loss=None,val_accuracy=None,**timing,**stats)
            if step%500==0:
                t=time.perf_counter();final,sr,ar,rr=observe(m,vx,vy,step);validation_seconds+=time.perf_counter()-t
                signals.extend(sr);activation.extend(ar);ratios.extend(rr);row.update(val_loss=final['loss'],val_accuracy=final['accuracy'])
                write_csv(out/'signal.csv',signals);write_csv(out/'activation.csv',activation);write_csv(out/'rms_ratios.csv',ratios)
                torch.save(dict(weights=m.weights,step=step,scale=s,generator=g.get_state(),batch_generator=bg.get_state(),config=cfg),out/'checkpoint.pt')
                losses.flush();abs_y.flush();selected.flush()
                dump(out/'progress.json',dict(step=step,validation=final,elapsed_seconds=time.perf_counter()-started,engine_seconds=engine_seconds,first_firing_divergence=first_div))
                print(f'GPU E17a seed{seed} step={step} val={final["accuracy"]:.3%} first_divergence={first_div}',flush=True)
            if mw is None:mw=csv.DictWriter(mf,fieldnames=list(row));mw.writeheader()
            mw.writerow(row);mf.flush();lf.flush();ff.flush()
    elapsed=time.perf_counter()-started;losses.flush();abs_y.flush();selected.flush()
    t=time.perf_counter();probe_rows.extend(probes(m,x,y,seed,'final'));final_probe_seconds=time.perf_counter()-t;write_csv(out/'probes.csv',probe_rows)
    torch.save(dict(weights=m.weights,scale=s,config=cfg),out/'model.pt')
    # Only final test call. CPU inference fixes evaluation protocol to original E17.
    test=evaluate(m,tx,ty)
    summary=dict(seed=seed,condition='GPU_E17a',status='success',blocks=8,width=76,num_params=100016,steps=12000,initial_validation=initial,final_validation=final,test=test,test_evaluations=1,evaluation_device='cpu',candidate_device='cuda',candidate_forward_equivalents=1536000,first_numeric_difference=first_numeric,first_firing_divergence=first_div,naive_reference_replayed_intervals=replayed,naive_reference_seconds=reference_seconds,elapsed_seconds=elapsed,engine_seconds=engine_seconds,gpu_workflow_milliseconds=gpu_milliseconds,validation_seconds=validation_seconds,initial_probe_seconds=initial_probe_seconds,final_probe_seconds=final_probe_seconds,gpu_setup_seconds=setup_seconds,layer_totals=totals,counter_histogram=hist,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    dump(out/'summary.json',summary);dump(out/'manifest.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'})
    print(f'GPU E17a seed{seed} complete; final test saved for aggregate after all3',flush=True)


def analyze():
    assert json.loads((ROOT/'status.json').read_text())['training_complete']
    old={int(r['seed']):r for r in read(OLD/'per_seed/results.csv') if r['condition']=='E17a'};rows=[]
    for seed in range(3):
        s=json.loads((ROOT/'per_seed'/f'seed{seed}'/'summary.json').read_text())
        rows.append(dict(seed=seed,cpu_test_percent=float(old[seed]['test_accuracy_percent']),gpu_trained_test_percent=s['test']['accuracy']*100,delta_pp=s['test']['accuracy']*100-float(old[seed]['test_accuracy_percent']),cpu_validation_percent=float(old[seed]['validation_accuracy_percent']),gpu_validation_percent=s['final_validation']['accuracy']*100,first_firing_divergence=s['first_firing_divergence'],elapsed_seconds=s['elapsed_seconds'],engine_seconds=s['engine_seconds']))
    vals=[r['gpu_trained_test_percent'] for r in rows];deltas=[r['delta_pp'] for r in rows];mean=statistics.mean(vals)
    agg=dict(gpu_test_mean_percent=mean,gpu_test_sample_std_percent=statistics.stdev(vals),cpu_test_mean_percent=statistics.mean(r['cpu_test_percent'] for r in rows),cpu_test_sample_std_percent=statistics.stdev(r['cpu_test_percent'] for r in rows),paired_delta_mean_pp=statistics.mean(deltas),paired_delta_sample_std_pp=statistics.stdev(deltas),fixed_reference_percent=90.637,lower_bound_percent=90.337,upper_bound_percent=90.937,accuracy_band_pass=90.337<=mean<=90.937,nondegradation_lower_bound_pass=mean>=90.337,strict_firing_sequence_equal=all(r['first_firing_divergence'] is None for r in rows),mean_training_seconds=statistics.mean(r['elapsed_seconds'] for r in rows))
    write_csv(ROOT/'per_seed/results.csv',rows);write_csv(ROOT/'aggregate/results.csv',[agg]);dump(ROOT/'report.json',agg)
    lines=['# GPU CUDA Graphs E17a：12000区間×3 seed再現','', '比較対象はE17a。8残差ブロック・幅76・18行列・100,016三値重み、A8＋ReLU。各seedは元と同じ初期三値重みから学習した。16ブロック/E18の長期学習ではない。','',f'GPU学習後test：**{mean:.4f} ± {statistics.stdev(vals):.4f}%**（3 seed平均±標本標準偏差）。CPU E17a：{agg["cpu_test_mean_percent"]:.4f} ± {agg["cpu_test_sample_std_percent"]:.4f}%。対応seed差：{agg["paired_delta_mean_pp"]:+.4f} ± {agg["paired_delta_sample_std_pp"]:.4f}pt。','',f'事前登録90.637±0.3ptの範囲判定：**{agg["accuracy_band_pass"]}**。下限90.337%以上：{agg["nondegradation_lower_bound_pass"]}。発火系列の完全一致：{agg["strict_firing_sequence_equal"]}。','', '| seed | CPU test % | GPU学習後test % | 差 pt | 最初の発火分岐 | 学習実時間 秒 |','|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f'| {r["seed"]} | {r["cpu_test_percent"]:.2f} | {r["gpu_trained_test_percent"]:.2f} | {r["delta_pp"]:+.2f} | {r["first_firing_divergence"]} | {r["elapsed_seconds"]:.2f} |')
    lines+=['', '最終精度は既存E17と同じCPU推論で統一し、各seedの最終12000区間でtestを1回だけ評価した。候補評価は固定したGPU BMM＋CUDA Graphs・FP32・TF32無効。精度維持とCPUビット一致は別の判断であり、3 seedの結果は統計的な同等性証明ではない。','', '候補・ミニバッチ・確率的投票の乱数消費順と、C8・閾値・タイブレーク・S更新は元のepochコード。発火が初めて分岐するまで独立CPU再生を実施し、再生された候補差・カウンタを既存E17aのログと照合した。数値による分岐後もGPUは自分の重み・Sで最後まで学習し、CPUへ戻す補完処理は使用していない。','', 'validationは初期＋500区間ごと。testはモデル選択・条件探索に使わない。初期/最終の層単独プローブと各時点のRMS・量子化・枝/ストリーム比は、学習後重みに対するCPU診断としてE17定義を継承した。GPU候補の128損失はcandidate_losses.npy、64候補差はabs_y.npyに全区間保存。','', 'GPU3 seedは逐次実行、既存CPU実験との資源共有あり。時間はengine/CPU reference/validation/setup/probesをseed別に分離した。過去CPU実時間との単純比を専有環境での速度比較とみなさない。GPUメモリはPyTorchの割当・予約カウンタであり、全プロセスVRAMとは異なる。','', 'config、実行時sources、データ・対照・成果物SHA-256、初期val、全区間/行列/発火CSV、チェックポイント、最終モデル、summaryを保存。監査結果はaudit.json。']
    if mean>90.937:lines+=['','上限超過は改善方向であり、精度劣化を意味しない。ただし事前登録した両側再現範囲には未達。']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')


def run_all():
    ROOT.mkdir(parents=True,exist_ok=True);assert not (ROOT/'sources').exists()
    names=['run_gpu_e17a.py','audit_gpu_e17a.py','gpu_evaluation_engines.py','allocation_engines.py','GPU_E17A_REPRODUCTION_PREREGISTRATION.md','train.py','residual_followup_models.py','residual_stream.py','activation_quantization.py','depth_diagnostics.py','run_residual_e17.py']
    (ROOT/'sources').mkdir()
    for n in names:shutil.copy2(HERE/n,ROOT/'sources'/n)
    dump(ROOT/'manifest.json',dict(preregistration_commit='c908462',sources={n:sha(ROOT/'sources'/n) for n in names},data={p.name:sha(p) for p in (HERE/'data/MNIST/raw').glob('*-ubyte')},cpu_reference_sha256=sha(OLD/'per_seed/results.csv'),cpu_source_manifest_sha256=sha(OLD/'manifest.json')))
    init();props=torch.cuda.get_device_properties(0)
    dump(ROOT/'environment.json',dict(torch_version=torch.__version__,cuda_version=torch.version.cuda,gpu_name=props.name,gpu_memory=props.total_memory,nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),tf32=False,cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG']))
    preflight()
    context=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True);events=[]
    runtime=dict(gpu_workers=1,seeds_sequential=True,cpu_threads=1,cpu_affinity=[15],existing_jobs=[l for l in context.splitlines() if 'run_fast_engine.py' in l or 'finish_fast_engine_report.py' in l],events=events)
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for seed in range(3):
        dump(ROOT/'status.json',dict(complete=False,training_complete=False,completed=seed,expected=3,active_seed=seed))
        with (ROOT/f'seed{seed}.log').open('w') as log:
            p=subprocess.Popen([sys.executable,__file__,'worker','--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,env=env)
            events.append(dict(seed=seed,pid=p.pid,start_time=time.time()));dump(ROOT/'runtime_workers.json',runtime)
            code=p.wait();events[-1].update(end_time=time.time(),returncode=code);dump(ROOT/'runtime_workers.json',runtime)
            if code:
                dump(ROOT/'status.json',dict(complete=False,training_complete=False,completed=seed,failed_seed=seed,returncode=code));raise RuntimeError((seed,code))
    dump(ROOT/'status.json',dict(complete=False,training_complete=True,completed=3,expected=3))
    analyze()
    subprocess.run([sys.executable,str(HERE/'audit_gpu_e17a.py')],check=True,env=env)
    print('GPU E17a all3 runs and audit complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker','preflight','analyze']);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
    if a.mode=='run':run_all()
    elif a.mode=='worker':worker(a.seed)
    elif a.mode=='preflight':preflight()
    else:analyze()
