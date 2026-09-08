"""E18/E19 TDT and E20 backprop runs under the committed fixed protocol."""
import argparse,csv,json,math,os,shutil,statistics,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import torch
import torch.nn.functional as F
from train import load_data,evaluate,epoch,candidate_pair,loss
from depth_diagnostics import SignalObserver,layer_events
from activation_quantization import ActivationObserver
from residual_followup_models import ResidualTDT,BPResidual,TDT_CONDITIONS,ternary_weight,divergence_reason
from run_residual_e17 import config,dump,write_csv,sha
HERE=Path(__file__).resolve().parent
ROOT=HERE/'results/residual-followups-e18-e20-20260908'
E17=HERE/'results/residual-stream-a8-e17-20260908'
ALL_CONDITIONS=list(TDT_CONDITIONS)+['E20a','E20b','E20c']


def setup(backprop=False):
    torch.set_num_threads(1)
    torch.set_grad_enabled(backprop)
    torch.use_deterministic_algorithms(True)


def observe(m, vx, vy, step):
    m.signal_observer = SignalObserver()
    m.activation_observer = ActivationObserver(len(m.shapes), m.activation_precision)
    val = evaluate(m, vx, vy)
    signals = [{'step': step, 'matrix': m.matrix_names[r['layer']], **r}
               for r in m.signal_observer.summary()]
    activation = []
    for r in m.activation_observer.summary():
        r['code_histogram'] = json.dumps(r['code_histogram'], sort_keys=True)
        activation.append({'step': step, 'matrix': m.matrix_names[r['layer']], **r})
    ratios = []
    for b in range(m.blocks):
        rows = {r['stage']: r for r in signals if r['layer'] == 1+2*b}
        stream = rows['stream_before']['rms']
        branch = rows['branch_output']['rms']
        ratios.append(dict(step=step, block=b, stream_rms=stream, branch_rms=branch,
                           branch_stream_rms_ratio=branch/stream if stream else None))
    m.signal_observer = m.activation_observer = None
    return val, signals, activation, ratios


def probes(m, x, y, seed, stage):
    rows = []
    offset = 0
    before = m.weights.clone()
    for layer, shape in enumerate(m.shapes):
        size = math.prod(shape)
        g = torch.Generator().manual_seed(700000+seed*1000+len(m.shapes)*20+layer)
        bg = torch.Generator().manual_seed(900000+seed*1000+len(m.shapes)*20+layer)
        for pair in range(64):
            indices = torch.randperm(size, generator=g)[:16]+offset
            plus, minus, _, _ = candidate_pair(m.weights, indices, g)
            batch = torch.randint(len(x), (128,), generator=bg)
            lp = loss(m, x[batch], y[batch], plus)
            lm = loss(m, x[batch], y[batch], minus)
            rows.append(dict(stage=stage, layer=layer, matrix=m.matrix_names[layer],
                pair=pair, perturbed_coordinates=16, loss_plus=float(lp),
                loss_minus=float(lm), abs_y=float((lp-lm).abs())))
        offset += size
    assert torch.equal(before, m.weights)
    return rows


def tdt_run(condition, seed, root, data):
    setup()
    a = config(seed, data)
    blocks,width,precision,expected_count = TDT_CONDITIONS[condition]
    activation='relu'
    m=ResidualTDT(seed,blocks,width,precision)
    assert m.num_params==expected_count
    out = root/'per_seed'/f'{condition}-seed{seed}'
    out.mkdir(parents=True, exist_ok=False)
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    cfg.update(condition=condition, architecture='residual_stream', width=width, blocks=blocks,
        activation_precision=precision, hidden_activation=activation, num_params=m.num_params,
        shapes=m.shapes, layer_scales=m.scales, matrix_names=m.matrix_names,
        rmsnorm_eps=1e-8, rmsnorm_trainable=False, stream_dtype='float32',
        weight_storage='int8 ternary', torch_version=torch.__version__,
        source_sha256=json.loads((root/'manifest.json').read_text())['sources'])
    dump(out/'config.json', cfg)
    (x, y), (vx, vy), (tx, ty) = load_data(a, m.device)
    g = torch.Generator().manual_seed(seed+1)
    bg = torch.Generator().manual_seed(seed+100000)
    initial, signals, activations, ratios = observe(m, vx, vy, 0)
    expected = [r for r in json.loads((root/'preflight.json').read_text())['initial_validation']
                if r['condition'] == condition and r['seed'] == seed][0]
    assert initial == {k: expected[k] for k in ('loss', 'accuracy')}
    probe_rows = probes(m, x, y, seed, 'initial')
    write_csv(out/'probes.csv', probe_rows)
    write_csv(out/'signal.csv', signals)
    write_csv(out/'activation.csv', activations)
    write_csv(out/'rms_ratios.csv', ratios)
    abs_y = np.lib.format.open_memmap(out/'abs_y.npy', mode='w+', dtype='float32', shape=(12000,64))
    totals = [dict(fires=0, selected_intervals=0, fire_intervals=0, selected_coordinates=0) for _ in range(len(m.shapes))]
    scale = .02
    calls = 0
    started = time.perf_counter()
    histogram = {}
    with (out/'metrics.csv').open('w', newline='') as mf, (out/'layer_metrics.csv').open('w', newline='') as lf:
        mw = None
        lw = csv.DictWriter(lf, fieldnames=['step','layer','parameters','selected_coordinates','selected_interval','fires','fire_interval'])
        lw.writeheader()
        for step in range(1, 12001):
            before = m.forward_calls
            proposal, indices, stats, scale = epoch(m, x, y, a, g, scale, bg)
            assert m.forward_calls-before == 128
            calls += 128
            values = np.asarray(stats.pop('abs_y_values'), dtype=np.float32)
            assert np.isfinite(values).all()
            abs_y[step-1] = values
            for k, v in stats.pop('counter_histogram').items():
                histogram[k] = histogram.get(k, 0)+v
            for e in layer_events(m, proposal, indices):
                lw.writerow(dict(step=step, **e))
                t = totals[e['layer']]
                for dst, src in [('fires','fires'), ('selected_intervals','selected_interval'),
                                 ('fire_intervals','fire_interval'), ('selected_coordinates','selected_coordinates')]:
                    t[dst] += e[src]
            m.weights.copy_(proposal)
            row = dict(step=step, elapsed_seconds=time.perf_counter()-started,
                train_forward_calls=calls, abs_y_mean=float(values.astype('float64').mean()),
                val_loss=None, val_accuracy=None, **stats)
            if step % 500 == 0:
                final, sr, ar, rr = observe(m, vx, vy, step)
                signals.extend(sr); activations.extend(ar); ratios.extend(rr)
                row.update(val_loss=final['loss'], val_accuracy=final['accuracy'])
                write_csv(out/'signal.csv', signals)
                write_csv(out/'activation.csv', activations)
                write_csv(out/'rms_ratios.csv', ratios)
                abs_y.flush()
                torch.save(dict(weights=m.weights, step=step, scale=scale, generator=g.get_state(),
                    batch_generator=bg.get_state(), config=cfg), out/'checkpoint.pt')
                dump(out/'progress.json', dict(step=step, validation=final, elapsed_seconds=time.perf_counter()-started))
                print(f'{condition} seed{seed} step={step} validation={final["accuracy"]:.3%}', flush=True)
            if mw is None:
                mw = csv.DictWriter(mf, fieldnames=list(row)); mw.writeheader()
            mw.writerow(row)
            mf.flush(); lf.flush()
    abs_y.flush()
    assert calls == 1536000
    probe_rows.extend(probes(m, x, y, seed, 'final'))
    write_csv(out/'probes.csv', probe_rows)
    # The only test evaluation, after all predetermined training is complete.
    test = evaluate(m, tx, ty)
    torch.save(dict(weights=m.weights, config=cfg), out/'model.pt')
    summary = dict(condition=condition, seed=seed, initial_validation=initial,
        final_validation=final, test=test, train_forward_calls=calls,
        diagnostic_probe_forward_calls=4*64*len(m.shapes), total_forward_calls=m.forward_calls,
        total_forward_examples=m.forward_examples, test_evaluations=1,
        elapsed_seconds=time.perf_counter()-started, layer_totals=totals,
        counter_histogram=histogram, num_params=m.num_params)
    summary.update(status='success',blocks=blocks,width=width,activation_precision=precision)
    dump(out/'summary.json', summary)
    dump(out/'manifest.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    return condition, seed



def clean_json(value):
    if isinstance(value,float) and not math.isfinite(value): return None
    if isinstance(value,dict): return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean_json(v) for v in value]
    return value


def model_digest(m):
    import hashlib
    return hashlib.sha256(b''.join(w.detach().numpy().tobytes() for w in m.latent)).hexdigest()


def weight_diagnostics(m,epoch_number):
    rows=[]
    for i,w in enumerate(m.latent):
        effective,codes,alpha=ternary_weight(w)
        rows.append(dict(epoch=epoch_number,layer=i,matrix=m.matrix_names[i],
            latent_rms=float(w.detach().square().mean().sqrt()),alpha=float(alpha),
            effective_rms=float(effective.square().mean().sqrt()),
            negative_codes=int((codes==-1).sum()),zero_codes=int((codes==0).sum()),
            positive_codes=int((codes==1).sum()),parameters=w.numel(),
            representation='active_W3' if m.condition=='E20c' else 'diagnostic_only_W3_of_FP32'))
    return rows


def bp_attempt(condition,seed,out,data,attempt):
    setup(True)
    out.mkdir(parents=True,exist_ok=False)
    m=BPResidual(condition,seed)
    a=config(seed,data)
    (x,y),(vx,vy),(tx,ty)=load_data(a,torch.device('cpu'))
    rescue=attempt==1
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    cfg.update(condition=condition,attempt=attempt,architecture='residual_stream',blocks=8,width=76,
        num_params=100016,shapes=m.shapes,matrix_names=m.matrix_names,optimizer='Adam',
        learning_rate=.0003 if rescue else .001,gradient_clip_norm=1. if rescue else None,
        weight_decay=0.,adam_betas=[.9,.999],adam_eps=1e-8,max_epochs=100,
        minimum_epochs=30,early_stopping_patience=20,validation_selection='minimum epoch-end loss',
        validation_schedule='initial, every epoch end, plus diagnostic-only every 500 optimizer updates',
        lr_scheduler=dict(factor=.5,patience=5,min_lr=1e-5),
        initialization='paired normal He: hidden sqrt(2/fan-in), output sqrt(1/fan-in)',
        initial_weight_sha256=model_digest(m),rmsnorm_eps=1e-8,rmsnorm_trainable=False,
        activation_precision=m.activation_precision,weight_representation='latent FP32 with dynamic W3' if condition=='E20c' else 'FP32',
        ste='identity on A8 and effective W3; no separate alpha derivative',
        weight_quantization='alpha=mean(abs(latent)), int8 round(w/alpha).clamp(-1,1), effective=alpha*q',
        explosion_rms_threshold=1e4,torch_version=torch.__version__,
        source_sha256=json.loads((out.parents[2]/'manifest.json').read_text())['sources'])
    for unused in ['steps','measurements','block_size','threshold','max_fires','counter_bits','leak','scale','scale_ema','min_scale','expected_params','zero_rate','gain']:
        cfg.pop(unused,None)
    dump(out/'config.json',cfg)
    with torch.no_grad():
        initial,signals,activations,ratios=observe(m,vx,vy,0)
    expected=next(r for r in json.loads((out.parents[2]/'preflight.json').read_text())['initial_validation'] if r['condition']==condition and r['seed']==seed)
    assert initial=={k:expected[k] for k in ('loss','accuracy')}
    with torch.no_grad(): initial_predictions=m(vx).argmax(1).numpy()
    np.savez_compressed(out/'initial_validation_predictions.npz',predictions=initial_predictions,labels=vy.numpy())
    # The above audit prediction forward is explicitly separate from optimizer updates.
    optimizer=torch.optim.Adam(m.parameters(),lr=cfg['learning_rate'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=5,min_lr=1e-5)
    g=torch.Generator().manual_seed(seed+100000)
    initial_weights={k:v.detach().clone() for k,v in m.state_dict().items()}
    torch.save(dict(state_dict=initial_weights,config=cfg),out/'initial_model.pt')
    history=[];gradient_rows=[];weights_rows=weight_diagnostics(m,0);update_validation=[]
    best=float('inf');best_epoch=0;stale=0;updates=0;exploded=False;failure=None
    started=time.perf_counter()
    for ep in range(1,101):
        m.train()
        order=torch.randperm(len(x),generator=g)
        gs=np.zeros(18);gm=np.zeros(18);gz=np.zeros(18,dtype=np.int64)
        batches=0;examples=0;train_loss=0.;correct=0;nonfinite=False
        for start in range(0,len(x),128):
            ids=order[start:start+128]
            optimizer.zero_grad(set_to_none=True)
            z=m(x[ids]);objective=F.cross_entropy(z,y[ids])
            if not torch.isfinite(z).all() or not torch.isfinite(objective):
                nonfinite=True;failure='nonfinite_loss_or_logits';break
            objective.backward()
            norms=[float(w.grad.norm()) if w.grad is not None else float('nan') for w in m.latent]
            if not all(math.isfinite(v) for v in norms):
                nonfinite=True;failure='nonfinite_or_missing_gradients';break
            gs+=norms;gm=np.maximum(gm,norms);gz+=(np.asarray(norms)==0);batches+=1
            if rescue: torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();updates+=1
            if not all(torch.isfinite(w).all() for w in m.latent):
                nonfinite=True;failure='nonfinite_latent_weights';break
            examples+=len(ids);train_loss+=float(objective.detach())*len(ids)
            correct+=int((z.detach().argmax(1)==y[ids]).sum())
            if updates%500==0:
                with torch.no_grad(): val500=evaluate(m,vx,vy)
                update_validation.append(dict(update=updates,epoch=ep,selection_eligible=False,**val500))
                write_csv(out/'update_validation.csv',update_validation)
        norms=(gs/max(1,batches)).tolist()
        gradient_rows.extend(dict(epoch=ep,layer=i,matrix=m.matrix_names[i],batches=batches,
            mean_gradient_norm=float(norms[i]),max_gradient_norm=float(gm[i]),zero_gradient_batches=int(gz[i]),
            gradient_before_clipping=True,partial_epoch=bool(nonfinite)) for i in range(18))
        with torch.no_grad():
            val,sr,ar,rr=observe(m,vx,vy,ep)
            wr=weight_diagnostics(m,ep)
        signals.extend(sr);activations.extend(ar);ratios.extend(rr);weights_rows.extend(wr)
        peak=max((r['rms'] for r in sr if r['stage']=='output'),default=0.)
        exploded=exploded or peak>1e4
        relu_zero=any(r['stage']=='branch_activation' and r['zero_fraction']==1 for r in sr)
        nonfinite=nonfinite or not math.isfinite(val['loss']) or any(r['nonfinite_count'] for r in sr)
        failure=failure or divergence_reason(nonfinite,exploded,relu_zero,norms)
        lr=optimizer.param_groups[0]['lr']
        improved=False
        if not failure:
            scheduler.step(val['loss'])
            improved=val['loss']<best
            if improved:
                best=val['loss'];best_epoch=ep;stale=0
                torch.save(dict(state_dict=m.state_dict(),config=cfg,epoch=ep,validation=val),out/'best_model.pt')
            else: stale+=1
        history.append(dict(epoch=ep,optimizer_updates=updates,train_loss=train_loss/max(1,examples),
            train_accuracy=correct/max(1,examples),val_loss=val['loss'],val_accuracy=val['accuracy'],
            lr=lr,best_epoch=best_epoch,selected_best=improved,explosion_seen=exploded,
            relu_branch_all_zero=relu_zero,all_layer_epoch_gradients_zero=all(v==0 for v in norms),
            failure=failure,elapsed_seconds=time.perf_counter()-started))
        for name,rows in [('training.csv',history),('gradient_metrics.csv',gradient_rows),('signal.csv',signals),
                          ('activation.csv',activations),('rms_ratios.csv',ratios),('weight_metrics.csv',weights_rows)]:
            write_csv(out/name,clean_json(rows))
        dump(out/'progress.json',clean_json(dict(epoch=ep,updates=updates,validation=val,failure=failure)))
        print(f'{condition} seed{seed} attempt{attempt} epoch={ep} val={val["accuracy"]:.3%} failure={failure}',flush=True)
        if failure:
            torch.save(dict(state_dict=m.state_dict(),optimizer=optimizer.state_dict(),generator=g.get_state(),
                epoch=ep,updates=updates,reason=failure,config=cfg),out/'failed_model.pt')
            break
        if ep>=30 and stale>=20: break
    if failure:
        summary=dict(status='failed',reason=failure,condition=condition,seed=seed,attempt=attempt,
            epochs=len(history),optimizer_updates=updates,test=None,test_evaluations=0,
            initial_validation=initial,elapsed_seconds=time.perf_counter()-started)
    else:
        checkpoint=torch.load(out/'best_model.pt',weights_only=False,map_location='cpu')
        m.load_state_dict(checkpoint['state_dict']);m.eval()
        with torch.no_grad():
            selected,ss,aa,rr=observe(m,vx,vy,best_epoch)
            assert selected==checkpoint['validation']
            # Only test evaluation in a successful attempt; never used for retry or selection.
            test=evaluate(m,tx,ty)
            # Validation-only predictions for an independent replay audit.
            val_pred=m(vx).argmax(1).numpy()
        np.savez_compressed(out/'selected_validation_predictions.npz',predictions=val_pred,labels=vy.numpy())
        for name,rows in [('selected_signal.csv',ss),('selected_activation.csv',aa),('selected_rms_ratios.csv',rr)]:
            write_csv(out/name,rows)
        torch.save(dict(state_dict=m.state_dict(),config=cfg,epoch=best_epoch,validation=selected),out/'model.pt')
        if condition=='E20c':
            encoded=[ternary_weight(w) for w in m.latent]
            torch.save(dict(codes=[r[1] for r in encoded],alphas=[r[2] for r in encoded],
                effective_weights=[r[0] for r in encoded],latent_checkpoint='model.pt'),out/'quantized_model.pt')
        summary=dict(status='success',condition=condition,seed=seed,attempt=attempt,
            epochs=len(history),best_epoch=best_epoch,optimizer_updates=updates,
            initial_validation=initial,selected_validation=selected,test=test,test_evaluations=1,
            elapsed_seconds=time.perf_counter()-started,initial_weight_sha256=cfg['initial_weight_sha256'],
            final_weight_sha256=model_digest(m),forward_calls=m.forward_calls,forward_examples=m.forward_examples)
    dump(out/'summary.json',clean_json(summary))
    dump(out/'manifest.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    return summary


def bp_run(condition,seed,root,data):
    out=root/'per_seed'/f'{condition}-seed{seed}'
    out.mkdir(parents=True,exist_ok=False)
    attempts=[]
    result=bp_attempt(condition,seed,out/'attempt0',data,0);attempts.append(result)
    if result['status']=='failed' and condition=='E20c':
        result=bp_attempt(condition,seed,out/'attempt1',data,1);attempts.append(result)
    dump(out/'summary.json',dict(condition=condition,seed=seed,status=result['status'],attempts=attempts,
        selected_attempt=result['attempt'] if result['status']=='success' else None,
        test=result.get('test'),elapsed_seconds=sum(r['elapsed_seconds'] for r in attempts)))
    dump(out/'manifest.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    return condition,seed


def preflight(root,data):
    setup()
    root.mkdir(parents=True,exist_ok=False)
    a=config(0,data)
    (x,y),(vx,vy),_=load_data(a,torch.device('cpu'))
    initial=[];timings=[]
    legacy_seconds=statistics.mean(json.loads((E17/'per_seed'/f'E17a-seed{s}/summary.json').read_text())['elapsed_seconds'] for s in range(3))
    for c in ALL_CONDITIONS:
        for seed in range(3):
            if c in TDT_CONDITIONS:
                b,d,p,n=TDT_CONDITIONS[c];m=ResidualTDT(seed,b,d,p)
            else: m=BPResidual(c,seed)
            with torch.no_grad():val=evaluate(m,vx,vy)
            initial.append(dict(condition=c,seed=seed,num_params=m.num_params,**val))
        if c in TDT_CONDITIONS:
            m=ResidualTDT(0,b,d,p)
            g=torch.Generator().manual_seed(1);bg=torch.Generator().manual_seed(100000)
            started=time.perf_counter();before=m.forward_calls
            for _ in range(3): epoch(m,x,y,a,g,.02,bg)  # Disposable, no update applied.
            sec=(time.perf_counter()-started)/3
            assert m.forward_calls-before==384
            timings.append(dict(condition=c,blocks=b,width=d,input_weights=90*d,branch_weights=2*b*d*d,
                output_weights=10*d,total_weights=n,mac_ratio_to_e17=n/100016,
                e17_actual_seconds_mean=legacy_seconds,mac_based_estimate_seconds=legacy_seconds*n/100016,
                measured_seconds_per_interval=sec,isolated_estimate_seconds=sec*12000,forwards_per_interval=128))
    write_csv(root/'preflight_initial_validation.csv',initial)
    write_csv(root/'preflight_runtime.csv',timings)
    dump(root/'preflight.json',dict(initial_validation=initial,timings=timings,test_evaluated=False,
        data_sha256={p.name:sha(p) for p in sorted((Path(data)/'MNIST/raw').glob('*-ubyte'))}))
    print(json.dumps(timings,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['preflight','run','worker'])
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--data',type=Path,default=HERE/'data')
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--condition',choices=ALL_CONDITIONS)
    p.add_argument('--seed',type=int,choices=[0,1,2])
    a=p.parse_args()
    if a.mode=='preflight':preflight(a.root,a.data);return
    if a.mode=='worker':
        if a.condition=='E18d':
            assert json.loads((a.root/'authorizations.json').read_text())['e18d_budget_exception_approved']
        (tdt_run if a.condition in TDT_CONDITIONS else bp_run)(a.condition,a.seed,a.root,a.data)
        return
    assert (a.root/'preflight.json').exists()
    assert not (a.root/'manifest.json').exists()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor','16485f4',revision],cwd=HERE,check=True)
    sources=['train.py','activation_quantization.py','depth_diagnostics.py','residual_stream.py',
             'run_residual_e17.py','residual_followup_models.py','run_residual_followups.py',
             'test_residual_followups.py','E18_E20_PREREGISTRATION.md']
    (a.root/'sources').mkdir()
    for name in sources:shutil.copy2(HERE/name,a.root/'sources'/name)
    dump(a.root/'manifest.json',dict(git_revision=revision,preregistration_commit='16485f4',
        sources={n:sha(a.root/'sources'/n) for n in sources},conditions=ALL_CONDITIONS,seeds=[0,1,2],
        data_sha256=json.loads((a.root/'preflight.json').read_text())['data_sha256'],
        e17_artifacts_sha256=sha(E17/'artifacts_sha256.json')))
    tasks=[(c,s) for c in ['E18c','E18b','E18a','E20c','E20a','E20b','E19a','E18d'] for s in range(3)]
    active={};completed=[];errors=[];events=[]
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    runtime=dict(workers_limit=a.workers,threads_per_worker=1,cpu_count=os.cpu_count(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),events=events)
    while tasks or active:
        approved=(a.root/'authorizations.json').exists() and json.loads((a.root/'authorizations.json').read_text()).get('e18d_budget_exception_approved',False)
        for task in list(tasks):
            if len(active)>=a.workers: break
            c,s=task
            if c=='E18d' and not approved: continue
            log=(a.root/f'{c}-seed{s}.log').open('w')
            cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--root',str(a.root),
                 '--data',str(a.data),'--condition',c,'--seed',str(s)]
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            active[task]=(process,log)
            tasks.remove(task)
            events.append(dict(event='start',condition=c,seed=s,pid=process.pid,time=time.time(),active_workers=len(active)))
        for task,(process,log) in list(active.items()):
            code=process.poll()
            if code is None:continue
            log.close();del active[task]
            events.append(dict(event='end',condition=task[0],seed=task[1],pid=process.pid,time=time.time(),returncode=code,active_workers=len(active)))
            if code==0:completed.append(task)
            else:errors.append(dict(condition=task[0],seed=task[1],returncode=code))
        dump(a.root/'runtime_workers.json',runtime)
        dump(a.root/'status.json',dict(complete=False,training_complete=len(completed)==24 and not errors,
            completed=len(completed),expected=24,active=[list(t) for t in active],pending=tasks,errors=errors,
            waiting_for_e18d_approval=any(t[0]=='E18d' for t in tasks) and not approved))
        if tasks or active:time.sleep(5)
    if errors:raise RuntimeError(errors)
    dump(a.root/'status.json',dict(complete=False,training_complete=True,completed=24,expected=24,errors=[]))
    print('All 24 runs finished; ready for independent final audit',flush=True)


if __name__=='__main__':main()
