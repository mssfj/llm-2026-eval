"""Independent E18-E20 final audit and preregistered comparisons; no test replay."""
import argparse,csv,json,math,statistics,shutil,time
from pathlib import Path
import numpy as np
import torch
from train import load_data,evaluate
from residual_followup_models import ResidualTDT,BPResidual,TDT_CONDITIONS,ternary_weight
from run_residual_followups import ROOT,E17,HERE,ALL_CONDITIONS,setup,observe,probes,config,model_digest
from run_residual_e17 import dump,write_csv,sha


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def verify_manifest(out):
    for name,digest in json.loads((out/'manifest.json').read_text()).items():
        assert sha(out/name)==digest,(out,name)


def aggregate_rows(rows,keys,metrics):
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    result=[]
    for key,rs in sorted(groups.items()):
        out=dict(zip(keys,key));out['seeds']=len(rs)
        for metric in metrics:
            vals=[float(r[metric]) for r in rs if r[metric] not in ('',None)]
            out[metric+'_mean']=statistics.mean(vals) if vals else None
            out[metric+'_sample_std']=statistics.stdev(vals) if len(vals)>1 else None
        result.append(out)
    return result


def tdt_audit(root,c,seed):
    out=root/'per_seed'/f'{c}-seed{seed}'
    verify_manifest(out)
    cfg=json.loads((out/'config.json').read_text());s=json.loads((out/'summary.json').read_text())
    blocks,width,precision,count=TDT_CONDITIONS[c]
    m=ResidualTDT(seed,blocks,width,precision)
    assert s['status']=='success' and s['num_params']==count
    assert len(m.shapes)==2*blocks+2 and cfg['shapes']==[list(shape) for shape in m.shapes]
    a=config(seed,cfg['data_dir'])
    for key in ['steps','measurements','block_size','threshold','batch_size','max_fires','counter_bits',
                'leak','scale','scale_ema','min_scale','zero_rate','gain','pool_shape','train_size','val_size',
                'test_size','data_seed','seed','batch_seed','threads','eval_every']:
        assert cfg[key]==getattr(a,key),(c,seed,key)
    metrics=read(out/'metrics.csv'); assert len(metrics)==12000
    selected=np.zeros((12000,len(m.shapes)),dtype=np.int16);fires=np.zeros_like(selected)
    with (out/'layer_metrics.csv').open() as f:
        for index,r in enumerate(csv.DictReader(f)):
            step,layer=divmod(index,len(m.shapes))
            assert int(r['step'])==step+1 and int(r['layer'])==layer
            assert int(r['parameters'])==math.prod(m.shapes[layer])
            selected[step,layer]=int(r['selected_coordinates']);fires[step,layer]=int(r['fires'])
            assert int(r['selected_interval'])==int(selected[step,layer]>0)
            assert int(r['fire_interval'])==int(fires[step,layer]>0)
    assert index+1==12000*len(m.shapes)
    assert (selected.sum(1)==16).all() and (fires.sum(1)<=1).all() and (fires<=selected).all()
    y=np.load(out/'abs_y.npy').astype('float64')
    assert y.shape==(12000,64) and np.isfinite(y).all()
    scale=.02;hist_count=0
    for i,r in enumerate(metrics):
        assert int(r['step'])==i+1 and float(r['scale'])==scale
        scale=max(1e-5,.9*scale+.1*sorted(y[i])[32])
        assert float(r['abs_y_mean'])==float(y[i].mean())
        assert int(r['zero_difference_count'])==int((y[i]==0).sum())
        assert float(r['zero_difference_fraction'])==float((y[i]==0).mean())
        assert int(r['fires'])==int(fires[i].sum())
        assert int(r['train_forward_calls'])==128*(i+1)
        assert int(r['counter_capacity'])==127 and int(r['counter_peak_abs'])<=64
        assert float(r['saturation_rate'])==0
        assert bool(r['val_accuracy'])==((i+1)%500==0)
        hist_count+=int(r['counter_count'])
    assert sum(s['counter_histogram'].values())==hist_count
    assert s['train_forward_calls']==1536000 and s['test_evaluations']==1
    assert s['diagnostic_probe_forward_calls']==len(m.shapes)*256
    assert s['total_forward_calls']==1536000+25+10+len(m.shapes)*256
    assert s['total_forward_examples']==1536000*128+35000+len(m.shapes)*256*128
    ck=torch.load(out/'model.pt',weights_only=False,map_location='cpu')
    assert json.loads(json.dumps(ck['config']))==cfg
    assert ck['weights'].dtype==torch.int8 and ck['weights'].numel()==count
    assert set(ck['weights'].unique().tolist()) <= {-1,0,1}
    (x,labels),(vx,vy),_=load_data(a,torch.device('cpu'))
    replay=[]
    for stage,step in [('initial',0),('final',12000)]:
        if stage=='final':m.weights.copy_(ck['weights'])
        val,sr,ar,rr=observe(m,vx,vy,step)
        assert val==s[stage+'_validation'],(c,seed,stage,'validation')
        for name,actual in [('signal.csv',sr),('activation.csv',ar),('rms_ratios.csv',rr)]:
            saved=[r for r in read(out/name) if int(r['step'])==step]
            assert len(saved)==len(actual)
            for first,second in zip(actual,saved):
                assert {k:'' if v is None else str(v) for k,v in first.items()}==second,(c,seed,stage,name)
        actual=probes(m,x,labels,seed,stage)
        saved=[r for r in read(out/'probes.csv') if r['stage']==stage]
        assert len(saved)==len(m.shapes)*64
        for first,second in zip(actual,saved):
            assert {k:str(v) for k,v in first.items()}==second,(c,seed,stage,'probes')
        replay.append(dict(stage=stage,validation=val,probe_pairs=len(actual)))
    firing=[];conditioned=[]
    for layer,t in enumerate(s['layer_totals']):
        assert t['fires']==int(fires[:,layer].sum())
        assert t['fire_intervals']==int((fires[:,layer]>0).sum())
        assert t['selected_intervals']==int((selected[:,layer]>0).sum())
        assert t['selected_coordinates']==int(selected[:,layer].sum())
        firing.append(dict(condition=c,seed=seed,layer=layer,matrix=m.matrix_names[layer],parameters=math.prod(m.shapes[layer]),**t,
            all_interval_firing_rate=t['fire_intervals']/12000,
            selected_interval_firing_rate=t['fire_intervals']/t['selected_intervals'] if t['selected_intervals'] else None))
        vals=y[selected[:,layer]>0].ravel()
        conditioned.append(dict(condition=c,seed=seed,layer=layer,mean_abs_y=float(vals.mean()),
            zero_fraction=float((vals==0).mean()),selected_intervals=len(vals)//64))
    audit=dict(condition=c,seed=seed,passed=True,validation_and_probes_replay=replay,
        audit_forward_calls=m.forward_calls,test_replayed=False)
    return s,firing,conditioned,audit


def bp_audit(root,c,seed):
    out=root/'per_seed'/f'{c}-seed{seed}';verify_manifest(out)
    s=json.loads((out/'summary.json').read_text())
    assert len(s['attempts']) in (1,2)
    if len(s['attempts'])==2:assert c=='E20c' and s['attempts'][0]['status']=='failed'
    audits=[]
    initial_hash=None
    for attempt,summary in enumerate(s['attempts']):
        d=out/f'attempt{attempt}';verify_manifest(d)
        cfg=json.loads((d/'config.json').read_text())
        assert cfg['condition']==c and cfg['seed']==seed and cfg['batch_seed']==seed+100000
        assert cfg['num_params']==100016 and cfg['blocks']==8 and cfg['width']==76
        assert cfg['learning_rate']==(.001 if attempt==0 else .0003)
        assert cfg['gradient_clip_norm']==(None if attempt==0 else 1.)
        assert cfg['minimum_epochs']==30 and cfg['max_epochs']==100 and cfg['early_stopping_patience']==20
        assert cfg['rmsnorm_eps']==1e-8 and cfg['rmsnorm_trainable'] is False
        m=BPResidual(c,seed)
        assert model_digest(m)==cfg['initial_weight_sha256']
        if initial_hash is None:initial_hash=model_digest(m)
        else:assert model_digest(m)==initial_hash
        a=config(seed,cfg['data_dir'])
        _,(vx,vy),_=load_data(a,torch.device('cpu'))
        with torch.no_grad():initial=evaluate(m,vx,vy)
        assert initial==summary['initial_validation']
        history=read(d/'training.csv');grad=read(d/'gradient_metrics.csv')
        assert len(grad)==18*len(history)
        for ep in range(1,len(history)+1):
            gs=[r for r in grad if int(r['epoch'])==ep]
            assert sorted(int(r['layer']) for r in gs)==list(range(18))
        if summary['status']=='success':
            selected=min(history,key=lambda r:float(r['val_loss']))
            assert int(selected['epoch'])==summary['best_epoch']
            best=float('inf');stale=0;best_epoch=0
            dummy=torch.nn.Parameter(torch.zeros(1))
            optimizer=torch.optim.Adam([dummy],lr=cfg['learning_rate'])
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=5,min_lr=1e-5)
            for index,r in enumerate(history):
                assert int(r['epoch'])==index+1 and int(r['optimizer_updates'])==79*(index+1)
                assert float(r['lr'])==optimizer.param_groups[0]['lr']
                v=float(r['val_loss']);scheduler.step(v)
                if v<best:best=v;best_epoch=index+1;stale=0
                else:stale+=1
                assert int(r['best_epoch'])==best_epoch
                if index+1<len(history):assert not (index+1>=30 and stale>=20)
            assert len(history)==100 or (len(history)>=30 and stale>=20)
            assert summary['test_evaluations']==1
            ck=torch.load(d/'model.pt',weights_only=False,map_location='cpu')
            assert json.loads(json.dumps(ck['config']))==cfg
            m.load_state_dict(ck['state_dict'])
            with torch.no_grad():
                val,sr,ar,rr=observe(m,vx,vy,summary['best_epoch'])
            assert val==summary['selected_validation']
            for name,actual in [('selected_signal.csv',sr),('selected_activation.csv',ar),('selected_rms_ratios.csv',rr)]:
                saved=read(d/name);assert len(saved)==len(actual)
                for first,second in zip(actual,saved):
                    assert {k:'' if v is None else str(v) for k,v in first.items()}==second,(c,seed,attempt,name)
            with np.load(d/'selected_validation_predictions.npz') as pp:
                with torch.no_grad():assert np.array_equal(m(vx).argmax(1).numpy(),pp['predictions'])
            if c=='E20c':
                q=torch.load(d/'quantized_model.pt',weights_only=False,map_location='cpu')
                for i,w in enumerate(m.latent):
                    effective,codes,alpha=ternary_weight(w)
                    assert torch.equal(q['codes'][i],codes) and codes.dtype==torch.int8
                    assert torch.equal(q['alphas'][i],alpha) and torch.equal(q['effective_weights'][i],effective)
            assert all(torch.isfinite(w).all() and w.dtype==torch.float32 for w in m.latent)
            assert model_digest(m)==summary['final_weight_sha256']
        else:
            assert summary['test'] is None and summary['test_evaluations']==0
            assert (d/'failed_model.pt').exists()
        audits.append(dict(attempt=attempt,status=summary['status'],passed=True,
            epochs=len(history),test_replayed=False))
    return s,dict(condition=c,seed=seed,passed=True,attempts=audits,initial_weight_sha256=initial_hash)


def main(root):
    setup()
    state=json.loads((root/'status.json').read_text())
    assert state['training_complete'] and state['completed']==24
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest['sources'].items():
        assert sha(root/'sources'/name)==digest and sha(HERE/name)==digest
    e17manifest=json.loads((E17/'manifest.json').read_text())
    assert manifest['data_sha256']==e17manifest['data_sha256']
    for name in ['train.py','activation_quantization.py','depth_diagnostics.py','residual_stream.py','run_residual_e17.py']:
        assert manifest['sources'][name]==sha(E17/'sources'/name)
    rows=[];audits=[];firing=[];conditioned=[];signals=[];ratios=[];activation=[];isolated=[];curves=[]
    bp_initial={};bp_grad=[];bp_signals=[];bp_ratios=[]
    for c in ALL_CONDITIONS:
        for seed in range(3):
            if c in TDT_CONDITIONS:
                s,fr,co,au=tdt_audit(root,c,seed)
                firing.extend(fr);conditioned.extend(co)
                out=root/'per_seed'/f'{c}-seed{seed}'
                b,d,p,n=TDT_CONDITIONS[c]
                y=np.load(out/'abs_y.npy')
                rows.append(dict(condition=c,seed=seed,status='success',blocks=b,width=d,num_params=n,
                    test_accuracy_percent=100*s['test']['accuracy'],val_accuracy_percent=100*s['final_validation']['accuracy'],
                    elapsed_seconds=s['elapsed_seconds'],selected_attempt=None,epochs=None,best_epoch=None,
                    mean_abs_y=float(y.astype('float64').mean()),zero_difference_fraction=float((y==0).mean())))
                for name,target in [('signal',signals),('rms_ratios',ratios),('activation',activation)]:
                    target.extend(dict(condition=c,seed=seed,**r) for r in read(out/f'{name}.csv'))
                pp=read(out/'probes.csv')
                for stage in ['initial','final']:
                    for layer in range(2*b+2):
                        vs=[float(r['abs_y']) for r in pp if r['stage']==stage and int(r['layer'])==layer]
                        isolated.append(dict(condition=c,seed=seed,stage=stage,layer=layer,
                            mean_abs_y=statistics.mean(vs),sample_std_abs_y=statistics.stdev(vs)))
                curves.append(dict(condition=c,seed=seed,step=0,val_accuracy=s['initial_validation']['accuracy']))
                curves.extend(dict(condition=c,seed=seed,step=int(r['step']),val_accuracy=float(r['val_accuracy']))
                    for r in read(out/'metrics.csv') if r['val_accuracy'])
            else:
                s,au=bp_audit(root,c,seed)
                if seed in bp_initial:assert bp_initial[seed]==au['initial_weight_sha256']
                else:bp_initial[seed]=au['initial_weight_sha256']
                selected=s['attempts'][s['selected_attempt']] if s['status']=='success' else None
                rows.append(dict(condition=c,seed=seed,status=s['status'],blocks=8,width=76,num_params=100016,
                    test_accuracy_percent=100*s['test']['accuracy'] if selected else None,
                    val_accuracy_percent=100*selected['selected_validation']['accuracy'] if selected else None,
                    elapsed_seconds=s['elapsed_seconds'],selected_attempt=s['selected_attempt'],
                    epochs=selected['epochs'] if selected else None,best_epoch=selected['best_epoch'] if selected else None,
                    mean_abs_y=None,zero_difference_fraction=None))
                for i,attempt in enumerate(s['attempts']):
                    out=root/'per_seed'/f'{c}-seed{seed}'/f'attempt{i}'
                    for name,target in [('gradient_metrics',bp_grad),('signal',bp_signals),('rms_ratios',bp_ratios)]:
                        target.extend(dict(condition=c,seed=seed,attempt=i,**r) for r in read(out/f'{name}.csv'))
            audits.append(au)
            print(f'Audited {c} seed{seed}',flush=True)
    for name,data in [('per_seed/results.csv',rows),('firing/matrices.csv',firing),
        ('signal/conditioned_candidates.csv',conditioned),('signal/metrics.csv',signals),
        ('signal/rms_ratios.csv',ratios),('activation/metrics.csv',activation),
        ('signal/isolated_candidates.csv',isolated),('aggregate/validation_curves.csv',curves),
        ('gradient/metrics.csv',bp_grad),('gradient/signals.csv',bp_signals),('gradient/rms_ratios.csv',bp_ratios)]:
        write_csv(root/name,data)
    aggregate=[]
    for c in ALL_CONDITIONS:
        rs=[r for r in rows if r['condition']==c];vs=[r['test_accuracy_percent'] for r in rs if r['status']=='success']
        aggregate.append(dict(condition=c,successful_seeds=len(vs),failed_seeds=3-len(vs),
            test_mean_percent=statistics.mean(vs) if len(vs)==3 else None,
            test_sample_std_percent=statistics.stdev(vs) if len(vs)==3 else None,
            runtime_mean_seconds=statistics.mean(r['elapsed_seconds'] for r in rs)))
    write_csv(root/'aggregate/results.csv',aggregate)
    for name,keys,metrics in [('firing/matrices.csv',['condition','layer','matrix','parameters'],['fires','all_interval_firing_rate','selected_interval_firing_rate']),
        ('signal/rms_ratios.csv',['condition','step','block'],['stream_rms','branch_rms','branch_stream_rms_ratio']),
        ('signal/metrics.csv',['condition','step','layer','stage'],['rms']),
        ('signal/isolated_candidates.csv',['condition','stage','layer'],['mean_abs_y']),
        ('activation/metrics.csv',['condition','step','layer'],['relative_squared_error','cosine_mean_valid'])]:
        path=root/name
        write_csv(path.with_name(path.stem+'_aggregate.csv'),aggregate_rows(read(path),keys,metrics))
    e17=read(E17/'per_seed/results.csv')
    baseline={int(r['seed']):float(r['test_accuracy_percent']) for r in e17 if r['condition']=='E17a'}
    def values(c):return [next(r['test_accuracy_percent'] for r in rows if r['condition']==c and r['seed']==seed) for seed in range(3)]
    def paired(label,left,right):
        diffs=[a-b for a,b in zip(left,right)] if all(v is not None for v in left+right) else None
        return dict(comparison=label,mean_pp=statistics.mean(diffs) if diffs else None,
            sample_std_pp=statistics.stdev(diffs) if diffs else None,
            seed0_pp=diffs[0] if diffs else None,seed1_pp=diffs[1] if diffs else None,seed2_pp=diffs[2] if diffs else None)
    base=[baseline[s] for s in range(3)]
    effects=[paired('E18a_minus_E18d',values('E18a'),values('E18d')),
        paired('E17a_minus_E19a_A4_cost',base,values('E19a')),
        paired('E20a_minus_E20b_A8_cost',values('E20a'),values('E20b')),
        paired('E20b_minus_E20c_W3_cost',values('E20b'),values('E20c')),
        paired('E20c_minus_E17a_TDT_comparison',values('E20c'),base),
        paired('E20a_minus_E14_reference',values('E20a'),[93.89]*3)]
    write_csv(root/'aggregate/paired_effects.csv',effects)
    if all(r['mean_pp'] is not None for r in effects[2:5]):
        assert math.isclose(sum(r['mean_pp'] for r in effects[2:5]),statistics.mean(values('E20a'))-statistics.mean(base),abs_tol=1e-12)
    depth=[dict(condition=c,mean_test_percent=statistics.mean(values(c)),
        change_from_fixed_e17_pp=statistics.mean(values(c))-90.637,
        passed=statistics.mean(values(c))>=89.637) for c in ['E18a','E18b','E18c']]
    a4cost=90.637-statistics.mean(values('E19a'))
    criteria=dict(E18=dict(passed=all(r['passed'] for r in depth),conditions=depth),
        E19=dict(passed=a4cost<=3 and min(values('E19a'))>67.17,A4_cost_pp=a4cost,serial_A4_cost_pp=19.86,
            all_seeds_above_67_17=min(values('E19a'))>67.17),
        E20=dict(all_conditions_three_seed_success=all(r['successful_seeds']==3 for r in aggregate if r['condition'].startswith('E20')),
            note='No accuracy pass cutoff was specified; report decomposition and failures without selecting conditions'))
    write_csv(root/'aggregate/depth_criteria.csv',depth)
    warnings=[]
    for c in ['E17a',*TDT_CONDITIONS]:
        rr=[r for r in (read(E17/'signal/rms_ratios.csv') if c=='E17a' else ratios) if r['condition']==c and int(r['step'])==12000]
        ss=[r for r in (read(E17/'signal/metrics.csv') if c=='E17a' else signals) if r['condition']==c and int(r['step'])==12000]
        layers=18 if c=='E17a' else 2*TDT_CONDITIONS[c][0]+2
        logits=[float(r['rms']) for r in ss if int(r['layer'])==layers-1 and r['stage']=='output']
        count=sum(float(r['branch_stream_rms_ratio'])>.5 for r in rr)
        warnings.append(dict(condition=c,ratio_exceed_count=count,ratio_total=len(rr),ratio_exceed_fraction=count/len(rr),
            logits_exceed_count=sum(v>10 for v in logits),logits_rms_mean=statistics.mean(logits),logits_rms_max=max(logits)))
    write_csv(root/'signal/scale_warnings.csv',warnings)
    dump(root/'criteria.json',criteria)
    lines=['# E18 / E19 / E20 結果','','全24 run。testは最終モデル/validation選択済みモデルでのみ評価。平均±標本標準偏差。', '',
        '| 条件 | test (%) | 成功seed | 実時間平均（分） |','| --- | ---: | ---: | ---: |',
        '| E17a（再利用） | 90.6367 ± 0.4300 | 3 | 36.38 |']
    for r in aggregate:
        score=f"{r['test_mean_percent']:.4f} ± {r['test_sample_std_percent']:.4f}" if r['successful_seeds']==3 else '失敗seedあり・正式3seed平均なし'
        lines.append(f"| {r['condition']} | {score} | {r['successful_seeds']} | {r['runtime_mean_seconds']/60:.2f} |")
    lines+=['',f"E18主判定: {'合格（深さ問題は反転した）' if criteria['E18']['passed'] else '未達（低下条件で深さ問題は残存）'}。閾値は固定参照90.637%−1.0pt。",
        f"E19主判定: {'合格（残差設計でA4は実用域）' if criteria['E19']['passed'] else '未達'}。A4コスト{a4cost:.4f}pt、直列設計の19.86ptと比較。",'',
        '| 比較 | 差（pt、対応seed平均±標本SD） |','| --- | ---: |']
    for r in effects:
        value=f"{r['mean_pp']:+.4f} ± {r['sample_std_pp']:.4f}" if r['mean_pp'] is not None else '失敗seedのため分解不可'
        lines.append(f"| {r['comparison']} | {value} |")
    lines+=['','E20c−E17aには学習則だけでなくW3スケール・初期化・予算・選択方法の差が含まれる。純粋な因果分離とは解釈しない。E18a−E18dも幅の変更を含む。',
        'E18dは指定の幅54・98,712重みを維持し、100k比−1.288%の例外を承認記録に明記。',
        'スケール警告はsignal/scale_warnings.csv。E18cを含め、分母の異なる深さを件数と割合の両方で比較する。',
        '全epochのBP勾配・RMSはgradient/、TDT全区間発火はper_seed各run/layer_metrics.csv、層別集計はfiring/とsignal/。',
        'E20失敗と救済の全attemptはper_seed配下。資源共有はruntime_workers.json。testによる救済・選択・再学習は行っていない。']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    shutil.copy2(Path(__file__),root/'sources'/Path(__file__).name)
    dump(root/'audit.json',dict(passed=True,runs=audits,test_replayed=False,source_sha256=sha(Path(__file__))))
    dump(root/'status.json',dict(complete=True,training_complete=True,completed=24,expected=24,errors=[],audited=True))
    dump(root/'artifacts_sha256.json',{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='artifacts_sha256.json'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();main(a.root)
