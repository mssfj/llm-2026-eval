"""Saved-weight equivalence, including every residual restart boundary."""
import json,math,types
from pathlib import Path
import torch
import train
from fast_engine import Schedule,candidate_losses,epoch,schedule
from residual_stream import ResidualStreamModel
from run_residual_e17 import config,setup,load_data,dump
ROOT=Path(__file__).resolve().parent
OLD=ROOT/'results/residual-stream-a8-e17-20260908'
OUT=ROOT/'results/fast-engine-e17a-20260908'

def decide(m,plan,losses,scale=.02):
    evidence=torch.zeros((m.num_params,2),dtype=torch.int8);counts=torch.zeros((m.num_params,2),dtype=torch.int32)
    g=torch.Generator().manual_seed(777);votes=[]
    # Fixed, identical rounding variates for both loss lists.
    for k in range(64):
        plus,minus=plan.weights[2*k:2*k+2]
        lo=torch.minimum(plus[plan.indices],minus[plan.indices]).long();edges=lo+1
        phi=(plus[plan.indices]-minus[plan.indices]).long()
        before=evidence.clone()
        train.accumulate(evidence,plan.indices,edges,-(losses[2*k]-losses[2*k+1])*phi/scale,g,1.,127)
        votes.append((evidence-before)[plan.indices].clone())
        counts[plan.indices,edges]+=1
    proposal,n=train.select_actions(m.weights,evidence,counts,plan.indices,8,1,scale)
    return torch.stack(votes),evidence,proposal,n

def main():
    setup();a=config(0,ROOT/'data');(x,y),_,_=load_data(a,torch.device('cpu'))
    m=ResidualStreamModel();state=torch.load(OLD/'per_seed/E17a-seed0/model.pt',weights_only=False)
    m.weights.copy_(state['weights'] if isinstance(state,dict) else state)
    g=torch.Generator().manual_seed(913);indices=torch.randperm(m.num_params,generator=g)[:16]
    tests=[('global',indices)];offset=0
    for l,s in enumerate(m.shapes):
        tests.append((f'matrix{l}',torch.randperm(math.prod(s),generator=g)[:16]+offset));offset+=math.prod(s)
    tests.append(('same_output_row',torch.arange(16)))
    rows=[]
    for name,indices in tests:
        g=torch.Generator().manual_seed(117);ws=[]
        for k in range(64):ws.extend(train.candidate_pair(m.weights,indices,g)[:2])
        plan=Schedule(indices,torch.arange(128).repeat(64,1),torch.stack(ws))
        naive=torch.stack([train.loss(m,x[:128],y[:128],w) for w in ws])
        fast,meta=candidate_losses(m,x,y,plan)
        rel=((fast-naive).abs()/naive.abs().clamp_min(1e-30));n=decide(m,plan,naive);f=decide(m,plan,fast)
        rows.append(dict(case=name,guard_fallbacks=meta['guard_fallbacks'],max_relative_error=float(rel.max()),loss_failures=int((rel>=1e-5).sum()),vote_mismatches=int((n[0]!=f[0]).sum()),counter_mismatches=int((n[1]!=f[1]).sum()),fire_mismatches=int((n[2]!=f[2]).sum())))
    m.activation_precision='a32'
    plan=schedule(m,x,a,torch.Generator().manual_seed(1),torch.Generator().manual_seed(100000))
    fp32_naive=torch.stack([train.loss(m,x[plan.batches[k//2]],y[plan.batches[k//2]],w) for k,w in enumerate(plan.weights)])
    fp32_fast,_=candidate_losses(m,x,y,plan,guard=False)
    algebra_error=float(((fp32_fast-fp32_naive).abs()/fp32_naive.abs()).max())
    m.activation_precision='a8'
    ga=torch.Generator().manual_seed(1);gb=torch.Generator().manual_seed(1);ba=torch.Generator().manual_seed(100000);bb=torch.Generator().manual_seed(100000)
    r1=train.epoch(m,x,y,a,ga,.02,ba);trace={};r2=epoch(m,x,y,a,gb,.02,bb,trace=trace)
    rng_equal=torch.equal(ga.get_state(),gb.get_state()) and torch.equal(ba.get_state(),bb.get_state())
    report=dict(a32_unguarded_algebra_max_relative_error=algebra_error,cases=rows,max_relative_error=max(r['max_relative_error'] for r in rows),rng_equal=rng_equal,real_epoch_proposal_equal=torch.equal(r1[0],r2[0]),real_epoch_scale_absolute_difference=abs(r1[3]-r2[3]))
    report['passed']=rng_equal and report['real_epoch_proposal_equal'] and all(not any(r[k] for k in ['loss_failures','vote_mismatches','counter_mismatches','fire_mismatches']) for r in rows)
    dump(OUT/'level1.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
