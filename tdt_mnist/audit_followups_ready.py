"""Audit immutable completed runs while other runs train; no test evaluation."""
import json,time
from analyze_residual_followups import ROOT,ALL_CONDITIONS,TDT_CONDITIONS,setup,tdt_audit_cached,bp_audit_cached,dump
setup()
finished=[]
while len(finished)<24:
    for c in ALL_CONDITIONS:
        for seed in range(3):
            if (c,seed) in finished:continue
            out=ROOT/'per_seed'/f'{c}-seed{seed}'
            if not (out/'manifest.json').exists():continue
            result=(tdt_audit_cached if c in TDT_CONDITIONS else bp_audit_cached)(ROOT,c,seed)
            assert result[-1]['passed']
            finished.append((c,seed))
            dump(ROOT/'audit_progress.json',dict(completed=len(finished),expected=24,finished=finished,test_replayed=False))
            print(f'Independent audit passed: {c} seed{seed} ({len(finished)}/24)',flush=True)
    if len(finished)<24:time.sleep(15)
print('All 24 independent audits complete',flush=True)
