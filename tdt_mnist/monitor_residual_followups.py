"""Read-only progress monitor; never opens test-bearing summaries."""
import argparse,json,time
from pathlib import Path
from run_residual_followups import ROOT,TDT_CONDITIONS


def progress(root):
    state=json.loads((root/'status.json').read_text())
    rows={}
    for c in TDT_CONDITIONS:
        values=[]
        for seed in range(3):
            p=root/'per_seed'/f'{c}-seed{seed}'/'progress.json'
            if p.exists():
                v=json.loads(p.read_text());values.append(dict(seed=seed,step=v['step'],val_percent=round(100*v['validation']['accuracy'],2)))
        rows[c]=values
    return dict(utc=time.strftime('%H:%M:%S',time.gmtime()),completed=state['completed'],expected=24,
        errors=state.get('errors',[]),waiting_for_e18d_approval=state.get('waiting_for_e18d_approval',False),progress=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
    while True:
        r=progress(a.root);print(json.dumps(r),flush=True)
        if not a.watch or r['completed']==24 or r['errors']:break
        time.sleep(45)
