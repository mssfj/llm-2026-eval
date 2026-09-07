"""Read current experiment progress without touching training state."""
from pathlib import Path
import csv,json,time,argparse
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'tdt_mnist/results/depth-activation-100k-20260907'
RUNS=ROOT/'tdt_mnist/runs/depth-activation-100k-20260907'
CONDITIONS=['relu-a3-threshold','identity-a32','identity-a3-threshold']
def snapshot():
    result={'utc':time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),'completed':0,'expected':108,'conditions':{}}
    for condition in CONDITIONS:
        directory=RUNS/condition;done=list(directory.glob('*/summary.json'));active=[]
        for p in directory.glob('*/metrics.csv'):
            if (p.parent/'summary.json').exists():continue
            with p.open('rb') as f:
                header=f.readline().decode();f.seek(0,2);size=f.tell();f.seek(max(0,size-5000));tail=f.read().decode(errors='replace').splitlines()
            try:
                row=next(csv.DictReader([header,tail[-2]])) # Avoid concurrently-written trailing row.
                active.append({'run':p.parent.name,'step':int(row['step']),'seconds':float(row['elapsed_seconds'])})
            except (IndexError,ValueError,TypeError,StopIteration):pass
        result['conditions'][condition]={'completed':len(done),'active':active};result['completed']+=len(done)
    try:result['probes']=json.loads((REPORT/'probe_progress.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):pass
    try:result['final_status']=json.loads((REPORT/'status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):pass
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        r=snapshot();print(json.dumps(r,ensure_ascii=False),flush=True)
        if not a.watch or r.get('final_status',{}).get('complete'):break
        time.sleep(60)
