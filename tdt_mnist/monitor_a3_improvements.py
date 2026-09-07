"""Read-only progress monitor for the nine approved A3 intervention runs."""
from pathlib import Path
import csv,json,time,argparse
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'tdt_mnist/results/a3-improvements-16layer-20260907'
R=ROOT/'tdt_mnist/runs/a3-improvements-16layer-20260907'
def snapshot():
    out={'utc':time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),'runs':[]}
    for d in sorted(R.iterdir()):
        if not d.is_dir():continue
        if (d/'summary.json').exists():
            s=json.loads((d/'summary.json').read_text());out['runs'].append({'run':d.name,'complete':True,'val':s['final_validation']['accuracy']});continue
        try:
            with (d/'metrics.csv').open('rb') as f:
                header=f.readline().decode();f.seek(0,2);size=f.tell();f.seek(max(0,size-5000));tail=f.read().decode().splitlines()
            r=next(csv.DictReader([header,tail[-2]]));out['runs'].append({'run':d.name,'step':int(r['step']),'seconds':float(r['elapsed_seconds'])})
        except (ValueError,IndexError,FileNotFoundError):pass
    try:out['status']=json.loads((P/'status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):pass
    try:out['verified']=json.loads((P/'verification.json').read_text())['passed']
    except (FileNotFoundError,json.JSONDecodeError):pass
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        s=snapshot();print(json.dumps(s),flush=True)
        if not a.watch or s.get('status',{}).get('complete'):break
        time.sleep(60)
