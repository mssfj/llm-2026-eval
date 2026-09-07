"""Execute all three activation conditions, then audit and compare saved results."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,sys,json,time
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'tdt_mnist/results/depth-activation-100k-20260907'
RUNS=ROOT/'tdt_mnist/runs/depth-activation-100k-20260907'
CONDITIONS=[('relu-a3-threshold','relu','a3'),('identity-a32','identity','a32'),('identity-a3-threshold','identity','a3')]
def run(c):
    label,activation,precision=c
    cmd=[sys.executable,str(ROOT/'tdt_mnist/sweep_depth_activation.py'),'--hidden-activation',activation,
         '--activation-precision',precision,'--a3-method',('mean_threshold' if precision=='a3' else 'absmax'),'--workers','4',
         '--report-dir',str(REPORT/label),'--output-dir',str(RUNS/label)]
    with (REPORT/(label+'.log')).open('w') as f:
        subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    return label
if __name__=='__main__':
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'status.json').write_text(json.dumps({'complete':False,'expected':108,'started':time.time()}))
    try:
        with ThreadPoolExecutor(max_workers=3) as pool: list(pool.map(run,CONDITIONS))
        subprocess.run([sys.executable,str(ROOT/'tdt_mnist/analyze_depth_activation.py'),str(REPORT)],check=True)
        (REPORT/'status.json').write_text(json.dumps({'complete':True,'completed':108,'expected':108,'finished':time.time()}))
    except Exception as e:
        (REPORT/'status.json').write_text(json.dumps({'complete':False,'error':repr(e)}));raise
