"""Use spare CPU capacity to probe completed checkpoints while training continues."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,subprocess,sys,time,hashlib
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'tdt_mnist/results/depth-activation-100k-20260907'
RUNS=ROOT/'tdt_mnist/runs/depth-activation-100k-20260907'
DEST=REPORT/'layer_isolated_probes'
def task(run):
    label=run.parent.name+'-'+run.name;dest=DEST/(label+'.csv');meta=DEST/(label+'.json')
    result=subprocess.run([sys.executable,str(ROOT/'tdt_mnist/probe_depth_activation.py'),str(run),str(dest)],capture_output=True,text=True,check=True)
    meta.write_text(result.stdout)
    return label
if __name__=='__main__':
    DEST.mkdir(parents=True,exist_ok=True);pending={};done=set()
    with ThreadPoolExecutor(max_workers=3) as pool:
        while len(done)<108:
            for f in RUNS.glob('*/*/model.pt'):
                run=f.parent;key=str(run)
                if key in done or key in pending or not (run/'summary.json').exists():continue
                pending[key]=pool.submit(task,run)
            for key,future in list(pending.items()):
                if future.done():
                    print(future.result(),flush=True);done.add(key);del pending[key]
            (REPORT/'probe_progress.json').write_text(json.dumps({'completed':len(done),'pending':len(pending),'expected':108}))
            if len(done)<108:time.sleep(20)
