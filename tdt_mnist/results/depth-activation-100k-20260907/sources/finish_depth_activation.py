"""Finish already-running activation sweeps, preserving all valid runs."""
from pathlib import Path
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'tdt_mnist/results/depth-activation-100k-20260907'
CONDITIONS=['relu-a3-threshold','identity-a32','identity-a3-threshold']
if __name__=='__main__':
    while True:
        statuses={}
        for c in CONDITIONS:
            try:statuses[c]=json.loads((REPORT/c/'status.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError):statuses[c]={'completed':0,'complete':False}
        errors={k:s.get('errors') for k,s in statuses.items() if s.get('errors')}
        progress={'completed':sum(s['completed'] for s in statuses.values()),'expected':108,
            'complete':False,'conditions':statuses,'updated':time.time()}
        (REPORT/'progress.json').write_text(json.dumps(progress,indent=2))
        if errors:raise RuntimeError(errors)
        if all(s['complete'] for s in statuses.values()):break
        time.sleep(30)
    subprocess.run([sys.executable,str(ROOT/'tdt_mnist/analyze_depth_activation.py'),str(REPORT)],check=True)
    (REPORT/'status.json').write_text(json.dumps({'complete':True,'completed':108,'expected':108,
        'analysis_verified':True,'finished':time.time()},indent=2))
    print('Complete: training, layer RMS, perturbation-layer |y| and independent layer probes.',flush=True)
