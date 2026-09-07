"""Drain only sweep schedulers; never interrupt their active training subprocesses."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os,signal,subprocess,time,json
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'tdt_mnist/results/depth-activation-100k-20260907'
def process(pid):
    p=Path('/proc')/str(pid)
    try:
        cmd=(p/'cmdline').read_bytes().split(b'\0');cmd=[s.decode() for s in cmd if s]
        stat=(p/'stat').read_text().rsplit(')',1)[1].split();return cmd,int(stat[1]),stat[0]
    except (FileNotFoundError,ProcessLookupError):return [],-1,'X'
def drain_and_resume(pid):
    cmd,_,state=process(pid)
    assert len(cmd)>2 and Path(cmd[1]).name=='sweep_depth_activation.py'
    assert cmd[cmd.index('--activation-precision')+1]=='a3'
    assert cmd[cmd.index('--workers')+1]=='4'
    report=Path(cmd[cmd.index('--report-dir')+1]);runs=Path(cmd[cmd.index('--output-dir')+1])
    os.kill(pid,signal.SIGSTOP)
    print('Paused scheduler only',pid,flush=True)
    while True:
        children=[]
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            c,ppid,st=process(int(p.name))
            if ppid==pid and st not in ['X','Z']:children.append(int(p.name))
        if not children:break
        time.sleep(20)
    dirs=[p for p in runs.iterdir() if p.is_dir()]
    assert all((p/'summary.json').exists() and (p/'model.pt').exists() for p in dirs)
    os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
    for _ in range(30):
        if process(pid)[2] in ['X','Z']:break
        time.sleep(1)
    assert process(pid)[2] in ['X','Z']
    cmd[cmd.index('--workers')+1]='6';cmd.append('--resume')
    log=REPORT/(report.name+'-resume.log')
    (report/'runtime_workers.json').write_text(json.dumps({'initial_workers':4,'resumed_workers':6,
        'drained_completed_runs':len(dirs),'training_subprocesses_interrupted':False,'command':cmd},indent=2))
    print('Resuming',report.name,'completed',len(dirs),'workers',6,flush=True)
    with log.open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    print('Finished',report.name,flush=True)
if __name__=='__main__':
    pids=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        cmd,_,state=process(int(p.name))
        if len(cmd)>2 and Path(cmd[1]).name=='sweep_depth_activation.py' and '--activation-precision' in cmd and cmd[cmd.index('--activation-precision')+1]=='a3':pids.append(int(p.name))
    assert len(pids)==2,pids
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(drain_and_resume,pids))
