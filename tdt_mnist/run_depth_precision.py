"""Nine A16/A8/A4 runs: 16 layers, 100k weights, no ReLU, threshold 8."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,csv,hashlib,shutil,subprocess,sys,os,argparse,time
from train import parser
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'tdt_mnist/results/depth-precision-16layer-100k-20260907'
R=ROOT/'tdt_mnist/runs/depth-precision-16layer-100k-20260907'
BASE=ROOT/'tdt_mnist/runs/depth-activation-100k-20260907/identity-a32'
CONDITIONS=['a16','a8','a4']
def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    out=Path('/tmp/tdt-depth-precision-smoke-runs') if a.smoke else R
    report=Path('/tmp/tdt-depth-precision-smoke-report') if a.smoke else P
    out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    assert not any(out.iterdir()) and not any(report.iterdir()),'Use empty output directories'
    names=[v.dest for v in parser()._actions if v.dest!='help']
    sources=['train.py','activation_quantization.py','depth_diagnostics.py','a3_improvements.py','run_depth_precision.py','diagnose_depth_precision.py','analyze_depth_precision.py']
    manifest={'conditions':CONDITIONS,'seeds':[0] if a.smoke else [0,1,2],'depth':16,'threshold':8,'smoke':a.smoke,
        'baseline_root':str(BASE),'sources':{n:hashlib.sha256((ROOT/'tdt_mnist'/n).read_bytes()).hexdigest() for n in sources},
        'baseline_config_sha256':{str(seed):hashlib.sha256((BASE/f'depth16-threshold8-seed{seed}'/'config.json').read_bytes()).hexdigest() for seed in [0,1,2]}}
    first=json.loads((BASE/'depth16-threshold8-seed0/config.json').read_text())
    manifest['data_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((Path(first['data_dir'])/'MNIST/raw').glob('*-ubyte'))}
    (report/'manifest.json').write_text(json.dumps(manifest,indent=2));(report/'sources').mkdir()
    for n in sources:shutil.copy2(ROOT/'tdt_mnist'/n,report/'sources'/n)
    def run(condition,seed):
        cfg=json.loads((BASE/f'depth16-threshold8-seed{seed}'/'config.json').read_text())
        run_dir=out/f'{condition}-seed{seed}'
        opts={n:cfg.get(n) for n in names};opts.update(a3_improvement='none',activation_precision=condition,output_dir=str(run_dir),download=False,loss_diagnostics=True,layer_diagnostics=True)
        if a.smoke:opts.update(steps=2,measurements=8,eval_every=1,train_size=64,val_size=32)
        cmd=[sys.executable,str(ROOT/'tdt_mnist/train.py')]
        for n,v in opts.items():
            if v is None:continue
            flag='--'+n.replace('_','-')
            if isinstance(v,bool):cmd.append(flag if v else '--no-'+n.replace('_','-'))
            elif isinstance(v,list):cmd.extend([flag,*map(str,v)])
            else:cmd.extend([flag,str(v)])
        with (out/f'{condition}-seed{seed}.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        return {'condition':condition,'seed':seed,'run_directory':str(run_dir),**json.loads((run_dir/'summary.json').read_text())}
    results=[];errors=[];tasks=[(c,s) for c in CONDITIONS for s in manifest['seeds']]
    def status():
        (report/'status.json').write_text(json.dumps({'complete':False,'training_complete':len(results)==len(tasks) and not errors,'completed':len(results),'expected':len(tasks),'errors':errors},indent=2))
        (report/'summaries.json').write_text(json.dumps(results,indent=2))
    status()
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures={pool.submit(run,*t):t for t in tasks}
        for f in as_completed(futures):
            try:r=f.result();results.append(r);print(f"{len(results)}/{len(tasks)} {futures[f]} val={r['final_validation']['accuracy']}",flush=True)
            except Exception as e:errors.append(f'{futures[f]}: {e}');print(errors[-1],flush=True)
            status()
    if errors:raise RuntimeError(errors)
    if not a.smoke:
        subprocess.run([sys.executable,str(ROOT/'tdt_mnist/analyze_depth_precision.py'),str(report)],check=True)
        (report/'status.json').write_text(json.dumps({'complete':True,'training_complete':True,'analysis_verified':True,'completed':9,'expected':9,'errors':[],'finished':time.time()},indent=2))
if __name__=='__main__':main()
