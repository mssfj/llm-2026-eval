"""Read-only initial/final activation diagnostics and layer-isolated probes."""
from pathlib import Path
from types import SimpleNamespace
import json,csv,hashlib,argparse
import torch,numpy as np
from train import TernaryModel,load_data,evaluate,candidate_pair,loss
from depth_diagnostics import SignalObserver
from activation_quantization import ActivationObserver

def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def diagnose(run,out):
    torch.set_num_threads(1);torch.set_grad_enabled(False);torch.use_deterministic_algorithms(True)
    cfg=json.loads((run/'config.json').read_text());summary=json.loads((run/'summary.json').read_text())
    args=SimpleNamespace(**cfg);args.data_dir=Path(args.data_dir);args.download=False
    kwargs={k:cfg[k] for k in ['pool_size','hidden_size','zero_rate','gain','device','seed','pool_shape','activation_precision','hidden_activation','a3_method','a3_threshold_factor','hidden_sizes']}
    model=TernaryModel(**kwargs,a3_improvement=cfg.get('a3_improvement','none'));model.extended_diagnostics=True
    initial=model.weights.clone();checkpoint=torch.load(run/'model.pt',map_location='cpu',weights_only=False)
    assert model.num_params==100000 and checkpoint['weights'].numel()==100000
    assert set(checkpoint['weights'].unique().tolist()).issubset({-1,0,1})
    assert json.loads(json.dumps(checkpoint['config']))==cfg
    (x,y),(vx,vy),_=load_data(args,model.device)
    activation=[];signals=[];replay=[]
    for stage,weights in [('initial',initial),('final',checkpoint['weights'])]:
        model.weights.copy_(weights);model.activation_observer=ActivationObserver(16,cfg['activation_precision']);model.signal_observer=SignalObserver()
        val=evaluate(model,vx,vy)
        assert val==summary['initial_validation' if stage=='initial' else 'final_validation'],(run,stage,val)
        replay.append({'stage':stage,**val})
        for r in model.activation_observer.summary():
            codes=r.pop('code_histogram');activation.append({'stage':stage,**r,
                'code_histogram':json.dumps(codes,sort_keys=True),'integer_code_zero_fraction':int(codes.get('0',0))/r['values'] if codes else None})
        signals.extend({'checkpoint_stage':stage,**r} for r in model.signal_observer.summary())
        model.activation_observer=None;model.signal_observer=None
    probes=[];offset=0;start_calls=model.forward_calls
    for layer,shape in enumerate(model.shapes):
        size=shape[0]*shape[1]
        for stage,weights in [('initial',initial),('final',checkpoint['weights'])]:
            model.weights.copy_(weights)
            g=torch.Generator().manual_seed(700000+args.seed*1000+16*20+layer)
            bg=torch.Generator().manual_seed(900000+args.seed*1000+16*20+layer)
            for pair in range(64):
                indices=torch.randperm(size,generator=g)[:16]+offset
                plus,minus,_,_=candidate_pair(model.weights,indices,g)
                batch=torch.randint(len(x),(128,),generator=bg)
                lp=loss(model,x[batch],y[batch],plus);lm=loss(model,x[batch],y[batch],minus);absolute=float((lp-lm).abs())
                assert np.isfinite(absolute)
                probes.append({'stage':stage,'layer':layer,'pair':pair,'perturbed_coordinates':16,
                    'loss_plus':float(lp),'loss_minus':float(lm),'abs_y':absolute})
        offset+=size
    assert torch.equal(model.weights,checkpoint['weights'])
    out.mkdir(parents=True,exist_ok=True)
    write(out/'activation.csv',activation);write(out/'signals.csv',signals);write(out/'probes.csv',probes)
    audit={'passed':True,'run':str(run),'validation_replay':replay,'probe_pairs':len(probes),
        'probe_forward_calls':model.forward_calls-start_calls,'checkpoint_sha256':hashlib.sha256((run/'model.pt').read_bytes()).hexdigest(),
        'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')}}
    (out/'verification.json').write_text(json.dumps(audit,indent=2));return audit
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('out',type=Path);a=p.parse_args();print(json.dumps(diagnose(a.run,a.out)))
