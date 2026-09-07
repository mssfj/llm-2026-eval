"""Read-only layer-isolated candidate-pair probes on initial/final checkpoints."""
import argparse,csv,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
import numpy as np
from train import TernaryModel,load_data,candidate_pair,loss

def probe(directory, destination, pairs=64):
    torch.set_num_threads(1);torch.set_grad_enabled(False);torch.use_deterministic_algorithms(True)
    config=json.loads((directory/'config.json').read_text())
    args=SimpleNamespace(**config);args.data_dir=Path(args.data_dir);args.download=False
    model=TernaryModel(**{k:config[k] for k in ['pool_size','hidden_size','zero_rate','gain','device','seed','pool_shape',
        'activation_precision','hidden_activation','a3_method','a3_threshold_factor','hidden_sizes']})
    initial=model.weights.clone()
    checkpoint=torch.load(directory/'model.pt',map_location='cpu',weights_only=False)
    final=checkpoint['weights'];(x,y),_,_=load_data(args,model.device)
    rows=[];offset=0
    for layer,shape in enumerate(model.shapes):
        size=shape[0]*shape[1]
        for stage,weights in [('initial',initial),('final',final)]:
            model.weights.copy_(weights)
            # Reset generators: same batch, coordinates and random orientations for both stages and methods.
            g=torch.Generator().manual_seed(700000+args.seed*1000+len(model.shapes)*20+layer)
            bg=torch.Generator().manual_seed(900000+args.seed*1000+len(model.shapes)*20+layer)
            for pair in range(pairs):
                indices=torch.randperm(size,generator=g)[:args.block_size]+offset
                plus,minus,_,_=candidate_pair(model.weights,indices,g)
                batch=torch.randint(len(x),(args.batch_size,),generator=bg)
                lp=loss(model,x[batch],y[batch],plus);lm=loss(model,x[batch],y[batch],minus)
                # FP32 subtraction matches training; Python subtraction of scalar exports need not.
                absolute=float((lp-lm).abs())
                assert np.isfinite(absolute)
                rows.append({'stage':stage,'layer':layer,'pair':pair,'perturbed_coordinates':len(indices),
                    'loss_plus':float(lp),'loss_minus':float(lm),'abs_y':absolute})
        offset+=size
    assert torch.equal(model.weights,final)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return {'run':str(directory),'probe_pairs':len(rows),'probe_forward_calls':model.forward_calls,
        'checkpoint_sha256':hashlib.sha256((directory/'model.pt').read_bytes()).hexdigest(),
        'csv_sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('destination',type=Path);p.add_argument('--pairs',type=int,default=64)
    a=p.parse_args();print(json.dumps(probe(a.run,a.destination,a.pairs)))
