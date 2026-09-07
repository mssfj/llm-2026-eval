"""Parameter-free A3 interventions: hidden RMS normalization, residuals, Lloyd iteration."""
import torch
import torch.nn.functional as F

def renormalize(x, epsilon=1e-8):
    rms=x.square().mean(dim=-1,keepdim=True).sqrt()
    gain=1/rms.clamp_min(epsilon)
    return x*gain,gain

def shortcut(x,width):
    if x.shape[-1]>=width:return x[...,:width]
    return F.pad(x,(0,width-x.shape[-1]))

def lloyd_encode(x,max_iterations=5,diagnostics=False):
    magnitude=x.abs()
    sigma=x.std(dim=-1,keepdim=True,unbiased=False)
    threshold=.6*sigma
    selected=magnitude>threshold
    converged=torch.zeros_like(sigma,dtype=torch.bool)
    iterations=torch.zeros_like(sigma)
    for _ in range(max_iterations):
        count=selected.sum(dim=-1,keepdim=True)
        beta=(magnitude*selected).sum(dim=-1,keepdim=True)/count.clamp_min(1)
        threshold=beta/2
        new_selected=magnitude>threshold
        stable=(new_selected==selected).all(dim=-1,keepdim=True)
        iterations+=~converged
        converged|=stable
        selected=new_selected
        if bool(converged.all()):break
    count=selected.sum(dim=-1,keepdim=True)
    beta=(magnitude*selected).sum(dim=-1,keepdim=True)/count.clamp_min(1)
    codes=(x.sign()*selected).to(torch.int8)
    info=None
    if diagnostics:
        stable=((magnitude>beta/2)==selected).all(dim=-1,keepdim=True)
        info={'lloyd_iterations':iterations,'lloyd_unconverged':(~stable).float(),
              'lloyd_threshold_sigma':threshold/sigma.clamp_min(1e-8),
              'lloyd_threshold':threshold,'lloyd_beta':beta}
    return codes,beta,info
