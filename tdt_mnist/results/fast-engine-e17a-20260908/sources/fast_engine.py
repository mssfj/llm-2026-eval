"""Batched sparse residual evaluation; legacy epoch decision path is unchanged."""
import math
import types
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import train
from activation_quantization import encode_activation, decode_activation

@dataclass
class Schedule:
    indices: torch.Tensor
    batches: torch.Tensor
    weights: torch.Tensor

def schedule(model, x, args, generator, batch_generator):
    # Replay every RNG consumption, including stochastic-rounding uniforms.
    g = torch.Generator().set_state(generator.get_state())
    bg = torch.Generator().set_state(batch_generator.get_state()) if batch_generator is not None else g
    indices = torch.randperm(model.num_params, generator=g)[:args.block_size]
    batches=[];weights=[]
    for _ in range(args.measurements):
        batches.append(torch.randint(len(x),(args.batch_size,),generator=bg))
        plus,minus,_,_=train.candidate_pair(model.weights,indices,g)
        weights.extend([plus,minus])
        torch.rand((len(indices),),generator=g)
    return Schedule(indices,torch.stack(batches),torch.stack(weights))

def quantize(x, precision):
    q,s=encode_activation(x,precision)
    return decode_activation(q,s)

@torch.no_grad()
def candidate_losses(model, x, y, plan, *, guard=True):
    matrices=model.matrices(model.weights);shapes=model.shapes
    blocks=(len(shapes)-2)//2
    batch=x[plan.batches]; labels=y[plan.batches].repeat_interleave(2,0)
    n,b,_=batch.shape
    ambiguous=torch.zeros(2*n,dtype=torch.bool)
    inputs=[];outputs=[];streams=[]
    def base_linear(z,l):
        q=quantize(z,model.activation_precision);out=F.linear(q,matrices[l])
        inputs.append(q);outputs.append(out);return out
    h=base_linear(batch.reshape(n*b,-1),0)
    for k in range(blocks):
        streams.append(h)
        z=base_linear(model.rmsnorm(h),1+2*k)
        if model.hidden_activation=='relu':z=F.relu(z)
        h=h+base_linear(z,2+2*k)
    base_linear(model.rmsnorm(h),len(shapes)-1)
    offsets=[0]
    for shape in shapes:offsets.append(offsets[-1]+math.prod(shape))
    affected={}
    for l,shape in enumerate(shapes):
        mask=(plan.indices>=offsets[l])&(plan.indices<offsets[l+1])
        idx=plan.indices[mask]
        if len(idx):affected[l]=idx
    first=min(affected)
    def expanded(t):return t.reshape(n,b,-1).repeat_interleave(2,0).reshape(2*n*b,-1)
    def correction(z,l,out):
        idx=affected.get(l)
        if idx is None:return out
        local=idx-offsets[l];rows=local//shapes[l][1];cols=local%shapes[l][1]
        delta=plan.weights[:,idx].float()*model.scales[l]-model.weights[idx].float()*model.scales[l]
        update=z.reshape(2*n,b,-1)[:,:,cols]*delta[:,None,:]
        out.view(2*n,b,-1).index_add_(2,rows,update)
        return out
    def linear(z,l):
        if l==first:
            q=expanded(inputs[l]);out=expanded(outputs[l])
        else:
            if guard and model.activation_precision=='a8':
                scale=z.abs().amax(-1,keepdim=True)/127
                scaled=z/torch.where(scale>0,scale,torch.ones_like(scale))
                near=(scaled.abs().frac()-.5).abs()<1e-4
                ambiguous.logical_or_(near.reshape(2*n,-1).any(-1))
            q=quantize(z,model.activation_precision);out=F.linear(q,matrices[l])
        return correction(q,l,out)
    def finish(logits):
        values=_losses(logits,labels)
        for p in ambiguous.nonzero().flatten().tolist():
            values[p]=train.loss(model,batch[p//2],labels[p],plan.weights[p])
        return values,dict(first=first,matrices=len(shapes),guard_fallbacks=int(ambiguous.sum()),cache_bytes=sum(t.numel()*t.element_size() for t in inputs+outputs+streams))
    if first==0:h=linear(None,0);start=0
    elif first==len(shapes)-1:
        logits=linear(None,first)
        return finish(logits)
    else:
        start=(first-1)//2;h=expanded(streams[start])
    for k in range(start,blocks):
        l=1+2*k
        if first==l+1:z=None
        else:
            z=linear(model.rmsnorm(h),l)
            if model.hidden_activation=='relu':z=F.relu(z)
        h=h+linear(z,l+1)
    logits=linear(model.rmsnorm(h),len(shapes)-1)
    return finish(logits)

def _losses(logits,labels):
    logits=logits.reshape(*labels.shape,10)
    # Each scalar uses the legacy CE reduction to avoid adding reduction drift.
    return torch.stack([F.cross_entropy(l,y) for l,y in zip(logits,labels)])

@torch.no_grad()
def epoch(model,x,y,args,generator,scale,batch_generator=None,*,engine='fast',trace=None):
    if engine=='naive':return train.epoch(model,x,y,args,generator,scale,batch_generator)
    if engine!='fast':raise ValueError(engine)
    plan=schedule(model,x,args,generator,batch_generator)
    losses,metadata=candidate_losses(model,x,y,plan)
    cursor=0
    def cached_loss(m,bx,by,weights=None):
        nonlocal cursor
        # Schedule construction is independent; the legacy path still owns RNG.
        if trace is not None:
            assert torch.equal(weights,plan.weights[cursor])
            assert torch.equal(bx,x[plan.batches[cursor//2]])
            assert torch.equal(by,y[plan.batches[cursor//2]])
        result=losses[cursor];cursor+=1
        return result
    globs=dict(train.epoch.__wrapped__.__globals__);globs['loss']=cached_loss
    original=types.FunctionType(train.epoch.__wrapped__.__code__,globs,'epoch',train.epoch.__wrapped__.__defaults__)
    result=original(model,x,y,args,generator,scale,batch_generator)
    assert cursor==2*args.measurements
    if trace is not None:trace.update(metadata,losses=losses.tolist())
    return result
