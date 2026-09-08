"""CPU-authoritative TDT decisions with GPU candidate loss evaluation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import math,time
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import train
from allocation_engines import private_function,buffer_candidate_function
from activation_quantization import encode_activation,decode_activation


def configure_gpu():
    torch.backends.cuda.matmul.fp32_precision='ieee'
    torch.backends.cudnn.conv.fp32_precision='ieee'
    torch.use_deterministic_algorithms(True)

@dataclass
class Plan:
    indices:torch.Tensor
    batches:torch.Tensor
    codes:torch.Tensor


def schedule(model,x,args,generator,batch_generator):
    g=torch.Generator().set_state(generator.get_state())
    bg=torch.Generator().set_state(batch_generator.get_state())
    indices=torch.randperm(model.num_params,generator=g)[:args.block_size]
    candidate=buffer_candidate_function([model.weights.clone(),model.weights.clone()])
    batches=[];codes=[]
    for _ in range(args.measurements):
        batches.append(torch.randint(len(x),(args.batch_size,),generator=bg))
        plus,minus,_,_=candidate(model.weights,indices,g)
        codes.extend([plus[indices].clone(),minus[indices].clone()])
        torch.rand((len(indices),),generator=g)
    return Plan(indices,torch.stack(batches),torch.stack(codes))


class GPUEvaluator:
    def __init__(self,model,x,y,mode,precision='a8'):
        assert mode in ['gpu_sequential','gpu_batched','gpu_graph']
        configure_gpu();self.mode=mode;self.precision=precision;self.shapes=model.shapes;self.blocks=model.blocks
        self.x=x.cuda();self.y=y.cuda();self.cpu_weights=model.weights.clone();self.codes=model.weights.cuda()
        self.scales=torch.cat([torch.full((math.prod(s),),alpha,dtype=torch.float32) for s,alpha in zip(model.shapes,model.scales)]).cuda()
        self.base=self.codes.float()*self.scales
        self.indices=torch.arange(16,device='cuda');self.batches=torch.zeros((64,128),dtype=torch.int64,device='cuda')
        self.candidate_codes=self.codes[:16].expand(128,16).clone()
        self.offsets=[0]
        for s in self.shapes:self.offsets.append(self.offsets[-1]+math.prod(s))
        self.graph=None
        torch.cuda.synchronize()
        if mode=='gpu_graph':
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):self.parallel()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.graph_losses=self.parallel()
            torch.cuda.synchronize()
    def sync_weights(self,model):
        changed=(self.cpu_weights!=model.weights).nonzero().flatten()
        if len(changed):
            indices=changed.cuda();codes=model.weights[changed].cuda()
            self.codes[indices]=codes
            self.base[indices]=codes.float()*self.scales[indices]
            self.cpu_weights[changed]=model.weights[changed]
    def inputs(self,plan):
        self.indices.copy_(plan.indices);self.batches.copy_(plan.batches);self.candidate_codes.copy_(plan.codes)
    def prepared(self):
        weights=self.base.expand(128,-1).clone()
        weights[:,self.indices]=self.candidate_codes.float()*self.scales[self.indices]
        x=self.x[self.batches].repeat_interleave(2,dim=0)
        y=self.y[self.batches].repeat_interleave(2,dim=0)
        return weights,x,y
    def q(self,x):return decode_activation(*encode_activation(x,self.precision))
    @staticmethod
    def norm(x):return x/torch.sqrt(x.square().mean(-1,keepdim=True)+1e-8)
    def parallel(self):
        weights,x,y=self.prepared()
        def linear(x,l):
            w=weights[:,self.offsets[l]:self.offsets[l+1]].view(128,*self.shapes[l])
            return torch.bmm(self.q(x),w.transpose(1,2))
        h=linear(x,0)
        for b in range(self.blocks):h=h+linear(F.relu(linear(self.norm(h),1+2*b)),2+2*b)
        logits=linear(self.norm(h),len(self.shapes)-1)
        return F.cross_entropy(logits.reshape(-1,10),y.flatten(),reduction='none').view(128,128).mean(1)
    def sequential(self):
        weights,x,y=self.prepared();losses=[]
        for p in range(128):
            def linear(z,l):return F.linear(self.q(z),weights[p,self.offsets[l]:self.offsets[l+1]].view(self.shapes[l]))
            h=linear(x[p],0)
            for b in range(self.blocks):h=h+linear(F.relu(linear(self.norm(h),1+2*b)),2+2*b)
            logits=linear(self.norm(h),len(self.shapes)-1)
            losses.append(F.cross_entropy(logits,y[p]))
        return torch.stack(losses)
    @torch.no_grad()
    def evaluate(self,model,plan):
        self.sync_weights(model);self.inputs(plan)
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record()
        if self.mode=='gpu_graph':self.graph.replay();values=self.graph_losses
        elif self.mode=='gpu_batched':values=self.parallel()
        else:values=self.sequential()
        end.record()
        cpu=values.cpu()  # Only the complete128-loss vector crosses to CPU.
        return cpu,dict(gpu_workflow_milliseconds=start.elapsed_time(end))


@torch.no_grad()
def epoch(model,x,y,args,g,scale,bg,evaluator,trace=None):
    t=time.perf_counter();plan=schedule(model,x,args,g,bg);schedule_seconds=time.perf_counter()-t
    losses,timing=evaluator.evaluate(model,plan)
    assert torch.isfinite(losses).all()
    cursor=0
    def loss(m,bx,by,weights=None):
        nonlocal cursor
        if trace is not None:
            assert torch.equal(weights[plan.indices],plan.codes[cursor])
            assert torch.equal(bx,x[plan.batches[cursor//2]]) and torch.equal(by,y[plan.batches[cursor//2]])
        result=losses[cursor];cursor+=1;return result
    bindings={'loss':loss}
    if trace is not None:
        trace.update(losses=list(losses.unbind()),votes=[],counters=[],schedule_indices=plan.indices.clone())
        def accumulate(evidence,indices,edges,signal,generator,leak,capacity):
            old=evidence[indices].clone();result=train.accumulate(evidence,indices,edges,signal,generator,leak,capacity)
            trace['votes'].append((evidence[indices]-old).clone());trace['counters'].append(evidence[indices].clone());return result
        bindings['accumulate']=accumulate
    result=private_function(train.epoch,bindings)(model,x,y,args,g,scale,bg)
    assert cursor==128
    return result,{**timing,'schedule_seconds':schedule_seconds}
