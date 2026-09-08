"""Independent follow-up engines. Original GPU evaluator and TDT code untouched."""
import time
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import train
from gpu_evaluation_engines import GPUEvaluator, Plan, epoch as original_epoch
from allocation_engines import private_function

MODES=['gpu_graph','cpu_compact','persistent_candidates','transfer_buffers','fused_graph']

class RecordedEvaluator(GPUEvaluator):
    @torch.no_grad()
    def evaluate(self,model,plan):
        values,timing=super().evaluate(model,plan)
        self.last_losses=values;self.last_plan=plan
        return values,timing
    def accepted(self,model,proposal,indices):pass
    def reset_model(self,model):self.sync_weights(model)


def capture(ev,forward=None):
    forward=forward or ev.parallel
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):forward()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    ev.graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(ev.graph):ev.graph_losses=forward()
    torch.cuda.synchronize()

class PersistentEvaluator(RecordedEvaluator):
    def __init__(self,model,x,y):
        super().__init__(model,x,y,'gpu_batched')
        self.candidate_weights=self.base.expand(128,-1).clone()
        self.previous_indices=self.indices.clone()
        capture(self);self.mode='gpu_graph'
    def sync_weights(self,model):
        changed=(self.cpu_weights!=model.weights).nonzero().flatten()
        super().sync_weights(model)
        if len(changed):
            changed=changed.cuda()
            self.candidate_weights[:,changed]=self.base[changed]
    def prepared(self):
        # Base already contains accepted state; overwrite, never delta-add.
        self.candidate_weights[:,self.previous_indices]=self.base[self.previous_indices]
        self.candidate_weights[:,self.indices]=self.candidate_codes.float()*self.scales[self.indices]
        self.previous_indices.copy_(self.indices)
        x=self.x[self.batches].repeat_interleave(2,dim=0)
        y=self.y[self.batches].repeat_interleave(2,dim=0)
        return self.candidate_weights,x,y

class TransferEvaluator(RecordedEvaluator):
    def __init__(self,model,x,y):
        super().__init__(model,x,y,'gpu_batched')
        self.host=torch.zeros(67728,dtype=torch.uint8,pin_memory=True)
        self.packed=self.host.cuda()
        self.indices=self.packed[:128].view(torch.int64)
        self.batches=self.packed[128:65664].view(torch.int64).view(64,128)
        self.candidate_codes=self.packed[65664:67712].view(torch.int8).view(128,16)
        self.update_index=self.packed[67712:67720].view(torch.int64)
        self.update_code=self.packed[67720:67721].view(torch.int8)
        # Valid unique indices before graph warmup/capture.
        self.host[:128].view(torch.int64).copy_(torch.arange(16))
        self.host[65664:67712].view(torch.int8).view(128,16).copy_(model.weights[:16].expand(128,16))
        self.host[67720:67721].view(torch.int8).copy_(model.weights[:1])
        self.packed.copy_(self.host)
        self.pending=None;self.start=torch.cuda.Event(enable_timing=True);self.end=torch.cuda.Event(enable_timing=True)
        self.download_done=torch.cuda.Event();self.host_losses=torch.empty(128,dtype=torch.float32,pin_memory=True)
        capture(self);self.mode='gpu_graph'
    def prepared(self):
        self.codes[self.update_index]=self.update_code
        self.base[self.update_index]=self.update_code.float()*self.scales[self.update_index]
        return super().prepared()
    def accepted(self,model,proposal,indices):
        local=(model.weights[indices]!=proposal[indices]).nonzero().flatten()
        assert len(local)<=1
        self.pending=None if not len(local) else (int(indices[local[0]]),int(proposal[indices[local[0]]]))
    def reset_model(self,model):
        self.codes.copy_(model.weights);self.base.copy_(self.codes.float()*self.scales)
        self.cpu_weights.copy_(model.weights);self.pending=None;torch.cuda.synchronize()
    @torch.no_grad()
    def evaluate(self,model,plan):
        # Previous evaluate waits for D2H completion; host input is safe to reuse.
        self.host[:128].view(torch.int64).copy_(plan.indices)
        self.host[128:65664].view(torch.int64).view(64,128).copy_(plan.batches)
        self.host[65664:67712].view(torch.int8).view(128,16).copy_(plan.codes)
        index,code=self.pending if self.pending is not None else (0,int(model.weights[0]))
        self.host[67712:67720].view(torch.int64)[0]=index
        self.host[67720:67721].view(torch.int8)[0]=code
        self.packed.copy_(self.host,non_blocking=True)
        self.start.record();self.graph.replay();self.end.record()
        self.host_losses.copy_(self.graph_losses,non_blocking=True)
        self.download_done.record();self.download_done.synchronize()
        # Stable per-interval values for logs; small CPU copy included in timing.
        values=self.host_losses.clone();self.last_losses=values;self.last_plan=plan
        return values,dict(gpu_workflow_milliseconds=self.start.elapsed_time(self.end))

class FusedEvaluator(RecordedEvaluator):
    def __init__(self,model,x,y):
        super().__init__(model,x,y,'gpu_batched')
        self.compiled=torch.compile(self.parallel,fullgraph=True,options={'triton.cudagraphs':False})
        capture(self,self.compiled);self.mode='gpu_graph'

def evaluator(model,x,y,mode):
    if mode in ['gpu_graph','cpu_compact']:return RecordedEvaluator(model,x,y,'gpu_graph')
    return {'persistent_candidates':PersistentEvaluator,'transfer_buffers':TransferEvaluator,'fused_graph':FusedEvaluator}[mode](model,x,y)

@dataclass
class CompactPlan(Plan):
    edges:torch.Tensor
    phi:torch.Tensor
    uniforms:torch.Tensor

def compact_schedule(model,x,args,g,bg):
    indices=torch.randperm(model.num_params,generator=g)[:args.block_size]
    current=model.weights[indices];local=torch.arange(len(indices))
    batches=[];codes=[];edges=[];phis=[];uniforms=[]
    for _ in range(args.measurements):
        batches.append(torch.randint(len(x),(args.batch_size,),generator=bg))
        plus,minus,edge,phi=train.candidate_pair(current,local,g)
        codes.extend([plus,minus]);edges.append(edge);phis.append(phi)
        uniforms.append(torch.rand((len(indices),),generator=g))
    return CompactPlan(indices,torch.stack(batches),torch.stack(codes),torch.stack(edges),torch.stack(phis),torch.stack(uniforms))

class UniformTorch:
    def __init__(self,uniform):self.uniform=uniform
    def __getattr__(self,name):return getattr(torch,name)
    def rand(self,shape,**kw):
        assert tuple(shape)==tuple(self.uniform.shape)
        return self.uniform

@torch.no_grad()
def compact_epoch(model,x,y,args,g,scale,bg,ev,trace=None):
    started=time.perf_counter();plan=compact_schedule(model,x,args,g,bg);schedule_seconds=time.perf_counter()-started
    losses,timing=ev.evaluate(model,plan);assert torch.isfinite(losses).all()
    n=len(plan.indices);local=torch.arange(n)
    evidence=torch.zeros((n,2),dtype=torch.int8);counts=torch.zeros((n,2),dtype=torch.int32)
    proxy=UniformTorch(plan.uniforms[0]);accumulate=private_function(train.accumulate,{'torch':proxy})
    clipped=saturated=nonzero=peak=0;differences=[]
    if trace is not None:trace.update(losses=list(losses.unbind()),votes=[],counters=[],schedule_indices=plan.indices.clone())
    for k in range(args.measurements):
        difference=losses[2*k]-losses[2*k+1];signal=-difference*plan.phi[k]/scale
        proxy.uniform=plan.uniforms[k];before=evidence.clone() if trace is not None else None
        c,s,z=accumulate(evidence,local,plan.edges[k],signal,g,args.leak,127)
        clipped+=c;saturated+=s;nonzero+=z
        peak=max(peak,int(evidence[local,plan.edges[k]].to(torch.int32).abs().max()))
        counts[local,plan.edges[k]]+=1;differences.append(float(difference.abs()))
        if trace is not None:trace['votes'].append(evidence-before);trace['counters'].append(evidence.clone())
    selected,fires=train.select_actions(model.weights[plan.indices],evidence,counts,local,args.threshold,args.max_fires,scale)
    proposal=model.weights.clone();proposal[plan.indices]=selected
    # Original full-state flatten order is global coordinate order, not randperm order.
    order=torch.argsort(plan.indices);statistics=train.counter_statistics(evidence[order],counts[order],127)
    # Integer sums <=1024 are exact FP32; include unvisited zeros in denominator.
    statistics['counter_all_mean']=float(evidence.float().sum()/(model.num_params*2))
    statistics['counter_all_abs_mean']=float(evidence.float().abs().sum()/(model.num_params*2))
    votes=args.measurements*n
    stats=dict(fires=fires,clip_rate=clipped/votes,zero_difference_count=sum(d==0 for d in differences),zero_difference_fraction=sum(d==0 for d in differences)/args.measurements,saturation_rate=saturated/votes,nonzero_vote_rate=nonzero/votes,scale=scale,counter_peak_abs=peak,**statistics)
    if getattr(args,'loss_diagnostics',False):stats['abs_y_values']=differences
    median=sorted(differences)[len(differences)//2];new_scale=max(args.min_scale,(1-args.scale_ema)*scale+args.scale_ema*median)
    return (proposal,plan.indices,stats,new_scale),{**timing,'schedule_seconds':schedule_seconds}

@torch.no_grad()
def epoch(model,x,y,args,g,scale,bg,ev,mode,trace=None):
    if mode=='cpu_compact':return compact_epoch(model,x,y,args,g,scale,bg,ev,trace)
    return original_epoch(model,x,y,args,g,scale,bg,ev,trace)
