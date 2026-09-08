"""Independent, exact-operation ablations: restored weights OR candidate buffers.

No changes to legacy files, no low-rank output correction, no batched matmul.
The original epoch/candidate code is reused with private function globals.
"""
import ast
import inspect
import math
import types
import torch
import train


def private_function(function,bindings):
    raw=getattr(function,'__wrapped__',function)
    return types.FunctionType(raw.__code__,{**raw.__globals__,**bindings},raw.__name__,raw.__defaults__,raw.__closure__)


_BUFFER_CODE=None

def buffer_candidate_function(buffers):
    global _BUFFER_CODE
    if _BUFFER_CODE is not None:
        return types.FunctionType(_BUFFER_CODE,{**train.candidate_pair.__globals__,'_candidate_buffers':buffers},'candidate_pair')
    tree=ast.parse(inspect.getsource(train.candidate_pair))
    class ReplaceClones(ast.NodeTransformer):
        def __init__(self):self.count=0
        def visit_Call(self,node):
            if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='weights' and node.func.attr=='clone':
                result=ast.Subscript(value=ast.Name(id='_candidate_buffers',ctx=ast.Load()),slice=ast.Constant(value=self.count),ctx=ast.Load());self.count+=1
                return ast.copy_location(result,node)
            return self.generic_visit(node)
    transform=ReplaceClones();tree=transform.visit(tree);assert transform.count==2
    ast.fix_missing_locations(tree);namespace={**train.candidate_pair.__globals__,'_candidate_buffers':buffers}
    exec(compile(tree,'<candidate_pair: two clones replaced by private buffers>','exec'),namespace)
    _BUFFER_CODE=namespace['candidate_pair'].__code__
    return namespace['candidate_pair']


class CachedMatrices:
    """Private per-epoch proxy; full FP32 matrices are transient, never learned."""
    def __init__(self,model):
        object.__setattr__(self,'model',model)
        matrices=model.matrices(model.weights)
        flat=torch.cat([w.flatten() for w in matrices])
        sizes=[math.prod(s) for s in model.shapes]
        views=[];offset=0
        for shape,size in zip(model.shapes,sizes):
            views.append(flat[offset:offset+size].view(shape));offset+=size
        object.__setattr__(self,'flat',flat);object.__setattr__(self,'views',views)
        object.__setattr__(self,'ends',torch.tensor(sizes,device=model.device).cumsum(0))
        object.__setattr__(self,'indices',None);object.__setattr__(self,'coordinate_scales',None)
    def __getattr__(self,name):return getattr(self.model,name)
    def __setattr__(self,name,value):setattr(self.model,name,value)
    def set_indices(self,indices):
        if self.indices is None:
            object.__setattr__(self,'indices',indices)
            layers=torch.bucketize(indices,self.ends,right=True)
            object.__setattr__(self,'coordinate_scales',torch.tensor(self.model.scales,dtype=torch.float32,device=self.model.device)[layers])
        else:assert torch.equal(indices,self.indices)
    def matrices(self,weights):
        # Assign original code*scale, never add a floating-point delta.
        self.flat[self.indices]=weights[self.indices].float()*self.coordinate_scales
        return self.views
    def __call__(self,x,weights=None):
        return type(self.model).__call__.__wrapped__(self,x,weights)


@torch.no_grad()
def epoch(model,x,y,args,generator,scale,batch_generator=None,*,engine='naive',trace=None):
    if engine not in ['naive','restore_cache','candidate_buffers']:raise ValueError(engine)
    if engine=='naive' and trace is None:return train.epoch(model,x,y,args,generator,scale,batch_generator)
    active=model;bindings={}
    if engine=='restore_cache':
        active=CachedMatrices(model)
        def candidate(weights,indices,g):
            active.set_indices(indices)
            return train.candidate_pair(weights,indices,g)
        bindings['candidate_pair']=candidate
    elif engine=='candidate_buffers':
        # Two initial full copies; all128 later full copies are eliminated.
        bindings['candidate_pair']=buffer_candidate_function([model.weights.clone(),model.weights.clone()])
    if trace is not None:
        trace.update(losses=[],votes=[],counters=[])
        def loss(m,bx,by,weights=None):
            result=train.loss(m,bx,by,weights)
            trace['losses'].append(result.clone())
            return result
        def accumulate(evidence,indices,edges,signal,g,leak,capacity):
            before=evidence[indices].clone()
            result=train.accumulate(evidence,indices,edges,signal,g,leak,capacity)
            trace['votes'].append((evidence[indices]-before).clone())
            trace['counters'].append(evidence[indices].clone())
            return result
        bindings.update(loss=loss,accumulate=accumulate)
    return private_function(train.epoch,bindings)(active,x,y,args,generator,scale,batch_generator)
