"""E18/E19 variable-depth TDT and E20 differentiable versions of E17."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from train import TernaryModel
from residual_stream import ResidualStreamModel
from activation_quantization import encode_activation, decode_activation


TDT_CONDITIONS = {
    'E18a': (16, 76, 'a8', 192432),
    'E18b': (24, 76, 'a8', 284848),
    'E18c': (32, 76, 'a8', 377264),
    'E18d': (16, 54, 'a8', 98712),
    'E19a': (8, 76, 'a4', 100016),
}


class ResidualTDT(ResidualStreamModel):
    def __init__(self, seed=0, blocks=8, width=76, precision='a8'):
        TernaryModel.__init__(self, pool_shape=(9,10), hidden_sizes=[width]*(2*blocks+1),
            seed=seed, activation_precision=precision, hidden_activation='relu')
        self.blocks, self.width = blocks, width
        self.matrix_names = ['W_in'] + [f'block{b}.{w}' for b in range(blocks)
                                       for w in ('W1','W2')] + ['W_out']
        assert self.num_params == 100*width + 2*blocks*width*width

    @torch.no_grad()
    def __call__(self, x, weights=None):
        self.forward_calls += 1
        self.forward_examples += len(x)
        matrices = self.matrices(self.weights if weights is None else weights)
        h = self.linear(x, 0, matrices)
        for b in range(self.blocks):
            h = self.block(h,b,matrices)
        return self.linear(self.rmsnorm(h),len(matrices)-1,matrices)


class ExactIdentitySTE(torch.autograd.Function):
    """Exactly return the quantized forward value, exactly pass arbitrary dL/dx."""
    @staticmethod
    def forward(ctx, original, quantized):
        return quantized.clone()

    @staticmethod
    def backward(ctx, upstream):
        return upstream, None


def ternary_weight(weight):
    # The effective W3 tensor is a transient, detached value. Adam owns only
    # the independent FP32 latent Parameter; alpha has no separate optimizer.
    with torch.no_grad():
        alpha = weight.abs().mean()
        if float(alpha) == 0:
            codes = torch.zeros_like(weight, dtype=torch.int8)
        else:
            codes = torch.round(weight/alpha).clamp(-1,1).to(torch.int8)
        effective = codes.float()*alpha
    return effective, codes, alpha


class BPResidual(nn.Module):
    rmsnorm = staticmethod(ResidualStreamModel.rmsnorm)
    record = ResidualStreamModel.record
    block = ResidualStreamModel.block

    def __init__(self, condition='E20a', seed=0):
        super().__init__()
        assert condition in ('E20a','E20b','E20c')
        self.condition = condition
        self.width, self.blocks = 76, 8
        self.shapes = [(76,90)] + [(76,76)]*16 + [(10,76)]
        self.matrix_names = ['W_in'] + [f'block{b}.{w}' for b in range(8)
                                       for w in ('W1','W2')] + ['W_out']
        self.num_params = sum(math.prod(s) for s in self.shapes)
        self.activation_precision = 'a32' if condition=='E20a' else 'a8'
        self.hidden_activation = 'relu'
        self.signal_observer = self.activation_observer = None
        g = torch.Generator().manual_seed(seed)
        self.latent = nn.ParameterList([nn.Parameter(torch.randn(shape,generator=g)*
            math.sqrt((1. if i==17 else 2.)/shape[1])) for i,shape in enumerate(self.shapes)])
        self.forward_calls = self.forward_examples = 0

    def effective_matrices(self):
        if self.condition != 'E20c':
            return list(self.latent)
        return [ExactIdentitySTE.apply(w,ternary_weight(w)[0]) for w in self.latent]

    def linear(self,x,index,matrices):
        with torch.no_grad():
            codes,scale = encode_activation(x,self.activation_precision)
            decoded = decode_activation(codes,scale)
        if self.activation_observer is not None:
            self.activation_observer.record(index,x.detach(),codes,decoded)
        self.record(index,'pre_quantization',x)
        q = x if self.activation_precision=='a32' else ExactIdentitySTE.apply(x,decoded)
        self.record(index,'input',q)
        out = F.linear(q,matrices[index])
        self.record(index,'output',out)
        return out

    def forward(self,x):
        self.forward_calls += 1
        self.forward_examples += len(x)
        matrices = self.effective_matrices()
        h = self.linear(x,0,matrices)
        for b in range(8):
            h = self.block(h,b,matrices)
        return self.linear(self.rmsnorm(h),17,matrices)


def divergence_reason(nonfinite, exploded, relu_zero, gradient_norms):
    if nonfinite:
        return 'nonfinite_loss_logits_weights_or_gradients'
    if exploded and relu_zero and all(v==0 for v in gradient_norms):
        return 'E15_amplitude_explosion_relu_zero_all_gradients_zero'
    return None
