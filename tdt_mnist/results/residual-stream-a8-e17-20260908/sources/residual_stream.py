"""E17: FP32 residual stream with ternary branch matrices; legacy TDT API."""
import math
import torch
import torch.nn.functional as F
from train import TernaryModel
from activation_quantization import encode_activation, decode_activation


class ResidualStreamModel(TernaryModel):
    def __init__(self, seed=0, precision='a8', activation='relu'):
        # 17 hidden widths give exactly the 18 required matrix shapes. Reuse
        # the legacy INT8 initialization, fixed scales, and flat coordinate map.
        super().__init__(pool_shape=(9, 10), hidden_sizes=[76]*17,
                         seed=seed, activation_precision=precision,
                         hidden_activation=activation)
        self.matrix_names = ['W_in'] + [f'block{b}.{w}' for b in range(8)
                                                    for w in ('W1', 'W2')] + ['W_out']
        assert self.num_params == 100016

    @staticmethod
    def rmsnorm(h):
        return h / torch.sqrt(h.square().mean(-1, keepdim=True) + 1e-8)

    def record(self, layer, stage, x):
        if self.signal_observer is not None:
            self.signal_observer.record(layer, stage, x)

    def linear(self, x, index, matrices):
        codes, scale = encode_activation(x, self.activation_precision)
        decoded = decode_activation(codes, scale)
        if self.activation_observer is not None:
            self.activation_observer.record(index, x, codes, decoded)
        self.record(index, 'pre_quantization', x)
        self.record(index, 'input', decoded)
        out = F.linear(decoded, matrices[index])
        self.record(index, 'output', out)
        return out

    def block(self, h, block, matrices):
        index = 1 + 2*block
        self.record(index, 'stream_before', h)
        z = self.linear(self.rmsnorm(h), index, matrices)
        if self.hidden_activation == 'relu':
            z = F.relu(z)
        self.record(index, 'branch_activation', z)
        branch = self.linear(z, index+1, matrices)
        self.record(index, 'branch_output', branch)
        out = h + branch
        self.record(index, 'stream_after', out)
        return out

    def matrices(self, weights):
        result = []
        offset = 0
        for shape, scale in zip(self.shapes, self.scales):
            size = math.prod(shape)
            result.append(weights[offset:offset+size].view(shape).float()*scale)
            offset += size
        return result

    @torch.no_grad()
    def __call__(self, x, weights=None):
        self.forward_calls += 1
        self.forward_examples += len(x)
        matrices = self.matrices(self.weights if weights is None else weights)
        h = self.linear(x, 0, matrices)
        for b in range(8):
            h = self.block(h, b, matrices)
        return self.linear(self.rmsnorm(h), 17, matrices)
