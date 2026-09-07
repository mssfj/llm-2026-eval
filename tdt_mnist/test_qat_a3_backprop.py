import unittest
import torch
import torch.nn.functional as F
from qat_a3_backprop import A3STE,QATMLP,ALL
from backprop_a3_inference import FP32MLP
from activation_quantization import encode_activation,decode_activation

class QATTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(9)
    def test_forward_is_exact_and_backward_is_identity(self):
        x=torch.tensor([[-100.,-1.,0.,.2,1e-20,4.],[0.,0.,0.,0.,0.,0.]],requires_grad=True)
        q,b=encode_activation(x.detach(),'a3','mean_threshold',.5);actual=A3STE.apply(x)
        self.assertTrue(torch.equal(actual,decode_activation(q,b)))
        incoming=torch.randn_like(x);(actual*incoming).sum().backward()
        self.assertTrue(torch.equal(x.grad,incoming))
    def test_training_forward_matches_ptq_and_reaches_all_weights(self):
        model=QATMLP();ref=FP32MLP().eval();ref.load_state_dict(model.state_dict())
        x=torch.randn(32,90);y=torch.arange(32)%10;z=model(x)
        with torch.no_grad():expected=ref(x,ALL)
        self.assertTrue(torch.equal(z,expected))
        F.cross_entropy(z,y).backward()
        for layer in model.layers:
            self.assertTrue(torch.isfinite(layer.weight.grad).all());self.assertGreater(float(layer.weight.grad.norm()),0.)

if __name__=='__main__':unittest.main()
