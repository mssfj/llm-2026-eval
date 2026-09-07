import unittest
import torch
import torch.nn.functional as F
from backprop_a3_inference import FP32MLP,digest,modes
from activation_quantization import encode_activation,decode_activation,ActivationObserver
from depth_diagnostics import SignalObserver

class BackpropInferenceTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(4)
    def test_fp32_backprop_reaches_every_layer(self):
        model=FP32MLP();self.assertEqual(sum(p.numel() for p in model.parameters()),95274)
        before=digest(model);x=torch.randn(32,90);y=torch.arange(32)%10
        loss=F.cross_entropy(model(x),y);loss.backward()
        for layer in model.layers:
            self.assertIsNotNone(layer.weight.grad);self.assertTrue(torch.isfinite(layer.weight.grad).all());self.assertGreater(float(layer.weight.grad.norm()),0.)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001);optimizer.step();self.assertNotEqual(before,digest(model))
        with self.assertRaises(ValueError):model(x,(1,))
    def test_selective_quantization_matches_forward_hook(self):
        model=FP32MLP().eval();x=torch.randn(7,90);qindices=(1,5,15);before=digest(model)
        sig=SignalObserver();act=ActivationObserver(16,'a3')
        with torch.no_grad():
            actual=model(x,qindices,sig,act)
            def hook(module,args):
                q,b=encode_activation(args[0],'a3','mean_threshold',.5)
                return (decode_activation(q,b),)
            handles=[model.layers[i].register_forward_pre_hook(hook) for i in qindices]
            expected=model(x)
            for handle in handles:handle.remove()
        self.assertTrue(torch.equal(actual,expected));self.assertEqual(digest(model),before)
        self.assertEqual([r['layer'] for r in act.summary()],list(qindices));self.assertEqual(len(sig.summary()),48)
        self.assertEqual(len(modes()),33)
        self.assertTrue(all(0<=i<16 for _,indices in modes() for i in indices))

if __name__=='__main__':unittest.main()
