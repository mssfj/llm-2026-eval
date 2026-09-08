import unittest
import torch
from residual_stream import ResidualStreamModel
from residual_followup_models import (ResidualTDT,BPResidual,ExactIdentitySTE,ternary_weight,
                                      TDT_CONDITIONS,divergence_reason)
from activation_quantization import encode_activation,decode_activation
from train import epoch
from run_residual_e17 import config


class FollowupTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.set_grad_enabled(True)

    def test_counts_and_e17_exact_equivalence(self):
        for blocks,width,precision,count in TDT_CONDITIONS.values():
            m=ResidualTDT(blocks=blocks,width=width,precision=precision)
            self.assertEqual(m.num_params,count)
            self.assertEqual(len(m.shapes),2*blocks+2)
            self.assertEqual(m.weights.dtype,torch.int8)
        old,new=ResidualStreamModel(),ResidualTDT()
        self.assertTrue(torch.equal(old.weights,new.weights))
        x=torch.randn(128,90); y=torch.arange(128)%10
        self.assertTrue(torch.equal(old(x),new(x)))
        a=config(0,'tdt_mnist/data')
        results=[]
        for m in (old,new):
            before=m.forward_calls
            results.append(epoch(m,x,y,a,torch.Generator().manual_seed(1),.02,
                                 torch.Generator().manual_seed(100000)))
            self.assertEqual(m.forward_calls-before,128)
        self.assertTrue(torch.equal(results[0][0],results[1][0]))
        self.assertEqual(results[0][2:],results[1][2:])

    def test_zero_branches_all_depths(self):
        for blocks,width,precision,_ in TDT_CONDITIONS.values():
            m=ResidualTDT(blocks=blocks,width=width,precision=precision)
            matrices=m.matrices(m.weights)
            h=torch.randn(7,width)
            for b in range(blocks):
                matrices[1+2*b].zero_();matrices[2+2*b].zero_()
                self.assertTrue(torch.equal(m.block(h,b,matrices),h))

    def test_identity_ste_arbitrary_upstream_and_exact_forward(self):
        for kind in ('activation','weight'):
            x=torch.randn(7,11,requires_grad=True)
            if kind=='weight':q=ternary_weight(x)[0]
            else:
                codes,scale=encode_activation(x.detach(),'a8');q=decode_activation(codes,scale)
            out=ExactIdentitySTE.apply(x,q)
            upstream=torch.randn_like(x)
            out.backward(upstream)
            self.assertTrue(torch.equal(out,q))
            self.assertTrue(torch.equal(x.grad,upstream))

    def test_latent_update_all_18_gradients_and_paired_initialization(self):
        m=BPResidual('E20c',0)
        for condition in ('E20a','E20b'):
            other=BPResidual(condition,0)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(m.latent,other.latent)))
        before=[w.detach().clone() for w in m.latent]
        quantized=[ternary_weight(w)[0] for w in m.latent]
        opt=torch.optim.Adam(m.parameters(),lr=.001)
        loss=torch.nn.functional.cross_entropy(m(torch.randn(128,90)),torch.arange(128)%10)
        loss.backward()
        self.assertTrue(all(w.grad is not None and torch.isfinite(w.grad).all() and float(w.grad.norm())>0 for w in m.latent))
        opt.step()
        self.assertTrue(all(not torch.equal(a,b) for a,b in zip(before,m.latent)))
        for saved,old,w in zip(quantized,before,m.latent):
            self.assertTrue(torch.equal(saved,ternary_weight(old)[0]))
            self.assertNotEqual(saved.data_ptr(),w.data_ptr())
            self.assertTrue(w.is_leaf)
            self.assertEqual(w.dtype,torch.float32)
        zero=torch.zeros(4,4)
        self.assertTrue(torch.equal(ternary_weight(zero)[0],zero))

    def test_failure_gate(self):
        self.assertIsNone(divergence_reason(False,True,True,[1.]*18))
        self.assertIsNotNone(divergence_reason(False,True,True,[0.]*18))
        self.assertIsNotNone(divergence_reason(True,False,False,[1.]*18))


if __name__=='__main__':unittest.main()
