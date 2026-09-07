import unittest,importlib.util
from pathlib import Path
import torch
import torch.nn.functional as F
from train import TernaryModel,epoch,parser
from a3_improvements import renormalize,shortcut,lloyd_encode
from activation_quantization import encode_activation,decode_activation

class ImprovementTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_norm_zero_and_unit_rms(self):
        x=torch.tensor([[0.,0.,0.],[-2.,0.,1.],[1e-12,0.,0.]])
        y,g=renormalize(x)
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.equal(y[0],x[0]))
        self.assertAlmostEqual(float(y[1].square().mean().sqrt()),1.,places=6)
        self.assertLess(float(y[2].square().mean().sqrt()),1.)
        self.assertTrue(torch.equal((x==0),(y==0)))
    def test_residual_uses_prequantization_stream_and_preserves_budget(self):
        m=TernaryModel(pool_shape=(2,2),hidden_sizes=[5,3,6],hidden_activation='identity',activation_precision='a3',a3_method='mean_threshold',a3_improvement='residual')
        x=torch.randn(7,4,generator=torch.Generator().manual_seed(7));expected=x;offset=0
        for i,(shape,scale) in enumerate(zip(m.shapes,m.scales)):
            prior=expected;n=shape[0]*shape[1]
            q,b=encode_activation(prior,'a3','mean_threshold')
            expected=F.linear(decode_activation(q,b),m.weights[offset:offset+n].view(shape).float()*scale)
            if 0<i<len(m.shapes)-1:
                p=prior[:,:shape[0]]
                if p.shape[-1]<shape[0]:p=F.pad(p,(0,shape[0]-p.shape[-1]))
                expected=p+expected
            offset+=n
        self.assertTrue(torch.equal(expected,m(x)))
        self.assertEqual(offset,m.num_params)
    def test_lloyd_degeneracy_and_descent(self):
        x=torch.cat([torch.zeros(1,80),torch.ones(1,80),torch.randn(16,80,generator=torch.Generator().manual_seed(2))])
        q,b,info=lloyd_encode(x,diagnostics=True)
        self.assertTrue(torch.isfinite(b).all())
        self.assertTrue(set(q.unique().tolist()).issubset({-1,0,1}))
        sel=x.abs()>.6*x.std(-1,keepdim=True,unbiased=False)
        initial_b=(x.abs()*sel).sum(-1,keepdim=True)/sel.sum(-1,keepdim=True).clamp_min(1)
        initial=x.sign()*sel*initial_b
        self.assertTrue(torch.all((x-q*b).square().mean(-1)<=(x-initial).square().mean(-1)+1e-6))
        self.assertTrue(torch.equal(q[0]*b[0],x[0]))
        self.assertTrue(torch.equal(q[1]*b[1],x[1]))
        stable=info['lloyd_unconverged'].flatten()==0
        self.assertTrue(torch.equal((q!=0)[stable],(x.abs()>b/2)[stable]))
    def test_legacy_trajectory_identical(self):
        oldpath=Path(__file__).parent/'results/depth-activation-100k-20260907/identity-a3-threshold/sources/train.py'
        spec=importlib.util.spec_from_file_location('legacy_train_improvement_test',oldpath);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        kwargs=dict(pool_shape=(2,2),hidden_sizes=[5,3,6],hidden_activation='identity',activation_precision='a3',a3_method='mean_threshold')
        models=[old.TernaryModel(**kwargs),TernaryModel(**kwargs)]
        gs=[torch.Generator().manual_seed(10) for _ in range(2)];scales=[.02,.02]
        x=torch.randn(20,4,generator=torch.Generator().manual_seed(8));y=torch.arange(20)%10
        args=parser().parse_args(['--measurements','8','--block-size','4','--batch-size','8','--threshold','8'])
        for _ in range(8):
            rs=[]
            for i,fn in enumerate([old.epoch,epoch]):
                r=fn(models[i],x,y,args,gs[i],scales[i]);models[i].weights.copy_(r[0]);scales[i]=r[3];rs.append(r)
            self.assertEqual(rs[0][2],rs[1][2]);self.assertEqual(scales[0],scales[1])
            self.assertTrue(torch.equal(models[0].weights,models[1].weights))
            self.assertTrue(torch.equal(gs[0].get_state(),gs[1].get_state()))

if __name__=='__main__':unittest.main()
