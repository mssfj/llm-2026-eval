import unittest
import torch
import torch.nn.functional as F
from train import TernaryModel, model_parameter_count, model_shapes, parser, validate
from depth_diagnostics import DEPTH_WIDTHS, SignalObserver, layer_events


class DepthTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_exact_active_parameter_counts_and_forward(self):
        x=torch.randn(4,90,generator=torch.Generator().manual_seed(4))
        initial=None
        for depth,widths in DEPTH_WIDTHS.items():
            model=TernaryModel(pool_shape=(9,10),hidden_sizes=widths)
            self.assertEqual(len(model.shapes),depth)
            self.assertEqual(model.num_params,100000)
            self.assertEqual(model_parameter_count(pool_shape=(9,10),hidden_sizes=widths),100000)
            if initial is None: initial=model.weights.clone()
            self.assertTrue(torch.equal(initial,model.weights))
            expected=x;offset=0
            for layer,((out_width,in_width),scale) in enumerate(zip(model.shapes,model.scales)):
                n=out_width*in_width
                matrix=model.weights[offset:offset+n].view(out_width,in_width).float()*scale
                expected=F.linear(expected,matrix)
                if layer<depth-1: expected=F.relu(expected)
                offset+=n
            self.assertEqual(offset,100000)
            self.assertTrue(torch.equal(expected,model(x)))
            model.signal_observer=SignalObserver()
            self.assertTrue(torch.equal(expected,model(x)))
            self.assertEqual(len(model.signal_observer.summary()),3*depth)

    def test_layer_boundaries_and_selection_opportunities(self):
        model=TernaryModel(pool_shape=(1,1),hidden_sizes=[2,3])
        # Layer sizes: 2, 6, 30. Select coordinates at each boundary.
        indices=torch.tensor([0,1,2,7,8,37])
        proposal=model.weights.clone()
        proposal[2]=0 if int(proposal[2]) else 1
        events=layer_events(model,proposal,indices)
        self.assertEqual([e['selected_coordinates'] for e in events],[2,2,2])
        self.assertEqual([e['fires'] for e in events],[0,1,0])
        self.assertEqual([e['selected_interval'] for e in events],[1,1,1])
        self.assertEqual([e['fire_interval'] for e in events],[0,1,0])
        events=layer_events(model,model.weights,torch.tensor([0,1]))
        self.assertEqual([e['selected_interval'] for e in events],[1,0,0])
        self.assertEqual(sum(e['fires'] for e in events),0)

    def test_signal_moments_and_dead_features_across_batches(self):
        obs=SignalObserver()
        x=torch.tensor([[-2.,0.,1.],[2.,0.,3.]])
        obs.record(0,'output',x[:1]);obs.record(0,'output',x[1:])
        r=obs.summary()[0]
        self.assertAlmostEqual(r['mean'],float(x.double().mean()))
        self.assertAlmostEqual(r['rms'],float(x.double().square().mean().sqrt()))
        self.assertAlmostEqual(r['std'],float(x.double().std(unbiased=False)))
        self.assertEqual(r['dead_features'],1)
        self.assertEqual(r['zero_fraction'],1/3)
        self.assertEqual(r['negative_fraction'],1/6)

    def test_legacy_shape_and_invalid_widths(self):
        self.assertEqual(model_shapes(pool_shape=(9,10),hidden_size=1000),[(1000,90),(10,1000)])
        with self.assertRaises(ValueError): model_shapes(hidden_size=1,hidden_sizes=[2])
        with self.assertRaises(ValueError): model_shapes(hidden_sizes=[0])
        p=parser();args=p.parse_args(['--pool-shape','9','10','--hidden-sizes','200','200','200','--expected-params','100000'])
        validate(args,p)


if __name__=='__main__': unittest.main()
