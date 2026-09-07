import unittest
import torch
from train import TernaryModel, epoch, parser

class LossDiagnosticsTests(unittest.TestCase):
    def test_observation_preserves_trajectory_and_rng(self):
        torch.set_num_threads(1)
        x=torch.randn(20,4,generator=torch.Generator().manual_seed(4));y=torch.arange(20)%10
        for activation,precision in [('relu','a3'),('identity','a32'),('identity','a3')]:
            models=[TernaryModel(pool_shape=(2,2),hidden_sizes=[5,5,5],hidden_activation=activation,
                activation_precision=precision,a3_method='mean_threshold') for _ in range(2)]
            generators=[torch.Generator().manual_seed(17) for _ in range(2)]
            scales=[.02,.02]
            for step in range(4):
                results=[]
                for i in range(2):
                    args=parser().parse_args(['--measurements','8','--block-size','4','--batch-size','8','--threshold','1'])
                    args.loss_diagnostics=bool(i)
                    result=epoch(models[i],x,y,args,generators[i],scales[i])
                    results.append(result);models[i].weights.copy_(result[0]);scales[i]=result[3]
                values=results[1][2].pop('abs_y_values')
                self.assertEqual(len(values),8)
                self.assertTrue(all(v>=0 for v in values))
                self.assertEqual(sum(v==0 for v in values),results[1][2]['zero_difference_count'])
                self.assertEqual(results[0][2],results[1][2])
                self.assertEqual(scales[0],scales[1])
                self.assertTrue(torch.equal(models[0].weights,models[1].weights))
                self.assertTrue(torch.equal(generators[0].get_state(),generators[1].get_state()))

if __name__=='__main__':unittest.main()
