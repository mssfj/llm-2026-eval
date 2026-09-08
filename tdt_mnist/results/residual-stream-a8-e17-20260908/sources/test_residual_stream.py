import unittest
import torch
from residual_stream import ResidualStreamModel
from train import epoch, parser


class ResidualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_zero_branches_are_exact_identity(self):
        for precision in ('a8', 'a32'):
            for activation in ('relu', 'identity'):
                m = ResidualStreamModel(precision=precision, activation=activation)
                matrices = m.matrices(m.weights)
                h = torch.randn(13, 76)
                for b in range(8):
                    matrices[1+2*b].zero_()
                    matrices[2+2*b].zero_()
                    self.assertTrue(torch.equal(m.block(h, b, matrices), h))

    def test_budget_and_legacy_epoch_calls(self):
        m = ResidualStreamModel()
        self.assertEqual(m.num_params, 100016)
        self.assertEqual(len(m.shapes), 18)
        self.assertEqual(m.weights.dtype, torch.int8)
        args = parser().parse_args(['--measurements', '64', '--threshold', '8', '--block-size', '16'])
        before = m.weights.clone()
        proposal, _, stats, _ = epoch(m, torch.randn(128, 90), torch.arange(128)%10,
            args, torch.Generator().manual_seed(1), .02, torch.Generator().manual_seed(100000))
        self.assertEqual(m.forward_calls, 128)
        self.assertEqual(m.forward_examples, 128*128)
        self.assertTrue(torch.equal(m.weights, before))
        self.assertLessEqual(int((proposal != before).sum()), 1)
        self.assertTrue(set(proposal.unique().tolist()) <= {-1, 0, 1})
        self.assertLessEqual(stats['counter_peak_abs'], 64)

    def test_formula_and_fp32(self):
        m = ResidualStreamModel(precision='a32')
        x = torch.randn(7, 90)
        w = m.matrices(m.weights)
        h = x @ w[0].T
        for b in range(8):
            z = h / (h.square().mean(-1, keepdim=True)+1e-8).sqrt()
            h = h + torch.relu(z @ w[1+2*b].T) @ w[2+2*b].T
        expected = m.rmsnorm(h) @ w[-1].T
        actual = m(x)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertFalse(actual.requires_grad)


if __name__ == '__main__':
    unittest.main()
