"""Algorithm invariants and actual no-backprop learning; no downloads."""
import unittest
import torch
from train import (TernaryModel, accumulate, candidate_pair, epoch, loss,
                   oracle_metrics, parser, select_actions)


class TDTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.generator = torch.Generator().manual_seed(42)

    def test_parameter_count_and_legal_candidates(self):
        model = TernaryModel()
        self.assertEqual(model.num_params, 1000)
        self.assertEqual(model.weights.dtype, torch.int8)
        self.assertFalse(model.weights.requires_grad)
        weights = torch.tensor([-1, 0, 1], dtype=torch.int8)
        for _ in range(100):
            plus, minus, edge, phi = candidate_pair(weights, torch.arange(3), self.generator)
            self.assertTrue(torch.all((plus - minus).long() == phi))
            self.assertTrue(torch.all((plus - weights).abs() <= 1))
            self.assertTrue(torch.all((minus - weights).abs() <= 1))
            self.assertTrue(torch.all(torch.minimum(plus, minus) == edge - 1))
            self.assertTrue(torch.all(torch.maximum(plus, minus) == edge))
        self.assertEqual(weights.tolist(), [-1, 0, 1])

    def test_rounding_mean_and_counter_overflow(self):
        size = 40000
        evidence = torch.zeros((size, 2), dtype=torch.int8)
        indices = torch.arange(size)
        edges = torch.zeros(size, dtype=torch.long)
        signal = torch.full((size,), 0.25)
        accumulate(evidence, indices, edges, signal, self.generator, 1, 127)
        self.assertAlmostEqual(float(evidence[:, 0].float().mean()), 0.25, delta=0.01)
        evidence[:] = 127
        clipped, saturated, _ = accumulate(evidence, indices, edges, signal * 8,
                                           self.generator, 1, 127)
        self.assertEqual(clipped, size)
        self.assertEqual(saturated, size)
        self.assertTrue(torch.all(evidence == 127))

    def test_outward_actions_and_competing_zero_edges(self):
        weights = torch.tensor([-1, 0, 1], dtype=torch.int8)
        counts = torch.ones((3, 2), dtype=torch.int32)
        evidence = torch.tensor([[-8, 127], [-5, 9], [127, 8]], dtype=torch.int8)
        result, fires = select_actions(weights, evidence, counts, torch.arange(3), 4, 3, 1)
        self.assertEqual(result.tolist(), [-1, 1, 1])
        self.assertEqual(fires, 1)
        evidence = torch.tensor([[8, 127], [-9, 5], [127, -8]], dtype=torch.int8)
        result, fires = select_actions(weights, evidence, counts, torch.arange(3), 4, 3, 1)
        self.assertEqual(result.tolist(), [0, -1, 0])
        self.assertEqual(fires, 3)
        _, fires = select_actions(weights, evidence, counts, torch.arange(3), 4, 1, 1)
        self.assertEqual(fires, 1)

    def test_forward_only_learning_and_heldout_oracle(self):
        args = parser().parse_args(["--pool-size", "1", "--threshold", "2",
                                    "--measurements", "8", "--scale", "0.01"])
        model = TernaryModel(pool_size=1)
        x = torch.ones((32, 1))
        y = torch.zeros(32, dtype=torch.long)
        initial = float(loss(model, x, y))
        scale = args.scale
        fires = 0
        for _ in range(100):
            calls = model.forward_calls
            before = model.weights.clone()
            proposal, indices, stats, scale = epoch(model, x, y, args, self.generator, scale)
            self.assertEqual(model.forward_calls - calls, 2 * args.measurements)
            self.assertTrue(torch.equal(model.weights, before))
            self.assertTrue(torch.all((proposal - before).abs() <= 1))
            self.assertLessEqual(stats["fires"], args.max_fires)
            oracle_metrics(model, proposal, indices, x, y, 1e-5)
            self.assertTrue(torch.equal(model.weights, before))
            fires += stats["fires"]
            model.weights.copy_(proposal)
        self.assertGreater(fires, 0)
        self.assertLess(float(loss(model, x, y)), initial - 0.1)
        self.assertTrue(torch.all(model.weights.abs() <= 1))


if __name__ == "__main__":
    unittest.main()
