import unittest
from unittest.mock import patch

import torch

from compare_counters import immediate_action, no_counter_epoch
from train import TernaryModel, epoch, parser


class CounterAblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_immediate_legal_moves_and_no_history(self):
        weights = torch.tensor([-1, 0, 1], dtype=torch.int8)
        indices = torch.arange(3)
        edges = torch.tensor([0, 0, 1])
        proposal, count = immediate_action(weights, indices, edges, torch.tensor([1, -1, -1]), 3)
        self.assertEqual(proposal.tolist(), [0, -1, 0])
        self.assertEqual(count, 3)
        proposal, count = immediate_action(weights, indices, edges, torch.zeros(3), 3)
        self.assertEqual(proposal.tolist(), weights.tolist())
        self.assertEqual(count, 0)

    def test_one_pair_matches_counter_at_threshold_one(self):
        args = parser().parse_args(["--block-size", "8", "--measurements", "1", "--threshold", "1"])
        model = TernaryModel()
        x = torch.randn((32, 100), generator=torch.Generator().manual_seed(99))
        y = torch.arange(32) % 10
        gen1, gen2 = [torch.Generator().manual_seed(1) for _ in range(2)]
        batch1, batch2 = [torch.Generator().manual_seed(2) for _ in range(2)]
        a, _, stats_a, scale_a = epoch(model, x, y, args, gen1, 0.02, batch1)
        b, stats_b, scale_b = no_counter_epoch(model, x, y, args, gen2, 0.02, batch2)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(stats_a["fires"], stats_b["fires"])
        self.assertEqual(scale_a, scale_b)
        self.assertTrue(torch.equal(gen1.get_state(), gen2.get_state()))
        self.assertTrue(torch.equal(batch1.get_state(), batch2.get_state()))

    def test_next_pair_uses_updated_weights_without_modifying_model(self):
        model = TernaryModel()
        model.weights.zero_()
        args = parser().parse_args(["--measurements", "4", "--threshold", "8"])
        seen = []

        def pair(weights, indices, generator):
            seen.append(int(weights[indices[0]]))
            edges = torch.ones(len(indices), dtype=torch.long)
            plus, minus = weights.clone(), weights.clone()
            plus[indices] = 1
            minus[indices] = 0
            return plus, minus, edges, torch.ones(len(indices), dtype=torch.long)

        with patch("compare_counters.candidate_pair", side_effect=pair), patch(
            "compare_counters.loss", side_effect=[torch.tensor(0.), torch.tensor(1.)] * 4
        ):
            proposal, stats, _ = no_counter_epoch(
                model, torch.zeros((2, 100)), torch.zeros(2, dtype=torch.long), args,
                torch.Generator().manual_seed(0), 1, torch.Generator().manual_seed(1))
        self.assertEqual(seen, [0, 1, 1, 1])
        self.assertEqual(stats["fires"], 1)
        self.assertEqual(int(proposal.sum()), 1)
        self.assertEqual(int(model.weights.sum()), 0)


if __name__ == "__main__":
    unittest.main()
