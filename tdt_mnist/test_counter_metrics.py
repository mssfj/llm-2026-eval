import unittest

import torch

from train import TernaryModel, counter_statistics, epoch, parser


class CounterMetricsTests(unittest.TestCase):
    def test_measured_edges_exclude_unvisited_zeros_and_include_saturation(self):
        evidence = torch.tensor([[-127, 127], [4, 0], [0, 0]], dtype=torch.int8)
        counts = torch.tensor([[1, 1], [2, 1], [0, 0]])
        stats = counter_statistics(evidence, counts, 127)
        self.assertEqual(stats["counter_count"], 4)
        self.assertEqual(stats["counter_min"], -127)
        self.assertEqual(stats["counter_max"], 127)
        self.assertEqual(stats["counter_mean"], 1.0)
        self.assertEqual(stats["counter_abs_mean"], 64.5)
        self.assertEqual(stats["counter_saturated_count"], 2)
        self.assertEqual(stats["counter_saturated_fraction"], 0.5)
        self.assertEqual(stats["counter_histogram"], {"-127": 1, "0": 1, "4": 1, "127": 1})
        self.assertAlmostEqual(stats["counter_all_mean"], 4 / 6)

    def test_batch_rng_is_independent_of_block_size(self):
        torch.set_num_threads(1)
        x = torch.arange(3200, dtype=torch.float32).view(32, 100) / 3200
        y = torch.arange(32) % 10
        states = []
        for block in (1, 8, 32):
            model = TernaryModel()
            args = parser().parse_args(["--measurements", "4", "--block-size", str(block)])
            batch_rng = torch.Generator().manual_seed(100)
            proposal, indices, stats, scale = epoch(
                model, x, y, args, torch.Generator().manual_seed(1), 0.02, batch_rng)
            states.append(batch_rng.get_state())
            self.assertEqual(sum(stats["counter_histogram"].values()), stats["counter_count"])
            self.assertLessEqual(stats["counter_abs_max"], stats["counter_peak_abs"])
            self.assertLessEqual(stats["counter_peak_abs"], args.measurements)
            self.assertEqual(stats["counter_capacity"], 127)
        self.assertTrue(all(torch.equal(states[0], state) for state in states[1:]))


if __name__ == "__main__":
    unittest.main()
