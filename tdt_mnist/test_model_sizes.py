import contextlib
import io
import unittest

import torch

from train import TernaryModel, model_parameter_count, parser, pooled_shape, validate


class ModelSizesTests(unittest.TestCase):
    def test_exact_10000_two_layer_forward(self):
        model = TernaryModel(pool_shape=(9, 10), hidden_size=100)
        self.assertEqual(model.num_params, 10000)
        self.assertEqual(model.shapes, [(100, 90), (10, 100)])
        self.assertEqual(model.pool_shape, (9, 10))
        self.assertEqual(model.weights.dtype, torch.int8)
        logits = model(torch.ones(4, 90))
        self.assertEqual(tuple(logits.shape), (4, 10))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertFalse(logits.requires_grad)
        self.assertTrue(torch.all(model.weights.abs() <= 1))

    def test_default_and_scaling(self):
        self.assertEqual(TernaryModel().num_params, 1000)
        self.assertEqual(model_parameter_count(hidden_size=16), 1760)
        self.assertEqual(model_parameter_count(hidden_size=100, pool_shape=(9, 10)), 10000)
        self.assertEqual(model_parameter_count(hidden_size=200, pool_shape=(9, 10)), 20000)
        self.assertEqual(pooled_shape(14), (14, 14))

    def test_cli_checks_exact_count_and_dimensions(self):
        p = parser()
        good = p.parse_args(["--pool-shape", "9", "10", "--hidden-size", "100", "--expected-params", "10000"])
        validate(good, p)
        for options in (["--expected-params", "10000"], ["--pool-shape", "0", "10"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                validate(p.parse_args(options), p)


if __name__ == "__main__":
    unittest.main()
