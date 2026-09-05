import unittest

import torch
import torch.nn.functional as F

from activation_quantization import (PRECISIONS, QMAX, ActivationObserver,
                                     encode_activation, decode_activation)
from train import TernaryModel


class ActivationQuantizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_float_formats_and_no_input_mutation(self):
        x = torch.tensor([[1.0001, -0.123456, 0.0]])
        before = x.clone()
        codes, scale = encode_activation(x, "a32")
        self.assertEqual(codes.data_ptr(), x.data_ptr())
        self.assertTrue(torch.equal(decode_activation(codes, scale), x))
        codes, scale = encode_activation(x, "a16")
        self.assertEqual(codes.dtype, torch.float16)
        self.assertTrue(torch.equal(decode_activation(codes, scale), x.half().float()))
        self.assertFalse(torch.equal(decode_activation(codes, scale), x))
        self.assertTrue(torch.equal(x, before))

    def test_integer_ranges_error_bounds_and_batch_independence(self):
        x = torch.randn((5, 100), generator=torch.Generator().manual_seed(20)) * 3
        x[0] = 0
        for precision, limit in QMAX.items():
            codes, scale = encode_activation(x, precision)
            restored = decode_activation(codes, scale)
            self.assertEqual(codes.dtype, torch.int8)
            self.assertLessEqual(int(codes.abs().max()), limit)
            self.assertTrue(torch.isfinite(restored).all())
            self.assertTrue(torch.all(codes[0] == 0))
            self.assertEqual(float(scale[0]), 1)
            self.assertTrue(torch.all((x - restored).abs() <= scale / 2 + 1e-6))
            for i in range(len(x)):
                row_codes, row_scale = encode_activation(x[i:i+1], precision)
                self.assertTrue(torch.equal(codes[i:i+1], row_codes))
                self.assertTrue(torch.equal(scale[i:i+1], row_scale))

    def test_ternary_means_three_values_and_ties_to_even(self):
        x = torch.tensor([[-1., -.5, 0., .5, 1.]])
        codes, scale = encode_activation(x, "a3")
        self.assertEqual(codes.tolist(), [[-1, 0, 0, 0, 1]])
        self.assertEqual(set(codes.flatten().tolist()), {-1, 0, 1})

    def test_a32_matches_original_forward_and_observer_is_read_only(self):
        model = TernaryModel(pool_shape=(9, 10), hidden_size=100)
        x = torch.randn((8, 90), generator=torch.Generator().manual_seed(1))
        w1 = model.weights[:9000].view(100, 90).float() * model.scales[0]
        w2 = model.weights[9000:].view(10, 100).float() * model.scales[1]
        expected = F.linear(F.relu(F.linear(x, w1)), w2)
        self.assertTrue(torch.equal(model(x), expected))
        model.activation_observer = ActivationObserver(2, "a32")
        self.assertTrue(torch.equal(model(x), expected))
        for row in model.activation_observer.summary():
            self.assertEqual(row["mse"], 0)

    def test_paired_initial_weights_and_a3_relu_codes(self):
        x = torch.randn((8, 90), generator=torch.Generator().manual_seed(1))
        initial = None
        for precision in PRECISIONS:
            model = TernaryModel(pool_shape=(9, 10), hidden_size=100, activation_precision=precision)
            if initial is None:
                initial = model.weights.clone()
            self.assertTrue(torch.equal(model.weights, initial))
            model.activation_observer = ActivationObserver(2, precision)
            output = model(x)
            self.assertTrue(torch.isfinite(output).all())
            self.assertFalse(output.requires_grad)
            self.assertEqual(model.num_params, 10000)
            rows = model.activation_observer.summary()
            self.assertEqual([r["values"] for r in rows], [8 * 90, 8 * 100])
            if precision == "a3":
                self.assertEqual(set(rows[0]["code_histogram"]), {"-1", "0", "1"})
                self.assertEqual(set(rows[1]["code_histogram"]), {"0", "1"})


if __name__ == "__main__":
    unittest.main()
