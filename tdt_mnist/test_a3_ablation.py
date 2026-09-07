import unittest
from unittest.mock import patch
import torch
from activation_quantization import encode_activation, decode_activation, ActivationObserver
from train import TernaryModel, epoch, parser


class A3AblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_threshold_and_scale_are_separate_and_empty_safe(self):
        x = torch.tensor([[-4., -1., 0., 1., 4.], [0., 0., 0., 0., 0.]])
        codes, scale = encode_activation(x, 'a3', 'mean_threshold', .5)
        self.assertEqual(codes.tolist(), [[-1, 0, 0, 0, 1], [0, 0, 0, 0, 0]])
        self.assertEqual(scale.tolist(), [[4.], [1.]])
        self.assertTrue(torch.isfinite(decode_activation(codes, scale)).all())
        x = torch.tensor([[-4., -2., 0., 2., 4.]])
        codes, scale = encode_activation(x, 'a3', 'mean_threshold', .5)
        self.assertEqual(codes.tolist(), [[-1, -1, 0, 1, 1]])
        self.assertEqual(float(scale[0]), 3.)
        best_error = (x - decode_activation(codes, scale)).square().sum()
        for beta in (2., 2.9, 3.1, 4.):
            self.assertLess(float(best_error), float((x - codes.float()*beta).square().sum()))

    def test_signed_hidden_codes_and_read_only_diagnostics(self):
        x = torch.randn(16, 90, generator=torch.Generator().manual_seed(9))
        for method in ('absmax', 'mean_threshold'):
            relu = TernaryModel(pool_shape=(9, 10), hidden_size=1000, activation_precision='a3', a3_method=method)
            signed = TernaryModel(pool_shape=(9, 10), hidden_size=1000, activation_precision='a3', a3_method=method, hidden_activation='identity')
            self.assertTrue(torch.equal(relu.weights, signed.weights))
            for model in (relu, signed):
                expected = model(x)
                model.activation_observer = ActivationObserver(2, 'a3')
                self.assertTrue(torch.equal(expected, model(x)))
            self.assertNotIn('-1', relu.activation_observer.summary()[1]['code_histogram'])
            self.assertIn('-1', signed.activation_observer.summary()[1]['code_histogram'])
            self.assertFalse(torch.equal(relu(x), signed(x)))

    def test_zero_loss_difference_counts_pairs_not_coordinates(self):
        model = TernaryModel()
        args = parser().parse_args(['--measurements', '4', '--block-size', '8'])
        # Four exact FP32 pair differences: zero, positive, negative, zero.
        losses = [torch.tensor(x) for x in [1., 1., 2., 1., 1., 2., 3., 3.]]
        with patch('train.loss', side_effect=losses):
            _, _, stats, _ = epoch(model, torch.zeros(8, 100), torch.zeros(8, dtype=torch.long), args,
                                   torch.Generator().manual_seed(0), .02)
        self.assertEqual(stats['zero_difference_count'], 2)
        self.assertEqual(stats['zero_difference_fraction'], .5)


if __name__ == '__main__':
    unittest.main()
