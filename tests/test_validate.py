import unittest

import torch

from validate import require_exact, validate_cpu


class ValidationTests(unittest.TestCase):
    def test_order_and_rounding_witnesses(self):
        self.assertEqual(validate_cpu(), {"witness_outputs": 24, "reordered_outputs_rejected": 16})

    def test_bitwise_comparison_distinguishes_signed_zero(self):
        with self.assertRaises(AssertionError):
            require_exact(torch.tensor([0.0]), torch.tensor([-0.0]), "fixture")

    def test_bitwise_comparison_rejects_dtype_and_shape_changes(self):
        reference = torch.ones(1, 2)
        for wrong in (reference.flatten(), reference.bfloat16()):
            with self.subTest(shape=wrong.shape, dtype=wrong.dtype), self.assertRaises(AssertionError):
                require_exact(wrong, reference, "fixture")
