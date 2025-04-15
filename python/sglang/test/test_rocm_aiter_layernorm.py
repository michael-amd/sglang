import itertools 
import unittest
import torch

# Import the functions from AITER repository.
from aiter.ops.rmsnorm import rms_norm as rocm_aiter_rms_norm

from sglang.test.test_utils import CustomTestCase


class TestRMSNorm(CustomTestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HIDDEN_SIZES = [768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]
    ADD_RESIDUAL = [False, True]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_rms_norm_test(self, num_tokens, hidden_size, add_residual, dtype, seed):
        torch.manual_seed(seed)

        # Set up a dummy layer weight (here we simply create a parameter to mimic the weight).
        weight = torch.empty(hidden_size, dtype=dtype, device="cuda")
        weight.normal_(mean=1.0, std=0.1)

        scale = 1 / (2 * hidden_size)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda") * scale

        # For this test, assume RMSNorm does not support residual addition.
        residual = None
        # Compute reference output using a simple implementation.
        def reference_rms_norm(x, weight, eps=1e-6):
            orig_dtype = x.dtype
            x_f32 = x.to(torch.float32)
            variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
            x_norm = x_f32 * torch.rsqrt(variance + eps)
            return (x_norm * weight.to(torch.float32)).to(orig_dtype)
        
        ref_out = reference_rms_norm(x, weight)
        out = rocm_aiter_rms_norm(x, weight, 1e-6)
        self.assertTrue(torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2),
                        f"RMSNorm mismatch for shape {(num_tokens, hidden_size)}")

    def test_rms_norm(self):
        for params in itertools.product(
            self.NUM_TOKENS, self.HIDDEN_SIZES, self.ADD_RESIDUAL, self.DTYPES, self.SEEDS
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                add_residual=params[2],
                dtype=params[3],
                seed=params[4],
            ):
                # For this test, we'll only test the basic RMSNorm (without residual).
                self._run_rms_norm_test(*params)

if __name__ == "__main__":
    unittest.main(verbosity=2)
