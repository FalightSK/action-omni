import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.vision_patch_projection import (
    SUPPORTED_CLASS_MODULE,
    install_qwen3vl_patch_projection,
    linearized_patch_projection,
    uninstall_qwen3vl_patch_projection,
)


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, *, kernel=(2, 2, 2), channels=3, width=7):
        super().__init__()
        self.temporal_patch_size, self.patch_size = kernel[0], kernel[1]
        self.in_channels, self.embed_dim = channels, width
        self.proj = nn.Conv3d(channels, width, kernel_size=kernel, stride=kernel, bias=True)

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.in_channels, self.temporal_patch_size,
                                           self.patch_size, self.patch_size)
        return self.proj(hidden_states.to(self.proj.weight.dtype)).view(-1, self.embed_dim)


Qwen3VLVisionPatchEmbed.__module__ = SUPPORTED_CLASS_MODULE


class VisionPatchProjectionTests(unittest.TestCase):
    def test_cpu_float32_matches_conv3d_and_preserves_state_dict(self):
        torch.manual_seed(104)
        module = Qwen3VLVisionPatchEmbed().eval()
        inputs = torch.randn(8, 3, 2, 2, 2, dtype=torch.float32)
        expected = module(inputs)
        keys_before = list(module.state_dict())
        values_before = {key: value.clone() for key, value in module.state_dict().items()}

        actual = linearized_patch_projection(module, inputs)
        flat_actual = linearized_patch_projection(module, inputs.view(8, -1))

        self.assertEqual(actual.shape, (8, 7))
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(flat_actual, expected, atol=1e-5, rtol=1e-4)
        self.assertEqual(list(module.state_dict()), keys_before)
        for key, value in values_before.items():
            torch.testing.assert_close(module.state_dict()[key], value, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for BF16 parity")
    def test_cuda_bfloat16_matches_conv3d_with_explicit_tolerance(self):
        torch.manual_seed(105)
        module = Qwen3VLVisionPatchEmbed().to(device="cuda", dtype=torch.bfloat16).eval()
        inputs = torch.randn(8, 3, 2, 2, 2, device="cuda", dtype=torch.float32)
        expected = module(inputs)
        actual = linearized_patch_projection(module, inputs)
        flat = inputs.view(8, -1).to(torch.bfloat16)
        separate_bias = torch.nn.functional.linear(
            flat, module.proj.weight.view(module.embed_dim, -1), None)
        separate_bias.add_(module.proj.bias)
        # BF16 operators may round differently across Conv3d and GEMM kernels.
        torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.01)
        # Conv3d's CUDA path rounds the convolution result before adding bias.
        torch.testing.assert_close(actual, separate_bias, atol=0, rtol=0)

    def test_guarded_installer_is_idempotent_and_does_not_change_state_dict(self):
        model = nn.Module()
        model.patch_embed = Qwen3VLVisionPatchEmbed()
        keys_before = list(model.state_dict())
        state_before = {key: value.clone() for key, value in model.state_dict().items()}
        with patch("transformers.__version__", "4.57.6"):
            first = install_qwen3vl_patch_projection(model, required=True)
            second = install_qwen3vl_patch_projection(model, required=True)
        self.assertTrue(first.enabled)
        self.assertEqual(first.patched_modules, 1)
        self.assertEqual(second.patched_modules, 0)
        self.assertEqual(list(model.state_dict()), keys_before)
        for key, value in state_before.items():
            torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
        self.assertEqual(uninstall_qwen3vl_patch_projection(model), 1)
        self.assertEqual(uninstall_qwen3vl_patch_projection(model), 0)

    def test_wrong_transformers_version_leaves_forward_untouched(self):
        model = nn.Module()
        model.patch_embed = Qwen3VLVisionPatchEmbed()
        original = model.patch_embed.forward
        with patch("transformers.__version__", "4.58.0"):
            report = install_qwen3vl_patch_projection(model)
        self.assertFalse(report.enabled)
        self.assertEqual(report.patched_modules, 0)
        self.assertIs(model.patch_embed.forward.__func__, original.__func__)

    def test_invalid_input_shape_fails_loudly(self):
        module = Qwen3VLVisionPatchEmbed()
        with self.assertRaisesRegex(ValueError, "Expected flattened patches"):
            linearized_patch_projection(module, torch.randn(8, 3, 1, 2, 2))


if __name__ == "__main__":
    unittest.main()
