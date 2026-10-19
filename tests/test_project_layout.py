"""Check import/CLI compatibility without loading weights or running experiments."""
import importlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


class ProjectLayoutTests(unittest.TestCase):
    def test_encoder_precision_matches_protocol_and_restores_global_flag(self):
        from configs.base_config import BaseVLAConfig
        from experiment_protocol import apply_normalization, cache_config
        from models.vla import VLAModel

        cfg = BaseVLAConfig()
        apply_normalization(cfg, {"normalization": {}})
        self.assertFalse(cache_config(cfg)["vlm_bf16_reduced_precision_reduction"])
        model = VLAModel.__new__(VLAModel)
        torch.nn.Module.__init__(model)
        model.config = cfg
        previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        try:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            def encode(_inputs):
                self.assertFalse(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
                self.assertFalse(torch.is_grad_enabled())
                return "encoded"
            with patch.object(model, "_encode_vlm", side_effect=encode):
                self.assertEqual(model.encode_vlm({}), "encoded")
            self.assertTrue(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
            with patch.object(model, "_encode_vlm", side_effect=RuntimeError("encoding failed")):
                with self.assertRaises(RuntimeError):
                    model.encode_vlm({})
            self.assertTrue(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous

    def test_legacy_commands_forward_to_the_canonical_pipeline(self):
        for legacy, canonical, options in (
            ("train", "train", ["--epochs", "1", "--dry-run"]),
            ("precompute_embeddings", "precompute", ["--max-samples", "5", "--output", "x.h5"]),
            ("inference", "evaluate", ["--episodes", "2", "--no-video"]),
        ):
            with self.subTest(command=legacy):
                with patch(f"scripts.{canonical}.main") as run:
                    module = importlib.import_module(legacy)
                    importlib.reload(module)  # importing must not execute a command
                    run.assert_not_called()
                    module.main(["--exp", "2", *options])
                    run.assert_called_once_with(
                        ["--dataset", "pusht", "--exp", "exp02a", *options]
                    )

    def test_legacy_public_imports_and_backbone_paths(self):
        from envs.pusht_env import PushTAgent, run_episode
        from inference import PushTAgent as old_agent, run_episode as old_episode
        from models.vla_train import VLATrainModel
        from train import VLATrainModel as old_model
        from models import backbones
        from scripts.analysis.latent_compare import backbones as old_backbones

        self.assertIs(old_agent, PushTAgent)
        self.assertIs(old_episode, run_episode)
        self.assertIs(old_model, VLATrainModel)
        self.assertIs(old_backbones.Backbone, backbones.Backbone)
        self.assertIs(old_backbones.load_backbone, backbones.load_backbone)
        self.assertEqual(backbones.MODELS, Path(__file__).resolve().parents[1] / "asset/models")

    def test_runtime_uses_the_shared_backbone_loader(self):
        from configs.base_config import BaseVLAConfig
        from models.vla import VLAModel

        cfg = BaseVLAConfig()
        cfg.vlm_backbone = "groot"
        backbone = SimpleNamespace(
            processor=object(), model=torch.nn.Linear(1, 1),
            final_norm=torch.nn.Identity(), hidden_size=cfg.vlm_hidden_size,
            image_token_id=123,
        )
        with patch("models.backbones.load_backbone", return_value=backbone) as load:
            with patch("models.vla.VLMTokenAdapter", return_value=torch.nn.Linear(1, 1)):
                with patch("models.vla.FlowMatchingDecoder", return_value=torch.nn.Linear(1, 1)):
                    model = VLAModel(cfg)
        load.assert_called_once_with("groot", device="cpu", dtype=torch.bfloat16)
        self.assertIs(model.vlm, backbone.model)
        self.assertIs(model.processor, backbone.processor)
        self.assertEqual(cfg.image_token_id, 123)
        self.assertTrue(all(not p.requires_grad for p in model.vlm.parameters()))


if __name__ == "__main__":
    unittest.main()
