"""Run: python -m unittest discover -s tests -v. Synthetic data only; no retraining."""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import random
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import torch

from configs.base_config import BaseVLAConfig
from data.hdf5_embeddings import HDF5EmbeddingDataset, HDF5EmbeddingWriter
from experiment_protocol import (apply_normalization, cache_config, checkpoint_config,
                                 digest, encoder_code_id, fixed_rng, load_protocol,
                                 protocol_arrays, restore_rng, rng_state,
                                 run_signature, sampling_seed, seed_everything, split_episodes,
                                 write_json)
from scripts.prepare_protocol import prepare
from scripts.analysis.paired_results import (compare_records, holm, paired_statistics,
                                            training_summary, wilson, export_report)
from scripts.analysis.camera_ready_diagnostics import classify_trace, rollout_diagnostics, offline_metrics


def fixture_protocol():
    episodes = [{"id": i, "task": "task", "name": str(i), "start": 2*i,
                 "stop": 2*i+2} for i in range(10)]
    p = {"schema": 1, "dataset": "aloha", "n_samples": 20, "split_seed": 42,
         "episodes": split_episodes(episodes), "normalization": {}}
    p["protocol_id"] = digest(p)
    return p


class ProtocolTests(unittest.TestCase):
    def test_canonical_padding_matches_batch_one_for_both_padding_sides(self):
        from models.vla import canonicalize_vlm_tokens
        tokens = torch.tensor([[[1.], [2.], [99.]], [[99.], [3.], [4.]]])
        attention = torch.tensor([[1, 1, 0], [0, 1, 1]])
        image = torch.tensor([[True, False, False], [False, True, False]])
        result, mask = canonicalize_vlm_tokens(tokens, image, attention, 4)
        self.assertTrue(torch.equal(result[:, :, 0], torch.tensor([[1.,2.,0.,0.],[3.,4.,0.,0.]])))
        self.assertTrue(torch.equal(mask, torch.tensor([[True,False,False,False]]*2)))
        multi, _ = canonicalize_vlm_tokens(tokens[:, None], image, attention, 4)
        self.assertTrue(torch.equal(multi[:, 0], result))
        with self.assertRaises(ValueError):
            canonicalize_vlm_tokens(tokens, image, attention, 1)

    def test_episode_split_is_stable_disjoint_and_stratified(self):
        p = fixture_protocol()
        self.assertEqual(p["episodes"], split_episodes(p["episodes"]))
        ids, val = protocol_arrays(p)
        self.assertEqual(int(val.sum()), 2)
        self.assertFalse(set(ids[val]) & set(ids[~val]))
        with self.assertRaises(ValueError):
            split_episodes([{ "id": 0, "task": "only" }])
        bad = deepcopy(p)
        bad["episodes"][1]["start"] = 1
        with self.assertRaises(ValueError):
            protocol_arrays(bad)

    def test_manifest_detects_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "protocol.json"
            p = fixture_protocol()
            write_json(path, p)
            self.assertEqual(load_protocol(path, "aloha"), p)
            p["split_seed"] = 99
            write_json(path, p)
            with self.assertRaises(ValueError):
                load_protocol(path)

    def test_normalization_uses_training_frames_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data/chunk-000").mkdir(parents=True)
            rows = []
            for ep in range(10):
                for frame in range(2):
                    rows.append({"index": 2*ep+frame, "episode_index": ep,
                                 "observation.state": [float(ep)]*14,
                                 "action": [float(ep + frame)]*14})
            path = root / "data/chunk-000/file-000.parquet"
            pd.DataFrame(rows).to_parquet(path)
            first = prepare("aloha", root)
            held_out = next(e["id"] for e in first["episodes"] if e["split"] == "validation")
            for row in rows:
                if row["episode_index"] == held_out:
                    row["observation.state"] = [1e6]*14
                    row["action"] = [-1e6]*14
            pd.DataFrame(rows).to_parquet(path)
            second = prepare("aloha", root)
            self.assertEqual(first["normalization"], second["normalization"])
            self.assertEqual(first["receiver_gripper"], second["receiver_gripper"])
            self.assertNotEqual(first["protocol_id"], second["protocol_id"])
            cfg = BaseVLAConfig(state_dim=14, action_dim=14)
            apply_normalization(cfg, first)
            self.assertEqual(len(cfg.state_mean), 14)

    def test_fixed_validation_preserves_all_rng_streams_and_resume(self):
        seed_everything(7)
        saved = rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        restore_rng(saved)
        with fixed_rng(4242):
            draw = torch.rand(3)
            random.random()
            np.random.rand()
        actual = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))
        with fixed_rng(4242):
            self.assertTrue(torch.equal(draw, torch.rand(3)))
        self.assertEqual(sampling_seed("aloha", 0, 0, 1), sampling_seed("aloha", 0, 0, 1))
        self.assertNotEqual(sampling_seed("aloha", 0, 0, 1), sampling_seed("aloha", 1, 0, 1))

    def test_resumed_shuffle_is_independent_of_worker_initialization(self):
        from torch.utils.data import DataLoader, RandomSampler
        data = torch.arange(20)
        shuffle = torch.Generator().manual_seed(0)
        loader = DataLoader(data, batch_size=4, sampler=RandomSampler(data, generator=shuffle),
                            generator=torch.Generator().manual_seed(0))
        list(loader)
        saved = shuffle.get_state()
        expected = torch.cat(list(loader))
        resumed = torch.Generator()
        resumed.set_state(saved)
        restarted = DataLoader(data, batch_size=4, sampler=RandomSampler(data, generator=resumed),
                               generator=torch.Generator().manual_seed(999))
        self.assertTrue(torch.equal(expected, torch.cat(list(restarted))))

    def test_checkpoint_rejects_wrong_arm_and_seed_signature(self):
        p, cfg = fixture_protocol(), BaseVLAConfig()
        ck = {"cfg": cfg, "protocol": p, "training_seed": 0,
              "run_signature": run_signature(cfg, p, 0)}
        self.assertIs(checkpoint_config(cfg, ck, "aloha"), cfg)
        wrong = deepcopy(cfg)
        wrong.dit_hidden_dim += 8
        with self.assertRaises(ValueError):
            checkpoint_config(wrong, ck, "aloha")
        ck["training_seed"] = 1
        with self.assertRaises(ValueError):
            checkpoint_config(cfg, ck, "aloha")

    def test_hdf5_metadata_roundtrip_and_partial_cache_rejection(self):
        p, cfg = fixture_protocol(), BaseVLAConfig()
        ids, val = protocol_arrays(p)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.h5"
            w = HDF5EmbeddingWriter(path, 20, 2, 3, 2, 2, 2, attrs={
                "protocol_id": p["protocol_id"], "full_dataset": True,
                "cache_id": "test", "cache_config": json.dumps(cache_config(cfg)),
                "encoder_code_id": encoder_code_id(), "vlm_extract_layers": [28]})
            w.f.create_dataset("episode_ids", data=ids)
            w.f.create_dataset("validation_mask", data=val)
            w.write(0, torch.zeros(20, 2, 3, dtype=torch.bfloat16),
                    torch.ones(20, 2, dtype=torch.bool), torch.zeros(20, 2), torch.zeros(20, 2, 2))
            w.close()
            ds = HDF5EmbeddingDataset(path, expect_layers=(28,), protocol=p, config=cfg)
            self.assertEqual(ds[0]["embedding"].shape, (2, 3))
            ds._h5.close()
            bad = deepcopy(cfg)
            bad.action_mean = (99.,)
            with self.assertRaises(ValueError):
                HDF5EmbeddingDataset(path, protocol=p, config=bad)
            bad = deepcopy(cfg)
            bad.vlm_bf16_reduced_precision_reduction = False
            with self.assertRaises(ValueError):
                HDF5EmbeddingDataset(path, protocol=p, config=bad)
            partial = Path(d) / "partial.h5"
            w = HDF5EmbeddingWriter(partial, 2, 2, 3, 2, 2, 2)
            with self.assertRaises(ValueError):
                w.close()
            with self.assertRaises(ValueError):
                HDF5EmbeddingDataset(partial)


class StatisticsTests(unittest.TestCase):
    def test_wilson_extremes_and_reference(self):
        self.assertAlmostEqual(wilson(5, 10)[0], 0.236593090512564, places=12)
        self.assertEqual(wilson(0, 10)[0], 0.)
        self.assertEqual(wilson(10, 10)[1], 1.)
        with self.assertRaises(ValueError):
            wilson(0, 0)

    def test_known_discordance_and_no_discordance(self):
        a = [1]*73 + [0]*55 + [1]*47 + [0]*25
        b = [1]*73 + [0]*55 + [0]*47 + [1]*25
        s = paired_statistics(a, b, resamples=1000)
        self.assertAlmostEqual(s["mcnemar_exact_p"], binomtest(47, 72).pvalue)
        self.assertAlmostEqual(s["difference_pp"], 11.)
        self.assertEqual(paired_statistics([1, 1], [1, 1])["mcnemar_exact_p"], 1.)
        with self.assertRaises(ValueError):
            paired_statistics([1], [0, 1])

    def test_correction_and_training_unit(self):
        self.assertEqual(holm([0.01, 0.04, 0.03], 3), [0.03, 0.06, 0.06])
        self.assertIsNone(training_summary([0.5])["sample_sd"])
        self.assertAlmostEqual(training_summary([0.4, 0.5, 0.6])["sample_sd"], 0.1)

    def test_pair_keys_and_sampling_seeds_must_match(self):
        left = {"dataset": "aloha", "training_seed": 0, "protocol_id": "protocol",
                "checkpoint_sha256": "a", "run_signature": "a", "cache_id": "a",
                "configuration": {"vlm_backbone": "groot", "batch_size": 128},
                "episodes": [{"initial_state_id": 0, "is_success": True,
                              "environment_seed": 42, "policy_sampling_seed": 123}]}
        right = deepcopy(left)
        right["checkpoint_sha256"] = "b"
        right["configuration"]["vlm_backbone"] = "qwen3vl"
        compare_records(left, right, resamples=10)
        right["episodes"][0]["policy_sampling_seed"] = 124
        with self.assertRaises(ValueError):
            compare_records(left, right)
        right["episodes"][0]["policy_sampling_seed"] = 123
        right["configuration"]["batch_size"] = 256
        with self.assertRaises(ValueError):
            compare_records(left, right)
        right = deepcopy(left)
        right["checkpoint_sha256"] = "b"
        right["episodes"].append(dict(right["episodes"][0]))
        with self.assertRaises(ValueError):
            compare_records(left, right)

    def test_operational_failure_labels(self):
        reward = np.array([1, 2, 2, 2])
        state = np.zeros((4, 14))
        action = state.copy()
        self.assertEqual(classify_trace(state, reward, action, .5), "receiver_never_opened")
        state[2:, 6] = 1
        self.assertEqual(classify_trace(state, reward, action, .5), "opened_not_closed")
        state[3, 6] = 0
        self.assertEqual(classify_trace(state, reward, action, .5), "opened_then_closed_without_transfer")
        reward[-1] = 4
        self.assertEqual(classify_trace(state, reward, action, .5), "success")

    def test_rollout_diagnostics_preserve_cohort_and_commands(self):
        p = fixture_protocol()
        p["receiver_gripper"] = {"dimension": 6, "low": 0., "high": 1.}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.npz"
            action = np.zeros((4, 14))
            action[:, 6] = 1
            np.savez(path, n_episodes=[1], initial_state_ids=[0],
                     ep0000_next_state=np.zeros((4, 14)), ep0000_action=action,
                     ep0000_reward=[1, 2, 2, 2], ep0000_attn_image_mass=[.8, .9])
            result = {"protocol_id": p["protocol_id"], "checkpoint_sha256": "sha", "dataset": "aloha",
                      "trace": str(path), "inference_horizon": 2,
                      "episodes": [{"initial_state_id": 0, "max_reward": 2, "is_success": False}]}
            diagnostic = rollout_diagnostics(result, p, "sha")["0.6"]
            self.assertEqual(diagnostic["counts"]["receiver_never_opened"], 1)
            self.assertEqual(diagnostic["never_opened_with_commanded_open"], 1)
            result["episodes"][0]["is_success"] = True
            with self.assertRaises(ValueError):
                rollout_diagnostics(result, p, "sha")

    def test_actual_head_forward_and_sampling_without_training(self):
        from models.vla_train import VLATrainModel
        cfg = BaseVLAConfig(vlm_hidden_size=16, vlm_adapter_dim=32, lora_rank=2,
                            pos_enc_dim=8, readout_heads=2, adapter_dropout=0.,
                            dit_hidden_dim=32, dit_num_layers=1, dit_num_heads=2,
                            decoder_dropout=0., action_horizon=2, img_seq_len=4,
                            img_grid_h=2, img_grid_w=2)
        seed_everything(0)
        model = VLATrainModel(cfg).eval()
        embed, state, action = torch.randn(2, 4, 16), torch.randn(2, 2), torch.randn(2, 2, 2)
        mask = torch.ones(2, 4, dtype=torch.bool)
        with torch.no_grad(), fixed_rng(4242):
            loss = model(embed, state, action, mask)
            pred = model.sample(embed, state, img_mask=mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(pred.shape, (2, 4))
        with torch.no_grad(), fixed_rng(4242):
            self.assertTrue(torch.equal(loss, model(embed, state, action, mask)))
        class TinyCache(torch.utils.data.Dataset):
            def __init__(self):
                self.actions = torch.randn(20, 2, 2)
                self.states = torch.randn(20, 2)
                self.embeddings = torch.randn(20, 4, 16)
            def __len__(self):
                return 20
            def __getitem__(self, i):
                return {"embedding": self.embeddings[i], "state": self.states[i],
                        "actions": self.actions[i], "img_mask": torch.ones(4, dtype=torch.bool)}
        cache = TinyCache()
        metrics = offline_metrics(model, cache, fixture_protocol(), "cpu", batch_size=1)
        self.assertEqual(metrics["validation_frames"], 2)
        self.assertEqual(metrics["validation_episodes"], 1)
        self.assertTrue(np.isfinite(metrics["flow_loss_frame_weighted"]))
        self.assertEqual(metrics, offline_metrics(model, cache, fixture_protocol(), "cpu", batch_size=1))

    def test_report_exports_only_supplied_training_values(self):
        summaries = {"Synthetic & check": {
            "left_success_rate": training_summary([0.4, 0.5, 0.6]),
            "right_success_rate": training_summary([0.3, 0.4, 0.5]),
            "difference_pp": training_summary([10., 10., 10.])}}
        with tempfile.TemporaryDirectory() as d:
            export_report(d, summaries, {})
            table = (Path(d) / "generated_results.tex").read_text(encoding="utf-8")
            self.assertIn(r"Synthetic \& check", table)
            self.assertIn("50.00", table)
            self.assertGreater((Path(d) / "performance.pdf").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
