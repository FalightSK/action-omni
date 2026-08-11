# References

Extracted from the retired `RESEARCH.md`, `README.md`, `MILESTONES.md` before
those were removed for the pipeline rework. Kept because the revised plan will
still need to position itself against this literature; everything else in those
files (the old hypotheses/phases/roadmap) was project narrative, not reference
material, and was not preserved.

## Supporting frozen-backbone evidence

| Paper | What it shows |
|---|---|
| **R3M** (Nair et al., 2022) | Frozen ResNet visual features transfer to robot policies. Vision-only, no language. |
| **MVP** (Radosavovic et al., 2023) | Frozen MAE-ViT features transfer to manipulation. Vision-only, no language. |
| **DINOv2 for robotics** (2023) | Frozen self-supervised ViT features used for robotics (cited in README as related but not detailed further). |
| **RoboFlamingo** (Li et al., 2023) | Frozen VLM + finetuned head. Note: only *partially* freezes the visual encoder. |

## Opposing position — full/partial VLM finetuning

| Paper | Their claim |
|---|---|
| **RT-2** (Brohan et al., 2023) | Finetuning the VLM on robot data produces emergent generalization. |
| **OpenVLA** (Kim et al., 2024) | A 7B finetuned VLM outperforms prior VLA systems. |
| **π0** (Black et al., 2024) | VLM + flow matching (backbone finetuned) for dexterous tasks — same decoder family as this project's DiT flow-matching decoder, but with a finetuned backbone. |
| **Octo** (Team, 2023) | Large-scale robot-data pretraining is the path to generalist policies. |

## Critical baselines

| Paper | Relevance |
|---|---|
| **CLIP** (Radford et al., 2021) | Contrastive image-text pretraining; zero-shot aligns color/object names with images. The baseline any color/attribute-OOD claim must beat. |
| **CLIPort** (2021) | CLIP-conditioned Diffusion Policy for language-conditioned manipulation — the baseline to beat for language-conditioned tasks specifically. |

## Note on gaps identified against this literature (may or may not still be relevant to the revised plan)

- None of the frozen-backbone papers (R3M, MVP, RoboFlamingo) test *joint* language-visual
  pretraining (VLM) against vision-only encoders (ViT) for generalization.
- None of the finetuning papers (RT-2, OpenVLA, π0, Octo) compare a finetuned VLM against a
  frozen VLM on the same out-of-distribution generalization test — they show finetuned VLMs
  work, not that finetuning is *necessary* relative to frozen.
