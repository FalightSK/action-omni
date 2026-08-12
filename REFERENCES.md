# References

Literature this study positions against. Kept short and load-bearing: each entry
is here because it constrains a claim made in `RESULTS.md`.

## Frozen visual backbones for control

| Work | What it shows | Relevance |
|---|---|---|
| **R3M** (Nair et al., 2022) | Frozen ResNet visual features transfer to robot policies | Vision-only, no language — establishes that a frozen encoder can drive control |
| **MVP** (Radosavovic et al., 2023) | Frozen MAE-ViT features transfer to manipulation | Same, with a ViT |
| **DINOv2** (Oquab et al., 2023) | Frozen self-supervised features used for robotics | Same, self-supervised |
| **RoboFlamingo** (Li et al., 2023) | Frozen VLM + trained head | Only *partially* freezes the visual encoder, so it is not a clean frozen-backbone reference |

None of these test whether **robot-data pretraining** of the backbone adds
anything over its stock base under a fixed head — the comparison this study runs.

## Full-finetuning VLAs — the opposing position

| Work | Their claim | What this study tests |
|---|---|---|
| **RT-2** (Brohan et al., 2023) | Finetuning the VLM on robot data produces emergent generalization | Whether the pretraining, rather than the complete system, is what produces the performance |
| **OpenVLA** (Kim et al., 2024) | A 7B finetuned VLM outperforms prior VLA systems | Same |
| **Octo** (Team, 2023) | Large-scale robot-data pretraining is the path to generalist policies | Same |
| **π₀ / π₀.₅** (Black et al., 2024; Intelligence et al., 2025) | VLM plus flow matching, backbone finetuned, for dexterous manipulation | The closest architectural neighbour: same decoder family as this study's DiT flow-matching head, but with a *finetuned* backbone and a much larger action expert. Also the source of the normalisation and timestep-schedule choices audited in `RESULTS.md` §5 |
| **GR00T N1** (NVIDIA, 2025) | Robot-pretrained VLM + diffusion head for humanoid manipulation | Supplies one arm of this study's pair; its `select_layer` sets the read depth |

This study does **not** contradict any of these published results — none of their
systems were run here. Their papers claim the complete system works; they do not
establish that the VLM's robot pretraining is what produces that performance.
That attribution is what is under test.

**These are cited as literature only.** Their reported numbers are not part of
this study's evidence base and are never pooled with, compared against, or
tabulated alongside the six arms in `RESULTS.md`. Every number in that document
comes from this stack, this head and this harness. Mixing an externally reported
success rate into a table of internally measured ones would compare across
different heads, action spaces, training budgets and evaluation protocols at
once — the exact confound the single-head design exists to avoid.

## Reference points on the benchmarks used

| Work | Relevance |
|---|---|
| **SmolVLA** (Shukor et al., 2025) | A full-finetuning VLA reproduced here on LIBERO-Spatial at 76.0% SR (100 rollouts). Also notable because its vision-language backbone ships bit-identical to stock SmolVLM2 — a published, working VLA that never finetuned its VLM |
| **LIBERO** (Liu et al., 2023) | Source of the LIBERO-Goal suite: 10 goals over one fixed scene, which is what isolates the instruction pathway |
| **ALOHA / ACT** (Zhao et al., 2023) | Source of the bimanual transfer-cube task and the receding-horizon action-chunking convention used here |

## Method

| Work | Relevance |
|---|---|
| **Rectified flow / flow matching** (Liu et al., 2022; Lipman et al., 2023) | The decoder's training objective: `x_t = (1-t)·x₀ + t·actions`, velocity regression |
| **DiT** (Peebles & Xie, 2023) | The decoder's architecture; adaLN conditioning is the global pathway, cross-attention the local one |
| **Diffusion Policy** (Chi et al., 2023) | Establishes action chunking with a receding horizon as the control convention |
