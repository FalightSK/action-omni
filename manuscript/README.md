# Camera-ready manuscript draft

`main.tex` is an editable IEEE-style revision draft created on 18 October 2026
at the author's request. It is not the original authoring source and is not
submission ready. Merge author details, venue-specific requirements, and any
material missing from the original source when that source is supplied.

The draft compiles with the Codex desktop LaTeX compiler. A first attempt timed
out; subsequent compilation of the current source succeeded, including the
six-page maximum guard. Open `main.tex` in the desktop editor for its PDF preview.
Compilation does not replace a visual layout check; exact pagination and final
visual review remain to be checked in the preview.

## What is written

- Neutral title, narrow questions, related work, and corrected per-benchmark methods.
- Explicit distinction between training seeds, scene seeds, and rollout batches.
- Episode-disjoint validation, train-only normalization, checkpoint selection,
  paired analysis, and training-variability reporting.
- All six original result tables, explicitly labeled as submitted-protocol
  observations: LIBERO loss/success, ALOHA rollout batches, ALOHA stages,
  receiving-gripper failures, offline diagnostics, and text ablation.
- Corrected interpretation of paired counts, failures, offline metrics, and scope.
- An evidence-status subsection recording **Pending**, with zero completed new
  policies; no historical table is presented as a new replication.
- Limitations and a provisional conclusion that do not invent a replicated result.

No new performance numbers have been inserted. No old policy is counted as an
episode-holdout replication. Remove the draft-status notice and replace the
pending abstract/results/conclusion only after eligible experiments finish.

## References

The bibliography restores the **same 13 works as `paper_76.pdf`**, in its original
reference order. All 13 have in-text citations. The previous nine-entry draft
had omitted six original works and introduced separate Cosmos and Diffusion
Policy entries. Those two replacements have been removed; the GR00T N1.7 model
card supports the remaining Cosmos-lineage mention.

- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [pi0](https://arxiv.org/abs/2410.24164)
- [GR00T N1](https://arxiv.org/abs/2503.14734)
- [GR00T N1.7 released model card](https://huggingface.co/nvidia/GR00T-N1.7-3B)
- [RoboVLMs / What matters in building VLA models](https://www.nature.com/articles/s42256-025-01168-7)
- [ALOHA / ACT](https://arxiv.org/abs/2304.13705)
- [LIBERO](https://arxiv.org/abs/2306.03310)
- [Vision-based manipulators need to also see from their hands](https://arxiv.org/abs/2203.12677)
- [Critical Interval MSE](https://arxiv.org/abs/2606.29898)
- [LIBERO-PRO](https://arxiv.org/abs/2510.03827)
- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)

Metadata corrections are deliberate: RoboVLMs' first author is **X. Li**, not
the submitted **X. Liu**; its publisher confirms volume 8, pages 158–172 (2026).
The N1.7 reference identifies the accessible released model card/configuration
rather than inventing metadata for the submitted generic “Technical Report.”
OpenVLA is identified by its 2024 arXiv record; DiT and Flow Matching use their
2023 conference versions, as in the submission. These checks verify identifiable
records and the limited background claims used here, not every assertion in
every cited work. The emergence reference supports a conceptual analogy only.

## Alignment with the submitted paper

The source was reconciled directly against all six pages of
`C:\Users\SK\Downloads\paper_76.pdf` on 18 October 2026. It retains the original
seven-section progression: introduction, related work, experimental design,
closed-loop comparisons, offline metrics, limitations, and conclusion. Neutral
section titles replace economic wording. The abstract again states the original
camera contrast, ALOHA contrast, and offline mismatch, with their evidence status.

Restored discussion covers camera configuration, LIBERO routing diagnostics,
checkpoint observations, staged handover behavior, harness checks, the six-axis
interpretation limit, offline ranking reversal, and fixed-instruction language
ablation. The original four figures are **not reconstructed in this draft**;
their principal numerical observations are represented in the tables and prose.
Figures for revised results remain subject to eligible new experiment records.

Reviewer-driven corrections remain in force: head sizes differ across benchmarks;
rollout batches are not head-training replications; the recovered exact paired
LIBERO test is approximately 0.473; pooled paired ALOHA statistics are unavailable;
conditional stage rates and failure categories are descriptive; gripper behavior
and attention do not establish causation. The language-ablation control is not a
universal transferable floor. No original numerical claim is silently assigned
to the new episode-holdout protocol.

Validation after reconciliation: all 13 unique references are cited; all six
tables have resolved source labels; the existing source compiles successfully
with the desktop compiler and six-page guard. Final visual review remains pending.

## Later results integration

The paired-results command can export `generated_results.tex` and
`performance.pdf` from eligible saved outcomes. The draft deliberately does not
automatically include a possibly stale generated table or figure. Review the
actual completed counts and interpretations before incorporating these exports.
The implementation does not modify the original submitted PDF.
