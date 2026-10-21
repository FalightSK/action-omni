# Paper 76 revision — 21 October 2026

This is a separate working copy. The original repository source is preserved in `baseline/main.tex`; no training or simulation was run for this revision.

## Evidence and manuscript changes

- **R1:** Separate independent head-training seeds from rollout scenes and repeated checkpoints. Retain the distinct three-seed, fixed-100-epoch ALOHA campaign with explicit repository-report provenance. Its reported counts, means, sample SDs and exact McNemar arithmetic are checked; raw execution and bootstrap output were unavailable for independent verification. Separate the later 37-checkpoint development sweep. ALOHA pairing is implemented; the per-policy `paired_claim:false` audit flag is not evidence of an unpaired experiment. Do not transfer the older matched cohort's outcome table to newer checkpoint digests. Historical gripper categories are observational and cannot establish causality.
- **R2:** Add all three completed LIBERO final evaluations: GR00T seeds 3/4, Qwen seed 3. State actual independent-run counts, GR00T mean 86.625% and sample SD 3.359 pp, and unavailable Qwen across-run SD. Disclose development-selected epochs 100/110/130 and unequal budgets. Document deterministic whole-episode validation, train-only normalization, metric averaging, sample counts, sampling RNG and three integration steps. Explicitly answer RQ2: flow loss and MAE disagree with seed-3 success ordering, MSE agrees. No universal offline-metric claim.
- **R3:** State that broader backbones, intermediate checkpoints, capacity controls, task coverage, physical robots and full-backbone fine-tuning are absent. Spatial 0/50 versus 0/50 is exploratory and includes a 152-to-192 token-budget deviation. Do not claim that the requested extensions were performed.
- **R4:** Use neutral frozen-endpoint empirical language; remove cost-benefit conclusions because costs were not measured. Restore the official submitted title exactly in substance, retaining its wording because no authority to rename it was supplied. The broader title remains a concern: abstract, results and conclusions explicitly narrow the claims. Preserve the original anonymous review author block until the user supplies camera-ready metadata.
- Remove the invalid/excluded one-camera condition as support for a camera-count claim. Keep campaigns distinct and do not pool historical, fixed-endpoint, development, and final rates.

## Verification completed

- Read all exported JSON/CSV/text evidence and verified all 60 manifest entries by byte count and SHA-256; no mismatch.
- Verified ALOHA and LIBERO protocol digests and disjoint episode IDs directly from saved local manifests. ALOHA validation IDs: 5, 20, 25, 39, 49; LIBERO uses 450/50 training/validation demonstrations.
- Recalculated independent-seed means/SDs and reported exact McNemar values; checked all 13 bibliography entries are cited and all LaTeX references resolve.
- Regenerated the sparse ALOHA figure from the 37 measured CSV rows only. Missing evaluations are never assigned zero.

## Remaining evidence limits

The six fixed-endpoint ALOHA raw evaluation records and their paired-bootstrap output are not locally present. The later export retains 447 raw cloud references but does not bundle those artifacts or split IDs. Latest sparse paired outcomes are not locally inspected. Historical submitted training provenance is incomplete. These limitations are disclosed rather than replaced with invented tests.

The parent task obtained and extracted the original attached PDF and supplied its exact title and anonymous author block. Those are restored in this review draft. Camera-ready author order, affiliations and email addresses still require user-supplied metadata. The original PDF itself was not materialized in this execution workspace. No private original PDF is included in the source package or Git revision.

## Build

Standard multi-file LaTeX: `pdflatex main.tex` twice, or `tectonic main.tex`, from this directory. References are embedded in `main.tex`; no separate `.bib` is required. Figure source is generated deterministically by the accompanying build script from the export CSV. Compiler/renderer dependencies and raw experiment exports must not be committed with the manuscript.
