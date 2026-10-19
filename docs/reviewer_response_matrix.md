# Reviewer response implementation matrix

Status: pre-retraining implementation draft, 18 October 2026. This is an internal
record, not a claim that the planned experiments have completed.

| Concern | Implemented response | Evidence/location | Remaining work |
|---|---|---|---|
| R1: paired ALOHA analysis | Exact paired tables/tests, paired bootstrap, corrected intervals and Holm reporting. | `scripts/analysis/paired_results.py`; synthetic tests; manuscript Statistics. | Actual matched policy outcomes and interpretation. |
| R1: two independent runs | Separate training seed, environment start, and action-sampling seed in metadata; historical rollout batches are not called training replications. | Training/evaluation metadata; manuscript Introduction and submitted-results subsection. | Complete independent training runs and report their actual count. |
| R1: scope | Neutral title and endpoint/frozen-head/simulation limitations. | `manuscript/main.tex`, Introduction, Discussion, Conclusion. | Rewrite final conclusions from actual results. |
| R1: gripper causality | Operational labels, one common threshold, sensitivity check, same-cohort accounting, commanded/observed distinction. | `camera_ready_diagnostics.py`; manuscript observed-failures subsection. | Real traces and simulator semantic check. |
| R2: training variance and mean/SD | Three-seed target; per-policy reporting; sample SD across heads, undefined for one policy. | Training seed support; paired reporter; manuscript evidence-status subsection. | Train/evaluate the target 12 primary policies after confirmation. |
| R2: LIBERO run count | Explicit target and zero completed revised policies; historical independent training counts unverified. | Manuscript evidence-status subsection; future run manifests. | Populate counts and outcomes from completed runs. |
| R2: validation separation | Episode-disjoint shared split and training-only ALOHA normalization. | Prepared manifests; split/normalization tests; manuscript Demonstrations and validation. | Cache actual features and verify cache/live parity. |
| R2: offline loss versus rollout success | Distinct datasets/quantities and checkpoint-selection versus diagnostic measures. | New holdout diagnostic command; manuscript methods and RQ2 subsection. | Compute metrics for new saved policies and answer RQ2 from those results. |
| R2: intermediate checkpoint | Optional Cosmos config is available as `aloha/exp07`. | Config registry and inherited ALOHA recipe. | Only run if authorized core work and deadline permit. |
| R2: additional families/datasets or narrower claims | Narrowed claims; no assertion that another checkpoint adds a new lineage. | Manuscript scope/limitations; `plan.md`. | Broader testing remains explicitly deferred. |
| R2: Qwen3.5 capacity control | Explicit deferral; not confused with a matched pretraining pair. | Plan and implementation guide. | Follow-up work if pursued. |
| R3: broader families/tasks | Better replication infrastructure and explicit remaining external-validity limit. | Manuscript limitations; plan. | Request is not experimentally satisfied by this implementation or by optional Cosmos alone. |
| R4: broad title | Uses “An Empirical Study of Robot Pretraining in Frozen VLA Backbones.” | Manuscript title. | Confirm with the author before final submission. |
| R4: economic wording | Removed cost-repayment/payoff rationale from the new narrative. | Revised manuscript prose. | Final search after results integration and author edits. |

Validation details and limitations are recorded in `camera_ready_validation.md`.
No new success rate, policy-training SD, or causal intervention is reported.
