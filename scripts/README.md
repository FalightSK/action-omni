# scripts/ layout

Core entrypoints live at `scripts/` root; one-off helpers are grouped by purpose.
Every subfolder script finds the repo root via `Path(__file__).parents[2]`
(root-level scripts use `parents[1]`).

## Root — core pipeline (referenced by launchers & docs; do not move)
| file | step |
|---|---|
| `precompute.py` | 1 · cache VLM embeddings (`--dataset --exp`) |
| `train.py` | 2 · train adapter + flow decoder |
| `evaluate.py` / `offline_eval.py` | 3 · rollout / offline eval |
| `convert_language_table.py` | build LT dataset from TFDS |
| `analysis.py`, `compare.py`, `export_bank.py` | shared analysis utilities |
| `train_lt_onthefly.py` | on-the-fly (no-cache) training variant |
| `launch_lt.ps1`, `run_lt_pipeline.ps1`, `migrate_assets.sh` | launchers |

## Subfolders
- **`data/`** — dataset build + embedding-cache tooling (`_lt_make_curated_subset`, `_pt_to_h5_cache`, `_h5_*`, downloaders).
- **`validation/`** — milestone validity gates (`_val_gate1_aloha`, `_val_gate1_lt`).
- **`eval/`** — policy rollout / success-rate eval (`eval_lt_*`, `eval_ood_*`, `eval_token_ablation`, `_lt_eval_indist`).
- **`analysis/`** — mechanistic / attention / PCA / token probes.
- **`plots/`** — figure generators + PDF report builders.
- **`media/`** — rollout videos / filmstrips.
- **`archive/`** — superseded: smolvla-era diagnostics, 0%-SR failure analysis, VRAM benches. Kept for reference, not part of the active pipeline.

Backward-compat shims (`train.py`, `precompute_embeddings.py`, `inference.py`,
`config_loader.py`) remain at the **repo root** — they are imported by
`data/`, `models/`, `envs/` and must not be relocated.
