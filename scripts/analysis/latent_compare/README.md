# latent_compare — probing utilities

Shared statistics used by the head diagnostics. `scripts/analysis/head_diagnostics.py`
imports from `analyze.py` directly:

| Function | What it does |
|---|---|
| `cv_r2` | Cross-validated R² for a ridge probe — reports generalisation, not fit, so a probe cannot claim information that is only memorised |
| `parallel_analysis` | Horn's parallel analysis: how many principal components exceed what random data of the same shape produces. Used to size a latent space before comparing arms |
| `_pcs` | Top-k principal components of a token matrix |

## Why cross-validated

A probe fit and scored on the same frames reports capacity, not availability. All
probe numbers in `RESULTS.md` are cross-validated, which is what licenses the
phrase "linearly readable" rather than "linearly fittable".

## Scope

These measure **availability** — that information is present and linearly
readable. They do not show a policy uses it. Necessity claims in this project
come only from closed-loop rollouts; see `OBJECTIVE.md` for why the distinction
is enforced.
