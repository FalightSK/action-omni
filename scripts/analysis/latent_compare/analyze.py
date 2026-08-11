"""
Step 3/3 — quantify and embed the latent geometry of each backbone.

Three questions, three families of measurement:

1. HOW MANY DIMENSIONS DOES THE REPRESENTATION ACTUALLY USE?
   Horn's parallel analysis. We do NOT use a "95% variance" cutoff: with
   n=2400 samples and d=1024-2048 features we sit in the regime where a pure
   noise matrix already produces large leading eigenvalues (Marchenko-Pastur),
   so any fixed-variance rule would mistake sampling noise for structure. PA
   builds the null explicitly — permute each feature column independently,
   which destroys inter-feature correlation while preserving every marginal —
   and retains only components beating the 95th percentile of that null at the
   same rank. Because the null is rebuilt at each arm's own (n, d), the
   retained count stays comparable across arms of different width.
   Participation ratio is reported alongside as a width-insensitive check.

2. WHAT DO THOSE DIMENSIONS ENCODE?
   Variance attributable to each factor the user asked about, measured in the
   retained PC space:
     instruction — eta^2 from one-way ANOVA over instruction id
     time        — ridge R^2 predicting normalised phase, + max |Spearman|
     action      — cross-validated ridge R^2 predicting the action chunk,
                   plus first canonical correlation (CCA)
     state       — cross-validated ridge R^2 predicting proprioceptive state
   Ridge probes are cross-validated because an unregularised in-sample fit from
   a k-dim latent to a 224-dim action chunk would report near-perfect R^2 on
   noise alone.

3. HOW ARE IMAGE AND TEXT ARRANGED RELATIVE TO EACH OTHER?
   Principal angles between the image-token and text-token PC subspaces, plus
   cross-predictability in both directions. Small angles / high cross-R^2 mean
   the arm has fused the two streams into a shared subspace; large angles mean
   it keeps them apart.

Outputs
  asset/analysis/latent_compare/metrics.json
  asset/analysis/latent_compare/embed_<model>_<key>.npz   (PC + UMAP coords)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.utils.extmath import randomized_svd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "asset" / "analysis" / "latent_compare"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbones import ARMS as MODELS  # noqa: E402
from backbones import GATE_KEYS as KEYS  # noqa: E402

POOLS = ["image", "text", "all"]
# Grid depths plus "doc". The per-arm analyses run at every tag so a reader can
# check any claim at any depth; the figures select which one they quote.
DEPTH_TAGS = {0: "d000", 25: "d025", 50: "d050", 75: "d075", 100: "d100",
              "doc": "doc"}

# 40 surrogates is enough here: the permutation null is near-degenerate (its p95
# is flat at ~2.7 with negligible spread across draws), so the percentile
# estimate is stable long before the usual 100+ draws.
N_PERM = 40
N_COMP = 120          # leading spectrum computed (k always lands well below)
# Depth used for probes/UMAP and every headline number.
#
# This was 100 — each arm's own last layer — and that choice is what made the
# study's original image/text finding a depth artifact: SmolVLM2 was probed at
# layer 32 against SmolVLA's 16, and Cosmos at 28 against GR00T's 16. Measured
# on Cosmos, the last layer differs from the layer GR00T actually reads by a
# relative norm of 4.64, so "depth 100" was not a neutral default; it was a
# systematic bias against every arm whose descendant truncates.
#
# "doc" is each arm's documented read layer (backbones.DOC_LAYER), so a
# finetuned arm and its stock control are compared at the SAME absolute layer
# and the only remaining difference is the weights.
PRIMARY_DEPTH = "doc"


# ── 1. dimensionality ────────────────────────────────────────────────────────

def _spectrum(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Leading eigenvalues of the covariance of column-standardised X."""
    _, s, _ = randomized_svd(X, n_components=k, n_iter=7, random_state=seed)
    return (s ** 2) / (X.shape[0] - 1)


def parallel_analysis(X: np.ndarray, n_perm: int = N_PERM, k: int = N_COMP,
                      pct: float = 95.0, seed: int = 0) -> dict:
    """Horn's parallel analysis with column-permutation surrogates."""
    Xz = X - X.mean(0)
    sd = Xz.std(0)
    sd[sd < 1e-8] = 1.0
    Xz = Xz / sd

    k = min(k, min(Xz.shape) - 1)
    obs = _spectrum(Xz, k, seed)

    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, k), dtype=np.float64)
    for i in range(n_perm):
        # permuted(axis=0) shuffles every column independently in one vectorised
        # call — this is the surrogate that kills inter-feature correlation while
        # leaving each feature's marginal distribution untouched.
        null[i] = _spectrum(rng.permuted(Xz, axis=0), k, seed=i)
    thresh = np.percentile(null, pct, axis=0)

    above = obs > thresh
    n_ret = int(np.argmin(above)) if not above.all() else k  # first failure ends it

    # Xz is column-standardised, so the correlation matrix has unit diagonal and
    # its trace is exactly d — no need to form the d x d covariance.
    total = float(Xz.shape[1])
    pr = float((obs.sum() ** 2) / (obs ** 2).sum())
    return {
        "n_retained": n_ret,
        "participation_ratio": pr,
        "var_explained_retained": float(obs[:n_ret].sum() / total) if n_ret else 0.0,
        "eig_obs": obs[:40].tolist(),
        "eig_null_p95": thresh[:40].tolist(),
        "ambient_dim": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
    }


# ── 2. factor structure ──────────────────────────────────────────────────────

def _pcs(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    Xz = X - X.mean(0)
    sd = Xz.std(0)
    sd[sd < 1e-8] = 1.0
    Xz /= sd
    U, s, _ = randomized_svd(Xz, n_components=max(k, 2), n_iter=7, random_state=seed)
    return U * s


def eta_squared(Z: np.ndarray, labels: np.ndarray) -> float:
    """Share of total PC variance explained by group identity (one-way ANOVA)."""
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan")
    gm = Z.mean(0)
    ss_b = sum(len(g := np.nonzero(labels == u)[0]) * ((Z[g].mean(0) - gm) ** 2).sum() for u in uniq)
    ss_t = ((Z - gm) ** 2).sum()
    return float(ss_b / ss_t) if ss_t > 0 else float("nan")


def cv_r2(Z: np.ndarray, Y: np.ndarray, seed: int = 0) -> float:
    """5-fold cross-validated ridge R^2, averaged over targets."""
    if Y.ndim == 1:
        Y = Y[:, None]
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    Ys = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)
    cv = KFold(5, shuffle=True, random_state=seed)
    # cross_val_predict ravels a single-column target back to (n,); without this
    # reshape the (n,1) - (n,) subtraction broadcasts to an (n,n) matrix and the
    # resulting "R^2" is silently garbage (it comes out near -1).
    pred = np.asarray(cross_val_predict(Ridge(alpha=1.0), Zs, Ys, cv=cv)).reshape(Ys.shape)
    ss_res = ((Ys - pred) ** 2).sum(0)
    ss_tot = ((Ys - Ys.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - ss_res / np.maximum(ss_tot, 1e-12)))


def first_canon_corr(Z: np.ndarray, Y: np.ndarray, seed: int = 0) -> float:
    n_c = int(min(8, Z.shape[1], Y.shape[1]))
    if n_c < 1:
        return float("nan")
    try:
        cca = CCA(n_components=n_c, max_iter=1000)
        a, b = cca.fit_transform(Z, Y)
        return float(np.corrcoef(a[:, 0], b[:, 0])[0, 1])
    except Exception:
        return float("nan")


def temporal_smoothness(X: np.ndarray, episode: np.ndarray, frame: np.ndarray) -> float:
    """Mean cosine between latents of probe frames adjacent within an episode."""
    order = np.lexsort((frame, episode))
    Xo, eo = X[order], episode[order]
    a, b = Xo[:-1], Xo[1:]
    same = eo[:-1] == eo[1:]
    if same.sum() < 2:
        return float("nan")
    a, b = a[same], b[same]
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return float(np.mean(num / np.maximum(den, 1e-12)))


def principal_angles(A: np.ndarray, B: np.ndarray, k: int) -> dict:
    """Principal angles between the top-k PC subspaces of A and B."""
    def basis(M):
        Mz = (M - M.mean(0)) / (M.std(0) + 1e-8)
        _, _, Vt = randomized_svd(Mz, n_components=k, n_iter=7, random_state=0)
        return Vt  # (k, d) orthonormal rows
    # different arms give image/text the same width, so the subspaces are comparable
    Va, Vb = basis(A), basis(B)
    if Va.shape[1] != Vb.shape[1]:
        return {"mean_cos": float("nan"), "min_angle_deg": float("nan")}
    s = np.linalg.svd(Va @ Vb.T, compute_uv=False)
    s = np.clip(s, 0, 1)
    return {
        "mean_cos": float(s.mean()),
        "min_angle_deg": float(np.degrees(np.arccos(s[0]))),
        "mean_angle_deg": float(np.degrees(np.arccos(s)).mean()),
    }


# ── driver ───────────────────────────────────────────────────────────────────

def load_probe(key: str) -> dict:
    with h5py.File(DIR / f"probe_{key}.h5", "r") as f:
        return {
            "actions": f["actions"][:].reshape(len(f["actions"]), -1),
            "states": f["states"][:],
            "phase": f["phase"][:],
            "episode": f["episode"][:],
            "frame": f["frame_in_ep"][:],
            "instr_id": f["instr_id"][:],
            "n_instr": int(f.attrs["n_instructions"]),
        }


def analyse(model: str, key: str, probe: dict, do_umap: bool, seed: int) -> dict:
    path = DIR / f"latents_{model}_{key}.h5"
    if not path.exists():
        return {}
    out: dict = {}
    with h5py.File(path, "r") as f:
        pools = {p: f[f"{DEPTH_TAGS[PRIMARY_DEPTH]}_{p}"][:] for p in POOLS}
        # dimensionality at every depth, for the depth-trend table
        dims: dict = {}
        for dpct, tag in DEPTH_TAGS.items():
            for p in POOLS:
                ds = f"{tag}_{p}"
                if ds in f:
                    dims[f"{tag}_{p}"] = parallel_analysis(f[ds][:], seed=seed)
        out["dimensionality"] = dims

    k_all = max(2, dims[f"{DEPTH_TAGS[PRIMARY_DEPTH]}_all"]["n_retained"])
    k_use = min(k_all, 64)  # probes use the retained space, capped for stability

    factors: dict = {}
    for p in POOLS:
        X = pools[p]
        Z = _pcs(X, k_use, seed)
        factors[p] = {
            "eta2_instruction": eta_squared(Z, probe["instr_id"]),
            "eta2_episode": eta_squared(Z, probe["episode"]),
            "r2_phase": cv_r2(Z, probe["phase"], seed),
            "max_abs_spearman_phase": float(
                max(abs(stats.spearmanr(Z[:, i], probe["phase"]).statistic)
                    for i in range(min(Z.shape[1], 20)))
            ),
            "r2_action": cv_r2(Z, probe["actions"], seed),
            "r2_state": cv_r2(Z, probe["states"], seed),
            "canon_corr_action": first_canon_corr(Z, probe["actions"], seed),
            "temporal_smoothness": temporal_smoothness(X, probe["episode"], probe["frame"]),
            "k_used": int(k_use),
        }
    out["factors"] = factors

    out["image_text_geometry"] = {
        **principal_angles(pools["image"], pools["text"], min(k_use, 32)),
        "r2_image_from_text": cv_r2(_pcs(pools["text"], k_use, seed),
                                    _pcs(pools["image"], k_use, seed), seed),
        "r2_text_from_image": cv_r2(_pcs(pools["image"], k_use, seed),
                                    _pcs(pools["text"], k_use, seed), seed),
        "cos_image_text_centroid": float(
            np.dot(pools["image"].mean(0), pools["text"].mean(0))
            / (np.linalg.norm(pools["image"].mean(0)) * np.linalg.norm(pools["text"].mean(0)) + 1e-12)
        ),
    }

    if do_umap:
        import umap

        emb = {}
        for p in POOLS:
            Z = _pcs(pools[p], min(k_use, 50), seed)
            red = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                            random_state=seed, n_components=2).fit_transform(Z)
            emb[f"umap_{p}"] = red.astype(np.float32)
            emb[f"pc_{p}"] = Z[:, :10].astype(np.float32)
        np.savez_compressed(DIR / f"embed_{model}_{key}.npz", **emb)

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--keys", nargs="+", default=KEYS)
    ap.add_argument("--no-umap", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    # The 18 (model, dataset) cells are fully independent — nothing is shared
    # across them and each writes its own embed_*.npz — so they parallelise by
    # simply running several processes. The only shared resource is the single
    # metrics.json, so a worker writes a shard instead and a merge step folds
    # the shards together. --out keeps the default single-process behaviour
    # byte-identical to before.
    ap.add_argument("--out", default="metrics.json",
                    help="output filename within the analysis dir")
    a = ap.parse_args()

    results: dict = {}
    for key in a.keys:
        probe = load_probe(key)
        for m in a.models:
            print(f"[{m} / {key}] analysing …", flush=True)
            r = analyse(m, key, probe, not a.no_umap, a.seed)
            if r:
                results.setdefault(key, {})[m] = r
                d = r["dimensionality"][f"{DEPTH_TAGS[PRIMARY_DEPTH]}_all"]
                fa = r["factors"]["all"]
                print(f"    k={d['n_retained']:3d} PR={d['participation_ratio']:6.1f} "
                      f"| eta2_instr={fa['eta2_instruction']:.3f} "
                      f"r2_act={fa['r2_action']:.3f} r2_phase={fa['r2_phase']:.3f}")

    (DIR / a.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {DIR / a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
