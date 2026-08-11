"""
Joint analysis — all three datasets pooled into one space per backbone.

Every earlier figure was per-dataset, which cannot answer "how does this model
organise *different tasks and scenes* relative to each other." This script pools
all 7,136 probe frames and asks two questions:

  Q1 (per model)  — when a backbone sees two different robots, two different
                    scenes and three different tasks at once, what does its
                    latent organise by? Visual appearance, or task?

                    The informative contrast is built into the probe set:
                      ALOHA transfer vs ALOHA insertion = SAME robot, SAME
                        camera, same visual world — DIFFERENT task.
                      ALOHA vs Language Table            = different robot,
                        different camera, different everything.
                    Any encoder trivially separates the second pair. The
                    diagnostic quantity is how large the first separation is
                    *relative* to the second — reported as `task_scene_ratio`.
                    A model that only does appearance scores near 0; a model
                    that represents task structure scores higher.

  Q2 (across models) — do the robot-pretrained backbones converge on a shared
                    way of organising the world that the stock ones don't?
                    Different backbones have different widths and incomparable
                    axes, so their points cannot share a UMAP. Representational
                    similarity analysis solves this: build each model's
                    frame x frame distance matrix (RDM), which lives in a space
                    that IS comparable across models, then correlate the RDMs.
                    That yields a model x model similarity matrix — the models
                    themselves placed in one space.

Primary pool is IMAGE tokens. Text tokens would separate the datasets almost by
definition (the instruction strings differ), which would answer the question by
construction rather than by measurement.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
from scipy import stats
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "asset" / "analysis" / "latent_compare"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbones import ARMS as MODELS  # noqa: E402
from backbones import KEYS  # noqa: E402
N_PC = 50
RSA_PER_DATASET = 500  # stratified subsample per dataset for the RDMs
SEED = 0


def load_joint(model: str, pool: str) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate a model's latents across every dataset in KEYS.

    Reads the `doc` tap — each arm's DOCUMENTED layer — not `d100`.

    d100 is each arm's own last layer, which is precisely the confound this
    study exists to remove: it compares SmolVLA's layer 16 against SmolVLM2's
    layer 32, and GR00T's 16 against Cosmos's 28. The frozen pair exposes it
    numerically. SmolVLA's VLM is 345/345 tensors identical to SmolVLM2's and
    their doc reads agree to cosine 0.999995, so any paired difference must be
    zero — yet under d100 this file reported RSA 0.962 and a task/scene delta of
    -0.0128, both pure read-depth artifact.
    """
    Xs, labels = [], []
    for li, key in enumerate(KEYS):
        p = DIR / f"latents_{model}_{key}.h5"
        with h5py.File(p, "r") as f:
            X = f[f"doc_{pool}"][:]
        Xs.append(X)
        labels.append(np.full(len(X), li, dtype=np.int64))
    return np.concatenate(Xs, 0), np.concatenate(labels, 0)


def _pc_space(X: np.ndarray, k: int = N_PC) -> np.ndarray:
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return PCA(n_components=min(k, min(Xz.shape) - 1), random_state=SEED).fit_transform(Xz)


def separation_metrics(Z: np.ndarray, y: np.ndarray) -> dict:
    """How the model arranges same-scene-different-task vs different-scene."""
    n = len(KEYS)
    cents = np.stack([Z[y == i].mean(0) for i in range(n)])
    d_task = float(np.linalg.norm(cents[0] - cents[1]))              # AT vs AI
    # "different scene" is the mean distance from each ALOHA task to every
    # non-ALOHA dataset. With Language Table alone this was 2 distances; adding
    # LIBERO-Goal makes it 4. Averaging keeps the ratio on the same scale and
    # comparable to the previously reported values, since each additional scene
    # contributes symmetrically.
    others = list(range(2, n))
    d_scene = float(np.mean([
        np.linalg.norm(cents[a] - cents[o]) for a in (0, 1) for o in others
    ]))

    # Same-scene task discrimination, measured properly rather than by centroid
    # distance alone (which ignores within-class spread).
    m = y < 2
    acc = float(np.mean(cross_val_score(
        LogisticRegression(max_iter=2000), Z[m], y[m], cv=5, scoring="accuracy"
    )))

    return {
        "task_scene_ratio": d_task / max(d_scene, 1e-9),
        "d_task_aloha_pair": d_task,
        "d_scene_aloha_vs_other": d_scene,
        "silhouette_3way": float(silhouette_score(Z, y, sample_size=3000, random_state=SEED)),
        "same_scene_task_acc": acc,
    }


def rdm(X: np.ndarray) -> np.ndarray:
    """Condensed cosine-distance matrix over frames."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Xn.T
    np.fill_diagonal(S, 1.0)
    return squareform(np.clip(1.0 - S, 0, 2), checks=False)


def main() -> int:
    rng = np.random.default_rng(SEED)
    out: dict = {"per_model": {}, "rsa": {}}

    # stratified subsample indices shared by every model (identical frames)
    _, y_full = load_joint(MODELS[0], "image")
    sub = np.concatenate([
        rng.choice(np.nonzero(y_full == i)[0], RSA_PER_DATASET, replace=False)
        for i in range(3)
    ])
    sub.sort()
    np.save(DIR / "joint_rsa_index.npy", sub)

    rdms: dict[str, np.ndarray] = {}
    embeds: dict[str, np.ndarray] = {}

    for pool in ("image", "all"):
        for m in MODELS:
            X, y = load_joint(m, pool)
            Z = _pc_space(X)
            met = separation_metrics(Z, y)
            out["per_model"].setdefault(pool, {})[m] = met
            print(f"[{pool:5s}] {m:10s} ratio={met['task_scene_ratio']:.3f} "
                  f"sil={met['silhouette_3way']:.3f} taskacc={met['same_scene_task_acc']:.3f}",
                  flush=True)

            if pool == "image":
                rdms[m] = rdm(X[sub])
                import umap
                embeds[m] = umap.UMAP(
                    n_neighbors=30, min_dist=0.1, metric="cosine",
                    random_state=SEED, n_components=2,
                ).fit_transform(Z).astype(np.float32)

    # ── RSA: correlate every pair of RDMs ────────────────────────────────────
    n = len(MODELS)
    R = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = stats.spearmanr(rdms[MODELS[i]], rdms[MODELS[j]]).statistic
            R[i, j] = R[j, i] = float(r)
    out["rsa"]["matrix"] = R.tolist()
    out["rsa"]["models"] = MODELS

    # place the models themselves in 2D from their RDM correlations
    D = 1.0 - R
    np.fill_diagonal(D, 0.0)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED,
              normalized_stress="auto").fit_transform(D)
    out["rsa"]["mds"] = mds.tolist()

    np.savez_compressed(
        DIR / "joint_embed.npz",
        labels=y_full,
        **{f"umap_{m}": embeds[m] for m in MODELS},
    )
    (DIR / "joint_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\nRSA matrix (Spearman between RDMs):")
    print("            " + "".join(f"{m:>11s}" for m in MODELS))
    for i, m in enumerate(MODELS):
        print(f"{m:>11s} " + "".join(f"{R[i, j]:11.3f}" for j in range(n)))
    print(f"\nwrote {DIR/'joint_metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
