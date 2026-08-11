"""
Extra figures for the written report — these make two things concrete that the
first figure set only reports as numbers:

  fig7 — an actual PCA scatter (PC1 vs PC2), not the UMAP projection. UMAP
         distorts distances to make clusters legible; a raw PC1-vs-PC2 plot
         shows what the top two directions of real variance look like
         unembellished, at the cost of being messier to look at.

  fig8 / fig9 — "how is R^2 retrieved" made visual: for a scalar target (phase;
         and the action chunk's own first PC, to reduce the multi-dim action to
         one axis) we plot the cross-validated ridge PREDICTION against the
         TRUE value. A diagonal cloud is high R^2 (phase, ALOHA); a flat cloud
         at zero is R^2 near 0 (phase, Language Table). This is the same
         cv_r2() computation in analyze.py, just kept in scatter form instead of
         collapsed to one number.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import h5py
import matplotlib as mpl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

warnings.filterwarnings("ignore")
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plots import (  # noqa: E402
    COLORS, GRID, INK, INK2, KEY_LABEL, KEYS, LABELS, MODELS, PHASE_CMAP,
    SURF, _style, available, load_metrics,
)

ROOT = HERE.parents[2]
DIR = ROOT / "asset" / "analysis" / "latent_compare"
FIG = DIR / "figures"


def load_probe(key: str) -> dict:
    with h5py.File(DIR / f"probe_{key}.h5", "r") as f:
        return {
            "actions": f["actions"][:].reshape(len(f["actions"]), -1),
            "phase": f["phase"][:],
            "episode": f["episode"][:],
        }


# ── Fig 7: raw PC1 vs PC2 scatter, coloured by phase ─────────────────────────

def fig_pca_scatter(M: dict):
    keys = [k for k in KEYS if k in M]
    mdls = available(M)
    fig, axes = plt.subplots(len(mdls), len(keys),
                             figsize=(3.3 * len(keys), 3.0 * len(mdls)), squeeze=False)
    sc = None
    for r, mdl in enumerate(mdls):
        for c, key in enumerate(keys):
            ax = axes[r][c]
            ax.set_axisbelow(True)
            for sp in ax.spines.values():
                sp.set_color(GRID)
            p = DIR / f"embed_{mdl}_{key}.npz"
            probe = load_probe(key)
            if not p.exists():
                ax.axis("off")
                continue
            pc = np.load(p)["pc_all"]
            var1, var2 = pc[:, 0].var(), pc[:, 1].var()
            sc = ax.scatter(pc[:, 0], pc[:, 1], c=probe["phase"], cmap=PHASE_CMAP,
                            s=3.0, linewidths=0, alpha=0.85)
            ax.set_xlabel(f"PC1 (var {var1:.0f})", fontsize=6.5)
            if c == 0:
                ax.set_ylabel(f"{LABELS[mdl]}\nPC2", fontsize=6.8)
            else:
                ax.set_ylabel(f"PC2 (var {var2:.0f})", fontsize=6.5)
            if r == 0:
                ax.set_title(KEY_LABEL[key], fontsize=10)
            ax.tick_params(labelsize=5.5, length=0)
            ax.grid(color=GRID, linewidth=0.5, alpha=0.6)
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes, fraction=0.014, pad=0.01)
        cb.set_label("phase within episode (0=start, 1=end)", fontsize=8)
        cb.outline.set_edgecolor(GRID)
    fig.suptitle(
        "Raw PCA (top 2 components, all tokens) — the actual axes UMAP re-projects",
        fontsize=11,
    )
    fig.savefig(FIG / "fig7_pca_scatter_raw.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


# ── Fig 8/9: predicted-vs-actual scatter — makes "R^2" a picture ─────────────

def _cv_predict_scalar(Z: np.ndarray, y: np.ndarray, seed: int = 0):
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    ys = (y - y.mean()) / (y.std() + 1e-8)
    cv = KFold(5, shuffle=True, random_state=seed)
    pred = cross_val_predict(Ridge(alpha=1.0), Zs, ys, cv=cv)
    r2 = 1 - ((ys - pred) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
    return ys, pred, float(r2)


def _r2_scatter_grid(M: dict, target_fn, target_name: str, ylabel: str, out_name: str):
    keys = [k for k in KEYS if k in M]
    mdls = available(M)
    fig, axes = plt.subplots(len(mdls), len(keys),
                             figsize=(3.1 * len(keys), 2.9 * len(mdls)), squeeze=False)
    for r, mdl in enumerate(mdls):
        for c, key in enumerate(keys):
            ax = axes[r][c]
            _style(ax)
            path = DIR / f"latents_{mdl}_{key}.h5"
            if not path.exists():
                ax.axis("off")
                continue
            probe = load_probe(key)
            with h5py.File(path, "r") as f:
                X = f["doc_all"][:]
            fa = M[key].get(mdl, {}).get("factors", {}).get("all", {})
            k = int(fa.get("k_used", 40))
            Z = PCA(n_components=k, random_state=0).fit_transform(
                (X - X.mean(0)) / (X.std(0) + 1e-8)
            )
            y = target_fn(probe)
            ys, pred, r2 = _cv_predict_scalar(Z, y)
            ax.scatter(ys, pred, s=3.5, color=COLORS[mdl], alpha=0.5, linewidths=0)
            lim = max(abs(ys).max(), abs(pred).max(), 1.0) * 1.05
            ax.plot([-lim, lim], [-lim, lim], color=GRID, lw=1, ls="--", zorder=0)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.text(0.05, 0.92, f"R²={r2:.2f}", transform=ax.transAxes,
                    fontsize=8, color=INK, fontweight="bold", va="top")
            if r == 0:
                ax.set_title(KEY_LABEL[key], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{LABELS[mdl]}\npredicted (z-scored)", fontsize=6.8)
            if r == len(mdls) - 1:
                ax.set_xlabel(f"actual {target_name} (z-scored)", fontsize=7)
            ax.tick_params(labelsize=5.5)
    fig.suptitle(
        f"{ylabel}\n(5-fold cross-validated ridge prediction vs. ground truth; "
        "dashed line = perfect prediction)",
        fontsize=10.5,
    )
    # tight_layout() alone leaves the two-line suptitle overlapping the top row's
    # column titles once there are four datasets — the grid grew wider without the
    # header band growing. Reserve the top strip explicitly.
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(FIG / out_name, bbox_inches="tight", dpi=130)
    plt.close(fig)


def fig_r2_phase(M: dict):
    _r2_scatter_grid(
        M, lambda p: p["phase"], "phase",
        "How the \"phase R²\" number is produced — predicted vs. actual episode phase",
        "fig8_r2_explain_phase.png",
    )


def fig_r2_action(M: dict):
    def action_pc1(probe):
        A = probe["actions"]
        Ac = A - A.mean(0)
        # first principal component of the action chunk itself, so a
        # multi-dimensional target (14-D joints / 2-D pushes) becomes one
        # comparable scalar the same way phase already is.
        _, _, Vt = np.linalg.svd(Ac, full_matrices=False)
        return Ac @ Vt[0]

    _r2_scatter_grid(
        M, action_pc1, "action (1st PC of the action chunk)",
        "How the \"action R²\" number is produced — predicted vs. actual "
        "leading action component",
        "fig9_r2_explain_action.png",
    )


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    M = load_metrics()
    fig_pca_scatter(M)
    fig_r2_phase(M)
    fig_r2_action(M)
    print("wrote fig7, fig8, fig9")
    return 0


if __name__ == "__main__":
    sys.exit(main())
