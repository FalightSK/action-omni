"""Task geometry — how far apart does each backbone place the datasets?

Three datasets gave three pairwise distances, and three distances define a
triangle exactly: the 2-D drawing was the measurement, with zero projection
error. Four datasets give six distances, and four points generally need three
dimensions. The exactness does not survive being flattened to a page.

Rather than quietly projecting and pretending, this reports three views with
their error stated:

  fig13  the exact triangle over the three ORIGINAL datasets, unchanged. Still
         zero-distortion, still directly comparable to earlier versions.

  fig18  all four datasets. Points are placed by classical MDS on the 4x4
         distance matrix, which for four points is exact in 3-D; the panel shows
         the 2-D projection that preserves the most variance, and prints both
         the residual (how much geometry the flattening cost) and every true
         edge length. Read the numbers, not the picture, wherever they disagree
         — that is why they are drawn on the edges.

  fig19  the 4x4 normalised distance matrices themselves, one heatmap per
         backbone. No projection at all, so nothing can be distorted. This is
         the authoritative view; the embeddings above are navigational aids.

Every backbone is normalised by its own mean pairwise distance, so shape is
comparable across arms of different width.

Measured on image tokens (text tokens separate the datasets by construction).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_joint import KEYS, MODELS, _pc_space, load_joint  # noqa: E402
from plots import (COLORS, GRID, INK, INK2, LABELS, SHORT, SURF, _style,  # noqa: E402
                   resolve_label_overlaps)

ROOT = HERE.parents[2]
DIR = ROOT / "asset" / "analysis" / "latent_compare"
FIG = DIR / "figures"

# Derived from KEYS, not hard-coded. These lists were written for four datasets
# and silently disagreed with KEYS the moment Language Table was dropped —
# matplotlib caught it as "3 locations vs 4 labels", but a figure indexed by
# position rather than by name could just as easily have mislabelled an axis.
_LABEL_OF = {
    "aloha_transfer": ("ALOHA\ntransfer-cube", "AT", "#2a78d6"),
    "aloha_insertion": ("ALOHA\ninsertion", "AI", "#eb6834"),
    "language_table": ("Language Table\n(curated)", "LT", "#1baf7a"),
    "libero_goal": ("LIBERO-Goal", "LG", "#8a63d2"),
}
TASK_LABEL = [_LABEL_OF[k][0] for k in KEYS]
TASK_SHORT = [_LABEL_OF[k][1] for k in KEYS]
TASK_COLOR = [_LABEL_OF[k][2] for k in KEYS]
POOL = "image"

# With Language Table gone there are exactly three datasets, and three pairwise
# distances determine a triangle EXACTLY by the law of cosines — so the drawing
# is the measurement again, with zero projection error. The tetrahedron figure
# (fig18) and its "2-D keeps 77-96%" caveat existed only because a fourth
# dataset forced a 3-D arrangement into a plane; both are now unnecessary.
TRI_IDX = list(range(len(KEYS)))


def triangle_from_distances(d01: float, d02: float, d12: float) -> np.ndarray:
    """Exact 2-D placement of three points given their pairwise distances."""
    p0 = np.array([0.0, 0.0])
    p1 = np.array([d01, 0.0])
    x = (d02 ** 2 + d01 ** 2 - d12 ** 2) / (2 * d01) if d01 > 1e-12 else 0.0
    y = float(np.sqrt(max(0.0, d02 ** 2 - x ** 2)))
    return np.stack([p0, p1, np.array([x, y])])


def classical_mds(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates and eigenvalues from a distance matrix (Torgerson scaling).

    For n points the embedding is exact in at most n-1 dimensions. The
    eigenvalues say how many are really needed: a near-zero third eigenvalue
    would mean the four datasets happen to lie in a plane and the 2-D drawing
    loses nothing.
    """
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    wp = np.clip(w, 0, None)
    return V * np.sqrt(wp), w


def compute() -> dict:
    out: dict = {}
    n = len(KEYS)
    for m in MODELS:
        X, y = load_joint(m, POOL)
        Z = _pc_space(X)
        cents = np.stack([Z[y == i].mean(0) for i in range(n)])
        spread = np.array([
            float(np.linalg.norm(Z[y == i] - cents[i], axis=1).mean()) for i in range(n)
        ])
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = float(np.linalg.norm(cents[i] - cents[j]))
        iu = np.triu_indices(n, 1)
        scale = float(D[iu].mean())
        Dn = D / scale

        coords, eig = classical_mds(Dn)
        # how much of the geometry a 2-D drawing keeps
        pos = np.clip(eig, 0, None)
        var2 = float(pos[:2].sum() / max(pos.sum(), 1e-12))
        emb2 = coords[:, :2]
        D2 = np.linalg.norm(emb2[:, None, :] - emb2[None, :, :], axis=-1)
        resid = float(np.abs(D2[iu] - Dn[iu]).mean())

        sub = Dn[np.ix_(TRI_IDX, TRI_IDX)]
        out[m] = {
            "keys": list(KEYS),
            "dist_norm": Dn.tolist(),
            "spread_norm": (spread / scale).tolist(),
            "raw_scale": scale,
            "mds2": emb2.tolist(),
            "mds_eigenvalues": eig.tolist(),
            "var_kept_2d": var2,
            "mean_abs_edge_error_2d": resid,
            # the original three-dataset triangle, still exact
            "tri": triangle_from_distances(sub[0, 1], sub[0, 2], sub[1, 2]).tolist(),
        }
        print(f"{m:10s} " + "  ".join(
            f"{TASK_SHORT[i]}-{TASK_SHORT[j]}={Dn[i, j]:.2f}" for i, j in zip(*iu))
            + f"  | 2-D keeps {var2:.0%}, mean edge err {resid:.3f}", flush=True)
    return out


# ── Fig 13: the exact triangle over the original three datasets ──────────────

def fig_triangles(G: dict):
    n = len(MODELS) + 1
    ncol = 4 if n > 6 else 3
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 4.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)

    for ax, m in zip(axes[:len(MODELS)], MODELS):
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        tri = np.array(G[m]["tri"]) - np.array(G[m]["tri"]).mean(0)
        Dn = np.array(G[m]["dist_norm"])[np.ix_(TRI_IDX, TRI_IDX)]
        spread = np.array(G[m]["spread_norm"])[TRI_IDX]

        closed = np.vstack([tri, tri[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=INK2, lw=1.2, zorder=1, alpha=0.5)
        for i in range(3):
            ax.add_patch(plt.Circle(tri[i], spread[i], color=TASK_COLOR[TRI_IDX[i]],
                                    alpha=0.16, lw=0, zorder=0))
            ax.scatter(*tri[i], s=95, color=TASK_COLOR[TRI_IDX[i]], edgecolor=SURF,
                       linewidth=1.8, zorder=3)
            off = tri[i] - tri.mean(0)
            off = off / (np.linalg.norm(off) + 1e-9)
            ax.annotate(TASK_LABEL[TRI_IDX[i]], tri[i], textcoords="offset points",
                        xytext=(off[0] * 34, off[1] * 30), ha="center", va="center",
                        fontsize=7.2, color=INK)
        for (a, b) in [(0, 1), (0, 2), (1, 2)]:
            mid = (tri[a] + tri[b]) / 2
            push = mid - tri.mean(0)
            push = push / (np.linalg.norm(push) + 1e-9)
            ax.annotate(f"{Dn[a, b]:.2f}", mid, textcoords="offset points",
                        xytext=(push[0] * 12, push[1] * 12), ha="center", va="center",
                        fontsize=8.2, color=INK2, fontweight="bold")
        ratio = Dn[0, 1] / ((Dn[0, 2] + Dn[1, 2]) / 2)
        ax.set_title(f"{LABELS[m]}\ntask/scene ratio {ratio:.2f}", fontsize=8.6)
        lim = np.abs(tri).max() + spread.max() + 0.42
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    ax = axes[len(MODELS)]
    ax.set_aspect("equal"); _style(ax)
    ax.set_xticks([]); ax.set_yticks([])
    allpts = []
    for m in MODELS:
        tri = np.array(G[m]["tri"]); tri = tri - tri.mean(0)
        allpts.append(tri)
        closed = np.vstack([tri, tri[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=COLORS[m], lw=2.0, alpha=0.9,
                label=SHORT[m].replace("\n", " "))
        ax.scatter(tri[:, 0], tri[:, 1], s=34, color=COLORS[m], zorder=3,
                   edgecolor=SURF, linewidth=1.1)
    P = np.concatenate(allpts)
    lim = np.abs(P).max() * 1.14
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim * 1.55)
    ax.legend(frameon=False, fontsize=7.4, labelcolor=INK, loc="upper center",
              ncol=2, handlelength=1.4, columnspacing=1.0)
    ax.set_title(f"All {len(MODELS)} overlaid\n(same normalisation and orientation)",
                 fontsize=8.6)

    fig.suptitle(
        "How each backbone places the three original tasks relative to each other\n"
        "exact construction — three distances determine a triangle, so there is no projection error",
        fontsize=11.5,
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig13_task_triangles.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


# ── Fig 18: all four datasets ────────────────────────────────────────────────

def fig_tetra(G: dict):
    n = len(MODELS)
    ncol = 4 if n > 6 else 3
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 4.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)

    nk = len(KEYS)
    iu = np.triu_indices(nk, 1)
    for ax, m in zip(axes, MODELS):
        ax.set_aspect("equal"); ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        P = np.array(G[m]["mds2"])
        P = P - P.mean(0)
        Dn = np.array(G[m]["dist_norm"])
        spread = np.array(G[m]["spread_norm"])

        edge_anns = []
        for i, j in zip(*iu):
            ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color=INK2, lw=1.0,
                    alpha=0.45, zorder=1)
            mid = (P[i] + P[j]) / 2
            edge_anns.append(ax.annotate(
                f"{Dn[i, j]:.2f}", mid, ha="center", va="center", fontsize=6.6,
                color=INK2, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc=SURF, ec="none", alpha=0.85)))
        # Label placement: a purely radial offset overprints AT and AI, which sit
        # almost on top of each other for most arms (their centroid distance is
        # ~0.4 against ~1.1 for every other pair). Push apart any labels that
        # would collide, then place radially from the adjusted positions.
        node_anns = []
        lab = P + 0.0
        for _ in range(60):
            moved = False
            for i in range(nk):
                for j in range(i + 1, nk):
                    d = lab[i] - lab[j]
                    r = float(np.linalg.norm(d))
                    if r < 0.42:
                        v = d / (r + 1e-9) * (0.42 - r) / 2
                        lab[i] += v; lab[j] -= v
                        moved = True
            if not moved:
                break
        for i in range(nk):
            ax.add_patch(plt.Circle(P[i], spread[i], color=TASK_COLOR[i], alpha=0.14,
                                    lw=0, zorder=0))
            ax.scatter(*P[i], s=85, color=TASK_COLOR[i], edgecolor=SURF, linewidth=1.7,
                       zorder=3)
            off = lab[i] - P.mean(0)
            off = off / (np.linalg.norm(off) + 1e-9)
            tip = P[i] + off * 0.30
            ax.plot([P[i, 0], tip[0]], [P[i, 1], tip[1]], color=TASK_COLOR[i],
                    lw=0.8, alpha=0.6, zorder=2)
            # xytext in DATA coords (not offset points) so the overlap resolver
            # below can move these too — otherwise a node label and an edge label
            # can still collide, which is how PaliGemma rendered "0.6AT".
            node_anns.append(ax.annotate(
                TASK_SHORT[i], xy=tuple(tip), xytext=(tip[0] + off[0] * 0.10,
                                                     tip[1] + off[1] * 0.09),
                ha="center", va="center", fontsize=8, color=INK, fontweight="bold"))
        ax.set_title(f"{LABELS[m]}\n2-D keeps {G[m]['var_kept_2d']:.0%} · "
                     f"edge err {G[m]['mean_abs_edge_error_2d']:.2f}", fontsize=8.2)
        lim = np.abs(P).max() + spread.max() + 0.62
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        # AT and AI nearly coincide for most arms, so the AT-LT and AI-LT edges
        # nearly coincide too and their midpoint labels land on top of each
        # other. Separate them by measured text extent; expand=False because the
        # limits above are already generous and re-expanding them here would
        # rescale the panel and undo the separation.
        # avoid=P also lifts the AT-AI label off the two markers it sits between:
        # those centroids nearly coincide, so their edge midpoint is underneath them.
        resolve_label_overlaps(fig, ax, edge_anns + node_anns, pad_px=1.5,
                               expand=False, avoid=P, avoid_r_px=9.0)

    handles = [plt.Line2D([], [], marker="o", ls="", color=TASK_COLOR[i],
                          label=f"{TASK_SHORT[i]} — {TASK_LABEL[i]}".replace("\n", " "))
               for i in range(nk)]
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=nk, labelcolor=INK,
               loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "All four datasets — four points need three dimensions, so this is a projection\n"
        "edge labels are the TRUE normalised distances; each panel states what the flattening cost",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "fig18_task_tetrahedron.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


# ── Fig 19: the distance matrices themselves — no projection ─────────────────

def fig_matrix(G: dict):
    n = len(MODELS)
    ncol = 4 if n > 6 else 3
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.1 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)

    vmax = max(np.array(G[m]["dist_norm"]).max() for m in MODELS)
    im = None
    for ax, m in zip(axes, MODELS):
        ax.grid(False)
        Dn = np.array(G[m]["dist_norm"])
        im = ax.imshow(Dn, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(KEYS))); ax.set_yticks(range(len(KEYS)))
        ax.set_xticklabels(TASK_SHORT, fontsize=7.5)
        ax.set_yticklabels(TASK_SHORT, fontsize=7.5)
        for i in range(len(KEYS)):
            for j in range(len(KEYS)):
                ax.text(j, i, f"{Dn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color=SURF if Dn[i, j] > vmax * 0.62 else INK)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(LABELS[m], fontsize=8.2)

    if im is not None:
        cb = fig.colorbar(im, ax=axes[:n].tolist(), fraction=0.02, pad=0.02)
        cb.set_label("normalised centroid distance", fontsize=8)
        cb.outline.set_edgecolor(GRID)
    fig.suptitle(
        "The measurement itself — normalised distance between dataset centroids\n"
        "no embedding, no projection: nothing here can be distorted",
        fontsize=11.5,
    )
    fig.savefig(FIG / "fig19_task_distance_matrix.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    FIG.mkdir(parents=True, exist_ok=True)
    G = compute()
    (DIR / "task_geometry.json").write_text(json.dumps(G, indent=2), encoding="utf-8")
    fig_triangles(G)
    fig_tetra(G)
    fig_matrix(G)
    print("\nwrote fig13, fig18, fig19")
    return 0


if __name__ == "__main__":
    sys.exit(main())
