"""
Figures for the cross-backbone latent study.

Design notes (why the charts look the way they do):
  * Model identity is carried by FACET POSITION in the UMAP grids, not by hue.
    Five categorical hues cannot stay colorblind-separable in a scatter (every
    pair is adjacent there), so hue inside a scatter is reserved for one ordered
    variable — phase — on a single-hue light-to-dark ramp.
  * Model hues appear only on bar/line forms, where the validated adjacent-pair
    ordering holds. Bars carry direct value labels: three of the five slots sit
    below 3:1 against the surface, which obliges visible labels.
  * No dual axes anywhere; magnitudes that differ in scale get their own panel.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import h5py
import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "asset" / "analysis" / "latent_compare"
FIG = DIR / "figures"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbones import ARMS as MODELS  # noqa: E402
from backbones import KEYS as _KEYS  # noqa: E402
from backbones import PAIRS  # noqa: E402

LABELS = {
    "qwen": "Qwen3.5-0.8B\n(frozen, ours)",
    "pi05": "Pi-0.5\n(robot-finetuned)",
    "paligemma": "PaliGemma-3B\n(stock control)",
    "smolvla": "SmolVLA\n(VLM frozen)",
    "smolvlm2": "SmolVLM2-500M\n(stock control)",
    "groot": "GR00T N1.7-3B\n(robot-finetuned)",
    # Cosmos is BOTH: a stock control for GR00T and a finetune of Qwen3-VL. The
    # label says so, because calling it "stock" hides the middle link of the
    # qwen3vl -> cosmos -> groot chain.
    "cosmos": "Cosmos-Reason2-2B\n(physical-AI FT)",
    "qwen3vl": "Qwen3-VL-2B\n(stock control)",
}
# Short forms for axis ticks — the full labels collide at five-per-panel.
SHORT = {
    "qwen": "Qwen-0.8B\n(frozen)",
    "pi05": "Pi-0.5\n(robot-FT)",
    "paligemma": "PaliGemma\n(stock)",
    "smolvla": "SmolVLA\n(VLM frozen)",
    "smolvlm2": "SmolVLM2\n(stock)",
    "groot": "GR00T N1.7\n(robot-FT)",
    "cosmos": "Cosmos-R2\n(phys-AI FT)",
    "qwen3vl": "Qwen3-VL-2B\n(stock)",
}
COLORS = {
    "qwen": "#2a78d6", "pi05": "#eb6834", "paligemma": "#1baf7a",
    "smolvla": "#eda100", "smolvlm2": "#e87ba4", "groot": "#8a63d2",
    # 8 categories is well past the point where hue alone is safely separable,
    # which is why every panel also labels its arms directly. The GR00T chain
    # (qwen3vl -> cosmos -> groot) is kept in one warm-to-cool family so the
    # three links read as related; brown and teal both sit far from the orange
    # and purple already in use.
    "cosmos": "#8a6244", "qwen3vl": "#0f9b8e",
}
# PAIRS and KEYS come from backbones so the roster cannot drift per file.
KEYS = _KEYS
KEY_LABEL = {
    "aloha_transfer": "ALOHA transfer-cube",
    "aloha_insertion": "ALOHA insertion",
    "language_table": "Language Table",
    "libero_goal": "LIBERO-Goal",
}
POOLS = ["image", "text", "all"]

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d8d8d4"
SURF = "#fcfcfb"
# sequential, single hue, light -> dark (never a rainbow for an ordered variable)
PHASE_CMAP = LinearSegmentedColormap.from_list(
    "phase", ["#cfe0f5", "#2a78d6", "#123a6b"]
)

mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
})


def _style(ax):
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    return ax


def load_metrics() -> dict:
    return json.loads((DIR / "metrics.json").read_text(encoding="utf-8"))


def available(m: dict) -> list[str]:
    return [k for k in MODELS if any(k in m.get(d, {}) for d in KEYS)]


# ── Fig 1: scree with the parallel-analysis null ─────────────────────────────

def fig_scree(M: dict):
    keys = [k for k in KEYS if k in M]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 3.9), squeeze=False)
    for ax, key in zip(axes[0], keys):
        _style(ax)
        for mdl in available(M):
            d = M[key].get(mdl, {}).get("dimensionality", {}).get("doc_all")
            if not d:
                continue
            obs = np.array(d["eig_obs"])
            nul = np.array(d["eig_null_p95"])
            x = np.arange(1, len(obs) + 1)
            ax.plot(x, obs, color=COLORS[mdl], lw=2, label=LABELS[mdl].replace("\n", " "))
            ax.plot(x, nul, color=COLORS[mdl], lw=1, ls=":", alpha=0.55)
            k = d["n_retained"]
            if 0 < k <= len(obs):
                ax.plot([k], [obs[k - 1]], "o", ms=7, color=COLORS[mdl],
                        mec=SURF, mew=1.6, zorder=5)
        ax.set_yscale("log")
        ax.set_xlabel("component rank")
        ax.set_title(KEY_LABEL[key], fontsize=10)
        if ax is axes[0][0]:
            ax.set_ylabel("eigenvalue (log)")
    axes[0][0].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.suptitle(
        "Eigenvalue spectrum vs. permutation null (dotted = 95th pct of null; dot = last retained)",
        fontsize=10.5, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig1_scree_parallel_analysis.png", bbox_inches="tight")
    plt.close(fig)


# ── Fig 2: retained dimensionality ───────────────────────────────────────────

def fig_dimensionality(M: dict):
    keys = [k for k in KEYS if k in M]
    mdls = available(M)
    fig, axes = plt.subplots(2, len(keys), figsize=(4.3 * len(keys), 6.4), squeeze=False)
    for col, key in enumerate(keys):
        for row, (metric, name) in enumerate(
            [("n_retained", "components retained (Horn PA)"),
             ("participation_ratio", "participation ratio")]
        ):
            ax = _style(axes[row][col])
            w = 0.26
            for pi, pool in enumerate(POOLS):
                xs, vs, cs = [], [], []
                for mi, mdl in enumerate(mdls):
                    d = M[key].get(mdl, {}).get("dimensionality", {}).get(f"doc_{pool}")
                    if not d:
                        continue
                    xs.append(mi + (pi - 1) * w)
                    vs.append(d[metric])
                    cs.append(COLORS[mdl])
                alpha = {"image": 1.0, "text": 0.62, "all": 0.32}[pool]
                b = ax.bar(xs, vs, width=w * 0.92, color=cs, alpha=alpha,
                           edgecolor=SURF, linewidth=1.4)
                # relief rule: three palette slots fall under 3:1, so label directly
                for rect, v in zip(b, vs):
                    ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.0f}",
                            ha="center", va="bottom", fontsize=6.2, color=INK2)
            ax.set_xticks(range(len(mdls)))
            ax.set_xticklabels([SHORT[m] for m in mdls], fontsize=6.2)
            if col == 0:
                ax.set_ylabel(name)
            if row == 0:
                ax.set_title(KEY_LABEL[key], fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=INK2,
                             alpha={"image": 1.0, "text": 0.62, "all": 0.32}[p])
               for p in POOLS]
    axes[0][0].legend(handles, ["image tokens", "text tokens", "all tokens"],
                      frameon=False, fontsize=7.5)
    fig.suptitle("Effective dimensionality of the latent space, by token role", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_dimensionality.png", bbox_inches="tight")
    plt.close(fig)


# ── Fig 3: what the dimensions encode ────────────────────────────────────────

FACTORS = [
    ("r2_action", "action chunk\n(CV ridge R²)"),
    ("r2_phase", "time / phase\n(CV ridge R²)"),
    ("r2_state", "proprio state\n(CV ridge R²)"),
    ("eta2_instruction", "instruction\n(η²)"),
    ("temporal_smoothness", "temporal\nsmoothness"),
]


def fig_factors(M: dict, pool: str = "all"):
    keys = [k for k in KEYS if k in M]
    mdls = available(M)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 3.6), squeeze=False)
    for ax, key in zip(axes[0], keys):
        _style(ax)
        n = len(FACTORS)
        w = 0.8 / len(mdls)
        for mi, mdl in enumerate(mdls):
            f = M[key].get(mdl, {}).get("factors", {}).get(pool, {})
            vs = [f.get(k, np.nan) for k, _ in FACTORS]
            xs = np.arange(n) + (mi - (len(mdls) - 1) / 2) * w
            b = ax.bar(xs, vs, width=w * 0.9, color=COLORS[mdl],
                       edgecolor=SURF, linewidth=1.2,
                       label=LABELS[mdl].replace("\n", " "))
            for rect, v in zip(b, vs):
                if np.isfinite(v):
                    ax.text(rect.get_x() + rect.get_width() / 2, v,
                            f"{v:.2f}", ha="center",
                            va="bottom" if v >= 0 else "top",
                            fontsize=5.6, color=INK2, rotation=90)
        ax.set_xticks(range(n))
        ax.set_xticklabels([lbl for _, lbl in FACTORS], fontsize=7)
        ax.axhline(0, color=GRID, lw=1)
        ax.set_title(KEY_LABEL[key], fontsize=10)
        if ax is axes[0][0]:
            ax.set_ylabel("variance explained")
    h, l = axes[0][0].get_legend_handles_labels()
    # labelcolor stays ink: the swatch carries identity, the text must not.
    fig.legend(h, l, frameon=False, fontsize=8, ncol=len(mdls), labelcolor=INK,
               loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"What the retained dimensions encode ({pool} tokens)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"fig3_factors_{pool}.png", bbox_inches="tight")
    plt.close(fig)


# ── Fig 4: UMAP grid coloured by phase ───────────────────────────────────────

def fig_umap(M: dict, pool: str = "all"):
    keys = [k for k in KEYS if k in M]
    mdls = available(M)
    fig, axes = plt.subplots(len(mdls), len(keys),
                             figsize=(3.35 * len(keys), 3.05 * len(mdls)), squeeze=False)
    sc = None
    for r, mdl in enumerate(mdls):
        for c, key in enumerate(keys):
            ax = axes[r][c]
            ax.grid(False)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(GRID)
            p = DIR / f"embed_{mdl}_{key}.npz"
            if not p.exists():
                ax.axis("off"); continue
            emb = np.load(p)[f"umap_{pool}"]
            with h5py.File(DIR / f"probe_{key}.h5", "r") as f:
                phase = f["phase"][:]
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=phase, cmap=PHASE_CMAP,
                            s=3.2, linewidths=0, alpha=0.85)
            if r == 0:
                ax.set_title(KEY_LABEL[key], fontsize=10)
            if c == 0:
                ax.set_ylabel(LABELS[mdl], fontsize=8)
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes, fraction=0.016, pad=0.012)
        cb.set_label("phase within episode (0 = start, 1 = end)", fontsize=8)
        cb.outline.set_edgecolor(GRID)
    fig.suptitle(f"UMAP of the retained PC space ({pool} tokens), coloured by episode phase",
                 fontsize=11)
    fig.savefig(FIG / f"fig4_umap_phase_{pool}.png", bbox_inches="tight")
    plt.close(fig)


# ── Fig 5: Language Table UMAP coloured by instruction verb ──────────────────

VERB_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # first three slots validate all-pairs


def fig_umap_instruction(M: dict, pool: str = "text"):
    key = "language_table"
    if key not in M:
        return
    mdls = available(M)
    with h5py.File(DIR / f"probe_{key}.h5", "r") as f:
        instr = json.loads(f.attrs["instructions"])
        iid = f["instr_id"][:]

    def verb(s: str) -> str:
        w = re.findall(r"[a-z]+", s.lower())
        return w[0] if w else "other"

    verbs = np.array([verb(instr[i]) for i in iid])
    top = [v for v, _ in sorted(((v, (verbs == v).sum()) for v in set(verbs)),
                                key=lambda t: -t[1])[:3]]
    # Wrap past four arms — see the same note in plots_joint.fig_joint_umap.
    ncol = min(len(mdls), 3) if len(mdls) > 4 else len(mdls)
    nrow = -(-len(mdls) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.6 * nrow), squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(mdls):]:
        ax.set_visible(False)
    for ax, mdl in zip(flat, mdls):
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        p = DIR / f"embed_{mdl}_{key}.npz"
        if not p.exists():
            ax.axis("off"); continue
        emb = np.load(p)[f"umap_{pool}"]
        other = ~np.isin(verbs, top)
        ax.scatter(emb[other, 0], emb[other, 1], c="#c9c9c4", s=3, linewidths=0,
                   alpha=0.6, label="other")
        for v, col in zip(top, VERB_COLORS):
            m = verbs == v
            ax.scatter(emb[m, 0], emb[m, 1], c=col, s=3.4, linewidths=0,
                       alpha=0.9, label=f'"{v}…"')
        ax.set_title(LABELS[mdl], fontsize=8)
    flat[0].legend(frameon=False, fontsize=7, markerscale=3, loc="best")
    fig.suptitle(f"Language Table — UMAP of {pool} tokens, coloured by instruction verb",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"fig5_umap_instruction_{pool}.png", bbox_inches="tight")
    plt.close(fig)


# ── Fig 6: image/text subspace geometry ──────────────────────────────────────

def fig_image_text(M: dict):
    keys = [k for k in KEYS if k in M]
    mdls = available(M)
    panels = [
        ("mean_angle_deg", "mean principal angle (°)\nlow = fused subspaces"),
        ("r2_text_from_image", "text predicted from image\n(CV ridge R²)"),
        ("r2_image_from_text", "image predicted from text\n(CV ridge R²)"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.5), squeeze=False)
    for ax, (metric, name) in zip(axes[0], panels):
        _style(ax)
        w = 0.8 / len(mdls)
        for mi, mdl in enumerate(mdls):
            vs = [M[k].get(mdl, {}).get("image_text_geometry", {}).get(metric, np.nan)
                  for k in keys]
            xs = np.arange(len(keys)) + (mi - (len(mdls) - 1) / 2) * w
            b = ax.bar(xs, vs, width=w * 0.9, color=COLORS[mdl],
                       edgecolor=SURF, linewidth=1.2,
                       label=LABELS[mdl].replace("\n", " "))
            for rect, v in zip(b, vs):
                if np.isfinite(v):
                    ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=5.8,
                            color=INK2, rotation=90)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([KEY_LABEL[k] for k in keys], fontsize=7.5)
        ax.set_title(name, fontsize=9)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, ncol=len(mdls), labelcolor=INK,
               loc="lower center", bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Geometry between the image-token and text-token subspaces", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_image_text_geometry.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    M = load_metrics()
    fig_scree(M)
    fig_dimensionality(M)
    for pool in ("all", "image", "text"):
        fig_factors(M, pool)
    fig_umap(M, "all")
    fig_umap(M, "image")
    # fig_umap_instruction is NOT generated: it is built entirely from Language
    # Table, which is excluded from every per-arm analysis. It was also never
    # referenced by the report, so it only produced an orphan PNG that could be
    # mistaken for current output. The function is kept for ad-hoc use.
    fig_image_text(M)
    print(f"figures written to {FIG}")
    for p in sorted(FIG.glob("*.png")):
        print("  ", p.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ── shared label placement ───────────────────────────────────────────────────

def resolve_label_overlaps(fig, ax, anns, pad_px: float = 3.0, iters: int = 400,
                           avoid=None, avoid_r_px: float = 13.0,
                           expand: bool = True):
    """Nudge annotation text apart until no two rendered boxes overlap.

    Offsetting labels by a fixed vector, or stacking them along one axis, both
    fail for the same reason: whether two labels collide depends on the RENDERED
    text width, which is unknown until the figure is drawn and has nothing to do
    with how far apart their anchor points are. "SmolVLA (robot-FT)" is far wider
    than "Pi-0.5", so equal spacing in data units gives unequal spacing on paper.

    This measures the actual boxes, pushes overlapping pairs apart along their
    axis of least overlap (the shorter move), and repeats. Leader lines are drawn
    by the caller's arrowprops, so a label that travels stays attached to its
    point.

    `avoid` is a sequence of data-space points (typically the markers themselves)
    that labels must also stay clear of. Without it a label can be moved out of
    another label's way and straight on top of a scatter point, which is just as
    unreadable — and is exactly what happened to the Qwen label in fig12.

    Call AFTER creating every annotation on the axes and BEFORE saving.
    """
    avoid_disp = (ax.transData.transform(np.asarray(avoid, dtype=float))
                  if avoid is not None and len(avoid) else None)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    for _ in range(iters):
        boxes = [a.get_window_extent(renderer=r) for a in anns]
        moved = False
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                bi, bj = boxes[i], boxes[j]
                ox = min(bi.x1, bj.x1) - max(bi.x0, bj.x0) + pad_px
                oy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0) + pad_px
                if ox <= 0 or oy <= 0:
                    continue                      # already clear
                # push along whichever axis needs the smaller correction
                if ox < oy:
                    d = ox / 2 + 0.5
                    sx = 1.0 if bi.x0 + bi.x1 >= bj.x0 + bj.x1 else -1.0
                    push = np.array([[sx * d, 0.0], [-sx * d, 0.0]])
                else:
                    d = oy / 2 + 0.5
                    sy = 1.0 if bi.y0 + bi.y1 >= bj.y0 + bj.y1 else -1.0
                    push = np.array([[0.0, sy * d], [0.0, -sy * d]])
                for k, a in ((0, anns[i]), (1, anns[j])):
                    p = np.array(a.get_position())
                    disp = ax.transData.transform(p) + push[k]
                    a.set_position(tuple(inv.transform(disp)))
                moved = True

        # labels vs markers
        if avoid_disp is not None:
            for i, a_ in enumerate(anns):
                b_ = a_.get_window_extent(renderer=r)
                cx, cy = (b_.x0 + b_.x1) / 2, (b_.y0 + b_.y1) / 2
                for (px, py) in avoid_disp:
                    inside_x = b_.x0 - avoid_r_px < px < b_.x1 + avoid_r_px
                    inside_y = b_.y0 - avoid_r_px < py < b_.y1 + avoid_r_px
                    if not (inside_x and inside_y):
                        continue
                    v = np.array([cx - px, cy - py], dtype=float)
                    n = float(np.linalg.norm(v))
                    v = v / n if n > 1e-6 else np.array([0.0, 1.0])
                    pos = np.array(a_.get_position())
                    disp = ax.transData.transform(pos) + v * (avoid_r_px * 0.9)
                    a_.set_position(tuple(inv.transform(disp)))
                    moved = True

        if not moved:
            break
        fig.canvas.draw()
        r = fig.canvas.get_renderer()

    # Grow the axes so every label sits inside it. Without this the labels stay
    # where they were pushed to and simply overprint the title or spill past the
    # frame — which is the failure this function exists to prevent.
    #
    # expand=False when the axes limits are already generous and fixed: growing
    # them changes the data->display transform, which invalidates the pixel-space
    # separation just computed, which can shrink the point cloud and reintroduce
    # the very overlaps that were resolved. Only expand when labels genuinely
    # need room beyond the current frame.
    if not expand:
        return
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    xs, ys = list(ax.get_xlim()), list(ax.get_ylim())
    for a in anns:
        b = a.get_window_extent(renderer=r)
        (x0, y0), (x1, y1) = inv.transform((b.x0, b.y0)), inv.transform((b.x1, b.y1))
        xs = [min(xs[0], x0, x1), max(xs[1], x0, x1)]
        ys = [min(ys[0], y0, y1), max(ys[1], y0, y1)]
    mx, my = 0.04 * (xs[1] - xs[0]), 0.06 * (ys[1] - ys[0])
    ax.set_xlim(xs[0] - mx, xs[1] + mx)
    ax.set_ylim(ys[0] - my, ys[1] + my)
