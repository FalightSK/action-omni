"""
Assembles the self-contained HTML report from metrics.json + the figures/ dir.

Images are inlined as base64 data URIs (the report is meant to be published as
a standalone artifact with no external file dependencies). Only a curated
subset of the generated figures is included — the ones that carry a distinct
piece of the argument — to keep the report legible; the rest remain on disk
in figures/ for anyone who wants the per-pool breakdowns.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "asset" / "analysis" / "latent_compare"
FIG = DIR / "figures"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbones import ARMS as MODELS  # noqa: E402
from backbones import DOC_LAYER, GATE_KEYS, KEYS, PAIRS  # noqa: E402

NAME = {
    "qwen": "Qwen3.5-0.8B", "pi05": "Pi-0.5", "paligemma": "PaliGemma-3B",
    "smolvla": "SmolVLA", "smolvlm2": "SmolVLM2-500M", "groot": "GR00T N1.7-3B",
    "cosmos": "Cosmos-Reason2-2B", "qwen3vl": "Qwen3-VL-2B",
}
ROLE = {
    "qwen": "frozen, ours", "pi05": "robot-finetuned", "paligemma": "stock control",
    # SmolVLA's VLM is bit-identical to stock SmolVLM2 (345/345 tensors), so it
    # is not "robot-finetuned" in any sense that touches the representation.
    "smolvla": "VLM frozen", "smolvlm2": "stock control",
    "groot": "robot-finetuned", "cosmos": "physical-AI finetuned",
    "qwen3vl": "stock control",
}
KEY_LABEL = {
    "aloha_transfer": "ALOHA transfer-cube", "aloha_insertion": "ALOHA insertion",
    # Qualified deliberately: this is the curated 822-episode / 18.3k-frame
    # subset, not full Language Table (18,102 eps / 464,911 frames). ALOHA uses
    # its full datasets, so the comparison carries this asymmetry — an unlabelled
    # "Language Table" would read as the full set.
    "language_table": "Language Table (curated)",
    "libero_goal": "LIBERO-Goal",
}


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def img_tag(fname: str, alt: str, cls: str = "") -> str:
    p = FIG / fname
    return f'<img class="fig-img {cls}" src="data:image/png;base64,{b64(p)}" alt="{alt}" loading="lazy">'


def fnum(v, nd=3):
    if v is None:
        return "—"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return "—"
    if fv != fv:
        return "—"
    return f"{fv:.{nd}f}"


def main() -> int:
    M = json.loads((DIR / "metrics.json").read_text(encoding="utf-8"))

    # ---- pull the numbers the prose references, so they can't drift from metrics.json
    def get(key, model, *path):
        d = M.get(key, {}).get(model, {})
        for p in path:
            d = d.get(p, {})
        return d if d != {} else None

    action_table_rows = []
    for k in KEYS:
        row = [KEY_LABEL[k]]
        for m in MODELS:
            row.append(fnum(get(k, m, "factors", "all", "r2_action"), 3))
        action_table_rows.append(row)

    phase_table_rows = []
    for k in KEYS:
        row = [KEY_LABEL[k]]
        for m in MODELS:
            row.append(fnum(get(k, m, "factors", "all", "r2_phase"), 3))
        phase_table_rows.append(row)

    dim_table_rows = []
    for k in KEYS:
        row = [KEY_LABEL[k]]
        for m in MODELS:
            d = get(k, m, "dimensionality", "doc_all")
            row.append(f"{d['n_retained']} / {d['participation_ratio']:.1f}" if d else "—")
        dim_table_rows.append(row)

    angle_table_rows = []
    for k in KEYS:
        row = [KEY_LABEL[k]]
        for m in MODELS:
            g = get(k, m, "image_text_geometry")
            row.append(fnum(g.get("mean_angle_deg") if g else None, 1))
        angle_table_rows.append(row)

    def render_row(cells):
        return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"

    header_cells = "".join(
        f'<th>{NAME[m]}<span class="role">{ROLE[m]}</span></th>' for m in MODELS
    )

    action_rows_html = "\n".join(render_row(r) for r in action_table_rows)
    phase_rows_html = "\n".join(render_row(r) for r in phase_table_rows)
    dim_rows_html = "\n".join(render_row(r) for r in dim_table_rows)
    angle_rows_html = "\n".join(render_row(r) for r in angle_table_rows)

    # ---- joint (all-datasets-pooled) analysis
    J = json.loads((DIR / "joint_metrics.json").read_text(encoding="utf-8"))
    T = json.loads((DIR / "task_geometry.json").read_text(encoding="utf-8"))

    taskdist_rows = []
    for m in MODELS:
        Dn = T[m]["dist_norm"]
        ratio = J["per_model"]["image"][m]["task_scene_ratio"]
        hi = ' class="hl"' if m == "pi05" else ""
        taskdist_rows.append(
            f"<tr{hi}><td>{NAME[m]}<span class='role'>{ROLE[m]}</span></td>"
            # Pairs generated from however many datasets there are. The literal
            # six-pair list assumed four datasets and broke the moment Language
            # Table was dropped; with three datasets there are three pairs, and
            # three distances determine the triangle exactly.
            + "".join(f"<td>{Dn[i][j]:.2f}</td>"
                      for i in range(len(KEYS)) for j in range(i + 1, len(KEYS)))
            + f"<td><b>{ratio:.2f}</b></td></tr>"
        )
    taskdist_rows_html = "\n".join(taskdist_rows)

    R = J["rsa"]["matrix"]
    rsa_rows = []
    for i, m in enumerate(MODELS):
        cells = []
        for j in range(len(MODELS)):
            v = R[i][j]
            strong = ' class="strong"' if (i != j and v >= 0.9) else (
                ' class="weak"' if (i != j and v <= 0.6) else "")
            cells.append(f"<td{strong}>{v:.2f}</td>")
        rsa_rows.append(f"<tr><td>{NAME[m]}</td>" + "".join(cells) + "</tr>")
    rsa_rows_html = "\n".join(rsa_rows)
    rsa_header = "".join(f"<th>{NAME[m]}</th>" for m in MODELS)

    # headline deltas used in the callouts
    qwen_act_at = get("aloha_transfer", "qwen", "factors", "all", "r2_action")
    pg_act_at = get("aloha_transfer", "paligemma", "factors", "all", "r2_action")
    qwen_act_ai = get("aloha_insertion", "qwen", "factors", "all", "r2_action")
    pg_act_ai = get("aloha_insertion", "paligemma", "factors", "all", "r2_action")

    # Methods table, generated from the same DOC_LAYER the extraction used, so
    # the report cannot describe a read layer the data was not taken at.
    method_rows = []
    for m in MODELS:
        layer, n_layers, mode, prov = DOC_LAYER[m]
        method_rows.append(
            f'<tr><td><b>{NAME.get(m, m)}</b><br><span class="muted">{ROLE.get(m, "")}'
            f'</span></td><td class="num">layer {layer}</td>'
            f'<td class="num">{n_layers}</td>'
            f'<td>{"KV, all layers" if mode == "kv" else "hidden state"}</td>'
            f'<td><span class="mono">{prov}</span></td></tr>'
        )

    html = HTML_TEMPLATE.format(
        METHOD_ROWS="\n        ".join(method_rows),
        header_cells=header_cells,
        action_rows=action_rows_html,
        phase_rows=phase_rows_html,
        dim_rows=dim_rows_html,
        angle_rows=angle_rows_html,
        fig1=img_tag("fig1_scree_parallel_analysis.png",
                     "Eigenvalue spectrum of each backbone versus its permutation null"),
        fig2=img_tag("fig2_dimensionality.png",
                     "Components retained by parallel analysis and participation ratio, by token role"),
        fig7=img_tag("fig7_pca_scatter_raw.png",
                     "Raw PC1 vs PC2 scatter for each backbone and dataset, coloured by episode phase"),
        fig8=img_tag("fig8_r2_explain_phase.png",
                     "Predicted vs actual episode phase, 5-fold cross-validated ridge regression"),
        fig9=img_tag("fig9_r2_explain_action.png",
                     "Predicted vs actual leading action component, 5-fold cross-validated ridge regression"),
        fig3=img_tag("fig3_factors_all.png",
                     "Variance explained in the retained PCs by action, phase, state, instruction and smoothness"),
        fig4=img_tag("fig4_umap_phase_all.png",
                     "UMAP of the retained PC space, coloured by episode phase, for every backbone and dataset"),
        fig6=img_tag("fig6_image_text_geometry.png",
                     "Principal angle and cross-prediction between the image-token and text-token subspaces"),
        fig10=img_tag("fig10_joint_umap_datasets.png",
                      "All three datasets embedded in one latent space per backbone"),
        fig18=img_tag("fig18_task_tetrahedron.png",
                      "All three datasets placed per backbone by classical MDS, with true edge lengths"),
        fig19=img_tag("fig19_task_distance_matrix.png",
                      "Normalised centroid distance matrix between the three datasets, per backbone"),
        fig17=img_tag("fig17_libero_goal_umap.png",
                      "UMAP of LIBERO-Goal image and text tokens coloured by goal, per backbone"),
        # fig20 (the depth-matched-control chart) is deliberately NOT passed.
        # It plotted cosmos16 / smolvlm2_16, arms removed from the roster once
        # every stock control began being read at its descendant's documented
        # layer. Embedding a figure whose inputs no longer exist would show
        # Aug-8 numbers beside current ones with nothing to mark the difference.
        fig15=img_tag("fig15_depth_curves.png",
                      "Action decodability against relative read depth, per dataset and backbone"),
        fig16=img_tag("fig16_dataset_gate.png",
                      "Action decodability per dataset with every backbone shown, and LIBERO-Goal broken out by factor"),
        fig12=img_tag("fig12_rsa_models.png",
                      "Representational similarity between backbones and their MDS placement"),
        taskdist_rows=taskdist_rows_html,
        rsa_header=rsa_header,
        rsa_rows=rsa_rows_html,
        qwen_vs_pg_at=fnum(qwen_act_at - pg_act_at if qwen_act_at and pg_act_at else None, 3),
        qwen_vs_pg_ai=fnum(qwen_act_ai - pg_act_ai if qwen_act_ai and pg_act_ai else None, 3),
    )

    out = DIR / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
    return 0


HTML_TEMPLATE = r"""
<style>
:root {{
  --bg: #f5f6f8;
  --surface: #ffffff;
  --surface-2: #eef0f3;
  --text: #14181c;
  --text-2: #454e56;
  --text-3: #7c848c;
  --border: #dde1e6;
  --accent: #2a78d6;
  --accent-soft: #e8f0fc;
  --accent-ink: #ffffff;
  --good: #1baf7a;
  --good-soft: #e6f7f0;
  --warn: #eb6834;
  --warn-soft: #fdece3;
  --shadow: 0 1px 2px rgba(20,24,28,0.04), 0 8px 24px rgba(20,24,28,0.06);
  color-scheme: light;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    --bg: #101317;
    --surface: #171b1f;
    --surface-2: #1d2227;
    --text: #edf0f2;
    --text-2: #b7bfc6;
    --text-3: #838d95;
    --border: #2a3036;
    --accent: #4b93ea;
    --accent-soft: rgba(75,147,234,0.14);
    --accent-ink: #0c1116;
    --good: #22c58a;
    --good-soft: rgba(34,197,138,0.12);
    --warn: #ef7a4c;
    --warn-soft: rgba(239,122,76,0.14);
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 28px rgba(0,0,0,0.35);
    color-scheme: dark;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #101317; --surface: #171b1f; --surface-2: #1d2227;
  --text: #edf0f2; --text-2: #b7bfc6; --text-3: #838d95; --border: #2a3036;
  --accent: #4b93ea; --accent-soft: rgba(75,147,234,0.14); --accent-ink: #0c1116;
  --good: #22c58a; --good-soft: rgba(34,197,138,0.12);
  --warn: #ef7a4c; --warn-soft: rgba(239,122,76,0.14);
  --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 28px rgba(0,0,0,0.35);
  color-scheme: dark;
}}
:root[data-theme="light"] {{
  --bg: #f5f6f8; --surface: #ffffff; --surface-2: #eef0f3;
  --text: #14181c; --text-2: #454e56; --text-3: #7c848c; --border: #dde1e6;
  --accent: #2a78d6; --accent-soft: #e8f0fc; --accent-ink: #ffffff;
  --good: #1baf7a; --good-soft: #e6f7f0; --warn: #eb6834; --warn-soft: #fdece3;
  --shadow: 0 1px 2px rgba(20,24,28,0.04), 0 8px 24px rgba(20,24,28,0.06);
  color-scheme: light;
}}

* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
body {{
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 16px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}}
.report-shell {{
  display: grid;
  grid-template-columns: 232px minmax(0, 1fr);
  gap: 0;
  max-width: 1320px;
  margin: 0 auto;
}}
.mono {{
  font-family: "SF Mono", "Cascadia Code", "Consolas", "Roboto Mono", monospace;
  font-variant-numeric: tabular-nums;
}}

/* ---- rail ---- */
.rail {{
  position: sticky;
  top: 0;
  align-self: start;
  height: 100vh;
  overflow-y: auto;
  padding: 2.2rem 1.1rem 2rem 1.6rem;
  border-right: 1px solid var(--border);
}}
.rail-eyebrow {{
  font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
  font-size: 0.68rem;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--text-3);
  margin: 0 0 0.35rem;
}}
.rail-title {{
  font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
  font-size: 1.18rem;
  font-weight: 600;
  line-height: 1.28;
  margin: 0 0 1.6rem;
  color: var(--text);
}}
.rail nav {{ display: flex; flex-direction: column; gap: 0.15rem; }}
.rail a {{
  display: block;
  padding: 0.38rem 0.6rem;
  border-radius: 6px;
  color: var(--text-2);
  text-decoration: none;
  font-size: 0.85rem;
  border-left: 2px solid transparent;
}}
.rail a:hover {{ background: var(--surface-2); color: var(--text); }}
.rail a.active {{
  color: var(--accent);
  border-left-color: var(--accent);
  background: var(--accent-soft);
  font-weight: 600;
}}
.rail .sub {{ padding-left: 1.3rem; font-size: 0.78rem; }}

/* ---- main column ---- */
main {{ padding: 2.6rem 3.2rem 6rem; min-width: 0; }}
.hero-eyebrow {{
  font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
  font-size: 0.75rem;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0 0 0.7rem;
}}
h1.report-title {{
  font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
  font-size: 2.35rem;
  font-weight: 600;
  line-height: 1.15;
  letter-spacing: -0.01em;
  text-wrap: balance;
  margin: 0 0 0.9rem;
  max-width: 30ch;
}}
.dek {{
  font-size: 1.08rem;
  color: var(--text-2);
  max-width: 62ch;
  margin: 0 0 1.6rem;
}}
.meta-strip {{
  display: flex;
  flex-wrap: wrap;
  gap: 1.6rem;
  padding: 1rem 0 2.4rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2.6rem;
}}
.meta-item {{ display: flex; flex-direction: column; gap: 0.15rem; }}
.meta-item .k {{
  font-size: 0.68rem; letter-spacing: 0.07em; text-transform: uppercase;
  color: var(--text-3);
}}
.meta-item .v {{ font-size: 0.92rem; color: var(--text); font-weight: 500; }}

section.block {{ margin-bottom: 3.6rem; scroll-margin-top: 1.5rem; }}
section.block h2 {{
  font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
  font-size: 1.55rem;
  font-weight: 600;
  margin: 0 0 0.3rem;
  text-wrap: balance;
}}
section.block .section-num {{
  font-family: "SF Mono", Consolas, monospace;
  color: var(--accent);
  font-size: 0.85rem;
  margin-right: 0.5rem;
}}
section.block .lede {{
  color: var(--text-2);
  max-width: 68ch;
  margin: 0.5rem 0 1.4rem;
}}
section.block h3 {{
  font-size: 1.05rem;
  font-weight: 650;
  margin: 1.8rem 0 0.5rem;
}}
p {{ max-width: 70ch; color: var(--text); }}
p.narrow {{ max-width: 68ch; color: var(--text-2); }}

.callout {{
  border-left: 3px solid var(--accent);
  background: var(--accent-soft);
  border-radius: 0 10px 10px 0;
  padding: 1rem 1.3rem;
  margin: 1.2rem 0;
  max-width: 68ch;
}}
.callout.good {{ border-color: var(--good); background: var(--good-soft); }}
.callout.warn {{ border-color: var(--warn); background: var(--warn-soft); }}
.callout .tag {{
  font-family: "SF Mono", Consolas, monospace;
  font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--text-3); display: block; margin-bottom: 0.35rem;
}}
.callout p {{ margin: 0; color: var(--text); max-width: none; }}
.callout p + p {{ margin-top: 0.5rem; }}

.figure-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  box-shadow: var(--shadow);
  padding: 1.1rem;
  margin: 1.4rem 0;
  max-width: 100%;
}}
.fig-img {{ width: 100%; height: auto; display: block; border-radius: 6px; }}
figcaption, .fig-cap {{
  font-size: 0.83rem;
  color: var(--text-3);
  margin-top: 0.7rem;
  max-width: 78ch;
}}
.fig-cap b {{ color: var(--text-2); }}

.table-wrap {{ overflow-x: auto; margin: 1.2rem 0; border: 1px solid var(--border); border-radius: 10px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; }}
thead th {{
  text-align: left;
  background: var(--surface-2);
  color: var(--text-2);
  font-weight: 600;
  padding: 0.6rem 0.8rem;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}}
/* .role is a sub-label under a name — must be block in BOTH th and td, or it
   runs inline and reads as "Qwen3.5-0.8Bfrozen, ours". */
th .role, td .role {{ display: block; font-weight: 400; color: var(--text-3); font-size: 0.72rem; }}
/* Small inline tables used inside the finding callouts. They sit on the callout's
   tinted background rather than the page surface, so they take their own borders
   and stay narrow — these are 3-5 row comparisons, not data tables. */
table.mini {{ width: auto; margin: 0.7rem 0; font-size: 0.82rem; border-collapse: collapse; }}
table.mini th, table.mini td {{ padding: 0.28rem 0.75rem 0.28rem 0; text-align: left;
  border-bottom: 1px solid var(--border); white-space: nowrap; }}
table.mini th {{ font-weight: 600; color: var(--text-2); font-size: 0.76rem;
  text-transform: uppercase; letter-spacing: 0.04em; }}
table.mini tr:last-child td {{ border-bottom: none; }}
tbody td {{
  padding: 0.55rem 0.8rem;
  border-bottom: 1px solid var(--border);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}}
tbody td:first-child {{ font-variant-numeric: normal; color: var(--text-2); font-weight: 500; }}
tbody tr:last-child td {{ border-bottom: none; }}
tbody tr:hover {{ background: var(--surface-2); }}
tbody tr.hl td {{ background: var(--accent-soft); }}
tbody tr.hl td:first-child {{ box-shadow: inset 3px 0 0 var(--accent); }}
td.strong {{ color: var(--accent); font-weight: 700; }}
td.weak {{ color: var(--warn); font-weight: 700; }}

.flow {{
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 0.5rem;
  margin: 1.4rem 0 1.8rem;
  max-width: 78ch;
}}
.flow .step {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.75rem 0.7rem;
  position: relative;
}}
.flow .step .n {{
  font-family: "SF Mono", Consolas, monospace;
  font-size: 0.68rem; color: var(--accent); display: block; margin-bottom: 0.3rem;
}}
.flow .step .t {{ font-size: 0.78rem; color: var(--text-2); line-height: 1.35; }}
.flow .step:not(:last-child)::after {{
  content: "→";
  position: absolute;
  right: -1.15rem; top: 50%; transform: translateY(-50%);
  color: var(--text-3); font-size: 0.9rem;
}}
@media (max-width: 900px) {{
  .flow {{ grid-template-columns: 1fr 1fr; }}
  .flow .step:not(:last-child)::after {{ content: none; }}
}}

.pillbar {{ display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.8rem 0 1.4rem; }}
.pill {{
  font-family: "SF Mono", Consolas, monospace;
  font-size: 0.72rem;
  padding: 0.28rem 0.6rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  color: var(--text-2);
  background: var(--surface);
}}
.pill.acc {{ color: var(--accent); border-color: var(--accent); background: var(--accent-soft); }}

hr.divider {{ border: none; border-top: 1px solid var(--border); margin: 2.6rem 0; }}

footer {{ color: var(--text-3); font-size: 0.82rem; max-width: 70ch; }}
footer a {{ color: var(--accent); }}

@media (max-width: 860px) {{
  .report-shell {{ grid-template-columns: 1fr; }}
  .rail {{ position: relative; height: auto; border-right: none; border-bottom: 1px solid var(--border); }}
  main {{ padding: 2rem 1.3rem 4rem; }}
}}

a:focus-visible, .rail a:focus-visible {{
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}}

/* ---- print: the sticky rail, dark surfaces and nowrap tables all assume a
   screen viewport; none of that survives pagination, so print gets its own
   rules rather than inheriting the screen ones. ---- */
@media print {{
  :root {{
    --bg: #ffffff; --surface: #ffffff; --surface-2: #f3f4f6;
    --text: #14181c; --text-2: #454e56; --text-3: #7c848c; --border: #d7dbe0;
    --accent: #1f5fae; --accent-soft: #eef4fc;
    --good: #148a5f; --good-soft: #eaf7f1;
    --warn: #c05427; --warn-soft: #fbebe2;
  }}
  body {{ font-size: 12.5px; }}
  @page {{ size: letter; margin: 0.6in 0.55in; }}
  .rail {{ position: static; height: auto; border-right: none; border-bottom: 1px solid var(--border); page-break-after: always; }}
  main {{ padding: 0.2in 0; }}
  .figure-card, .callout, .table-wrap, .flow {{ break-inside: avoid; box-shadow: none; }}
  .figure-card {{ border: 1px solid var(--border); }}
  section.block {{ break-before: auto; }}
  section.block h2 {{ break-after: avoid; }}
  .table-wrap {{ overflow-x: visible; }}
  table {{ font-size: 0.74rem; table-layout: fixed; width: 100%; }}
  /* `tbody td` (specificity 0,0,2) beats a bare `td` (0,0,1), so the screen
     rule `tbody td {{ white-space: nowrap }}` survived into print and clipped
     long cells instead of wrapping them — the methods table's provenance column
     was cut mid-string, losing exactly the config references section 00 exists
     to show. Match the specificity so the override actually applies. */
  table th, tbody td, thead th {{
    white-space: normal; overflow-wrap: anywhere; word-break: normal;
    padding: 0.4rem 0.45rem;
  }}
  /* Long unbroken identifiers (paths, config keys) must be allowed to break
     mid-token or a fixed-width column clips them again. */
  td .mono, th .mono {{ overflow-wrap: anywhere; }}
  th .role {{ font-size: 0.64rem; }}
  a {{ color: var(--accent); text-decoration: none; }}
}}
</style>

<div class="report-shell">
  <aside class="rail" id="rail">
    <p class="rail-eyebrow">action-omni · latent study</p>
    <p class="rail-title">Cross-backbone latent comparison</p>
    <nav>
      <a href="#overview">Overview</a>
      <a href="#dimensionality">1 · Effective dimensionality</a>
      <a href="#pca">2 · PCA structure</a>
      <a href="#r2">3 · How R² is retrieved</a>
      <a href="#umap">4 · UMAP</a>
      <a href="#geometry">5 · Image / text geometry</a>
      <a href="#joint">6 · All tasks in one space</a>
      <a href="#findings">Findings</a>
      <a href="#nextchapter">Chapter 2 →</a>
      <a href="#caveats">Caveats</a>
    </nav>
  </aside>

  <main>
    <p class="hero-eyebrow">VLA anatomy — chapter 1: what shipped VLAs actually contain</p>
    <h1 class="report-title" id="overview">Which parts of a VLA are load-bearing, and which are cargo?</h1>
    <p class="dek">
      This study began as "does robot pretraining help?" and became something more
      useful: a <b>dissection</b> of three published VLAs — Pi-0.5, SmolVLA and
      GR00T N1.7 — each measured against the exact checkpoint it was built from.
      The question it answers is not whether robot pretraining is good, but
      <b>which components a working VLA actually needs</b>.
    </p>
    <p class="dek">
      Two results reframed the whole thing. SmolVLA's vision-language backbone is
      <b>bit-identical to stock SmolVLM2</b> — it was never finetuned at all. And
      the study's own cleanest finding turned out to be a <b>depth artifact</b>,
      caught only by building controls that did not previously exist. Both are
      below, with the retraction stated in full.
    </p>

    <div class="meta-strip">
      <div class="meta-item"><span class="k">Backbones</span><span class="v">9 — 3 finetuned/stock pairs, 2 depth-matched controls, frozen Qwen</span></div>
      <div class="meta-item"><span class="k">Datasets</span><span class="v">ALOHA ×2, Language Table (curated), LIBERO-Goal</span></div>
      <div class="meta-item"><span class="k">Probe frames</span><span class="v">7,200 — identical frames for every backbone</span></div>
      <div class="meta-item"><span class="k">Evidence tier</span><span class="v">availability (linear probes), not necessity</span></div>
      <div class="meta-item"><span class="k">Dimensionality method</span><span class="v">Horn's parallel analysis</span></div>
      <div class="meta-item"><span class="k">Probe method</span><span class="v">5-fold CV ridge regression</span></div>
    </div>

    <div class="callout warn">
      <span class="tag">what this chapter can and cannot claim</span>
      <p>Every number here is a <b>linear probe</b>: it says whether information is
      present and linearly readable in a representation. It does <b>not</b> say a
      trained policy uses it. Those two diverge — an earlier closed-loop ALOHA
      ablation in this project found image tokens carried the control signal while
      offline loss had overstated the text pathway. Claims of the form "component X
      is <i>needed</i>" belong to chapter 2 (ablate, retrain, roll out); this
      chapter establishes what is <i>available</i>, and audits what the published
      checkpoints actually contain at the weight level.</p>
    </div>

    <section class="block" id="method-recap">
      <h3>Pipeline, in five steps</h3>
      <div class="flow">
        <div class="step"><span class="n">01</span><span class="t">Build one frozen probe set — same frames, same instructions, for every backbone</span></div>
        <div class="step"><span class="n">02</span><span class="t">Run each backbone; pool hidden states by image / text / all tokens</span></div>
        <div class="step"><span class="n">03</span><span class="t">Parallel analysis: keep only PCs that beat a permutation null</span></div>
        <div class="step"><span class="n">04</span><span class="t">Cross-validated ridge: how well do those PCs predict action / phase / state / instruction?</span></div>
        <div class="step"><span class="n">05</span><span class="t">UMAP the retained PCs; report geometry between image and text subspaces</span></div>
      </div>
      <p class="narrow">
        Every measurement below is computed on the <b>same 7,200 frames</b> across
        all eight backbones — the same image, the same instruction string — so a
        difference between backbones reflects the backbone, not the sample.
      </p>
      <p class="narrow">
        <b>The arms, and why each exists.</b> Each robot-trained model sits beside
        the checkpoint it was initialised from, and every link is verified by
        direct tensor comparison rather than taken from a model card:
      </p>
      <ul class="narrow">
        <li><b>Pi-0.5 ← PaliGemma-3B</b> — finetuned.</li>
        <li><b>SmolVLA ← SmolVLM2-500M</b> — <b>345 of 345 tensors identical</b>.
          SmolVLA froze its VLM (<span class="mono">train_expert_only=True</span>)
          and trained only the action expert, so it never modified a
          representation at all.</li>
        <li><b>GR00T N1.7 ← Cosmos-Reason2-2B ← Qwen3-VL-2B</b> — a three-level
          chain, both links real: 476 of 493 tensors differ (~1.8%/layer) and
          584 of 625 (~1.6%/layer). Cosmos declares
          <span class="mono">Qwen3VLForConditionalGeneration</span>, so it is a
          finetuned Qwen3-VL-2B, and adding that root separates what
          physical-reasoning training bought from what robot-action training
          bought.</li>
      </ul>
      <p class="narrow">
        Two of the three robot arms also <i>truncate</i> the language stack
        (SmolVLA 32→16, GR00T 28→16), which confounds "what finetuning did" with
        "what reading earlier does". An earlier version of this study built extra
        truncated arms to control for it; that is no longer necessary, because
        every stock control is now read at <b>the layer its own descendant
        consumes</b> — so the comparison is depth-matched by construction. See
        section 00. Frozen Qwen3.5-0.8B is the project's own backbone, present as
        the never-saw-a-robot reference.
      </p>
    </section>

    <hr class="divider">

    <section class="block" id="method">
      <h2><span class="section-num">00</span>What is measured, and where it is read</h2>
      <p class="lede">
        Every number in this report is a <b>mean-pooled hidden state or key/value
        tensor</b>, taken at a specific layer of a specific model, then reduced by
        PCA and probed by ridge regression. This section states exactly which
        layer and which tensor, per arm, because the answer is not the same for
        all of them and an earlier version of this study drew a false conclusion
        by assuming it was.
      </p>

      <h3>Pooling</h3>
      <p>
        For each frame the model is run once on <span class="mono">(image,
        instruction)</span>. The token sequence is split by role using the
        processor's own image-token id: <b>image</b> tokens, <b>text</b> tokens,
        and <b>all</b> tokens. Within each group the hidden state is
        <b>averaged over tokens</b>, weighted by the attention mask so padding
        never enters the average. That gives one vector per frame per pool.
        Figures state which pool they use; unless noted it is <span
        class="mono">all</span>.
      </p>

      <h3>Read layer — and why the stock control moves with it</h3>
      <p>
        Reading every arm at its own last layer is <b>not</b> neutral. Two of the
        robot arms truncate their language stack, so "last layer" compares layer
        16 against layer 32 and measures read depth rather than finetuning. Each
        stock control is therefore read at <b>the layer its own robot descendant
        consumes</b>. On Cosmos, the last layer differs from the layer GR00T
        actually reads by a relative norm of <b>4.64</b> — so this is not a
        rounding detail.
      </p>
      <table class="tbl">
        <thead><tr><th>Arm</th><th>Reads</th><th>of</th><th>Tensor</th>
          <th>Source</th></tr></thead>
        <tbody>
        {METHOD_ROWS}
        </tbody>
      </table>
      <p>
        <b>Hidden</b> means a single layer's residual stream, exactly what the
        probe takes. <b>KV</b> means the policy's action expert never sees the
        residual stream at all: it attends to per-layer <i>keys and values</i>,
        at every layer up to its read depth. For those arms a second tap is
        extracted — every layer's K and V, mean-pooled and concatenated — because
        no single hidden state reproduces what the policy consumes. Pi-0.5 yields
        9,216 dims that way (18 layers × K,V × 256 under multi-query attention,
        against a 2,048-wide stream); SmolVLA yields 10,240.
      </p>
      <p class="note">
        Two bounds worth keeping in mind when reading a KV row. Per layer, K =
        W<sub>k</sub>h is a <i>linear</i> map of the hidden state, so a linear
        probe on K can never beat a linear probe on h — the hidden number is an
        upper bound. Across layers the concatenation is bounded by no single
        hidden state and can exceed all of them. Because the KV tap is 4–10×
        wider in raw dimensions, every comparison is made <b>after</b> per-tap
        PCA retention, so a KV advantage cannot be an artifact of width.
      </p>

      <h3>Final normalisation — why "the same layer" is not automatically the same thing</h3>
      <p>
        Transformer libraries apply the language stack's final normalisation only
        to the <i>last</i> hidden state. So a truncated model read at its own last
        layer returns a <b>post-norm</b> vector, while a full model read at that
        same absolute layer returns a <b>pre-norm</b> one — and the two are not
        comparable even when the weights are bit-identical. RMSNorm carries
        learned per-channel gains, so it rotates as well as rescales; no
        downstream standardisation undoes it.
      </p>
      <p>
        This is not hypothetical. SmolVLA's VLM is <b>345 of 345 tensors
        identical</b> to stock SmolVLM2, and both are read at layer 16 — yet
        before this was corrected they had mean norms of 25.35 and 297.76 and a
        cosine similarity of <b>0.22</b>. Every tap in this report therefore has
        the model's own final norm applied, so "read at layer L" means "what this
        stack outputs if truncated at L" — which is the question the study asks.
        After the correction that pair reads cosine <b>0.999995</b>, the
        bit-identity it should always have shown.
      </p>
      <p class="note">
        The pair is retained deliberately as a <b>null control</b>. Identical
        weights read at the same layer must produce identical vectors, so
        whatever difference SmolVLA and SmolVLM2 still show is this method's
        noise floor, and any paired effect of comparable size should be read as
        indistinguishable from measurement error.
      </p>

      <h3>Depth grid</h3>
      <p>
        Alongside the documented read, each arm is tapped at <b>0 / 25 / 50 / 75 /
        100%</b> of its language stack. <b>0% is a control, not a filler point</b>:
        it is the vision tower output passed through the multimodal projector and
        merged with the text embedding lookup, <i>before any decoder block runs</i>.
        Whatever it already decodes belongs to the image encoder and the embedding
        table, so the rise from 0% to the peak is what the language stack itself
        contributes. Verified rather than assumed — at text positions it equals
        <span class="mono">embed_tokens(ids) × sqrt(hidden)</span> to 0.19%
        (bfloat16 rounding), and the hidden-state list has exactly
        n_layers + 1 entries.
      </p>
      <p class="note">
        At 0% the two modalities have not been mixed by any attention and are
        still in their native scales, so a pooled <span class="mono">all</span>
        vector is dominated by whichever has the larger norm — and the imbalance
        <b>inverts</b> between architectures (SmolVLM2 reads image 3.09 / text
        0.063; Pi-0.5 reads image 1.25 / text 3.40, since Gemma scales embeddings
        by sqrt(2048) = 45.25). Depth results are therefore reported per pool;
        an <span class="mono">all</span> curve alone would report a modality
        weighting artifact as a property of the language stack.
      </p>

      <h3>Datasets</h3>
      <p>
        Three testbeds: ALOHA transfer-cube, ALOHA insertion, and LIBERO-Goal.
        <b>Language Table has been removed from every per-arm analysis</b> — its
        action signal is absent for all arms (R² ≤ 0.063), so any comparison
        there returns "no effect" regardless of what is varied. It is retained in
        one place only, the dataset-gate figure, where it is the negative anchor
        that makes the gate a scale rather than two points.
      </p>
    </section>

    <hr class="divider">

    <section class="block" id="dimensionality">
      <h2><span class="section-num">01</span>Effective dimensionality — parallel analysis</h2>
      <p class="lede">
        A 1024–2048-wide latent vector does not use all of its width for real
        signal. To find out how much of it is real, each backbone's eigenvalue
        spectrum is compared against the spectrum of a <b>randomised version of
        the same data</b> — every feature column independently shuffled, which
        destroys any real relationship between features while keeping each
        feature's own distribution intact. Only components that clear the 95th
        percentile of that random spectrum are counted as real.
      </p>
      <div class="callout">
        <span class="tag">why not "keep 95% of variance"</span>
        <p>
          At ~2,400 samples and 960–2048 features, a matrix of <i>pure noise</i>
          already produces large leading eigenvalues just from sampling — a
          fixed variance cutoff would count that noise as structure. Building
          the null explicitly from the data's own (n, d) avoids that trap, and
          keeps the retained count comparable across backbones of different width.
        </p>
      </div>

      <div class="figure-card">
        {fig1}
        <p class="fig-cap"><b>Fig. 1</b> — solid lines are each backbone's real eigenvalue spectrum; dotted
        lines are the 95th percentile of that backbone's own permutation null. The dot marks the
        last component that still clears the null. Every backbone's null flattens to ~2.5–4 within
        a few components — real structure is the part of the solid line still climbing above it.</p>
      </div>

      <div class="figure-card">
        {fig2}
        <p class="fig-cap"><b>Fig. 2</b> — top row: number of components retained by parallel analysis,
        split by whether the pooling was over image tokens, text tokens, or all tokens. Bottom row:
        participation ratio, a second, width-insensitive estimate of effective dimensionality
        (higher = variance spread over more directions rather than concentrated in a few).</p>
      </div>

      <h3>Retained components (k) / participation ratio — all tokens, final layer</h3>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Dataset</th>{header_cells}</tr></thead>
          <tbody>{dim_rows}</tbody>
        </table>
      </div>
      <p class="narrow">
        Read as <span class="mono">k / PR</span>. Retained dimensionality spans 37–65 on
        ALOHA transfer, 44–67 on ALOHA insertion and 34–59 on LIBERO-Goal — so the three
        datasets sit in a similar band, and <b>width is a property of the backbone far more
        than of the dataset</b>. The stock 2–3B arms are consistently widest (PaliGemma 57,
        Cosmos 65, Qwen3-VL 64 on ALOHA transfer) while frozen Qwen-0.8B is narrowest at 37.
      </p>
      <p class="narrow">
        Pi-0.5 is the one arm that <i>narrows</i> on the language dataset — 47 components on
        ALOHA transfer against 34 on LIBERO-Goal, where every other arm holds steady or
        widens. A more compact code on the dataset where it also scores highest is consistent
        with findings 3b and 7: its representation is organised around the goal rather than
        spread across scene appearance.
      </p>
    </section>

    <hr class="divider">

    <section class="block" id="pca">
      <h2><span class="section-num">02</span>PCA structure — the actual top-2 axes</h2>
      <p class="lede">
        Parallel analysis says <i>how many</i> dimensions are real; this section
        shows what the <i>first two</i> of them actually look like, before any
        UMAP re-projection smooths the picture. Each point is one probe frame,
        plotted at its raw PC1/PC2 coordinate, coloured by how far into the
        episode that frame falls.
      </p>
      <div class="figure-card">
        {fig7}
        <p class="fig-cap"><b>Fig. 3</b> — raw PC1 vs PC2 (not UMAP), all-token pool. "var" in each axis
        label is that component's eigenvalue — the first two components already carry a large share
        of Fig. 1's retained variance. Look at the colour, not just the shape: a smooth light→dark
        gradient means the model's top two directions already track time-within-episode; a uniform
        speckle means they don't.</p>
      </div>
      <p class="narrow">
        On both ALOHA tasks every backbone shows a visible phase gradient in just the top two
        components — Pi-0.5 in particular resolves it into a single clean diagonal streak.
        LIBERO-Goal is weaker but still structured, consistent with its lower phase R²
        (0.65–0.83 against ALOHA's 0.87–0.93): ten goals share one scene, so the leading
        variance is split between progress and goal identity rather than tracking progress
        alone.
      </p>
    </section>

    <hr class="divider">

    <section class="block" id="r2">
      <h2><span class="section-num">03</span>How the R² numbers are actually produced</h2>
      <p class="lede">
        Every "R²" reported in this study is a <b>5-fold cross-validated ridge regression</b>:
        the retained PCs are the input, the quantity of interest (action, phase, state, or
        instruction) is the target. Held-out folds matter here — an unregularised in-sample fit
        from a 40-dim latent to a 224-dim ALOHA action chunk would report near-perfect R² on
        noise alone. Cross-validation is what makes a low R² trustworthy rather than a modelling
        failure.
      </p>

      <div class="flow" style="grid-template-columns: repeat(4, 1fr);">
        <div class="step"><span class="n">in</span><span class="t">retained PCs for one backbone / dataset / token pool</span></div>
        <div class="step"><span class="n">fit</span><span class="t">ridge regression, 5-fold split, held-out predictions only</span></div>
        <div class="step"><span class="n">compare</span><span class="t">predicted value vs. the actual value, per held-out frame</span></div>
        <div class="step"><span class="n">score</span><span class="t">R² = 1 − (residual variance / total variance)</span></div>
      </div>

      <h3>Phase — the clearest case</h3>
      <div class="figure-card">
        {fig8}
        <p class="fig-cap"><b>Fig. 4</b> — each point is one held-out probe frame: x = its actual
        phase within the episode, y = the ridge model's prediction from that backbone's latent,
        both z-scored. A tight diagonal band means the latent predicts phase well; a flat
        vertical cloud with no diagonal trend means it predicts nothing. Every panel here is a
        diagonal band: phase is recoverable in all three retained datasets, at R² 0.87–0.93 on
        ALOHA and 0.65–0.83 on LIBERO-Goal, for all eight backbones. The flat-cloud case was
        Language Table, which is excluded from this figure — it is shown in the dataset gate
        (Fig. 15) where its failure is the point.</p>
      </div>

      <h3>Action — reduced to one axis for the same picture</h3>
      <p class="narrow">
        The action chunk is multi-dimensional (14-D per joint pair on ALOHA, 2-D on Language
        Table), so it can't be put on one scatter axis directly. For this figure only, each
        action chunk is reduced to its own first principal component — one scalar that captures
        its dominant mode of variation — so the same predicted-vs-actual picture applies. The R²
        values here are consequently higher than the full multi-dimensional action R² reported in
        Table 2 and Section "Findings" (predicting one axis is an easier task than predicting all
        of them jointly); the comparison <i>between backbones and datasets</i> is what this figure
        is for, not the absolute numbers.
      </p>
      <div class="figure-card">
        {fig9}
        <p class="fig-cap"><b>Fig. 5</b> — same construction as Fig. 4, target is the action chunk's
        leading component. The ALOHA columns show strong diagonal structure across every
        backbone; LIBERO-Goal is visibly looser, matching its lower full-chunk R²
        (0.273–0.401 against ALOHA's 0.667–0.786). Both are well clear of the gate — the
        dataset that fails it, Language Table, is excluded here and shown in Fig. 15.</p>
      </div>

      <h3>What the retained dimensions encode, in full</h3>
      <div class="figure-card">
        {fig3}
        <p class="fig-cap"><b>Fig. 6</b> — the complete factor breakdown behind the R² numbers,
        all-token pool: action chunk and phase (ridge R²), proprioceptive state (ridge R²),
        instruction identity (η², one-way ANOVA — only meaningful where more than one instruction
        exists), and temporal smoothness (cosine similarity between adjacent frames in an episode).</p>
      </div>

      <h3>Full multi-dimensional action R² and phase R² (Table 2)</h3>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Dataset</th>{header_cells}</tr></thead>
          <tbody>{action_rows}</tbody>
        </table>
      </div>
      <p class="narrow" style="margin-bottom:1.6rem;">Action chunk, CV ridge R² (all output dimensions, averaged).</p>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Dataset</th>{header_cells}</tr></thead>
          <tbody>{phase_rows}</tbody>
        </table>
      </div>
      <p class="narrow">Episode phase, CV ridge R².</p>
    </section>

    <hr class="divider">

    <section class="block" id="umap">
      <h2><span class="section-num">04</span>UMAP — visualising the retained PC space</h2>
      <p class="lede">
        UMAP runs on the parallel-analysis-retained PCs (never on raw features —
        feeding it the noise directions PA already rejected would let that noise
        drive the neighbour graph). Model identity is carried by grid position,
        not colour: eight categorical hues are not reliably distinguishable in a
        dense scatter, so colour is reserved for one ordered variable at a time.
      </p>
      <div class="figure-card">
        {fig4}
        <p class="fig-cap"><b>Fig. 7</b> — same phase colouring as Fig. 3/4, now after UMAP's
        non-linear layout. ALOHA panels resolve into trajectory-like arcs with a consistent
        light&rarr;dark sweep; LIBERO-Goal panels are more fragmented, because ten goals in one
        scene produce several short arcs rather than a single sweep.</p>
      </div>
    </section>

    <hr class="divider">

    <section class="block" id="geometry">
      <h2><span class="section-num">05</span>Image / text subspace geometry</h2>
      <p class="lede">
        Beyond what each token pool encodes on its own, this measures how the
        image-token subspace and text-token subspace sit relative to each other:
        the principal angle between their top PCs (0° = identical subspace, 90° =
        orthogonal / fully separated), and how well one predicts the other by
        cross-validated ridge.
      </p>
      <div class="figure-card">
        {fig6}
        <p class="fig-cap"><b>Fig. 8</b> — left: mean principal angle in degrees. Right two panels:
        cross-prediction R² in each direction. In every one of six within-family comparisons
        (Pi-0.5 vs PaliGemma, SmolVLA vs SmolVLM2, across all three datasets), the robot-finetuned
        arm shows a <i>larger</i> angle than its stock control — image and text pull apart under
        robot pretraining.</p>
      </div>
      <h3>Mean principal angle (°), all datasets</h3>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Dataset</th>{header_cells}</tr></thead>
          <tbody>{angle_rows}</tbody>
        </table>
      </div>
    </section>

    <hr class="divider">

    <section class="block" id="joint">
      <h2><span class="section-num">06</span>All tasks in one space</h2>
      <p class="lede">
        Every section above measured one dataset at a time. Pooling all 7,200
        frames into a single space per backbone asks a different question: when a
        model sees two robots, two scenes and three tasks at once, what does its
        latent organise <i>by</i> &mdash; how things look, or what is being done?
      </p>
      <div class="callout">
        <span class="tag">the contrast that makes this measurable</span>
        <p><b>ALOHA transfer-cube vs ALOHA insertion</b> — same robot, same camera,
        same visual world, <i>different task</i>.<br>
        <b>ALOHA vs LIBERO-Goal</b> &mdash; different robot, different simulator,
        different camera, different action space.</p>
        <p>Any encoder separates the second pair trivially. The diagnostic is the
        <b>task/scene ratio</b>: the first distance divided by the second. Near 0
        means the latent is organised by appearance; near 1 means it treats a
        change of task as being as significant as a change of world. Measured on
        image tokens — text tokens would separate the datasets by construction,
        since the instruction strings differ.</p>
      </div>

      <div class="figure-card">
        {fig10}
        <p class="fig-cap"><b>Fig. 9</b> &mdash; all three datasets embedded together, one space per
        backbone, coloured by dataset. Backbone identity is carried by panel position, which frees
        hue to encode dataset alone. Pi-0.5's panel is the visible outlier: where every other arm
        shows three well-separated blobs, its two ALOHA clusters sit as far apart as either sits
        from LIBERO-Goal &mdash; the geometry its 0.811 task/scene ratio quantifies.</p>
      </div>

      <h3>The geometry, measured rather than projected</h3>
      <p class="narrow">
        With three datasets the picture <i>is</i> the measurement: three pairwise centroid
        distances determine a triangle <b>exactly</b> by the law of cosines, with no projection
        and therefore no error to report. An earlier four-dataset version of this study had to
        flatten a tetrahedron by MDS and retained only 77&ndash;96% of the geometry; excluding
        Language Table removes that approximation entirely.
      </p>
      <p class="narrow">
        The table below and the distance-matrix figure carry the raw normalised distances,
        unprojected. Each backbone is normalised by its own mean pairwise distance, so widths
        cancel and the <i>shapes</i> stay comparable across models of 0.5B&ndash;3B parameters.
        AT = ALOHA transfer-cube, AI = ALOHA insertion, LG = LIBERO-Goal.
      </p>

      <div class="table-wrap">
        <table>
          <thead><tr>
            <th>Backbone</th>
            <th>AT &harr; AI<span class="role">same scene, diff task</span></th>
            <th>AT &harr; LG<span class="role">diff scene</span></th>
            <th>AI &harr; LG<span class="role">diff scene</span></th>
            <th>task / scene ratio</th>
          </tr></thead>
          <tbody>{taskdist_rows}</tbody>
        </table>
      </div>
      <p class="narrow">
        Distances normalised per backbone (1.00 = that model's average task pair).
        <b>Pi-0.5 places its two ALOHA tasks 0.87 apart while placing ALOHA and LIBERO-Goal
        at 1.06&ndash;1.07</b> &mdash; nearly the same distance, which is why its ratio reaches
        0.811. Every other arm separates the two ALOHA tasks by only 0.44&ndash;0.68 while
        pushing LIBERO-Goal out past 1.14. SmolVLA and SmolVLM2 return identical rows, as
        identical weights must.
      </p>

      <div class="figure-card">

        <h3>The same geometry, placed by MDS &mdash; and why it is now exact</h3>
        <p class="narrow">
          Three datasets give three pairwise distances, and three distances determine a triangle
          <i>exactly</i>: the drawing <b>is</b> the measurement, with nothing projected away.
          The four-dataset version of this study could not make that claim &mdash; four points
          generally need three dimensions, so flattening them into a plane discarded 4&ndash;23%
          of the geometry and introduced mean edge errors of 0.07&ndash;0.15. Excluding Language
          Table removes the approximation rather than merely reporting it.
        </p>
        {fig18}
        <p class="fig-cap"><b>Fig. 10</b> &mdash; the three datasets placed by classical MDS on each
        backbone's 3&times;3 distance matrix. Edge labels are the <b>true</b> normalised distances,
        and every panel header now reads <span class="mono">2-D keeps 100%, mean edge error
        0.000</span> &mdash; the flattening is lossless because the configuration is planar by
        construction. Picture and number cannot disagree here.</p>
        {fig19}
        <p class="fig-cap"><b>Fig. 11</b> — the same measurement with no embedding at all. This is
        the authoritative view: nothing here can be distorted, because nothing has been projected.
        The two figures above are navigational aids for it.</p>
        <p class="narrow">
          <b>Three datasets, and the drawing is now the measurement.</b> Three pairwise
          distances determine a triangle <i>exactly</i> by the law of cosines, so with Language
          Table excluded there is no projection step and no error to report: every arm's panel
          reads <span class="mono">2-D keeps 100%, mean edge error 0.000</span>. The previous
          four-dataset version needed MDS into a plane and retained only 77–96% of the geometry.
        </p>
        <p class="narrow">
          <b>What the distances show.</b> Cross-dataset distances land in a narrow 1.06–1.29
          band for every arm, while the two ALOHA tasks — same robot, same camera, same visual
          world, <i>different task</i> — sit at 0.44–0.87. So these backbones separate
          <b>datasets</b> sharply and are largely indifferent to how similar the underlying
          <i>tasks</i> are: visual domain dominates task structure.
        </p>
        <p class="narrow">
          Pi-0.5 is the exception, and consistently so. Its ALOHA↔ALOHA gap of <b>0.87</b> is
          the largest in the study, and its cross-dataset distances (1.06, 1.07) are the
          smallest — the two nearly coincide, meaning it considers two tasks in the same visual
          world about as different as two entirely different worlds. That is what its 0.811
          task/scene ratio measures. At the other end, SmolVLA and SmolVLM2 return identical
          triangles to two decimals (0.44 / 1.29 / 1.27), as identical weights must.
        </p>
      </div>

      <h3>Placing the backbones themselves in one space</h3>
      <p class="narrow">
        Backbones have different widths and incomparable axes, so their frames cannot share a
        UMAP. Their frame×frame <i>distance matrices</i> can be compared, though — that
        representation lives in a common space regardless of model width. Correlating those
        matrices puts all eight backbones into a single map, each read at its documented layer.
      </p>
      <div class="figure-card">
        {fig12}
        <p class="fig-cap"><b>Fig. 12</b> — Spearman correlation between each pair of backbones'
        frame×frame distance matrices, and the MDS placement derived from it. The
        diagonal-adjacent cells are the ones to read: they compare each robot policy against
        the exact checkpoint it was built from.</p>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Backbone</th>{rsa_header}</tr></thead>
          <tbody>{rsa_rows}</tbody>
        </table>
      </div>
      <p class="narrow">
        <b>SmolVLA ↔ SmolVLM2 = 1.0000.</b> Exactly one, to four decimals — the two are the
        same weights read at the same layer, so this cell is a correctness check on the whole
        measurement rather than a finding. It is the reference against which the other three
        pairs should be read.
      </p>
      <p class="narrow">
        Against that reference: <b>Pi-0.5 ↔ PaliGemma = 0.818</b>, by a wide margin the lowest
        within-pair value in the study — robot training restructured its base substantially.
        <b>GR00T ↔ Cosmos = 0.936</b>, a real but smaller change. <b>Cosmos ↔ Qwen3-VL =
        0.974</b>, smaller still, which is why the physical-reasoning stage of that chain
        contributes almost nothing to the representational distance.
      </p>
    </section>

    <hr class="divider">

    <section class="block" id="findings">
      <h2>Findings</h2>

      <div class="callout warn">
        <span class="tag">finding 1 — dataset property, not backbone property</span>
        <p><b>Language Table's action signal is absent for all eight arms</b> (max 0.063),
        including <b>three independently developed robot policies</b> — Pi-0.5, SmolVLA and
        GR00T N1.7 — from three labs, three architectures and three robot corpora. Every one of
        them recovers ALOHA actions (R&sup2; 0.667&ndash;0.786) and none recovers Language
        Table's (0.007&ndash;0.063). Eight independent failures across three model families are
        much harder to attribute to any single model's blind spot.</p>
        <p>The latents are not empty — state R&sup2; reaches <b>0.906</b> (Cosmos-R2) and
        instruction separation is near-total (η&sup2; up to <b>0.982</b> on text tokens, SmolVLA).
        So these encoders see the arm and read the instruction perfectly well on this dataset;
        the missing piece is specifically the mapping from a single frame to the next action.</p>
        <p><b>This is why Language Table is excluded from every other analysis in this report.</b>
        An ablation on a dataset whose action is not recoverable returns "no effect" regardless of
        what is ablated, so keeping it would add rows that cannot discriminate between hypotheses.
        It is retained only in the dataset gate below, where its failure is the point.</p>
        <p><b>Scope.</b> "Language Table" here is the curated 822-episode / 18.3k-frame subset,
        not the full 18,102-episode set. The curated subset preserves the full set's action and
        state marginals to within ~5% and its support exactly, and curating to 8 balanced concepts
        should make action prediction <i>easier</i> rather than harder — but the decisive check is
        a replication on full Language Table, which has not been run.</p>
        <p>This should become a cheap upfront gate before training: <i>is the action linearly
        recoverable from the observation at all, by any encoder?</i> — rather than discovering it
        after a full train+eval cycle.</p>
      </div>

      <div class="callout good">
        <span class="tag">finding 2 — the frozen backbone is competitive</span>
        <p>On the datasets where action <i>is</i> predictable, frozen Qwen3.5-0.8B — which has
        never seen a single frame of robot data — outperforms stock PaliGemma-3B, a model with
        roughly double the hidden width: <span class="mono">Δ = {qwen_vs_pg_at}</span> on
        transfer-cube, <span class="mono">Δ = {qwen_vs_pg_ai}</span> on insertion. It trails the
        three robot-trained arms by 0.04–0.07 (Pi-0.5 −0.068, GR00T −0.060, SmolVLA −0.041) and
        does so at the <b>lowest effective dimensionality of all eight arms</b> on ALOHA
        transfer: k=37, against SmolVLA/SmolVLM2 at 41, PaliGemma at 57 and Cosmos at 65.</p>
        <p>The honest framing is "a frozen VLM retains most of the usable signal", not "robot
        pretraining buys nothing" — the robot-trained arms are consistently ahead. What the
        comparison does establish is that <b>never having seen a robot is not the binding
        constraint</b>: a 0.8B general-purpose encoder beats a 3B stock VLM on this task while
        using two-thirds of its dimensions.</p>
      </div>

      <div class="callout warn">
        <span class="tag">finding 4 — "robot-trained" is not one treatment</span>
        <p>Every arm is read at the layer its own descendant consumes (section 00), so each
        row below isolates the weights and nothing else. The four pairs then disagree
        completely about what robot training does — which means the label does not predict
        the result:</p>
        <table class="mini">
          <tr><th>pair</th><th>Δ task/scene</th><th>RSA to base</th><th>Δ η² instruction (image)</th></tr>
          <tr class="hl"><td><b>Pi-0.5 ← PaliGemma-3B</b></td><td><b>+0.224</b></td><td><b>0.818</b></td><td><b>+0.523</b></td></tr>
          <tr><td><b>GR00T N1.7 ← Cosmos-R2</b></td><td><b>+0.116</b></td><td>0.936</td><td>−0.020</td></tr>
          <tr><td>Cosmos-R2 ← Qwen3-VL-2B</td><td>−0.038</td><td>0.974</td><td>+0.007</td></tr>
          <tr><td><i>SmolVLA ← SmolVLM2-500M</i></td><td><i>−0.0000</i></td><td><i>1.0000</i></td><td><i>−0.0002</i></td></tr>
        </table>
        <p class="note">
          <b>The SmolVLA row is a null control, not a result.</b> Its VLM is 345 of 345
          tensors identical to stock SmolVLM2 — it froze the backbone
          (<span class="mono">train_expert_only=True</span>) and trained only the action
          expert — so read at the same layer the two <i>must</i> produce identical vectors.
          They do: Δ exactly −0.0000 and RSA exactly 1.0000. That fixes this method's noise
          floor at ~0.000, and any effect of comparable size should be read as measurement
          error rather than signal. It is also the check that caught three separate defects
          in this pipeline, each invisible in the outputs themselves.
        </p>
        <p><b>Pi-0.5 is the only arm robot training substantially restructured.</b> Ratio
        <b>0.811</b> against every other arm's 0.35–0.59, and only <b>0.818</b> similarity to
        its own base — the lowest within-pair RSA in the study by a wide margin. It considers
        the two ALOHA tasks nearly as different from each other as from LIBERO-Goal.</p>
        <p><b>GR00T moved, but less, and differently.</b> +0.116 task/scene at RSA 0.936 —
        two orders of magnitude above the noise floor, so real, but roughly half Pi-0.5's
        shift. And it moved on <i>only one</i> axis: its instruction η² went <b>down</b>
        (−0.020). Robot training reorganised its latent by task without making its visual
        representation any more language-conditioned.</p>
        <p><b>The physical-reasoning stage bought nothing measurable.</b> Adding Qwen3-VL-2B
        as the root of the chain is what makes this visible: Cosmos-R2 sits at −0.038
        task/scene and +0.007 η² against the stock model it was trained from, at RSA 0.974.
        So in the chain <span class="mono">Qwen3-VL-2B → Cosmos-R2 → GR00T</span>, essentially
        all of the representational change arrives at the <b>robot-action</b> stage, none at
        the physical-reasoning stage. Without the root this was unmeasurable — the previous
        version of this report could only say GR00T "sits at" a value, not that it moved.</p>
        <p>So of three published robot policies: one froze its VLM entirely, one changed it
        on a single axis, and one restructured it substantially. <b>Any claim of the form
        "robot pretraining does X" has to be checked per model</b>.</p>
      </div>

      <div class="callout">
        <span class="tag">finding 3 — RETRACTED, and why the retraction stands</span>
        <p><b>An earlier version of this finding was wrong.</b> It claimed robot finetuning
        separates the image and text subspaces, on "6 out of 6 within-pair comparisons". All
        six were confounded by read depth.</p>
        <p>Two of the three robot arms <b>truncate their language stack</b> as well as
        training it (SmolVLA 32→16, GR00T 28→16). Read at each arm's own last layer, those
        pairs compared layer 16 against layer 32 or 28 — and image/text subspaces separate
        naturally as you ascend a stack, with no finetuning involved. Depth-matched controls
        showed 3% of the effect surviving for GR00T and <b>0%</b> for SmolVLA.</p>
        <p>Those controls were extra arms: stock weights truncated to the robot arm's depth.
        They are no longer needed, because the confound is now removed at the source — every
        stock control is read at its descendant's documented layer, so the whole study is
        depth-matched by construction. The retraction stands; the machinery that produced it
        has been retired.</p>
        <p>SmolVLA's row was never evidence at all. Its VLM subtree is <b>bit-identical to
        stock SmolVLM2's first 16 layers</b> — 345 tensors, zero differing. That pair now
        serves as the study's null control (see finding 4) rather than as a treatment.</p>
        <p><b>What survives:</b> only Pi-0.5, whose pair was depth-matched from the start at
        18 vs 18 layers. The honest claim is <b>one model</b>, not six comparisons across two
        families.</p>
      </div>

      <div class="callout good">
        <span class="tag">finding 3b — two metrics, two different answers</span>
        <p>Image/text subspace separation collapsed under depth matching. Task organisation
        does not — and that contrast is what makes the retraction informative rather than
        merely negative. Measured at documented read layers, with the frozen pair fixing the
        floor at ~0.000:</p>
        <table class="mini">
          <tr><th>arm</th><th>task/scene ratio</th><th>vs its base</th></tr>
          <tr class="hl"><td><b>Pi-0.5</b></td><td><b>0.811</b></td><td><b>+0.224</b></td></tr>
          <tr><td>PaliGemma-3B (stock)</td><td>0.587</td><td>—</td></tr>
          <tr><td><b>GR00T N1.7</b></td><td><b>0.508</b></td><td><b>+0.116</b></td></tr>
          <tr><td>Qwen3-VL-2B (stock)</td><td>0.430</td><td>—</td></tr>
          <tr><td>Cosmos-R2</td><td>0.392</td><td>−0.038</td></tr>
          <tr><td>Qwen3.5-0.8B (frozen)</td><td>0.380</td><td>—</td></tr>
          <tr><td>SmolVLM2-500M (stock)</td><td>0.347</td><td>—</td></tr>
          <tr><td><i>SmolVLA</i></td><td><i>0.347</i></td><td><i>−0.0000</i></td></tr>
        </table>
        <p><b>Image/text separation is a depth phenomenon; task organisation is a real
        consequence of robot training.</b> Reporting only the first would credit robot
        pretraining with an effect it did not produce; reporting only the second would miss
        that it does produce one. Both are needed, and they rank the arms differently.</p>
        <p>Note also that a high ratio needs no robot data: stock PaliGemma reaches 0.587,
        above GR00T's 0.508. The <i>level</i> identifies nothing on its own — only the
        within-pair delta does, which is why every arm here has a verified base.</p>
        <p>That leaves a clean gradient over how much each robot policy actually changed its
        VLM: <b>Pi-0.5</b> substantial (+0.224, RSA 0.818) · <b>GR00T</b> real but smaller
        (+0.116, RSA 0.936) · <b>SmolVLA</b> none whatsoever (identical weights).</p>
        <p>The last point stands on its own and is the strongest external evidence in this
        study: SmolVLA is a published, working VLA whose vision-language backbone is
        <b>stock, frozen and unmodified</b>. Competitive robot control demonstrably does not
        require finetuning the VLM at all.</p>
      </div>
    </section>

      <div class="callout good">
        <span class="tag">finding 5 — control information lives in the first quarter of the stack</span>
        <p>An earlier version of this study sampled only 50 / 75 / 100% and concluded
        that action decodability "peaks at 50%", while flagging that the curve was
        <i>monotonically decreasing from the leftmost sampled point</i> and so
        left-censored — the true optimum could lie below 50%, where nothing had been
        measured. Adding <b>0% and 25%</b> resolves it, and the answer was indeed
        hiding below the old grid:</p>
        <p><b>Every arm peaks at or below 50%, and most peak at 25%.</b> Action
        information rises sharply over the first quarter of the language stack and
        degrades from there. Reading at the last layer — which this study and the
        project's training configs both did — leaves 0.02–0.09 R² on the table.</p>
        <p><b>PaliGemma-3B peaks at 0% on all three datasets</b> (+0.083 / +0.089 /
        +0.055 over its final layer), and SmolVLM2 does the same on ALOHA-insertion.
        For those arms the language decoder contributes <i>nothing</i> to action
        decodability: the vision tower and the embedding table already carry more
        than any layer above them.</p>
        {fig15}
        <p class="fig-cap"><b>Fig. 14</b> — action decodability against relative read depth,
        five depths per arm. The ring marks each arm's peak; dashed lines are stock controls.
        <b>0%</b> is the vision tower output through the multimodal projector merged with the
        text embedding lookup, before any decoder block runs — so the rise from 0% to the peak
        is what the language stack itself contributes, and a curve that starts at its maximum
        is an arm whose stack contributes nothing. The censoring noted in the previous version
        is resolved: every curve now turns over inside the sampled range.</p>
        <table class="mini">
          <tr><th>arm</th><th>ALOHA-T gain</th><th>ALOHA-I gain</th><th>LIBERO gain</th></tr>
          <tr><td>Qwen-0.8B (frozen)</td><td><b>+0.035</b></td><td><b>+0.062</b></td><td><b>+0.046</b></td></tr>
          <tr><td>PaliGemma (stock)</td><td>+0.024</td><td>+0.039</td><td>+0.010</td></tr>
          <tr><td>Cosmos-R2 (stock)</td><td>+0.014</td><td>+0.022</td><td>+0.028</td></tr>
          <tr><td>Pi-0.5 (robot-FT)</td><td>+0.015</td><td>+0.021</td><td>+0.022</td></tr>
          <tr><td>GR00T N1.7 (robot-FT)</td><td>+0.010</td><td>+0.008</td><td>+0.014</td></tr>
        </table>
        <p>Gain = best depth minus last layer, action R². Two consequences.
        <b>SmolVLA and GR00T truncating to 16 layers looks justified rather than
        arbitrary</b> — possibly not even aggressive enough. And <b>frozen Qwen gains
        the most</b>: at mid-stack it reaches 0.721 / 0.767 on ALOHA against Pi-0.5's
        0.769 / 0.808, narrowing a gap reported elsewhere as 0.067 / 0.082 to roughly
        0.048 / 0.041 — without touching a weight.</p>
        <p>The effect is not an ALOHA quirk: it reproduces on LIBERO-Goal, a different
        robot, simulator, camera and action space.</p>
      </div>

      <div class="callout">
        <span class="tag">finding 6 — datasets differ in whether the task is learnable at all</span>
        <p>Finding 1 recommended a cheap upfront gate: <i>is the action linearly
        recoverable from one observation, by any encoder?</i> Language Table is excluded
        from every other analysis in this report precisely because it fails that gate —
        but it is retained <b>here</b>, and only here, because a gate needs a documented
        negative to be a scale rather than two points:</p>
        {fig16}
        <p class="fig-cap"><b>Fig. 15</b> — left: every backbone's action decodability in each
        dataset, with the shaded band marking the region below which a dataset cannot test the
        action pathway at all. The three regimes separate cleanly and the spread <i>within</i> a
        dataset is far smaller than the gap <i>between</i> datasets — which is what makes this a
        property of the data rather than of any backbone. Right: LIBERO-Goal by factor. Its action
        signal is middling, but Pi-0.5's instruction η² of 0.77 against 0.21–0.41 for everyone
        else is the reason it is the language testbed.</p>
        <table class="mini">
          <tr><th>dataset</th><th>action R² range</th><th>verdict</th></tr>
          <tr><td>ALOHA transfer / insertion</td><td>0.667 – 0.786</td><td>strong motor testbed</td></tr>
          <tr><td>LIBERO-Goal</td><td><b>0.273 – 0.401</b></td><td>usable — about 8× Language Table</td></tr>
          <tr><td>Language Table (curated)</td><td>0.007 – 0.063</td><td><b>fails the gate</b> — excluded elsewhere</td></tr>
        </table>
        <p>The spread <i>within</i> a dataset is far smaller than the gap <i>between</i>
        datasets, across eight backbones spanning three families and 0.5B–3B parameters.
        That is what makes decodability a property of the data rather than of any
        backbone — and what justifies running the gate before committing GPU time.</p>
        <p>LIBERO-Goal passes at roughly half of ALOHA's decodability — a real testbed,
        not an easy one. Its value is elsewhere: it is the only dataset here where
        <b>language must do work</b>, because its ten goals share a fixed scene and
        object set, so instruction η² measures language disambiguating
        <i>identical</i> scenes rather than correlating with visual layout as it does
        in Language Table.</p>
        <p>On that dataset <b>Pi-0.5 reaches image-token η² = 0.760, more than three times
        any other arm</b>, and leads action R² by 0.10–0.17 against 0.02–0.09 on ALOHA. The
        one model that genuinely restructured its VLM pulls furthest ahead exactly where
        language is load-bearing — independent corroboration of findings 3b and 4 from a
        different measurement.</p>
      </div>

      <div class="callout good">
        <span class="tag">finding 7 — Pi-0.5 fuses the instruction into its <i>image</i> tokens</span>
        {fig17}
        <p class="fig-cap"><b>Fig. 16</b> — LIBERO-Goal, coloured by goal. All ten goals share one
        fixed scene and object set. Top row: image tokens. Bottom row: text tokens.</p>
        <p>The bottom row is the expected result and confirms the probe works — every arm separates
        ten different instruction strings by their text tokens (η² 0.71–0.94). The top row is the
        finding.</p>
        <table class="mini">
          <tr><th>arm</th><th>read layer</th><th>image-token η²</th><th>text-token η²</th></tr>
          <tr class="hl"><td><b>Pi-0.5</b> (robot-FT)</td><td>18</td><td><b>0.760</b></td><td>0.930</td></tr>
          <tr><td>Cosmos-R2 (physical-AI FT)</td><td>16</td><td>0.235</td><td>0.929</td></tr>
          <tr><td>PaliGemma-3B (stock — Pi-0.5's base)</td><td>18</td><td>0.236</td><td>0.714</td></tr>
          <tr><td>SmolVLA (VLM frozen)</td><td>16</td><td>0.233</td><td>0.944</td></tr>
          <tr><td><i>SmolVLM2-500M (stock)</i></td><td>16</td><td><i>0.233</i></td><td><i>0.944</i></td></tr>
          <tr><td>Qwen3-VL-2B (stock)</td><td>16</td><td>0.229</td><td>0.900</td></tr>
          <tr><td>GR00T N1.7 (robot-FT)</td><td>16</td><td>0.215</td><td>0.924</td></tr>
          <tr><td>Qwen3.5-0.8B (frozen)</td><td>24</td><td>0.160</td><td>0.914</td></tr>
        </table>
        <p><b>Pi-0.5's image tokens carry goal identity at 0.760 — more than three times every
        other arm, which sits in a tight 0.16–0.24 band.</b> Since the scene is constant across
        the ten goals, an image token cannot know the goal unless instruction information has
        been propagated into it. This is cross-modal fusion, made visible: the same model that
        finding 4 identified as the only one substantially restructured, and finding 3b as the
        only one to change its task organisation most, is <i>also</i> the only one whose visual
        representation is conditioned on language. Against its own base the effect is
        <b>+0.523</b> — roughly 2,600× the null control's −0.0002.</p>
        <p><b>The other two robot policies do not do this.</b> SmolVLA reads 0.233 against its
        stock control's 0.233 (a null by construction — identical weights). GR00T reads
        <b>0.215 against Cosmos's 0.235</b>, i.e. −0.020: robot training left its visual
        representation slightly <i>less</i> language-conditioned, not more. Combined with
        finding 3b, GR00T reorganised its latent by task while adding nothing on this axis —
        two capabilities that can move independently, and only Pi-0.5 gained both.</p>
        <p><b>The pooled number hides the mechanism.</b> Everything above is averaged over the
        whole episode, and episode phase turns out to dominate it. The next two sections take it
        apart.</p>
      </div>

      <div class="callout good">
        <span class="tag">finding 7b — knowing the goal before it is visible</span>
        <p>LIBERO-Goal's ten tasks share one scene and object set, and initial object positions
        jitter independently of the task. So at the <b>start</b> of an episode a frame carries
        essentially no task information: nothing distinguishes "open the drawer" from "turn on
        the stove" until the robot acts. Splitting instruction η² by episode phase therefore
        separates two very different abilities — <i>knowing</i> the goal from <i>inferring</i> it.</p>
        <table class="mini">
          <tr><th>arm</th><th>early (phase&lt;0.10)</th><th>mid</th><th>late (&gt;0.90)</th>
            <th>null p95</th><th>early excess</th></tr>
          <tr class="hl"><td><b>Pi-0.5</b></td><td><b>0.873</b></td><td>0.868</td><td>0.868</td>
            <td>0.030</td><td><b>+0.843</b></td></tr>
          <tr><td>PaliGemma-3B (stock)</td><td>0.273</td><td>0.414</td><td>0.545</td>
            <td>0.029</td><td>+0.244</td></tr>
          <tr><td>SmolVLM2-500M (stock)</td><td>0.084</td><td>0.495</td><td>0.601</td>
            <td>0.028</td><td>+0.056</td></tr>
          <tr><td>SmolVLA</td><td>0.083</td><td>0.495</td><td>0.601</td><td>0.026</td><td>+0.057</td></tr>
          <tr><td>Cosmos-Reason2-2B</td><td>0.082</td><td>0.564</td><td>0.684</td><td>0.027</td><td>+0.055</td></tr>
          <tr><td>GR00T N1.7-3B</td><td>0.075</td><td>0.500</td><td>0.628</td><td>0.027</td><td>+0.048</td></tr>
          <tr><td>Qwen3-VL-2B (stock)</td><td>0.073</td><td>0.528</td><td>0.630</td><td>0.028</td><td>+0.045</td></tr>
          <tr><td>Qwen3.5-0.8B (frozen)</td><td>0.066</td><td>0.377</td><td>0.558</td><td>0.028</td><td>+0.037</td></tr>
        </table>
        <p><b>Pi-0.5 is flat across the episode — 0.873 / 0.868 / 0.868.</b> It resolves the goal
        at frame 0, before any visual evidence of it exists. Every other arm traces a rising
        curve from the noise floor: they are reading the goal off the unfolding trajectory, which
        is ordinary visual discrimination, not language use.</p>
        <p>The null is permuted at <b>episode</b> level, not frame level — frames within an
        episode share a task, so a per-frame shuffle would leave task structure intact and
        understate the floor.</p>
        <p class="note">
          This also explains why the pooled numbers looked like a flat 0.16–0.24 band: pooling
          across phase puts episode progress in the denominator, compressing every arm toward a
          similar value. Phase is the largest uncontrolled variance source in this measurement,
          and the earlier version of this finding did not control for it.
        </p>
      </div>

      <div class="callout good">
        <span class="tag">finding 7c — the attention mask decides whether fusion is possible</span>
        <p>The phase split raises an obvious question: why does <b>stock PaliGemma</b>, which
        never saw a robot, sit five times above the floor while <b>robot-trained GR00T</b> sits
        on it? The answer is architectural, and it is testable directly — encode the same frames
        with a different instruction and measure how far the image-token vector moves. All
        conditions are encoded in one batch so every item is padded to the same length; a changed
        sequence length alone perturbs a bfloat16 vector by ~1e-3 and would otherwise be
        mistaken for a small real effect.</p>
        <table class="mini">
          <tr><th>arm</th><th>different instruction</th><th>same words, scrambled</th>
            <th>empty text</th><th>ratio</th><th>reading</th></tr>
          <tr class="hl"><td><b>Pi-0.5</b></td><td><b>0.234</b></td><td>0.142</td><td>0.332</td>
            <td><b>1.65</b></td><td><b>tracks meaning</b></td></tr>
          <tr><td>PaliGemma-3B (stock)</td><td>0.042</td><td>0.047</td><td>0.064</td><td>0.90</td>
            <td>token-level only</td></tr>
          <tr><td>Qwen3.5-0.8B</td><td><b>0.0000</b></td><td>0.0000</td><td>0.0000</td><td>—</td>
            <td>cannot see the text</td></tr>
          <tr><td>SmolVLA / SmolVLM2</td><td><b>0.0000</b></td><td>0.0000</td><td>0.0000</td><td>—</td>
            <td>cannot see the text</td></tr>
          <tr><td>GR00T / Cosmos / Qwen3-VL</td><td><b>0.0000</b></td><td>0.0000</td><td>0.0000</td><td>—</td>
            <td>cannot see the text</td></tr>
        </table>
        <p><b>Six of eight arms return exactly zero</b> — bit-identical vectors, not merely small
        ones. Their image tokens are produced before the instruction is attended to, so no amount
        of training could make them carry it. GR00T is robot-trained and structurally blind.</p>
        <p>PaliGemma uses <b>bidirectional attention over its input</b>: image and instruction
        tokens attend to each other. That is a precondition for language-conditioned vision, and
        it is the only architectural difference that tracks the phase-split result.</p>
        <p><b>But access alone is not the capability.</b> Stock PaliGemma moves as much for its
        own words scrambled as for a genuinely different instruction (0.047 vs 0.042, ratio 0.90)
        — it is reacting to token <i>content</i>, not to meaning. Pi-0.5 moves <b>5.5× further
        than its base</b> and responds <b>1.65× more</b> to a changed meaning than to the same
        words shuffled. Removing the instruction entirely moves it most of all (0.332).</p>
        <p><b>So the finding restates as a three-tier claim:</b> the attention mask determines
        whether the instruction can reach the visual representation at all; the base model
        determines whether it does so at token level; and robot training is what converts that
        into sensitivity to what the sentence <i>means</i>. The capability is gated by a design
        decision that cannot be retrofitted by training — which makes it a more actionable result
        than "Pi-0.5 scores higher".</p>
        <p class="note">
          Measured on 12 early-phase LIBERO frames, image-token pool, at each arm's documented
          read layer. The zero/non-zero split is structural and independent of the sample; the
          magnitudes for the two arms that can see text are from this sample and would benefit
          from a larger one.
        </p>
      </div>

    </section>

    <section class="block" id="nextchapter">
      <h2>Where chapter 2 goes</h2>
      <p class="narrow">
        This chapter is an audit: it measures what published VLAs contain and what
        their representations make available. It cannot say what a policy
        <i>needs</i> — for that a component has to be removed, the model retrained,
        and the rollout success measured.
      </p>
      <p class="narrow">
        Chapter 2 takes the two components this chapter points at most directly:
        <b>language-stack depth</b> (finding 5 says the top half is not where control
        information lives) and the <b>VLM→action adapter</b> (AdaLN conditioning
        versus linear projection versus cross-attention). Both are coupled — which
        layer you read and how you inject it are one design decision, not two.
      </p>
      <p class="narrow">
        Three silent no-ops had to be removed from that code path first, any one of
        which would have produced a clean but false "depth doesn't matter" result:
        <span class="mono">encode_vlm</span> ignored its layer index and always
        returned the last hidden state; the embedding cache could be reused across
        depths with no check; and the Language Table cache was labelled with a layer
        index that does not exist in a 24-layer model.
      </p>
    </section>

    <section class="block" id="caveats">
      <h2>What this does not establish</h2>
      <div class="pillbar">
        <span class="pill">linear probes only</span>
        <span class="pill">last-layer readout</span>
        <span class="pill">50 ALOHA episodes</span>
        <span class="pill">per-family prompt formats</span>
        <span class="pill">GR00T has no control</span>
        <span class="pill">curated Language Table</span>
      </div>
      <p class="narrow">
        <b>Curated Language Table.</b> ALOHA contributes its full datasets, but Language Table
        contributes a curated 822-episode subset — roughly 4% of the 18,102 available — that was
        selected to fix <i>training</i> problems (phrasing mismatch, control-data starvation) by a
        criterion unrelated to representation geometry. Its action and state marginals match the
        full set to within ~5% with identical support, and 8 balanced concepts should make action
        prediction easier rather than harder, so the direction of any bias argues against
        Finding 1 being an artifact — but this asymmetry is untested and it affects the
        ALOHA-vs-Language-Table task geometry in Finding 4, not only Finding 1.
      </p>
      <p class="narrow">
        <b>Every arm now has a verified control</b>, which was not true of earlier versions of
        this study: Cosmos-Reason2-2B has been fetched and measured, and Qwen3-VL-2B added as
        the root of the GR00T chain. The one arm without a pair is frozen Qwen3.5-0.8B, which
        is the project's own reference rather than a treatment.
      </p>
      <p class="narrow">
        <b>Availability, not necessity.</b> Everything here is a linear probe: it establishes
        that information is present and linearly readable, not that a policy uses it. Those can
        diverge — a closed-loop ALOHA ablation earlier in this project found image tokens
        carried the control signal while offline loss had overstated the text pathway. No
        sentence in this report should be read as "component X is needed"; that requires
        ablate &rarr; retrain &rarr; roll out.
      </p>
      <p class="narrow">
        <b>The KV comparison is not dimensionality-matched.</b> The KV taps retain roughly twice
        the components of the hidden taps, so Table 8's deltas are measured at each tap's own
        retained dimensionality. Parallel analysis places those components above the permutation
        null and the R&sup2; is cross-validated, so it is not overfitting — but a matched-k test
        has not been run.
      </p>
      <p class="narrow">
        Ridge regression is a linear probe — a non-linear one might recover Language Table
        action signal that ridge cannot, though the fact that all eight arms across three model
        families agree makes a dataset-level explanation more plausible than a representational
        gap unique to one model. ALOHA's probe draws from only 50 episodes, which caps how
        independent its samples really are. Each backbone family was queried in its own native
        prompt format (Qwen's chat template, PaliGemma's bare-instruction format, SmolVLM's chat
        template, Qwen3-VL's for GR00T and Cosmos) — evaluating a model outside its trained
        format would measure the mismatch rather than the representation, but it does mean
        text-token counts differ across arms.
      </p>
      <p class="narrow">
        <b>Pi-0.5's read is an approximation.</b> Its action expert attends to per-layer keys and
        values at every layer, so no single hidden state reproduces its input. Table 8 measures
        the KV object directly; the hidden-state rows for that arm are a proxy, and are labelled
        as such in section 00. GR00T is the clean case — it ships only the 16 layers it uses and
        reads a single hidden state, so its documented read is exact.
      </p>
    </section>

    <footer>
      Full numeric tables: <span class="mono">asset/analysis/latent_compare/tables.md</span>.
      Method and rationale: <span class="mono">scripts/analysis/latent_compare/README.md</span>.
      Extended per-pool figures (image-only / text-only breakdowns) remain in
      <span class="mono">asset/analysis/latent_compare/figures/</span>.
    </footer>
  </main>
</div>

<script>
(function() {{
  var links = Array.prototype.slice.call(document.querySelectorAll('.rail a'));
  var sections = links.map(function(a) {{
    return document.querySelector(a.getAttribute('href'));
  }}).filter(Boolean);
  if (!('IntersectionObserver' in window) || !sections.length) return;
  var io = new IntersectionObserver(function(entries) {{
    entries.forEach(function(e) {{
      var link = links[sections.indexOf(e.target)];
      if (!link) return;
      if (e.isIntersecting) {{
        links.forEach(function(l) {{ l.classList.remove('active'); }});
        link.classList.add('active');
      }}
    }});
  }}, {{ rootMargin: '-15% 0px -70% 0px', threshold: 0 }});
  sections.forEach(function(s) {{ io.observe(s); }});
}})();
</script>
"""

if __name__ == "__main__":
    sys.exit(main())
