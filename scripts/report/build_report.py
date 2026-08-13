"""
scripts/report/build_report.py
──────────────────────────────
Builds the study report as a self-contained HTML file and prints it to PDF.

Organised by CLAIM, not by experiment. Each claim is a self-contained card: the
claim in one sentence, the evidence that supports it, the figure that carries it,
and the limitation that bounds it. That ordering exists so a drafter writing the
manuscript cannot pick up a claim without also picking up its bound — the failure
mode of a report that defers every caveat to a limitations section at the end.

Self-contained matters: figures are inlined as base64 data URIs so the HTML can
be emailed, uploaded or opened from any directory without carrying an asset
folder alongside it. asset/ is gitignored, so a report that linked to the PNGs
would render blank for anyone who cloned the repo.

PDF is produced by headless Chrome rather than a Python HTML-to-PDF library:
Chrome is already present on this machine, and it is the same engine that renders
the HTML, so the two outputs cannot drift apart in layout.

Numbers are hard-coded from the measured results rather than recomputed here.
This file is a presentation layer; recomputing statistics in a report builder
would create a second source of truth that can silently disagree with
RESULTS.md. Every figure IS read from disk, so figures cannot go stale.

    python scripts/report/build_report.py
"""
from __future__ import annotations

import base64
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The final numbered set from plots_paper.py, not the exploratory figures that
# accumulated under head_diagnostics/ as results arrived.
FIG = ROOT / "asset" / "analysis" / "paper_figures"
# report/ is at the repo root, NOT under asset/ — asset/ is gitignored, and a
# report that only exists on the machine that built it is not a deliverable.
OUT = ROOT / "report"
HTML = OUT / "vla_anatomy_report.html"
PDF = OUT / "vla_anatomy_report.pdf"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]


def img(name: str, caption: str) -> str:
    """Inline a figure as a data URI, or a visible placeholder if it is missing.

    A missing figure must be loud, not silent — a report that quietly drops a
    panel looks complete and is not.
    """
    p = FIG / name
    if not p.exists():
        return (f'<figure class="missing"><div>MISSING FIGURE: {name}<br>'
                f'<small>run scripts/analysis/plots_paper.py</small>'
                f'</div><figcaption>{caption}</figcaption></figure>')
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>')


CSS = """
:root{--ink:#12120f;--ink2:#55554f;--rule:#d9d9d3;--surf:#ffffff;
      --accent:#c0392b;--pre:#eb6834;--stock:#1baf7a;--band:#f5f5f1;}
*{box-sizing:border-box}
body{margin:0;background:var(--surf);color:var(--ink);
     font:15px/1.62 "Iowan Old Style","Palatino Linotype",Georgia,serif;}
.page{max-width:52rem;margin:0 auto;padding:3rem 2rem 4rem;}
h1{font-size:2rem;line-height:1.18;margin:0 0 .5rem;letter-spacing:-.015em}
h2{font-size:1.25rem;margin:2.6rem 0 .7rem;padding-bottom:.3rem;
   border-bottom:2px solid var(--ink);letter-spacing:-.01em}
.sub{color:var(--ink2);font-size:1.02rem;margin:0 0 .3rem}
.meta{color:var(--ink2);font-size:.82rem;font-family:ui-monospace,Consolas,monospace;
      border-top:1px solid var(--rule);padding-top:.7rem;margin-top:1.1rem}
p{margin:.7rem 0}
table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.89rem}
th,td{padding:.4rem .6rem;border-bottom:1px solid var(--rule);text-align:left;
      vertical-align:top}
th{font-weight:600;border-bottom:1.5px solid var(--ink);white-space:nowrap}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr.hl td{background:#fdf6ec}
figure{margin:1.2rem 0;padding:0}
figure img{width:100%;height:auto;display:block;border:1px solid var(--rule);border-radius:3px}
figcaption{color:var(--ink2);font-size:.82rem;margin-top:.42rem;line-height:1.45}
figure.missing div{border:2px dashed var(--accent);color:var(--accent);
    padding:2.4rem 1rem;text-align:center;font-family:ui-monospace,monospace;font-size:.85rem}
code{font-family:ui-monospace,Consolas,monospace;font-size:.87em;
     background:var(--band);padding:.08em .32em;border-radius:3px}
ul,ol{margin:.55rem 0;padding-left:1.25rem}
li{margin:.28rem 0}
.pre{color:var(--pre);font-weight:600}
.stock{color:var(--stock);font-weight:600}
.lede{font-size:1.05rem;color:var(--ink2);margin:1rem 0 1.5rem}

/* claim cards -- the report's organising unit */
.claim{border:1px solid var(--rule);border-left:5px solid var(--ink);
       border-radius:4px;padding:1.15rem 1.3rem 1.3rem;margin:1.5rem 0 2rem;
       background:var(--surf)}
.claim-id{font-family:ui-monospace,Consolas,monospace;font-size:.75rem;
          letter-spacing:.09em;text-transform:uppercase;color:var(--ink2);
          margin-bottom:.35rem}
.claim-title{font-size:1.12rem;line-height:1.33;font-weight:600;margin:0 0 .2rem}
.claim h4{font-size:.78rem;letter-spacing:.07em;text-transform:uppercase;
          color:var(--ink2);margin:1.1rem 0 .35rem;font-weight:600}
.bound{background:#fdf3f1;border-left:3px solid var(--accent);
       padding:.7rem .95rem;margin:1rem 0 0;font-size:.87rem}
.bound strong{color:var(--accent)}
.headline{background:var(--band);border-left:3px solid var(--ink);
          padding:.8rem 1rem;margin:1rem 0;font-size:.94rem}
.summary-tbl td:first-child{font-weight:600;white-space:nowrap}

@media print{
  .page{max-width:none;padding:0 .2in}
  h2{break-after:avoid}
  .claim,figure,table,.bound,.headline{break-inside:avoid}
  body{font-size:10.3pt}
}
@page{size:A4;margin:14mm 15mm}
"""

BODY = f"""
<div class="page">

<h1>What robot pretraining buys a vision-language-action model</h1>
<p class="sub">A controlled component-interaction study across two embodiments</p>
<p class="meta">Six arms · one frozen-backbone head · 2,600 closed-loop rollouts ·
LIBERO-Goal and ALOHA transfer-cube</p>

<p class="lede">Existing work studies how to choose data and how to choose a
backbone. This study asks what happens <em>between</em> the components: how
backbone pretraining, camera configuration, action space and task structure
interact to decide whether a policy actually succeeds in the loop.</p>

<h2>The six claims</h2>

<table class="summary-tbl">
<tr><th>#</th><th>Claim</th><th class="n">Key number</th></tr>
<tr><td>C1</td><td>A camera matters far more than a pretrained backbone</td><td class="n">+29.0 vs +2.5 pts</td></tr>
<tr><td>C2</td><td>Pretraining helps on one task and not the other, at every stage of training</td><td class="n">+9.5 pts, p = 0.0067</td></tr>
<tr><td>C3</td><td>The whole advantage sits in one transition</td><td class="n">p = 0.0009</td></tr>
<tr><td>C4</td><td>The gap is in failure rate, not kind — so offline metrics miss it</td><td class="n">&lt;2% vs 15.5%</td></tr>
<tr><td>C5</td><td>Language ablation has a floor that must be measured</td><td class="n">1.14–1.24×</td></tr>
<tr><td>C6</td><td>Visual reliance tracks what the action is defined relative to</td><td class="n">0.038 vs 0.11</td></tr>
</table>

<h2>Setup</h2>

<p>One identical 19.2M-parameter head — token adapter plus DiT flow-matching
decoder — is trained against <strong>frozen</strong> published backbones and
compared by closed-loop rollout. Every arm shares the head, the read depth, the
optimiser schedule and the evaluation protocol, so a difference in success rate is
attributable to the factor that moved.</p>

<p>The pair spans one lineage: stock <span class="stock">Qwen3-VL-2B</span> →
Cosmos-Reason2-2B → <span class="pre">GR00T&nbsp;N1.7</span>. Both hops are
verified real finetunes (584/625 and 476/493 tensors differ), so the comparison
covers the whole robot-pretraining treatment rather than one weak hop.</p>

<p>Two controls make the pair comparable. <strong>Layer matching:</strong> both
arms are read at layer 16 — GR00T's own <code>select_layer</code>, intermediate
for Qwen3-VL's 28 layers — because reading each at its own last layer would
compare depth 16 against depth 28 and attribute a depth effect to pretraining.
<strong>Final-norm correction:</strong> since layer 16 is final for one arm and
intermediate for the other, the language stack's final RMSNorm is applied to the
intermediate read, without which the pair would differ by normalisation rather
than by weights.</p>

<h2>C1 · A camera matters far more than a pretrained backbone</h2>

<div class="claim">
<div class="claim-id">Claim 1</div>
<p class="claim-title">On single-arm manipulation, observation configuration
outweighs backbone pretraining by an order of magnitude.</p>

<h4>Evidence</h4>
<table>
<tr><th>Arm</th><th>Backbone</th><th class="n">Cameras</th><th class="n">Success (n = 200)</th></tr>
<tr><td>exp03</td><td class="pre">GR00T N1.7</td><td class="n">1</td><td class="n">62.5%</td></tr>
<tr><td>exp04</td><td class="stock">Qwen3-VL-2B</td><td class="n">1</td><td class="n">68.0%</td></tr>
<tr class="hl"><td>exp05</td><td class="pre">GR00T N1.7</td><td class="n">2</td><td class="n"><strong>91.5%</strong></td></tr>
<tr class="hl"><td>exp06</td><td class="stock">Qwen3-VL-2B</td><td class="n">2</td><td class="n"><strong>89.0%</strong></td></tr>
</table>

<p>Adding the wrist camera: <strong>+29.0 points</strong> (GR00T) and
<strong>+21.0 points</strong> (Qwen3-VL), p&nbsp;&lt;&nbsp;10<sup>−7</sup>.
Swapping the backbone at the benchmark's own two-camera specification:
<strong>+2.5 points</strong>, p&nbsp;=&nbsp;0.40.</p>

<p>Supporting detail: the instruction is load-bearing for every arm (0/200 under a
swapped instruction, all ten tasks), and both two-view arms converge to the same
routing despite different backbone weights — wrist attention 51.6% vs 54.5%,
ablation cost ×11.94 vs ×11.64.</p>

<h4>Figure</h4>
{img("fig1_main_result.png",
     "The study in one figure. Left: on LIBERO-Goal a second camera is worth +29.0 "
     "points while swapping the backbone at that same observation spec is worth 2.5 "
     "(p = 0.40). Right: on bimanual ALOHA the identical backbone swap is worth +9.5 "
     "points (p = 0.0067). Error bars are Wilson 95% intervals.")}

<div class="bound"><strong>Bounded to:</strong> one task suite (LIBERO-Goal) and
one backbone pair. The single-view arms are reported for completeness but carry no
claim — single-view puts the pretrained arm outside its trained input
configuration while leaving the stock arm inside its own, an asymmetry pointing
the same way as the result.</div>
</div>

<h2>C2 · Pretraining's payoff is task-specific, at every stage of training</h2>

<div class="claim">
<div class="claim-id">Claim 2</div>
<p class="claim-title">Robot pretraining is not a general sample-efficiency prior:
it is decisive on one task and absent on the other, from the first checkpoint to
the last.</p>

<h4>Evidence — the bimanual task</h4>
<p>ALOHA transfer-cube was chosen as the pre-registered falsifier for C1's null.
Bimanual manipulation is inside GR00T's pretraining distribution, so the venue is
biased <em>toward</em> finding an effect; a null here would have been strong
evidence. It did not return a null.</p>

<table>
<tr><th>Arm</th><th class="n">Run 1</th><th class="n">Run 2</th><th class="n">Pooled (n = 400)</th><th class="n">Wilson 95% CI</th></tr>
<tr class="hl"><td class="pre">GR00T N1.7</td><td class="n">60.0%</td><td class="n">62.5%</td><td class="n"><strong>61.25%</strong></td><td class="n">[56.4, 65.9]</td></tr>
<tr><td class="stock">Qwen3-VL-2B</td><td class="n">49.0%</td><td class="n">54.5%</td><td class="n">51.75%</td><td class="n">[46.9, 56.6]</td></tr>
</table>

<p><strong>+9.5 points, z = 2.71, p = 0.0067</strong> — clears the Bonferroni bar
for the six comparisons in this study (0.05/6&nbsp;=&nbsp;0.0083), and the
intervals do not overlap. The two runs used disjoint seed ranges, so this is a
genuine replication rather than a resampling of the policy's own noise.</p>

<h4>Evidence — the whole training curve</h4>
<table>
<tr><th>ALOHA epoch</th><th class="n">25</th><th class="n">50</th><th class="n">100</th><th class="n">150</th><th class="n">200</th><th class="n">300</th></tr>
<tr><td class="pre">GR00T</td><td class="n">8.0%</td><td class="n">28.0%</td><td class="n">46.0%</td><td class="n">54.0%</td><td class="n">72.0%</td><td class="n">66.0%</td></tr>
<tr><td class="stock">Qwen3-VL</td><td class="n"><strong>0.0%</strong></td><td class="n"><strong>0.0%</strong></td><td class="n">22.0%</td><td class="n">28.0%</td><td class="n">48.0%</td><td class="n">46.0%</td></tr>
</table>
<table>
<tr><th>LIBERO epoch</th><th class="n">25</th><th class="n">50</th><th class="n">75</th><th class="n">100</th></tr>
<tr><td class="pre">GR00T</td><td class="n">61.0%</td><td class="n">83.0%</td><td class="n">85.0%</td><td class="n">85.0%</td></tr>
<tr><td class="stock">Qwen3-VL</td><td class="n"><strong>77.0%</strong></td><td class="n">78.0%</td><td class="n">89.0%</td><td class="n"><strong>95.0%</strong></td></tr>
</table>

<div class="headline">
On ALOHA the stock backbone <strong>cannot do the task at all</strong> before
epoch 100 — 0/50 at epochs 25 and 50 while the pretrained arm is at 28% (McNemar
p&nbsp;=&nbsp;0.0001) — and needs ~2× the epochs to reach any given rate. On
LIBERO the <em>stock</em> arm leads at epoch 25 by 16 points. The early-training
effect, the largest in the study, does not transfer.
</div>

<h4>Figure</h4>
{img("fig3_checkpoint_ladders.png",
     "Both checkpoint ladders on a common axis. Left: on bimanual ALOHA the pretrained "
     "arm leads at every checkpoint. Right: on single-arm LIBERO-Goal it leads at none. "
     "Paired seeds throughout; LIBERO's 50 fixed initial states make its pairing exact.")}

<div class="bound"><strong>Bounded to:</strong> the two testbeds differ on
<em>six axes simultaneously</em> — instruction variation (10 vs 1), degrees of
freedom (7 vs 14), arm count, action space, camera count, and distribution match
to GR00T's pretraining corpus. The DOF hypothesis, the bimanual hypothesis and the
distribution-match hypothesis all predict what was observed, and this design
cannot separate them. Naming bimanual coordination as <em>the</em> cause is an
interpretation, not a result.
<br><br>
Two further cautions: on LIBERO neither nominally significant point
(p&nbsp;=&nbsp;0.0195, 0.0213) survives Bonferroni over four comparisons, so the
claim there is the negative one. And both ladders disagree with their own headline
levels — read the ladders for <em>shape</em>, the pooled runs for
<em>magnitude</em>.</div>
</div>

<h2>C3 · The whole advantage sits in one transition</h2>

<div class="claim">
<div class="claim-id">Claim 3</div>
<p class="claim-title">Pretraining does not raise general competence — it raises
one conditional probability.</p>

<h4>Evidence</h4>
<p>ALOHA scores contact / lift / handover / success. <code>max_reward == 3</code>
never occurs in 800 episodes, so once the receiving gripper touches the cube the
episode always completes, and the ladder is contact → lift → handover.</p>

<table>
<tr><th>Stage</th><th class="n">GR00T</th><th class="n">Qwen3-VL</th><th class="n">Δ</th><th class="n">p</th></tr>
<tr><td>P(contact)</td><td class="n">89.2%</td><td class="n">92.5%</td><td class="n">−3.3</td><td class="n">0.11</td></tr>
<tr><td>P(lift | contact)</td><td class="n">95.8%</td><td class="n">93.8%</td><td class="n">+2.0</td><td class="n">0.22</td></tr>
<tr class="hl"><td><strong>P(handover | lift)</strong></td><td class="n"><strong>71.6%</strong></td><td class="n"><strong>59.7%</strong></td><td class="n"><strong>+12.0</strong></td><td class="n"><strong>0.0009</strong></td></tr>
</table>

<p>Both early stages run slightly <em>against</em> the pretrained arm, and the
localisation is tighter (p&nbsp;=&nbsp;0.0009) than the top-line result
(p&nbsp;=&nbsp;0.0067). Decomposing success along the environment's own reward
ladder converts an aggregate difference into a mechanism at no additional rollout
cost — a method other closed-loop studies can adopt directly.</p>

<h4>Figure</h4>
{img("fig2_stage_decomposition.png",
     "Left: both arms reach contact and lift at the same rate — the curves separate only "
     "at the final transition. Right: the same data as conditional probabilities. "
     "P(handover | lift) carries the entire gap.")}

<div class="bound"><strong>Bounded to:</strong> one task. Whether the pattern
generalises to other multi-stage manipulation tasks is untested.</div>
</div>

<h2>C4 · The gap is in failure rate, not failure kind</h2>

<div class="claim">
<div class="claim-id">Claim 4</div>
<p class="claim-title">Offline metrics cannot rank these policies — and the
instrumented rollouts show exactly why.</p>

<h4>Evidence — the blindness</h4>
<table>
<tr><th>Measure (stock − pretrained, % of pretrained)</th><th class="n">Δ</th></tr>
<tr><td>velocity loss (overall)</td><td class="n">+0.1%</td></tr>
<tr><td>velocity loss (handover phase)</td><td class="n">+1.9%</td></tr>
<tr><td>open-loop action error (nMAE, 14 dims)</td><td class="n">−0.4%</td></tr>
<tr><td>PE sensitivity</td><td class="n">+1.1%</td></tr>
<tr><td>attention mass on image</td><td class="n">−5.2%</td></tr>
<tr class="hl"><td><strong>closed-loop success rate</strong></td><td class="n"><strong>−15.5%</strong></td></tr>
</table>

<p>Four accuracy measures differ by under 2% against a gap the rollouts resolve at
p&nbsp;=&nbsp;0.0067. The training objective <em>does</em> register that the
handover window is hard — mid-phase loss is 1.8× the late-phase value for both
arms — but not that one arm is worse at it.</p>

<h4>Evidence — the reason</h4>
<p>Instrumented rollouts (n = 200 per arm) log per-step 14-DOF state, commanded
action, the reward timeline and per-replan attention. Arm identity was
<em>measured</em>, not assumed: the picking arm is whichever block moves first in
successful episodes. Classifying episodes that lifted the cube but failed the
handover:</p>

<table>
<tr><th>Failure class</th><th class="n">GR00T</th><th class="n">Qwen3-VL</th><th class="n">p</th></tr>
<tr><td>premature receiver close</td><td class="n">59.6% (31)</td><td class="n">59.4% (41)</td><td class="n"><strong>0.98</strong></td></tr>
<tr><td>receiver never engaged</td><td class="n">19.2% (10)</td><td class="n">26.1% (18)</td><td class="n">0.38</td></tr>
<tr><td>grasp lost after lift</td><td class="n">21.2% (11)</td><td class="n">14.5% (10)</td><td class="n">0.34</td></tr>
<tr class="hl"><td><strong>total lifted-but-failed</strong></td><td class="n"><strong>52</strong></td><td class="n"><strong>69</strong></td><td class="n">—</td></tr>
</table>

<div class="headline">
The failure <strong>mix is statistically identical; only the count differs</strong>
— 33% more of the same failure. Two policies that fail the same way, differing
only in how often they enter an unrecoverable configuration, leave
<strong>no per-step action-error signature</strong>. The difference is a
compounding closed-loop property, not a per-step accuracy property — exactly what
an error averaged over a 16-step chunk and 2,000 frames cannot see.
</div>

<p>Attention was recorded <strong>during rollout</strong>, not on dataset frames:
dataset frames are successful expert demonstrations and cannot be linked to the
policy's own failures. Over the five replans after the lift the stock arm
allocates less mass to image tokens — 0.8407 vs 0.8663 in successful episodes
(Welch t&nbsp;=&nbsp;9.01) and 0.8380 vs 0.8546 in failed ones
(t&nbsp;=&nbsp;5.24), both p&nbsp;&lt;&nbsp;10<sup>−4</sup>.</p>

<h4>Figures</h4>
{img("fig4_offline_blindness.png",
     "Every offline diagnostic as a between-arm difference, on the same axis as the "
     "closed-loop result. Four accuracy measures sit inside the ±2% band.")}

{img("fig5_failure_taxonomy.png",
     "Instrumented rollouts. Left: the failure mix is identical. Centre: only the count "
     "differs. Right: image attention during the handover window, split by outcome so "
     "the comparison cannot be an artifact of failed episodes running longer.")}

<div class="bound"><strong>Bounded to:</strong> these are correlational
measurements on frozen checkpoints; they locate where two policies differ but
cannot prove the difference causes the gap. The absolute class proportions depend
on the gripper-closure threshold, though the between-arm comparison — 59.6% vs
59.4% — does not. And because the attention difference appears in successful
episodes too, it is a stable policy-level habit, <em>not</em> a signature of
impending failure; it must not be described as the policy recognising
anything.</div>
</div>

<h2>C5 · Language ablation has a floor that must be measured</h2>

<div class="claim">
<div class="claim-id">Claim 5</div>
<p class="claim-title">Zeroing text tokens costs something even when the text
carries no information — so ablation ratios must be read against that floor, not
against 1.0.</p>

<h4>Evidence</h4>
<p>ALOHA has a single fixed instruction, so zeroing its text tokens removes no
task-discriminative information. Loss nonetheless rises <strong>1.14×</strong>
(GR00T) and <strong>1.24×</strong> (Qwen3-VL). That residual is the cost of an
off-distribution perturbation alone — a control condition an ablation on a
multi-instruction dataset cannot supply for itself.</p>

<table>
<tr><th>Arm</th><th>Instructions</th><th class="n">Ablation ratio</th></tr>
<tr class="hl"><td class="pre">GR00T · ALOHA</td><td>1 (fixed)</td><td class="n">1.14× — <em>floor</em></td></tr>
<tr class="hl"><td class="stock">Qwen3-VL · ALOHA</td><td>1 (fixed)</td><td class="n">1.24× — <em>floor</em></td></tr>
<tr><td class="stock">Qwen3-VL · 2 views</td><td>10</td><td class="n">5.39×</td></tr>
<tr><td class="pre">GR00T · 2 views</td><td>10</td><td class="n">6.12×</td></tr>
<tr><td class="stock">Qwen3-VL · 1 view</td><td>10</td><td class="n">7.05×</td></tr>
<tr><td class="pre">GR00T · 1 view</td><td>10</td><td class="n">7.41×</td></tr>
</table>

<p>A second reading falls out of the same table: the two-view arms depend on
language measurably <em>less</em> than the one-view arms (5.39×/6.12× against
7.05×/7.41×), consistent with a wrist camera recovering information the
instruction previously had to supply.</p>

<h4>Figure</h4>
{img("fig6_text_ablation_floor.png",
     "Text-ablation ratio across all six arms. The grey band is the floor measured on "
     "ALOHA's single fixed instruction — the cost of the perturbation alone.")}

<div class="bound"><strong>Bounded to:</strong> the floor is measured on one
dataset and one head; its magnitude on other setups is unknown. The
methodological point — that a floor exists and is not 1.0 — does not depend on its
exact value.</div>
</div>

<h2>C6 · Visual reliance tracks what the action is defined relative to</h2>

<div class="claim">
<div class="claim-id">Claim 6</div>
<p class="claim-title">A policy leans on image position when the action is a
displacement it cannot infer from one frame, and not when the action is a
destination the frame already implies.</p>

<h4>Evidence</h4>
<p>Zeroing the 2D positional encoding shifts actions by
<strong>0.097–0.111</strong> under end-effector control but only
<strong>0.038</strong> under 14-DOF joint-space control with full
proprioception.</p>

<p>The obvious objection is that ALOHA's scene is simply visually simpler. A
gradient or Jacobian analysis cannot settle that — an impoverished scene produces
low image sensitivity under either explanation. A probe of what the tokens
<em>contain</em> can:</p>

<table>
<tr><th>Arm</th><th class="n">Image tokens</th><th class="n">R²(arm state)</th><th class="n">R²(action)</th></tr>
<tr><td>LIBERO GR00T 2-view</td><td class="n">128</td><td class="n">0.832</td><td class="n">−0.215</td></tr>
<tr><td>LIBERO Qwen3-VL 2-view</td><td class="n">128</td><td class="n">0.828</td><td class="n">−0.192</td></tr>
<tr><td>ALOHA GR00T</td><td class="n">54</td><td class="n">0.817</td><td class="n">0.771</td></tr>
<tr><td>ALOHA Qwen3-VL</td><td class="n">54</td><td class="n">0.791</td><td class="n">0.743</td></tr>
</table>

<div class="headline">
Arm configuration is <strong>equally readable on both testbeds</strong> (0.830 vs
0.804) — and ALOHA reaches that from <strong>54 image tokens against LIBERO's
128</strong>. The scene is not information-poor, so "visually simpler" does not
explain the lower PE sensitivity.
<br><br>
The second column identifies the operative variable. LIBERO commands end-effector
<em>deltas</em>, which one frame does not determine (R² −0.20); ALOHA commands
<em>absolute joint targets</em>, which sit close to the pose the image already
encodes (R² +0.76). The driver is what the action is defined relative to — not
degrees of freedom, not arm count.
</div>

<h4>Figure</h4>
{img("fig7_action_space.png",
     "Left: PE sensitivity across all six arms. Centre: arm configuration is equally "
     "readable from image tokens on both testbeds, and ALOHA does it with fewer tokens. "
     "Right: action recoverability from a single frame, the variable that separates them.")}

<div class="bound"><strong>Bounded to:</strong> the action-recoverability contrast
is <em>partly definitional</em> — absolute targets are near the current state by
construction — so it is reported as <em>explaining</em> the PE result rather than
as independent evidence for it. Token geometry also differs between the testbeds
(54 vs 128) and is not separately controlled.</div>
</div>

<h2>Validity</h2>

<p><strong>The harness is correct.</strong> With the policy removed from the loop
and the demonstrations' own actions replayed through the same env construction,
success detector and step budget, LIBERO replays at 90.0% (no settling) and 92.0%
(5 settling steps) — the band published methods report, so this is not a stricter
harness. It is a replay rate, not a policy ceiling: the two-view arms exceed it on
two tasks.</p>

<p><strong>Training budget is not the constraint.</strong> Each arm takes 53,760
optimizer steps at batch 128 — 7.2× more samples and ~34× more passes over the
data than a published LeRobot-family recipe on LIBERO. Validation flattens after
epoch ~75 and open-loop action correlation reaches 0.979–0.997.</p>

<p><strong>Instrumentation did not perturb the policies.</strong> The traced run
reproduced 60.5% / 51.5% against the original 60.0% / 49.0% on the same seeds.</p>

<h2>What this study cannot settle</h2>

<ul>
<li><strong>Two backbones, one lineage.</strong> Every backbone claim rests on one pair.</li>
<li><strong>The six-axis confound</strong> (see C2) — two testbeds cannot separate six covarying differences.</li>
<li><strong>One head architecture.</strong> "Backbone barely matters on LIBERO" may be specific to a head with this capacity and cross-attention design.</li>
<li><strong>Correlational diagnostics.</strong> A causal test would swap handover-window tokens between arms.</li>
<li><strong>Simulation only.</strong> No physical robot.</li>
<li><strong>Nulls are not equivalence.</strong> At n&nbsp;=&nbsp;200 per LIBERO condition the standard error is 3.3 points.</li>
</ul>

<h2>Summary</h2>

<div class="headline">
Robot pretraining of a frozen VLM backbone is <strong>not a general
sample-efficiency prior</strong>. Its payoff is task-specific, and where it fails
it fails at every point in training rather than only at convergence.
<br><br>
On single-arm pick-and-place at the benchmark's own two-camera specification it
contributes nothing measurable — 2.5 points at p&nbsp;=&nbsp;0.40, and no
advantage at any checkpoint — while a second camera contributes +21 to +29. On
bimanual handover it is the difference between 0% and 28% success at 50 epochs,
halves the epochs to any target, and leaves a replicated +9.5-point advantage that
localises to a single transition.
<br><br>
Throughout, the offline metrics these systems are trained on and selected by
failed to rank the resulting policies — and the instrumented rollouts show why:
the policies fail in the same way, differing only in how often.
</div>

<p class="meta">Full design, statistics and defect log: <code>RESULTS.md</code> ·
Framing and component ledger: <code>OBJECTIVE.md</code> ·
Figures: <code>scripts/analysis/plots_paper.py</code></p>

</div>
"""


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)

    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>What robot pretraining buys a VLA</title>'
            f'<style>{CSS}</style></head><body>{BODY}</body></html>')
    HTML.write_text(html, encoding="utf-8")
    print(f"  wrote {HTML}  ({HTML.stat().st_size/1024:.0f} KB)")

    chrome = next((c for c in CHROME_CANDIDATES if Path(c).exists()), None) \
        or shutil.which("chrome") or shutil.which("msedge")
    if not chrome:
        print("  [warn] no Chrome/Edge found — HTML written, PDF skipped")
        return 0

    # --no-pdf-header-footer suppresses Chrome's default URL/date furniture, which
    # would otherwise stamp a file:// path across the top of every page.
    cmd = [chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={PDF}", HTML.as_uri()]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if PDF.exists():
        print(f"  wrote {PDF}  ({PDF.stat().st_size/1024:.0f} KB)")
    else:
        print(f"  [warn] PDF not produced (exit {r.returncode})")
        print((r.stderr or "").strip()[:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
