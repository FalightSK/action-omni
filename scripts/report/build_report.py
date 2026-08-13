"""
scripts/report/build_report.py
──────────────────────────────
Builds the study report as a self-contained HTML file and prints it to PDF.

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
FIG = ROOT / "asset" / "analysis" / "head_diagnostics" / "aloha" / "figures"
# report/ is at the repo root, NOT under asset/ — asset/ is gitignored, and a
# report that only exists on the machine that built it is not a deliverable.
# Both outputs are self-contained (figures inlined), so two tracked files carry
# the whole thing at ~1.2 MB.
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
                f'<small>run scripts/analysis/plots_head_diagnostics_aloha.py</small>'
                f'</div><figcaption>{caption}</figcaption></figure>')
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>')


CSS = """
:root{--ink:#12120f;--ink2:#55554f;--rule:#d9d9d3;--surf:#ffffff;
      --accent:#c0392b;--pre:#eb6834;--stock:#1baf7a;--band:#f4f4f0;}
*{box-sizing:border-box}
body{margin:0;background:var(--surf);color:var(--ink);
     font:15px/1.62 "Iowan Old Style","Palatino Linotype",Georgia,serif;}
.page{max-width:52rem;margin:0 auto;padding:3.2rem 2rem 4rem;}
h1{font-size:2.05rem;line-height:1.18;margin:0 0 .5rem;letter-spacing:-.015em}
h2{font-size:1.32rem;margin:2.9rem 0 .7rem;padding-bottom:.32rem;
   border-bottom:2px solid var(--ink);letter-spacing:-.01em}
h3{font-size:1.06rem;margin:1.9rem 0 .5rem;color:var(--ink)}
.sub{color:var(--ink2);font-size:1.02rem;margin:0 0 .3rem}
.meta{color:var(--ink2);font-size:.83rem;font-family:ui-monospace,Consolas,monospace;
      border-top:1px solid var(--rule);padding-top:.7rem;margin-top:1.1rem}
p{margin:.72rem 0}
table{border-collapse:collapse;width:100%;margin:1.05rem 0;font-size:.9rem}
th,td{padding:.42rem .6rem;border-bottom:1px solid var(--rule);text-align:left;
      vertical-align:top}
th{font-weight:600;border-bottom:1.5px solid var(--ink);white-space:nowrap}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr.hl td{background:#fdf6ec}
figure{margin:1.5rem 0;padding:0}
figure img{width:100%;height:auto;display:block;border:1px solid var(--rule);border-radius:3px}
figcaption{color:var(--ink2);font-size:.83rem;margin-top:.45rem;line-height:1.45}
figure.missing div{border:2px dashed var(--accent);color:var(--accent);
    padding:2.4rem 1rem;text-align:center;font-family:ui-monospace,monospace;font-size:.85rem}
.key{background:var(--band);border-left:3px solid var(--ink);
     padding:.85rem 1.05rem;margin:1.15rem 0;font-size:.95rem}
.key strong{letter-spacing:.01em}
.warn{background:#fdf3f1;border-left:3px solid var(--accent);
      padding:.85rem 1.05rem;margin:1.15rem 0;font-size:.92rem}
code{font-family:ui-monospace,Consolas,monospace;font-size:.87em;
     background:var(--band);padding:.08em .32em;border-radius:3px}
ul,ol{margin:.6rem 0;padding-left:1.3rem}
li{margin:.3rem 0}
.pre{color:var(--pre);font-weight:600}
.stock{color:var(--stock);font-weight:600}
.lede{font-size:1.06rem;color:var(--ink2);margin:1rem 0 1.6rem}
@media print{
  .page{max-width:none;padding:0 .2in}
  h2{break-after:avoid} h3{break-after:avoid}
  figure,table,.key,.warn{break-inside:avoid}
  body{font-size:10.5pt}
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

<div class="key">
<strong>Four findings.</strong>
<ol>
<li><strong>Cameras dominate backbones.</strong> Adding a wrist camera is worth
+21 to +29 points; swapping a robot-pretrained backbone for its stock root is
worth 2.5 points (p&nbsp;=&nbsp;0.40).</li>
<li><strong>That null is bounded.</strong> On a bimanual task the same backbone
swap is worth <strong>+9.5 points (p&nbsp;=&nbsp;0.0067)</strong>, and the entire
gap is one transition — P(handover&nbsp;|&nbsp;lift), p&nbsp;=&nbsp;0.0009.</li>
<li><strong>Pretraining buys ~2× training efficiency on the bimanual task</strong>
and the gap does not close by epoch 300 — but the same ladder on LIBERO shows no
advantage at any checkpoint, so this is <em>not</em> a general
sample-efficiency prior.</li>
<li><strong>Offline metrics cannot rank these policies.</strong> On the bimanual
pair they are not merely weak predictors; they carry no signal at all.</li>
</ol>
</div>

<h2>1. Design</h2>

<p>One identical 19.2M-parameter head — token adapter plus DiT flow-matching
decoder — is trained against <strong>frozen</strong> published backbones and
compared by closed-loop rollout. Every arm shares the head, the read depth, the
optimiser schedule and the evaluation protocol, so a difference in success rate
is attributable to the factor that moved.</p>

<p>The pair spans one lineage: stock <span class="stock">Qwen3-VL-2B</span> →
Cosmos-Reason2-2B → <span class="pre">GR00T&nbsp;N1.7</span>. Both hops are
verified real finetunes (584/625 and 476/493 tensors differ), so the comparison
covers the whole robot-pretraining treatment rather than one weak hop.</p>

<h3>Two controls that make the comparison honest</h3>
<p><strong>Layer matching.</strong> Both arms are read at layer 16 — GR00T's own
<code>select_layer</code>, intermediate for Qwen3-VL's 28 layers. Reading each at
its own last layer would compare depth 16 against depth 28 and attribute a depth
effect to pretraining.</p>
<p><strong>Final-norm correction.</strong> Because layer 16 is final for one arm
and intermediate for the other, the language stack's final RMSNorm is applied to
the intermediate read. Without it the pair would differ by normalisation rather
than by weights.</p>

<h2>2. Cameras dominate backbones</h2>

<p>LIBERO-Goal: 10 goals over one fixed scene, 200 rollouts per arm.</p>

<table>
<tr><th>Arm</th><th>Backbone</th><th class="n">Views</th><th class="n">Val loss</th>
    <th class="n">Success</th><th class="n">Swapped</th></tr>
<tr><td>exp03</td><td class="pre">GR00T N1.7</td><td class="n">1</td><td class="n">0.0316</td><td class="n">62.5%</td><td class="n">0.0%</td></tr>
<tr><td>exp04</td><td class="stock">Qwen3-VL-2B</td><td class="n">1</td><td class="n">0.0332</td><td class="n">68.0%</td><td class="n">0.0%</td></tr>
<tr class="hl"><td>exp05</td><td class="pre">GR00T N1.7</td><td class="n">2</td><td class="n">0.0352</td><td class="n"><strong>91.5%</strong></td><td class="n">0.0%</td></tr>
<tr class="hl"><td>exp06</td><td class="stock">Qwen3-VL-2B</td><td class="n">2</td><td class="n">0.0352</td><td class="n"><strong>89.0%</strong></td><td class="n">0.0%</td></tr>
</table>

<p>The camera effect is <strong>+29.0 points</strong> (GR00T) and
<strong>+21.0 points</strong> (Qwen3-VL), p&nbsp;&lt;&nbsp;10⁻⁷. The backbone
effect, measured at the benchmark's own two-camera specification, is
<strong>2.5 points</strong> at p&nbsp;=&nbsp;0.40. An order of magnitude separates
the observation configuration from the backbone.</p>

<p>The instruction is load-bearing for every arm: <strong>0/200</strong> under a
swapped instruction, all ten tasks. And both two-view arms converge to the same
solution despite different backbone weights — wrist attention 51.6% vs 54.5%,
ablation cost ×11.94 vs ×11.64.</p>

<h2>3. The null does not survive a bimanual task</h2>

<p>ALOHA transfer-cube was chosen as the pre-registered falsifier. Bimanual
manipulation is inside GR00T's pretraining distribution, so this venue is biased
<em>toward</em> finding a pretraining effect; a null here would have been strong
evidence. It did not return a null.</p>

<table>
<tr><th>Arm</th><th class="n">Run 1</th><th class="n">Run 2</th><th class="n">Pooled (n=400)</th><th class="n">Wilson 95% CI</th></tr>
<tr class="hl"><td class="pre">GR00T N1.7</td><td class="n">60.0%</td><td class="n">62.5%</td><td class="n"><strong>61.25%</strong> (245/400)</td><td class="n">[56.4, 65.9]</td></tr>
<tr><td class="stock">Qwen3-VL-2B</td><td class="n">49.0%</td><td class="n">54.5%</td><td class="n">51.75% (207/400)</td><td class="n">[46.9, 56.6]</td></tr>
</table>

<p><strong>+9.5 points, z&nbsp;=&nbsp;2.71, p&nbsp;=&nbsp;0.0067</strong> — clears
the Bonferroni bar for the six comparisons in this study (0.05/6&nbsp;=&nbsp;0.0083),
and the intervals do not overlap. The two runs used disjoint seed ranges, so this
is a genuine replication rather than a resampling of the policy's own noise.</p>

<h3>The gap is one transition, not general competence</h3>

<p>ALOHA scores touch / lift / handover / success. <code>max_reward == 3</code>
never occurs in 800 episodes, so once the receiving gripper contacts the cube the
episode always completes, and the ladder is touch → lift → handover.</p>

<table>
<tr><th>Stage</th><th class="n">GR00T</th><th class="n">Qwen3-VL</th><th class="n">Δ</th><th class="n">p</th></tr>
<tr><td>P(touch)</td><td class="n">89.2%</td><td class="n">92.5%</td><td class="n">−3.3</td><td class="n">0.11</td></tr>
<tr><td>P(lift | touch)</td><td class="n">95.8%</td><td class="n">93.8%</td><td class="n">+2.0</td><td class="n">0.22</td></tr>
<tr class="hl"><td><strong>P(handover | lift)</strong></td><td class="n"><strong>71.6%</strong></td><td class="n"><strong>59.7%</strong></td><td class="n"><strong>+12.0</strong></td><td class="n"><strong>0.0009</strong></td></tr>
</table>

<p>Both early stages run slightly <em>against</em> the pretrained arm. This is not
a uniform competence advantage — it is localised to the one stage that bimanual
pretraining plausibly covers, and the localisation is tighter than the top-line
result.</p>

<h2>4. Speed or skill?</h2>

{img("figA5_ladder.png",
     "Closed-loop checkpoint ladder, paired seeds, 50 episodes per point. Left: the "
     "pretrained arm leads at every checkpoint and the stock arm scores 0/50 through "
     "epoch 50. Right: epochs required to reach a given success rate. Dashed lines mark "
     "the n=400 anchors — the ladder establishes the shape, the pooled runs the magnitude.")}

<table>
<tr><th class="n">Epoch</th><th class="n">25</th><th class="n">50</th><th class="n">100</th><th class="n">150</th><th class="n">200</th><th class="n">300</th></tr>
<tr><td class="pre">GR00T</td><td class="n">8.0%</td><td class="n">28.0%</td><td class="n">46.0%</td><td class="n">54.0%</td><td class="n">72.0%</td><td class="n">66.0%</td></tr>
<tr><td class="stock">Qwen3-VL</td><td class="n"><strong>0.0%</strong></td><td class="n"><strong>0.0%</strong></td><td class="n">22.0%</td><td class="n">28.0%</td><td class="n">48.0%</td><td class="n">46.0%</td></tr>
</table>

<ul>
<li>The stock backbone <strong>cannot do the task early at all</strong> — 0/50 at
epochs 25 and 50 while the pretrained arm is at 28% (McNemar p&nbsp;=&nbsp;0.0001).</li>
<li>Stock needs <strong>~2× the training</strong> to reach any given rate:
2.4× / 2.8× / 2.2× / 1.9× at the 20/30/40/46% targets.</li>
<li>The gap <strong>does not close</strong> by epoch 300.</li>
</ul>

<h3>The same ladder on LIBERO points the other way</h3>

{img("figA6_ladder_both.png",
     "Both checkpoint ladders on a common axis. Left: on bimanual ALOHA the pretrained "
     "arm leads at every checkpoint and the stock arm cannot do the task at all before "
     "epoch 100. Right: on single-arm LIBERO-Goal the stock arm leads at epoch 25 and "
     "again at epoch 100. The early-training effect does not transfer.")}

<table>
<tr><th class="n">Epoch</th><th class="n">25</th><th class="n">50</th><th class="n">75</th><th class="n">100</th></tr>
<tr><td class="pre">GR00T</td><td class="n">61.0%</td><td class="n">83.0%</td><td class="n">85.0%</td><td class="n">85.0%</td></tr>
<tr><td class="stock">Qwen3-VL</td><td class="n"><strong>77.0%</strong></td><td class="n">78.0%</td><td class="n">89.0%</td><td class="n"><strong>95.0%</strong></td></tr>
</table>

<p>n&nbsp;=&nbsp;100 per point, canonical condition only. Pairing is exact and
needs no seed offset — LIBERO supplies 50 <em>fixed</em> initial states per task,
so every snapshot of every arm saw identical starts.</p>

<p>At epoch 25 the <strong>stock</strong> arm leads by 16 points, the opposite of
ALOHA, where the pretrained arm led 8% to 0% at the same epoch and 28% to 0% by
epoch 50. <strong>ALOHA's early-training effect — the largest in this study — has
no LIBERO counterpart.</strong></p>

<div class="warn">
<strong>Neither significant point survives correction.</strong> Four comparisons,
Bonferroni bar 0.05/4&nbsp;=&nbsp;0.0125; the epoch-25 and epoch-100 gaps sit at
p&nbsp;=&nbsp;0.0195 and 0.0213. The defensible statement is the negative one: no
LIBERO checkpoint shows a pretraining advantage, and the two points that reach
nominal significance both favour the stock arm. A prediction was registered
before this run that the LIBERO curves would <em>converge early</em>; they do not
converge, they cross. The conclusion survives in a stronger form than predicted,
but that specific prediction failed.
</div>

<div class="warn">
<strong>Why the ladder's magnitude is not quotable.</strong> Its 50 paired scenes
are a fixed and slightly unrepresentative sample: at epoch 300 GR00T reads 66.0%
against the 61.25% anchor (+4.8) and Qwen3-VL reads 46.0% against 51.75% (−5.8).
Each deviation sits inside one standard error (~6.7 points), so nothing is broken
— but the ladder establishes the <em>shape</em> and the pooled n&nbsp;=&nbsp;400
runs establish the <em>magnitude</em> (+9.5 points). The +24-point ladder gap is
not the population gap.
<br><br>
The LIBERO ladder carries the same warning and disagrees with its headline in
<em>both</em> directions: at epoch 100 GR00T reads 85.0% against the 91.5% anchor
and Qwen3-VL 95.0% against 89.0%. Two causes — <code>best.pt</code> is epoch 111
for both arms, past the ladder's last snapshot, and the ladder runs 10
episodes/task against the headline's 20, so it sees half the initial states.
</div>

<h2>5. Offline metrics cannot rank these policies</h2>

{img("figA2_dissociation.png",
     "Every offline diagnostic expressed as a between-arm difference, on the same axis "
     "as the closed-loop result. Four accuracy measures are flat against a gap resolved "
     "at p = 0.0067. Attention allocation is the only offline quantity that moves.")}

<p>This is stronger than "offline ranks them wrongly." On the bimanual pair,
offline carries <strong>no signal</strong> about a difference the rollouts resolve
decisively. Loss by episode phase shows the objective <em>does</em> know the
handover is hard — 0.0295 vs 0.0161 late — it simply does not know that one arm
is worse at it.</p>

<p>The one offline quantity that moves is attention <em>allocation</em>: the stock
arm spends 5.2% less mass on image tokens and correspondingly more on text tokens
that carry zero information on this task. Correlational, one testbed — the only
surviving mechanistic candidate, not a finding.</p>

{img("figA4_phase_perdim.png",
     "Left: velocity loss by episode phase. The handover window is hardest for both arms "
     "and equally so. Right: per-dimension open-loop error across all 14 joints — no "
     "arm-specific deficit survives.")}

<h2>6. Two measurement lessons</h2>

<h3>Text ablation has a floor, and it must be measured</h3>

{img("figA1_text_ablation_floor.png",
     "Text-ablation ratio across all six arms. ALOHA has a single fixed instruction, so "
     "its 1.14–1.24x is the cost of the perturbation alone — the floor against which the "
     "LIBERO ratios must be read.")}

<p>Zeroing the text tokens of a <em>constant</em> instruction removes no task
information, yet loss still rises 1.14×–1.24×. That residual is the cost of an
off-distribution perturbation — the metric's floor, and a control condition an
ablation on a multi-instruction dataset cannot supply for itself. Reporting an
ablation ratio without its floor overstates the effect.</p>

<p>Note also that the two-view arms depend on text <em>less</em> than the one-view
arms (5.39×/6.12× against 7.05×/7.41×): with a wrist camera available, some of
what the instruction supplied becomes recoverable from vision.</p>

<h3>Action space determines how much the policy uses vision</h3>

{img("figA3_routing.png",
     "Left and centre: cross-attention mass and entropy across the six DiT blocks. "
     "Right: PE sensitivity across all six arms — joint-space control needs image "
     "position far less than end-effector control does.")}

<p>Zeroing the 2D positional encoding shifts actions by 0.097–0.111 under
end-effector control but only <strong>0.038</strong> under joint control with full
14-DOF proprioception. Where the state already determines the arm configuration,
the head sources far less from image position.</p>

<p>This was predicted the wrong way round before measurement — the pre-registered
expectation was that bimanual handover would be <em>more</em> position-sensitive,
on the reasoning that handover is spatial registration. It is not; the action
space dominates.</p>

<h2>7. Validity</h2>

<p><strong>The harness is correct.</strong> With the policy removed from the loop
and the demonstrations' own actions replayed through the same env construction,
success detector and step budget, LIBERO replays at 90.0% (no settling) and 92.0%
(5 settling steps). That coincides with the band published methods report, so this
is not a stricter harness. It is a replay rate, not a policy ceiling — the
two-view arms exceed it on two tasks.</p>

<p><strong>Training budget is not the constraint.</strong> Each arm takes 53,760
optimizer steps at batch 128 — 7.2× more samples and ~34× more passes over the
data than a published LeRobot-family recipe on LIBERO. Validation flattens after
epoch ~75 and open-loop action correlation reaches 0.979–0.997.</p>

<h2>8. What this study cannot settle</h2>

<div class="warn">
<strong>The six-axis confound.</strong> LIBERO-Goal and ALOHA differ
simultaneously in instruction variation (10 vs 1), degrees of freedom (7 vs 14),
arm count, action space (end-effector vs joint), camera count (2 vs 1), and
distribution match to GR00T's pretraining data. Two testbeds cannot separate six
axes. The DOF hypothesis, the bimanual hypothesis and the distribution-match
hypothesis all predict exactly what was observed, and this study cannot
distinguish among them. Any statement naming one axis as <em>the</em> cause is an
interpretation, not a result.
</div>

<ul>
<li><strong>Two backbones, one lineage.</strong> Every backbone claim rests on one pair.</li>
<li><strong>One head architecture.</strong> "Backbone barely matters on LIBERO" may
be specific to a head with this capacity and cross-attention design.</li>
<li><strong>Head diagnostics are correlational.</strong> They locate where two heads
differ on frozen checkpoints; they cannot prove that difference causes the
handover gap. A causal test would swap handover-window tokens between arms.</li>
<li><strong>Simulation only.</strong> No physical robot.</li>
<li><strong>Nulls are not equivalence.</strong> At n&nbsp;=&nbsp;200 per LIBERO
condition the standard error is 3.3 points, so a null means "no detectable
difference at this readout capacity," never "no difference."</li>
</ul>

<h2>9. Summary</h2>

<div class="key">
Robot pretraining of a frozen VLM backbone is <strong>not a general
sample-efficiency prior</strong>. Its payoff is task-specific, and on the testbed
where it fails it fails at every point in training, not only at convergence.
<br><br>
On single-arm pick-and-place at the benchmark's own two-camera specification it
contributes nothing measurable — 2.5 points at p&nbsp;=&nbsp;0.40 at convergence,
and no advantage at any checkpoint from epoch 25 onward — while a second camera
contributes +21 to +29. On bimanual handover it makes the difference between 0%
and 28% success at 50 epochs, cuts the epochs to any target by roughly half, and
leaves a replicated +9.5-point advantage that localises to a single transition
and does not close by epoch 300.
<br><br>
Throughout, offline velocity loss — the quantity these systems are trained on and
selected by — failed to rank the resulting policies, and on the bimanual pair
carried no signal about them at all.
</div>

<p class="meta">Full design, statistics and defect log: <code>RESULTS.md</code> ·
Framing and component ledger: <code>OBJECTIVE.md</code></p>

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
