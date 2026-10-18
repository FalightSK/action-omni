# Camera-ready manuscript outline

Updated: 18 October 2026 (Asia/Tokyo).

**Current detailed plan: [plan.md](plan.md).** This active outline replaces the pre-review writing guidelines retained below. The revised plan allows independent training replications, episode-disjoint validation, corrected analysis, and a time-gated Cosmos extension. Do not follow the historical "no new experiments" instruction or preserve its predetermined numerical conclusions.

## Working title

**An Empirical Study of Robot Pretraining in Frozen VLA Backbones**

Use neutral performance language. Do not imply a cost-benefit analysis or a general answer across backbone families and manipulation settings.

## Argument and research questions

The paper is a controlled empirical comparison of released frozen backbone endpoints under matched settings within each benchmark. Its conclusions must follow the revised results, including independent head-training variability. Do not assert in advance that robot pretraining is generally ineffective, universally useful, or confined to a proven causal mechanism.

1. How do the tested frozen endpoints compare in closed-loop performance across independently trained heads on ALOHA transfer-cube and two-camera LIBERO-Goal?
2. Do the selected offline validation measures reflect those observed closed-loop comparisons?

If the optional Cosmos experiment completes, discuss its position within the same lineage on ALOHA. It does not establish another independent pretraining family.

## Evidence and experiment priority

1. Correct the validation split, normalization provenance, training/evaluation seed reporting, and methods description.
2. Target three training seeds for each of the four primary benchmark/backbone configurations: 12 policies, 200 evaluation starts per policy.
3. Recompute paired statistics, training mean/sample SD, offline diagnostics, and same-cohort failure analysis.
4. Add compatible one-camera LIBERO controls after the primary comparisons. State their actual independent training count.
5. Add three Cosmos ALOHA seeds only if the measured workload plus contingency fits before the experiment cutoff.
6. Defer Push-T, Qwen3.5, additional lineages, full checkpoint ladders, and causal interventions.

Historical policies/results cannot be pooled with the new episode-holdout primary experiments. Do not copy old success rates, p-values, rollout counts, or training-efficiency conclusions into the new manuscript without eligible regenerated evidence.

## Six-page structure

| Content | Target pages | Purpose |
|---|---:|---|
| Introduction and related work | 1.00 | State the two specific questions and bounded contributions. |
| Methods and statistics | 1.25 | Describe actual per-benchmark heads/settings, episode split, training seeds, selection, rollout pairing, and uncertainty. |
| Results | 2.25 | Replicated backbone comparisons; eligible camera controls; observed ALOHA failures; offline comparison; optional completed Cosmos evidence. |
| Discussion, limitations, conclusion | 0.75 | Answer both questions directly and explain unresolved external validity and mechanism. |
| References | 0.75 | Retain verified, relevant primary references. |
| Total | 6.00 | Maintain the conference's hard page limit. |

Use a primary table with seed-specific rates and mean +/- sample SD. Prefer three principal visual elements: replicated performance, same-cohort failure patterns, and offline-versus-rollout comparisons. Label episode intervals separately from training variability.

## Required editorial corrections

- Distinguish independently trained heads from separate rollout batches; do not call scene resampling training replication.
- Report actual within-benchmark matched settings. Current ALOHA and LIBERO configurations do not have an identical 19.2M head or identical training recipe across benchmarks.
- Use paired tests/difference intervals. A nonsignificant result is not equivalence or a proven effect smaller than approximately seven points.
- A large camera effect does not establish power for a small backbone effect or prove an exclusive observability bottleneck.
- Describe the receiving gripper's behavior as an observed failure pattern. Do not claim that an exclusively visual cue or attention allocation causes it.
- Use the same evaluation cohort for success decomposition and failure-category accounting.
- Do not infer equal policies from similar routing diagnostics or equality from nonsignificant stage contrasts.
- Explain that action-error aggregation may dilute critical behavior differences; remove claims that arithmetic guarantees blindness.
- State whether the selected offline diagnostics reflected the new comparisons. Do not presume the original offline-mismatch result survives the revised validation protocol.
- Remove economic wording, unreproduced checkpoint-ladder conclusions, and unsupported generalization across pretraining families.
- Preserve limitations: few training seeds, one fixed demonstration split, one lineage, frozen backbones, simulation, cross-benchmark differences, and no causal intervention.

## Delivery checkpoints

- Editable manuscript: the author will supply the original source.
- Experiment cutoff: **20 October 2026, 20:59 JST**.
- Conference deadline: **21 October 2026, 20:59 JST**, equivalent to 20 October at 23:59 Anywhere on Earth.
- Final numerical, reference, and rendered-PDF checks follow the checklist in [plan.md](plan.md).

---

## Historical pre-review outline (superseded)

The text below is preserved only for provenance. Its instructions, numerical claims, title suggestions, and "final" status are historical and are not the active plan. In particular, it predates the reviewer-driven replication, validation, statistical, and causal-language corrections documented above.

<details>
<summary>Archived writing guideline - not current instructions</summary>

### Previous single-thesis writing guideline

Replaces the four-contribution structure of the first draft (`main.pdf`), and
the earlier two-thesis draft of this outline. **No new experiments.** Every
number below is already on disk.

---

## The thesis

**Robot pretraining of a frozen VLM backbone is not a general prior. Where it
helps, the help is narrow.**

That is the paper. The offline-blindness material is **not a second thesis** —
it is §V, and its job is to make the thesis trustworthy (see below).

### Framing rules

**Never write "pretraining does not help" unqualified.** Table II says otherwise
(+9.5, replicated, *p* = 0.0067) and a reviewer will find it. The sentence that
carries the intended meaning and survives review:

> Pretraining is not a general prior. On the language-conditioned benchmark it
> confers no advantage at any checkpoint; on the one task inside its pretraining
> distribution it raises exactly one of three transitions, while the other two
> run slightly against it.

The *narrowing* is the contribution. Framed this way ALOHA stops being a
counterexample to the thesis and becomes the evidence for it.

**The camera result is a yardstick, not a claim.** Its only job: establish that
the harness resolves a 21–29 point effect at n = 200, so the backbone null is a
property of the backbone and not of the measurement. Delete every "order of
magnitude" phrasing — a ratio whose denominator is a null is not a legal
operation, and §III-D already concedes the null is unresolved.

**§V exists to protect the null, not to make a second claim.** The anticipated
reviewer objection to §IV-B is *"maybe you just measured badly."* §V answers it:
the cheap instruments report **no difference** even on ALOHA, where a 9.5-point
difference demonstrably exists — therefore their agreement on LIBERO carries no
information, and only the rollouts do. Write §V so it visibly serves that
purpose.

**Separate measured from inferred.** Two places explain *why*, and in both the
explanation interprets measurements rather than being one. Mark them — see
IV-B(3) and IV-C(3). One hedging clause each removes the paper's largest
overreach exposure.

---

## Section plan (6 pages, IEEE two-column)

| § | Content | Budget |
|---|---|---|
| I | Does robot pretraining pay, and where? | 0.5 col |
| II | Related work — 3 short paragraphs | 0.4 col |
| III | Design + statistics + harness validity | 1.2 col |
| IV | **The payoff is narrow** | 2.2 col |
| V | **Why the cheap instruments can't see it** | 1.0 col |
| VI | Limitations (5) + Conclusion | 0.7 col |

---

## §IV — The payoff is narrow

### IV-A Calibrating the scale (0.3 col)

Wrist camera: **+29.0** (GR00T), **+21.0** (Qwen3-VL), *p* < 10⁻⁷. One sentence
of purpose: *the harness resolves a 21–29 point effect at n = 200, so a null
below is informative rather than under-powered.*

### IV-B LIBERO-Goal — competitive, and why (0.9 col)

Result: 91.5% vs 89.0%, *p* = 0.40; the pretrained arm leads at **no**
checkpoint (ladder 61/83/85/85 vs 77/78/89/95, neither nominally significant
point surviving correction).

The draft asserts the null and stops. Four lines of support, in this order:

1. **The residual differences are idiosyncratic, not systematic.** Paired 2×2
   over the 50 fixed initial states: 165 both succeed, 4 both fail, 18 only
   GR00T, 13 only Qwen. 31 of 35 failures are disjoint. The arms are not
   ordered — they fail on *different* episodes. Strongest item; lead with it.
2. **The head reaches the same solution from either backbone.** Routing
   diagnostics: wrist-attention mass 51.6% vs 54.5%, wrist-ablation cost ×11.94
   vs ×11.64, text-ablation cost ×6.12 vs ×5.39, PE sensitivity 0.111 vs 0.097.
   Supporting sentence, already measured: backbone token representations are
   substantially shared — cross-backbone cosine 0.712 mean / 0.900 median,
   against 0.362 within-backbone across frames.
3. **Interpretation — the task's demand on the representation is met by both.**
   LIBERO-Goal is one kitchen scene; the representation must localize a few
   objects and discriminate ten instructions. The binding constraint turns out
   to be *observability* — the wrist view, +21/+29 — not representation quality.
   ⚠ **Mark as interpretation.** We did not probe whether instruction identity
   is equally decodable from both backbones' tokens. Write it as *"consistent
   with the task's representational demand being met by both backbones, which we
   infer rather than measure."* Guard against the weaker reading a reviewer will
   otherwise adopt — *"the backbone is irrelevant here"* — by pointing at the
   ablation costs in (2): both arms depend heavily on their tokens.
4. **Headroom, stated honestly.** Both arms sit at 89.0–91.5% against a 90–92%
   demonstration-replay rate, so the null is partly a property of a benchmark
   largely solved at two views. Do **not** call it a ceiling — two-view arms
   exceed the replay rate on two tasks.

Close with the bound: at n = 200 the SE on a difference is 3.3 points, so this
bounds the backbone effect below ≈7 points; it does not establish zero.

**Cut:** the per-task table. Per-task SE is 9.4 points at n = 20, so the
±20–25-point excursions are noise and the table invites over-reading. One
sentence replaces it: *per-task gaps up to ±25 points exist but do not
accumulate in any direction, and are within the ±2 SE expected across ten tasks.*

### IV-C ALOHA — where it pays, and what it buys (0.8 col)

Result: **+9.5** pooled (n = 400/arm), replicated across two disjoint seed runs,
*p* = 0.0067, clears Bonferroni 0.0083. Ladder: stock scores 0/50 at epochs 25
and 50 against 8/50 and 28/50 (McNemar *p* = 0.0001).

Narrow it immediately — the narrowing is the contribution:

1. **Not general competence.** Reward-ladder decomposition: P(contact) 89.2 vs
   92.5 (*p* = 0.11, **against** the pretrained arm), P(lift|contact) 95.8 vs
   93.8 (*p* = 0.22), P(handover|lift) 71.6 vs 59.7 (*p* = 0.0009). Two of three
   transitions do not favor pretraining, and the surviving one localizes more
   tightly than the top-line result.
2. **What the one transition is.** Failure taxonomy over post-lift episodes:
   receiving gripper never opens 14.5% vs **26.2%** (*p* = 0.0068); the other
   two modes run the other way (opened-never-closed 13.3 vs 9.3;
   closed-on-nothing 2.3 vs 4.7). The +11.7-point excess accounts for 11.9
   points of gap. The stock policy does not aim badly, mistime, or drop — it
   fails to *initiate the receive*.
3. **Interpretation — why that transition.** Contact and lift are driven by
   proprioception plus coarse object location. Opening the receiving gripper is
   cued *only* visually: the cube arriving in the receiver's workspace has no
   proprioceptive correlate, and ALOHA's single fixed instruction supplies no
   language cue. ⚠ **Mark as interpretation** — a reading of the decomposition
   and the taxonomy, not an independent measurement.
4. **The correlate, at its real strength.** Over the five post-lift replans the
   stock arm puts less attention mass on image tokens: 0.8407 vs 0.8663
   (successes), 0.8380 vs 0.8546 (failures). **Three sentences maximum.** State
   that it appears in successes too, so it is a policy-level disposition rather
   than a failure signature — a correlate, not a causal chain. **Drop the *t*
   statistics**; reporting *t* = 9.01 and then explaining it overstates
   independence is worse than reporting neither.

### IV-D The bound (0.3 col)

The six-axis confound — instruction variation, DOF, arm count, action space,
camera count, distribution match — stated **once**, here, and referenced from
Limitations rather than repeated. Bimanual coordination, DOF, and distribution
match all predict the observed pattern; this design cannot separate them, and
naming one would be interpretation, not result.

---

## §V — Why the cheap instruments can't see it (1.0 col)

**Do not claim "offline metrics don't predict success."** That is known,
already cited [9], and the strongest-looking evidence for it here — the
LIBERO validation-loss reversal — runs *across camera configurations*, where the
boring explanation (covariate shift: the wrist camera aids recovery from states
reached only in closed loop, which loss on expert demos cannot see) is standard
and will be raised.

The defensible and novel claim is narrower:

> **Offline metrics cannot adjudicate component comparisons.** Every instrument
> a practitioner would use to decide *"is the pretrained backbone better?"*
> reports no difference on the pair where a 9.5-point difference exists.
> Component ablations validated offline are therefore not reliable evidence
> about closed-loop behavior.

### V-A The unconfounded case (0.5 col)

Same head, same read depth, same input configuration, same training budget —
**only the backbone differs.** Velocity loss +0.1%, handover-phase loss +1.9%,
nMAE −0.4%, PE sensitivity +1.1%, attention mass −5.2%, against closed-loop
SR −15.5% relative. Add **open-loop action correlation 0.979–0.997** (already in
§III-C): near-perfect open-loop agreement alongside a 9.5-point closed-loop gap
is the cleanest one-line statement available.

Add a clarifying clause to the attention row: *(dataset frames; cf. §IV-C for
rollout)*. The draft reports attention as both a failed offline metric and a
working rollout correlate without telling the reader these are different
measurements.

### V-B Why the blindness is structural, not incidental (0.4 col)

The mechanism from IV-C is what makes the blindness explicable rather than
merely observed: one dimension of fourteen, held at a plausible constant value,
inside a chunk whose other thirteen dimensions are accurate. Chunk-averaged
error **cannot** express it. This is the contribution over the known
offline–online disconnect — not that a gap exists, but a concrete case where the
arithmetic of the metric guarantees the gap.

### V-C What this buys §IV-B (0.1 col)

Close the section by discharging the objection it exists to answer: because
these instruments agree on ALOHA where a real difference exists, their agreement
on LIBERO is uninformative, and the LIBERO null rests on the rollouts alone.

One-sentence aside, with its caveat attached, if space allows: on LIBERO the two
arms with the worst validation loss are the two with the highest success
(0.0352 → 91.5/89.0% vs 0.0316/0.0332 → 62.5/68.0%), though that contrast is
across camera configurations and admits a covariate-shift explanation.

### Cut from §V entirely

- The "probes don't rescue it" material. PE sensitivity, attention mass, and
  linear decodability were built to describe representations, not to predict
  success; faulting them for a job never claimed is a strawman.
- The reference-frame / R² ≈ +0.76 vs −0.2 discussion — cross-testbed, partly
  definitional by the draft's own admission, serves nothing. Retain one clause
  in Limitations: arm configuration is equally decodable from image tokens on
  both testbeds (R² 0.83 vs 0.80), so neither scene is information-poor.
- The "two-view arms depend on language less" rider — four numbers, no test, not
  on a common loss scale.

### Language-ablation floor — relocate, do not expand

Two sentences at the end of §III or as a footnote in §V-A: ALOHA has one fixed
instruction, so zeroing text removes no task-discriminative information, yet
loss rises 1.14×/1.24×. Ablation ratios must be read against that measured
floor, not against 1.0. It is a real methodological point but it belongs to
neither §IV nor §V's argument.

---

## §VI — Limitations (7 → 5) + Conclusion

1. Two backbones, one lineage.
2. The six-axis confound (pointer to §IV-D, not a restatement).
3. One head architecture (19.2M cross-attention); a different capacity or
   conditioning path could reveal a backbone effect.
4. Correlational diagnostics — attention and failure-mode measurements locate
   where the policies differ but do not prove causation; the causal test is
   activation patching, not run here.
5. Simulation only; frozen-backbone scope only; nulls bound rather than
   establish equivalence.

**Delete the standalone "Implications" paragraph** — §IV and §V each end with
their own implication and the Conclusion states them a third time.

**Conclusion** leads with the replicated n = 400 result (+9.5) and its
localization to P(handover|lift), **not** with "0% vs 28% at 50 epochs" — that
is a single ladder point at n = 50 and too fragile to headline. The ladder is
supporting evidence.

---

## Global editing rules

1. **No ratios of an effect to a null.** Bound the null, report the effect, let
   the reader compare.
2. **Every caveat appears once.** The draft states the six-axis confound twice
   and the n = 200 SE twice.
3. **A number appears only if a claim depends on it.** The training-budget
   paragraph (7.2× samples / 34× passes), the per-task table, and the *t*
   statistics all fail this test.
4. **Cosmos-Reason2 is not an arm.** The abstract must not imply three backbones
   were trained; it is a weight-level lineage check only. Rewrite as: *two frozen
   backbones, the endpoints of one pretraining lineage whose intermediate hop we
   verify at the weight level but do not train against.*
5. **Reconcile the rollout count.** The abstract says 2,600; §III accounts for
   1,600. Either total the ladder and instrumented rollouts explicitly or quote
   1,600 for the headline arms.
6. **Claim insufficiency, not necessity.** We showed offline metrics failed to
   rank these policies. That licenses "offline metrics were insufficient here,"
   not "closed-loop mechanism-level evaluation is necessary."
7. **Move the single-view disclaimer into Table I's caption.** The table invites
   the 62.5-vs-68.0 comparison (stock *wins*) half a column before the
   disclaimer arrives.

---

## Title

One thesis, no ratio claim, no second clause. E.g.
*What Robot Pretraining Buys a Frozen VLM Backbone: A Controlled Study Across
Two Embodiments.*

</details>
