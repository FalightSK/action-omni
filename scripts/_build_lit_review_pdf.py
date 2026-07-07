"""
Executive summary PDF: literature review verdict + recommended position for Paper 1.
Frozen-VLM VLA capstone project.
"""
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, ListFlowable, ListItem
)
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "docs" / "LitReview_Paper1_ExecutiveSummary.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

styles = getSampleStyleSheet()

NAVY = colors.HexColor("#1a2b4c")
ACCENT = colors.HexColor("#c0392b")
GOOD = colors.HexColor("#1e7a3d")
GREY = colors.HexColor("#555555")
LIGHTGREY = colors.HexColor("#f2f2f2")

styles.add(ParagraphStyle(name="DocTitle", fontSize=20, leading=24, fontName="Helvetica-Bold",
                           textColor=NAVY, spaceAfter=4))
styles.add(ParagraphStyle(name="DocSubtitle", fontSize=11.5, leading=15, fontName="Helvetica",
                           textColor=GREY, spaceAfter=14))
styles.add(ParagraphStyle(name="H1", fontSize=14, leading=17, fontName="Helvetica-Bold",
                           textColor=NAVY, spaceBefore=16, spaceAfter=6))
styles.add(ParagraphStyle(name="H2", fontSize=11.5, leading=14, fontName="Helvetica-Bold",
                           textColor=colors.HexColor("#333333"), spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle(name="Body", fontSize=10, leading=14.5, fontName="Helvetica",
                           alignment=TA_JUSTIFY, spaceAfter=6))
styles.add(ParagraphStyle(name="BodyBold", parent=styles["Body"], fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="Verdict", fontSize=11, leading=16, fontName="Helvetica-Bold",
                           textColor=colors.white, spaceAfter=0))
styles.add(ParagraphStyle(name="BulletP", fontSize=10, leading=14, fontName="Helvetica",
                           spaceAfter=3, alignment=TA_JUSTIFY))
styles.add(ParagraphStyle(name="Small", fontSize=8.3, leading=11, fontName="Helvetica",
                           textColor=GREY))
styles.add(ParagraphStyle(name="TableCell", fontSize=8.6, leading=11, fontName="Helvetica"))
styles.add(ParagraphStyle(name="TableCellBold", fontSize=8.6, leading=11, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="TableHead", fontSize=8.8, leading=11, fontName="Helvetica-Bold",
                           textColor=colors.white))

def P(text, style="Body"):
    return Paragraph(text, styles[style])

def bullets(items, style="BulletP"):
    return ListFlowable(
        [ListItem(P(i, style), leftIndent=6, bulletColor=NAVY) for i in items],
        bulletType="bullet", start="circle", leftIndent=14, spaceBefore=2, spaceAfter=8,
    )

story = []

# ── Header ────────────────────────────────────────────────────────────────
story.append(P("Literature Review — Executive Summary", "DocTitle"))
story.append(P("Frozen-VLM Vision-Language-Action (VLA) Project &nbsp;|&nbsp; Position for Paper 1 "
               "&nbsp;|&nbsp; Prepared 2026-07-02", "DocSubtitle"))
story.append(HRFlowable(width="100%", thickness=1.2, color=NAVY, spaceAfter=12))

# ── Verdict banner ───────────────────────────────────────────────────────
verdict_tbl = Table(
    [[P("VERDICT: The original Paper-1 hypothesis &mdash; \"a frozen, non-robot-pretrained VLM + "
        "diffusion/flow action head needs no backbone robot-data pretraining\" &mdash; is already "
        "published (SmolVLA, EF-VLA, ReMem-VLA). The architecture is not novel. A narrower, still-open "
        "question survives: <b>why</b> does freezing work in some reports (SmolVLA/EF-VLA, high SR) and "
        "fail in others (Knowledge Insulation, ~0% SR)? That contradiction is unresolved and is the "
        "recommended new position for Paper 1.", "Verdict")]],
    colWidths=[6.6 * inch],
)
verdict_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), ACCENT),
    ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
]))
story.append(verdict_tbl)
story.append(Spacer(1, 14))

# ── 1. Original hypothesis ──────────────────────────────────────────────
story.append(P("1. The Hypothesis Under Review", "H1"))
story.append(P(
    "Paper 1's hypothesis, as originally framed: to turn a VLM into a VLA via a Diffusion-Policy-style "
    "method, large-scale robot-data pretraining of the backbone is unnecessary. A frozen, web-pretrained "
    "VLM already emits vision-language features that are \"good enough\"; the diffusion/flow-matching "
    "decoder (with AdaLN pulling global scene+state context together with the VLM vectors) does the "
    "heavy lifting. This review asks two things: (a) has this already been shown, and (b) why do "
    "VLA papers finetune the backbone at all &mdash; what does that tell us about where the real "
    "difficulty lives?", "Body"))

# ── 2. Two camps ─────────────────────────────────────────────────────────
story.append(P("2. The Field Has Already Split Into Two Camps", "H1"))
story.append(P(
    "A focused search of 2025&ndash;2026 VLA literature shows the field is mid-argument over exactly "
    "this question. Two camps disagree, using near-identical architectures to reach opposite "
    "conclusions.", "Body"))

camp_data = [
    [P("CAMP A &mdash; Frozen backbone works", "TableHead"), P("CAMP B &mdash; Freezing fails", "TableHead")],
    [P("<b>SmolVLA</b> (HuggingFace, Jun 2025) &mdash; frozen VLM, flow-matching action expert "
       "trained from scratch, cross-attn + self-attn, state as single token, community data only. "
       "<i>Nearly identical to our architecture.</i>", "TableCell"),
     P("<b>Knowledge Insulation</b> (Physical Intelligence, May 2025) &mdash; states frozen backbone "
       "\"does not have sufficient representations for robotics&mdash;freezing doesn't work,\" reports "
       "~0% SR frozen. Proposes stop-gradient \"insulation\" instead of full freeze or full finetune.",
       "TableCell")],
    [P("<b>EF-VLA</b> (ICLR 2025) &mdash; frozen CLIP, early fusion, kept frozen specifically to "
       "preserve generalization. 85% SR on unseen goal descriptions, +20% on compositional tasks.",
       "TableCell"),
     P("<b>VLM4VLA</b> (Jan 2026) &mdash; ablates 24 VLMs. Freezing the <b>vision</b> encoder is "
       "catastrophic (&minus;28 to &minus;40 pts SimplerEnv). Freezing word embeddings is nearly free. "
       "General VLM capability does <b>not</b> predict control performance; embodied-task finetuning "
       "actively hurts.", "TableCell")],
    [P("<b>ReMem-VLA</b> &mdash; frozen VLM + learnable action queries + action diffusion, explicitly "
       "\"without large-scale pre-training.\"", "TableCell"),
     P("", "TableCell")],
]
camp_tbl = Table(camp_data, colWidths=[3.3 * inch, 3.3 * inch])
camp_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (0, 0), GOOD), ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#8a3324")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHTGREY]),
]))
story.append(camp_tbl)
story.append(Spacer(1, 10))

story.append(P("2.1 Why This Sinks the Original Framing", "H2"))
story.append(bullets([
    "The architecture (frozen VLM &rarr; cross-attention &rarr; flow/diffusion action head, state as a "
    "conditioning token, no backbone robot pretraining) is SmolVLA's architecture, published two months "
    "before this session's review. We cannot claim the design as a contribution.",
    "The \"language grounding transfers from a frozen backbone\" result is not novel either &mdash; "
    "EF-VLA published it at ICLR 2025 with stronger numbers (85% unseen-goal SR) than our current "
    "Language Table results (46&ndash;74% on held-out verbs, pending the reconciliation noted in &sect;5).",
    "\"You don't need robot pretraining\" is already Camp A's accepted position, contested by Camp B. "
    "A third voice restating either side, at capstone compute scale, adds little against "
    "Physical Intelligence / HuggingFace / DeepMind-scale efforts.",
]))

# ── 3. Why the field finetunes ──────────────────────────────────────────
story.append(P("3. Why VLA Papers Finetune the Backbone &mdash; the Real Justifications", "H1"))
story.append(P(
    "Four distinct reasons recur in the literature for adapting the VLM on robot data. Distinguishing "
    "them matters: only one is neutralized by a diffusion/flow action head (our design choice); the "
    "others remain open questions our design does not automatically answer.", "Body"))

reason_data = [
    [P("Reason", "TableHead"), P("Neutralized by a flow/diffusion head?", "TableHead")],
    [P("<b>(1) Action-output formatting.</b> RT-2 represents actions as discretized text tokens and "
       "must finetune the LM head to emit them &mdash; the single largest driver of backbone finetuning "
       "in the RT-2 lineage.", "TableCell"),
     P("<b>YES &mdash; decisively.</b> A diffusion/flow head consumes VLM features; the VLM never has "
       "to emit actions. This is the strongest structural argument our design has.", "TableCell")],
    [P("<b>(2) Embodiment / camera domain gap.</b> Web images differ from robot camera views "
       "(gripper, tabletop, viewpoint). Finetuning adapts visual features to this domain.", "TableCell"),
     P("<b>NO &mdash; only partially.</b> This is VLM4VLA's core finding: freezing the vision encoder "
       "is catastrophic. This is the live threat to our claim and must be confronted directly, not "
       "assumed away.", "TableCell")],
    [P("<b>(3) Catastrophic forgetting / preserve web knowledge</b> during robot co-training.",
       "TableCell"),
     P("YES, trivially &mdash; a frozen backbone cannot forget.", "TableCell")],
    [P("<b>(4) Cross-embodiment / dexterous scale</b> (&pi;0, OpenVLA on Open X-Embodiment) &mdash; "
       "adapting to many robot bodies and high-frequency dexterous control.", "TableCell"),
     P("Out of scope &mdash; concede this boundary explicitly rather than overclaim.", "TableCell")],
]
reason_tbl = Table(reason_data, colWidths=[3.5 * inch, 3.1 * inch])
reason_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHTGREY]),
]))
story.append(reason_tbl)
story.append(Spacer(1, 8))
story.append(P(
    "<b>Closest cousin and biggest threat: &pi;0</b> (Physical Intelligence) pairs a VLM (PaliGemma) "
    "with a flow-matching action expert &mdash; architecturally close to ours &mdash; yet still "
    "co-finetunes the backbone. Our entire differentiation from &pi;0 is the word <i>frozen</i>. "
    "Any version of Paper 1 must explicitly answer: <i>why can we freeze where &pi;0 chose not to?</i> "
    "The honest answer is scope &mdash; &pi;0 targets high-frequency dexterous cross-embodiment control "
    "requiring tight visual adaptation; our tasks target semantic/language grounding, where web "
    "pretraining plausibly already suffices. This boundary is a limitation to own, not a gap to hide.",
    "Body"))
story.append(P(
    "<b>Reference point, not a threat:</b> Diffusion Policy (Chi et al., RSS 2023) shows a conditional "
    "diffusion decoder on plain visual features already solves image&rarr;action control with no VLM "
    "and no language. This sets the floor: any VLA claim must be measured on the "
    "<b>language</b> axis, not the control axis &mdash; control alone was solved without a VLM.", "Body"))

story.append(PageBreak())

# ── 4. Novelty map ───────────────────────────────────────────────────────
story.append(P("4. Novelty Map — What's Taken vs What's Open", "H1"))

nov_data = [
    [P("Claim", "TableHead"), P("Status", "TableHead"), P("Source", "TableHead")],
    [P("Frozen VLM + diffusion/flow head, no backbone robot-pretraining works", "TableCell"),
     P("TAKEN", "TableCellBold"), P("SmolVLA, ReMem-VLA", "TableCell")],
    [P("Frozen backbone preserves language generalization to unseen goals", "TableCell"),
     P("TAKEN", "TableCellBold"), P("EF-VLA (ICLR 2025)", "TableCell")],
    [P("Freezing the vision encoder is catastrophic; freezing language/word embeddings is nearly free",
       "TableCell"), P("TAKEN (partial)", "TableCellBold"), P("VLM4VLA", "TableCell")],
    [P("General VLM benchmark capability does not predict downstream control performance", "TableCell"),
     P("TAKEN", "TableCellBold"), P("VLM4VLA", "TableCell")],
    [P("Reconciling <i>why</i> Camp A and Camp B reach opposite SR outcomes on near-identical "
       "frozen-backbone designs, using mechanistic tooling (not just benchmark SR)", "TableCell"),
     P("OPEN", "TableCellBold"), P("&mdash; no paper found", "TableCell")],
    [P("Isolating AdaLN / global-context conditioning specifically (vs. cross-attention generally) as "
       "the mechanism that lets a frozen backbone's language signal reach the action decoder",
       "TableCell"), P("OPEN", "TableCellBold"), P("&mdash; no paper found", "TableCell")],
]
nov_tbl = Table(nov_data, colWidths=[3.6 * inch, 1.0 * inch, 2.0 * inch])
nov_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHTGREY]),
    ("TEXTCOLOR", (1, 4), (1, 4), ACCENT), ("TEXTCOLOR", (1, 1), (1, 3), ACCENT),
    ("TEXTCOLOR", (1, 5), (1, 6), GOOD),
]))
story.append(nov_tbl)
story.append(Spacer(1, 10))

# ── 5. Recommended position ─────────────────────────────────────────────
story.append(P("5. Recommended Position for Paper 1", "H1"))
pos_tbl = Table(
    [[P("Reframed thesis: <b>\"When and why does a frozen VLM suffice for a diffusion-based VLA?\"</b> "
        "A mechanistic account showing the <u>language pathway survives backbone freezing</u> while the "
        "<u>visual pathway is the true bottleneck</u> &mdash; reconciling the frozen-works "
        "(SmolVLA/EF-VLA) vs. frozen-fails (Knowledge Insulation) contradiction, and localizing the "
        "mechanism (AdaLN global-context conditioning vs. cross-attention) by which the surviving "
        "language signal reaches the action decoder.", "Verdict")]],
    colWidths=[6.6 * inch])
pos_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), GOOD),
    ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
]))
story.append(pos_tbl)
story.append(Spacer(1, 10))

story.append(P("5.1 Why This Position Is Defensible", "H2"))
story.append(bullets([
    "It does not compete with SmolVLA/EF-VLA on architecture novelty or benchmark SR &mdash; it explains "
    "<i>their disagreement</i>, which neither paper attempts.",
    "It is analysis-shaped, not scale-shaped: viable on single-GPU capstone compute, unlike a "
    "SOTA-chasing system paper competing with Physical-Intelligence-scale resourcing.",
    "It reuses interpretability tooling already built this project (LoRA-as-projection, cross-attention "
    "load-bearing ablation, anti-random rollout proof, language-causal action-divergence probe) &mdash; "
    "no new infrastructure required to start.",
    "It directly engages the one concrete gap VLM4VLA leaves open: it isolates the vision/language "
    "freeze dichotomy at the level of components (which layer, which pathway) rather than only at the "
    "level of aggregate benchmark SR.",
]))

story.append(P("5.2 What Must Be Shown to Land This Claim", "H2"))
story.append(bullets([
    "<b>AdaLN isolation (highest priority, currently unverified).</b> The project's named mechanism is "
    "\"AdaLN pulls global context.\" Existing ablations only isolated cross-attention "
    "(removing it: +261% loss) &mdash; AdaLN itself has not been cleanly turned on/off. Run: AdaLN "
    "conditioning ON vs. OFF (zeroed/fixed global context), decoder otherwise identical.",
    "<b>Vision-vs-language freeze cross-check on our own setup.</b> Replicate VLM4VLA's core ablation "
    "(freeze vision encoder only vs. freeze language pathway only vs. freeze both) on our architecture "
    "and task suite, to confirm or complicate their vision-bottleneck finding under a diffusion/flow "
    "head specifically (their ablation used autoregressive-token VLAs, not flow/diffusion &mdash; this "
    "gap is itself worth stating explicitly in related work).",
    "<b>Backbone-sufficiency control.</b> Frozen VLM vs. frozen random-init encoder, same decoder. If "
    "random features collapse and VLM features work, the claim that decoder capacity alone is not doing "
    "all the work is bounded and defensible.",
    "<b>Explicit related-work reconciliation section.</b> Cite SmolVLA, EF-VLA, ReMem-VLA (Camp A) and "
    "Knowledge Insulation, VLM4VLA (Camp B) directly; state the contradiction in their reported numbers "
    "(0% vs. 85% SR under \"frozen\"); position this paper's contribution as resolving it mechanistically "
    "rather than re-litigating which camp is right by SR alone.",
]))

story.append(P("5.3 Risks to Manage", "H2"))
story.append(bullets([
    "<b>Scope creep back to the old framing.</b> Any writing that reads as \"we show frozen VLMs work\" "
    "will read as redundant to a reviewer who knows SmolVLA/EF-VLA. Every section should be phrased as "
    "\"why/when,\" not \"whether.\"",
    "<b>The &pi;0 boundary must be stated, not implied.</b> Reviewers familiar with &pi;0 will ask why we "
    "freeze where they didn't; pre-empt this in related work (dexterous cross-embodiment control vs. "
    "semantic/language grounding &mdash; see &sect;3).",
    "<b>Venue calibration.</b> This framing is workshop-realistic (CoRL/IROS workshop) on current "
    "evidence; a main-venue submission needs the AdaLN and vision/language freeze results in hand before "
    "the claim can be judged strong enough to attempt it.",
]))

story.append(Spacer(1, 10))
story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#999999")))
story.append(Spacer(1, 6))
story.append(P(
    "Sources reviewed: SmolVLA (arXiv 2506.01844), EF-VLA (ICLR 2025, OpenReview KBSHR4h8XV), "
    "ReMem-VLA (arXiv 2603.12942), Knowledge Insulation (arXiv 2505.23705), VLM4VLA (arXiv 2601.03309), "
    "&pi;0 (arXiv 2410.24164), RT-2 (Zitkovich et al., CoRL 2023), Diffusion Policy (Chi et al., RSS 2023). "
    "Paper 2 (data-pipeline thesis) intentionally out of scope for this summary per current focus.",
    "Small"))

doc = SimpleDocTemplate(
    str(OUT), pagesize=LETTER,
    topMargin=0.65 * inch, bottomMargin=0.65 * inch,
    leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    title="Literature Review Executive Summary — Paper 1 Position",
    author="Frozen-VLM VLA Project",
)
doc.build(story)
print(f"saved -> {OUT}")
