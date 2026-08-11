"""Held-out paraphrases for LIBERO-Goal — the OOD instruction test set.

Design
──────
Training uses ONLY the ten canonical instruction strings that ship with
LIBERO-Goal. These paraphrases are never seen during training or in the
precomputed embedding cache; they exist solely to be substituted at evaluation
time. That makes the contrast clean: the goal, the scene, the objects and the
required trajectory are all identical to training, and the *only* thing that
changed is the wording. A drop in success rate is therefore attributable to
instruction encoding, not to a harder task.

Why this matters for this project specifically
──────────────────────────────────────────────
LIBERO-Goal holds the scene fixed across all ten goals, so language is the only
signal that disambiguates them. Chapter 1 measured that Pi-0.5 encodes the
instruction unusually strongly — its image tokens carry goal identity at
eta^2 = 0.761 against 0.16-0.24 for every other backbone. Whether that
representational fact produces *robust* instruction following is exactly what a
paraphrase test answers, and it cannot be answered offline.

Paraphrase construction rules, applied deliberately:
  * same goal, same objects, same spatial relation — only surface form changes
  * vary the verb ("put" -> "place"/"set"), the determiner, and the preposition
  * keep one paraphrase per goal close to the original (a mild rewording) and
    one further away (different verb AND different phrasing), so the eval can
    report a difficulty gradient rather than a single number
  * no synonyms for the OBJECTS themselves in the "near" set — swapping "bowl"
    for "dish" tests object grounding, which is a different question from
    instruction robustness, so those live in the "far" set and are reported
    separately
"""

from __future__ import annotations

# canonical -> {"near": [...], "far": [...]}
#   near : reworded, same content words
#   far  : different verb and/or object synonym — a harder generalisation
PARAPHRASES: dict[str, dict[str, list[str]]] = {
    "open the middle drawer of the cabinet": {
        "near": ["pull open the middle drawer of the cabinet",
                 "open up the cabinet's middle drawer"],
        "far":  ["slide the centre drawer of the dresser out"],
    },
    "open the top drawer and put the bowl inside": {
        "near": ["open the top drawer and place the bowl inside",
                 "open the upper drawer, then put the bowl in it"],
        "far":  ["pull out the highest drawer and set the dish within"],
    },
    "push the plate to the front of the stove": {
        "near": ["push the plate toward the front of the stove",
                 "slide the plate to the stove's front"],
        "far":  ["nudge the dish to the near edge of the cooktop"],
    },
    "put the bowl on the plate": {
        "near": ["place the bowl on the plate",
                 "set the bowl down on the plate"],
        "far":  ["stack the dish onto the platter"],
    },
    "put the bowl on the stove": {
        "near": ["place the bowl on the stove",
                 "set the bowl onto the stove"],
        "far":  ["move the dish onto the cooktop"],
    },
    "put the bowl on top of the cabinet": {
        "near": ["place the bowl on top of the cabinet",
                 "set the bowl above the cabinet"],
        "far":  ["rest the dish atop the cupboard"],
    },
    "put the cream cheese in the bowl": {
        "near": ["place the cream cheese in the bowl",
                 "put the cream cheese into the bowl"],
        "far":  ["drop the cheese block into the dish"],
    },
    "put the wine bottle on the rack": {
        "near": ["place the wine bottle on the rack",
                 "set the wine bottle onto the rack"],
        "far":  ["store the bottle of wine in the holder"],
    },
    "put the wine bottle on top of the cabinet": {
        "near": ["place the wine bottle on top of the cabinet",
                 "set the wine bottle above the cabinet"],
        "far":  ["stand the bottle of wine atop the cupboard"],
    },
    "turn on the stove": {
        "near": ["switch on the stove",
                 "turn the stove on"],
        "far":  ["ignite the cooktop burner"],
    },
}


def canonical_instructions() -> list[str]:
    return list(PARAPHRASES)


def variants(canonical: str, tier: str) -> list[str]:
    """tier is 'canonical', 'near' or 'far'."""
    if tier == "canonical":
        return [canonical]
    return PARAPHRASES[canonical][tier]


def all_eval_instructions(tier: str) -> dict[str, list[str]]:
    """{canonical goal: [instruction strings to evaluate with]} for one tier."""
    return {c: variants(c, tier) for c in PARAPHRASES}


def assert_disjoint() -> None:
    """No paraphrase may coincide with any canonical string.

    A collision would silently turn part of the OOD set into training data, and
    the resulting success rate would look like generalisation when it is
    memorisation. Cheap to check, so check it.
    """
    canon = set(PARAPHRASES)
    for c, tiers in PARAPHRASES.items():
        for tier, xs in tiers.items():
            for x in xs:
                assert x not in canon, f"paraphrase collides with a canonical string: {x!r}"
                assert x != c, f"paraphrase identical to its canonical: {x!r}"


assert_disjoint()
