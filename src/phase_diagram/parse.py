"""Turn a model's reply into a phase label. The reply is only ever pattern-matched."""

from __future__ import annotations

import json
import re

SYNONYMS = {
    "solid": "solid",
    "ice": "solid",
    "liquid": "liquid",
    "vapor": "vapor",
    "vapour": "vapor",
    "gas": "vapor",
    "gaseous": "vapor",
    "steam": "vapor",
    "supercritical": "supercritical",
    "super-critical": "supercritical",
}
CLASSES = ("solid", "liquid", "vapor", "supercritical")


def parse_answer(text: str | None) -> tuple[str | None, str]:
    """Return (label, status).

    status is "schema" for a schema-constrained reply, "exact" for a bare class word,
    "synonym" for a bare synonym, "embedded" for one unambiguous class word inside
    longer text, and "invalid" otherwise.
    """
    if not text:
        return None, "invalid"
    try:
        phase = json.loads(text).get("phase", "").lower()
        if phase in CLASSES:
            return phase, "schema"
    except (ValueError, AttributeError):
        pass
    words = re.findall(r"[a-z]+(?:-[a-z]+)?", text.lower())
    if len(words) == 1 and words[0] in CLASSES:
        return words[0], "exact"
    if len(words) == 1 and words[0] in SYNONYMS:
        return SYNONYMS[words[0]], "synonym"
    hits = {SYNONYMS[w] for w in words if w in SYNONYMS}
    if len(hits) == 1:
        return hits.pop(), "embedded"
    return None, "invalid"
