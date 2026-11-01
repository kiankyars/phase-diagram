"""Grade every model that has answers and write the tables in results/.

Each model is graded under the conditions it has answers for. "instant" means no
reasoning at all. "reasoned" means reasoning is allowed, which is the reasoned arm for
most models and the direct arm for the six that cannot turn it off. Allowed is not the
same as used: `share_thought` says how often a model actually reasoned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from phase_diagram import grade, grid
from phase_diagram.framings import CANONICAL
from phase_diagram.pipeline import ROOT, load_answers, load_grid
from phase_diagram.registry import MODELS, Model, batch_cost

RESULTS = ROOT / "results"
TREND_METRICS = [
    "balanced_accuracy",
    "ba_familiar",
    "ba_unfamiliar",
    "saturation_mae",
    "melting_mae",
    "saturation_bias",
    "melting_bias",
    "melting_bias_low",
    "melting_bias_high",
    "ordered",
    "physical_change",
    "one_region",
    "ba_hot_gas",
    "ba_hot_fluid",
    "ba_any_fluid",
]
NOTATION_TREND_METRICS = ["balanced_accuracy", "saturation_mae", "melting_mae"]
SHARED_SPAN_START = "2025-09-29"  # Sonnet 4.5, the oldest model Anthropic still serves
NOTATION_METRICS = ["balanced_accuracy", "ba_familiar", "ba_unfamiliar", *(f"wrong_{name}" for name in grade.REGION_NAMES)]
NOTATION_CONTRASTS = [("t_unit", "C", "K"), ("p_unit", "atm", "Pa"), ("p_unit", "atm", "MPa"), ("p_unit", "MPa", "Pa")]


def identity(model: Model) -> dict:
    return {
        "model": model.id,
        "label": model.label,
        "provider": model.provider,
        "family": model.family,
        "released": model.released,
        "thinks": model.thinks,
    }


def usage(answers: pd.DataFrame, model: Model, arm: str) -> dict:
    chosen = answers[answers["arm"] == arm]
    return {
        "output_tokens": chosen["output_tokens"].mean(),
        "reasoning_tokens": chosen["reasoning_tokens"].mean(),
        "share_thought": (chosen["reasoning_tokens"] > 0).mean(),
        "usd_per_1k": 1000 * batch_cost(model, chosen["input_tokens"].mean(), chosen["output_tokens"].mean()),
    }


def notation_effects(framings: pd.DataFrame, condition: str, level: int) -> list[dict]:
    """Whether a unit helps or hurts across models, which a model-by-model view cannot say."""
    chosen = framings[(framings["condition"] == condition) & (framings["level"] == level)]
    rows = []
    for metric in NOTATION_METRICS:
        for unit, first, second in NOTATION_CONTRASTS:
            means = chosen.groupby(["model", unit])[metric].mean().unstack()
            rows.append(
                {"condition": condition, "level": level, "metric": metric, "first": first, "second": second}
                | grade.paired_effect(means[first], means[second])
            )
    return rows


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    points = load_grid()
    scores, units, framings, thought = [], [], [], []
    for model in MODELS:
        answers = load_answers(model)
        if not len(answers):
            continue
        for condition in ("instant", "reasoned"):
            arm = grade.arm_for(model, condition)
            if arm is None or not (answers["arm"] == arm).any():
                continue
            base = identity(model) | {"condition": condition}
            frame = grade.attempt(answers, points, arm=arm)
            finest = grade.complete_level(frame)
            if finest is None:
                continue
            spend = usage(answers, model, arm)
            if condition == "reasoned":
                thought.append(base | spend | grade.thought_focus(answers, points, arm))
            for level in range(finest, grid.MAX_LEVEL + 1):
                scores.append(base | spend | grade.score_map(frame, level))
            for level in (1, 2, 3):
                rows = grade.score_framings(answers, points, level, arm)
                framings.extend(base | row for row in rows)
                if len(rows) == 6:
                    units.append(base | {"level": level} | grade.score_units(answers, points, level, arm))

    scores = pd.DataFrame(scores)
    scores.to_csv(RESULTS / "scores.csv", index=False)
    units = pd.DataFrame(units)
    units.to_csv(RESULTS / "units.csv", index=False)
    framings = pd.DataFrame(framings)
    framings.to_csv(RESULTS / "framings.csv", index=False)
    pd.DataFrame(thought).to_csv(RESULTS / "thought.csv", index=False)

    references = []
    truth_pass = points.assign(pred=points["truth"].astype(np.int8))
    sketch_pass = points.assign(pred=grade.sketch_phase(points["T_K"], points["p_Pa"]).astype(np.int8))
    for name, frame in (("IAPWS", truth_pass), ("Four-fact sketch", sketch_pass)):
        references.extend({"reference": name} | grade.score_map(frame, level) for level in grid.SIDES)
    pd.DataFrame(references).to_csv(RESULTS / "references.csv", index=False)

    trends = []
    shared = grade.shared_level(scores)

    def fit(rows: pd.DataFrame, metrics: list[str], **labels) -> None:
        rows = rows.sort_values("released").reset_index(drop=True)
        trends.extend(labels | {"metric": metric} | grade.trend(rows["released"], rows[metric]) for metric in metrics)

    for (condition, provider), group in scores.groupby(["condition", "provider"]):
        at_level = group[group["level"] == shared]
        for span, rows in (("all", at_level), ("shared", at_level[at_level["released"] >= SHARED_SPAN_START])):
            fit(rows, TREND_METRICS, condition=condition, provider=provider, level=shared, span=span, framing=CANONICAL)
    level = grade.unit_level(units, "instant")
    chosen = framings[(framings["condition"] == "instant") & (framings["level"] == level)]
    for (provider, framing), rows in chosen.groupby(["provider", "framing"]):
        for span, part in (("all", rows), ("shared", rows[rows["released"] >= SHARED_SPAN_START])):
            fit(part, NOTATION_TREND_METRICS, condition="instant", provider=provider, level=level, span=span, framing=framing)
    pd.DataFrame(trends).to_csv(RESULTS / "trends.csv", index=False)
    notation = [row for c in ("instant", "reasoned") for row in notation_effects(framings, c, grade.unit_level(units, c))]
    pd.DataFrame(notation).to_csv(RESULTS / "notation.csv", index=False)
    print(f"graded {scores['model'].nunique()} models; finest shared level: {shared}")


if __name__ == "__main__":
    main()
