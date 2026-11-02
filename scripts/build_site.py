"""Bundle the results into one self-contained page, results/site/index.html.

Every answer map goes in as a string with one character per grid cell, so the page can
draw any model, condition and unit system without loading anything else.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from markdown_it import MarkdownIt

from phase_diagram import grade, grid, truth
from phase_diagram.framings import CANONICAL, FRAMINGS
from phase_diagram.pipeline import ROOT, load_answers, load_grid, spent
from phase_diagram.registry import MODELS

TEMPLATE = ROOT / "site" / "template.html"
PAGE = ROOT / "site" / "page.md"
WIDGETS = ROOT / "site" / "widgets.html"
OUT = ROOT / "results" / "site" / "index.html"
RESULTS = ROOT / "results"

CHARS = np.array(list(".x0123"))  # MISSING, INVALID, then the phase codes
SCORE_COLUMNS = [
    "balanced_accuracy",
    "ba_familiar",
    "ba_unfamiliar",
    "saturation_bias",
    "saturation_mae",
    "melting_bias",
    "melting_mae",
    "boils_at",
    "freezes_at",
    "critical_T",
    "critical_logp",
    "critical_T_error",
    "critical_logp_error",
    "triple_T",
    "triple_logp",
    "one_region",
    "reasoning_tokens",
    "share_thought",
    "output_tokens",
    *(f"wrong_{name}" for name in grade.REGION_NAMES),
]
UNIT_COLUMNS = ["cross", "cross_t_unit", "cross_p_unit", "retest", "gap", "gap_low", "gap_high"]
THOUGHT_LEVELS = 9  # reasoning tokens are drawn on a scale of this many steps


def encode(image: np.ndarray) -> str:
    return "".join(CHARS[image.ravel() + 2])


def clean(value):
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else round(float(value), 6)  # coarser rounding here would round the page's figures the wrong way
    return value


def records(frame: pd.DataFrame, columns: list[str]) -> dict:
    return {column: clean(frame[column].iloc[0]) for column in columns if column in frame}


def thought_cells(answers: pd.DataFrame, arm: str, points: pd.DataFrame, limit: float) -> tuple[int, str] | None:
    chosen = answers[(answers["arm"] == arm) & (answers["framing"] == CANONICAL) & (answers["rep"] == 0)]
    frame = points.merge(chosen[["pid", "reasoning_tokens"]], on="pid")
    level = grade.complete_level(frame)
    if level is None or not (frame["reasoning_tokens"] > 0).any():
        return None
    at = frame[frame["level"] >= level]
    side = grid.SIDES[level]
    image = np.zeros((side, side), dtype=int)
    scaled = np.ceil(np.clip(at["reasoning_tokens"].to_numpy() / limit, 0, 1) * THOUGHT_LEVELS).astype(int)
    image[at["i"].to_numpy() >> level, at["j"].to_numpy() >> level] = scaled
    return level, "".join(str(v) for v in image.ravel())


def model_curves(frame: pd.DataFrame, level: int) -> dict[str, list[list[float]]]:
    """A model's boiling and melting curves, for the chart that overlays every model."""
    lines = grade.boundary_lines(frame, level)
    return {name: [[round(float(t), 1), round(float(logp), 2)] for t, logp in rows] for name, rows in lines.items()}


def share_cells(wrong: list[np.ndarray]) -> str:
    """The share of models that missed each cell, on a scale of THOUGHT_LEVELS steps."""
    return "".join(str(v) for v in np.round(np.mean(wrong, axis=0) * THOUGHT_LEVELS).astype(int).ravel())


def spend_note(all_answers: dict[str, pd.DataFrame]) -> str:
    """What the whole run cost, from the tokens each API reported, at the prices paid."""
    paid = {"anthropic": 0.0, "openai": 0.0}
    for model in MODELS:
        paid[model.provider] += spent(model, all_answers[model.id])
    return (
        f"The run cost ${sum(paid.values()):,.0f}: ${paid['anthropic']:,.0f} at Anthropic and ${paid['openai']:,.0f} at OpenAI, "
        "all at Batch or Flex prices, half the standard rate."
    )


def render_section(text: str) -> str:
    """One `##` section: text up to the first chart or `###` goes in the header."""
    title, body = text.split("\n", 1)
    md = MarkdownIt()
    match = re.search(r"^(###|\{\{)", body, re.M)
    cut = match.start() if match else len(body)
    head, rest = body[:cut], body[cut:]
    if rest.startswith("###"):
        notes = "".join(f"<div>{md.render('### ' + note)}</div>" for note in rest.split("### ")[1:])
        rest = f'<div class="notes">{notes}</div>'
    else:
        rest = md.render(rest)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return f'<section class="section" aria-labelledby="{slug}"><header><h2 id="{slug}">{title}</h2>{md.render(head)}</header>{rest}</section>'


def render_page(spend: str) -> str:
    """Turn site/page.md into HTML, with each {{name}} line replaced by its chart from site/widgets.html."""
    parts = re.split(r"<!-- widget: ([a-z-]+) -->\n", WIDGETS.read_text())
    widgets = dict(zip(parts[1::2], parts[2::2]))
    intro, *sections = re.split(r"^## ", PAGE.read_text(), flags=re.M)
    intro = MarkdownIt().render(intro).replace("<p>We asked", '<p class="lede">We asked', 1).replace("<p>Do newer", '<p class="answer">Do newer', 1)
    html = f'<div class="intro">{intro}</div>' + "".join(map(render_section, sections))
    used = re.findall(r"<p>\{\{([a-z-]+)\}\}</p>", html)
    assert sorted(used) == sorted([*widgets, "spend"]), "site/page.md must use every chart exactly once"
    html = re.sub(r"<p>\{\{([a-z-]+)\}\}</p>", lambda m: f"<p>{spend}</p>" if m[1] == "spend" else widgets[m[1]], html)
    return html.replace("<ol>", '<ol class="sources">')


def curves() -> list[list[list[float]]]:
    lines = []
    for t, p in truth.boundary_curves(160).values():
        keep = (t >= grid.T_MIN) & (np.log10(p) >= grid.LOGP_MIN)
        lines.append([[round(float(a), 2), round(float(b), 3)] for a, b in zip(t[keep], np.log10(p[keep]))])
    log_pc = round(float(np.log10(truth.P_C)), 3)
    top = round(float(np.log10(truth.p_melt_high(np.array([truth.T_C]))[0])), 3)
    lines.append([[truth.T_C, log_pc], [truth.T_C, top]])
    lines.append([[truth.T_C, log_pc], [grid.T_MAX, log_pc]])
    return lines


def main() -> None:
    points = load_grid().sort_values("pid")
    scores = pd.read_csv(RESULTS / "scores.csv")
    units = pd.read_csv(RESULTS / "units.csv")
    framings = pd.read_csv(RESULTS / "framings.csv")
    references = pd.read_csv(RESULTS / "references.csv")
    thoughts = pd.read_csv(RESULTS / "thought.csv").set_index("model")
    shared = grade.shared_level(scores)
    unit_levels = {c: grade.unit_level(units, c) for c in ("instant", "reasoned")}

    side = grid.N_FINE
    truth_image = grade.to_image(points.assign(pred=points["truth"].astype(np.int8)), 0)
    sketch_image = grade.to_image(points.assign(pred=grade.sketch_phase(points["T_K"], points["p_Pa"]).astype(np.int8)), 0)

    maps, models, thought = {}, [], {}
    wrong = {"instant": [], "reasoned": []}
    all_answers = {m.id: load_answers(m) for m in MODELS}
    tokens = [a.loc[a["reasoning_tokens"] > 0, "reasoning_tokens"] for a in all_answers.values() if len(a)]
    limit = float(np.quantile(pd.concat(tokens), 0.95)) if tokens else 1.0
    for model in MODELS:
        answers = all_answers[model.id]
        entry = {
            "id": model.id,
            "label": model.label,
            "provider": model.provider,
            "family": model.family,
            "released": model.released,
            "thinks": model.thinks,
            "scores": {},
            "units": {},
            "byFraming": {},
            "boils": {},
            "curves": {},
        }
        if not len(answers):
            models.append(entry)
            continue
        for condition in ("instant", "reasoned"):
            arm = grade.arm_for(model, condition)
            if arm is None or not (answers["arm"] == arm).any():
                continue
            for framing in FRAMINGS:
                frame = grade.attempt(answers, points, framing=framing.key, arm=arm)
                level = grade.complete_level(frame)
                if level is not None:
                    own = {k: clean(v) for k, v in grade.score_map(frame, level).items() if k in SCORE_COLUMNS}
                    asked = answers[(answers["arm"] == arm) & (answers["framing"] == framing.key) & (answers["rep"] == 0)]
                    own["share_thought"] = clean(float((asked["reasoning_tokens"] > 0).mean()))
                    maps[f"{model.id}|{condition}|{framing.key}"] = {"level": level, "cells": encode(grade.to_image(frame, level)), "scores": own}
            mine = scores[(scores["model"] == model.id) & (scores["condition"] == condition)]
            at_level = mine[mine["level"] == shared]
            if len(at_level):
                entry["scores"][condition] = records(at_level, SCORE_COLUMNS)
                frame = grade.attempt(answers, points, arm=arm)
                entry["curves"][condition] = model_curves(frame, shared)
                missed = frame.assign(wrong=(frame["pred"] != frame["truth"]).astype(np.int8))
                wrong[condition].append(grade.to_image(missed, shared, "wrong"))
            mine = units[(units["model"] == model.id) & (units["condition"] == condition) & (units["level"] == unit_levels[condition])]
            if len(mine):
                entry["units"][condition] = records(mine, UNIT_COLUMNS)
            mine = framings[(framings["model"] == model.id) & (framings["condition"] == condition)]
            by_framing = entry["byFraming"].setdefault(condition, {})
            for _, row in mine.iterrows():
                by_framing.setdefault(row["framing"], {})[str(int(row["level"]))] = clean(row["balanced_accuracy"])
            read = mine[mine["level"] == unit_levels[condition]].dropna(subset=["boils_at"])
            if condition == "instant" and len(read):
                entry["boils"] = {r["framing"]: clean(r["boils_at"]) for _, r in read.iterrows()}
            shown = thought_cells(answers, arm, points, limit)
            if shown and condition == "reasoned":
                focus = thoughts.loc[model.id]
                thought[model.id] = {"level": shown[0], "cells": shown[1], "near": clean(focus["thought_near"]), "far": clean(focus["thought_far"])}
        models.append(entry)

    data = {
        "totalAnswers": int(sum(len(a) for a in all_answers.values())),
        "side": side,
        "T": [round(float(v), 2) for v in points["T_K"]],
        "p": [float(f"{v:.7g}") for v in points["p_Pa"]],
        "levelOf": "".join(str(v) for v in points["level"]),
        "bounds": {"tMin": grid.T_MIN, "tMax": grid.T_MAX, "logpMin": grid.LOGP_MIN, "logpMax": grid.LOGP_MAX},
        "truth": encode(truth_image),
        "sketch": encode(sketch_image),
        "curves": curves(),
        "models": models,
        "maps": maps,
        "thought": thought,
        "thoughtLimit": round(limit),
        "missed": {c: {"level": shared, "models": len(w), "cells": share_cells(w)} for c, w in wrong.items()},
        "level": shared,
        "unitLevels": unit_levels,
        "regions": [[f"wrong_{name}", label] for name, label in grade.REGION_NAMES.items()],
        "framings": [{"key": f.key, "label": f.label, "t": f.t_unit, "p": f.p_unit} for f in FRAMINGS],
        "references": {
            name: {str(int(r["level"])): clean(r["balanced_accuracy"]) for _, r in group.iterrows()}
            for name, group in references.groupby("reference")
        },
    }
    html = TEMPLATE.read_text().replace("<!--PAGE-->", render_page(spend_note(all_answers))).replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB, {len(maps)} maps")


if __name__ == "__main__":
    main()
