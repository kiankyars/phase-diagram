"""Write every answer map to one JSON file for the blog post's explorer.

Each map is a string with one character per grid cell, so the post can draw any model,
condition and set of units and show the exact prompt behind any cell.

    uv run python -W ignore scripts/export_maps.py [out.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from phase_diagram import grade, grid, truth
from phase_diagram.framings import FRAMINGS
from phase_diagram.pipeline import ROOT, load_answers, load_grid
from phase_diagram.registry import MODELS

OUT = ROOT / "results" / "maps.json"
READINGS = ["balanced_accuracy", "saturation_mae", "melting_mae", "boils_at", "freezes_at"]
CHARS = np.array(list(".x0123"))  # MISSING, INVALID, then the phase codes


def encode(image: np.ndarray) -> str:
    return "".join(CHARS[image.ravel() + 2])


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
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    points = load_grid().sort_values("pid")
    scores = pd.read_csv(ROOT / "results" / "scores.csv")
    shared = grade.shared_level(scores)
    atlas = scores[scores["level"] == shared].set_index(["model", "condition"])["balanced_accuracy"]
    maps, models = {}, []
    for model in MODELS:
        answers = load_answers(model)
        if not len(answers):
            continue
        atlas_scores = {c: round(float(atlas[(model.id, c)]), 2) for c in ("instant", "reasoned") if (model.id, c) in atlas.index}
        models.append({"id": model.id, "label": model.label, "provider": model.provider, "released": model.released, "thinks": model.thinks, "atlas": atlas_scores})
        for condition in ("instant", "reasoned"):
            arm = grade.arm_for(model, condition)
            if arm is None or not (answers["arm"] == arm).any():
                continue
            for framing in FRAMINGS:
                frame = grade.attempt(answers, points, framing=framing.key, arm=arm)
                level = grade.complete_level(frame)
                if level is not None:
                    graded = grade.score_map(frame, level)
                    readings = {k: None if pd.isna(graded.get(k)) else round(float(graded[k]), 3) for k in READINGS}
                    asked = answers[(answers["arm"] == arm) & (answers["framing"] == framing.key) & (answers["rep"] == 0)]
                    readings["share_thought"] = round(float((asked["reasoning_tokens"] > 0).mean()), 3)
                    maps[f"{model.id}|{condition}|{framing.key}"] = {"level": level, "cells": encode(grade.to_image(frame, level)), "scores": readings}
    data = {
        "side": grid.N_FINE,
        "T": [round(float(v), 2) for v in points["T_K"]],
        "p": [float(f"{v:.7g}") for v in points["p_Pa"]],
        "levelOf": "".join(str(v) for v in points["level"]),
        "bounds": {"tMin": grid.T_MIN, "tMax": grid.T_MAX, "logpMin": grid.LOGP_MIN, "logpMax": grid.LOGP_MAX},
        "truth": encode(grade.to_image(points.assign(pred=points["truth"].astype(np.int8)), 0)),
        "curves": curves(),
        "framings": [{"key": f.key, "label": f.label, "t": f.t_unit, "p": f.p_unit} for f in FRAMINGS],
        "atlasGrid": grid.SIDES[shared],
        "models": sorted(models, key=lambda m: (m["provider"], m["released"])),
        "maps": maps,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"{out}: {out.stat().st_size / 1e6:.2f} MB, {len(maps)} maps")


if __name__ == "__main__":
    main()
