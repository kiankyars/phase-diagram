"""Draw every figure from the tables in results/."""

from __future__ import annotations

import pandas as pd

from phase_diagram import figures, grade
from phase_diagram.pipeline import ROOT, load_grid
from phase_diagram.registry import BY_ID

POSTER_ROWS = 3  # models from each end of the ranking by how much notation moves them


def main() -> None:
    results = ROOT / "results"
    scores = pd.read_csv(results / "scores.csv")
    references = pd.read_csv(results / "references.csv")
    units = pd.read_csv(results / "units.csv")
    framings = pd.read_csv(results / "framings.csv")
    for condition in ("instant", "reasoned"):
        figures.atlas(condition, scores, references)
    figures.reasoning_effect(scores, references)
    figures.generations(scores, references)
    figures.boundary_fan(scores)
    figures.error_share(scores)
    figures.region_matrix(scores)
    figures.familiar_vs_unfamiliar(scores)
    figures.plausible_vs_accurate(scores, references)
    levels = {condition: grade.unit_level(units, condition) for condition in ("instant", "reasoned")}
    for condition, level in levels.items():
        figures.units_gap(units, condition, level)
        figures.units_matrix(framings, condition, level)
    figures.units_boiling(framings, "instant", levels["instant"])
    chosen = units[(units["condition"] == "instant") & (units["level"] == levels["instant"])].sort_values("gap")
    poster = pd.concat([chosen.tail(POSTER_ROWS).iloc[::-1], chosen.head(POSTER_ROWS)])
    figures.units_poster([BY_ID[m] for m in poster["model"]], "instant", poster, levels["instant"])
    deciders = ["claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5", "gpt-6-astra"]
    figures.thought_map([BY_ID[m] for m in deciders], load_grid())


if __name__ == "__main__":
    main()
