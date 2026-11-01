"""Napkin budget: what each wave costs each model at batch prices.

Tokens per request come from the answers collected so far, or from the probe for a
model with no answers yet, so running this again after any wave re-budgets the rest.
"""

from __future__ import annotations

import json

import pandas as pd

from phase_diagram import grid
from phase_diagram.pipeline import DATA, WAVES, load_answers, load_grid, spent
from phase_diagram.registry import MODELS, Model, batch_cost

DIRECT_WAVES = ["pilot", "units", "core", "retest", "detail", "fine", "units-fine"]


def wave_sizes(points: pd.DataFrame) -> dict[str, int]:
    """New requests per model in each wave, given the waves before it."""
    seen: set = set()
    sizes = {}
    for wave in DIRECT_WAVES:
        keys = {(s.framing, s.rep, int(pid)) for s in WAVES[wave] for pid in grid.at_level(points, s.level)["pid"]}
        sizes[wave] = len(keys - seen)
        seen |= keys
    return sizes


def tokens_per_request(model: Model, probe: dict) -> tuple[float, float, str]:
    direct = load_answers(model).query("arm == 'direct'")
    if len(direct):
        return direct["input_tokens"].mean(), direct["output_tokens"].mean(), "measured"
    answers = probe[model.id]["answers"]
    mean = lambda key: sum(a[key] for a in answers) / len(answers)
    return mean("input_tokens"), mean("output_tokens"), "probe"


def main() -> None:
    points = load_grid()
    probe = json.loads((DATA / "probe.json").read_text())
    sizes = wave_sizes(points)
    rows = []
    for model in MODELS:
        tokens_in, tokens_out, source = tokens_per_request(model, probe)
        per = batch_cost(model, tokens_in, tokens_out)
        row = {"model": model.label, "provider": model.provider, "in": round(tokens_in), "out": round(tokens_out)}
        row["$/1k"] = round(1000 * per, 2)
        row |= {wave: round(per * n, 2) for wave, n in sizes.items()}
        row["to detail"] = round(per * sum(sizes[w] for w in DIRECT_WAVES[:5]), 2)
        row["everything"] = round(per * sum(sizes.values()), 2)
        row["spent"] = round(spent(model, load_answers(model)), 2)
        row["tokens"] = source
        rows.append(row)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.max_columns", None, "display.max_rows", None)
    print("new requests per model:", sizes)
    print(table.to_string(index=False))
    print(table.groupby("provider")[[*DIRECT_WAVES, "to detail", "everything", "spent"]].sum().round(2).to_string())


if __name__ == "__main__":
    main()
