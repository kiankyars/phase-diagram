# What does a language model think water does?

This repository asks 28 models from Anthropic and OpenAI about water at thousands of temperatures and pressures, grades the phase diagram each one implies against the IAPWS reference equations, and repeats the same conditions in six sets of units. The method follows [How does a blind model see the Earth?](https://outsidetext.substack.com/p/how-does-a-blind-model-see-the-earth) by henry.

**[Results, method and limits](https://kiankyars.github.io/machine_learning/2026/11/03/phase-diagram.html)**. [`PLAN.md`](PLAN.md) has the design decisions and the budget.

![](results/figures/atlas_instant.png)

![](results/figures/atlas_reasoned.png)

## Reproduce

`run_wave.py` asks the models again, which needs `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`. The rest reads the stored answers.

```bash
uv sync
uv run pytest -q
uv run python -W ignore scripts/grade_models.py
uv run python -W ignore scripts/make_figures.py
uv run python -W ignore scripts/export_maps.py
uv run python -W ignore scripts/claims.py
uv run python -W ignore scripts/budget.py
uv run python -W ignore scripts/run_wave.py pilot units core retest detail --provider anthropic
```

## Tree

| Path | Contents |
|---|---|
| `src/phase_diagram/truth.py` | IAPWS curves and the true phase of any point |
| `src/phase_diagram/grid.py` | The nested grid of randomly placed points |
| `src/phase_diagram/framings.py` | The six sets of units and the prompt |
| `src/phase_diagram/sealed.py` | The allowlists that keep tools out |
| `src/phase_diagram/registry.py` | Models, prices and reasoning settings |
| `src/phase_diagram/pipeline.py` | Batch and Flex runs, caps, storage |
| `src/phase_diagram/grade.py` | Every score |
| `src/phase_diagram/figures.py` | The static figures |
| `scripts/` | Entry points. `export_maps.py` writes the answer maps the blog post draws |
| `data/answers/` | One folder per model, one parquet per batch |
| `data/raw/` | The raw replies, gzipped, one file per batch |
| `data/batches/` | One manifest per batch |
| `results/` | Scores as CSV, `claims.json`, figures, and `maps.json` for the blog post's explorer |
