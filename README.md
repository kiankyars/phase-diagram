# What does a language model think water does?

This repository asks 28 models from Anthropic and OpenAI about water at thousands of temperatures and pressures, grades the phase diagram each one implies against the IAPWS reference equations, and repeats the same conditions in six sets of units. The method follows [How does a blind model see the Earth?](https://outsidetext.substack.com/p/how-does-a-blind-model-see-the-earth) by henry.

**[Results, method and limits](https://kiankyars.github.io/phase-diagram/)**. [`PLAN.md`](PLAN.md) has the design decisions and the budget.

![Every model answering instantly](results/figures/atlas_instant.png)

## Reproduce

`run_wave.py` asks the models again, which costs money and needs `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`. The rest reads the stored answers. To change the page, edit `site/page.md` and run `build_site.py`.

```bash
uv sync
uv run pytest -q
uv run python -W ignore scripts/grade_models.py
uv run python -W ignore scripts/make_figures.py
uv run python -W ignore scripts/build_site.py
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
| `scripts/` | Entry points |
| `site/page.md` | The text of the results page, with one `{{name}}` line for each chart |
| `site/widgets.html` | The charts and controls that those lines stand for |
| `site/template.html` | The page styles and scripts |
| `.github/workflows/pages.yml` | Publishes `results/site/` to GitHub Pages |
| `data/answers/` | One folder per model, one parquet per batch |
| `data/raw/` | The raw replies, gzipped, one file per batch |
| `data/batches/` | One manifest per batch |
| `results/` | Scores as CSV, `claims.json`, figures, and the built page |
