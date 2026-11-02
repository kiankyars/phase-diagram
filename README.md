# What does a language model think water does?

This repository asks 28 models from Anthropic and OpenAI about water at thousands of
temperatures and pressures, grades the phase diagram each one implies against the IAPWS
reference equations, and repeats the same conditions in six notations. The method follows
[How does a blind model see the Earth?](https://outsidetext.substack.com/p/how-does-a-blind-model-see-the-earth)
by henry.

**[Interactive results](https://kiankyars.github.io/phase-diagram/)**

![Every model answering instantly](results/figures/atlas_instant.png)

![Every model with reasoning allowed](results/figures/atlas_reasoned.png)

## What the maps show

Answering instantly, the models draw maps that look like phase diagrams, but none beats a
sketch made from four remembered facts. Allowed to reason, half of the 18 models tested
both ways beat it, the best are almost exact, and release date adds no resolved gain in
accuracy.

Scores are balanced accuracy in kelvin and megapascals on the same 64 × 64 points: 0.25 is
guessing and 1 is perfect. Resolved means the 95% bootstrap interval on the yearly change,
a Theil-Sen slope, excludes zero.

**Tidy is not the same as right.** On instant maps 84–98% of points sit in the largest
connected region of their phase, yet balanced accuracy runs only from 0.63 to 0.83. The
misses gather in the liquid above the critical pressure (68% wrong) and the vapor below
0 °C (56%).

**Four facts beat every instant map.** A sketch built from the triple point, the critical
point, ice melting at 0 °C and one latent heat scores 0.94. No instant map passes 0.83.
The sketch's layout is built in, it was chosen after the pilot, and it has no ice above
0 °C, so 21 of the 22 maps beat it on high-pressure ice.

**The boiling curve closed in at OpenAI, the top of the melting curve overshot.** OpenAI's
instant boiling error fell from about 2 decades of pressure on GPT-4o mini to about 0.2 on
GPT-5.5 and GPT-6 Sol. That fall is resolved over its whole series but not over the span
both providers cover, and Anthropic resolves no trend in the error. Melting-curve error
did not fall. Above 209 MPa, OpenAI's three oldest models sit 46–193 K too cold and those
from GPT-5.4 on 65–351 K too hot. Thirteen of the 22 turn hot compressed liquid into ice
below 1 GPa, where the true melting pressure is 3 to 12 GPa.

![Every model's boundaries](results/figures/boundary_fan.png)

**Allowing reasoning lifted all 18 models tested both ways.** Balanced accuracy rose by a
median of 0.20 and at least 0.06. Median melting error fell from 77 K to 20 K and boiling
error from 0.58 to 0.006 decades. GPT-5.6 Sol, GPT-6 Astra and GPT-6.1 Sol match IAPWS at
all but one or two of 39 pressures.

![The effect of allowing reasoning](results/figures/reasoning_effect.png)

![Accuracy and boundary error by release date](results/figures/generations.png)

**Instant answers depend on the notation.** All 22 instant models change their answers
with the units more than they do on a plain repeat. Sonnet 4.6 agrees with itself on 96%
of repeats and 76% of unit changes. Opus 5 calls hot gas below the critical pressure
supercritical 85–89% of the time in kelvin and 18–22% in Celsius. With reasoning allowed
the gap between repeats and unit changes is 0.06 at most.

![Six models in six notations](results/figures/units_poster.png)

**Reasoning concentrates near the lines between labels.** Eight of the nine models that
often skip reasoning reason more often near a boundary or critical line, and all nine
spend more tokens there. Opus 5.5 reasons on 79% of nearby points and 20% of the rest,
Sonnet 5.5 on 76% and 17%.

![Where models reason](results/figures/thought_map.png)

## How it works

| Step | What happens |
|---|---|
| Grid | One jittered point per cell, from 16 × 16 up to 128 × 128, over 170–900 K and 0.01 Pa to 20 GPa. Points within 2 K or 0.03 decades of a boundary or critical line are redrawn in the same cell |
| Prompt | `Pure water at equilibrium, at 450.00 kelvin and 2.000 megapascals: solid, liquid, vapor, or supercritical? Do not say anything else.` One user message and no system prompt, though Anthropic adds one describing the schema |
| Units control | Six notations, K or °C by Pa, MPa or atm, two of them asked twice to measure noise |
| Sandbox | The request code accepts a fixed list of keys, none of which can carry a tool, and raises on anything else. A provider-enforced schema limits replies to the four phase words |
| Conditions | Instant (no reasoning) and reasoning allowed, called `reasoned` in the results tables. The six models that cannot turn reasoning off sit in arm `direct` in `data/answers`. Allowed is not the same as used: see `share_thought` |
| Truth | [IAPWS R14-08](https://iapws.org/technical-guidance/release/MeltSub) for melting and sublimation and the Wagner and Pruss saturation equation from [IAPWS SR1-86(1992)](https://iapws.org/technical-guidance/release/Supp-sat). Supercritical means above both the critical temperature and pressure |
| Scoring | Balanced accuracy averages the four phases, because 61% of the grid counts as vapor. Boundary error is the mean absolute offset from the true boundary, over 33 isotherms for boiling and 39 isobars for melting |
| Tier | Batch at both providers, and OpenAI's Flex tier, at the same price, for seven models |

[`PLAN.md`](PLAN.md) has the design decisions and the budget.

## Reproduce

Grading needs no API keys. It reads the answers stored in `data/`.

```bash
uv sync
uv run pytest -q
uv run python -W ignore scripts/grade_models.py
uv run python -W ignore scripts/make_figures.py
uv run python -W ignore scripts/build_site.py
uv run python -W ignore scripts/claims.py
uv run python -W ignore scripts/budget.py
```

`claims.py` recomputes the numbers quoted on the page and here into `results/claims.json`.
A push to `main` that changes `results/site/` redeploys the page.

Asking again spends real money and needs `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in
`~/.env`. A run never asks an answered key again, so clear `data/answers` and
`data/batches` first. Retired models may no longer answer.

```bash
uv run python -W ignore scripts/run_wave.py pilot units core retest detail --provider anthropic
```

The original run also used `--models`, `--cap` and `--flex`, so these commands alone will
not match it exactly.

## Repository map

| Path | Contents |
|---|---|
| `src/phase_diagram/truth.py` | IAPWS curves and the true phase of any point |
| `src/phase_diagram/grid.py` | The nested jittered grid |
| `src/phase_diagram/framings.py` | The six notations and the prompt |
| `src/phase_diagram/sealed.py` | The allowlists that keep tools out |
| `src/phase_diagram/registry.py` | Models, prices and reasoning settings |
| `src/phase_diagram/pipeline.py` | Batch and Flex runs, caps, storage |
| `src/phase_diagram/grade.py` | Every score |
| `src/phase_diagram/figures.py` | The static figures |
| `scripts/` | Entry points |
| `site/template.html` | The results page, filled by `build_site.py` |
| `.github/workflows/pages.yml` | Publishes `results/site/` to GitHub Pages |
| `data/answers/` | One folder per model, one parquet per batch |
| `data/raw/` | The raw replies, gzipped, one file per batch |
| `data/batches/` | One manifest per batch |
| `results/` | Scores as CSV, `claims.json`, figures, and the built page |

## Limits

- Retirements cut Anthropic's history off at September 2025 and OpenAI's at July 2024, so
  the two lines cover different spans. Release date stands in for generation.
- Six models cannot turn reasoning off: Fable 5, Fable 5.1, Opus 5.5, o3, GPT-6 Astra and
  GPT-6.1 Sol. They count only as reasoning allowed. GPT-4o, GPT-4o mini, GPT-4.1 and
  Opus 4.7 are instant only.
- Reasoning settings differ by model, and newer models reason less at both providers, so a
  reasoned trend does not isolate generation.
- A change of unit can also change the number's form, so a gap between notations is not
  purely a unit effect.
- Intervals are not adjusted for how many were examined: 26 of 143 trends and 13 of 84
  notation contrasts exclude zero. They resample models, and consecutive releases are
  related.
- The critical lines that bound supercritical are a naming convention. Accepting the word
  on all fluid lifts the best instant score from 0.83 to 0.92, and the sketch still leads
  at 0.94, but that also hides ordinary liquid a map is missing.
- Ice VII above the critical temperature is barely sampled.
- Six GPT-6.1 Terra answers survive without their raw replies, after an id collision that
  is now fixed.
- The harness, the analysis and this text were written with Claude Code running Claude
  Opus 5.5, which is one of the graded models.
- `data/` holds model outputs kept as a research record, many of them wrong. They are not
  offered as training data.
- Default sampling applies throughout, and each prompt is asked once or twice.

## Cost

The run cost $796: $462 at Anthropic and $334 at OpenAI, at Batch or Flex prices. The
dearest model was Fable 5 at $75. `scripts/budget.py` prints the split by model.
