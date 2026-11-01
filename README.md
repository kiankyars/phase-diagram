# What does a language model think water does?

Ask a language model about pure water thousands of times, take a one-word answer each
time, color each answer, and you get the phase diagram its answers imply in the notation
it was asked in. This repository does that for 28 models from Anthropic and OpenAI, grades each
diagram against the IAPWS reference equations, and repeats the same conditions in six
notations to see whether the picture survives a change of units. The method follows
[How does a blind model see the Earth?](https://outsidetext.substack.com/p/how-does-a-blind-model-see-the-earth)
by henry, which draws a world map from questions about latitude and longitude.

**[Interactive results](https://kiankyars.github.io/phase-diagram/)**: an atlas of every
model, an explorer that shows the exact prompt behind each cell, and all the charts. The
page is built into [`results/site/`](results/site/), and GitHub Pages serves it from
there.

![Every model answering instantly](results/figures/atlas_instant.png)

![Every model with reasoning allowed](results/figures/atlas_reasoned.png)

## What the maps show

Answering instantly, the models draw maps that look like phase diagrams, but on balanced
accuracy none beats a sketch made from four remembered facts. Allowed to reason, half of
the 18 models tested both ways beat that sketch, and the best are almost exact.

Across generations the answer depends on the curve. Answering instantly, over its whole
series OpenAI's boiling curve moved toward the truth in all six notations, and in kelvin
and megapascals the top of its melting line, above 209 MPa, swung past the truth, from too
cold to too hot. Neither move is resolved over the span both providers cover. Anthropic's
one-year instant series resolves no fall in error on either curve in any notation, though
in kelvin and megapascals its boiling line did rise from far too low to the truth, and the
two providers' falls in boiling error cannot be told apart. Instant maps change with the
notation, so they are not one stable picture of water. With reasoning allowed most are,
and release date adds no resolved gain in accuracy. Resolved means the 95% bootstrap
interval on the yearly change, a Theil-Sen slope, excludes zero.

Scores are balanced accuracy in kelvin and megapascals, on the same 64 × 64 points for
every model, unless a sentence says otherwise: 0.25 is guessing and 1 is perfect.

**Tidy is not the same as right.** Instant maps are tidy: 84–98% of points sit in the
largest connected region of their phase, against 100% for the truth. Their balanced
accuracy still runs only from 0.63 to 0.83. Physical order means more pressure moves water
to a denser phase and more heat to a lighter one, ranking solid, liquid, supercritical,
vapor. Ice Ih breaks this rule and jitter adds false steps, so even the truth falls short
of 100%. Where neighboring answers differ, the order is physical 54–69% of the time,
against 94% for the truth on the same jittered points (99% at 128 × 128) and about 50% for
a random map. Inside a region a stray answer adds two physical and two unphysical steps,
which pulls the figure toward 50%: half the unphysical steps on the median map touch a
cell none of its four neighbors matches. Six of the eight regions are mostly right, at
9–27% wrong on average. The misses gather in the liquid above the critical pressure (68%)
and the vapor below 0 °C (56%), and on five maps most of the ordinary liquid, below the
critical pressure, is missing.

**Four facts beat every instant map on balanced accuracy.** A sketch built from the
triple point, the critical point, ice melting at 0 °C and one latent heat scores 0.94.
No model answering instantly passes 0.83, and the best of the 132 maps across all six
notations at 32 × 32 is Opus 4.7 in kelvin and pascals, at 0.90. The sketch is not better
everywhere: it has no ice above 0 °C, so 21 of the 22 maps beat it on high-pressure ice,
GPT-4o and GPT-6 Luna beat its melting-curve error, and it uses the textbook definition of
supercritical, above both critical values, which the prompt does not state. Its layout is
built in rather than derived from those facts, and it was chosen after the pilot. Of the 18
models tested both ways,
11 of the 17 with a reading boil water within 15 K of the truth at 1 atm instantly, and
all 18 do with reasoning allowed.

**The boiling curve closed in at OpenAI, the top of the melting curve overshot.** At
OpenAI, instant boiling-curve error fell from about 2 decades of pressure on GPT-4o mini,
whose line there mostly marks where vapor turns to supercritical, to about 0.2 on GPT-5.5
and GPT-6 Sol. That fall is resolved in all six notations over the whole series, but not
over the span both providers cover (a fall of 0.47 decades a year, −0.27 to 0.87). At
Anthropic no notation resolves a trend in the error: Sonnet 4.5's 2 decades appears only
in kelvin and megapascals, 0.6 to 1.2 elsewhere, Sonnet 4.6 under five months later reads
1.6 to 4.7, and while Sonnet 5.5 reaches 0.2, Haiku 5.5 is at 1.2. Its boiling line did
rise, though, 1.55 decades a year (0.43 to 2.94), from far too low on Sonnet 4.5 to the
truth on Sonnet 5.5. The two providers' falls in error cannot be told apart (0.49 and 0.75
decades a year; difference 0.26, −0.9 to 1.5). Melting-curve error did not fall: it runs
from 27 to 107 K, and the worst values are on recent models, which put high-pressure ice
far too hot. OpenAI's line moved 42 K a year hotter on average (15 to 74), all of it above
209 MPa, where the three oldest models sit 46–193 K too cold and those from GPT-5.4 on
65–351 K too hot. Below 209 MPa it moved 14 K a year colder, only just clear of zero (0.7
to 27), and over the shared span the swing is not resolved (−47 to 93). Thirteen of the
22, 12 of them from 2026, turn hot compressed liquid into ice below 1 GPa, where the true
melting pressure is 3 to 12 GPa. Across notations at 32 × 32 that count runs from 9 to 14.
With reasoning allowed, 10 of the 12 of these that have a reasoned arm put that line above
1 GPa.

![Every model's boundaries](results/figures/boundary_fan.png)

**Allowing reasoning lifted all 18 models tested both ways.** Balanced accuracy rose on
all 18, by a median of 0.20 and at least 0.06. Over the year both providers cover, neither instant trend is resolved in kelvin and megapascals: OpenAI −0.03
to 0.25 a year, Anthropic −0.21 to 0.11. Across notations at 32 × 32 the verdict moves: in
all three Celsius notations OpenAI's trend clears zero even over that year (0.03 to 0.19,
0.03 to 0.17, 0.04 to 0.28), while in kelvin and pascals zero is inside even over its
whole series (−0.02 to 0.08). With reasoning allowed, the melting curve improves on all
18, from a median of 77 K to 20 K. GPT-5.6 Sol, and GPT-6 Astra and GPT-6.1 Sol, which
cannot turn reasoning off, match IAPWS at all but one or two of 39 pressures, on a grid
with cells 11 K wide, including 8 or 9 of the 10 above 209 MPa, where the curve climbs.
Reasoning also cuts the median boiling error from 0.58 to 0.006 decades and leaves 23 of
24 maps at or below the best instant map, 0.18.

![The effect of allowing reasoning](results/figures/reasoning_effect.png)

![Accuracy and boundary error by release date](results/figures/generations.png)

**Instant answers depend on the notation.** At 32 × 32, all 22 instant models change
their answers with the units more than they do on a plain repeat. Sonnet 4.6 agrees with
itself on 96% of repeats and 76% of unit changes. Opus 5 calls hot gas below the critical
pressure supercritical 85–89% of the time in kelvin and 18–22% in Celsius. With reasoning
allowed, at 16 × 16, the gap between repeats and unit changes is 0.06 at most, and for 18
of 24 models it cannot be told from zero. No unit is bad for everyone, and most of the
damage is specific to each model.

![Six models in six notations](results/figures/units_poster.png)

**Reasoning concentrates near the lines between labels.** Eight of the nine models that
often skip reasoning reason more often within a sixteenth of the map from a boundary or
critical line, and all nine spend more tokens there: Opus 5.5 reasons on 79% of nearby
points and 20% of the rest, Sonnet 5.5 on 76% and 17%. Sonnet 5 reasons on fewer points
near a line, 80% against 88% far from one, but spends more tokens there, 151 a reply
against 93. Drawn on the map, the reasoning of Opus 5.5 and Sonnet 5.5 traces the
sublimation and boiling curves and the critical pressure, then fills the map from about
200 MPa on Opus 5.5, apart from its coldest ice, and from about 1 GPa on Sonnet 5.5, even
in the hot corner where no boundary lies. Where it can be tested, this tracks the model's
own instant errors on three of four models: Sonnet 5.5 reasons on 82% of the points it
gets wrong instantly and 28% of those it gets right, Opus 4.8 on 79% and 49%, Opus 5 on
75% and 65%. Much of that is location: those gaps of 53, 31 and 11 points shrink to 5, 9
and 7 within 16 × 16 cells. Sonnet 5 shows no such link overall, at 85% and 85%, though
within cells it reasons 4 points more on its errors, and Opus 5.5 has no instant arm to
test.

![Where models reason](results/figures/thought_map.png)

## How it works

| Step | What happens |
|---|---|
| Grid | One jittered point per cell, from 16 × 16 up to 128 × 128, over 170–900 K and 0.01 Pa to 20 GPa. A draw within 2 K or 0.03 decades of a boundary or critical line is redrawn in the same cell, so no cell is empty and no label turns on where a curve sits within its IAPWS uncertainty |
| Prompt | `Pure water at equilibrium, at 450.00 kelvin and 2.000 megapascals: solid, liquid, vapor, or supercritical? Do not say anything else.` One user message. Pressures are plain decimals from 0.001 up to 1,000,000 of the unit and scientific notation outside that range, such as `1.230e-8 megapascals`. The request sets no system prompt, though Anthropic adds one describing the schema |
| Units control | Six notations, K or °C by Pa, MPa or atm, and two of them asked twice so that a unit effect can be told from noise |
| Sandbox | The request code accepts a fixed list of keys, none of which can carry a tool, and raises on anything else. A provider-enforced schema limits a reply to the four phase words. Of the 23 replies that escape it, two bare phase words count as that word, and the 21 refusals and token cutoffs count as wrong. Fallbacks are off |
| Conditions | Instant (no reasoning) and reasoning allowed, called `reasoned` in the results tables. In `data/answers` the six models that cannot turn reasoning off sit in arm `direct`, and `grade.arm_for` picks the arm. Allowed is not the same as used: see `share_thought` |
| Truth | [IAPWS R14-08](https://iapws.org/technical-guidance/release/MeltSub) for melting and sublimation and the Wagner and Pruss saturation equation from [IAPWS SR1-86(1992)](https://iapws.org/technical-guidance/release/Supp-sat), which agrees with IAPWS-95 to within 0.01%. Supercritical means above both the critical temperature and pressure; hot gas below the critical pressure counts as vapor and cold fluid above it as liquid |
| Scoring | Balanced accuracy averages the four phases, because 61% of the grid counts as vapor. Boundary error is the mean absolute offset of a model's boundary from the one the true labels give on the same points, over 33 isotherms for boiling and 39 isobars for melting. Resolved means the 95% bootstrap interval on the yearly change, a Theil-Sen slope, excludes zero |
| Tier | The cheapest on each platform: Batch at both, and OpenAI's Flex tier, at the same price, for four models whose Batch queue crawled and for three models' last maps |

[`PLAN.md`](PLAN.md) has the design decisions, what the pilot changed, and the budget.

## Reproduce

Grading needs no API keys. It reads the answers stored in `data/`.

```bash
uv sync
```

```bash
uv run pytest -q
```

```bash
uv run python -W ignore scripts/grade_models.py
```

```bash
uv run python -W ignore scripts/make_figures.py
```

```bash
uv run python -W ignore scripts/build_site.py
```

A push to `main` that changes `results/site/` redeploys the page.

`scripts/claims.py` recomputes the scores, trends and counts quoted on the page and in this
file into `results/claims.json`. Run records such as caps, Flex use and timings are not in
it.

```bash
uv run python -W ignore scripts/claims.py
```

`scripts/budget.py` prints the cost of each wave and needs no keys.

```bash
uv run python -W ignore scripts/budget.py
```

Asking again spends real money and needs `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in
`~/.env`, or in the file named by `PHASE_DIAGRAM_ENV`. A run resumes where it stopped and
never asks an answered key again. This repository already holds every answer, so the
command below sends nothing until `data/answers` and `data/batches` are cleared, and
retired models may no longer answer.

```bash
uv run python -W ignore scripts/run_wave.py pilot units core retest detail --provider anthropic
```

That covers the direct arm at Anthropic up to 64 × 64 in kelvin and megapascals, which is
the instant condition for 11 models and reasoning allowed for Fable 5, Fable 5.1 and Opus
5.5. Add `fine units-fine` for the finer waves and `reasoned reasoned-units reasoned-retest reasoned-core reasoned-detail
reasoned-fine` for reasoning allowed, and repeat with `--provider openai`.

The run itself was uneven. `--models` limited `units-fine` at Anthropic to five models and
`reasoned-fine` there to Sonnet 5.5 and Haiku 5.5, `--cap` let seven models past the $45
model cap, and six of them finished above it, and `--flex` moved seven OpenAI models to
Flex. Without those flags the commands
above stop at the caps or spend on waves those models never ran.

## Repository map

| Path | Contents |
|---|---|
| `src/phase_diagram/truth.py` | IAPWS curves and the true phase of any point |
| `src/phase_diagram/grid.py` | The nested jittered grid and the boundary buffer |
| `src/phase_diagram/framings.py` | The six notations and the prompt |
| `src/phase_diagram/sealed.py` | The allowlists and reply audits that keep tools out |
| `src/phase_diagram/registry.py` | Models, prices and reasoning settings |
| `src/phase_diagram/pipeline.py` | Batch and Flex runs, caps, manifests, storage |
| `src/phase_diagram/grade.py` | Every score |
| `src/phase_diagram/figures.py` | The static figures |
| `scripts/` | Entry points: `run_wave`, `budget`, `grade_models`, `make_figures`, `build_site`, `claims`, `recover_refusals`, `probe` |
| `site/template.html` | The results page, filled by `build_site.py` |
| `.github/workflows/pages.yml` | Publishes `results/site/` to GitHub Pages |
| `data/answers/` | One folder per model, one parquet per batch |
| `data/raw/` | The raw replies, gzipped, one file per batch, with thinking signatures blanked (six replies are lost: see Limits) |
| `data/batches/` | One manifest per batch |
| `results/` | Scores as CSV, `claims.json`, figures, and the built page |

## Limits

- Retirements cut Anthropic's history off at September 2025 and OpenAI's at July 2024, so
  the two lines cover different spans. Release date stands in for generation.
- GPT-5 itself needs OpenAI organization verification, which the organization used here
  did not have, and is missing here.
- The OpenAI panel is the flagship line plus the current lineup: seven working variants,
  mostly small models, were left out by choice, and o1, o3-mini, o4-mini and GPT-4.1 nano
  are retired, so trends describe this panel rather than every release.
- Six models cannot turn reasoning off: Fable 5, Fable 5.1, Opus 5.5, o3, GPT-6 Astra and
  GPT-6.1 Sol. They count only as reasoning allowed, at their lowest effort. GPT-4o,
  GPT-4o mini and GPT-4.1 have no reasoning. Opus 4.7 never engaged adaptive thinking in
  15 hand-run calls, which are not stored here, and it rejects a fixed budget. All four
  are instant only.
- Reasoning settings differ by model, so the condition is comparable only as "allowed",
  and the reasoning models use falls with release date at both providers, so a reasoned
  trend does not isolate generation.
- Headline scores are in kelvin and megapascals. Across notations the best instant map
  and the instant trends both move.
- A change of unit can also change the number's form, so a gap between notations is not
  purely a unit effect, and no arm rewords a prompt without changing its units. At the
  switch to `1.000e6 pascals`, where the true answer does not change, vapor answers on
  Haiku 5.5, GPT-5.2 and GPT-6 Luna fall 24 to 48 percentage points more than the average
  at the same pressures in megapascals and atmospheres, and clearly more than in either
  alone, at 64 × 64. GPT-5.1 falls 49 points, but its megapascals fall 38 at the same
  place, with no scientific notation involved. Only those three show a step that neither
  control shares, against about 1 of the 16 at cuts where pascals do not change form. The
  contrast with the average alone clears zero on 7 at the switch and about 4 at those
  cuts, so a count of models clear of zero says little. Megapascals cross 1.000 at the
  same pressure, and within 0.15 decades of it GPT-4.1, GPT-5.1 and GPT-4o lose 30 to 41
  points of vapor.
- Intervals are not adjusted for how many were examined: 26 of 143 distinct trends and 13
  of 84 notation contrasts exclude zero, and 24 of the 26 trends are on OpenAI's instant
  series. They resample models, and consecutive releases are related, so an interval just
  clear of zero is weak evidence by itself.
- Agreement across units is measured at 32 × 32 instantly and 16 × 16 with reasoning
  allowed, the finest grids where every model has a repeat.
- The critical temperature and pressure lines that bound supercritical are a naming
  convention rather than phase boundaries, and a model can be wrong by them without being
  wrong on the physics. That holds beside those lines, for hot gas and compressed liquid
  alike. Accepting the word on all fluid in both regions, even water at 300 K, lifts the
  best instant score from 0.83 to 0.92, and the sketch still leads at 0.94. About half the
  weight of balanced accuracy sits on points whose class is set by these two lines, and
  OpenAI's instant trend depends on them: 0.05 a year (0.01 to 0.10) as scored and −0.02
  (−0.09 to 0.06) once the word is accepted on all fluid. That tolerance also credits a map
  for calling water at 300 K supercritical, so it hides ordinary liquid the map is missing.
- A boiling reading at 1 atm needs three liquid answers below its split, so maps with
  hardly any liquid near 1 atm have none, and readings are rough: repeats move them by 10 K at the
  median.
- Ice VII above the critical temperature is barely sampled.
- Six GPT-6.1 Terra answers survive without their raw replies, after an id collision
  that is now fixed.
- The published raw replies have Anthropic's thinking signatures blanked, because they can
  identify the account. Nothing reads them, and the rest of each reply is as returned.
- The harness, the analysis and this text were written with Claude Code running Claude
  Opus 5.5, which is one of the graded models.
- `data/` holds model outputs kept as a research record, many of them wrong, which is what
  the test measures. They are not offered as training data.
- Default sampling applies throughout, and each prompt is asked once or twice.

## Cost

The run cost $796: $462 at Anthropic and $334 at OpenAI, all at Batch or Flex prices, half
the standard rate, computed from the tokens each API reported. The dearest model was
Fable 5 at $75; GPT-6 Astra took $57. `scripts/budget.py` prints the split by model. The page and this file agree on the
total, and `PLAN.md` has the budget and how many models finished each grid.
