# Plan: what do LLMs think water does?

Rebuild the phase diagram each model's answers imply and grade it against IAPWS. Two
questions:

1. Across generations, do the boundaries move toward where they really are, or do the
   regions merely look plausible?
2. Is the diagram a stable picture of water, or does it change when the same physical
   conditions are written in different units?

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| Truth | IAPWS R14-08 melting and sublimation curves, and the Wagner and Pruss saturation equation from IAPWS SR1-86(1992), which agrees with IAPWS-95 to within 0.01% | The international reference for water |
| Domain | 170–900 K, 0.01 Pa – 20 GPa, log pressure | Reaches ice VII above Tc and low-pressure vapor, so unfamiliar conditions dominate |
| Sampling | One jittered point per grid cell, levels nesting 16 ⊂ 32 ⊂ 64 ⊂ 128 | Round values appear only by chance, and refining never repeats a question |
| Excluded points | A draw within ±2 K or ±0.03 decades of any boundary or critical line is redrawn in the same cell | No label turns on where a curve sits within its IAPWS uncertainty |
| Labels | Supercritical needs T > Tc and p > Pc. Fluid above Pc but below Tc is liquid, fluid above Tc but below Pc is vapor, and below both the saturation curve decides | Standard convention, stated so the grading is unambiguous |
| Prompt | `Pure water at equilibrium, at {T} and {p}: solid, liquid, vapor, or supercritical? Do not say anything else.` | Closed the way the [map experiment](https://outsidetext.substack.com/p/how-does-a-blind-model-see-the-earth) closed its prompt |
| Units control | 2×3 factorial: K or °C by Pa, MPa or atm, temperatures to two decimals and pressures to four significant figures in every unit | The same physical point in six notations |
| Noise floor | Two framings are asked twice | A unit effect only means something beyond test–retest disagreement |
| Sandbox | An allowlist of request keys with no tool field; a request with any other key raises before it is sent | Tools are not forbidden in the prompt, they are impossible to express |
| Reply format | Provider-enforced JSON schema whose only values are the four phase words | Models told to say nothing else still reason aloud once thinking is off |
| Reasoning | Two conditions: instant (none) and reasoning allowed (a low setting that lets each model reason, or the lowest effort for the six models that cannot turn it off) | The instant arm matches the map experiment; the second arm exists because of what the pilot found (below) |
| Tier | Batch API on both platforms (50% off), and OpenAI's flex tier for GPT-5.4, GPT-6 Luna, GPT-6.1 Sol and GPT-6.1 Terra, and for the 128×128 reasoned maps of GPT-5.1, GPT-5.2 and GPT-5.5 | The cheapest tier on each. Flex costs the same as Batch, and the first four stalled on Batch |
| Sampling parameters | None | Current models reject them, so every model uses its default |
| Logprobs | Not used, unlike the map experiment | Only OpenAI's older models offer them, which would break the comparison between providers |
| Fallbacks | Off | They would silently put another model's answers on this model's map |

Both SDKs send one user message, no system prompt and the same schema. Anthropic adds a
system prompt of its own describing the schema. Nothing a model writes is executed.

## Panel: 14 + 14

**Anthropic:** Sonnet 4.5, Haiku 4.5, Opus 4.5, Opus 4.6, Sonnet 4.6, Opus 4.7,
Opus 4.8, Fable 5, Sonnet 5, Opus 5, Fable 5.1, Opus 5.5, Sonnet 5.5, Haiku 5.5.

**OpenAI:** GPT-4o mini, GPT-4o, GPT-4.1, o3, GPT-5.1, GPT-5.2, GPT-5.4, GPT-5.5,
GPT-5.6 Sol, GPT-6 Astra, GPT-6 Sol, GPT-6 Luna, GPT-6.1 Sol, GPT-6.1 Terra.

The OpenAI list is the flagship line plus the current lineup. Seven lesser-known variants
were cut so that these models could run deeper.

OpenAI's history reaches July 2024, Anthropic's only September 2025, because older models
are retired. Trends are compared within a provider, and only the overlap between them.

## What the pilot changed

The 16×16 pilot was meant to validate the pipeline, and it also changed the design.
Every model that reasoned at all scored 0.96–0.99, even on about 30 hidden tokens,
while models answering with no reasoning scored 0.64–0.87. Comparing them directly
would have measured the reasoning switch and called it generational progress.

So reasoning became a second condition, measured within each model:

- **Instant:** no reasoning. 22 models.
- **Reasoning allowed:** a low setting that lets the model reason. 24 models.
  Allowed is not the same as used: `share_thought` in `results/scores.csv` runs from
  11% of replies on Fable 5.1 to 100%.

Six models cannot turn reasoning off, so they count only as reasoned, at their lowest
effort. GPT-4o, GPT-4o mini, GPT-4.1 and Opus 4.7 are instant only. Effort labels are not
comparable between models, so the Claude settings were chosen model by model.

The pilot also suggested a baseline: a sketch built from four remembered facts (triple
point, critical point, ice melting at 0 °C, one latent heat) joined by the
Clausius-Clapeyron equation. It scores 0.94, and was chosen after the pilot.

## Napkin budget

Estimates made before the run, at batch prices. `scripts/budget.py` prints the same table
from the real answers.

| Wave | New requests per model | What it adds | Anthropic | OpenAI |
|---|---:|---|---:|---:|
| pilot | 256 | 16×16, K and MPa | $3.56 | $1.22 |
| units | 1,280 | 16×16, the other five framings | $17.85 | $6.07 |
| core | 4,608 | 32×32, all six framings | $64.24 | $21.88 |
| retest | 2,048 | 32×32 again in two framings | $28.56 | $9.74 |
| detail | 3,072 | 64×64, K and MPa | $42.83 | $14.59 |
| **Core protocol** | **11,264** | | **$157** | **$53** |
| fine | 12,288 | 128×128, K and MPa | $171.33 | $58.33 |
| units-fine | 15,360 | 64×64, the other five framings | $214.18 | $72.93 |
| **Everything** | **38,912** | | **$543** | **$185** |

Credits: about $698 at Anthropic and $1,000 at OpenAI.

The expensive models, core protocol and then everything:

| Model | Core | Everything |
|---|---:|---:|
| Fable 5 | $38 | $132 |
| Fable 5.1 | $36 | $124 |
| Opus 5.5 | $19 | $66 |
| GPT-6 Astra | $17 | $60 |
| o3 | $14 | $50 |

**Astra is not a threat to the budget.** Even with the whole protocol it costs 6% of
the OpenAI credits, because a one-word answer is tiny.

The reasoned arm cost $130 at Anthropic and $137 at OpenAI.

Rules:

- Each model finishes the core protocol before its finer waves.
- A batch that would cross a cap is refused before it is sent. Caps: $45 per model, raised
  by hand, and $350 Anthropic / $400 OpenAI. Anthropic's was raised to $450 and then $480
  for finer reasoned maps and the units control on five models.
- OpenAI, having room, then gets the fine waves. Anthropic gets 128×128 where budget allows.

## What ran

1,180,160 answers from 28 models, for $796: $462 at Anthropic and $334 at OpenAI, out of
about $698 and $1,000 in credits. `scripts/budget.py` prints the split by model. The
OpenAI credits were left over by choice, because more grid would have changed no finding.

The finest grid every listed model finished, by condition:

| Grid | Instant | Reasoning allowed |
|---|---:|---:|
| Models | 22 | 24 |
| 128×128, K and MPa | 22 | 16 |
| 64×64, K and MPa | 22 | 24 |
| 64×64, all six framings | 16 | 3 |
| 32×32, all six framings | 22 | 6 |
| 16×16, all six framings | 22 | 24 |

Every model repeated two framings, at 32×32 instantly, and at 16×16 with reasoning
allowed, or 32×32 for the six models that cannot turn reasoning off.

## Grading

- **Accuracy:** balanced accuracy is the headline, because 61% of the grid counts as
  vapor and a model that always says vapor would otherwise score well. Macro-F1 and
  confusion too, plus how the score moves if the word supercritical is accepted on fluid
  beside the critical lines, which are naming conventions rather than phase boundaries.
- **Boundaries:** how far each model's saturation curve sits from the truth in log
  pressure, its melting curve in kelvin, and where it implies the critical and triple
  points are. Sublimation is not trended, because many maps have no such boundary to
  measure: its error is reported as the region `Vapor below 0 °C` instead.
- **Regions:** the share wrong in eight named regions that partition the grid, so a
  miss has an address.
- **Coherence:** connected regions, and whether phases come in a physical order where
  neighboring answers differ. This is the test of plausible versus correct: a map can be
  tidy and wrong.
- **Representation:** agreement across framings compared with test–retest agreement,
  plus the main effects of temperature units and pressure units (`results/notation.csv`).
- **Generations:** fifteen scores against release date, with exact bootstrap intervals,
  over each provider's whole series and over the span both cover.

## Results

- An atlas of small maps, IAPWS and the sketch first, then each provider's models by
  release date, with the true boundaries overlaid. This is the counterpart of the map
  experiment's world maps.
- A units poster: six models' six maps side by side, the three notation moves most and
  the three it moves least.
- A generation timeline, and a plausible-versus-accurate scatter.
- A map of where models reason, and a table of errors by region.
- An interactive explorer that compares any model and framing against the truth.
- `results/claims.json`, written by `scripts/claims.py`: the numbers the page and README
  quote, recomputed from the tables and answers.
