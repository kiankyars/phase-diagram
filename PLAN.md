# Plan: what do LLMs think water does?

Rebuild the phase diagram each model's answers imply and grade it against IAPWS. Two questions:

1. Across generations, do the boundaries move toward where they really are, or do the regions merely look plausible?
2. Is the diagram a stable picture of water, or does it change when the same physical conditions are written in different units?

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| Truth | IAPWS R14-08 melting and sublimation curves, and the Wagner and Pruss saturation equation from IAPWS SR1-86(1992), which agrees with IAPWS-95 to within 0.01% | The international reference for water |
| Domain | 170–900 K, 0.01 Pa – 20 GPa, log pressure | Reaches ice VII above the critical temperature and low-pressure vapor, so unfamiliar conditions dominate |
| Sampling | One randomly placed point per grid cell, levels nesting 16 ⊂ 32 ⊂ 64 ⊂ 128 | Round values appear only by chance, and refining never repeats a question |
| Excluded points | A draw within ±2 K or ±0.03 decades of any boundary or critical line is redrawn in the same cell | No label turns on where a curve sits within its IAPWS uncertainty |
| Labels | Supercritical needs temperature above the critical temperature Tc and pressure above the critical pressure Pc. Fluid above Pc but below Tc is liquid, fluid above Tc but below Pc is vapor, and below both the saturation curve decides | Standard convention, stated so the grading is unambiguous |
| Prompt | `Pure water at equilibrium, at {T} and {p}: solid, liquid, vapor, or supercritical? Do not say anything else.` | Ends the way the prompt in the [map experiment](https://outsidetext.substack.com/p/how-does-a-blind-model-see-the-earth) ends |
| Units | K or °C with Pa, MPa or atm, temperatures to two decimals and pressures to four significant figures in every unit | The same physical point in six sets of units |
| Noise floor | K with MPa and °C with atm are asked a second time | A unit effect only means something if it is bigger than the noise |
| Sandbox | An allowlist of request keys with no tool field; a request with any other key raises before it is sent | The prompt does not forbid tools. A request simply cannot express them |
| Reply format | A JSON schema whose only values are the four phase words: `output_config.format` at Anthropic and strict `response_format` at OpenAI | Models told to say nothing else still reason aloud once thinking is off |
| Reasoning | Two conditions: instant (none) and reasoning allowed (a low setting that lets each model reason, or the lowest effort for the six models that cannot turn it off) | Instant matches the map experiment. Reasoning allowed was added because of what the pilot found (below) |
| Tier | Batch on both providers (50% off), and OpenAI's Flex tier for seven models | The cheapest tier on each. Flex costs the same as Batch, and four models stalled on Batch |
| Sampling parameters | None | Current models reject them, so every model uses its default |
| Logprobs | Not used, unlike the map experiment | Only OpenAI's older models offer them |
| Fallbacks | Off | They would put another model's answers on this model's map |

Both SDKs send one user message, no system prompt and the same schema. Anthropic adds a system prompt of its own describing the schema. Nothing a model writes is executed.

## Panel: 14 + 14

**Anthropic:** Sonnet 4.5, Haiku 4.5, Opus 4.5, Opus 4.6, Sonnet 4.6, Opus 4.7, Opus 4.8, Fable 5, Sonnet 5, Opus 5, Fable 5.1, Opus 5.5, Sonnet 5.5, Haiku 5.5.

**OpenAI:** GPT-4o mini, GPT-4o, GPT-4.1, o3, GPT-5.1, GPT-5.2, GPT-5.4, GPT-5.5, GPT-5.6 Sol, GPT-6 Astra, GPT-6 Sol, GPT-6 Luna, GPT-6.1 Sol, GPT-6.1 Terra.

Seven lesser-known OpenAI variants were cut so that these models could run deeper.

OpenAI's history reaches July 2024, Anthropic's only September 2025, because older models are retired. Trends are measured within each provider, and the two providers are compared only over the period both cover.

## What the pilot changed

The 16×16 pilot was meant to test the pipeline, and it also changed the design. Every model that reasoned at all scored 0.96–0.99, even on about 30 hidden tokens, while models answering with no reasoning scored 0.64–0.87. Comparing them directly would have measured the reasoning switch and called it progress.

So reasoning became a second condition, measured within each model:

- **Instant:** no reasoning. 22 models.
- **Reasoning allowed:** a low setting that lets the model reason. 24 models. Allowed is not the same as used: `share_thought` in `results/scores.csv` runs from 11% of replies on Fable 5.1 to 100%.

Six models cannot turn reasoning off, so they count only as reasoning allowed, at their lowest effort. GPT-4o, GPT-4o mini, GPT-4.1 and Opus 4.7 are instant only. Effort labels mean different things on different models, so the Claude settings were chosen model by model.

The pilot also suggested a baseline: a sketch built from four remembered facts (triple point, critical point, ice melting at 0 °C, one latent heat) joined by the Clausius-Clapeyron equation. It scores 0.94, and was chosen after the pilot.

## Napkin budget

Estimates made before the run, at Batch prices. `scripts/budget.py` prints the same table from the real answers.

| Wave | New requests per model | What it adds | Anthropic | OpenAI |
|---|---:|---|---:|---:|
| pilot | 256 | 16×16, K and MPa | $3.56 | $1.22 |
| units | 1,280 | 16×16, the other five sets of units | $17.85 | $6.07 |
| core | 4,608 | 32×32, all six sets of units | $64.24 | $21.88 |
| retest | 2,048 | 32×32 again in two sets of units | $28.56 | $9.74 |
| detail | 3,072 | 64×64, K and MPa | $42.83 | $14.59 |
| **Core protocol** | **11,264** | | **$157** | **$53** |
| fine | 12,288 | 128×128, K and MPa | $171.33 | $58.33 |
| units-fine | 15,360 | 64×64, the other five sets of units | $214.18 | $72.93 |
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

Even GPT-6 Astra costs only 6% of the OpenAI credits for the whole protocol, because a one-word answer is tiny.

The reasoning-allowed runs cost $130 at Anthropic and $137 at OpenAI.

Rules:

- Each model finishes the core protocol before its finer waves.
- A batch that would cross a cap is refused before it is sent. Caps: $45 per model, raised by hand, and $350 Anthropic / $400 OpenAI. Anthropic's was raised to $450 and then $480 for finer reasoned maps and the units test on five models.
- OpenAI, having room, then gets the fine waves. Anthropic gets 128×128 where budget allows.

## What ran

1,180,160 answers from 28 models, for $796: $462 at Anthropic and $334 at OpenAI, out of about $698 and $1,000 in credits. The OpenAI credits were left over by choice, because more grid would have changed no finding.

How many models finished each grid:

| Grid | Instant | Reasoning allowed |
|---|---:|---:|
| Models | 22 | 24 |
| 128×128, K and MPa | 22 | 16 |
| 64×64, K and MPa | 22 | 24 |
| 64×64, all six sets of units | 16 | 3 |
| 32×32, all six sets of units | 22 | 6 |
| 16×16, all six sets of units | 22 | 24 |

Every model repeated K with MPa and °C with atm: at 32×32 instantly, and at 16×16 with reasoning allowed, or 32×32 for the six models that cannot turn reasoning off.

## Grading

- **Accuracy:** balanced accuracy, because 61% of the grid counts as vapor and a model that always says vapor would otherwise score well. We also report macro-F1, confusion matrices, and how the score moves if "supercritical" is accepted near the critical lines.
- **Boundaries:** how far each model's boiling curve sits from the truth in decades of pressure, its melting curve in kelvin, and where it puts the critical and triple points. Sublimation is scored as the region `Vapor below 0 °C`, because many maps have no sublimation curve to measure.
- **Regions:** the share wrong in eight named regions that cover the grid, so each miss has a place.
- **Coherence:** connected regions, and whether neighboring answers come in a physical order. A map can be tidy and wrong.
- **Units:** agreement across units compared with agreement on a repeat, plus the effects of temperature units and pressure units (`results/notation.csv`).
- **Generations:** fifteen scores against release date, with bootstrap intervals, over each provider's full history and over the period both cover.

## Results

- An atlas of small maps: IAPWS and the sketch first, then each provider's models by release date, with the true boundaries on top.
- A units poster: six models' six maps side by side, the three that units change most and the three they change least.
- A timeline by release date, and a tidy-versus-accurate scatter.
- A map of where models reason, and a table of errors by region.
- An interactive explorer that compares any model and set of units against the truth.
- `results/claims.json`, written by `scripts/claims.py`: the numbers the page and README quote, recomputed from the tables and answers.
