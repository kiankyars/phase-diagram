# What does a language model think water does?

We asked 28 language models 1,180,160 questions about pure water at different temperatures and pressures. From their answers we drew the phase diagram each model implies, and we graded it against the IAPWS reference equations. Answering instantly, no model beats a simple sketch made from four remembered facts. Allowed to reason, half of the 18 models tested both ways beat the sketch, and the best are almost exact.

Do newer models do better? It depends on the curve. When OpenAI's models answer instantly, their boiling curve has moved toward the truth and the top of their melting curve has moved past it. Neither trend is clear since September 2025, the period both providers cover. Anthropic's instant answers show no clear fall in error. Instant maps also change with the units in the question. With reasoning allowed, most maps do not, and newer models are not clearly more accurate.

{{stats}}

[Code, stored answers and raw replies on GitHub](https://github.com/kiankyars/phase-diagram)

## The atlas

Each square is one model's phase diagram, built only from its answers. Temperature runs left to right and pressure runs bottom to top. Black lines show the true IAPWS boundaries and the critical temperature and pressure.

Scores are balanced accuracy, the average share correct across the four phases. A score of 0.25 is random guessing and 1 is perfect. Scores here use kelvin and megapascals and the same 64 × 64 points for every model, even where a map is drawn finer.

Models marked * cannot turn reasoning off, so they appear only under "Reasoning allowed". Opus 4.7, GPT-4o, GPT-4o mini and GPT-4.1 did not reason, so they appear only under "Answering instantly".

{{atlas}}

## Explore one model's diagram

Every cell is one question, asked on its own with no tools. Hover over a cell to see the exact prompt, the model's answer and the true phase.

{{explorer}}

## What the maps show

Scores use kelvin and megapascals at 64 × 64 unless a sentence says otherwise.

### Tidy is not the same as right

Instant maps look tidy: 84–98% of points sit in the largest connected region of their phase, against 100% for the truth. But their balanced accuracy is only 0.63 to 0.83.

We also checked physical order. We rank the phases solid, liquid, supercritical, vapor, from densest to lightest. More pressure should push water toward the solid end of that list, and more heat toward the vapor end. Where neighboring answers differ, the order is physical 54–69% of the time. The truth scores 94% on the same points and a random map scores about 50%. The truth falls short of 100% for two reasons: ordinary ice is less dense than water, which breaks this ranking, and the random placement of points adds false steps.

Six of the eight regions of the map are mostly right, at 9–27% wrong on average. The big misses are liquid above the critical pressure (68% wrong) and vapor below 0 °C (56% wrong). On five maps, most of the ordinary liquid below the critical pressure is missing.

### Four facts beat every instant map

A sketch built from the triple point, the critical point, ice melting at 0 °C and one latent heat scores 0.94. No model answering instantly scores above 0.83. Across all six sets of units at 32 × 32, the best of the 132 maps is Opus 4.7 in kelvin and pascals, at 0.90.

The sketch is not better everywhere. It has no ice above 0 °C, so 21 of the 22 instant maps beat it on high-pressure ice. GPT-4o and GPT-6 Luna also beat its melting curve. The sketch's layout does not follow from the four facts. It was chosen after a small trial run. It also uses the textbook definition of supercritical, which the prompt does not state.

At 1 atm, water boils at 373 K. Answering instantly, 11 of the 18 models tested both ways get within 15 K of that, and one gives too few liquid answers to read a boiling point. With reasoning allowed, all 18 get within 15 K.

### OpenAI's boiling curve improved, and its melting curve overshot

Boiling curve error is measured in decades of pressure, and one decade is a factor of 10. OpenAI's instant error fell from about 2 decades on GPT-4o mini to about 0.2 on GPT-5.5 and GPT-6 Sol. This fall is clear in all six sets of units over OpenAI's full history, but not since September 2025.

Anthropic shows no clear trend in this error in any units. Its boiling curve did rise, though, from far too low on Sonnet 4.5 to the truth on Sonnet 5.5. The two providers' falls in error are too close to tell apart.

Melting curve error did not fall. It runs from 27 to 107 K, and the worst values are on recent models, which put high-pressure ice far too hot. Thirteen of the 22 instant models, 12 of them from 2026, turn hot compressed liquid into ice below 1 GPa. At those temperatures the true melting pressure is 3 to 12 GPa. Twelve of these models were also tested with reasoning allowed, and 10 of them then put the melting curve above 1 GPa.

### Allowing reasoning improved all 18 models tested both ways

Balanced accuracy rose on all 18, by a median of 0.20 and by at least 0.06. Median melting curve error fell from 77 K to 20 K. Median boiling curve error fell from 0.58 to 0.006 decades, and 23 of the 24 maps with reasoning allowed have an error at or below that of the best instant map, 0.18 decades.

The melting curves of GPT-5.6 Sol, GPT-6 Astra and GPT-6.1 Sol match IAPWS at all but one or two of 39 pressures, on a grid with cells 11 K wide.

### Instant answers depend on the units

At 32 × 32, all 22 instant models change their answers more when the units change than on a plain repeat. Sonnet 4.6 agrees with itself on 96% of repeats but only 76% of unit changes. Opus 5 calls hot gas below the critical pressure supercritical 85–89% of the time in kelvin and 18–22% of the time in Celsius.

With reasoning allowed, at 16 × 16, the gap between repeats and unit changes is at most 0.06. For 18 of 24 models it is too small to tell from zero.

### Models reason most near the boundaries

Nine models often skip reasoning. Eight of them reason more often near a boundary or critical line, and all nine spend more tokens there. Opus 5.5 reasons on 79% of nearby points and 20% of the rest. Sonnet 5.5 reasons on 76% and 17%. Sonnet 5 is the exception. It reasons on fewer points near a line, 80% against 88%, but still spends more tokens there, 151 a reply against 93.

Three of the four models we can test also reason more often on the points they get wrong when answering instantly. Sonnet 5.5 reasons on 82% of the points it gets wrong instantly and 28% of those it gets right. Opus 4.8 reasons on 79% and 49%, and Opus 5 on 75% and 65%. Much of this is location: when we compare only points in the same 16 × 16 grid cell, those gaps of 53, 31 and 11 percentage points shrink to 5, 9 and 7. Sonnet 5 shows no such link overall, at 85% and 85%.

## What allowing reasoning changes

Each row is one model. A hollow circle is the model answering instantly. A filled circle is the same model with reasoning allowed. The vertical line is the four-fact sketch.

Haiku 5.5 gains the most, from 0.63 to 0.96, because instantly it almost never says liquid. If we accept the word supercritical for any fluid, GPT-5.5 gains the most instead, 0.31 against 0.19 for Haiku 5.5. Sonnet 5 gains the least, from 0.79 to 0.85. With reasoning allowed it calls 59% of hot gas supercritical, up from 24% instantly. It is the only one of the 18 whose vapor score falls.

Fourteen of 24 models beat the sketch with reasoning allowed. Nine of them are among the 18 tested both ways: seven of eight at OpenAI and two of ten at Anthropic. The two providers share no reasoning setting, so this split is not a fair comparison. GPT-5.1 and Sonnet 4.5 fall just short of the sketch and GPT-6 Luna just passes it, but all three are within sampling error on the 4,096 points of the 64 × 64 grid. On its 128 × 128 map, Luna leads by 0.012 (0.006 to 0.018).

Allowed does not mean used. Settings differ between models, and some rarely reason: Fable 5.1 does so on 11% of replies. GPT-6 Astra averages 26 reasoning tokens a reply and scores 0.99. The table under "Every score" gives each model's share of replies that reasoned.

{{reasoning-chart}}

## Do the boundaries move toward the truth?

Partly. When models answer instantly, only OpenAI's boiling curve shows a clear fall in error, over its full history. With reasoning allowed, only Anthropic's melting curve does, and only just.

Each line marks where one model stops saying vapor (boiling) or solid (melting) at 64 × 64, whatever it says next. Each line is smoothed by taking the median of five neighboring grid lines, and it is colored by release date. The heavy lines are IAPWS.

When models answer instantly, boiling curves spread over orders of magnitude of pressure, and newer models gather around the truth. Melting lines gather in the wrong place. Of the 14 lines from 2026, 13 sit at 0.7 GPa or below at 400 K, where the truth is 2.8 GPa, and 10 stay at 2 GPa or below at 600 K, where the truth is 9 GPa. The 14th, Haiku 5.5, goes the other way and stops at 342 K.

The melting error comes from the top of the map. Over the 29 pressures below 209 MPa it runs from 8 to 43 K. Over the 10 pressures above 209 MPa it runs from 58 K to at least 382 K. Most recent models put ice far too hot there, and GPT-4o mini, GPT-4o, GPT-4.1 and both Haikus put it too cold.

The number of models that turn hot compressed liquid into ice below 1 GPa depends on the units. It is 13 of 22 in kelvin and megapascals at 64 × 64, and 9 to 14 across units at 32 × 32.

With reasoning allowed, boiling curves land on the truth and most melting lines rise along it. Twelve of the 13 models that put ice below 1 GPa were tested with reasoning allowed, and six of the 12 reach about 6 GPa between 400 and 640 K, which is what the true phases give on the same points. The ten melting lines closest to the truth are all from 2026, but so are 18 of the 24 models, and some 2026 models, such as Sonnet 5, still sit far below the truth.

{{fan-charts}}

## Across generations

This chart shows balanced accuracy by release date in kelvin and megapascals. Lines join models of the same family. Yearly changes are Theil-Sen slopes, a line fit that outliers barely move, with 95% bootstrap intervals in brackets. A trend is clear when its interval does not include zero.

Since September 2025, the period both providers cover, neither instant trend is clear. OpenAI gains 0.05 a year (−0.03 to 0.25) and Anthropic 0.00 (−0.21 to 0.11). Over OpenAI's full history the gain is the same 0.05 a year, with a narrower interval (0.01 to 0.10). That result depends on GPT-4o mini, GPT-4o and GPT-4.1. Remove any one of them, or any one of the three newest instant models, and the interval includes zero again.

Across the six sets of units at 32 × 32, OpenAI's slope runs from 0.04 to 0.11, and whether the trend is clear changes too. In all three sets of units that use Celsius, the trend is clear even since September 2025. In kelvin and pascals it is not clear even over the full history (−0.02 to 0.08).

With reasoning allowed, neither provider shows a clear gain in accuracy: OpenAI 0.01 a year (−0.02 to 0.06) and Anthropic 0.02 (−0.05 to 0.11).

Instant boiling curve error falls 0.49 decades a year at OpenAI (0.13 to 0.85) and 0.75 at Anthropic (−0.13 to 1.99). These two falls are too close to tell apart (difference 0.26, −0.9 to 1.5). Anthropic's curve itself rose 1.55 decades a year (0.43 to 2.94), from far too low on Sonnet 4.5 to the truth on Sonnet 5.5. OpenAI's shows no such move.

In kelvin and megapascals, instant melting curve error rises at both providers, but neither rise is clear. OpenAI's melting curve moved 42 K a year hotter on average (15 to 74), and it went past the truth, not onto it. It went from 13–45 K too cold on the three oldest models to 18–83 K too hot from GPT-5.4 on. All of that move is above 209 MPa, where those models went from 46–193 K too cold to 65–351 K too hot. Since September 2025 this swing is not clear (−47 to 93).

With reasoning allowed, melting curve error falls about 27 K a year at Anthropic (0.5 to 60). This is only just clear. Remove the Fable, Haiku or Opus models, or any one of eight of Anthropic's 13 models, and the interval includes zero. At OpenAI it falls 7.5 K a year (−6 to 22), which is not clear.

Answering instantly, small models lag on the boiling curve, ranking 15th to 21st of 22. GPT-6 Luna is an odd case. Its boiling error is 0.82 decades, yet it has the best instant balanced accuracy, 0.83, and the second-lowest melting error, 27 K. Haiku 5.5 has the worst balanced accuracy, 0.63, and it says no liquid on 25 of the 33 lines used to measure boiling error.

{{time-charts}}

## Where the picture breaks

Darker cells are points where more models answered wrong, at 64 × 64.

When models answer instantly, errors pile up in four places. The first is liquid above the critical pressure, which models call supercritical or ice. The second is vapor below 0 °C. The third is the top of the boiling curve. The fourth is hot gas just below the critical pressure, where 92% of the wrong answers are the word supercritical.

With reasoning allowed, most of this clears. On the 18 models tested both ways, the share wrong falls from 61% to 14% across three regions: liquid above the critical pressure, vapor below 0 °C, and vapor within a decade below the critical pressure. What remains lies along the top of the melting curve, in the hot high-pressure corner and in the cold low-pressure corner. There is also a pale wash over hot vapor, where Sonnet 5, Opus 5, Sonnet 4.6 and Haiku 4.5 say supercritical. Across all 24 models, that wash holds 36% of the remaining wrong answers.

{{missed-maps}}

The table below shows the same errors by region: the share of each region's points a model got wrong. "Vapor below 0 °C" is the vapor side of the sublimation curve.

Answering instantly in kelvin and megapascals, Sonnet 4.5, GPT-4o mini, GPT-4.1, Haiku 4.5 and Opus 4.7 miss 92–100% of the vapor below 0 °C. They draw a wall of ice down to near vacuum. From GPT-5.2 on, most OpenAI models go the other way. They get this vapor right by putting the sublimation curve 0.8 to 3.6 decades too high, and they lose ordinary ice instead, missing 22–54% of it.

"Liquid above the critical pressure" is the largest miss on average: 68% instantly and 13% with reasoning allowed. Supercritical answers make up 39 of those 68 percentage points and 1 of the 13. Right next to the critical lines, the word is a naming miss and not a physics miss. But only 13 of the 39 percentage points lie within 65 K of the critical temperature. The other 26 lie further below it, 6 of them below 400 K, where water is an ordinary dense liquid.

If we count every supercritical answer as right in the rows "Liquid above the critical pressure" and "Vapor above the critical temperature", the liquid row falls to 29% instantly and 11% with reasoning allowed. "Vapor below 0 °C" is then the largest miss instantly, at 56%.

{{region-table}}

## The same conditions in different units

We asked every point of the 32 × 32 grid in six sets of units (16 × 16 with reasoning allowed): kelvin or Celsius, with pascals, megapascals or atmospheres. We also asked kelvin with megapascals and Celsius with atmospheres a second time, which measures noise. A model has a units problem only if it agrees with itself less across units than across repeats. In the chart below, its two dots then sit apart. Answering instantly, all 22 do, by up to 0.20. Noise can hide a gap: GPT-4o has the lowest agreement across units of any model, 75%, but it agrees with itself on only 81% of repeats, so its gap is only 0.06.

No unit is bad for every model. Celsius or kelvin makes no overall difference to instant balanced accuracy (0.01, −0.01 to 0.03). Atmospheres help only on everyday water (250–400 K, 10 kPa to 10 MPa), by 0.05 over pascals (0.01 to 0.08). Most of the damage is specific to each model. Sonnet 4.5 misses all of the vapor below 0 °C in kelvin and megapascals and almost none of it in atmospheres. Sonnet 4.6 misses 73–100% of it in every set of units except Celsius and megapascals, where it misses 4%.

Changing the unit can also change how a number is written, so a gap is not purely a unit effect. Pressures in pascals switch to scientific notation at `1.000e6 pascals`, and the true answer does not change there. Yet at 64 × 64, vapor answers from Haiku 5.5, GPT-5.2 and GPT-6 Luna drop at that switch by 24 to 48 percentage points more than the average drop at the same pressures in megapascals and atmospheres.

Choose a model to compare its maps. Numbers here are at 32 × 32 unless a sentence says otherwise, and agreement with reasoning allowed is at 16 × 16. So miss rates differ from the 64 × 64 table above.

{{units-maps}}

This chart shows the boiling point at 1 atm, read off each instant map. It always shows instant answers, even if you choose "Reasoning allowed" above. Each row runs from the lowest to the highest reading across up to six sets of units, and the line marks 373 K. A reading needs at least three liquid answers below the boiling point, so only 104 of a possible 132 maps give one, and Haiku 5.5 gives none. Readings are rough: on a plain repeat, a reading moves by a median of 10 K.

Haiku 4.5 ranges over 206 K, from 414 K to 619 K. Sonnet 4.6 never comes near the truth, at 451 K to 635 K. GPT-6 Sol's six readings span 17 K, all within 12 K of the truth.

{{boil-chart}}

## Where models reason

Nine models skip reasoning on 11% to 89% of replies. Darker cells took more reasoning. Near means less than a sixteenth of the way across the map from a boundary or critical line.

Eight of the nine reason more often near a line, and all nine spend more tokens there. Points near a line are harder for three of the four models with instant answers to compare. They miss 28–41% of nearby points and 5–12% of the rest. Opus 5 does not fit, at 37% and 38%, because most of its errors are hot gas far from any line.

Drawn on the map, the reasoning of Opus 5.5 and Sonnet 5.5 traces the sublimation curve, the boiling curve and the critical pressure. It then fills the map above about 200 MPa on Opus 5.5 and above about 1 GPa on Sonnet 5.5, even in the hot corner where there is no boundary.

Below 100 MPa the pattern is clean: Opus 5.5 reasons on 71% of nearby points and 9% of the rest. Above 1 GPa, six of the nine reason on 90–100% of points, Opus 4.8 on 78%, Sonnet 5 on 66% and Fable 5.1 on 35%.

Each panel uses that model's finest grid: 64 × 64 for Opus 4.8, Sonnet 5 and Opus 5, and 128 × 128 for the rest. Models that reason on nearly every reply are left out.

{{thoughts}}

## Every score

Scores use kelvin and megapascals at 64 × 64, so a row can differ from the explorer, which scores each map on the grid it is drawn on. The columns for replies that reasoned and for reasoning tokens pool every grid and set of units.

{{score-table}}

## How this was done

### Questions

Each question is one user message: "Pure water at equilibrium, at 450.00 kelvin and 2.000 megapascals: solid, liquid, vapor, or supercritical? Do not say anything else." We set no system prompt, but Anthropic adds its own that describes the reply format. Pressures are plain decimals from 0.001 up to 1,000,000 of the unit, and scientific notation outside that range, such as 1.230e-8 megapascals.

The grid has one randomly placed point per cell, from 16 × 16 up to 128 × 128, over 170–900 K and 0.01 Pa to 20 GPa. A point within 2 K or 0.03 decades of a boundary or critical line is redrawn in the same cell, so no label depends on the small uncertainty in the IAPWS curves.

### Truth

We use IAPWS R14-08 for the melting and sublimation curves, and the saturation equation of Wagner and Pruss from IAPWS SR1-86(1992), which agrees with IAPWS-95 to within 0.01%. Supercritical means above both the critical temperature and the critical pressure. Hot gas below the critical pressure counts as vapor, and cold fluid above it counts as liquid. Those two edges are naming conventions, not phase boundaries, and the prompt does not state them.

### No tools

The request code accepts a fixed list of keys. None of them can carry a tool, and any other key raises an error. A JSON schema limits a reply to the four phase words: `output_config.format` at Anthropic and strict `response_format` at OpenAI. Of about 1.2 million replies, 23 got past it. Two were a bare phase word and count as that word. The other 21 count as wrong. Automatic fallbacks to another model are off.

### Two conditions

Instant means no reasoning. Reasoning allowed means a setting that let the model reason in tests. That was a 1,024-token budget on the five oldest Claude models, adaptive thinking at low effort on Opus 4.8, Opus 5, Sonnet 5.5 and Haiku 5.5, medium effort on Sonnet 5, and low effort on eight OpenAI models. The six models that cannot turn reasoning off run at their lowest effort. GPT-4o, GPT-4o mini and GPT-4.1 have no reasoning. Opus 4.7 never used adaptive thinking in 15 test calls, and it rejects a fixed budget.

### Scoring

Balanced accuracy averages the four phases, because 61% of the grid counts as vapor. Boundary error compares a model's boundary with the one the true labels give on the same points, so a perfect map scores 0. Sublimation is scored by region, because many maps have no such boundary to measure. Trends in kelvin and megapascals use the 64 × 64 grid, the finest that every model has. Trends across units use 32 × 32.

These scores use the textbook definition of supercritical. Accepting the word for hot gas below the critical pressure moves the best instant score from 0.83 to 0.84. Also accepting it for fluid above the critical pressure, at 0.9 of the critical temperature or hotter, moves it to 0.87. Accepting it at any temperature moves it to 0.92. The sketch stays at 0.94, so it still beats every instant map.

About half the weight of balanced accuracy sits on points whose true phase depends on the critical temperature and pressure lines, and OpenAI's instant trend depends on those points. It is 0.05 a year (0.01 to 0.10) as scored, and −0.02 (−0.09 to 0.06) if supercritical is accepted at any temperature. But that rule also counts "supercritical" as right for water at 300 K and 100 MPa, which is plainly a liquid.

### Cost

{{spend}}

### Limits

Model retirements cut Anthropic's history off at September 2025 and OpenAI's at July 2024, so the two providers cover different periods. Release date stands in for generation.

Reasoning settings differ between models, so the reasoning comparison is not fully controlled.

Models use their default sampling settings, so asking the same question twice can give different answers.

We tested 143 trends and 84 unit gaps without correcting for the number of tests. By chance alone about 7 trends and 4 gaps would look clear. 26 and 13 did.

Ice VII above the critical temperature is barely sampled.

The experiment was conducted by Claude Opus 5.5, which is one of the graded models.

### Sources

1. [IAPWS R14-08(2011)](https://iapws.org/technical-guidance/release/MeltSub), the pressure along the melting and sublimation curves.
2. [IAPWS SR1-86(1992)](https://iapws.org/technical-guidance/release/Supp-sat), saturation properties.
3. [How does a blind model see the Earth?](https://outsidetext.substack.com/p/how-does-a-blind-model-see-the-earth) by henry, the map experiment this one follows.
