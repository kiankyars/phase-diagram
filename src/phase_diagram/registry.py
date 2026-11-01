"""Every model in the experiment, with the request settings that make them comparable.

`floor` holds the lowest reasoning each API allows, so that the "direct" arm asks all
models for the same thing: answer now, without deliberating. Six models cannot turn
reasoning off; `thinks` marks them, and their token use is reported with their results.
No request ever sets a sampling parameter, because current models reject them, so every
model answers at its default temperature.

The pilot showed that reasoning, even a few dozen tokens of it, matters more than
generation. `reasoned` therefore holds the settings for a second arm: a low setting
that lets each model reason on this prompt, chosen per model from five test conditions.
The six `thinks` models are already allowed to, so their direct answers double as
reasoned answers. "Low" alone would not do: adaptive thinking at low effort skipped
thinking altogether on Opus 4.6, and on most calls for Sonnet 4.6 and Sonnet 5.

Allowed does not mean used. Under the adaptive settings a model decides reply by reply,
and several reason on a minority of questions, so `share_thought` in the results says
how often each one did.

GPT-4o, GPT-4o mini and GPT-4.1 have no reasoning to turn on. Opus 4.7 has no reasoned
arm either: adaptive thinking never engaged in 15 calls at any effort, and it rejects a
fixed budget.

Prices are standard USD per million tokens, read from the providers' pricing pages on
2026-10-31. Batch is half of standard on both platforms, and it is the tier used here.

Left out on purpose:
  search / deep-research models   they have built-in web access, which breaks the test
  *-pro tiers                     $15-$600 per MTok of heavy reasoning, unaffordable at any grid
  *-codex                         coding-specialised variants of models already included
  *chat-latest aliases            moving targets with no pinned snapshot
  mythos, rosalind, cyber         restricted access
  obscure variants                GPT-4o Nov, GPT-4.1 mini, GPT-5 nano, GPT-5.4 mini and
                                  nano, GPT-5.6 Terra and Luna. All work, but the panel is
                                  OpenAI's flagship line plus its current lineup, and each
                                  of those is run deeper instead.

Not reachable from the API organization used (HTTP 404 on 2026-10-31):
  gpt-3.5-turbo*, gpt-4, gpt-4-turbo, gpt-4o-2024-05-13, gpt-4.1-nano, o1, o3-mini,
  o4-mini, and the base models; gpt-5 and gpt-5-mini need organization verification.
  Claude models older than Sonnet 4.5 are retired.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BATCH_DISCOUNT = 0.5

LOW_EFFORT = {"output_config": {"effort": "low"}}
THINKING_OFF = {"thinking": {"type": "disabled"}}
BETWEEN_TOOLS = {"thinking": {"type": "between_tools"}}  # how Sonnet 5.5 turns thinking off

NO_REASONING = {"reasoning_effort": "none"}
LOW = {"reasoning_effort": "low"}

ADAPTIVE_LOW = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}
ADAPTIVE_MEDIUM = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}
SMALLEST_BUDGET = {"thinking": {"type": "enabled", "budget_tokens": 1024}}

# OpenAI batch queue limits for the organization used, in tokens. None means the
# limits page gives no number for the model.
SMALL_QUEUE = 1_350_000
LARGE_QUEUE = 20_000_000


@dataclass(frozen=True)
class Model:
    id: str
    label: str
    provider: str  # anthropic | openai
    family: str
    released: str
    price_in: float
    price_out: float
    floor: dict = field(default_factory=dict)
    thinks: bool = False
    queue_tokens: int | None = None
    reasoned: dict | None = None  # settings for the reasoned arm, if it needs a second pass

    @property
    def max_tokens(self) -> int:
        return 4096 if self.thinks else 32


def _claude(id, label, family, released, price_in, price_out, floor=None, thinks=False, reasoned=ADAPTIVE_LOW):
    reasoned = None if thinks else reasoned
    return Model(id, label, "anthropic", family, released, price_in, price_out, floor or {}, thinks, None, reasoned)


def _gpt(id, label, family, released, price_in, price_out, floor=None, thinks=False, queue=SMALL_QUEUE):
    reasoned = LOW if floor == NO_REASONING else None
    return Model(id, label, "openai", family, released, price_in, price_out, floor or {}, thinks, queue, reasoned)


MODELS = (
    _claude("claude-sonnet-4-5-20250929", "Sonnet 4.5", "sonnet", "2025-09-29", 3, 15, reasoned=SMALLEST_BUDGET),
    _claude("claude-haiku-4-5-20251001", "Haiku 4.5", "haiku", "2025-10-15", 1, 5, reasoned=SMALLEST_BUDGET),
    _claude("claude-opus-4-5-20251101", "Opus 4.5", "opus", "2025-11-24", 5, 25, reasoned=SMALLEST_BUDGET),
    _claude("claude-opus-4-6", "Opus 4.6", "opus", "2026-02-04", 5, 25, reasoned=SMALLEST_BUDGET),
    _claude("claude-sonnet-4-6", "Sonnet 4.6", "sonnet", "2026-02-17", 3, 15, reasoned=SMALLEST_BUDGET),
    _claude("claude-opus-4-7", "Opus 4.7", "opus", "2026-04-14", 5, 25, reasoned=None),
    _claude("claude-opus-4-8", "Opus 4.8", "opus", "2026-05-28", 5, 25),
    _claude("claude-fable-5", "Fable 5", "fable", "2026-06-07", 10, 50, LOW_EFFORT, thinks=True),
    _claude("claude-sonnet-5", "Sonnet 5", "sonnet", "2026-06-29", 2, 10, THINKING_OFF, reasoned=ADAPTIVE_MEDIUM),
    _claude("claude-opus-5", "Opus 5", "opus", "2026-07-24", 5, 25, THINKING_OFF),
    _claude("claude-fable-5-1", "Fable 5.1", "fable", "2026-08-28", 10, 50, LOW_EFFORT, thinks=True),
    _claude("claude-opus-5-5", "Opus 5.5", "opus", "2026-09-21", 4, 20, LOW_EFFORT, thinks=True),
    _claude("claude-sonnet-5-5", "Sonnet 5.5", "sonnet", "2026-09-28", 2, 10, BETWEEN_TOOLS),
    _claude("claude-haiku-5-5", "Haiku 5.5", "haiku", "2026-10-07", 0.10, 0.50, THINKING_OFF),
    _gpt("gpt-4o-mini-2024-07-18", "GPT-4o mini", "small", "2024-07-18", 0.15, 0.60, queue=LARGE_QUEUE),
    _gpt("gpt-4o-2024-08-06", "GPT-4o", "gpt", "2024-08-06", 2.50, 10, queue=None),
    _gpt("gpt-4.1-2025-04-14", "GPT-4.1", "gpt", "2025-04-14", 2, 8),
    _gpt("o3-2025-04-16", "o3", "o-series", "2025-04-16", 2, 8, LOW, thinks=True),
    _gpt("gpt-5.1-2025-11-13", "GPT-5.1", "gpt", "2025-11-13", 1.25, 10, NO_REASONING),
    _gpt("gpt-5.2-2025-12-11", "GPT-5.2", "gpt", "2025-12-11", 1.75, 14, NO_REASONING),
    _gpt("gpt-5.4-2026-03-05", "GPT-5.4", "gpt", "2026-03-05", 2.50, 15, NO_REASONING),
    _gpt("gpt-5.5-2026-04-23", "GPT-5.5", "gpt", "2026-04-23", 5, 30, NO_REASONING),
    _gpt("gpt-5.6-sol", "GPT-5.6 Sol", "gpt", "2026-06-23", 4, 20, NO_REASONING),
    _gpt("gpt-6-astra", "GPT-6 Astra", "gpt", "2026-08-27", 10, 50, LOW, thinks=True),
    _gpt("gpt-6-sol", "GPT-6 Sol", "gpt", "2026-09-14", 2, 10, NO_REASONING),
    _gpt("gpt-6-luna", "GPT-6 Luna", "small", "2026-09-14", 0.10, 0.50, NO_REASONING, queue=LARGE_QUEUE),
    _gpt("gpt-6.1-sol", "GPT-6.1 Sol", "gpt", "2026-09-27", 2, 10, LOW, thinks=True),
    _gpt("gpt-6.1-terra", "GPT-6.1 Terra", "gpt", "2026-10-13", 1, 5, NO_REASONING),
)
BY_ID = {m.id: m for m in MODELS}


def batch_cost(model: Model, input_tokens: float, output_tokens: float) -> float:
    return BATCH_DISCOUNT * (input_tokens * model.price_in + output_tokens * model.price_out) / 1e6
