"""Stage 0: one small live call per model and setting, to learn what each one accepts.

For every model this finds the lowest reasoning setting the API takes and measures
what a few answers cost in tokens. Results go to data/probe.json. It uses the
synchronous APIs at standard prices, so it stays deliberately small.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import anthropic
import openai

from phase_diagram import framings, sealed
from phase_diagram.clients import anthropic_client, openai_client
from phase_diagram.parse import parse_answer
from phase_diagram.registry import MODELS, Model

OUT = Path("data/probe.json")

# One condition from each region, including one a person would need to think about.
POINTS = [(451.37, 2.138e6), (701.42, 3.17e7), (268.91, 1.846e8), (612.33, 7.412e6)]

OPENAI_FLOORS = [
    {"reasoning_effort": "none"},
    {"reasoning_effort": "minimal"},
    {"reasoning_effort": "low"},
    {},
]
CANONICAL = framings.BY_KEY[framings.CANONICAL]


def call(client, model: Model, prompt: str, arm: str = "direct") -> dict:
    body = sealed.build_body(model, prompt, arm)
    started = time.monotonic()
    if model.provider == "anthropic":
        raw = client.messages.create(**body).model_dump(mode="json")
        record = sealed.audit_messages(raw)
    else:
        raw = client.chat.completions.create(**body).model_dump(mode="json")
        record = sealed.audit_chat(raw)
    record["seconds"] = round(time.monotonic() - started, 2)
    record["usage"] = raw.get("usage")
    return record


def find_floor(client, model: Model) -> tuple[Model | None, str | None]:
    """Return the model with the first floor its API accepts, for models that reason."""
    prompt = framings.render(*POINTS[0], CANONICAL)
    if model.provider == "anthropic" or not model.floor:
        candidates = [model.floor]
    else:
        candidates = OPENAI_FLOORS
    error = None
    for floor in candidates:
        attempt = replace(model, floor=floor)
        try:
            call(client, attempt, prompt)
            return attempt, None
        except (anthropic.BadRequestError, openai.BadRequestError) as exc:
            error = str(exc)[:200]
        except (anthropic.NotFoundError, openai.NotFoundError) as exc:
            return None, f"not found: {str(exc)[:160]}"
    return None, error


def probe(model: Model) -> dict:
    client = anthropic_client(max_retries=4) if model.provider == "anthropic" else openai_client(max_retries=4)
    try:
        usable, error = find_floor(client, model)
        if usable is None:
            return {"id": model.id, "ok": False, "error": error}
        answers = []
        for point in POINTS:
            reply = call(client, usable, framings.render(*point, CANONICAL))
            label, status = parse_answer(reply["text"])
            answers.append({**reply, "label": label, "status": status})
    except (anthropic.APIError, openai.APIError) as exc:
        return {"id": model.id, "ok": False, "error": str(exc)[:200]}
    return {"id": model.id, "ok": True, "floor": usable.floor, "answers": answers}


def main() -> None:
    wanted = set(sys.argv[1:])
    provider = {"anthropic", "openai"} & wanted
    chosen = [m for m in MODELS if m.provider in provider or m.id in wanted or not wanted]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(probe, chosen))
    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    existing.update({r["id"]: r for r in results})
    OUT.write_text(json.dumps(existing, indent=1))
    print(f"{'model':32s} {'floor':24s} {'in':>4s} {'out':>5s} {'reas':>5s} {'sec':>5s}  answers")
    for r in results:
        if not r["ok"]:
            print(f"{r['id']:32s} FAILED {r['error']}")
            continue
        a = r["answers"]
        mean = lambda key: sum((x[key] or 0) for x in a) / len(a)
        words = " ".join(repr(x["text"].strip())[:16] for x in a)
        print(
            f"{r['id']:32s} {json.dumps(r['floor'])[:24]:24s} {mean('input_tokens'):4.0f} "
            f"{mean('output_tokens'):5.0f} {mean('reasoning_tokens'):5.0f} {mean('seconds'):5.1f}  {words}"
        )


if __name__ == "__main__":
    main()
