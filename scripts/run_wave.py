"""Walk each model through waves in order, e.g. `run_wave.py core retest --provider openai`."""

from __future__ import annotations

import argparse

from phase_diagram.pipeline import WAVES, run, run_flex
from phase_diagram.registry import BY_ID, MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("waves", nargs="+", choices=sorted(WAVES))
    parser.add_argument("--provider", choices=["anthropic", "openai"])
    parser.add_argument("--models", nargs="*", help="model ids; default is every model")
    parser.add_argument("--skip", nargs="*", default=[], help="model ids to leave out")
    parser.add_argument("--poll", type=int, default=60, help="seconds between polls")
    parser.add_argument("--cap", type=float, help="per-model cap in USD for this run")
    parser.add_argument("--flex", action="store_true", help="use OpenAI's flex tier instead of Batch")
    args = parser.parse_args()

    models = [BY_ID[m] for m in args.models] if args.models else list(MODELS)
    models = [m for m in models if m.id not in args.skip]
    if args.provider:
        models = [m for m in models if m.provider == args.provider]
    caps = {m.id: args.cap for m in models} if args.cap else None
    (run_flex if args.flex else run)(args.waves, models, poll_seconds=args.poll, caps=caps)


if __name__ == "__main__":
    main()
