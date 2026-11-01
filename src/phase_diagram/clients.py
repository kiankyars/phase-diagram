"""API clients. Keys come from the .env file and never reach a log or a data file."""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
import openai
from dotenv import dotenv_values

ENV_FILE = Path(os.environ.get("PHASE_DIAGRAM_ENV", "~/.env")).expanduser()


def _key(name: str) -> str:
    value = dotenv_values(ENV_FILE).get(name)
    if not value:
        raise RuntimeError(f"{name} is not set in {ENV_FILE}")
    return value


def anthropic_client(**options) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key=_key("ANTHROPIC_API_KEY"), base_url="https://api.anthropic.com", **options
    )


def openai_client(**options) -> openai.OpenAI:
    return openai.OpenAI(api_key=_key("OPENAI_API_KEY"), base_url="https://api.openai.com/v1", **options)
