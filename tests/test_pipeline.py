import json

import pandas as pd
import pytest

from phase_diagram import pipeline
from phase_diagram.pipeline import blank_signatures, committed, slug, spent
from phase_diagram.registry import BY_ID

MODEL = BY_ID["gpt-6-sol"]


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ANSWERS", tmp_path / "answers")
    monkeypatch.setattr(pipeline, "BATCHES", tmp_path / "batches")
    return tmp_path


def manifest(root, name, state, estimate):
    folder = root / "batches" / slug(MODEL.id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps({"state": state, "estimate_usd": estimate}))


def test_money_committed_counts_answers_and_open_batches_only(state):
    folder = state / "answers" / slug(MODEL.id)
    folder.mkdir(parents=True)
    answers = pd.DataFrame({"input_tokens": [1_000_000], "output_tokens": [1_000_000]})
    answers.to_parquet(folder / "one.parquet", index=False)
    manifest(state, "open", "open", 2.5)
    manifest(state, "done", "done", 9.0)
    assert committed("openai") == pytest.approx(spent(MODEL, answers) + 2.5)
    assert committed("anthropic") == 0


def test_signatures_are_blanked_and_nothing_else_changes():
    thinking = {"type": "thinking", "thinking": "hot gas", "signature": "CAIS"}
    data = {"result": {"type": "succeeded", "message": {"content": [thinking, {"type": "text", "text": "vapor"}]}}}
    blocks = blank_signatures(data)["result"]["message"]["content"]
    assert blocks == [{"type": "thinking", "thinking": "hot gas", "signature": None}, {"type": "text", "text": "vapor"}]
    assert blank_signatures({"result": {"type": "errored"}}) == {"result": {"type": "errored"}}
