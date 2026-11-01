"""Rebuild answers OpenAI filed under `refusal`, from the raw replies, without asking again.

Until `sealed.audit_chat` read that field, such a reply was stored as an empty, invalid
answer, and its key counted as done. Running this twice changes nothing.
"""

from __future__ import annotations

import gzip
import json

import pandas as pd

from phase_diagram.parse import parse_answer
from phase_diagram.pipeline import ANSWERS, RAW, slug, split_id
from phase_diagram.registry import MODELS
from phase_diagram.sealed import audit_chat


def refused(path) -> dict[str, str]:
    """The text of every reply in a raw file that has a refusal and no content."""
    found = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            body = (data.get("response") or {}).get("body") or {}
            message = (body.get("choices") or [{}])[0].get("message") or {}
            if message.get("refusal") and not message.get("content"):
                found[data["custom_id"]] = audit_chat(body)["text"]
    return found


def main() -> None:
    kept = dropped = 0
    for model in MODELS:
        if model.provider != "openai":
            continue
        for path in sorted((RAW / slug(model.id)).glob("*.jsonl.gz")):
            found = refused(path)
            if not found:
                continue
            answers = ANSWERS / slug(model.id) / path.name.replace(".jsonl.gz", ".parquet")
            table = pd.read_parquet(answers)
            for cid, text in found.items():
                arm, framing, pid, rep = split_id(cid)
                row = (table["arm"] == arm) & (table["framing"] == framing) & (table["pid"] == pid) & (table["rep"] == rep)
                label, status = parse_answer(text)
                table.loc[row, ["label", "status", "text"]] = [label, status, text[:200]]
                kept += label is not None
                dropped += label is None
                print(f"{model.id} {cid}: {text[:60]!r} -> {label}")
            unfinished = answers.with_suffix(".tmp")
            table.to_parquet(unfinished, index=False)
            unfinished.replace(answers)
    print(f"{kept} answers recovered, {dropped} true refusals left invalid")


if __name__ == "__main__":
    main()
