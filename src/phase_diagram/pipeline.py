"""Run waves of questions through the batch APIs, the cheapest tier on both platforms.

OpenAI's flex tier costs the same as Batch and is the route for a model whose Batch
queue has stalled, one request at a time instead of waiting in a queue (`run_flex`).

A wave is a list of `Spec`s: which arm, framing, grid level and replicate to ask. Each
model walks the waves in order, coarsest first, so a model that turns out to be
expensive can stop early without leaving a gap on the map, and a slow model holds back
only itself. Levels nest, so refining a model only asks for points it has not answered.

State lives on disk and the loop is restartable: a key is pending until an answer for
it exists, and each model has at most one batch open at a time. Spending is computed
from the token counts the APIs report, and a batch is refused before it is sent if
its estimate would cross the model's cap or the provider's cap.
"""

from __future__ import annotations

import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import numpy as np
import openai
import pandas as pd

from . import framings, grid, sealed
from .clients import anthropic_client, openai_client
from .framings import CANONICAL, FRAMINGS
from .parse import parse_answer
from .registry import MODELS, Model, batch_cost

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
ANSWERS = DATA / "answers"
BATCHES = DATA / "batches"
LOG = DATA / "run.log"

PROVIDER_CAP = {"anthropic": 480.0, "openai": 400.0}
MODEL_CAP = 45.0
MAX_ROUNDS = 6  # batches in a row that answer nothing new before a model leaves a wave
SAFETY = 1.25  # margin on the measured cost per request
ANTHROPIC_QUEUE = 280_000
ANTHROPIC_CHUNK = 50_000
OPENAI_CHUNK = 45_000
DEFAULT_QUEUE_TOKENS = 1_000_000
QUEUE_FILL = 0.8
FLEX_WORKERS = 32
FLEX_CHUNK = 500  # answers between saves
FLEX_PAUSE = 20  # seconds to wait after a chunk the tier mostly turned away

# A batch can sit for hours on its last few requests. Past this age and share, cancel it:
# the finished answers come back and the rest are asked again.
STRAGGLER_MINUTES = 20
STRAGGLER_SHARE = 0.98

# Cost-per-request guesses for a model's first batch, before anything is measured.
FIRST_INPUT_TOKENS = {"anthropic": 300, "openai": 120}
FIRST_OUTPUT_TOKENS = {False: 20, True: 300}


@dataclass(frozen=True)
class Spec:
    framing: str
    level: int
    rep: int = 0
    arm: str = "direct"


OTHERS = [f.key for f in FRAMINGS[1:]]

WAVES = {
    "pilot": [Spec(CANONICAL, 3)],
    "units": [Spec(f, 3) for f in OTHERS],
    "core": [Spec(f.key, 2) for f in FRAMINGS],
    "retest": [Spec(CANONICAL, 2, rep=1), Spec("C_atm", 2, rep=1)],
    "detail": [Spec(CANONICAL, 1)],
    "fine": [Spec(CANONICAL, 0)],
    "units-fine": [Spec(f, 1) for f in OTHERS],
    "reasoned": [Spec(CANONICAL, 3, arm="reasoned")],
    "reasoned-units": [Spec(f, 3, arm="reasoned") for f in OTHERS],
    "reasoned-retest": [Spec(CANONICAL, 3, rep=1, arm="reasoned"), Spec("C_atm", 3, rep=1, arm="reasoned")],
    "reasoned-core": [Spec(CANONICAL, 2, arm="reasoned")],
    "reasoned-detail": [Spec(CANONICAL, 1, arm="reasoned")],
    "reasoned-fine": [Spec(CANONICAL, 0, arm="reasoned")],
}


def log(message: str) -> None:
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def load_grid() -> pd.DataFrame:
    path = DATA / "grid.parquet"
    if not path.exists():
        grid.build_grid().to_parquet(path, index=False)
    return pd.read_parquet(path)


def custom_id(arm: str, framing: str, pid: int, rep: int) -> str:
    return f"{arm}-{framing}-{pid}-{rep}"


def split_id(cid: str) -> tuple[str, str, int, int]:
    arm, framing, pid, rep = cid.rsplit("-", 3)
    return arm, framing, int(pid), int(rep)


def slug(model_id: str) -> str:
    return model_id.replace(".", "_")


def manifest_paths(model: Model) -> list[Path]:
    return sorted((BATCHES / slug(model.id)).glob("*.json"))


def load_answers(model: Model) -> pd.DataFrame:
    files = sorted((ANSWERS / slug(model.id)).glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["arm", "framing", "pid", "rep", "input_tokens", "output_tokens"])
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def pending_ids(model: Model, wave: str, points: pd.DataFrame, answers: pd.DataFrame) -> list[str]:
    done = set(zip(answers["arm"], answers["framing"], answers["pid"], answers["rep"]))
    wanted: dict[str, None] = {}
    for spec in WAVES[wave]:
        if spec.arm == "reasoned" and model.reasoned is None:
            continue
        for pid in grid.at_level(points, spec.level)["pid"].to_numpy():
            if (spec.arm, spec.framing, int(pid), spec.rep) not in done:
                wanted[custom_id(spec.arm, spec.framing, int(pid), spec.rep)] = None
    ids = list(wanted)
    seed = int.from_bytes(f"{model.id}|{wave}".encode(), "little") % 2**32
    order = np.random.default_rng(seed).permutation(len(ids))
    return [ids[i] for i in order]


def next_work(model: Model, waves: list[str], points: pd.DataFrame, answers: pd.DataFrame) -> tuple[str, list[str]] | None:
    """The first of these waves with anything left to ask this model."""
    for wave in waves:
        ids = pending_ids(model, wave, points, answers)
        if ids:
            return wave, ids
    return None


class Progress:
    """Drops a model from a wave once batches stop answering anything new."""

    def __init__(self) -> None:
        self.fewest: dict[tuple[str, str], int] = {}
        self.fruitless: dict[tuple[str, str], int] = {}

    def waves_left(self, model: Model, waves: list[str]) -> list[str]:
        return [w for w in waves if self.fruitless.get((model.id, w), 0) < MAX_ROUNDS]

    def note(self, model: Model, wave: str, left: int) -> None:
        key = (model.id, wave)
        if left < self.fewest.get(key, left + 1):
            self.fewest[key] = left
            self.fruitless[key] = 0
        else:
            self.fruitless[key] += 1
            if self.fruitless[key] == MAX_ROUNDS:
                log(f"{model.id}: giving up on {left} keys in {wave}")


def committed(provider: str) -> float:
    """What a provider's models have spent, plus the estimate of every batch still open."""
    total = 0.0
    for model in MODELS:
        if model.provider != provider:
            continue
        answers = load_answers(model)
        total += spent(model, answers) if len(answers) else 0.0
        manifests = [json.loads(path.read_text()) for path in manifest_paths(model)]
        total += sum(m["estimate_usd"] for m in manifests if m["state"] == "open")
    return total


def per_request(model: Model, answers: pd.DataFrame, arm: str) -> tuple[float, float]:
    """Mean input and output tokens per request, measured once anything is measured."""
    seen = answers[answers["arm"] == arm]
    if len(seen):
        return float(seen["input_tokens"].mean()), float(seen["output_tokens"].mean())
    output = FIRST_OUTPUT_TOKENS[model.thinks or arm != "direct"] * (1 if arm == "direct" else 10)
    return FIRST_INPUT_TOKENS[model.provider], output


def spent(model: Model, answers: pd.DataFrame) -> float:
    return batch_cost(model, float(answers["input_tokens"].sum()), float(answers["output_tokens"].sum()))


def request_body(model: Model, cid: str, by_pid: pd.DataFrame, tier: str | None = None) -> dict:
    arm, framing, pid, _ = split_id(cid)
    row = by_pid.loc[pid]
    prompt = framings.render(float(row["T_K"]), float(row["p_Pa"]), framings.BY_KEY[framing])
    return sealed.build_body(model, prompt, arm, tier)


def record_answer(model: Model, cid: str, audit: dict, batch_id: str) -> dict:
    arm, framing, pid, rep = split_id(cid)
    label, status = parse_answer(audit["text"])
    if audit["tool_use"]:
        label, status = None, "tool_use"
    return {
        "arm": arm,
        "framing": framing,
        "pid": pid,
        "rep": rep,
        "label": label,
        "status": status,
        "text": audit["text"][:200],
        "stop": audit["stop"],
        "input_tokens": audit["input_tokens"],
        "output_tokens": audit["output_tokens"],
        "reasoning_tokens": audit["reasoning_tokens"],
        "resolved_model": audit["resolved_model"],
        "batch_id": batch_id,
    }


def save_results(model: Model, batch_id: str, raw_lines: list[str], rows: list[dict]) -> None:
    raw_path = RAW / slug(model.id) / f"{batch_id}.jsonl.gz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw_path, "wt", compresslevel=9) as f:
        f.write("\n".join(raw_lines) + "\n")
    if rows:
        out = ANSWERS / slug(model.id) / f"{batch_id}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        unfinished = out.with_suffix(".tmp")
        pd.DataFrame(rows).to_parquet(unfinished, index=False)
        unfinished.replace(out)  # another run may be reading this directory


def write_manifest(model: Model, batch_id: str, wave: str, n: int, estimate: float) -> None:
    path = BATCHES / slug(model.id) / f"{batch_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "batch_id": batch_id,
                "model": model.id,
                "provider": model.provider,
                "wave": wave,
                "n": n,
                "estimate_usd": round(estimate, 4),
                "state": "open",
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=1,
        )
    )


def close_manifest(path: Path, state: str, **extra) -> None:
    manifest = json.loads(path.read_text())
    manifest.update(state=state, closed=datetime.now(timezone.utc).isoformat(timespec="seconds"), **extra)
    path.write_text(json.dumps(manifest, indent=1))


def submit_anthropic(client, model: Model, ids: list[str], by_pid: pd.DataFrame) -> str:
    requests = [{"custom_id": cid, "params": request_body(model, cid, by_pid)} for cid in ids]
    return client.messages.batches.create(requests=requests).id


def blank_signatures(data: dict) -> dict:
    """Blank thinking signatures, opaque blobs that can identify the account."""
    if data["result"]["type"] == "succeeded":
        for block in data["result"]["message"]["content"]:
            if "signature" in block:
                block["signature"] = None
    return data


def collect_anthropic(client, model: Model, path: Path) -> bool:
    manifest = json.loads(path.read_text())
    batch_id = manifest["batch_id"]
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        return False
    raw_lines, rows = [], []
    for entry in client.messages.batches.results(batch_id):
        data = blank_signatures(entry.model_dump(mode="json"))
        raw_lines.append(json.dumps(data, separators=(",", ":")))
        if data["result"]["type"] == "succeeded":
            audit = sealed.audit_messages(data["result"]["message"])
            rows.append(record_answer(model, data["custom_id"], audit, batch_id))
    save_results(model, batch_id, raw_lines, rows)
    counts = batch.request_counts
    close_manifest(path, "done", answered=len(rows), errored=counts.errored, expired=counts.expired)
    return True


CHAT_ENDPOINT = "/v1/chat/completions"


def submit_openai(client, model: Model, ids: list[str], by_pid: pd.DataFrame) -> str:
    lines = [
        json.dumps(
            {"custom_id": cid, "method": "POST", "url": CHAT_ENDPOINT, "body": request_body(model, cid, by_pid)},
            separators=(",", ":"),
        )
        for cid in ids
    ]
    upload = client.files.create(file=("requests.jsonl", "\n".join(lines).encode()), purpose="batch")
    return client.batches.create(input_file_id=upload.id, endpoint=CHAT_ENDPOINT, completion_window="24h").id


def collect_openai(client, model: Model, path: Path) -> bool:
    manifest = json.loads(path.read_text())
    batch_id = manifest["batch_id"]
    batch = client.batches.retrieve(batch_id)
    if batch.status not in ("completed", "failed", "expired", "cancelled"):
        counts = batch.request_counts
        late = time.time() - batch.created_at > STRAGGLER_MINUTES * 60
        if batch.status == "in_progress" and late and counts.completed >= STRAGGLER_SHARE * counts.total:
            client.batches.cancel(batch_id)
            log(f"{model.id}: cancelled {batch_id} to recover {counts.completed} of {counts.total}")
        return False
    raw_lines, rows = [], []
    for file_id in (batch.output_file_id, batch.error_file_id):
        if file_id:
            raw_lines.extend(line for line in client.files.content(file_id).text.splitlines() if line)
    for line in raw_lines:
        data = json.loads(line)
        response = data.get("response") or {}
        if response.get("status_code") == 200:
            audit = sealed.audit_chat(response["body"])
            rows.append(record_answer(model, data["custom_id"], audit, batch_id))
    if raw_lines:
        save_results(model, batch_id, raw_lines, rows)
    state = "done" if batch.status == "completed" else batch.status
    errors = [e.model_dump(mode="json") for e in (batch.errors.data if batch.errors else [])][:3]
    close_manifest(path, state, answered=len(rows), errors=errors)
    return True


def ask_flex(client, model: Model, cid: str, by_pid: pd.DataFrame) -> dict | None:
    """One request at the flex tier, as the line a batch output file would hold."""
    try:
        reply = client.chat.completions.create(**request_body(model, cid, by_pid, sealed.FLEX))
    except openai.APIError:
        return None
    return {"custom_id": cid, "response": {"status_code": 200, "body": reply.model_dump(mode="json")}}


def settle_batches(client, model: Model, poll_seconds: int) -> None:
    """Wait out any batch still open, so that no key is ever asked twice."""
    while True:
        waiting = [p for p in manifest_paths(model) if json.loads(p.read_text())["state"] == "open"]
        if not waiting:
            return
        for path in waiting:
            try:
                collect_openai(client, model, path)
            except openai.APIError as exc:
                log(f"{model.id}: poll failed, will retry: {str(exc)[:120]}")
        time.sleep(poll_seconds)


def run_flex(waves: list[str], models: list[Model], poll_seconds: int = 60, caps: dict | None = None, workers: int = FLEX_WORKERS) -> None:
    """Drive `waves` through OpenAI's flex tier, each model on its own thread."""
    caps = {model.id: MODEL_CAP for model in models} | (caps or {})
    points = load_grid()
    by_pid = points.set_index("pid")
    client = openai_client(max_retries=6, timeout=900)
    flying: dict[str, float] = {}  # the estimate of each chunk other threads are asking now

    def ask_chunk(model: Model, wave: str, chunk: list[str], pool: ThreadPoolExecutor, estimate: float) -> int:
        replies = [r for r in pool.map(lambda cid: ask_flex(client, model, cid, by_pid), chunk) if r]
        batch_id = f"flex-{datetime.now(timezone.utc):%Y%m%d%H%M%S%f}"
        rows = [record_answer(model, r["custom_id"], sealed.audit_chat(r["response"]["body"]), batch_id) for r in replies]
        if rows:
            save_results(model, batch_id, [json.dumps(r, separators=(",", ":")) for r in replies], rows)
        write_manifest(model, batch_id, wave, len(chunk), estimate)
        close_manifest(BATCHES / slug(model.id) / f"{batch_id}.json", "done", answered=len(rows))
        log(f"{model.id}: flex answered {len(rows)} of {len(chunk)} in {wave}")
        return len(rows)

    def drive(model: Model) -> None:
        settle_batches(client, model, poll_seconds)
        progress = Progress()
        with ThreadPoolExecutor(workers) as pool:
            while True:
                answers = load_answers(model)
                work = next_work(model, progress.waves_left(model, waves), points, answers)
                if work is None:
                    return
                wave, ids = work
                progress.note(model, wave, len(ids))
                chunk = ids[:FLEX_CHUNK]
                tokens_in, tokens_out = per_request(model, answers, split_id(chunk[0])[0])
                estimate = SAFETY * batch_cost(model, tokens_in * len(chunk), tokens_out * len(chunk))
                so_far = spent(model, answers) if len(answers) else 0.0
                provider_total = committed("openai") + sum(flying.values())
                if so_far + estimate > caps[model.id] or provider_total + estimate > PROVIDER_CAP["openai"]:
                    log(f"{model.id}: capped at ${so_far:.2f}, next chunk ${estimate:.2f}")
                    return
                flying[model.id] = estimate
                answered = ask_chunk(model, wave, chunk, pool, estimate)
                del flying[model.id]
                if answered < len(chunk) / 2:
                    time.sleep(FLEX_PAUSE)

    with ThreadPoolExecutor(len(models)) as pool:
        list(pool.map(drive, models))
    table = status_table(models)
    log(f"flex run finished for {len(models)} models, total ${table['usd'].sum():.2f}")


def chunk_size(model: Model, input_tokens: float) -> int:
    if model.provider == "anthropic":
        return ANTHROPIC_CHUNK
    queue = model.queue_tokens or DEFAULT_QUEUE_TOKENS
    return min(OPENAI_CHUNK, int(QUEUE_FILL * queue / max(input_tokens, 1)))


def status_table(models: list[Model]) -> pd.DataFrame:
    rows = []
    for model in models:
        answers = load_answers(model)
        rows.append(
            {
                "model": model.id,
                "provider": model.provider,
                "answers": len(answers),
                "usd": round(spent(model, answers), 3) if len(answers) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def run(waves: list[str], models: list[Model], poll_seconds: int = 60, caps: dict | None = None) -> None:
    """Drive each model through `waves` until every key is answered or a cap stops it."""
    caps = {model.id: MODEL_CAP for model in models} | (caps or {})
    points = load_grid()
    by_pid = points.set_index("pid")
    clients = {}
    if any(m.provider == "anthropic" for m in models):
        clients["anthropic"] = anthropic_client(max_retries=6, timeout=600)
    if any(m.provider == "openai" for m in models):
        clients["openai"] = openai_client(max_retries=6, timeout=600)

    progress = Progress()
    capped: set[str] = set()
    while True:
        by_provider = {provider: committed(provider) for provider in clients}
        open_requests = 0
        waiting = False
        for model in models:
            client = clients[model.provider]
            collect = collect_anthropic if model.provider == "anthropic" else collect_openai
            open_manifests = [p for p in manifest_paths(model) if json.loads(p.read_text())["state"] == "open"]
            for path in open_manifests:
                try:
                    finished = collect(client, model, path)
                except (anthropic.APIError, openai.APIError) as exc:
                    log(f"{model.id}: poll failed, will retry: {str(exc)[:120]}")
                    finished = False
                if finished:
                    manifest = json.loads(path.read_text())
                    log(f"{model.id}: batch {manifest['batch_id']} {manifest['state']} answered={manifest.get('answered')}")
                else:
                    open_requests += json.loads(path.read_text())["n"]
                    waiting = True
            if any(json.loads(p.read_text())["state"] == "open" for p in open_manifests):
                continue

            answers = load_answers(model)
            work = next_work(model, progress.waves_left(model, waves), points, answers)
            if work is None or model.id in capped:
                continue
            wave, ids = work
            progress.note(model, wave, len(ids))
            arm = split_id(ids[0])[0]
            tokens_in, tokens_out = per_request(model, answers, arm)
            take = min(len(ids), chunk_size(model, tokens_in))
            if model.provider == "anthropic" and open_requests + take > ANTHROPIC_QUEUE:
                waiting = True
                continue
            estimate = SAFETY * batch_cost(model, tokens_in * take, tokens_out * take)
            so_far = spent(model, answers) if len(answers) else 0.0
            if so_far + estimate > caps[model.id]:
                log(f"{model.id}: capped at ${so_far:.2f} of ${caps[model.id]:.2f}, next batch ${estimate:.2f}")
                capped.add(model.id)
                continue
            if by_provider[model.provider] + estimate > PROVIDER_CAP[model.provider]:
                log(f"{model.id}: provider cap reached at ${by_provider[model.provider]:.2f}")
                capped.add(model.id)
                continue
            submit = submit_anthropic if model.provider == "anthropic" else submit_openai
            try:
                batch_id = submit(client, model, ids[:take], by_pid)
            except (anthropic.APIError, openai.APIError) as exc:
                log(f"{model.id}: submit failed, skipping this model: {str(exc)[:300]}")
                capped.add(model.id)
                continue
            write_manifest(model, batch_id, wave, take, estimate)
            open_requests += take
            by_provider[model.provider] += estimate
            waiting = True
            log(f"{model.id}: submitted {take} of {len(ids)} pending in {wave}, ~${estimate:.2f}, batch {batch_id}")
        if not waiting:
            break
        time.sleep(poll_seconds)
    table = status_table(models)
    log(f"run of {', '.join(waves)} finished for {len(models)} models, total ${table['usd'].sum():.2f}")
