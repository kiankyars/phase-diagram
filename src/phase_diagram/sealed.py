"""The sandbox: build the only requests that can leave, and audit every reply.

The models get no tools because no tool can be expressed. Each API has an allowlist
of request keys that contains no tool, plugin or search field, and a request with any
other key raises before it is sent. The request sets no system prompt and holds no
conversation: each request is one question with no memory of any other.

The reply is held to a JSON schema whose only value is one of the four phase words,
enforced by constrained decoding on the provider's side. This does two jobs. Newer
models told to say nothing else often reason aloud anyway once thinking is off, so
without the schema a "direct" answer would not be direct and generations would not be
comparable. And it leaves a model no way to say anything else at all. Nothing a model
writes is ever executed.

After the fact, `audit_*` looks for any sign of tool activity in the raw reply. An
answer that shows any is thrown out, and the count is reported.

Refusal fallbacks (`fallbacks`) are off on purpose. They would silently swap in a
different model and put its answers on this model's map.
"""

from __future__ import annotations

from .parse import CLASSES
from .registry import Model

PHASE_SCHEMA = {
    "type": "object",
    "properties": {"phase": {"type": "string", "enum": list(CLASSES)}},
    "required": ["phase"],
    "additionalProperties": False,
}
ANTHROPIC_FORMAT = {"type": "json_schema", "schema": PHASE_SCHEMA}
OPENAI_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "phase", "strict": True, "schema": PHASE_SCHEMA},
}

ALLOWED_KEYS = {
    "messages": {"model", "max_tokens", "messages", "thinking", "output_config"},
    "chat": {
        "model",
        "messages",
        "max_completion_tokens",
        "max_tokens",
        "reasoning_effort",
        "response_format",
        "store",
        "service_tier",
    },
}
ALLOWED_NESTED = {"output_config": {"effort", "format"}, "thinking": {"type", "budget_tokens"}}

REASONED_TOKENS = 4096
FLEX = "flex"


def seal(api: str, body: dict) -> dict:
    """Refuse any request that could give a model more than one question to answer."""
    extra = set(body) - ALLOWED_KEYS[api]
    if extra:
        raise ValueError(f"forbidden request keys for {api}: {sorted(extra)}")
    for key, allowed in ALLOWED_NESTED.items():
        if key in body and set(body[key]) - allowed:
            raise ValueError(f"forbidden keys inside {key}: {sorted(set(body[key]) - allowed)}")
    if body.get("output_config", {}).get("format", ANTHROPIC_FORMAT) != ANTHROPIC_FORMAT:
        raise ValueError("output format must be the phase schema")
    if body.get("response_format", OPENAI_FORMAT) != OPENAI_FORMAT:
        raise ValueError("response format must be the phase schema")
    if body.get("service_tier", FLEX) != FLEX:
        raise ValueError("the only tier a request may name is flex")
    return body


def build_body(model: Model, prompt: str, arm: str = "direct", tier: str | None = None) -> dict:
    if arm == "reasoned" and model.reasoned is None:
        raise ValueError(f"{model.id} has no reasoned arm")
    settings = model.floor if arm == "direct" else model.reasoned
    limit = model.max_tokens if arm == "direct" else REASONED_TOKENS
    message = [{"role": "user", "content": prompt}]
    if model.provider == "anthropic":
        config = {**settings.get("output_config", {}), "format": ANTHROPIC_FORMAT}
        body = {"model": model.id, "max_tokens": limit, "messages": message, **settings, "output_config": config}
        return seal("messages", body)
    body = {
        "model": model.id,
        "max_completion_tokens": limit,
        "messages": message,
        "store": False,
        "response_format": OPENAI_FORMAT,
        **settings,
    }
    if tier:
        body["service_tier"] = tier
    return seal("chat", body)


def audit_messages(message: dict) -> dict:
    """Normalize an Anthropic message and flag any block that is not text or thinking."""
    blocks = message.get("content") or []
    usage = message.get("usage") or {}
    server_tools = usage.get("server_tool_use") or {}
    used_server_tool = any(v for v in server_tools.values()) if isinstance(server_tools, dict) else bool(server_tools)
    return {
        "text": "".join(b.get("text", "") for b in blocks if b.get("type") == "text"),
        "stop": message.get("stop_reason"),
        "resolved_model": message.get("model"),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens", 0),
        "tool_use": used_server_tool
        or any(b.get("type") not in ("text", "thinking", "redacted_thinking") for b in blocks),
    }


def audit_chat(body: dict) -> dict:
    """Normalize an OpenAI reply. A few schema-valid answers arrive under `refusal`."""
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "text": message.get("content") or message.get("refusal") or "",
        "stop": choice.get("finish_reason"),
        "resolved_model": body.get("model"),
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "reasoning_tokens": details.get("reasoning_tokens", 0),
        "tool_use": bool(message.get("tool_calls") or message.get("function_call") or message.get("annotations")),
    }
