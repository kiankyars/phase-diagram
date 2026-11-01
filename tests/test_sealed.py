import pytest

from phase_diagram.registry import BY_ID, MODELS
from phase_diagram.sealed import FLEX, audit_chat, audit_messages, build_body, seal


REASONED = [m for m in MODELS if m.reasoned]
CASES = [(m, "direct") for m in MODELS] + [(m, "reasoned") for m in REASONED]


@pytest.mark.parametrize("model, arm", CASES, ids=lambda x: getattr(x, "id", x))
def test_no_request_can_carry_a_tool(model, arm):
    body = build_body(model, "question", arm)
    for forbidden in (
        "tools",
        "tool_choice",
        "functions",
        "web_search_options",
        "mcp_servers",
        "container",
        "system",
        "temperature",
    ):
        assert forbidden not in body


@pytest.mark.parametrize(
    "api, body",
    [
        ("messages", {"model": "m", "tools": [{"type": "web_search_20260209"}]}),
        ("messages", {"model": "m", "system": "use your tools"}),
        ("messages", {"model": "m", "fallbacks": "default"}),
        ("chat", {"model": "m", "tools": []}),
        ("chat", {"model": "m", "web_search_options": {}}),
        ("chat", {"model": "m", "functions": []}),
        ("messages", {"model": "m", "output_config": {"format": {}}}),
        ("messages", {"model": "m", "output_config": {"format": {"type": "json_schema", "schema": {}}}}),
        ("chat", {"model": "m", "response_format": {"type": "json_object"}}),
    ],
)
def test_seal_rejects_anything_outside_the_allowlist(api, body):
    with pytest.raises(ValueError):
        seal(api, body)


@pytest.mark.parametrize("model, arm", CASES, ids=lambda x: getattr(x, "id", x))
def test_every_reply_is_held_to_the_phase_words(model, arm):
    body = build_body(model, "question", arm)
    if model.provider == "anthropic":
        schema = body["output_config"]["format"]["schema"]
    else:
        schema = body["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["phase"]["enum"] == ["solid", "liquid", "vapor", "supercritical"]
    assert schema["additionalProperties"] is False


def test_audit_flags_tool_activity():
    assert audit_messages({"content": [{"type": "server_tool_use"}], "usage": {}})["tool_use"]
    assert audit_messages(
        {"content": [{"type": "text", "text": "x"}], "usage": {"server_tool_use": {"web_search_requests": 1}}}
    )["tool_use"]
    assert audit_chat({"choices": [{"message": {"tool_calls": [{}]}}], "usage": {}})["tool_use"]


def test_audit_passes_a_plain_answer():
    reply = audit_messages(
        {
            "model": "m",
            "stop_reason": "end_turn",
            "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "Liquid"}],
            "usage": {"input_tokens": 40, "output_tokens": 12},
        }
    )
    assert reply["text"] == "Liquid"
    assert not reply["tool_use"]
    assert (reply["input_tokens"], reply["output_tokens"]) == (40, 12)


def test_every_model_that_can_reason_is_covered_and_the_rest_are_refused():
    cannot = {"gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06", "gpt-4.1-2025-04-14", "claude-opus-4-7"}
    for model in MODELS:
        if model.thinks or model.id in cannot:
            assert model.reasoned is None
            with pytest.raises(ValueError):
                build_body(model, "question", "reasoned")
        else:
            assert model.reasoned


def test_the_reasoned_arm_is_the_setting_the_floor_models_already_use():
    by_id = {m.id: m for m in MODELS}
    assert by_id["gpt-6-sol"].reasoned == by_id["gpt-6-astra"].floor
    assert by_id["claude-opus-5"].reasoned["output_config"] == by_id["claude-opus-5-5"].floor["output_config"]


def test_flex_is_the_only_tier_a_request_can_name():
    gpt = BY_ID["gpt-5.2-2025-12-11"]
    assert build_body(gpt, "question", tier=FLEX)["service_tier"] == FLEX
    with pytest.raises(ValueError):
        build_body(gpt, "question", tier="priority")
    with pytest.raises(ValueError):
        seal("messages", {"model": "any", "service_tier": FLEX})


def test_an_answer_filed_under_refusal_is_kept():
    body = {"choices": [{"finish_reason": "stop", "message": {"content": None, "refusal": '{"phase":"vapor"}'}}]}
    assert audit_chat(body)["text"] == '{"phase":"vapor"}'
