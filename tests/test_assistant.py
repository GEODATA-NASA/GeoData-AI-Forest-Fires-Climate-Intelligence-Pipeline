"""Assistant loop, tools and endpoint, driven by a scripted fake Claude client (no network, no cost)."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api.assistant.knowledge import KnowledgeBase
from api.assistant.service import AssistantService
from api.assistant.tools import TOOLS, ToolError, ToolExecutor
from api.main import create_app


def _text(text):
    return SimpleNamespace(type="text", text=text)


def _tool(name, args, tool_id):
    return SimpleNamespace(type="tool_use", name=name, input=args, id=tool_id)


def _response(content, stop_reason):
    usage = SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=50, cache_creation_input_tokens=0)
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage, model="claude-opus-5-5")


class FakeMessages:
    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def create(self, **kwargs):
        # Snapshot the messages: the service keeps appending to the same list.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.script.pop(0)


class FakeClient:
    def __init__(self, script):
        self.api_key = "test-key"
        self.auth_token = None
        self.messages = FakeMessages(script)
        self.beta = SimpleNamespace(messages=self.messages)


@pytest.fixture()
def runtime(client):
    return client.app.state.runtime


def _service(runtime, script, **kwargs):
    return AssistantService(runtime, client=FakeClient(script), knowledge=KnowledgeBase.from_project(runtime.settings.frontend_dir.parent), **kwargs)


def test_tool_schemas_are_strict_and_closed():
    for tool in TOOLS:
        assert tool["strict"] is True
        assert tool["input_schema"]["additionalProperties"] is False
        for prop in tool["input_schema"].get("properties", {}).values():
            assert not {"minimum", "maximum", "minLength", "maxLength"} & set(prop)


def test_loop_runs_tools_and_returns_answer(runtime):
    service = _service(runtime, [
        _response([_tool("get_fire_statistics", {"start_date": "2025-01-01", "end_date": "2025-01-31"}, "t1"),
                   _tool("get_model_info", {}, "t2")], "tool_use"),
        _response([_text("There were 3 detections in January 2025.")], "end_turn"),
    ])
    answer = service.ask("How many fires in January 2025?")
    assert answer.text == "There were 3 detections in January 2025."
    assert [call.name for call in answer.tool_calls] == ["get_fire_statistics", "get_model_info"]
    assert all(call.ok for call in answer.tool_calls)
    assert answer.rounds == 2 and answer.usage["input_tokens"] == 200

    requests = service.client.messages.requests
    first, second = requests
    assert first["model"] == "claude-opus-5-5"
    assert first["fallbacks"] == "default" and first["betas"] == ["server-side-fallback-2026-07-01"]
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "thinking" not in first  # always on for this model; effort is the control
    assert first["output_config"] == {"effort": "medium"}
    # Both tool results return in one user message, in order, matching the tool_use ids.
    results = second["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    stats = json.loads(results[0]["content"])
    assert stats["total_detections"] == 3 and stats["severe_detections"] == 2
    assert stats["by_date"][0] == {"date": "2025-01-02", "count": 2, "high_frp_count": 1}


def test_tool_errors_go_back_to_the_model(runtime):
    service = _service(runtime, [
        _response([_tool("get_fire_statistics", {"start_date": "2025-02-01", "end_date": "2025-01-01"}, "t1")], "tool_use"),
        _response([_text("The dates were reversed.")], "end_turn"),
    ])
    answer = service.ask("Stats?")
    assert answer.tool_calls[0].ok is False and "start_date" in answer.tool_calls[0].error
    result = service.client.messages.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True


def test_refusal_and_round_limit(runtime):
    refused = _service(runtime, [_response([], "refusal")])
    assert refused.ask("...").text == "I can't help with that request."

    looping = _service(runtime, [_response([_tool("get_model_info", {}, f"t{i}")], "tool_use") for i in range(3)], max_rounds=3)
    assert "lookup limit" in looping.ask("Loop").text


def test_history_is_sent_before_the_question(runtime):
    service = _service(runtime, [_response([_text("ok")], "end_turn")])
    service.ask("And in March?", history=[{"role": "user", "content": "Fires in January?"}, {"role": "assistant", "content": "3."}])
    messages = service.client.messages.requests[0]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[-1]["content"].startswith("And in March?")


def test_fallbacks_can_be_switched_off(runtime):
    service = _service(runtime, [_response([_text("ok")], "end_turn")], use_fallbacks=False)
    service.ask("Hi")
    assert "fallbacks" not in service.client.messages.requests[0]


def test_tool_executor_validation_and_outputs(runtime):
    tools = ToolExecutor(runtime, KnowledgeBase.from_project(runtime.settings.frontend_dir.parent))
    coverage = tools.run("get_data_coverage", {})
    assert coverage["fire_date_range"] == {"start": "2025-01-02", "end": "2025-03-16"}
    listed = tools.run("list_fires", {"start_date": "2025-01-01", "end_date": "2025-12-31", "limit": 2, "include_predictions": True})
    assert listed["count"] == 2 and listed["fires"][0]["frp"] == 250.0
    assert 0 <= listed["fires"][0]["high_frp_probability"] <= 1
    scored = tools.run("predict_fire_severity", {"latitude": 30.0, "longitude": 31.0, "acq_date": "2025-01-02", "brightness": 340})
    assert scored["weather_matched"] is True and "scan" in scored["missing_inputs_imputed"]
    with pytest.raises(ToolError):
        tools.run("get_fire_statistics", {"start_date": "2025-01-01", "end_date": "2025-01-31", "bbox": [10, 5, -10, 6]})
    with pytest.raises(ToolError):
        tools.run("delete_everything", {})


def test_documentation_search_finds_relevant_sections(runtime):
    from api.assistant.knowledge import DATA_DICTIONARY

    # A fixed corpus, so edits to the README cannot change the expected hit.
    kb = KnowledgeBase({"data dictionary": DATA_DICTIONARY})
    hits = kb.search("what does frp mean fire radiative power", limit=4)
    assert any("fire radiative power" in hit["text"].lower() for hit in hits)
    assert kb.search("unrelated words like spaceship banana") == []

    project = KnowledgeBase.from_project(runtime.settings.frontend_dir.parent)
    assert any(hit["source"] == "README.md" for hit in project.search("docker compose", limit=4))


def test_endpoint_status_and_errors(settings):
    app = create_app(settings)
    with TestClient(app) as http:
        # No credentials in the test environment -> disabled, 503 on ask.
        app.state.assistant.client = SimpleNamespace(api_key=None, auth_token=None)
        status = http.get("/api/v1/ask/status").json()
        assert status["enabled"] in (False, True)
        if not status["enabled"]:
            assert http.post("/api/v1/ask", json={"question": "How many fires?"}).status_code == 503

        app.state.assistant.client = FakeClient([_response([_text("Three.")], "end_turn")])
        response = http.post("/api/v1/ask", json={"question": "How many fires?"})
        assert response.status_code == 200
        assert response.json()["answer"] == "Three."
        assert http.post("/api/v1/ask", json={"question": "x"}).status_code == 422
        bad_history = {"question": "Hi there", "history": [{"role": "assistant", "content": "hello"}]}
        assert http.post("/api/v1/ask", json=bad_history).status_code == 422
