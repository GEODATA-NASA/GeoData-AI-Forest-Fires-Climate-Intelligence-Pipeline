"""The question-answering loop: Claude plus the read-only data tools.

A manual tool loop (rather than the SDK's beta tool runner) keeps three things
explicit: a hard cap on tool rounds per question, a trace of every lookup that
is returned to the caller, and tool errors reported back to the model so it
can correct its input.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import anthropic

from api.assistant.knowledge import KnowledgeBase
from api.assistant.tools import TOOLS, ToolError, ToolExecutor, serialize

LOGGER = logging.getLogger("geodata.api.assistant")

SYSTEM_PROMPT = """You answer questions about the GeoData AI wildfire project for analysts and engineers on the team.

The project ingests NASA FIRMS satellite fire detections (MODIS and VIIRS) and NASA POWER daily weather, stores them in PostgreSQL/PostGIS, and trains a random-forest model that estimates the probability that a detected fire is severe, meaning its fire radiative power (FRP) is at or above a threshold (10 MW by default). The model does not predict whether fires start.

Use the tools for every number you report. Do not estimate counts, shares or model scores from memory, and do not round them beyond what helps the reader. If a tool reports that the data is a random sample, say so and treat shares and rankings as representative but scale absolute counts as the tool explains. State the dates and area a number refers to. When a place is named, pass an approximate bounding box and say it is approximate.

Each row is one detection, not one fire: a single large fire produces many detections. Keep that distinction when you describe counts.

If the question cannot be answered from these tools, say what is missing instead of guessing. Answer in a few short paragraphs or a short list. Lead with the answer."""


@dataclass
class ToolCall:
    name: str
    input: dict
    ok: bool
    duration_ms: float
    error: str | None = None


@dataclass
class Answer:
    text: str
    model: str
    stop_reason: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    rounds: int = 0


class AssistantUnavailable(Exception):
    """Raised when the assistant cannot run (disabled, no credentials, auth failure)."""


def credentials_available(client: anthropic.Anthropic) -> bool:
    """True when the SDK can find credentials (API key, auth token, or an `ant` profile)."""
    if getattr(client, "api_key", None) or getattr(client, "auth_token", None):
        return True
    if os.getenv("ANTHROPIC_PROFILE") or os.getenv("ANTHROPIC_FEDERATION_RULE_ID"):
        return True
    return (Path.home() / ".config" / "anthropic").is_dir()


class AssistantService:
    def __init__(
        self,
        runtime,
        model: str = "claude-opus-5-5",
        effort: str = "medium",
        max_rounds: int = 8,
        client=None,
        knowledge: KnowledgeBase | None = None,
        enabled: bool = True,
        use_fallbacks: bool = True,
    ):
        self.runtime = runtime
        self.model = model
        self.effort = effort
        self.max_rounds = max_rounds
        self.use_fallbacks = use_fallbacks
        self.client = client if client is not None else anthropic.Anthropic(timeout=120.0, max_retries=2)
        self.knowledge = knowledge or KnowledgeBase.from_project(Path(runtime.settings.frontend_dir).parent)
        self.enabled = enabled
        self.tools = ToolExecutor(runtime, self.knowledge)

    @property
    def available(self) -> bool:
        return self.enabled and credentials_available(self.client)

    def status(self) -> dict:
        reason = None
        if not self.enabled:
            reason = "The assistant is switched off (ASSISTANT_ENABLED=0)."
        elif not credentials_available(self.client):
            reason = "No Anthropic credentials found. Set ANTHROPIC_API_KEY for the API process."
        return {"enabled": reason is None, "model": self.model, "reason": reason}

    def _request(self, messages: list) -> object:
        params = dict(
            model=self.model,
            max_tokens=16000,
            # The system prompt and tools never change between requests, so they
            # form a cacheable prefix; the top-level marker also caches the growing
            # message history between rounds of one question.
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            tools=TOOLS,
            messages=messages,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
        )
        if self.use_fallbacks:
            # If a safety classifier declines, the API re-runs the request on
            # Anthropic's recommended fallback model instead of refusing.
            return self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **params)
        return self.client.messages.create(**params)

    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        if not self.available:
            raise AssistantUnavailable(self.status()["reason"])

        today = date.today().isoformat()
        messages: list = []
        for turn in history or []:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": f"{question}\n\n(Today's date: {today}.)"})

        answer = Answer(text="", model=self.model, stop_reason=None)
        usage_totals = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        started = time.monotonic()

        for round_number in range(1, self.max_rounds + 1):
            try:
                response = self._request(messages)
            except anthropic.AuthenticationError as error:
                raise AssistantUnavailable("The Anthropic API rejected the configured credentials.") from error
            except anthropic.PermissionDeniedError as error:
                raise AssistantUnavailable("The configured Anthropic credentials cannot use this model.") from error
            answer.rounds = round_number
            answer.model = getattr(response, "model", self.model)
            answer.stop_reason = response.stop_reason
            for key in usage_totals:
                usage_totals[key] += getattr(response.usage, key, 0) or 0

            if response.stop_reason == "refusal":
                answer.text = "I can't help with that request."
                break
            # Append the assistant turn unchanged (thinking blocks included) so the
            # conversation stays append-only between rounds.
            messages.append({"role": "assistant", "content": response.content})
            tool_uses = [block for block in response.content if block.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                answer.text = "\n\n".join(block.text for block in response.content if block.type == "text").strip()
                if response.stop_reason == "max_tokens":
                    answer.text += "\n\n(The answer was cut off at the length limit.)"
                break

            results = []
            for block in tool_uses:
                call_started = time.perf_counter()
                try:
                    payload = self.tools.run(block.name, dict(block.input or {}))
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": serialize(payload)})
                    call = ToolCall(block.name, dict(block.input or {}), True, 0.0)
                except (ToolError, KeyError, TypeError, ValueError) as error:
                    message = str(error) or type(error).__name__
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": f"Error: {message}", "is_error": True})
                    call = ToolCall(block.name, dict(block.input or {}), False, 0.0, error=message)
                except Exception as error:  # noqa: BLE001 - report to the model, keep the loop alive
                    LOGGER.exception("Tool %s failed", block.name)
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": "Error: the lookup failed on the server.", "is_error": True})
                    call = ToolCall(block.name, dict(block.input or {}), False, 0.0, error=type(error).__name__)
                call.duration_ms = round((time.perf_counter() - call_started) * 1000, 1)
                answer.tool_calls.append(call)
            # All results for one assistant turn go back in a single user message.
            messages.append({"role": "user", "content": results})
        else:
            answer.text = "I could not finish within the lookup limit for one question. Try a narrower question."

        answer.usage = usage_totals
        LOGGER.info(
            "Answered in %.1f s, %s round(s), %s tool call(s), tokens in/out %s/%s (cache read %s).",
            time.monotonic() - started, answer.rounds, len(answer.tool_calls),
            usage_totals["input_tokens"], usage_totals["output_tokens"], usage_totals["cache_read_input_tokens"],
        )
        return answer
