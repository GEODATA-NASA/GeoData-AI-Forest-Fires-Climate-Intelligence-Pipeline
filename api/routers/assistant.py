"""Natural-language questions about the data, answered by Claude with data tools."""

import logging
from typing import Literal

import anthropic
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api.assistant.service import AssistantUnavailable

LOGGER = logging.getLogger("geodata.api.assistant")
router = APIRouter(prefix="/api/v1/ask", tags=["assistant"])


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=12, description="Earlier questions and answers, oldest first")


class ToolCallOut(BaseModel):
    name: str
    input: dict
    ok: bool
    duration_ms: float
    error: str | None = None


class AskResponse(BaseModel):
    answer: str
    model: str
    stop_reason: str | None
    rounds: int
    tool_calls: list[ToolCallOut]
    usage: dict


class AssistantStatus(BaseModel):
    enabled: bool
    model: str | None
    reason: str | None = None


def _assistant(request: Request):
    return getattr(request.app.state, "assistant", None)


@router.get("/status", response_model=AssistantStatus, summary="Whether the assistant can answer questions")
def status(request: Request) -> AssistantStatus:
    assistant = _assistant(request)
    if assistant is None:
        return AssistantStatus(enabled=False, model=None, reason="The assistant is not configured.")
    return AssistantStatus(**assistant.status())


@router.post("", response_model=AskResponse, summary="Ask a question about the fire data, the model or the project")
def ask(payload: AskRequest, request: Request) -> AskResponse:
    assistant = _assistant(request)
    if assistant is None:
        raise HTTPException(status_code=503, detail="The assistant is not configured.")
    history = [turn.model_dump() for turn in payload.history]
    if history and history[0]["role"] != "user":
        raise HTTPException(status_code=422, detail="history must start with a user turn.")
    try:
        answer = assistant.ask(payload.question, history)
    except AssistantUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except anthropic.RateLimitError as error:
        raise HTTPException(status_code=429, detail="The language model is rate limited. Try again in a minute.") from error
    except anthropic.APIConnectionError as error:
        raise HTTPException(status_code=502, detail="Could not reach the Anthropic API.") from error
    except anthropic.APIStatusError as error:
        LOGGER.warning("Anthropic API error %s: %s", error.status_code, error.message)
        raise HTTPException(status_code=502, detail=f"The language model returned an error ({error.status_code}).") from error
    return AskResponse(
        answer=answer.text,
        model=answer.model,
        stop_reason=answer.stop_reason,
        rounds=answer.rounds,
        tool_calls=[ToolCallOut(**call.__dict__) for call in answer.tool_calls],
        usage=answer.usage,
    )
