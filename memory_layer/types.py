"""Typed objects shared by the standalone memory methods."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class Memory(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    id: str
    text: str


class MemoryProvenance(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    kind: Literal['mail', 'slack', 'calendar', 'chat']
    source_id: str
    chat_id: str | None = None


class MemoryRecord(Memory):
    is_persona: bool
    provenance: list[MemoryProvenance]


class Message(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    role: Literal['user', 'assistant']
    content: str


class ProviderMessage(BaseModel):
    role: Literal['system', 'user']
    content: str


class ReadRequest(BaseModel):
    query: str
    memories: list[Memory]


class ChangeRequest(BaseModel):
    instruction: str
    context_mode: Literal['messages', 'summary']
    context: list[Message] | str
    memories: list[Memory]
    historical_time: datetime | None = None


class ReadDecision(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    ids: list[str]


class ChangeDecision(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    action: Literal['create', 'update', 'delete', 'clarify', 'none']
    text: str
    ids: list[str]
    question: str


class ChangeResult(BaseModel):
    status: Literal['created', 'updated', 'deleted', 'clarification_needed', 'no_change']
    ids: list[str] | None = None
    memory: str | None = None
    question: str | None = None
