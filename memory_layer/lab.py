"""Local-only manual test interface for the standalone memory agent."""
import json
import logging
import os
import re
import sqlite3
import traceback
import uuid
from contextlib import closing
from datetime import datetime
import time
from collections.abc import Awaitable, Callable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import httpx2
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from .agent import SESSION_ID, AgentEvent, AgentResult, MemoryAgent
from .chat_store import Chat, ChatStore
from .mock_sources import MockSources
from .persona import get_persona, seed_persona
from .service import MemoryLayer
from .store import Store
from .types import MemoryRecord
from .source_snapshot import SourceSnapshot, source_snapshot

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).with_name('lab.html')
WORKSPACE_PAGE = ROOT / 'frontend' / 'dist' / 'memory-lab.html'
WORKSPACE_ASSETS = ROOT / 'frontend' / 'dist' / 'assets'
WORKSPACE_UNBUILT = 'React workspace is not built. Run npm run build in frontend.'
PERSONA = ROOT / 'data' / 'persona.json'
DATABASE = Path(os.getenv('MEMORY_LAB_DB', str(ROOT / 'memory-lab.sqlite3')))
DEFAULT_MODEL = 'gpt-5.4-mini'
MODEL_ENV = 'OPENAI_MODEL'
IPV4_ENV = 'AGENT_OPENAI_IPV4_ONLY'
IPV4_ADDRESS = '0.0.0.0'
LOCAL_HOSTS = ['localhost', '127.0.0.1']
ORIGIN_HEADER = 'origin'
INVALID_ORIGIN = 'This request must come from the local Memory Lab page.'
PROVIDER_ERROR = 'The memory request failed. Check the backend log and try again.'
CHAT_FAILURE_LOG = (
    'memory_lab.chat_failed handler=%s run_id=%s chat_id=%s message_id=%s error_type=%s '
    'source_file=%s source_function=%s source_line=%s status_code=%s code=%s param=%s api_type=%s request_id=%s'
)
CHAT_HANDLER = 'chat'
CHAT_STREAM_HANDLER = 'chat_stream'
NDJSON_MEDIA_TYPE = 'application/x-ndjson'
LAB_LOG = 'memory_lab'
logger = logging.getLogger(LAB_LOG)
load_dotenv(ROOT / '.env')


def log_chat_failure(exc: Exception, handler: str, chat_id: str, message_id: str, run_id: str) -> None:
    frames = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []
    source = next((frame for frame in reversed(frames) if Path(frame.filename).is_relative_to(ROOT)), None)
    logger.error(CHAT_FAILURE_LOG, handler, run_id, chat_id, message_id, type(exc).__name__,
                 Path(source.filename).relative_to(ROOT).as_posix() if source else 'unknown',
                 source.name if source else 'unknown', source.lineno if source else 0,
                 getattr(exc, 'status_code', None), getattr(exc, 'code', None), getattr(exc, 'param', None),
                 getattr(exc, 'type', None), getattr(exc, 'request_id', None))
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('memory_lab.chat_failure_detail run_id=%s provider_message=%s', run_id,
                     getattr(exc, 'message', ''))


class SourcePollingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if (isinstance(args, tuple) and len(args) == 5 and args[1] == 'GET'
                and isinstance(args[2], str) and args[2].split('?', 1)[0] == '/api/sources' and args[4] == 200):
            logging.getLogger('uvicorn.error').debug(record.msg, *args)
            return False
        return True


logging.getLogger('uvicorn.access').addFilter(SourcePollingFilter())


class Persona(BaseModel):
    as_of: str
    email: str


class PersonaRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=10000)]
    expected_text: str | None


class PersonaResponse(BaseModel):
    text: str | None


class MemoryEditRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)

    expected_text: Annotated[str, StringConstraints(min_length=1, max_length=10000)]
    text: Annotated[str, StringConstraints(min_length=1, max_length=10000)]


class MemoryForgetRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)

    expected_text: Annotated[str, StringConstraints(min_length=1, max_length=10000)]


class MemoryForgetResponse(BaseModel):
    status: Literal['forgotten']


class ManualContactRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=200)
    email: str = Field(min_length=3, max_length=320)
    org: str = Field(default='', max_length=200)
    role: str = Field(default='', max_length=200)

    @field_validator('email')
    @classmethod
    def synthetic_email(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.example", value, re.I):
            raise ValueError('Email must be a synthetic .example address.')
        return value


class ContactEditRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    expected: ManualContactRequest
    contact: ManualContactRequest


class IncomingMailRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)

    thread_id: str | None = Field(default=None, min_length=1, max_length=200)
    subject: str | None = Field(default=None, min_length=1, max_length=300)
    sender: str = Field(min_length=3, max_length=320)
    body: str = Field(min_length=1, max_length=10000)
    timestamp: datetime
    unread: bool = True

    @model_validator(mode='after')
    def valid_thread(self) -> 'IncomingMailRequest':
        if bool(self.thread_id) == bool(self.subject):
            raise ValueError('Choose an existing thread or provide a new subject.')
        return self

    @field_validator('sender')
    @classmethod
    def synthetic_sender(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.example", value, re.I):
            raise ValueError('Sender must be a synthetic .example address.')
        return value

    @field_validator('timestamp')
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value is None or value.utcoffset() is None:
            raise ValueError('Timestamp must include a timezone.')
        return value


class IncomingMailCreateRequest(IncomingMailRequest):
    timestamp: datetime | None = None


class MailThreadEditRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)

    expected_subject: str = Field(min_length=1, max_length=300)
    subject: str = Field(min_length=1, max_length=300)


class IncomingMailEditRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    expected: IncomingMailRequest
    message: IncomingMailRequest


class CalendarEventRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)

    person: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=300)
    start: datetime
    end: datetime
    all_day: bool = False

    @model_validator(mode='after')
    def valid_interval(self) -> 'CalendarEventRequest':
        if self.start.utcoffset() is None or self.end.utcoffset() is None or self.end <= self.start:
            raise ValueError('Start and end must include timezones, and end must be after start.')
        return self


class CalendarEventEditRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    expected: CalendarEventRequest
    event: CalendarEventRequest


class SlackChannelRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)

    name: str = Field(min_length=2, max_length=80, pattern=r'^#[a-z0-9][a-z0-9_-]*$')
    members: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]] = Field(max_length=100)

    @field_validator('members')
    @classmethod
    def unique_members(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError('Members must be unique.')
        return value


class SlackChannelEditRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    expected: SlackChannelRequest
    channel: SlackChannelRequest


class SlackMessageRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)

    channel: str = Field(min_length=2, max_length=80)
    author: str = Field(min_length=1, max_length=100)
    ts: datetime
    text: str = Field(min_length=1, max_length=10000)

    @field_validator('ts')
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value is None or value.utcoffset() is None:
            raise ValueError('Timestamp must include a timezone.')
        return value


class SlackMessageCreateRequest(SlackMessageRequest):
    ts: datetime | None = None


class SlackMessageEditRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    expected: SlackMessageRequest
    message: SlackMessageRequest


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    text: str = Field(min_length=1, max_length=10000)
    chat_id: str = Field(default=SESSION_ID, min_length=1, max_length=64)


@lru_cache(maxsize=1)
def get_sources() -> MockSources:
    try:
        sources = MockSources(DATABASE)
        sources.seed()
        return sources
    except Exception:
        logger.exception('sources.initialize status=error')
        raise


@lru_cache(maxsize=1)
def get_agent() -> MemoryAgent:
    transport = httpx2.HTTPTransport(local_address=IPV4_ADDRESS) if os.getenv(IPV4_ENV, '').lower() == 'true' else None
    http_client = httpx2.Client(transport=transport, timeout=60)
    model_name = os.getenv(MODEL_ENV, DEFAULT_MODEL)
    client = OpenAI(http_client=http_client)
    memory = MemoryLayer(DATABASE, client, model_name)
    seed_persona(memory.store, PERSONA)
    sources = get_sources()
    model = ChatOpenAI(model=model_name, http_client=http_client, store=False,
                       model_kwargs={'parallel_tool_calls': False})
    checkpointer = SqliteSaver(sqlite3.connect(DATABASE, check_same_thread=False))
    persona = Persona.model_validate_json(PERSONA.read_text(encoding='utf-8'))
    return MemoryAgent(memory, model, checkpointer, sources, persona.as_of, persona.email, PERSONA)


app = FastAPI()
app.add_middleware(TrustedHostMiddleware, allowed_hosts=LOCAL_HOSTS)


@app.middleware('http')
async def local_origin(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    origin = request.headers.get(ORIGIN_HEADER)
    if origin and origin != str(request.base_url).rstrip('/'):
        return Response(INVALID_ORIGIN, status_code=403)
    return await call_next(request)


@app.get('/', response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(PAGE)


@app.get('/workspace', response_class=FileResponse)
def workspace() -> FileResponse:
    if not WORKSPACE_PAGE.is_file():
        raise HTTPException(status_code=503, detail=WORKSPACE_UNBUILT)
    return FileResponse(WORKSPACE_PAGE)


@app.get('/api/sources')
def sources_view(sources: Annotated[MockSources, Depends(get_sources)]) -> SourceSnapshot:
    started = time.monotonic()
    try:
        persona = Persona.model_validate_json(PERSONA.read_text(encoding='utf-8'))
        result = source_snapshot(sources, persona.email, persona.as_of)
        logging.getLogger('uvicorn.error').debug(
            'sources.snapshot status=ok contacts=%s mail=%s events=%s slack=%s duration_ms=%.1f',
            len(result.contacts), len(result.mail), len(result.calendar_events), len(result.slack_messages),
            (time.monotonic() - started) * 1000)
        return result
    except Exception as exc:
        logger.exception('sources.snapshot status=error error_type=%s', type(exc).__name__)
        raise HTTPException(status_code=503, detail='Could not load source data. Check the backend log.') from exc


@app.post('/api/sources/contacts', status_code=201)
def add_contact(payload: ManualContactRequest, sources: Annotated[MockSources, Depends(get_sources)]) -> dict[str, str]:
    contact_id = 'manual:' + uuid.uuid4().hex
    with closing(sources.connect()) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        if conn.execute('SELECT 1 FROM contacts WHERE lower(email) = lower(?)', (payload.email,)).fetchone():
            raise HTTPException(status_code=409, detail='A contact with this email already exists.')
        conn.execute('INSERT INTO contacts(id,name,email,org,role) VALUES (?,?,?,?,?)',
                     (contact_id, payload.name, payload.email, payload.org, payload.role))
    logging.getLogger('uvicorn.error').info('source.contact status=created contact_id=%s', contact_id)
    return {'id': contact_id}


@app.put('/api/sources/contacts/{contact_id}')
def edit_contact(contact_id: str, payload: ContactEditRequest,
                 sources: Annotated[MockSources, Depends(get_sources)]) -> dict[str, str]:
    with closing(sources.connect()) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT name,email,org,role FROM contacts WHERE id = ?', (contact_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail='Contact not found.')
        previous = payload.expected
        if row != (previous.name, previous.email, previous.org, previous.role):
            raise HTTPException(status_code=409, detail='Contact changed since loading; refresh and retry.')
        contact = payload.contact
        if conn.execute('SELECT 1 FROM contacts WHERE lower(email) = lower(?) AND id != ?',
                        (contact.email, contact_id)).fetchone():
            raise HTTPException(status_code=409, detail='A contact with this email already exists.')
        conn.execute('UPDATE contacts SET name=?,email=?,org=?,role=? WHERE id=?',
                     (contact.name, contact.email, contact.org, contact.role, contact_id))
    logging.getLogger('uvicorn.error').info('source.contact status=updated contact_id=%s', contact_id)
    return {'id': contact_id}


@app.post('/api/sources/incoming-mail', status_code=201)
def add_incoming_mail(payload: IncomingMailCreateRequest, sources: Annotated[MockSources, Depends(get_sources)]) -> dict[str, str]:
    persona = Persona.model_validate_json(PERSONA.read_text(encoding='utf-8'))
    if payload.sender.casefold() == persona.email.casefold():
        raise HTTPException(status_code=422, detail='Incoming sender must differ from the user address.')
    message_id = 'manual:' + uuid.uuid4().hex
    with closing(sources.connect()) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        thread_id = payload.thread_id
        if thread_id is None:
            thread_id = 'manual:' + uuid.uuid4().hex
            conn.execute('INSERT INTO mail_threads(id, subject) VALUES (?, ?)', (thread_id, payload.subject))
        elif not conn.execute('SELECT 1 FROM mail_threads WHERE id = ?', (thread_id,)).fetchone():
            raise HTTPException(status_code=404, detail='Mail thread not found.')
        position = conn.execute('SELECT COALESCE(MAX(position), -1) + 1 FROM mail_messages WHERE thread_id = ?',
                                (thread_id,)).fetchone()[0]
        conn.execute('INSERT INTO mail_messages VALUES (?, ?, ?, ?, ?, ?, ?)',
                     (message_id, thread_id, payload.sender, payload.timestamp.isoformat() if payload.timestamp else persona.as_of, payload.unread, payload.body, position))
        conn.execute('INSERT INTO mail_recipients VALUES (?, ?, ?, ?)', (message_id, 'to', 0, persona.email))
    logging.getLogger('uvicorn.error').info('source.incoming_mail status=created thread_id=%s message_id=%s', thread_id, message_id)
    return {'id': message_id, 'thread_id': thread_id}


@app.put('/api/sources/incoming-mail/{message_id}')
def edit_incoming_mail(message_id: str, payload: IncomingMailEditRequest,
                       sources: Annotated[MockSources, Depends(get_sources)]) -> dict[str, str]:
    persona = Persona.model_validate_json(PERSONA.read_text(encoding='utf-8'))
    with closing(sources.connect()) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT thread_id,sender,ts,body,unread FROM mail_messages WHERE id = ?', (message_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail='Email not found.')
        if row[1].lower() == persona.email.lower() or not conn.execute(
                "SELECT 1 FROM mail_recipients WHERE message_id=? AND lower(address)=lower(?)", (message_id, persona.email)).fetchone():
            raise HTTPException(status_code=400, detail='Only incoming email can be edited here.')
        previous, message = payload.expected, payload.message
        if message.sender.casefold() == persona.email.casefold():
            raise HTTPException(status_code=422, detail='Incoming sender must differ from the user address.')
        if (row[0] != previous.thread_id or row[1] != previous.sender or row[3] != previous.body
                or bool(row[4]) != previous.unread or datetime.fromisoformat(row[2]) != previous.timestamp):
            raise HTTPException(status_code=409, detail='Email changed since loading; refresh and retry.')
        if message.thread_id != row[0]:
            raise HTTPException(status_code=400, detail='Moving email to another thread is not supported.')
        conn.execute('UPDATE mail_messages SET sender=?,ts=?,body=?,unread=? WHERE id=?',
                     (message.sender, message.timestamp.isoformat(), message.body, message.unread, message_id))
    logging.getLogger('uvicorn.error').info('source.incoming_mail status=updated thread_id=%s message_id=%s', row[0], message_id)
    return {'id': message_id, 'thread_id': row[0]}


@app.put('/api/sources/mail-threads/{thread_id}')
def edit_mail_thread(thread_id: str, payload: MailThreadEditRequest,
                     sources: Annotated[MockSources, Depends(get_sources)]) -> dict[str, str]:
    with closing(sources.connect()) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT subject FROM mail_threads WHERE id=?', (thread_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail='Mail thread not found.')
        if row[0] != payload.expected_subject:
            raise HTTPException(status_code=409, detail='Mail thread changed since loading; refresh and retry.')
        conn.execute('UPDATE mail_threads SET subject=? WHERE id=?', (payload.subject, thread_id))
    logging.getLogger('uvicorn.error').info('source.mail_thread status=updated thread_id=%s', thread_id)
    return {'id': thread_id}


@app.post('/api/sources/calendar-events', status_code=201)
def add_calendar_event(payload: CalendarEventRequest,
                       sources: Annotated[MockSources, Depends(get_sources)]) -> dict[str, str]:
    event_id = 'manual:' + uuid.uuid4().hex
    with closing(sources.connect()) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        if not conn.execute('SELECT 1 FROM calendar_people WHERE id=?', (payload.person,)).fetchone():
            raise HTTPException(status_code=404, detail='Calendar account not found.')
        conn.execute('INSERT INTO calendar_events VALUES (?,?,?,?,?,?)',
                     (event_id, payload.person, payload.title, payload.start.isoformat(), payload.end.isoformat(), payload.all_day))
    logging.getLogger('uvicorn.error').info('source.calendar status=created event_id=%s', event_id)
    return {'id': event_id}


@app.put('/api/sources/calendar-events/{event_id}')
def edit_calendar_event(event_id: str, payload: CalendarEventEditRequest,
                        sources: Annotated[MockSources, Depends(get_sources)]) -> dict[str, str]:
    with closing(sources.connect()) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT person_id,title,start,end,all_day FROM calendar_events WHERE id=?', (event_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail='Calendar event not found.')
        previous, event = payload.expected, payload.event
        if (row[0] != previous.person or row[1] != previous.title or bool(row[4]) != previous.all_day
                or datetime.fromisoformat(row[2]) != previous.start or datetime.fromisoformat(row[3]) != previous.end):
            raise HTTPException(status_code=409, detail='Calendar event changed since loading; refresh and retry.')
        if not conn.execute('SELECT 1 FROM calendar_people WHERE id=?', (event.person,)).fetchone():
            raise HTTPException(status_code=404, detail='Calendar account not found.')
        conn.execute('UPDATE calendar_events SET person_id=?,title=?,start=?,end=?,all_day=? WHERE id=?',
                     (event.person, event.title, event.start.isoformat(), event.end.isoformat(), event.all_day, event_id))
    logging.getLogger('uvicorn.error').info('source.calendar status=updated event_id=%s', event_id)
    return {'id': event_id}


def save_slack_members(conn: sqlite3.Connection, name: str, members: list[str]) -> None:
    conn.execute('DELETE FROM slack_members WHERE channel_id=?', (name,))
    for position, member in enumerate(members):
        conn.execute('INSERT OR IGNORE INTO slack_people(id) VALUES (?)', (member,))
        conn.execute('INSERT INTO slack_members(channel_id,person_id,position) VALUES (?,?,?)', (name, member, position))


@app.post('/api/sources/slack-channels', status_code=201)
def add_slack_channel(payload: SlackChannelRequest,
                      sources: Annotated[MockSources, Depends(get_sources)]) -> dict[str, str]:
    with closing(sources.connect()) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        if conn.execute('SELECT 1 FROM slack_channels WHERE id=?', (payload.name,)).fetchone():
            raise HTTPException(status_code=409, detail='Slack channel already exists.')
        conn.execute('INSERT INTO slack_channels(id) VALUES (?)', (payload.name,))
        save_slack_members(conn, payload.name, payload.members)
    logging.getLogger('uvicorn.error').info('source.slack_channel status=created channel=%s', payload.name)
    return {'id': payload.name}


@app.put('/api/sources/slack-channels/{channel_id}')
def edit_slack_channel(channel_id: str, payload: SlackChannelEditRequest,
                       sources: Annotated[MockSources, Depends(get_sources)]) -> dict[str, str]:
    with closing(sources.connect()) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        if not conn.execute('SELECT 1 FROM slack_channels WHERE id=?', (channel_id,)).fetchone():
            raise HTTPException(status_code=404, detail='Slack channel not found.')
        members = [row[0] for row in conn.execute('SELECT person_id FROM slack_members WHERE channel_id=? ORDER BY position', (channel_id,))]
        if payload.expected.name != channel_id or payload.expected.members != members:
            raise HTTPException(status_code=409, detail='Slack channel changed since loading; refresh and retry.')
        name = payload.channel.name
        if name != channel_id:
            if conn.execute('SELECT 1 FROM slack_channels WHERE id=?', (name,)).fetchone():
                raise HTTPException(status_code=409, detail='Slack channel already exists.')
            for table in ('mock_pending_actions', 'mock_pending_action'):
                if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                    pending = conn.execute(f"SELECT payload FROM {table} WHERE kind='slack'").fetchall()
                    if any(json.loads(row[0]).get('channel') == channel_id for row in pending):
                        raise HTTPException(status_code=409, detail='A pending Slack action uses this channel; resolve it before renaming.')
            conn.execute('PRAGMA defer_foreign_keys=ON')
            conn.execute('UPDATE slack_channels SET id=? WHERE id=?', (name, channel_id))
            conn.execute('UPDATE slack_members SET channel_id=? WHERE channel_id=?', (name, channel_id))
            conn.execute('UPDATE slack_messages SET channel_id=? WHERE channel_id=?', (name, channel_id))
            conn.execute('INSERT OR REPLACE INTO source_migrations(name) VALUES (?)',
                         ('renamed_slack:' + channel_id + '->' + name,))
        save_slack_members(conn, name, payload.channel.members)
    logging.getLogger('uvicorn.error').info('source.slack_channel status=updated previous_channel=%s channel=%s', channel_id, name)
    return {'id': name}


@app.post('/api/sources/slack-messages', status_code=201)
def add_slack_message(payload: SlackMessageCreateRequest,
                      sources: Annotated[MockSources, Depends(get_sources)]) -> dict[str, str]:
    message_id = 'manual:' + uuid.uuid4().hex
    with closing(sources.connect()) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        if not conn.execute('SELECT 1 FROM slack_channels WHERE id=?', (payload.channel,)).fetchone():
            raise HTTPException(status_code=404, detail='Slack channel not found.')
        conn.execute('INSERT OR IGNORE INTO slack_people(id) VALUES (?)', (payload.author,))
        conn.execute('INSERT INTO slack_messages VALUES (?,?,?,?,?)',
                     (message_id, payload.channel, payload.author, payload.ts.isoformat() if payload.ts else
                      Persona.model_validate_json(PERSONA.read_text(encoding='utf-8')).as_of, payload.text))
    logging.getLogger('uvicorn.error').info('source.slack_message status=created message_id=%s channel=%s', message_id, payload.channel)
    return {'id': message_id}


@app.put('/api/sources/slack-messages/{message_id}')
def edit_slack_message(message_id: str, payload: SlackMessageEditRequest,
                       sources: Annotated[MockSources, Depends(get_sources)]) -> dict[str, str]:
    with closing(sources.connect()) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT channel_id,author_id,ts,text FROM slack_messages WHERE id=?', (message_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail='Slack message not found.')
        previous, message = payload.expected, payload.message
        if row[0] != previous.channel or row[1] != previous.author or row[3] != previous.text or datetime.fromisoformat(row[2]) != previous.ts:
            raise HTTPException(status_code=409, detail='Slack message changed since loading; refresh and retry.')
        if not conn.execute('SELECT 1 FROM slack_channels WHERE id=?', (message.channel,)).fetchone():
            raise HTTPException(status_code=404, detail='Slack channel not found.')
        conn.execute('INSERT OR IGNORE INTO slack_people(id) VALUES (?)', (message.author,))
        conn.execute('UPDATE slack_messages SET channel_id=?,author_id=?,ts=?,text=? WHERE id=?',
                     (message.channel, message.author, message.ts.isoformat(), message.text, message_id))
    logging.getLogger('uvicorn.error').info('source.slack_message status=updated message_id=%s channel=%s', message_id, message.channel)
    return {'id': message_id}


@app.get('/api/persona')
def persona_view() -> PersonaResponse:
    store = Store(DATABASE)
    seed_persona(store, PERSONA)
    return PersonaResponse(text=get_persona(store))


@app.get('/api/memories')
def list_memories() -> list[MemoryRecord]:
    return Store(DATABASE).records()


@app.put('/api/memories/{memory_id}')
def edit_memory(memory_id: str, payload: MemoryEditRequest) -> MemoryRecord:
    if not memory_id.isdecimal():
        raise HTTPException(status_code=404, detail='Memory not found.')
    try:
        return Store(DATABASE).edit(memory_id, payload.expected_text, payload.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail='Memory changed since loading; reload and retry.') from exc
    except ValueError as exc:
        status = 404 if str(exc) == 'Memory not found.' else 409
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@app.delete('/api/memories/{memory_id}', response_model=MemoryForgetResponse)
def forget_memory(memory_id: str, payload: MemoryForgetRequest) -> MemoryForgetResponse:
    if not memory_id.isdecimal():
        raise HTTPException(status_code=404, detail='Memory not found.')
    try:
        Store(DATABASE).forget(memory_id, payload.expected_text)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail='Memory changed since loading; reload and retry.') from exc
    except ValueError as exc:
        status = 404 if str(exc) == 'Memory not found.' else 409
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    logger.info('memory.forget status=ok memory_id=%s', memory_id)
    return MemoryForgetResponse(status='forgotten')


@app.get('/api/chats')
def list_chats() -> list[Chat]:
    return ChatStore(DATABASE).list_chats()


@app.post('/api/chats')
def create_chat() -> Chat:
    return ChatStore(DATABASE).create()


def prepare_chat(store: ChatStore, chat_id: str, agent: MemoryAgent) -> Chat:
    if chat_id == SESSION_ID:
        store.ensure_legacy()
    chat = store.get(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail='Chat not found.')
    if store.needs_import(chat_id):
        store.import_events(chat_id, agent.history(chat_id))
    return chat


@app.get('/api/chats/{chat_id}')
def chat_history(chat_id: str, agent: Annotated[MemoryAgent, Depends(get_agent)]) -> dict[str, object]:
    store = ChatStore(DATABASE)
    chat = prepare_chat(store, chat_id, agent)
    return {'chat': chat, 'events': store.events(chat_id), 'pending': agent.pending_action(chat_id)}


@app.post('/api/chat')
def chat(payload: ChatRequest, agent: Annotated[MemoryAgent, Depends(get_agent)]) -> AgentResult:
    store = ChatStore(DATABASE)
    prepare_chat(store, payload.chat_id, agent)
    message_id = store.append(payload.chat_id, {'kind': 'user', 'data': payload.text})
    try:
        result = agent.answer(payload.text, payload.chat_id, message_id)
        store.append(payload.chat_id, {'kind': 'answer', 'data': result.answer,
                                       'references': [reference.model_dump() for reference in result.references]})
        return result
    except Exception as exc:
        log_chat_failure(exc, CHAT_HANDLER, payload.chat_id, message_id, agent.run_id)
        raise HTTPException(status_code=503, detail=PROVIDER_ERROR) from exc


@app.post('/api/chat/stream')
def chat_stream(payload: ChatRequest, agent: Annotated[MemoryAgent, Depends(get_agent)]) -> StreamingResponse:
    store = ChatStore(DATABASE)
    prepare_chat(store, payload.chat_id, agent)
    message_id = store.append(payload.chat_id, {'kind': 'user', 'data': payload.text})
    def events() -> Iterator[str]:
        try:
            yield json.dumps({'kind': 'user_id', 'message_id': message_id}) + '\n'
            for event in agent.stream_answer(payload.text, payload.chat_id, message_id):
                store.append(payload.chat_id, event.model_dump())
                yield event.model_dump_json() + '\n'
        except Exception as exc:
            log_chat_failure(exc, CHAT_STREAM_HANDLER, payload.chat_id, message_id, agent.run_id)
            event = AgentEvent(kind='error', data=PROVIDER_ERROR)
            store.append(payload.chat_id, event.model_dump())
            yield event.model_dump_json() + '\n'
    return StreamingResponse(events(), media_type=NDJSON_MEDIA_TYPE)


app.mount('/assets', StaticFiles(directory=WORKSPACE_ASSETS, check_dir=False), name='workspace-assets')
