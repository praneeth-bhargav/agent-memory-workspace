"""Replay historical messages as evidence without invoking chat or action tools."""
import argparse
import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from .chat_store import ChatStore
from .persona import seed_persona
from .semantic import DEFAULT_MODEL, MODEL_ENV
from .service import CHANGE_PROMPT, CHAT_GROUNDING_PROMPT, MemoryLayer
from .types import ChangeResult, Message

ROOT = Path(__file__).resolve().parent.parent
CONTEXT_MESSAGES = 20
REPLAY_VERSION = 1
DEFAULT_AGENT_MODEL = 'gpt-5.4-mini'
MODEL_ENVIRONMENT_KEY = 'OPENAI_MODEL'
REPLAY_SCHEMA = '''
CREATE TABLE IF NOT EXISTS history_replay_runs (
    input_hash TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL UNIQUE REFERENCES lab_chats(id),
    message_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS history_replay_messages (
    input_hash TEXT NOT NULL REFERENCES history_replay_runs(input_hash),
    history_id TEXT NOT NULL,
    message_row_id INTEGER NOT NULL UNIQUE REFERENCES lab_chat_events(id),
    PRIMARY KEY(input_hash, history_id)
);
CREATE TABLE IF NOT EXISTS history_replay_decisions (
    input_hash TEXT NOT NULL,
    history_id TEXT NOT NULL,
    status TEXT NOT NULL,
    outcome TEXT NOT NULL,
    PRIMARY KEY(input_hash, history_id),
    FOREIGN KEY(input_hash, history_id)
        REFERENCES history_replay_messages(input_hash, history_id)
);
'''
REPLAY_TITLE = 'Historical replay'
NEW_CHAT_TITLE = 'New chat'
USER_EVENT = 'user'
ASSISTANT_EVENT = 'answer'
REPLAY_CHANNEL = 'channel'
REPLAY_TIME = 'ts'
REPLAY_HISTORY_ID = 'history_id'


class HistoricalMessage(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    id: str = Field(min_length=1)
    ts: datetime
    channel: str = Field(min_length=1)
    role: Literal['user', 'assistant']
    text: str


class ReplayOutcome(BaseModel):
    status: Literal['created', 'updated', 'deleted', 'no_change', 'clarification_needed', 'rejected']
    result: ChangeResult | None = None
    error_type: str | None = None


class ReplayReport(BaseModel):
    chat_id: str
    messages: int
    created: int
    updated: int
    other: int
    deleted: int = 0
    skipped: int = 0
    clarification: int = 0
    rejected: int = 0


def _input_hash(messages: list[HistoricalMessage], agent_model: str, embedding_model: str) -> str:
    prompt_hash = hashlib.sha256((CHANGE_PROMPT + CHAT_GROUNDING_PROMPT).encode('utf-8')).hexdigest()
    identity = {'messages': [message.model_dump(mode='json') for message in messages],
                'agent_model': agent_model, 'embedding_model': embedding_model,
                'decision_prompt_hash': prompt_hash, 'replay_version': REPLAY_VERSION}
    content = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def _require_existing_run(database: Path, input_hash: str) -> None:
    uri = database.resolve().as_uri() + '?mode=ro'
    try:
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'history_replay_runs' not in tables:
                raise ValueError('No matching replay run exists in this database.')
            row = conn.execute('SELECT 1 FROM history_replay_runs WHERE input_hash=?', (input_hash,)).fetchone()
    except sqlite3.Error as exc:
        raise ValueError('Replay database is unreadable.') from exc
    if row is None:
        raise ValueError('No matching replay run exists in this database.')


def _open_run(memory: MemoryLayer, messages: list[HistoricalMessage], input_hash: str,
              require_existing: bool) -> str:
    ChatStore(memory.store.path)
    with memory.store.connect() as conn, conn:
        conn.executescript(REPLAY_SCHEMA)
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT chat_id,message_count FROM history_replay_runs WHERE input_hash=?',
                           (input_hash,)).fetchone()
        if row:
            if row['message_count'] != len(messages):
                raise RuntimeError('Replay input identity has an inconsistent message count.')
            return str(row['chat_id'])
        if require_existing:
            raise ValueError('No matching replay run exists in this database.')
        chat_id = uuid.uuid4().hex
        conn.execute('INSERT INTO lab_chats(id,title,imported) VALUES (?,?,1)', (chat_id, REPLAY_TITLE))
        conn.execute('INSERT INTO history_replay_runs(input_hash,chat_id,message_count) VALUES (?,?,?)',
                     (input_hash, chat_id, len(messages)))
        return chat_id


def _append_message(memory: MemoryLayer, input_hash: str, chat_id: str, item: HistoricalMessage) -> str:
    kind = USER_EVENT if item.role == 'user' else ASSISTANT_EVENT
    event = {'kind': kind, 'data': item.text, REPLAY_HISTORY_ID: item.id,
             REPLAY_TIME: item.ts.isoformat(), REPLAY_CHANNEL: item.channel}
    with memory.store.connect() as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        prior = conn.execute('SELECT message_row_id FROM history_replay_messages WHERE input_hash=? AND history_id=?',
                             (input_hash, item.id)).fetchone()
        if prior:
            return f"m{prior['message_row_id']:03d}"
        cursor = conn.execute('INSERT INTO lab_chat_events(chat_id,event) VALUES (?,?)',
                              (chat_id, json.dumps(event, ensure_ascii=False)))
        message_row_id = int(cursor.lastrowid)
        conn.execute('INSERT INTO history_replay_messages(input_hash,history_id,message_row_id) VALUES (?,?,?)',
                     (input_hash, item.id, message_row_id))
        if item.role == 'user':
            conn.execute('UPDATE lab_chats SET title=CASE WHEN title=? THEN ? ELSE title END, '
                         "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=?",
                         (NEW_CHAT_TITLE, item.text[:60], chat_id))
        return f'm{message_row_id:03d}'


def _load_outcome(memory: MemoryLayer, input_hash: str, history_id: str) -> ReplayOutcome | None:
    with memory.store.connect() as conn:
        row = conn.execute('SELECT outcome FROM history_replay_decisions WHERE input_hash=? AND history_id=?',
                           (input_hash, history_id)).fetchone()
    return ReplayOutcome.model_validate_json(row['outcome']) if row else None


def _record_outcome(conn: sqlite3.Connection, input_hash: str, history_id: str, outcome: ReplayOutcome) -> None:
    conn.execute('INSERT INTO history_replay_decisions(input_hash,history_id,status,outcome) VALUES (?,?,?,?)',
                 (input_hash, history_id, outcome.status, outcome.model_dump_json()))


def _count(outcome: ReplayOutcome, counts: dict[str, int]) -> None:
    counts[outcome.status] = counts.get(outcome.status, 0) + 1
    if outcome.status not in ('created', 'updated'):
        counts['other'] += 1


def replay(memory: MemoryLayer, rows: list[dict[str, object]], *, require_existing: bool = False) -> ReplayReport:
    messages = [HistoricalMessage.model_validate(row) for row in rows]
    if len({message.id for message in messages}) != len(messages):
        raise ValueError('History message IDs must be unique')
    if any(message.ts.utcoffset() is None for message in messages):
        raise ValueError('Historical timestamps must include a timezone')
    if messages != sorted(messages, key=lambda message: message.ts):
        raise ValueError('History messages must be in chronological order')
    input_hash = _input_hash(messages, memory.model, memory.index.model)
    chat_id = _open_run(memory, messages, input_hash, require_existing)
    counts = {'created': 0, 'updated': 0, 'other': 0, 'deleted': 0, 'no_change': 0,
              'clarification_needed': 0, 'rejected': 0}
    context: list[Message] = []
    chats = ChatStore(memory.store.path)
    for item in messages:
        message_id = _append_message(memory, input_hash, chat_id, item)
        if item.role == 'user':
            outcome = _load_outcome(memory, input_hash, item.id)
            if outcome is None:
                source = chats.message(message_id)
                if source is None:
                    raise RuntimeError('Historical user message mapping is missing.')

                def checkpoint(conn: sqlite3.Connection, result: ChangeResult) -> None:
                    _record_outcome(conn, input_hash, item.id,
                                    ReplayOutcome(status=result.status, result=result))

                try:
                    result = memory.change(item.text, context_mode='messages', context=context[-CONTEXT_MESSAGES:],
                                           chat_sources=[source], historical_time=item.ts, on_commit=checkpoint)
                    outcome = ReplayOutcome(status=result.status, result=result)
                except ValueError as exc:
                    outcome = ReplayOutcome(status='rejected', error_type=type(exc).__name__)
                if _load_outcome(memory, input_hash, item.id) is None:
                    with memory.store.connect() as conn, conn:
                        conn.execute('BEGIN IMMEDIATE')
                        _record_outcome(conn, input_hash, item.id, outcome)
            _count(outcome, counts)
        context = [*context[-CONTEXT_MESSAGES:], Message(role=item.role, content=item.text)]
    return ReplayReport(chat_id=chat_id, messages=len(messages), created=counts['created'],
                        updated=counts['updated'], other=counts['other'], deleted=counts['deleted'],
                        skipped=counts['no_change'], clarification=counts['clarification_needed'],
                        rejected=counts['rejected'])


def main() -> None:
    parser = argparse.ArgumentParser(description='Replay a chronological sample into an isolated SQLite database.')
    parser.add_argument('--database', type=Path, required=True, help='SQLite database for this replay')
    parser.add_argument('--limit', type=int, required=True, help='Number of chronological messages to replay')
    parser.add_argument('--resume', action='store_true', help='Resume only if this database has the matching input')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    if args.resume and not args.database.exists():
        parser.error('--resume requires an existing replay database')
    if not args.resume and args.database.exists():
        parser.error('Database already exists; choose a new path or pass --resume')
    rows: list[dict[str, object]] = []
    for path in sorted((ROOT / 'data' / 'history').glob('*.jsonl')):
        with path.open(encoding='utf-8') as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    rows.sort(key=lambda row: HistoricalMessage.model_validate(row).ts)
    selected = [HistoricalMessage.model_validate(row) for row in rows[:args.limit]]
    if any(message.ts.utcoffset() is None for message in selected):
        parser.error('Historical timestamps must include a timezone')
    agent_model = os.getenv(MODEL_ENVIRONMENT_KEY, DEFAULT_AGENT_MODEL)
    embedding_model = os.getenv(MODEL_ENV, DEFAULT_MODEL)
    if args.resume:
        try:
            _require_existing_run(args.database, _input_hash(selected, agent_model, embedding_model))
        except ValueError as exc:
            parser.error(str(exc))
    memory = MemoryLayer(args.database, OpenAI(), agent_model)
    seed_persona(memory.store, ROOT / 'data' / 'persona.json')
    report = replay(memory, rows[:args.limit], require_existing=args.resume)
    print(report.model_dump_json())


if __name__ == '__main__':
    main()
