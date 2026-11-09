"""Persistent chat list and user-visible event history."""
import json
import re
import sqlite3
import uuid
from contextlib import closing
from os import PathLike

from pydantic import BaseModel

LEGACY_CHAT = 'memory-lab'
NEW_CHAT = 'New chat'
MESSAGE_ID = re.compile(r'^m([0-9]+)$')
SCHEMA = '''
CREATE TABLE IF NOT EXISTS lab_chats (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, imported INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE TABLE IF NOT EXISTS lab_chat_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT NOT NULL REFERENCES lab_chats(id),
    event TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_chat_sources (
    memory_id INTEGER NOT NULL REFERENCES memories(id),
    message_row_id INTEGER NOT NULL REFERENCES lab_chat_events(id),
    PRIMARY KEY(memory_id, message_row_id)
);
'''


class Chat(BaseModel):
    id: str
    title: str


class ChatStore:
    def __init__(self, path: str | PathLike[str]) -> None:
        self.path = path
        with closing(self.connect()) as conn, conn:
            conn.executescript(SCHEMA)
            tables = {row['name'] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            checkpoint = 'checkpoints' in tables and conn.execute(
                'SELECT 1 FROM checkpoints WHERE thread_id = ? LIMIT 1', (LEGACY_CHAT,)).fetchone()
            pending = ('mock_pending_action' in tables and conn.execute('SELECT 1 FROM mock_pending_action WHERE id=1').fetchone()) or (
                'mock_pending_actions' in tables and conn.execute('SELECT 1 FROM mock_pending_actions WHERE chat_id=?', (LEGACY_CHAT,)).fetchone())
            if checkpoint or pending:
                conn.execute('INSERT OR IGNORE INTO lab_chats(id,title,imported) VALUES (?,?,0)', (LEGACY_CHAT, 'Previous chat'))

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys = ON')
        return conn

    def create(self) -> Chat:
        chat = Chat(id=uuid.uuid4().hex, title=NEW_CHAT)
        with closing(self.connect()) as conn, conn:
            conn.execute('INSERT INTO lab_chats(id,title) VALUES (?,?)', (chat.id, chat.title))
        return chat

    def ensure_legacy(self) -> None:
        with closing(self.connect()) as conn, conn:
            conn.execute('INSERT OR IGNORE INTO lab_chats(id,title,imported) VALUES (?,?,0)', (LEGACY_CHAT, 'Previous chat'))

    def list_chats(self) -> list[Chat]:
        with closing(self.connect()) as conn:
            return [Chat(id=row['id'], title=row['title']) for row in conn.execute(
                'SELECT id,title FROM lab_chats ORDER BY updated_at DESC, rowid DESC')]

    def get(self, chat_id: str) -> Chat | None:
        with closing(self.connect()) as conn:
            row = conn.execute('SELECT id,title FROM lab_chats WHERE id=?', (chat_id,)).fetchone()
            return Chat(id=row['id'], title=row['title']) if row else None

    def needs_import(self, chat_id: str) -> bool:
        with closing(self.connect()) as conn:
            row = conn.execute('SELECT imported FROM lab_chats WHERE id=?', (chat_id,)).fetchone()
            return bool(row and not row['imported'])

    def import_events(self, chat_id: str, events: list[dict[str, object]]) -> None:
        with closing(self.connect()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT imported FROM lab_chats WHERE id=?', (chat_id,)).fetchone()
            if row and not row['imported']:
                conn.executemany('INSERT INTO lab_chat_events(chat_id,event) VALUES (?,?)',
                                 [(chat_id, json.dumps(event)) for event in events])
                conn.execute('UPDATE lab_chats SET imported=1 WHERE id=?', (chat_id,))

    def append(self, chat_id: str, event: dict[str, object]) -> str:
        with closing(self.connect()) as conn, conn:
            cursor = conn.execute('INSERT INTO lab_chat_events(chat_id,event) VALUES (?,?)', (chat_id, json.dumps(event)))
            data = event.get('data')
            if event.get('kind') == 'user' and isinstance(data, str):
                conn.execute('UPDATE lab_chats SET title=CASE WHEN title=? THEN ? ELSE title END, '
                             "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=?",
                             (NEW_CHAT, data[:60], chat_id))
            return f'm{cursor.lastrowid:03d}'

    @staticmethod
    def message_in_connection(conn: sqlite3.Connection, message_id: str) -> dict[str, str] | None:
        match = MESSAGE_ID.fullmatch(message_id)
        if not match:
            return None
        row = conn.execute('SELECT id,chat_id,event FROM lab_chat_events WHERE id=?', (int(match.group(1)),)).fetchone()
        if not row:
            return None
        event = json.loads(row['event'])
        if event.get('kind') != 'user' or not isinstance(event.get('data'), str):
            return None
        result = {'chat_id': row['chat_id'], 'message_id': f"m{row['id']:03d}", 'text': event['data']}
        if isinstance(event.get('ts'), str):
            result['timestamp'] = event['ts']
        return result

    def message(self, message_id: str) -> dict[str, str] | None:
        with closing(self.connect()) as conn:
            return self.message_in_connection(conn, message_id)

    def user_messages(self, chat_id: str, before_message_id: str) -> list[dict[str, str]]:
        match = MESSAGE_ID.fullmatch(before_message_id)
        if not match:
            return []
        with closing(self.connect()) as conn:
            rows = conn.execute('SELECT id FROM lab_chat_events WHERE chat_id=? AND id<? ORDER BY id DESC LIMIT 50',
                                (chat_id, int(match.group(1)))).fetchall()
            messages = [self.message_in_connection(conn, f"m{row['id']:03d}") for row in reversed(rows)]
            return [message for message in messages if message is not None]

    def events(self, chat_id: str) -> list[dict[str, object]]:
        with closing(self.connect()) as conn:
            return [{**json.loads(row['event']), 'message_id': f"m{row['id']:03d}"} for row in conn.execute(
                'SELECT id,event FROM lab_chat_events WHERE chat_id=? ORDER BY id', (chat_id,))]
