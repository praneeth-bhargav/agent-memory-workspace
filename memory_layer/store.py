"""Private SQLite storage for the independent memory layer."""
import re
import sqlite3
from contextlib import closing
from collections.abc import Callable
from os import PathLike
from typing import Final, Literal

from .types import Memory, MemoryProvenance, MemoryRecord

CREATE: Final = 'create'
UPDATE: Final = 'update'
DELETE: Final = 'delete'
MEMORIES_CHANGED = 'Memories changed during the request; retry'
FILE_REQUIRED = 'A SQLite database file is required'
IN_MEMORY_PATH = ':memory:'
ID_COLUMN = 'id'
TEXT_COLUMN = 'text'

FORGETTABLE_WORDS = frozenset({'a', 'an', 'are', 'as', 'at', 'be', 'do', 'for', 'from', 'i', 'in', 'is',
                               'it', 'like', 'likes', 'my', 'of', 'on', 'or', 'prefer', 'prefers', 'the',
                               'that', 'this', 'to', 'user', 'with',
                               'without', 'you'})
SOURCE_CITATION = re.compile(r'\[(mail|slack|calendar):([^\]]+)\]', re.I)
FORGET_CITATION = re.compile(r'\[(?:mail|slack|calendar|chat):[^\]]+\]', re.I)
FORGET_REAUTHORIZATION = re.compile(r'\b(?:remember|save|keep)\b.{0,80}\b(?:again|re-?remember)\b|\b(?:again|re-?remember)\b.{0,80}\b(?:remember|save|keep)\b', re.I)
FORGOTTEN_MEMORY = 'This fact was forgotten. The user must explicitly ask to remember it again.'
FORGET_OVERLAP_THRESHOLD = 2 / 3
StoreCommitCallback = Callable[[sqlite3.Connection, list[str]], None]

CREATE_TABLES = '''
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1
    );
    CREATE TABLE IF NOT EXISTS memory_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id INTEGER NOT NULL REFERENCES memories(id),
        action TEXT NOT NULL,
        text TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS memory_forget_suppressions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
'''
ENABLE_FOREIGN_KEYS = 'PRAGMA foreign_keys = ON'
ENABLE_SECURE_DELETE = 'PRAGMA secure_delete = ON'
SELECT_ACTIVE = 'SELECT id, text FROM memories WHERE active = 1 ORDER BY id'
BEGIN_TRANSACTION = 'BEGIN IMMEDIATE'
INSERT_MEMORY = 'INSERT INTO memories(text) VALUES (?)'
UPDATE_MEMORY = 'UPDATE memories SET text = ? WHERE id = ? AND active = 1'
DELETE_MEMORY = 'UPDATE memories SET active = 0 WHERE id = ? AND active = 1'
INSERT_HISTORY = 'INSERT INTO memory_history(memory_id, action, text) VALUES (?, ?, ?)'


class Store:
    def __init__(self, path: str | PathLike[str]) -> None:
        if str(path) == IN_MEMORY_PATH:
            raise ValueError(FILE_REQUIRED)
        self.path = path
        with closing(self.connect()) as conn, conn:
            conn.executescript(CREATE_TABLES)
            conn.execute(BEGIN_TRANSACTION)
            columns = {row['name'] for row in conn.execute('PRAGMA table_info(memories)')}
            if 'is_persona' not in columns:
                conn.execute('ALTER TABLE memories ADD COLUMN is_persona INTEGER NOT NULL DEFAULT 0')
            conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS one_persona ON memories(is_persona) WHERE is_persona = 1')

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute(ENABLE_FOREIGN_KEYS)
        conn.execute(ENABLE_SECURE_DELETE)
        return conn

    @staticmethod
    def active(conn: sqlite3.Connection) -> list[Memory]:
        return [Memory(id=str(row[ID_COLUMN]), text=row[TEXT_COLUMN]) for row in conn.execute(SELECT_ACTIVE)]

    def snapshot(self) -> list[Memory]:
        with closing(self.connect()) as conn:
            return self.active(conn)

    def records(self) -> list[MemoryRecord]:
        with closing(self.connect()) as conn:
            rows = conn.execute('SELECT id,text,is_persona FROM memories WHERE active=1 ORDER BY id').fetchall()
            has_chat_sources = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_chat_sources'").fetchone()
            records: list[MemoryRecord] = []
            for row in rows:
                provenance = [MemoryProvenance(kind=kind.lower(), source_id=source_id)
                              for kind, source_id in SOURCE_CITATION.findall(row['text'])]
                if has_chat_sources:
                    links = conn.execute('''SELECT e.id,e.chat_id FROM memory_chat_sources s
                        JOIN lab_chat_events e ON e.id=s.message_row_id WHERE s.memory_id=? ORDER BY e.id''',
                        (row['id'],)).fetchall()
                    provenance.extend(MemoryProvenance(kind='chat', source_id=f"m{link['id']:03d}",
                                                       chat_id=link['chat_id']) for link in links)
                records.append(MemoryRecord(id=str(row['id']), text=row['text'],
                                            is_persona=bool(row['is_persona']), provenance=provenance))
            return records

    @staticmethod
    def _forget_tokens(text: str) -> set[str]:
        return {word for word in re.findall(r'[^\W_]+', FORGET_CITATION.sub('', text).casefold())
                if word not in FORGETTABLE_WORDS}

    @classmethod
    def _matches_forget_text(cls, candidate: str, forgotten: str) -> bool:
        current, target = cls._forget_tokens(candidate), cls._forget_tokens(forgotten)
        if not current or not target:
            return False
        shared = len(current & target)
        if current == target:
            return True
        return len(current) > 1 and len(target) > 1 and shared / min(len(current), len(target)) >= FORGET_OVERLAP_THRESHOLD

    def is_forgotten(self, text: str) -> bool:
        with closing(self.connect()) as conn:
            return self._is_forgotten_in_connection(conn, text)

    @staticmethod
    def explicitly_reauthorized(instruction: str) -> bool:
        return bool(FORGET_REAUTHORIZATION.search(instruction))

    def forget(self, memory_id: str, expected_text: str) -> None:
        with closing(self.connect()) as conn, conn:
            conn.execute(BEGIN_TRANSACTION)
            row = conn.execute('SELECT text,is_persona,active FROM memories WHERE id=?', (memory_id,)).fetchone()
            if row is None or not row['active']:
                raise ValueError('Memory not found.')
            if row['is_persona']:
                raise ValueError('Edit the persona in the workspace instead of forgetting it here.')
            if row['text'] != expected_text:
                raise RuntimeError(MEMORIES_CHANGED)
            conn.execute('INSERT INTO memory_forget_suppressions(text) VALUES (?)', (row['text'],))
            conn.execute(DELETE_MEMORY, (memory_id,))
            conn.execute(INSERT_HISTORY, (memory_id, DELETE, row['text']))
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='semantic_vectors'").fetchone():
                conn.execute('DELETE FROM semantic_vectors WHERE memory_id=?', (memory_id,))
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_chat_sources'").fetchone():
                conn.execute('DELETE FROM memory_chat_sources WHERE memory_id=?', (memory_id,))

    def edit(self, memory_id: str, expected_text: str, text: str) -> MemoryRecord:
        with closing(self.connect()) as conn, conn:
            conn.execute(BEGIN_TRANSACTION)
            row = conn.execute('SELECT text,is_persona,active FROM memories WHERE id=?', (memory_id,)).fetchone()
            if row is None or not row['active']:
                raise ValueError('Memory not found.')
            if row['is_persona']:
                raise ValueError('Edit the persona in the workspace instead.')
            if row['text'] != expected_text:
                raise RuntimeError(MEMORIES_CHANGED)
            if self._is_forgotten_in_connection(conn, text):
                raise ValueError(FORGOTTEN_MEMORY)
            conn.execute(UPDATE_MEMORY, (text, memory_id))
            conn.execute(INSERT_HISTORY, (memory_id, UPDATE, text))
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='semantic_vectors'").fetchone():
                conn.execute('DELETE FROM semantic_vectors WHERE memory_id=?', (memory_id,))
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_chat_sources'").fetchone():
                conn.execute('DELETE FROM memory_chat_sources WHERE memory_id=?', (memory_id,))
        return next(record for record in self.records() if record.id == memory_id)

    def is_persona(self, memory_id: str) -> bool:
        with closing(self.connect()) as conn:
            row = conn.execute('SELECT is_persona FROM memories WHERE id = ?', (memory_id,)).fetchone()
            return bool(row and row['is_persona'])

    def chat_source_ids(self, memory_id: str) -> set[str]:
        with closing(self.connect()) as conn:
            rows = conn.execute('SELECT message_row_id FROM memory_chat_sources WHERE memory_id=?', (memory_id,))
            return {f'm{row[0]:03d}' for row in rows}

    @classmethod
    def _is_forgotten_in_connection(cls, conn: sqlite3.Connection, text: str) -> bool:
        return any(cls._matches_forget_text(text, row['text'])
                   for row in conn.execute('SELECT text FROM memory_forget_suppressions'))

    @classmethod
    def _clear_matching_suppressions(cls, conn: sqlite3.Connection, text: str) -> None:
        rows = conn.execute('SELECT id,text FROM memory_forget_suppressions').fetchall()
        conn.executemany('DELETE FROM memory_forget_suppressions WHERE id=?',
                         [(row['id'],) for row in rows if cls._matches_forget_text(text, row['text'])])

    def apply(self, snapshot: list[Memory], action: Literal['create', 'update', 'delete'],
              text: str, ids: list[str], source_guard: Callable[[sqlite3.Connection], bool] | None = None,
              source_message_row_id: int | None = None, allow_forgotten: bool = False,
              on_commit: StoreCommitCallback | None = None) -> list[str]:
        with closing(self.connect()) as conn, conn:
            conn.execute(BEGIN_TRANSACTION)
            if self.active(conn) != snapshot:
                raise RuntimeError(MEMORIES_CHANGED)
            if source_guard is not None and not source_guard(conn):
                raise RuntimeError('Source changed during memory decision; retry')
            if action in (CREATE, UPDATE) and not allow_forgotten and self._is_forgotten_in_connection(conn, text):
                raise ValueError(FORGOTTEN_MEMORY)
            if allow_forgotten:
                self._clear_matching_suppressions(conn, text)
            has_chat_sources = bool(conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_chat_sources'").fetchone())
            if action == CREATE:
                new_id = conn.execute(INSERT_MEMORY, (text,)).lastrowid
                ids = [str(new_id)]
            elif action == UPDATE:
                for memory_id in ids:
                    conn.execute(UPDATE_MEMORY, (text, memory_id))
                    if has_chat_sources:
                        conn.execute('DELETE FROM memory_chat_sources WHERE memory_id=?', (memory_id,))
            elif action == DELETE:
                for memory_id in ids:
                    original = next(memory.text for memory in snapshot if memory.id == memory_id)
                    conn.execute('INSERT INTO memory_forget_suppressions(text) VALUES (?)', (original,))
                    conn.execute(DELETE_MEMORY, (memory_id,))
                    if has_chat_sources:
                        conn.execute('DELETE FROM memory_chat_sources WHERE memory_id=?', (memory_id,))
                    conn.execute(INSERT_HISTORY, (memory_id, action, original))
                if on_commit is not None:
                    on_commit(conn, ids)
                return ids
            for memory_id in ids:
                if source_message_row_id is not None:
                    conn.execute('INSERT INTO memory_chat_sources(memory_id,message_row_id) VALUES (?,?)',
                                 (memory_id, source_message_row_id))
                conn.execute(INSERT_HISTORY, (memory_id, action, text))
            if on_commit is not None:
                on_commit(conn, ids)
            return ids
