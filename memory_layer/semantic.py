"""Persisted embedding index for candidate memory retrieval."""
import hashlib
import json
import logging
import sqlite_vec  # type: ignore[import-untyped]
import os
from contextlib import closing
from collections.abc import Callable

from openai import OpenAI
from pydantic import BaseModel

from .store import Store
from .types import Memory

MODEL_ENV = 'OPENAI_EMBEDDING_MODEL'
DEFAULT_MODEL = 'text-embedding-3-small'
MAX_RESULTS = 8
HASH_COLUMN = 'content_hash'
MODEL_COLUMN = 'model'
VECTOR_COLUMN = 'vector'
TEXT_COLUMN = 'text'
CREATE_INDEX = '''CREATE TABLE IF NOT EXISTS semantic_vectors (
    memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL, model TEXT NOT NULL, vector TEXT NOT NULL
)'''
SELECT_VECTOR = 'SELECT content_hash, model, vector FROM semantic_vectors WHERE memory_id = ?'
SELECT_ACTIVE_TEXT = 'SELECT text FROM memories WHERE id = ? AND active = 1'
UPSERT_VECTOR = '''INSERT INTO semantic_vectors(memory_id, content_hash, model, vector)
    VALUES (?, ?, ?, ?) ON CONFLICT(memory_id) DO UPDATE SET
    content_hash=excluded.content_hash, model=excluded.model, vector=excluded.vector'''
EMBEDDING_LOG = 'memory.embedding run_id=%s chat_id=%s model=%s purpose=%s memory_id=%s input=%s'
CANDIDATES_TABLE = 'CREATE TEMP TABLE candidates(memory_id TEXT, position INTEGER, vector TEXT)'
SCORE_CANDIDATES = '''SELECT memory_id,
    CASE WHEN json_array_length(vector) = ?
         THEN COALESCE(1.0 - vec_distance_cosine(?, vector), 0.0)
         ELSE 0.0 END AS similarity
    FROM candidates ORDER BY similarity DESC, position LIMIT ?'''
logger = logging.getLogger('uvicorn.error')


class Vector(BaseModel):
    values: list[float]


class SemanticIndex:
    def __init__(self, store: Store, client: OpenAI, trace_context: Callable[[], tuple[str, str]]) -> None:
        self.store = store
        self.client = client
        self.trace_context = trace_context
        self.model = os.getenv(MODEL_ENV, DEFAULT_MODEL)
        with closing(self.store.connect()) as conn, conn:
            conn.execute(CREATE_INDEX)

    def _embed(self, text: str, purpose: str, memory_id: str = '') -> list[float]:
        run_id, chat_id = self.trace_context()
        logger.info(EMBEDDING_LOG, run_id, chat_id, self.model, purpose, memory_id, json.dumps(text, ensure_ascii=False))
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def _vector(self, memory: Memory) -> list[float]:
        digest = hashlib.sha256(memory.text.encode('utf-8')).hexdigest()
        with closing(self.store.connect()) as conn:
            row = conn.execute(SELECT_VECTOR, (memory.id,)).fetchone()
        if row and row[HASH_COLUMN] == digest and row[MODEL_COLUMN] == self.model:
            return Vector.model_validate_json(row[VECTOR_COLUMN]).values
        vector = self._embed(memory.text, 'memory', memory.id)
        with closing(self.store.connect()) as conn, conn:
            current = conn.execute(SELECT_ACTIVE_TEXT, (memory.id,)).fetchone()
            if current and current[TEXT_COLUMN] == memory.text:
                conn.execute(UPSERT_VECTOR, (memory.id, digest, self.model, Vector(values=vector).model_dump_json()))
        return vector

    def search(self, query: str, memories: list[Memory]) -> list[Memory]:
        if not memories:
            run_id, chat_id = self.trace_context()
            logger.info('memory.search run_id=%s chat_id=%s query=%s total=0 limit=%s',
                        run_id, chat_id, json.dumps(query, ensure_ascii=False), MAX_RESULTS)
            return []
        query_vector = self._embed(query, 'query')
        vectors = [self._vector(memory) for memory in memories]
        with closing(self.store.connect()) as conn:
            conn.enable_load_extension(True)
            try:
                sqlite_vec.load(conn)
            finally:
                conn.enable_load_extension(False)
            conn.execute(CANDIDATES_TABLE)
            conn.executemany('INSERT INTO candidates VALUES (?, ?, ?)',
                             ((memory.id, position, json.dumps(vector))
                              for position, (memory, vector) in enumerate(zip(memories, vectors))))
            ranked_ids = conn.execute(SCORE_CANDIDATES, (len(query_vector), json.dumps(query_vector), MAX_RESULTS)).fetchall()
        by_id = {memory.id: memory for memory in memories}
        run_id, chat_id = self.trace_context()
        logger.info('memory.search run_id=%s chat_id=%s query=%s total=%s limit=%s',
                    run_id, chat_id, json.dumps(query, ensure_ascii=False), len(memories), MAX_RESULTS)
        for position, row in enumerate(ranked_ids):
            memory = by_id[str(row['memory_id'])]
            score = float(row['similarity'])
            logger.info('memory.candidate run_id=%s chat_id=%s score=%.6f selected=%s memory=%s',
                        run_id, chat_id, score, position < MAX_RESULTS, memory.model_dump_json())
        return [by_id[str(row['memory_id'])] for row in ranked_ids]
