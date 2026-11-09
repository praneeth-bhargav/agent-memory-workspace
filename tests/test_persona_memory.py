import sqlite3
from types import SimpleNamespace

from fastapi.testclient import TestClient

from memory_layer import lab
from memory_layer.store import Store
from memory_layer.service import MemoryLayer
from memory_layer.types import ChangeDecision


def test_persona_is_read_only_and_synchronized(tmp_path, monkeypatch):
    database = tmp_path / 'test.sqlite3'
    monkeypatch.setattr(lab, 'DATABASE', database)
    with TestClient(lab.app, base_url='http://localhost') as client:
        initial = client.get('/api/persona')
        assert initial.status_code == 200
        text = initial.json()['text']
        assert 'Rohan Menon' in text and 'Kestrel Works' in text and 'Priya Nair' in text
        assert '2026-05-18T09:30:00+05:30' not in text
        with sqlite3.connect(database) as conn:
            memory_id = conn.execute('SELECT id FROM memories WHERE is_persona = 1').fetchone()[0]
            conn.execute('CREATE TABLE semantic_vectors (memory_id INTEGER PRIMARY KEY, content_hash TEXT, model TEXT, vector TEXT)')
            conn.execute('INSERT INTO semantic_vectors VALUES (?, ?, ?, ?)', (memory_id, 'old', 'model', '[]'))
        response = client.put('/api/persona', json={'text': 'Rohan runs Acme Labs.', 'expected_text': text})
        assert response.status_code == 405
        with sqlite3.connect(database) as conn:
            conn.execute('UPDATE memories SET text = ? WHERE id = ?', ('Outdated persona', memory_id))
        assert client.get('/api/persona').json()['text'] == text
        with sqlite3.connect(database) as conn:
            assert conn.execute('SELECT id, text FROM memories WHERE is_persona = 1').fetchall() == [(memory_id, text)]
            assert conn.execute('SELECT * FROM semantic_vectors').fetchall() == []
            assert conn.execute('SELECT action FROM memory_history WHERE memory_id = ? ORDER BY id', (memory_id,)).fetchall() == [('create',), ('update',)]


def test_fixed_persona_is_restored_if_inactive(tmp_path, monkeypatch):
    database = tmp_path / 'test.sqlite3'
    monkeypatch.setattr(lab, 'DATABASE', database)
    with TestClient(lab.app, base_url='http://localhost') as client:
        assert client.get('/api/persona').json()['text']
        store = Store(database)
        snapshot = store.snapshot()
        store.apply(snapshot, 'delete', '', [snapshot[0].id])
        assert 'Rohan Menon' in client.get('/api/persona').json()['text']
        assert client.put('/api/persona', json={'text': 'A replacement persona.', 'expected_text': None}).status_code == 405
        assert len(store.snapshot()) == 1


def test_agent_does_not_overwrite_or_delete_whole_persona_for_one_fact(tmp_path):
    class Completions:
        def __init__(self):
            self.actions = iter(['update', 'delete'])

        def parse(self, **kwargs):
            result = ChangeDecision(action=next(self.actions), text='New company', ids=['1'], question='')
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=result))])

    embeddings = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(data=[SimpleNamespace(embedding=[1.0])]))
    provider = SimpleNamespace(chat=SimpleNamespace(completions=Completions()), embeddings=embeddings)
    memory = MemoryLayer(tmp_path / 'test.sqlite3', provider, 'test-model')
    from memory_layer.persona import seed_persona
    seed_persona(memory.store, lab.PERSONA)
    original = memory.store.snapshot()
    for instruction in ['Change my company', 'Forget my company']:
        result = memory.change(instruction, context_mode='messages', context=[])
        assert result.status == 'clarification_needed'
        assert 'persona' in result.question.lower()
        assert memory.store.snapshot() == original


def test_existing_memory_schema_is_preserved(tmp_path, monkeypatch):
    database = tmp_path / 'test.sqlite3'
    with sqlite3.connect(database) as conn:
        conn.execute('CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1)')
        conn.execute("INSERT INTO memories(text) VALUES ('An existing memory')")
    monkeypatch.setattr(lab, 'DATABASE', database)
    with TestClient(lab.app, base_url='http://localhost') as client:
        assert client.get('/api/persona').status_code == 200
    with sqlite3.connect(database) as conn:
        assert conn.execute('SELECT text, is_persona FROM memories ORDER BY id').fetchall()[0] == ('An existing memory', 0)
        assert conn.execute('SELECT count(*) FROM memories WHERE is_persona = 1').fetchone()[0] == 1
