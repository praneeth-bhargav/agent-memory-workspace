import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from memory_layer import lab
from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.graph.message import add_messages

from memory_layer.agent import forgotten_message_removals
from memory_layer.chat_store import ChatStore
from memory_layer.persona import seed_persona
from memory_layer.service import MemoryLayer
from memory_layer.store import Store
from memory_layer.types import ChangeDecision


class Client:
    def __init__(self, *decisions):
        self.decisions = iter(decisions)
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self.parse))
        self.embeddings = SimpleNamespace(create=lambda **kwargs:
            SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])]))

    def parse(self, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=next(self.decisions)))])


def decision(action, text='', ids=None):
    return ChangeDecision(action=action, text=text, ids=ids or [], question='')


def setup_lab(tmp_path, monkeypatch):
    database = tmp_path / 'memories.sqlite3'
    monkeypatch.setattr(lab, 'DATABASE', database)
    lab.get_sources.cache_clear()
    memory = MemoryLayer(database, None, 'synthetic')
    seed_persona(memory.store, lab.PERSONA)
    chat_store = ChatStore(database)
    chat = chat_store.create()
    chat_store.append(chat.id, {'kind': 'user', 'data': 'Original user preference.'})
    return database, memory


def test_memory_api_lists_provenance_and_protects_persona(tmp_path, monkeypatch):
    _, memory = setup_lab(tmp_path, monkeypatch)
    with memory.store.connect() as conn, conn:
        conn.execute("INSERT INTO memories(text) VALUES ('Prefers coffee [chat:m001]')")
        conn.execute('INSERT INTO memory_chat_sources(memory_id,message_row_id) VALUES (2,1)')
    try:
        with TestClient(lab.app, base_url='http://localhost') as client:
            response = client.get('/api/memories')
            assert response.status_code == 200
            records = response.json()
            assert [record['text'] for record in records] == [memory.store.snapshot()[0].text,
                                                               'Prefers coffee [chat:m001]']
            assert records[0]['is_persona'] is True
            assert records[1]['provenance'] == [{'kind': 'chat', 'source_id': 'm001',
                                                  'chat_id': records[1]['provenance'][0]['chat_id']}]
            assert records[1]['provenance'][0]['chat_id']
            assert client.put('/api/memories/1', json={'expected_text': records[0]['text'],
                                                       'text': 'Changed persona'}).status_code == 409
            assert client.request('DELETE', '/api/memories/1', json={'expected_text': records[0]['text']}).status_code == 409
    finally:
        lab.get_sources.cache_clear()


def test_memory_api_edit_is_stale_safe_and_invalidates_old_provenance_and_vectors(tmp_path, monkeypatch):
    _, memory = setup_lab(tmp_path, monkeypatch)
    memory.store.apply(memory.store.snapshot(), 'create', 'Prefers coffee [chat:m001] [slack:slack:1]', [])
    with memory.store.connect() as conn, conn:
        conn.execute('INSERT INTO memory_chat_sources(memory_id,message_row_id) VALUES (2,1)')
        conn.execute("INSERT INTO semantic_vectors(memory_id,content_hash,model,vector) VALUES (2,'old','test','{}')")
    try:
        with TestClient(lab.app, base_url='http://localhost') as client:
            stale = client.put('/api/memories/2', json={'expected_text': 'Changed elsewhere', 'text': 'Prefers tea'})
            assert stale.status_code == 409
            updated = client.put('/api/memories/2', json={'expected_text': 'Prefers coffee [chat:m001] [slack:slack:1]',
                                                           'text': 'Prefers tea'})
            assert updated.status_code == 200
            assert updated.json()['text'] == 'Prefers tea'
            assert updated.json()['provenance'] == []
        with memory.store.connect() as conn:
            assert conn.execute('SELECT COUNT(*) FROM semantic_vectors WHERE memory_id=2').fetchone()[0] == 0
            assert conn.execute('SELECT COUNT(*) FROM memory_chat_sources WHERE memory_id=2').fetchone()[0] == 0
            assert conn.execute('SELECT text FROM memory_history WHERE memory_id=2 ORDER BY id DESC LIMIT 1').fetchone()[0] == 'Prefers tea'
    finally:
        lab.get_sources.cache_clear()


def test_forgetting_suppresses_paraphrase_until_explicit_reauthorization(tmp_path):
    provider = Client(decision('create', 'User prefers black coffee without sugar.'),
                      decision('create', 'The user likes plain coffee with no sugar.'),
                      decision('create', 'The user likes plain coffee with no sugar.'))
    memory = MemoryLayer(tmp_path / 'memories.sqlite3', provider, 'test')
    assert memory.change('Remember that I like black coffee without sugar', context_mode='summary', context='').status == 'created'
    original = memory.store.snapshot()[0]
    memory.store.forget(original.id, original.text)
    suppressed = memory.change('Remember that I like black coffee without sugar', context_mode='summary', context='')
    assert suppressed.status == 'no_change'
    assert memory.store.snapshot() == []
    assert memory.change('I explicitly want you to remember again that I like plain coffee with no sugar',
                         context_mode='summary', context='').status == 'created'
    assert len(memory.store.snapshot()) == 1
    with memory.store.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM memory_forget_suppressions').fetchone()[0] == 0


def test_forgotten_messages_are_removed_from_model_history_without_deleting_chat_events(tmp_path):
    database = tmp_path / 'memories.sqlite3'
    memory = MemoryLayer(database, None, 'test')
    memory.store.apply(memory.store.snapshot(), 'create', 'Prefers coffee', [])
    record = memory.store.snapshot()[0]
    memory.store.forget(record.id, record.text)
    messages = [HumanMessage(id='forgotten', content='I prefer coffee'),
                HumanMessage(id='unrelated', content='My meeting is at noon')]

    removals = forgotten_message_removals(messages, memory.store)

    assert removals == [RemoveMessage(id='forgotten')]
    assert add_messages(messages, removals) == [messages[1]]
    with memory.store.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM memory_history WHERE action="delete"').fetchone()[0] == 1


def test_memory_api_forget_blocks_resurrection_and_keeps_original_chat(tmp_path, monkeypatch):
    database, memory = setup_lab(tmp_path, monkeypatch)
    memory.store.apply(memory.store.snapshot(), 'create', 'Prefers coffee', [])
    try:
        with TestClient(lab.app, base_url='http://localhost') as client:
            deleted = client.request('DELETE', '/api/memories/2', json={'expected_text': 'Prefers coffee'})
            assert deleted.status_code == 200
            assert deleted.json()['status'] == 'forgotten'
            remaining = client.get('/api/memories').json()
            assert [record['text'] for record in remaining] == [memory.store.snapshot()[0].text]
            memory.store.apply(memory.store.snapshot(), 'create', 'Prefers tea', [])
            blocked_edit = client.put('/api/memories/3', json={'expected_text': 'Prefers tea',
                                                                'text': 'Prefers coffee'})
            assert blocked_edit.status_code == 409
        provider = Client(decision('create', 'Prefers coffee'))
        retry = MemoryLayer(database, provider, 'test').change('I like coffee',
                                                              context_mode='summary', context='')
        assert retry.status == 'no_change'
        assert [item.text for item in memory.store.snapshot()] == [remaining[0]['text'], 'Prefers tea']
        with memory.store.connect() as conn:
            assert conn.execute('SELECT COUNT(*) FROM lab_chat_events').fetchone()[0] == 1
            assert conn.execute('SELECT COUNT(*) FROM memory_history WHERE action = "delete"').fetchone()[0] == 1
    finally:
        lab.get_sources.cache_clear()
