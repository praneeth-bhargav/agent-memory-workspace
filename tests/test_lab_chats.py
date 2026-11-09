import sqlite3

from memory_layer.chat_store import ChatStore
from memory_layer.mock_actions import MockActions, MockEmail
from memory_layer.mock_sources import MockSources
from memory_layer import lab
from memory_layer.agent import AgentEvent
from fastapi.testclient import TestClient


def test_chat_events_and_titles_persist_across_store_instances(tmp_path):
    database = tmp_path / 'chats.sqlite3'
    store = ChatStore(database)
    first = store.create()
    second = store.create()
    store.append(first.id, {'kind': 'user', 'data': 'First question'})
    store.append(first.id, {'kind': 'answer', 'data': 'First answer'})
    store.append(second.id, {'kind': 'user', 'data': 'Other question'})
    reloaded = ChatStore(database)
    assert reloaded.get(first.id).title == 'First question'
    assert [event['data'] for event in reloaded.events(first.id)] == ['First question', 'First answer']
    assert [event['data'] for event in reloaded.events(second.id)] == ['Other question']
    assert len(reloaded.list_chats()) == 2


def test_chat_api_creates_reopens_and_streams_separate_histories(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, 'DATABASE', tmp_path / 'api.sqlite3')

    class FakeAgent:
        def history(self, chat_id):
            return []

        def pending_action(self, chat_id):
            return None

        def stream_answer(self, text, chat_id, message_id):
            yield AgentEvent(kind='answer', data=f'{chat_id}: {text}')

    monkeypatch.setitem(lab.app.dependency_overrides, lab.get_agent, lambda: FakeAgent())
    with TestClient(lab.app, base_url='http://localhost') as client:
        first = client.post('/api/chats').json()['id']
        second = client.post('/api/chats').json()['id']
        assert first != second
        assert client.post('/api/chat/stream', json={'text': 'First message', 'chat_id': first}).status_code == 200
        assert client.post('/api/chat/stream', json={'text': 'Second message', 'chat_id': second}).status_code == 200
        assert [e['data'] for e in client.get(f'/api/chats/{first}').json()['events']] == ['First message', f'{first}: First message']
        assert [e['data'] for e in client.get(f'/api/chats/{second}').json()['events']] == ['Second message', f'{second}: Second message']
        assert client.get('/api/chats/missing').status_code == 404
        assert client.post('/api/chat/stream', json={'text': 'No', 'chat_id': 'missing'}).status_code == 404


def test_legacy_chat_imports_once_when_reopened(tmp_path, monkeypatch):
    database = tmp_path / 'legacy.sqlite3'
    with sqlite3.connect(database) as conn:
        conn.execute('CREATE TABLE checkpoints (thread_id TEXT)')
        conn.execute("INSERT INTO checkpoints VALUES ('memory-lab')")
    monkeypatch.setattr(lab, 'DATABASE', database)

    class FakeAgent:
        def history(self, chat_id):
            return [{'kind': 'user', 'data': 'Existing conversation'}]

        def pending_action(self, chat_id):
            return None

    monkeypatch.setitem(lab.app.dependency_overrides, lab.get_agent, lambda: FakeAgent())
    with TestClient(lab.app, base_url='http://localhost') as client:
        assert client.get('/api/chats').json()[0]['id'] == 'memory-lab'
        for _ in range(2):
            assert client.get('/api/chats/memory-lab').json()['events'] == [
                {'kind': 'user', 'data': 'Existing conversation', 'message_id': 'm001'}]


def test_pending_actions_are_isolated_and_legacy_action_is_preserved(tmp_path):
    database = tmp_path / 'mock.sqlite3'
    sources = MockSources(database)
    sources.seed()
    email = MockEmail(to=['arjun@kestrelworks.example'], subject='Legacy draft', body='Legacy body')
    with sqlite3.connect(database) as conn:
        conn.execute('CREATE TABLE mock_pending_action (id INTEGER PRIMARY KEY CHECK(id=1), kind TEXT NOT NULL, payload TEXT NOT NULL)')
        conn.execute('INSERT INTO mock_pending_action VALUES (1, ?, ?)', ('email', email.model_dump_json()))
    assert ChatStore(database).get('memory-lab').title == 'Previous chat'
    old = MockActions(sources, '2026-05-18T09:30:00+05:30', 'rohan@kestrelworks.example')
    fresh = MockActions(sources, '2026-05-18T09:30:00+05:30', 'rohan@kestrelworks.example', 'new-chat')
    assert old.pending().payload.subject == 'Legacy draft'
    with sqlite3.connect(database) as conn:
        assert conn.execute('SELECT count(*) FROM mock_pending_action').fetchone()[0] == 0
    assert fresh.pending() is None
    fresh.prepare_email(MockEmail(to=['arjun@kestrelworks.example'], subject='New draft', body='New body'))
    fresh.confirm()
    assert old.pending().payload.subject == 'Legacy draft'
    assert not sources.mail('Legacy body')
    assert sum(message.body == 'New body' for thread in sources.mail('New body') for message in thread.messages) == 1
    assert MockActions(sources, '2026-05-18T09:30:00+05:30', 'rohan@kestrelworks.example').pending().payload.subject == 'Legacy draft'
