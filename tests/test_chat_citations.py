import json
import logging
import sqlite3
from contextlib import closing
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import Field, PrivateAttr

from memory_layer import lab
from memory_layer.agent import MemoryAgent
from memory_layer.chat_store import ChatStore
from memory_layer.mock_sources import MockSources
from memory_layer.service import ChatChangeDecision, MemoryLayer
from memory_layer.source_memory import GroundingDecision
from memory_layer.source_references import SourceReferenceSelection, validate_source_references
from memory_layer.types import ReadDecision


class ScriptedModel(BaseChatModel):
    responses: list[AIMessage] = Field(default_factory=list)
    _index: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return 'synthetic'

    def bind_tools(self, tools: Any, **kwargs: Any) -> 'ScriptedModel':
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        response = self.responses[self._index]
        self._index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


def call(name, args, identifier):
    return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': identifier, 'type': 'tool_call'}])


def build_system(tmp_path, monkeypatch):
    database = tmp_path / 'isolated.sqlite3'
    memory = MemoryLayer(database, None, 'synthetic')
    memory.index.search = lambda query, memories: memories
    sources = MockSources(database)
    sources.seed()
    model = ScriptedModel(responses=[
        call('get_action_context', {'action': 'email'}, 'context'),
        call('prepare_mock_email', {'to': ['pooja@synthetic.example'], 'subject': 'Birla assignment',
                                    'body': 'Please lead Birla onboarding.'}, 'prepare'),
        AIMessage(content='The draft is prepared.'),
        call('memory_read', {'query': 'Who was asked to lead Birla?'}, 'read'),
        AIMessage(content='Pooja was asked to lead Birla onboarding.'),
        call('get_action_context', {'action': 'email'}, 'context-2'),
        call('prepare_mock_email', {'to': ['pooja@synthetic.example'], 'subject': 'Rejected assignment',
                                    'body': 'Please lead the rejected assignment.'}, 'prepare-2'),
        AIMessage(content='The draft is prepared.'),
    ])
    checkpointer = SqliteSaver(sqlite3.connect(database, check_same_thread=False))
    agent = MemoryAgent(memory, model, checkpointer, sources, '2026-05-18T09:30:00+05:30', 'rohan@kestrelworks.example')
    def decide(prompt, payload, response_type):
        if response_type is ChatChangeDecision:
            source = next(item for item in payload.chat_sources if 'Ask Pooja' in item['text'])
            return ChatChangeDecision(action='create', text='Pooja was asked to lead Birla onboarding.',
                                      ids=[], question='', source_message_id=source['message_id'])
        if response_type is ReadDecision:
            return ReadDecision(ids=[item.id for item in payload.memories])
        if response_type is GroundingDecision:
            return GroundingDecision(needs_current_sources=False, supported_by_sources=True,
                                     references=[SourceReferenceSelection(kind='chat', source_id='m001')])
        raise AssertionError(response_type)
    memory._decide = decide
    monkeypatch.setattr(lab, 'DATABASE', database)
    monkeypatch.setitem(lab.app.dependency_overrides, lab.get_agent, lambda: agent)
    return database, memory, agent


def send(client, chat_id, text):
    response = client.post('/api/chat/stream', json={'chat_id': chat_id, 'text': text})
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert not any(event['kind'] == 'error' for event in events), events
    return events


def test_assignment_yes_new_chat_citation_and_no_cancellation(tmp_path, monkeypatch, caplog):
    database, memory, agent = build_system(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO, logger='uvicorn.error')
    with TestClient(lab.app, base_url='http://localhost') as client:
        first = client.post('/api/chats').json()['id']
        events = send(client, first, 'Ask Pooja to lead Birla onboarding by email.')
        original = events[0]['message_id']
        assert original == 'm001'
        assert 'nothing has been sent' in events[-1]['data']
        assert memory.store.snapshot() == []
        assert agent.actions.pending() is not None
        confirmed = send(client, first, 'yes')
        assert 'Confirmed mock email' in confirmed[-1]['data']
        assert memory.store.snapshot()[0].text.endswith(f'[chat:{original}]')
        second = client.post('/api/chats').json()['id']
        answer = send(client, second, 'Who was asked to lead Birla?')[-1]
        assert answer['references'] == [{'kind': 'chat', 'source_id': original,
                                         'label': 'Message · #m001', 'chat_id': first}]
        assert client.get(f'/api/chats/{first}').json()['events'][0]['message_id'] == original
        third = client.post('/api/chats').json()['id']
        send(client, third, 'Prepare a mock email about a different assignment.')
        cancelled = send(client, third, 'no')
        assert 'Cancelled' in cancelled[-1]['data']
        assert len(memory.store.snapshot()) == 1
        with closing(sqlite3.connect(database)) as conn:
            assert conn.execute("SELECT count(*) FROM mail_threads WHERE subject='Birla assignment'").fetchone()[0] == 1
            assert conn.execute("SELECT count(*) FROM mail_threads WHERE subject='Rejected assignment'").fetchone()[0] == 0
            assert conn.execute('SELECT count(*) FROM mock_pending_actions').fetchone()[0] == 0
            assert conn.execute('SELECT memory_id,message_row_id FROM memory_chat_sources').fetchall() == [(1, 1)]
    assert 'memory.confirmed_action_review' in caplog.text
    assert 'mock.cancel' in caplog.text


def test_direct_assignment_memory_cites_user_statement(tmp_path, monkeypatch):
    database, memory, agent = build_system(tmp_path, monkeypatch)
    model = ScriptedModel(responses=[
        call('memory_change', {'query': 'Birla onboarding assignment'}, 'save'),
        AIMessage(content='Pooja leads Birla onboarding.'),
    ])
    agent = MemoryAgent(memory, model, SqliteSaver(sqlite3.connect(database, check_same_thread=False)),
                        agent.sources, '2026-05-18T09:30:00+05:30', 'rohan@kestrelworks.example')
    monkeypatch.setitem(lab.app.dependency_overrides, lab.get_agent, lambda: agent)
    previous_decide = memory._decide
    def decide(prompt, payload, response_type):
        if response_type is ChatChangeDecision:
            return ChatChangeDecision(action='create', text='Pooja leads Birla onboarding.', ids=[],
                                      question='', source_message_id=payload.chat_sources[0]['message_id'])
        return previous_decide(prompt, payload, response_type)
    memory._decide = decide
    with TestClient(lab.app, base_url='http://localhost') as client:
        chat_id = client.post('/api/chats').json()['id']
        send(client, chat_id, 'Pooja leads Birla onboarding.')
    assert memory.store.snapshot()[0].text.endswith('[chat:m001]')
    with closing(sqlite3.connect(database)) as conn:
        assert conn.execute('SELECT memory_id,message_row_id FROM memory_chat_sources').fetchall() == [(1, 1)]


def test_source_validation_rejects_invented_and_non_user_messages(tmp_path):
    database = tmp_path / 'lab.sqlite3'
    chats = ChatStore(database)
    chat = chats.create()
    assistant_id = chats.append(chat.id, {'kind': 'answer', 'data': 'Pooja leads Birla.'})
    assert chats.message(assistant_id) is None
    raw = json.dumps({'tool': 'memory_read', 'result': json.dumps({'current_sources': []})})
    assert validate_source_references([SourceReferenceSelection(kind='chat', source_id='m999')], [raw]) == []


def test_confirmation_rejects_unrelated_or_changed_source_without_writing_memory(tmp_path):
    database = tmp_path / 'guard.sqlite3'
    memory = MemoryLayer(database, None, 'synthetic')
    chats = ChatStore(database)
    chat = chats.create()
    original = chats.append(chat.id, {'kind': 'user', 'data': 'Ask Pooja to lead Birla.'})
    confirmation = chats.append(chat.id, {'kind': 'user', 'data': 'yes'})
    later = chats.append(chat.id, {'kind': 'user', 'data': 'Another request.'})
    sources = chats.user_messages(chat.id, confirmation)
    assert [source['message_id'] for source in sources] == [original]
    memory.index.search = lambda query, memories: memories
    def decision_for(source_id):
        return lambda prompt, payload, response_type: response_type(
            action='create', text='Pooja was asked to lead Birla.', ids=[], question='', source_message_id=source_id)
    memory._decide = decision_for(later)
    with pytest.raises(ValueError, match='original user message'):
        memory.change('Review confirmation', context_mode='messages', context=[], chat_sources=sources)
    memory._decide = decision_for(original)
    with closing(sqlite3.connect(database)) as conn, conn:
        conn.execute('UPDATE lab_chat_events SET event=? WHERE id=1',
                     (json.dumps({'kind': 'user', 'data': 'changed'}),))
    with pytest.raises(RuntimeError, match='Source changed'):
        memory.change('Review confirmation', context_mode='messages', context=[], chat_sources=sources)
    assert memory.store.snapshot() == []
