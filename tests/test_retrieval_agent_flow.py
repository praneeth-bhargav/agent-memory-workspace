import json
import logging
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import PrivateAttr

from memory_layer import lab
from memory_layer.agent import MemoryAgent
from memory_layer.mock_actions import MockSlackPost
from memory_layer.mock_sources import MailMessage, MailThread, MockSources
from memory_layer import source_db
from memory_layer.service import ChatChangeDecision, MemoryLayer
from memory_layer.source_memory import GroundingDecision, SourceSelection, SourceSelections
from memory_layer.source_references import SourceReferenceSelection
from memory_layer.types import ChangeDecision, ReadDecision


class ScriptedModel(BaseChatModel):
    _responses: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self):
        return 'scripted-local'

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._responses.pop(0))])


def call(name, args, identity):
    return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': identity, 'type': 'tool_call'}])


def test_api_questions_execute_search_memory_and_sql_with_recorded_trace(tmp_path, monkeypatch, caplog):
    database = tmp_path / 'lab.db'
    monkeypatch.setattr(lab, 'DATABASE', database)
    lab.get_sources.cache_clear()
    with TestClient(lab.app, base_url='http://localhost') as client, caplog.at_level(logging.INFO, logger='uvicorn.error'):
        assert client.post('/api/sources/slack-channels', json={'name': '#qa-oncall', 'members': ['rudra']}).status_code == 201
        created = client.post('/api/sources/slack-messages', json={'channel': '#qa-oncall', 'author': 'rudra',
            'text': 'I will handle support ticket tis week'}).json()['id']
        contact = client.post('/api/sources/contacts', json={'name': 'Rudra QA', 'email': 'rudra@qa.example',
            'org': 'QA', 'role': 'Engineer'}).json()['id']
        provider = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kwargs:
            SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])))
        memory = MemoryLayer(database, provider, 'test')
        def decide(prompt, payload, response_type):
            if response_type is GroundingDecision:
                kind, identity = ('contact', contact) if 'query_sources_sql' in str(payload.current_sources) else ('slack', created)
                return GroundingDecision(needs_current_sources=True, supported_by_sources=True,
                    references=[SourceReferenceSelection(kind=kind, source_id=identity)])
            if response_type is SourceSelections:
                return SourceSelections(sources=[SourceSelection(kind='slack', source_id=created, query='support on call May 18')])
            if response_type is ChangeDecision:
                return ChangeDecision(action='create', text='Rudra handles support 2026-05-18 through 2026-05-24', ids=[], question='')
            if response_type is ReadDecision:
                return ReadDecision(ids=['1'])
            raise AssertionError(response_type)
        memory._decide = decide
        model = ScriptedModel()
        model._responses = [call('search_slack', {'query': 'support tickets this week'}, 's1'),
                            call('search_slack', {'query': 'support tickets'}, 's2'),
                            AIMessage(content='Rudra handles support this week.'),
                            call('memory_read', {'query': 'support assignment May 18 2026'}, 'm1'),
                            AIMessage(content='Rudra handles support this week.'),
                            call('query_sources_sql', {'sql': 'SELECT id,name,email FROM contacts WHERE email=?',
                                 'parameters': ['rudra@qa.example']}, 'q1'),
                            AIMessage(content='Rudra QA: rudra@qa.example.')]
        agent = MemoryAgent(memory, model, InMemorySaver(), lab.get_sources(), '2026-05-18T09:30:00+05:30', 'rohan@kestrelworks.example')
        lab.app.dependency_overrides[lab.get_agent] = lambda: agent
        try:
            first_id = client.post('/api/chats').json()['id']
            first = client.post('/api/chat/stream', json={'text': 'Who is handling support tickets this week?', 'chat_id': first_id})
            events = [json.loads(line) for line in first.text.splitlines()]
            assert any(event.get('name') == 'search_slack' and created in event.get('data', '') for event in events)
            assert any(event.get('kind') == 'answer' and 'Rudra' in event['data'] for event in events)
            with memory.store.connect() as conn:
                assert created in conn.execute('SELECT text FROM memories WHERE active=1').fetchone()['text']
            second_id = client.post('/api/chats').json()['id']
            second = client.post('/api/chat/stream', json={'text': 'Who was on support during May 18?', 'chat_id': second_id})
            events = [json.loads(line) for line in second.text.splitlines()]
            assert any(event.get('name') == 'memory_read' and 'Rudra' in event.get('data', '') for event in events)
            third_id = client.post('/api/chats').json()['id']
            third = client.post('/api/chat/stream', json={'text': 'Which contact has email rudra@qa.example?', 'chat_id': third_id})
            events = [json.loads(line) for line in third.text.splitlines()]
            assert any(event.get('name') == 'query_sources_sql' and contact in event.get('data', '') for event in events)
            assert not any(event.get('name') == 'memory_read' for event in events)
            assert all(name in caplog.text for name in ('tool.start', 'search_slack', 'memory_read', 'query_sources_sql'))
        finally:
            lab.app.dependency_overrides.clear()
    lab.get_sources.cache_clear()


def test_wrong_mail_direction_is_rejected_and_retry_returns_only_recipients(tmp_path):
    database = tmp_path / 'mail.db'
    sources = MockSources(database)
    with sources.connect() as conn:
        source_db.insert_thread(conn, MailThread(thread_id='incoming', subject='Incoming', messages=[
            MailMessage(id='incoming-id', **{'from': 'qa@example.example'}, to=['rohan@kestrelworks.example'], cc=[],
                        ts='2026-05-18T09:00:00+05:30', unread=False, body='Incoming')]))
        source_db.insert_thread(conn, MailThread(thread_id='outgoing', subject='Outgoing', messages=[
            MailMessage(id='outgoing-id', **{'from': 'rohan@kestrelworks.example'}, to=['qa@example.example'], cc=[],
                        ts='2026-05-18T10:00:00+05:30', unread=False, body='Outgoing')]))
    provider = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kwargs:
        SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])))
    memory = MemoryLayer(database, provider, 'test')
    def decide(prompt, payload, response_type):
        if response_type is GroundingDecision:
            return GroundingDecision(needs_current_sources=True, supported_by_sources=True,
                references=[SourceReferenceSelection(kind='mail', source_id='outgoing-id')])
        if response_type is SourceSelections:
            return SourceSelections(sources=[])
        raise AssertionError(response_type)
    memory._decide = decide
    model = ScriptedModel()
    model._responses = [call('search_mail_by_address', {'direction':'from','address':'qa@example.example','limit':3}, 'wrong'),
                        call('search_mail_by_address', {'direction':'to','address':'qa@example.example','limit':3}, 'right'),
                        AIMessage(content='One email was sent to qa@example.example: Outgoing.')]
    agent = MemoryAgent(memory, model, InMemorySaver(), sources, '2026-05-18T09:30:00+05:30', 'rohan@kestrelworks.example')
    events = list(agent.stream_answer('List emails sent to qa@example.example', 'mail-direction'))
    assert any(event.kind == 'tool_result' and event.name == 'search_mail_by_address'
               and event.status == 'error' and 'direction=to' in event.data for event in events)
    success = [event for event in events if event.kind == 'tool_result' and event.name == 'search_mail_by_address'
               and event.status == 'success']
    assert len(success) == 1 and 'outgoing-id' in success[0].data and 'incoming-id' not in success[0].data
    assert 'One email was sent to' in events[-1].data


def test_confirmation_controls_mock_post_and_memory_review(tmp_path):
    from memory_layer.mock_sources import MockSources

    database = tmp_path / 'lab.db'
    sources = MockSources(database)
    sources.seed()
    provider = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kwargs:
        SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])))
    memory = MemoryLayer(database, provider, 'test')
    def decide(prompt, payload, response_type):
        if response_type is GroundingDecision:
            return GroundingDecision(needs_current_sources=False, supported_by_sources=True)
        if response_type is ChatChangeDecision:
            return ChatChangeDecision(action='create', text='Alice on call 2026-05-18 to 2026-05-24', ids=[],
                                      question='', source_message_id=payload.chat_sources[0]['message_id'])
        raise AssertionError(response_type)
    memory._decide = decide
    model = ScriptedModel()
    model._responses = [AIMessage(content='Cancelled.')]
    agent = MemoryAgent(memory, model, InMemorySaver(), sources, '2026-05-18T09:30:00+05:30', 'rohan@kestrelworks.example')
    agent.actions.chat_id = 'cancel'
    agent.actions.prepare_slack(MockSlackPost(channel='#ops', text='QA cancelled assignment'))
    assert 'Cancelled' in agent.answer('no', 'cancel').answer
    assert not sources.slack('QA cancelled assignment')
    assert memory.store.snapshot() == []
    confirm_chat = agent.chat_store.create().id
    agent.chat_store.append(confirm_chat, {'kind': 'user', 'data': 'Ask Alice to be on call 2026-05-18 to 2026-05-24'})
    agent.actions.chat_id = confirm_chat
    agent.actions.prepare_slack(MockSlackPost(channel='#ops', text='QA confirmed assignment'))
    confirmation = agent.chat_store.append(confirm_chat, {'kind': 'user', 'data': 'yes'})
    assert 'Confirmed mock slack' in agent.answer('yes', confirm_chat, confirmation).answer
    assert sources.slack('QA confirmed assignment')
    assert 'Alice on call' in memory.store.snapshot()[0].text
