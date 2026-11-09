import threading
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

from memory_layer.agent import MemoryAgent
from memory_layer.mock_actions import MockActions
from memory_layer.mock_sources import MockSources
from memory_layer.service import MemoryLayer
from memory_layer.chat_store import ChatStore


class StatefulGraph:
    def __init__(self):
        self.states = {
            'current-chat': [HumanMessage(id='old-current', content='I prefer coffee'),
                             HumanMessage(id='current-safe', content='My meeting is at noon')],
            'other-chat': [HumanMessage(id='old-other', content='I prefer coffee')],
        }

    def get_state(self, config):
        chat_id = config['configurable']['thread_id']
        return SimpleNamespace(values={'messages': self.states.get(chat_id, [])})

    def update_state(self, config, update):
        chat_id = config['configurable']['thread_id']
        self.states[chat_id] = add_messages(self.states.get(chat_id, []), update['messages'])

    def stream(self, agent_input, config, stream_mode):
        yield {'model': {'messages': [AIMessage(content='Done.')]}}


class FakeGraph:
    def get_state(self, config):
        return SimpleNamespace(values={'messages': []})

    def stream(self, agent_input, config, stream_mode):
        yield {'SummarizationMiddleware.before_model': {'messages': [AIMessage(content='', tool_calls=[{
            'name': 'old_tool', 'args': {}, 'id': 'old-call', 'type': 'tool_call'
        }])]}}
        yield {'model': {'messages': [AIMessage(content='', tool_calls=[{
            'name': 'search_contacts', 'args': {'query': 'Maya'}, 'id': 'call-1', 'type': 'tool_call'
        }])]}}
        yield {'tools': {'messages': [ToolMessage(content='{"items": []}', name='search_contacts',
                                                   tool_call_id='call-1')]}}
        yield {'model': {'messages': [AIMessage(content='No matches.')]}}


def test_agent_streams_tool_start_result_and_answer(tmp_path):
    sources = MockSources(tmp_path / 'mock.sqlite3')
    agent = object.__new__(MemoryAgent)
    agent.actions = MockActions(sources, '2026-05-18T09:30:00+05:30', 'rohan@kestrelworks.example')
    agent.lock = threading.Lock()
    agent.agent = FakeGraph()
    agent.persona_path = Path('data/persona.json')
    events = list(agent.stream_answer('Find Maya'))
    assert [event.kind for event in events] == ['tool_start', 'tool_result', 'answer']
    assert events[0].name == 'search_contacts' and 'Maya' in events[0].data
    assert events[1].tool_call_id == 'call-1' and 'items' in events[1].data
    assert events[2].data == 'No matches.'


def test_agent_prunes_forgotten_checkpoint_context_across_chats_but_keeps_chat_events(tmp_path):
    database = tmp_path / 'lab.sqlite3'
    sources = MockSources(database)
    memory = MemoryLayer(database, None, 'test')
    memory.store.apply(memory.store.snapshot(), 'create', 'Prefers coffee', [])
    saved = memory.store.snapshot()[0]
    memory.store.forget(saved.id, saved.text)
    memory._decide = lambda *args: SimpleNamespace(needs_current_sources=False,
                                                    supported_by_sources=True, references=[])
    chats = ChatStore(database)
    chat = chats.create()
    chats.append(chat.id, {'kind': 'user', 'data': 'I prefer coffee'})
    graph = StatefulGraph()
    agent = object.__new__(MemoryAgent)
    agent.actions = MockActions(sources, '2026-05-18T09:30:00+05:30', 'rohan@kestrelworks.example')
    agent.sources = sources
    agent.memory = memory
    agent.lock = threading.Lock()
    agent.agent = graph
    agent.persona_path = Path('data/persona.json')

    list(agent.stream_answer('What meetings do I have?', chat_id='current-chat'))
    list(agent.stream_answer('What meetings do I have?', chat_id='other-chat'))

    assert [message.id for message in graph.states['current-chat']] == ['current-safe']
    assert graph.states['other-chat'] == []
    assert chats.events(chat.id)[0]['data'] == 'I prefer coffee'
