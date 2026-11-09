import json

import pytest

from memory_layer.action_tools import ActionContext, action_tools
from memory_layer.mock_actions import MockActions
from memory_layer.mock_sources import MockSources
from memory_layer.types import Memory


class MemoryLookup:
    def __init__(self):
        self.queries = []
        self.fail = False

    def read(self, query):
        self.queries.append(query)
        if self.fail:
            raise RuntimeError('provider unavailable')
        return [Memory(id='1', text='The user is Rohan and signs emails as R.')]


@pytest.fixture
def setup(tmp_path):
    sources = MockSources(tmp_path / 'mock.sqlite3')
    sources.seed()
    actions = MockActions(sources, '2026-05-18T09:30:00+05:30', 'rohan@kestrelworks.example')
    memory = MemoryLookup()
    run = ['turn-1']
    context = ActionContext()
    tools = {tool.name: tool for tool in action_tools(actions, lambda: run[0], memory.read, lambda: 'prepare the action', context)}
    return actions, memory, run, tools, context


@pytest.mark.parametrize(('kind', 'tool', 'args'), [
    ('email', 'prepare_mock_email', {'to': ['arjun@kestrelworks.example'], 'subject': 'Hi', 'body': 'Hi Arjun,\nR'}),
    ('slack', 'prepare_mock_slack_post', {'channel': '#ops', 'text': 'Hello team'}),
    ('calendar', 'prepare_mock_booking', {'people': ['rohan'], 'title': 'Meeting',
     'start': '2026-05-25T12:00:00+05:30', 'end': '2026-05-25T12:30:00+05:30'}),
])
def test_preparation_requires_same_action_context_in_current_turn(setup, kind, tool, args):
    actions, memory, run, tools, context = setup
    assert 'get_action_context' in tools[tool].invoke(args)
    assert actions.pending() is None
    output = tools['get_action_context'].invoke({'action': kind})
    result = json.loads(output)
    assert 'get_action_context' in tools[tool].invoke(args)  # Same-batch drafts cannot use an unseen result.
    context.acknowledge(run[0], output)
    assert result['action'] == kind and result['memories'][0]['text'].startswith('The user is Rohan')
    assert len(result['memories']) == 1
    run[0] = 'turn-2'
    assert 'get_action_context' in tools[tool].invoke(args)
    assert actions.pending() is None
    output = tools['get_action_context'].invoke({'action': kind})
    context.acknowledge(run[0], output)
    assert json.loads(tools[tool].invoke(args))['kind'] in ('email', 'slack', 'booking')
    assert actions.pending() is not None
    assert len(memory.queries) == 4


def test_failed_context_and_unresolved_email_placeholder_never_prepare(setup):
    actions, memory, run, tools, context = setup
    email = {'to': ['arjun@kestrelworks.example'], 'subject': 'Hi', 'body': 'Best, [Your Name]'}
    memory.fail = True
    assert 'failed' in tools['get_action_context'].invoke({'action': 'email'}).lower()
    assert 'get_action_context' in tools['prepare_mock_email'].invoke(email)
    assert actions.pending() is None
    memory.fail = False
    output = tools['get_action_context'].invoke({'action': 'email'})
    context.acknowledge(run[0], output)
    assert 'placeholder' in tools['prepare_mock_email'].invoke(email).lower()
    assert actions.pending() is None
    memory.fail = True
    assert 'failed' in tools['get_action_context'].invoke({'action': 'email'}).lower()
    assert 'get_action_context' in tools['prepare_mock_email'].invoke({**email, 'body': 'Best, R'})
    assert actions.pending() is None


def test_action_context_is_not_shared_between_action_types(setup):
    actions, _, run, tools, context = setup
    output = tools['get_action_context'].invoke({'action': 'slack'})
    context.acknowledge(run[0], output)
    response = tools['prepare_mock_email'].invoke({'to': ['arjun@kestrelworks.example'], 'subject': 'Hi', 'body': 'R'})
    assert 'get_action_context' in response
    assert actions.pending() is None
