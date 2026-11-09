import pytest

from memory_layer.action_tools import ActionContext, action_tools
from memory_layer.mock_actions import MockActions, MockBooking, MockEmail, MockSlackPost
from memory_layer.mock_sources import MockSources


def test_actions_require_confirmation_and_persist_once(tmp_path):
    sources = MockSources(tmp_path / 'mock.sqlite3')
    sources.seed()
    actions = MockActions(sources, '2026-05-18T09:30:00+05:30', 'rohan@kestrelworks.example')
    draft = MockEmail(to=['maya.rao@acme-industries.example'], subject='Hello', body='A mock message')
    actions.prepare_email(draft)
    with pytest.raises(ValueError, match='already awaiting'):
        actions.prepare_slack(MockSlackPost(channel='#ops', text='Another action'))
    assert not sources.mail('A mock message')
    actions.confirm()
    assert len(sources.mail('A mock message')) == 1
    with pytest.raises(ValueError):
        actions.confirm()
    sources.seed()
    assert len(sources.mail('A mock message')) == 1
    actions.prepare_slack(MockSlackPost(channel='#ops', text='Mock post'))
    assert not sources.slack('Mock post')
    actions.confirm()
    assert len(sources.slack('Mock post')) == 1


def test_booking_checks_conflicts_at_confirmation(tmp_path):
    sources = MockSources(tmp_path / 'mock.sqlite3')
    sources.seed()
    actions = MockActions(sources, '2026-05-18T09:30:00+05:30', 'rohan@kestrelworks.example')
    booking = MockBooking(people=['rohan'], title='Mock meeting', start='2026-05-25T10:30:00+05:30',
                          end='2026-05-25T11:30:00+05:30')
    actions.prepare_booking(booking)
    with pytest.raises(ValueError, match='busy'):
        actions.confirm()
    assert not any(e.title == 'Mock meeting' for e in sources.calendar('rohan', '2026-05-25', '2026-05-25'))
    actions.cancel()
    free = MockBooking(people=['Rohan', 'Priya'], title='Mock meeting', start='2026-05-25T12:00:00+05:30',
                       end='2026-05-25T12:30:00+05:30')
    actions.prepare_booking(free)
    actions.confirm()
    assert any(e.title == 'Mock meeting' for e in sources.calendar('rohan', '2026-05-25', '2026-05-25'))
    assert any(e.title == 'Mock meeting' for e in sources.calendar('priya', '2026-05-25', '2026-05-25'))


def test_rejects_real_addresses_and_unknown_channels(tmp_path):
    sources = MockSources(tmp_path / 'mock.sqlite3')
    sources.seed()
    actions = MockActions(sources, '2026-05-18T09:30:00+05:30', 'rohan@kestrelworks.example')
    with pytest.raises(ValueError, match='synthetic'):
        MockEmail(to=['somebody@example.com'], subject='Hello', body='Body')
    with pytest.raises(ValueError, match='does not exist'):
        actions.prepare_slack(MockSlackPost(channel='#missing', text='Hello'))
    context = ActionContext()
    tools = action_tools(actions, lambda: 'run', lambda query: [], lambda: 'post a message', context)
    output = tools[3].invoke({'action': 'slack'})
    context.acknowledge('run', output)
    result = tools[1].invoke(MockSlackPost(channel='#missing', text='Hello').model_dump())
    assert 'does not exist' in result
