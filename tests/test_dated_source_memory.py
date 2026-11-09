from types import SimpleNamespace

from memory_layer import source_db
from memory_layer.mock_sources import MockSources, SlackChannel, SlackMessage
from memory_layer.service import MemoryLayer
from memory_layer.source_memory import remember_source
from memory_layer.types import ChangeDecision, ReadDecision


def test_assignments_from_different_source_messages_do_not_overwrite(tmp_path):
    database = tmp_path / 'lab.db'
    sources = MockSources(database)
    with sources.connect() as conn:
        source_db.insert_channel(conn, SlackChannel(name='#oncall', members=['alice', 'bob']))
        for identity, author, text in [('week1', 'alice', 'On call May 18-24'), ('week2', 'bob', 'On call May 25-31')]:
            source_db.insert_slack(conn, identity, SlackMessage(channel='#oncall', author=author,
                ts='2026-05-18T09:00:00+05:30', text=text))
    client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kwargs:
        SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])))
    memory = MemoryLayer(database, client, 'test')
    decisions = iter([ChangeDecision(action='create', text='Alice is on call 2026-05-18 through 2026-05-24', ids=[], question=''),
                      ChangeDecision(action='update', text='Bob is on call 2026-05-25 through 2026-05-31', ids=['1'], question=''),
                      ReadDecision(ids=['1', '2'])])
    memory._decide = lambda *args: next(decisions)
    assert remember_source(memory, 'slack', 'week1', 'on call first week', 'Who is on call?').status == 'created'
    assert remember_source(memory, 'slack', 'week2', 'on call second week', 'Who is on call?').status == 'created'
    assert [item.text for item in memory.read('on call May 2026')] == [
        'Alice is on call 2026-05-18 through 2026-05-24 [slack:week1]',
        'Bob is on call 2026-05-25 through 2026-05-31 [slack:week2]']


def test_source_review_cannot_resurrect_forgotten_fact_without_reauthorization(tmp_path):
    database = tmp_path / 'lab.db'
    sources = MockSources(database)
    with sources.connect() as conn:
        source_db.insert_channel(conn, SlackChannel(name='#preferences', members=['alice']))
        source_db.insert_slack(conn, 'coffee', SlackMessage(channel='#preferences', author='alice',
            ts='2026-05-18T09:00:00+05:30', text='I prefer coffee in the morning'))
    memory = MemoryLayer(database, None, 'test')
    memory.index.search = lambda query, memories: memories
    decision = ChangeDecision(action='create', text='Prefers coffee in the morning', ids=[], question='')
    memory._decide = lambda *args: decision

    assert remember_source(memory, 'slack', 'coffee', 'coffee preference', 'What does this say?').status == 'created'
    saved = memory.store.snapshot()[0]
    memory.store.forget(saved.id, saved.text)

    assert remember_source(memory, 'slack', 'coffee', 'coffee preference', 'What does this say?').status == 'no_change'
    assert memory.store.snapshot() == []
    assert memory.store.is_forgotten('Prefers coffee in the morning [slack:coffee]')


def test_changed_dates_on_same_source_do_not_overwrite(tmp_path):
    database = tmp_path / 'lab.db'
    sources = MockSources(database)
    with sources.connect() as conn:
        source_db.insert_channel(conn, SlackChannel(name='#oncall', members=['alice']))
        source_db.insert_slack(conn, 'schedule', SlackMessage(channel='#oncall', author='alice',
            ts='2026-05-18T09:00:00+05:30', text='On call May 18-24'))
    client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kwargs:
        SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])))
    memory = MemoryLayer(database, client, 'test')
    decisions = iter([ChangeDecision(action='create', text='Alice on call 2026-05-18 to 2026-05-24', ids=[], question=''),
                      ChangeDecision(action='update', text='Alice on call 2026-05-25 to 2026-05-31', ids=['1'], question='')])
    memory._decide = lambda *args: next(decisions)
    remember_source(memory, 'slack', 'schedule', 'on call', 'Who is on call?')
    with sources.connect() as conn:
        conn.execute("UPDATE slack_messages SET text='On call May 25-31' WHERE id='schedule'")
    assert remember_source(memory, 'slack', 'schedule', 'on call', 'Who is on call?').status == 'created'
    assert len(memory.store.snapshot()) == 2
