import sqlite3

import pytest

from memory_layer.chat_store import ChatStore
from memory_layer import history_replay
from memory_layer.history_replay import _require_existing_run, replay
from memory_layer.service import ChatChangeDecision, MemoryLayer


def test_replay_reuses_appended_event_after_decision_provider_failure(tmp_path):
    database = tmp_path / 'history.sqlite3'
    memory = MemoryLayer(database, None, 'test')
    memory.index.search = lambda query, memories: memories
    decisions = iter([RuntimeError('synthetic interruption'), None])

    def decide(*args):
        decision = next(decisions)
        if isinstance(decision, Exception):
            raise decision
        payload = args[1]
        source_id = payload.chat_sources[0]['message_id']
        return ChatChangeDecision(action='create', text='Prefers tea', ids=[], question='',
                                  source_message_id=source_id)

    memory._decide = decide
    rows = [{'id': 'h1', 'ts': '2026-04-18T10:00:00+05:30', 'channel': 'web', 'role': 'user', 'text': 'I prefer tea.'}]

    with pytest.raises(RuntimeError, match='synthetic interruption'):
        replay(memory, rows)
    report = replay(memory, rows)

    chats = ChatStore(database).list_chats()
    assert report.created == 1
    assert len(chats) == 1
    assert len(ChatStore(database).events(chats[0].id)) == 1


def test_replay_does_not_duplicate_memory_when_interrupted_after_commit(tmp_path):
    database = tmp_path / 'history.sqlite3'
    memory = MemoryLayer(database, None, 'test')
    memory.index.search = lambda query, memories: memories
    decision_calls = 0

    def decide(*args):
        nonlocal decision_calls
        decision_calls += 1
        source_id = args[1].chat_sources[0]['message_id']
        return ChatChangeDecision(action='create', text='Prefers tea', ids=[], question='',
                                  source_message_id=source_id)

    memory._decide = decide
    original_change = memory.change
    interrupted = True

    def fail_after_committed_change(*args, **kwargs):
        nonlocal interrupted
        result = original_change(*args, **kwargs)
        if interrupted:
            interrupted = False
            raise RuntimeError('synthetic interruption after commit')
        return result

    memory.change = fail_after_committed_change
    rows = [{'id': 'h1', 'ts': '2026-04-18T10:00:00+05:30', 'channel': 'web', 'role': 'user', 'text': 'I prefer tea.'}]
    with pytest.raises(RuntimeError, match='after commit'):
        replay(memory, rows)
    memory.change = original_change

    report = replay(memory, rows)

    chats = ChatStore(database).list_chats()
    assert report.created == 1
    assert decision_calls == 1
    assert [item.text for item in memory.store.snapshot()] == ['Prefers tea [chat:m001]']
    assert len(chats) == 1
    assert len(ChatStore(database).events(chats[0].id)) == 1


def test_memory_write_and_replay_checkpoint_rollback_together_on_checkpoint_failure(tmp_path):
    database = tmp_path / 'history.sqlite3'
    memory = MemoryLayer(database, None, 'test')
    memory.index.search = lambda query, memories: memories
    memory._decide = lambda *args: ChatChangeDecision(action='create', text='Prefers tea', ids=[], question='',
                                                       source_message_id=args[1].chat_sources[0]['message_id'])
    rows = [{'id': 'h1', 'ts': '2026-04-18T10:00:00+05:30', 'channel': 'web', 'role': 'user', 'text': 'I prefer tea.'}]

    def fail_checkpoint(conn, result):
        conn.execute('INSERT INTO history_replay_decisions(input_hash,history_id,status,outcome) '
                     'VALUES (?,?,?,?)', ('missing-run', 'h1', result.status, result.model_dump_json()))

    # Simulate a checkpoint failure inside the memory-write transaction.
    original_change = memory.change

    def broken_change(*args, **kwargs):
        kwargs['on_commit'] = fail_checkpoint
        return original_change(*args, **kwargs)

    memory.change = broken_change
    with pytest.raises(sqlite3.IntegrityError):
        replay(memory, rows)

    assert memory.store.snapshot() == []
    with memory.store.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM memory_history').fetchone()[0] == 0


def test_replay_does_not_recreate_a_fact_after_historical_forget(tmp_path):
    db = tmp_path / 'history.sqlite3'
    memory = MemoryLayer(db, None, 'test')
    memory.index.search = lambda query, memories: memories
    memory.store.apply(memory.store.snapshot(), 'create', 'Prefers coffee', [])
    forgotten = memory.store.snapshot()[0]
    memory.store.forget(forgotten.id, forgotten.text)
    decisions = iter([
        ChatChangeDecision(action='create', text='Prefers coffee', ids=[], question='', source_message_id='m001'),
        ChatChangeDecision(action='create', text='Prefers coffee', ids=[], question='', source_message_id='m003'),
    ])
    memory._decide = lambda *args: next(decisions)
    rows = [
        {'id': 'h1', 'ts': '2026-04-18T10:00:00+05:30', 'channel': 'web', 'role': 'user', 'text': 'I prefer coffee.'},
        {'id': 'h2', 'ts': '2026-04-18T10:01:00+05:30', 'channel': 'web', 'role': 'assistant', 'text': 'Noted.'},
        {'id': 'h3', 'ts': '2026-04-18T10:02:00+05:30', 'channel': 'web', 'role': 'user', 'text': 'I like coffee.'},
    ]

    report = replay(memory, rows)

    assert report.created == 0 and report.updated == 0 and report.other == 2
    assert memory.store.snapshot() == []
    assert memory.store.is_forgotten('Prefers coffee')


def test_replay_applies_historical_forget_before_later_paraphrase(tmp_path):
    db = tmp_path / 'history.sqlite3'
    memory = MemoryLayer(db, None, 'test')
    memory.index.search = lambda query, memories: memories
    decisions = iter([
        ChatChangeDecision(action='create', text='Prefers coffee', ids=[], question='', source_message_id='m001'),
        ChatChangeDecision(action='delete', text='', ids=['1'], question='', source_message_id='m003'),
        ChatChangeDecision(action='create', text='Prefers coffee', ids=[], question='', source_message_id='m005'),
    ])
    memory._decide = lambda *args: next(decisions)
    rows = [
        {'id': 'h1', 'ts': '2026-04-18T10:00:00+05:30', 'channel': 'web', 'role': 'user', 'text': 'I prefer coffee.'},
        {'id': 'h2', 'ts': '2026-04-18T10:01:00+05:30', 'channel': 'web', 'role': 'assistant', 'text': 'Noted.'},
        {'id': 'h3', 'ts': '2026-04-18T10:02:00+05:30', 'channel': 'web', 'role': 'user', 'text': 'Forget my coffee preference.'},
        {'id': 'h4', 'ts': '2026-04-18T10:03:00+05:30', 'channel': 'web', 'role': 'assistant', 'text': 'Forgot it.'},
        {'id': 'h5', 'ts': '2026-04-18T10:04:00+05:30', 'channel': 'web', 'role': 'user', 'text': 'I like coffee.'},
    ]

    report = replay(memory, rows)
    resumed = replay(memory, rows, require_existing=True)

    assert report.created == resumed.created == 1
    assert report.updated == resumed.updated == 0
    assert report.other == resumed.other == 2
    assert memory.store.snapshot() == []
    assert memory.store.is_forgotten('Prefers coffee')


def test_resume_preflight_does_not_modify_an_unrelated_database(tmp_path):
    database = tmp_path / 'unrelated.sqlite3'
    with sqlite3.connect(database) as conn:
        conn.execute('CREATE TABLE preserved(value TEXT)')

    with pytest.raises(ValueError, match='No matching replay run'):
        _require_existing_run(database, 'not-a-run')

    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='preserved'").fetchone()
        assert not conn.execute("SELECT name FROM sqlite_master WHERE name='history_replay_runs'").fetchone()


def test_replay_rejects_timezone_naive_timestamps(tmp_path):
    memory = MemoryLayer(tmp_path / 'history.sqlite3', None, 'test')

    with pytest.raises(ValueError, match='must include a timezone'):
        replay(memory, [{'id': 'h1', 'ts': '2026-04-18T10:00:00', 'channel': 'web',
                        'role': 'user', 'text': 'I prefer tea.'}])


def test_resume_rejects_a_different_message_input(tmp_path):
    database = tmp_path / 'history.sqlite3'
    memory = MemoryLayer(database, None, 'test')
    memory.index.search = lambda query, memories: memories
    memory._decide = lambda *args: ChatChangeDecision(action='none', text='', ids=[], question='',
                                                       source_message_id=args[1].chat_sources[0]['message_id'])
    rows = [{'id': 'h1', 'ts': '2026-04-18T10:00:00+05:30', 'channel': 'web', 'role': 'user', 'text': 'No durable fact.'}]
    replay(memory, rows)
    changed_rows = [{**rows[0], 'text': 'A changed input.'}]

    with pytest.raises(ValueError, match='No matching replay run'):
        replay(memory, changed_rows, require_existing=True)
    changed_model = MemoryLayer(database, None, 'different-model')
    with pytest.raises(ValueError, match='No matching replay run'):
        replay(changed_model, rows, require_existing=True)


def test_resume_rejects_a_changed_decision_prompt(tmp_path, monkeypatch):
    memory = MemoryLayer(tmp_path / 'history.sqlite3', None, 'test')
    memory.index.search = lambda query, memories: memories
    memory._decide = lambda *args: ChatChangeDecision(action='none', text='', ids=[], question='',
                                                       source_message_id=args[1].chat_sources[0]['message_id'])
    rows = [{'id': 'h1', 'ts': '2026-04-18T10:00:00+05:30', 'channel': 'web', 'role': 'user', 'text': 'No durable fact.'}]
    replay(memory, rows)
    monkeypatch.setattr(history_replay, 'CHANGE_PROMPT', history_replay.CHANGE_PROMPT + ' changed')

    with pytest.raises(ValueError, match='No matching replay run'):
        replay(memory, rows, require_existing=True)


def test_invalid_memory_target_is_checkpointed_as_rejected(tmp_path):
    database = tmp_path / 'history.sqlite3'
    memory = MemoryLayer(database, None, 'test')
    memory.store.apply(memory.store.snapshot(), 'create', 'Prefers coffee', [])
    memory.index.search = lambda query, memories: memories
    calls = 0

    def decide(*args):
        nonlocal calls
        calls += 1
        source_id = args[1].chat_sources[0]['message_id']
        return ChatChangeDecision(action='update', text='Prefers tea', ids=['999'], question='',
                                  source_message_id=source_id)

    memory._decide = decide
    rows = [{'id': 'h1', 'ts': '2026-04-18T10:00:00+05:30', 'channel': 'web', 'role': 'user', 'text': 'Change coffee to tea.'}]

    first = replay(memory, rows)
    second = replay(memory, rows, require_existing=True)

    assert first.rejected == second.rejected == 1
    assert calls == 1
    assert [item.text for item in memory.store.snapshot()] == ['Prefers coffee']


def test_assistant_only_claims_do_not_trigger_memory_decisions(tmp_path):
    memory = MemoryLayer(tmp_path / 'history.sqlite3', None, 'test')
    memory._decide = lambda *args: (_ for _ in ()).throw(AssertionError('assistant messages are not facts'))

    report = replay(memory, [{'id': 'h1', 'ts': '2026-04-18T10:00:00+05:30', 'channel': 'web',
                              'role': 'assistant', 'text': 'You prefer coffee.'}])

    assert report.messages == 1
    assert report.created == 0
    assert memory.store.snapshot() == []


def test_replay_uses_original_assistant_context_and_records_user_citations(tmp_path):
    db = tmp_path / 'history.sqlite3'
    memory = MemoryLayer(db, None, 'test')
    memory.index.search = lambda query, memories: memories
    calls = []
    def decide(prompt, payload, response_type):
        assert 'preserve every supported unrelated detail' in prompt
        calls.append(payload)
        source = payload.chat_sources[0]['message_id']
        if len(calls) == 1:
            return ChatChangeDecision(action='create', text='The morning brief is at 9:30; check calendar first and Slack second.',
                                       ids=[], question='', source_message_id=source)
        assert payload.context[-1].content == 'The brief is at 9:30.'
        return ChatChangeDecision(action='update',
                                  text='The morning brief is at 8:30; check calendar first and Slack second.',
                                  ids=['1'], question='', source_message_id=source)
    memory._decide = decide
    rows = [
        {'id': 'm1', 'ts': '2026-04-18T10:00:00+05:30', 'channel': 'web', 'role': 'user', 'text': 'Brief at 9:30.'},
        {'id': 'm2', 'ts': '2026-04-18T10:01:00+05:30', 'channel': 'web', 'role': 'assistant', 'text': 'The brief is at 9:30.'},
        {'id': 'm3', 'ts': '2026-04-18T10:02:00+05:30', 'channel': 'web', 'role': 'user', 'text': 'Change that to 8:30.'},
    ]
    report = replay(memory, rows)
    assert calls[0].historical_time.isoformat() == '2026-04-18T10:00:00+05:30'
    assert calls[1].historical_time.isoformat() == '2026-04-18T10:02:00+05:30'
    assert report.created == 1 and report.updated == 1
    assert len(memory.store.snapshot()) == 1
    assert memory.store.snapshot()[0].text == (
        'The morning brief is at 8:30; check calendar first and Slack second. [chat:m003]')
    from memory_layer.chat_store import ChatStore
    events = ChatStore(db).events(report.chat_id)
    assert [event['history_id'] for event in events] == ['m1', 'm2', 'm3']
    assert [event['kind'] for event in events] == ['user', 'answer', 'user']
    assert events[0]['ts'] == rows[0]['ts']
