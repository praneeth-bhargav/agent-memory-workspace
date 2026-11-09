from types import SimpleNamespace

import pytest

from memory_layer import MemoryLayer
from memory_layer.chat_store import ChatStore
from memory_layer.service import ChatChangeDecision
from memory_layer.types import ChangeDecision, Memory, Message, ReadDecision


class Responses:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=next(self.responses)))])


class Embeddings:
    def create(self, **kwargs):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])


def client(*responses):
    completions = Responses(*responses)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions), embeddings=Embeddings()), completions


def decision(action, text='', ids=None, question=''):
    return ChangeDecision(action=action, text=text, ids=ids or [], question=question)


def test_explicit_changes_persist_and_read_across_instances(tmp_path):
    provider, _ = client(decision('create', 'Use R as my email signature'), ReadDecision(ids=['1']))
    memory = MemoryLayer(tmp_path / 'memories.sqlite3', provider, 'test-model')
    assert memory.change('Use R as my email signature', context_mode='messages', context=[]).status == 'created'
    reloaded = MemoryLayer(tmp_path / 'memories.sqlite3', provider, 'test-model')
    assert [item.text for item in reloaded.read('email signature')] == ['Use R as my email signature']


def test_create_decision_does_not_use_persona_as_a_target(tmp_path):
    from pathlib import Path

    from memory_layer.persona import seed_persona

    provider, _ = client(decision('create', 'Prefers coffee in the morning', ['1']))
    memory = MemoryLayer(tmp_path / 'memories.sqlite3', provider, 'test-model')
    seed_persona(memory.store, Path(__file__).resolve().parents[1] / 'data' / 'persona.json')
    memory.index.search = lambda query, memories: memories

    result = memory.change('I prefer coffee in the morning', context_mode='messages', context=[])

    records = memory.store.records()
    assert result.status == 'created'
    assert [record.is_persona for record in records] == [True, False]
    assert records[1].text == 'Prefers coffee in the morning'


def test_chat_memory_discards_model_supplied_source_tags_and_uses_actual_provenance(tmp_path):
    source_text = 'I prefer coffee in the morning.'
    database = tmp_path / 'memories.sqlite3'
    chats = ChatStore(database)
    chat = chats.create()
    message_id = chats.append(chat.id, {'kind': 'user', 'data': source_text})
    response = ChatChangeDecision(action='create', text='Prefers coffee [chat:m999]', ids=[], question='',
                                  source_message_id=message_id)
    provider, _ = client(response)
    memory = MemoryLayer(database, provider, 'test-model')
    memory.index.search = lambda query, memories: memories

    source = chats.message(message_id)
    assert source is not None
    result = memory.change(source_text, context_mode='messages', context=[], chat_sources=[source])

    record = memory.store.records()[0]
    assert result.status == 'created'
    assert record.text == f'Prefers coffee [chat:{message_id}]'
    assert [(item.source_id, item.chat_id) for item in record.provenance] == [(message_id, chat.id)]


def test_update_delete_and_clarification_keep_unrelated_facts(tmp_path):
    provider, _ = client(
        decision('create', 'Use R as my signature'),
        decision('create', 'Prefer concise emails'),
        decision('update', 'Use Sam as my signature', ['1']),
        decision('clarify', question='Which preference?'),
        decision('delete', ids=['1']),
        ReadDecision(ids=['2']),
    )
    memory = MemoryLayer(tmp_path / 'memories.sqlite3', provider, 'test-model')
    memory.change('Use R as my signature', context_mode='summary', context='')
    memory.change('Remember that I prefer concise emails', context_mode='summary', context='')
    assert memory.change('Change my signature to Sam', context_mode='summary', context='').status == 'updated'
    assert memory.change('Change that', context_mode='summary', context='').status == 'clarification_needed'
    assert memory.change('Forget my signature', context_mode='summary', context='').status == 'deleted'
    assert [item.text for item in memory.read('email preferences')] == ['Prefer concise emails']
    with memory.store.connect() as conn:
        history = conn.execute('SELECT action, text FROM memory_history WHERE memory_id = 1 ORDER BY id').fetchall()
        assert [(row['action'], row['text']) for row in history] == [
            ('create', 'Use R as my signature'), ('update', 'Use Sam as my signature'), ('delete', 'Use Sam as my signature')]
        row = conn.execute('SELECT text, active FROM memories WHERE id = 1').fetchone()
        assert (row['text'], row['active']) == ('Use Sam as my signature', 0)


def test_different_dated_assignments_do_not_replace_each_other(tmp_path):
    provider, _ = client(decision('create', 'Alice on call 2026-05-18 to 2026-05-24'),
                         decision('update', 'Bob on call 2026-05-25 to 2026-05-31', ['1']))
    memory = MemoryLayer(tmp_path / 'memories.sqlite3', provider, 'test-model')
    memory.change('Alice on call May 18-24', context_mode='messages', context=[])
    assert memory.change('Bob on call May 25-31', context_mode='messages', context=[]).status == 'created'
    assert len(memory.store.snapshot()) == 2


def test_clarification_may_identify_a_candidate_without_changing_it(tmp_path):
    provider, _ = client(decision('create', 'Use R as my signature'),
                         decision('clarify', ids=['1'], question='Which name?'))
    memory = MemoryLayer(tmp_path / 'memories.sqlite3', provider, 'test-model')
    memory.change('Use R as my signature', context_mode='messages', context=[])
    result = memory.change('Change it', context_mode='messages', context=[])
    assert result.status == 'clarification_needed' and result.question == 'Which name?'
    assert memory.store.snapshot()[0].text == 'Use R as my signature'


def test_implicit_statement_does_not_change_memory(tmp_path):
    provider, _ = client(decision('none'))
    memory = MemoryLayer(tmp_path / 'memories.sqlite3', provider, 'test-model')
    result = memory.change('I am working from home today', context_mode='messages',
                           context=[Message(role='user', content='Hello')])
    assert result.status == 'no_change'
    assert memory.store.snapshot() == []


def test_memory_decision_logs_include_correlated_inputs_candidates_and_result(tmp_path, caplog):
    import logging
    provider, _ = client(decision('none'))
    memory = MemoryLayer(tmp_path / 'memories.sqlite3', provider, 'test-model')
    with memory.store.connect() as conn:
        conn.execute("INSERT INTO memories(text) VALUES ('Existing email preference')")
    memory.trace_context = lambda: ('run-example', 'chat-example')
    with caplog.at_level(logging.INFO, logger='uvicorn.error'):
        result = memory.change('End my email with Best & Regards', context_mode='messages',
                               context=[Message(role='user', content='Earlier message')], query='email sign-off')
    assert result.status == 'no_change'
    logs = caplog.text
    for value in ('run-example', 'chat-example', 'email sign-off', 'Existing email preference',
                  'Earlier message', 'End my email with Best & Regards', '"action":"none"', 'score=',
                  'purpose=query', 'purpose=memory'):
        assert value in logs
    for record in caplog.records:
        assert 'run_id=run-example' in record.message and 'chat_id=chat-example' in record.message


def test_model_failure_logs_type_without_exception_text(tmp_path, caplog):
    import logging
    provider, _ = client()
    def fail(**kwargs):
        raise ValueError('provider-secret')
    provider.chat.completions.parse = fail
    memory = MemoryLayer(tmp_path / 'memories.sqlite3', provider, 'test-model')
    memory.trace_context = lambda: ('run-error', 'chat-error')
    with caplog.at_level(logging.INFO, logger='uvicorn.error'), pytest.raises(ValueError):
        memory.change('Remember tea', context_mode='messages', context=[])
    assert 'memory.decision.error run_id=run-error chat_id=chat-error' in caplog.text
    assert 'error_type=ValueError' in caplog.text
    assert 'provider-secret' not in caplog.text


def test_invalid_model_ids_do_not_change_storage(tmp_path):
    provider, _ = client(decision('delete', ids=['999']))
    memory = MemoryLayer(tmp_path / 'memories.sqlite3', provider, 'test-model')
    with pytest.raises(ValueError):
        memory.change('Forget my signature', context_mode='messages', context=[])
    assert memory.read('signature') == []


def test_concurrent_change_is_not_overwritten(tmp_path):
    provider, _ = client(decision('create', 'Prefer tea'))
    memory = MemoryLayer(tmp_path / 'memories.sqlite3', provider, 'test-model')
    original = memory._decide
    def concurrent_change(*args):
        result = original(*args)
        with memory.store.connect() as conn:
            conn.execute("INSERT INTO memories(text) VALUES ('Prefer coffee')")
        return result
    memory._decide = concurrent_change
    with pytest.raises(RuntimeError, match='Memories changed'):
        memory.change('Remember I prefer tea', context_mode='summary', context='')
    assert memory.store.snapshot() == [Memory(id='1', text='Prefer coffee')]


def test_history_does_not_store_full_instruction(tmp_path):
    provider, _ = client(decision('create', 'Prefers tea'))
    memory = MemoryLayer(tmp_path / 'memories.sqlite3', provider, 'test-model')
    memory.change('Remember I prefer tea; unrelated private detail: SECRET', context_mode='summary', context='')
    with memory.store.connect() as conn:
        assert 'SECRET' not in repr([tuple(row) for row in conn.execute('SELECT * FROM memory_history')])


def test_memory_database_is_explicitly_rejected():
    provider, _ = client()
    with pytest.raises(ValueError, match='file'):
        MemoryLayer(':memory:', provider, 'test-model')


def test_context_mode_validates_input_before_calling_model(tmp_path):
    provider, calls = client()
    memory = MemoryLayer(tmp_path / 'memories.sqlite3', provider, 'test-model')
    with pytest.raises(ValueError):
        memory.change('Remember tea', context_mode='messages', context='not messages')
    with pytest.raises(ValueError):
        memory.change('Remember tea', context_mode='invalid', context='')
    assert calls.calls == []


def test_objects_serialize_as_json(tmp_path):
    provider, _ = client(decision('create', 'Prefers tea'))
    result = MemoryLayer(tmp_path / 'memories.sqlite3', provider, 'test-model').change(
        'Remember I prefer tea', context_mode='messages', context=[])
    assert '"status":"created"' in result.model_dump_json()
