import json

import pytest

from memory_layer import source_db
from memory_layer.mock_sources import MailMessage, MailThread, MockSources
from memory_layer.source_query import source_query_tool
from memory_layer.source_tools import source_tools
from memory_layer.source_references import SourceReferenceSelection, validate_source_references


def setup_mail_sources(database):
    sources = MockSources(database)
    with sources.connect() as conn:
        source_db.insert_thread(conn, MailThread(thread_id='incoming', subject='Incoming', messages=[
            MailMessage(id='m-in', **{'from': 'qa@example.example'}, to=['rohan@example.example'], cc=[],
                        ts='2026-05-18T09:00:00+05:30', unread=False, body='Incoming review')]))
        source_db.insert_thread(conn, MailThread(thread_id='outgoing', subject='Outgoing', messages=[
            MailMessage(id='m-out', **{'from': 'rohan@example.example'}, to=['qa@example.example'], cc=[],
                        ts='2026-05-18T10:00:00+05:30', unread=False, body='Outgoing review')]))
        source_db.insert_thread(conn, MailThread(thread_id='copied', subject='Copied', messages=[
            MailMessage(id='m-cc', **{'from': 'rohan@example.example'}, to=['other@example.example'], cc=['qa@example.example'],
                        ts='2026-05-18T11:00:00+05:30', unread=False, body='Copied review')]))
    return sources


def test_directional_mail_search_matches_only_requested_party(tmp_path):
    sources = setup_mail_sources(tmp_path / 'mail.db')
    assert [message.id for message in sources.mail_by_address('from', 'qa@example.example', 5)] == ['m-in']
    assert [message.id for message in sources.mail_by_address('to', 'qa@example.example', 5)] == ['m-out']
    assert [message.id for message in sources.mail_by_address('cc', 'qa@example.example', 5)] == ['m-cc']


def test_directional_mail_tool_rejects_wrong_direction_and_sql_bypass(tmp_path):
    sources = setup_mail_sources(tmp_path / 'mail.db')
    tools = source_tools(sources, lambda: 'run', lambda: 'List emails sent to qa@example.example')
    search = next(item for item in tools if item.name == 'search_mail_by_address')
    wrong = search.invoke({'direction': 'from', 'address': 'qa@example.example', 'limit': 5})
    assert 'direction=to' in wrong
    result = json.loads(search.invoke({'direction': 'to', 'address': 'qa@example.example', 'limit': 5}))
    assert result['direction'] == 'to'
    assert result['matched_address'] == 'qa@example.example'
    assert [item['id'] for item in result['items']] == ['m-out']
    raw = json.dumps({'tool': 'search_mail_by_address', 'arguments': {'direction': 'to'},
                      'result': json.dumps(result)})
    references = validate_source_references([SourceReferenceSelection(kind='mail', source_id='m-out')], [raw])
    assert references[0].source_id == 'm-out'
    content = next(item for item in tools if item.name == 'search_mail')
    rejected = content.invoke({'query': 'qa@example.example'})
    assert 'search_mail_by_address' in rejected
    sql = source_query_tool(sources, lambda: 'run', lambda: 'List emails sent to qa@example.example')
    blocked = sql.invoke({'sql': 'SELECT id FROM mail_messages WHERE lower(sender)=lower(?)',
                          'parameters': ['qa@example.example']})
    assert 'search_mail_by_address' in blocked
