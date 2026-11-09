from memory_layer import source_db
import json

import pytest

from memory_layer.mock_sources import MockSources, SlackChannel, SlackMessage
from memory_layer.source_tools import source_tools


def test_slack_fts_stems_and_indexes_updates(tmp_path):
    sources = MockSources(tmp_path / 'sources.db')
    with sources.connect() as conn:
        source_db.insert_channel(conn, SlackChannel(name='#support', members=['rudra']))
        source_db.insert_slack(conn, 'test:1', SlackMessage(channel='#support', author='rudra',
            ts='2026-05-18T09:00:00+05:30', text='I handle support ticket tis week'))
    assert [item.id for item in sources.slack('support tickets')] == ['test:1']
    with sources.connect() as conn:
        conn.execute("UPDATE slack_messages SET text='database maintenance' WHERE id='test:1'")
    assert sources.slack('support tickets') == []
    assert [item.id for item in sources.slack('database maintenance')] == ['test:1']


def test_search_tool_reports_truncated_results(tmp_path):
    sources = MockSources(tmp_path / 'sources.db')
    with sources.connect() as conn:
        source_db.insert_channel(conn, SlackChannel(name='#support', members=['rudra']))
        for index in range(55):
            source_db.insert_slack(conn, str(index), SlackMessage(channel='#support', author='rudra',
                ts='2026-05-18T09:00:00+05:30', text='support ticket'))
    tool = next(tool for tool in source_tools(sources, lambda: 'test') if tool.name == 'search_slack')
    result = json.loads(tool.invoke({'query': 'support tickets'}))
    assert len(result['items']) == 50
    assert result['truncated'] is True


def test_search_rejects_oversized_input_and_treats_fts_operators_as_text(tmp_path):
    sources = MockSources(tmp_path / 'sources.db')
    sources.seed()
    with pytest.raises(ValueError, match='too long'):
        sources.slack('x' * 501)
    with pytest.raises(ValueError, match='too many terms'):
        sources.mail('word ' * 21)
    assert sources.slack('" NOT *') == []


def test_existing_mail_is_searchable_with_stemming(tmp_path):
    sources = MockSources(tmp_path / 'sources.db')
    sources.seed()
    assert any(thread.thread_id == 'T-ACME-01' for thread in sources.mail('Acme proposals'))
    with sources.connect() as conn:
        conn.execute("UPDATE mail_threads SET subject='Quartz marbles' WHERE id='T-ACME-01'")
    assert any(thread.thread_id == 'T-ACME-01' for thread in sources.mail('quartz marble'))
    with sources.connect() as conn:
        conn.execute("UPDATE mail_threads SET subject='Budget forecasts' WHERE id='T-ACME-01'")
    assert any(thread.thread_id == 'T-ACME-01' for thread in sources.mail('budget forecast'))
    assert not any(thread.thread_id == 'T-ACME-01' for thread in sources.mail('quartz marble'))
    reopened = MockSources(tmp_path / 'sources.db')
    assert any(thread.thread_id == 'T-ACME-01' for thread in reopened.mail('budget forecasts'))
