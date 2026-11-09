import sqlite3

import pytest

from memory_layer.agent import AGENT_PROMPT
from memory_layer.mock_sources import MockSources
from memory_layer.source_memory import GROUNDING_PROMPT
from memory_layer.source_query import SCHEMA_CONTEXT, query_sources


def test_mail_query_guidance_distinguishes_sender_from_recipient():
    assert 'emails from' in SCHEMA_CONTEXT and 'mail_messages.sender' in SCHEMA_CONTEXT
    assert 'one row per conversation' in SCHEMA_CONTEXT
    assert 'one row per individual email' in SCHEMA_CONTEXT
    assert 'subject is stored only in mail_threads' in SCHEMA_CONTEXT
    assert 'sender and recipient' in GROUNDING_PROMPT
    assert 'do not substitute emails from' in AGENT_PROMPT
    assert 'never use generic SQL or content FTS' in AGENT_PROMPT


def test_structured_queries_are_parameterized_bounded_and_read_only(tmp_path):
    sources = MockSources(tmp_path / 'sources.db')
    sources.seed()
    result = query_sources(sources, 'SELECT id,name,email FROM contacts WHERE email=?', ['no-match\' OR 1=1 --'])
    assert result['rows'] == []
    assert len(query_sources(sources, 'SELECT id FROM calendar_events', [])['rows']) <= 50
    with pytest.raises(sqlite3.DatabaseError):
        query_sources(sources, 'DELETE FROM contacts', [])
    assert sources.contacts('maya')
