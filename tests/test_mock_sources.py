import pytest

import memory_layer.mock_sources as mock_sources
from memory_layer.mock_sources import MockSources
from memory_layer.source_tools import SearchArguments, source_tools


def test_seed_and_read_sources_are_repeatable(tmp_path):
    sources = MockSources(tmp_path / 'mock.sqlite3')
    sources.seed()
    sources.seed()
    assert len(sources.contacts('maya')) == 2
    threads = sources.mail('Acme proposal')
    assert any(thread.thread_id == 'T-ACME-01' for thread in threads)
    assert sources.mail_thread('T-ACME-01').messages
    assert any(message.channel == '#ops' for message in sources.slack('#ops'))
    assert sources.calendar('rohan', '2026-05-25', '2026-05-26')
    assert any(person.key == 'rohan' for person in sources.calendar_people())
    slots = sources.available_slots(['rohan'], '2026-05-25T10:00:00+05:30',
                                    '2026-05-25T12:00:00+05:30', 30)
    assert slots[0].start == '2026-05-25T11:00:00+05:30'


def test_source_tools_return_and_log_results(tmp_path, caplog):
    sources = MockSources(tmp_path / 'mock.sqlite3')
    sources.seed()
    tools = source_tools(sources, lambda: 'example-run')
    with caplog.at_level('INFO', logger='uvicorn.error'):
        output = tools[0].invoke(SearchArguments(query='Maya').model_dump())
    assert 'Maya Rao' in output and 'Maya Iyer' in output
    assert 'example-run' in caplog.text and 'tool.start' in caplog.text and 'tool.end' in caplog.text


def test_invalid_source_json_logs_filename(tmp_path, monkeypatch, caplog):
    (tmp_path / 'contacts.json').write_text('{invalid', encoding='utf-8')
    monkeypatch.setattr(mock_sources, 'SOURCE_DIR', tmp_path)
    sources = MockSources(tmp_path / 'mock.sqlite3')
    with pytest.raises(ValueError):
        sources.seed()
    assert 'mock.json_parse_error' in caplog.text and 'contacts.json' in caplog.text
