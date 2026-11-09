import json
import sqlite3
from pathlib import Path

import pytest

from memory_layer.mock_actions import MockActions, MockEmail, MockSlackPost
from memory_layer.mock_sources import MockSources

SOURCE_DIR = Path(__file__).resolve().parent.parent / 'data' / 'sources'


def test_normalized_tables_separate_slack_people_and_enforce_relationships(tmp_path):
    sources = MockSources(tmp_path / 'normalized.sqlite3')
    sources.seed()
    with sources.connect() as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {'contacts', 'slack_people', 'slack_channels', 'slack_members', 'slack_messages',
                'mail_threads', 'mail_messages', 'mail_recipients', 'calendar_people', 'calendar_events'} <= tables
        assert conn.execute('SELECT count(*) FROM contacts').fetchone()[0] == 9
        assert conn.execute('SELECT count(*) FROM slack_people').fetchone()[0] == 5
        assert conn.execute('SELECT count(*) FROM slack_members').fetchone()[0] == 13
        assert conn.execute("SELECT count(*) FROM slack_people WHERE id = 'agent'").fetchone()[0] == 1
        assert conn.execute('PRAGMA foreign_key_list(slack_people)').fetchall() == []
        assert conn.execute('PRAGMA foreign_key_check').fetchall() == []
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO slack_messages(id, channel_id, author_id, ts, text) VALUES (?, ?, ?, ?, ?)",
                         ('invalid', '#missing', 'agent', '2026-01-01', 'invalid'))


def test_confirmed_actions_are_written_to_separate_tables(tmp_path):
    sources = MockSources(tmp_path / 'normalized.sqlite3')
    sources.seed()
    actions = MockActions(sources, '2026-05-18T09:30:00+05:30', 'rohan@kestrelworks.example')
    actions.prepare_slack(MockSlackPost(channel='#ops', text='new message'))
    actions.confirm()
    actions.prepare_email(MockEmail(to=['maya.rao@acme-industries.example'], subject='new subject', body='hello'))
    actions.confirm()
    with sources.connect() as conn:
        assert conn.execute("SELECT count(*) FROM slack_messages WHERE text = 'new message' AND channel_id = '#ops' AND author_id = 'rohan'").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM mail_messages JOIN mail_threads ON mail_threads.id = mail_messages.thread_id WHERE mail_threads.subject = 'new subject'").fetchone()[0] == 1
        assert conn.execute('PRAGMA foreign_key_check').fetchall() == []


def test_existing_legacy_actions_are_migrated_once(tmp_path):
    database = tmp_path / 'legacy.sqlite3'
    slack = json.loads((SOURCE_DIR / 'slack.json').read_text())
    calendar = json.loads((SOURCE_DIR / 'calendar.json').read_text())
    mail = json.loads((SOURCE_DIR / 'mail.json').read_text())
    with sqlite3.connect(database) as conn:
        conn.execute('CREATE TABLE mock_source_records (kind TEXT, id TEXT, payload TEXT, PRIMARY KEY(kind, id))')
        for channel in slack['channels']:
            conn.execute('INSERT INTO mock_source_records VALUES (?, ?, ?)', ('slack_channels', channel['name'], json.dumps(channel)))
        conn.execute('INSERT INTO mock_source_records VALUES (?, ?, ?)',
                     ('slack_messages', 'mock:previous', json.dumps({'channel': '#ops', 'author': 'rohan',
                      'ts': '2026-05-18T09:30:00+05:30', 'text': 'Previous mock action'})))
        conn.execute('INSERT INTO mock_source_records VALUES (?, ?, ?)', ('mail', mail[0]['thread_id'], json.dumps(mail[0])))
        conn.execute('INSERT INTO mock_source_records VALUES (?, ?, ?)', ('calendar_people', 'rohan', json.dumps(calendar['people']['rohan'])))
        event = {'person': 'rohan', 'title': 'Previous booking', 'start': '2026-05-25T11:00:00+05:30',
                 'end': '2026-05-25T11:30:00+05:30'}
        conn.execute('INSERT INTO mock_source_records VALUES (?, ?, ?)', ('calendar_events', 'mock:booking:rohan', json.dumps(event)))
        conn.execute('CREATE TABLE memories (id INTEGER PRIMARY KEY, text TEXT)')
        conn.execute('INSERT INTO memories(text) VALUES (?)', ('Keep this memory',))
    sources = MockSources(database)
    sources.seed()
    MockSources(database).seed()
    assert len(sources.slack('Previous mock action')) == 1
    assert len(sources.mail_thread(mail[0]['thread_id']).messages) == len(mail[0]['messages'])
    assert any(event.title == 'Previous booking' for event in sources.calendar('rohan', '2026-05-25', '2026-05-25'))
    with sources.connect() as conn:
        assert conn.execute('SELECT text FROM memories').fetchone()[0] == 'Keep this memory'
        assert conn.execute("SELECT count(*) FROM source_migrations WHERE name='relational_sources_v1'").fetchone()[0] == 1
        assert conn.execute('PRAGMA foreign_key_check').fetchall() == []
