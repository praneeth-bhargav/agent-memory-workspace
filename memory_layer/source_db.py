"""Relational storage for synthetic source records."""
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mock_sources import (CalendarEvent, CalendarEventEntry, CalendarPerson, CalendarPersonEntry, Contact,
                               MailThread, SlackChannel, SlackMessage, SlackMessageEntry)

SCHEMA = '''
CREATE TABLE IF NOT EXISTS contacts (id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL, org TEXT NOT NULL, role TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS slack_people (id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS slack_channels (id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS slack_members (channel_id TEXT NOT NULL REFERENCES slack_channels(id), person_id TEXT NOT NULL REFERENCES slack_people(id), position INTEGER NOT NULL, PRIMARY KEY(channel_id, person_id));
CREATE TABLE IF NOT EXISTS slack_messages (id TEXT PRIMARY KEY, channel_id TEXT NOT NULL REFERENCES slack_channels(id), author_id TEXT NOT NULL REFERENCES slack_people(id), ts TEXT NOT NULL, text TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS mail_threads (id TEXT PRIMARY KEY, subject TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS mail_messages (id TEXT PRIMARY KEY, thread_id TEXT NOT NULL REFERENCES mail_threads(id), sender TEXT NOT NULL, ts TEXT NOT NULL, unread INTEGER NOT NULL, body TEXT NOT NULL, position INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS mail_recipients (message_id TEXT NOT NULL REFERENCES mail_messages(id), kind TEXT NOT NULL CHECK(kind IN ('to', 'cc')), position INTEGER NOT NULL, address TEXT NOT NULL, PRIMARY KEY(message_id, kind, position));
CREATE TABLE IF NOT EXISTS calendar_people (id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL, timezone TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS calendar_events (id TEXT PRIMARY KEY, person_id TEXT NOT NULL REFERENCES calendar_people(id), title TEXT NOT NULL, start TEXT NOT NULL, end TEXT NOT NULL, all_day INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS source_migrations (name TEXT PRIMARY KEY);
'''
MIGRATION = 'relational_sources_v1'
LEGACY_EXISTS = "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'mock_source_records'"
MIGRATION_EXISTS = 'SELECT 1 FROM source_migrations WHERE name = ?'
LEGACY_RECORDS = 'SELECT id, payload FROM mock_source_records WHERE kind = ? ORDER BY id'
INSERT_MIGRATION = 'INSERT INTO source_migrations(name) VALUES (?)'


def insert_contact(conn: sqlite3.Connection, contact: 'Contact') -> None:
    conn.execute('INSERT OR IGNORE INTO contacts VALUES (?, ?, ?, ?, ?)',
                 (contact.id, contact.name, contact.email, contact.org, contact.role))


def insert_thread(conn: sqlite3.Connection, thread: 'MailThread') -> None:
    conn.execute('INSERT OR IGNORE INTO mail_threads VALUES (?, ?)', (thread.thread_id, thread.subject))
    for position, message in enumerate(thread.messages):
        conn.execute('INSERT OR IGNORE INTO mail_messages VALUES (?, ?, ?, ?, ?, ?, ?)',
                     (message.id, thread.thread_id, message.sender, message.ts, message.unread, message.body, position))
        for kind, addresses in (('to', message.to), ('cc', message.cc)):
            for index, address in enumerate(addresses):
                conn.execute('INSERT OR IGNORE INTO mail_recipients VALUES (?, ?, ?, ?)', (message.id, kind, index, address))


def insert_person(conn: sqlite3.Connection, key: str, person: 'CalendarPerson') -> None:
    conn.execute('INSERT OR IGNORE INTO calendar_people VALUES (?, ?, ?, ?)', (key, person.name, person.email, person.timezone))


def insert_event(conn: sqlite3.Connection, key: str, event: 'CalendarEvent') -> None:
    conn.execute('INSERT OR IGNORE INTO calendar_events VALUES (?, ?, ?, ?, ?, ?)',
                 (key, event.person, event.title, event.start, event.end, event.all_day))


def renamed_channel(conn: sqlite3.Connection, name: str) -> str:
    visited: set[str] = set()
    while name not in visited:
        visited.add(name)
        if conn.execute('SELECT 1 FROM slack_channels WHERE id = ?', (name,)).fetchone():
            return name
        prefix = 'renamed_slack:' + name + '->'
        row = conn.execute('SELECT name FROM source_migrations WHERE substr(name, 1, ?) = ? ORDER BY rowid DESC LIMIT 1',
                           (len(prefix), prefix)).fetchone()
        if row is None:
            return name
        name = row[0][len(prefix):]
    return name


def insert_channel(conn: sqlite3.Connection, channel: 'SlackChannel') -> None:
    prefix = 'renamed_slack:' + channel.name + '->'
    if conn.execute('SELECT 1 FROM source_migrations WHERE substr(name, 1, ?) = ?', (len(prefix), prefix)).fetchone():
        return
    inserted = conn.execute('INSERT OR IGNORE INTO slack_channels VALUES (?)', (channel.name,)).rowcount
    if not inserted:
        return
    for position, member in enumerate(channel.members):
        conn.execute('INSERT OR IGNORE INTO slack_people VALUES (?)', (member,))
        conn.execute('INSERT OR IGNORE INTO slack_members VALUES (?, ?, ?)', (channel.name, member, position))


def insert_slack(conn: sqlite3.Connection, key: str, message: 'SlackMessage') -> None:
    conn.execute('INSERT OR IGNORE INTO slack_people VALUES (?)', (message.author,))
    conn.execute('INSERT OR IGNORE INTO slack_messages VALUES (?, ?, ?, ?, ?)',
                 (key, renamed_channel(conn, message.channel), message.author, message.ts, message.text))


def migrate_legacy(conn: sqlite3.Connection) -> None:
    from .mock_sources import CalendarEvent, CalendarPerson, Contact, MailThread, SlackChannel, SlackMessage

    if conn.execute(MIGRATION_EXISTS, (MIGRATION,)).fetchone():
        return
    if conn.execute(LEGACY_EXISTS).fetchone():
        for _, raw in conn.execute(LEGACY_RECORDS, ('contacts',)).fetchall():
            insert_contact(conn, Contact.model_validate_json(raw))
        for _, raw in conn.execute(LEGACY_RECORDS, ('mail',)).fetchall():
            insert_thread(conn, MailThread.model_validate_json(raw))
        for key, raw in conn.execute(LEGACY_RECORDS, ('calendar_people',)).fetchall():
            insert_person(conn, key, CalendarPerson.model_validate_json(raw))
        for key, raw in conn.execute(LEGACY_RECORDS, ('calendar_events',)).fetchall():
            insert_event(conn, key, CalendarEvent.model_validate_json(raw))
        for _, raw in conn.execute(LEGACY_RECORDS, ('slack_channels',)).fetchall():
            insert_channel(conn, SlackChannel.model_validate_json(raw))
        for key, raw in conn.execute(LEGACY_RECORDS, ('slack_messages',)).fetchall():
            insert_slack(conn, key, SlackMessage.model_validate_json(raw))
    conn.execute(INSERT_MIGRATION, (MIGRATION,))


def contacts(conn: sqlite3.Connection) -> list['Contact']:
    from .mock_sources import Contact
    return [Contact(id=row[0], name=row[1], email=row[2], org=row[3], role=row[4])
            for row in conn.execute('SELECT id, name, email, org, role FROM contacts ORDER BY id')]


def mail(conn: sqlite3.Connection, thread_ids: list[str] | None = None) -> list['MailThread']:
    from .mock_sources import MailMessage, MailThread
    threads = []
    if thread_ids is None:
        rows = conn.execute('SELECT id, subject FROM mail_threads ORDER BY id').fetchall()
    else:
        rows = [row for thread_id in thread_ids
                if (row := conn.execute('SELECT id, subject FROM mail_threads WHERE id=?', (thread_id,)).fetchone())]
    for thread_id, subject in rows:
        messages = []
        for mid, sender, ts, unread, body in conn.execute(
                'SELECT id, sender, ts, unread, body FROM mail_messages WHERE thread_id = ? ORDER BY position', (thread_id,)).fetchall():
            to = [row[0] for row in conn.execute("SELECT address FROM mail_recipients WHERE message_id = ? AND kind = 'to' ORDER BY position", (mid,))]
            cc = [row[0] for row in conn.execute("SELECT address FROM mail_recipients WHERE message_id = ? AND kind = 'cc' ORDER BY position", (mid,))]
            messages.append(MailMessage.model_construct(id=mid, sender=sender, ts=ts, unread=bool(unread), body=body, to=to, cc=cc))
        threads.append(MailThread(thread_id=thread_id, subject=subject, messages=messages))
    return threads


def mail_by_address(conn: sqlite3.Connection, direction: str, address: str, limit: int) -> list['MailAddressResult']:
    from .mock_sources import MailAddressResult
    if direction == 'from':
        rows = conn.execute('''SELECT m.id,m.thread_id,t.subject,m.sender,m.ts,m.unread,m.body
            FROM mail_messages m JOIN mail_threads t ON t.id=m.thread_id
            WHERE lower(m.sender)=lower(?) ORDER BY datetime(m.ts) DESC,m.position DESC LIMIT ?''',
            (address, limit)).fetchall()
    else:
        rows = conn.execute('''SELECT m.id,m.thread_id,t.subject,m.sender,m.ts,m.unread,m.body
            FROM mail_messages m JOIN mail_threads t ON t.id=m.thread_id
            JOIN mail_recipients r ON r.message_id=m.id
            WHERE lower(r.address)=lower(?) AND r.kind=?
            ORDER BY datetime(m.ts) DESC,m.position DESC LIMIT ?''', (address, direction, limit)).fetchall()
    results = []
    for message_id, thread_id, subject, sender, ts, unread, body in rows:
        recipients = {kind: [] for kind in ('to', 'cc')}
        for kind, recipient in conn.execute(
                'SELECT kind,address FROM mail_recipients WHERE message_id=? ORDER BY kind,position', (message_id,)):
            recipients[kind].append(recipient)
        results.append(MailAddressResult(id=message_id, thread_id=thread_id, subject=subject, sender=sender,
            to=recipients['to'], cc=recipients['cc'], ts=ts, unread=bool(unread), body=body))
    return results


def calendar_people(conn: sqlite3.Connection) -> list['CalendarPersonEntry']:
    from .mock_sources import CalendarPersonEntry
    return [CalendarPersonEntry(key=row[0], name=row[1], email=row[2], timezone=row[3])
            for row in conn.execute('SELECT id, name, email, timezone FROM calendar_people ORDER BY id')]


def calendar_events(conn: sqlite3.Connection) -> list['CalendarEventEntry']:
    from .mock_sources import CalendarEventEntry
    return [CalendarEventEntry(id=row[0], person=row[1], title=row[2], start=row[3], end=row[4], all_day=bool(row[5]))
            for row in conn.execute('SELECT id, person_id, title, start, end, all_day FROM calendar_events ORDER BY id')]


def slack_channels(conn: sqlite3.Connection) -> list['SlackChannel']:
    from .mock_sources import SlackChannel
    channels = []
    for (name,) in conn.execute('SELECT id FROM slack_channels ORDER BY id').fetchall():
        members = [row[0] for row in conn.execute('SELECT person_id FROM slack_members WHERE channel_id = ? ORDER BY position', (name,))]
        channels.append(SlackChannel(name=name, members=members))
    return channels


def slack_messages(conn: sqlite3.Connection) -> list['SlackMessageEntry']:
    from .mock_sources import SlackMessageEntry
    return [SlackMessageEntry(id=row[0], channel=row[1], author=row[2], ts=row[3], text=row[4])
            for row in conn.execute('SELECT id, channel_id, author_id, ts, text FROM slack_messages ORDER BY id')]
