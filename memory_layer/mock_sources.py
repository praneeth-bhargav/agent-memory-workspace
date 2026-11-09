"""SQLite-backed synthetic contacts, email, calendars, and Slack."""
import logging
import re
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field, RootModel

from . import source_db, source_fts

SOURCE_DIR = Path(__file__).resolve().parent.parent / 'data' / 'sources'
CONTACTS_FILE = 'contacts.json'
MAIL_FILE = 'mail.json'
CALENDAR_FILE = 'calendar.json'
SLACK_FILE = 'slack.json'
CALENDAR_ID = 'calendar:'
SLACK_ID = 'slack:'
MISSING_THREAD = 'Mail thread not found'
INVALID_DATE_RANGE = 'Start date must not be after end date'
INVALID_SLOT_WINDOW = 'Use timezone-aware start/end within one day and a duration of 1–480 minutes.'
UNKNOWN_CALENDAR = 'All participants must have a mock calendar.'
SLOT_INTERVAL_MINUTES = 15
MAX_SLOTS = 10
MAX_SLOT_WINDOW_HOURS = 24
MAX_DURATION_MINUTES = 480
SEED_LOG = 'mock.seed contacts=%s threads=%s calendar_events=%s channels=%s slack_messages=%s'
SEED_ERROR = 'mock.seed_error file=%s error_type=%s'
PARSE_ERROR = 'mock.json_parse_error file=%s details=%s'
SourceModel = TypeVar('SourceModel', bound=BaseModel)
logger = logging.getLogger('uvicorn.error')


class Contact(BaseModel):
    id: str
    name: str
    email: str
    org: str
    role: str


class MailMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str
    sender: str = Field(alias='from')
    to: list[str]
    cc: list[str]
    ts: str
    unread: bool
    body: str


class MailThread(BaseModel):
    thread_id: str
    subject: str
    messages: list[MailMessage]


class MailAddressResult(BaseModel):
    id: str
    thread_id: str
    subject: str
    sender: str
    to: list[str]
    cc: list[str]
    ts: str
    unread: bool
    body: str


class CalendarPerson(BaseModel):
    name: str
    email: str
    timezone: str


class CalendarPersonEntry(CalendarPerson):
    key: str


class CalendarSlot(BaseModel):
    start: str
    end: str


class CalendarEvent(BaseModel):
    person: str
    title: str
    start: str
    end: str
    all_day: bool = False


class CalendarEventEntry(CalendarEvent):
    id: str


class CalendarSource(BaseModel):
    people: dict[str, CalendarPerson]
    events: list[CalendarEvent]


class SlackChannel(BaseModel):
    name: str
    members: list[str]


class SlackMessage(BaseModel):
    channel: str
    ts: str
    author: str
    text: str


class SlackMessageEntry(SlackMessage):
    id: str


class SlackSource(BaseModel):
    channels: list[SlackChannel]
    history: list[SlackMessage]


class ContactsSource(RootModel[list[Contact]]):
    pass


class MailSource(RootModel[list[MailThread]]):
    pass


class MockSources:
    def __init__(self, database: str | Path) -> None:
        self.database = database
        with closing(self.connect()) as conn:
            conn.executescript(source_db.SCHEMA)
            with conn:
                conn.execute('BEGIN IMMEDIATE')
                source_db.migrate_legacy(conn)
            source_fts.initialize(conn)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database, timeout=5)
        conn.execute('PRAGMA foreign_keys = ON')
        return conn

    @staticmethod
    def _load(filename: str) -> str:
        try:
            return (SOURCE_DIR / filename).read_text(encoding='utf-8')
        except OSError as exc:
            logger.error(SEED_ERROR, filename, type(exc).__name__)
            raise

    def _parse(self, filename: str, model: type[SourceModel]) -> SourceModel:
        try:
            return model.model_validate_json(self._load(filename))
        except ValueError as exc:
            logger.error(PARSE_ERROR, filename, str(exc))
            raise

    def seed(self) -> None:
        contacts = self._parse(CONTACTS_FILE, ContactsSource).root
        mail = self._parse(MAIL_FILE, MailSource).root
        calendar = self._parse(CALENDAR_FILE, CalendarSource)
        slack = self._parse(SLACK_FILE, SlackSource)
        with closing(self.connect()) as conn, conn:
            for contact in contacts:
                source_db.insert_contact(conn, contact)
            for thread in mail:
                source_db.insert_thread(conn, thread)
            for key, person in calendar.people.items():
                source_db.insert_person(conn, key, person)
            for index, event in enumerate(calendar.events):
                source_db.insert_event(conn, CALENDAR_ID + str(index), event)
            for channel in slack.channels:
                source_db.insert_channel(conn, channel)
            for index, message in enumerate(slack.history):
                source_db.insert_slack(conn, SLACK_ID + str(index), message)
        logger.info(SEED_LOG, len(contacts), len(mail), len(calendar.events), len(slack.channels), len(slack.history))

    @staticmethod
    def _matches(query: str, raw: str) -> bool:
        words = re.findall(r'\w+', query.casefold())
        return all(word in raw.casefold() for word in words)

    def contacts(self, query: str) -> list[Contact]:
        with closing(self.connect()) as conn:
            return [item for item in source_db.contacts(conn) if self._matches(query, item.model_dump_json())]

    def mail(self, query: str) -> list[MailThread]:
        with closing(self.connect()) as conn:
            expression = source_fts.expression(query)
            if not expression:
                ids = [row[0] for row in conn.execute('SELECT id FROM mail_threads ORDER BY id LIMIT ?',
                                                       (source_fts.MAX_RESULTS + 1,))]
            else:
                ids = [row[0] for row in conn.execute('''SELECT m.thread_id FROM mail_fts
                    JOIN mail_messages m ON m.rowid=mail_fts.rowid WHERE mail_fts MATCH ?
                    GROUP BY m.thread_id ORDER BY min(rank) LIMIT ?''', (expression, source_fts.MAX_RESULTS + 1))]
            return source_db.mail(conn, ids)

    def mail_by_address(self, direction: str, address: str, limit: int) -> list[MailAddressResult]:
        if direction not in ('from', 'to', 'cc') or not address.strip() or not 1 <= limit <= source_fts.MAX_RESULTS:
            raise ValueError('Provide direction from/to/cc, an address and limit from 1 to 50.')
        with closing(self.connect()) as conn:
            return source_db.mail_by_address(conn, direction, address.strip(), limit + 1)

    def mail_thread(self, thread_id: str) -> MailThread:
        with closing(self.connect()) as conn:
            matches = source_db.mail(conn, [thread_id])
        if not matches:
            raise ValueError(MISSING_THREAD)
        return matches[0]

    def slack(self, query: str) -> list[SlackMessageEntry]:
        with closing(self.connect()) as conn:
            expression = source_fts.expression(query)
            if not expression:
                return source_db.slack_messages(conn)[:source_fts.MAX_RESULTS + 1]
            rows = conn.execute('''SELECT m.id,m.channel_id,m.author_id,m.ts,m.text FROM slack_fts
                JOIN slack_messages m ON m.rowid=slack_fts.rowid WHERE slack_fts MATCH ?
                ORDER BY rank LIMIT ?''', (expression, source_fts.MAX_RESULTS + 1))
            return [SlackMessageEntry(id=row[0], channel=row[1], author=row[2], ts=row[3], text=row[4]) for row in rows]

    def calendar_people(self) -> list[CalendarPersonEntry]:
        with closing(self.connect()) as conn:
            return source_db.calendar_people(conn)

    def available_slots(self, people: list[str], start: str, end: str, duration_minutes: int) -> list[CalendarSlot]:
        try:
            window_start, window_end = datetime.fromisoformat(start), datetime.fromisoformat(end)
            if (window_start.utcoffset() is None or window_end.utcoffset() is None or window_end <= window_start
                    or window_end - window_start > timedelta(hours=MAX_SLOT_WINDOW_HOURS)
                    or not 1 <= duration_minutes <= MAX_DURATION_MINUTES):
                raise ValueError(INVALID_SLOT_WINDOW)
        except (ValueError, OverflowError) as exc:
            raise ValueError(INVALID_SLOT_WINDOW) from exc
        normalized_people = {person.strip().casefold() for person in people}
        known = {person.key for person in self.calendar_people()}
        if not normalized_people or not normalized_people <= known:
            raise ValueError(UNKNOWN_CALENDAR)
        with closing(self.connect()) as conn:
            events = source_db.calendar_events(conn)
        busy = [event for event in events if event.person in normalized_people]
        slots: list[CalendarSlot] = []
        current = window_start
        duration = timedelta(minutes=duration_minutes)
        while current + duration <= window_end and len(slots) < MAX_SLOTS:
            finish = current + duration
            if not any(current < datetime.fromisoformat(event.end) and finish > datetime.fromisoformat(event.start)
                       for event in busy):
                slots.append(CalendarSlot(start=current.isoformat(), end=finish.isoformat()))
            current += timedelta(minutes=SLOT_INTERVAL_MINUTES)
        return slots

    def calendar(self, person: str, start: str, end: str) -> list[CalendarEventEntry]:
        start_date, end_date = date.fromisoformat(start), date.fromisoformat(end)
        if start_date > end_date:
            raise ValueError(INVALID_DATE_RANGE)
        if person.strip().casefold() not in {entry.key for entry in self.calendar_people()}:
            raise ValueError(UNKNOWN_CALENDAR)
        with closing(self.connect()) as conn:
            events = source_db.calendar_events(conn)
        return [event for event in events if event.person.casefold() == person.strip().casefold()
                and date.fromisoformat(event.start[:10]) <= end_date
                and date.fromisoformat(event.end[:10]) >= start_date]
