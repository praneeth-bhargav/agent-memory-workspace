"""Prepare and atomically confirm synthetic side effects in SQLite."""
import re
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self

from . import source_db
from .mock_sources import CalendarEvent, MailMessage, MailThread, MockSources, SlackMessage

PENDING_TABLE = '''CREATE TABLE IF NOT EXISTS mock_pending_actions (
    chat_id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL
)'''
SET_PENDING = 'INSERT INTO mock_pending_actions(chat_id, kind, payload) VALUES (?, ?, ?)'
GET_PENDING = 'SELECT kind, payload FROM mock_pending_actions WHERE chat_id = ?'
CLEAR_PENDING = 'DELETE FROM mock_pending_actions WHERE chat_id = ?'
LEGACY_CHAT = 'memory-lab'
SELECT_CHANNEL = 'SELECT 1 FROM slack_channels WHERE id = ?'
SELECT_PERSON = 'SELECT 1 FROM calendar_people WHERE id = ?'
EMAIL: Final = 'email'
SLACK: Final = 'slack'
BOOKING: Final = 'booking'
MOCK_PREFIX = 'mock:'
NO_PENDING = 'No mock action is awaiting confirmation.'
PENDING_EXISTS = 'Another mock action is already awaiting confirmation.'
INVALID_ADDRESS = 'Recipients must be synthetic .example email addresses.'
UNKNOWN_CHANNEL = 'Slack channel does not exist.'
UNKNOWN_PERSON = 'All participants must have a mock calendar.'
BUSY = 'A participant is busy during that time; no booking was created.'
INVALID_TIME = 'Booking requires timezone-aware start and end times, with end after start.'
ADDRESS_PATTERN = re.compile(r'^[A-Za-z0-9.!#$%&\'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.example$', re.I)


class MockEmail(BaseModel):
    to: list[str] = Field(min_length=1)
    subject: str = Field(min_length=1, max_length=500)
    body: str = Field(min_length=1, max_length=10000)

    @field_validator('to')
    @classmethod
    def synthetic_addresses(cls, addresses: list[str]) -> list[str]:
        if any(not ADDRESS_PATTERN.fullmatch(address) for address in addresses):
            raise ValueError(INVALID_ADDRESS)
        return addresses


class MockSlackPost(BaseModel):
    channel: str
    text: str = Field(min_length=1, max_length=10000)


class MockBooking(BaseModel):
    people: list[str] = Field(min_length=1)
    title: str = Field(min_length=1, max_length=500)
    start: str
    end: str

    @field_validator('people')
    @classmethod
    def normalize_people(cls, people: list[str]) -> list[str]:
        return [person.strip().casefold() for person in people]

    @model_validator(mode='after')
    def valid_times(self) -> Self:
        try:
            start, end = datetime.fromisoformat(self.start), datetime.fromisoformat(self.end)
            if start.utcoffset() is None or end.utcoffset() is None or end <= start:
                raise ValueError(INVALID_TIME)
        except (ValueError, OverflowError) as exc:
            raise ValueError(INVALID_TIME) from exc
        if len(set(self.people)) != len(self.people):
            raise ValueError(UNKNOWN_PERSON)
        return self


class PendingAction(BaseModel):
    kind: Literal['email', 'slack', 'booking']
    payload: MockEmail | MockSlackPost | MockBooking


class MockActions:
    def __init__(self, sources: MockSources, as_of: str, sender: str, chat_id: str = LEGACY_CHAT) -> None:
        self.sources = sources
        self.as_of = as_of
        self.sender = sender
        self.chat_id = chat_id
        with closing(sources.connect()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute(PENDING_TABLE)
            conn.execute('CREATE TABLE IF NOT EXISTS mock_action_migrations (name TEXT PRIMARY KEY)')
            migrated = conn.execute("SELECT 1 FROM mock_action_migrations WHERE name='per_chat_pending'").fetchone()
            legacy = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='mock_pending_action'").fetchone()
            if not migrated:
                if legacy:
                    conn.execute('INSERT INTO mock_pending_actions(chat_id,kind,payload) '
                                 'SELECT ?,kind,payload FROM mock_pending_action WHERE id=1', (LEGACY_CHAT,))
                    conn.execute('DELETE FROM mock_pending_action WHERE id=1')
                conn.execute("INSERT INTO mock_action_migrations(name) VALUES ('per_chat_pending')")

    def _prepare(self, kind: str, payload: BaseModel) -> PendingAction:
        with closing(self.sources.connect()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            if conn.execute(GET_PENDING, (self.chat_id,)).fetchone():
                raise ValueError(PENDING_EXISTS)
            conn.execute(SET_PENDING, (self.chat_id, kind, payload.model_dump_json()))
        return self.pending_required()

    def prepare_email(self, email: MockEmail) -> PendingAction:
        return self._prepare(EMAIL, email)

    def prepare_slack(self, post: MockSlackPost) -> PendingAction:
        with closing(self.sources.connect()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            if not conn.execute(SELECT_CHANNEL, (post.channel,)).fetchone():
                raise ValueError(UNKNOWN_CHANNEL)
            if conn.execute(GET_PENDING, (self.chat_id,)).fetchone():
                raise ValueError(PENDING_EXISTS)
            conn.execute(SET_PENDING, (self.chat_id, SLACK, post.model_dump_json()))
        return self.pending_required()

    def prepare_booking(self, booking: MockBooking) -> PendingAction:
        with closing(self.sources.connect()) as conn:
            self._validate_people(conn, booking)
        return self._prepare(BOOKING, booking)

    def pending(self) -> PendingAction | None:
        with closing(self.sources.connect()) as conn:
            row = conn.execute(GET_PENDING, (self.chat_id,)).fetchone()
        if not row:
            return None
        kind, raw = row
        if kind == EMAIL:
            return PendingAction(kind=EMAIL, payload=MockEmail.model_validate_json(raw))
        if kind == SLACK:
            return PendingAction(kind=SLACK, payload=MockSlackPost.model_validate_json(raw))
        return PendingAction(kind=BOOKING, payload=MockBooking.model_validate_json(raw))

    def pending_required(self) -> PendingAction:
        pending = self.pending()
        if pending is None:
            raise ValueError(NO_PENDING)
        return pending

    def cancel(self) -> None:
        with closing(self.sources.connect()) as conn, conn:
            conn.execute(CLEAR_PENDING, (self.chat_id,))

    @staticmethod
    def _validate_people(conn: sqlite3.Connection, booking: MockBooking) -> None:
        if any(not conn.execute(SELECT_PERSON, (person,)).fetchone() for person in booking.people):
            raise ValueError(UNKNOWN_PERSON)

    @staticmethod
    def _check_busy(conn: sqlite3.Connection, booking: MockBooking) -> None:
        start, end = datetime.fromisoformat(booking.start), datetime.fromisoformat(booking.end)
        for event in source_db.calendar_events(conn):
            if event.person in booking.people and start < datetime.fromisoformat(event.end) and end > datetime.fromisoformat(event.start):
                raise ValueError(BUSY)

    def confirm(self) -> PendingAction:
        with closing(self.sources.connect()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute(GET_PENDING, (self.chat_id,)).fetchone()
            if not row:
                raise ValueError(NO_PENDING)
            kind, raw = row
            identity = MOCK_PREFIX + uuid.uuid4().hex
            if kind == EMAIL:
                email = MockEmail.model_validate_json(raw)
                message = MailMessage.model_construct(id=identity, sender=self.sender, to=email.to, cc=[],
                                                      ts=self.as_of, unread=False, body=email.body)
                thread = MailThread(thread_id=identity, subject=email.subject, messages=[message])
                source_db.insert_thread(conn, thread)
                result = PendingAction(kind=EMAIL, payload=email)
            elif kind == SLACK:
                post = MockSlackPost.model_validate_json(raw)
                slack_message = SlackMessage(channel=post.channel, ts=self.as_of, author='rohan', text=post.text)
                source_db.insert_slack(conn, identity, slack_message)
                result = PendingAction(kind=SLACK, payload=post)
            elif kind == BOOKING:
                booking = MockBooking.model_validate_json(raw)
                self._validate_people(conn, booking)
                self._check_busy(conn, booking)
                for person in booking.people:
                    event = CalendarEvent(person=person, title=booking.title, start=booking.start, end=booking.end)
                    source_db.insert_event(conn, identity + ':' + person, event)
                result = PendingAction(kind=BOOKING, payload=booking)
            else:
                raise ValueError(NO_PENDING)
            conn.execute(CLEAR_PENDING, (self.chat_id,))
        return result
