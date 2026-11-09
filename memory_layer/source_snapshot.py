"""Read a consistent snapshot of the standalone mock-source database."""
from contextlib import closing

from pydantic import BaseModel

from . import source_db
from .mock_sources import (CalendarEventEntry, CalendarPersonEntry, Contact, MailThread, MockSources,
                           SlackChannel, SlackMessageEntry)


class SourceSnapshot(BaseModel):
    user_email: str
    as_of: str
    contacts: list[Contact]
    mail: list[MailThread]
    calendar_people: list[CalendarPersonEntry]
    calendar_events: list[CalendarEventEntry]
    slack_channels: list[SlackChannel]
    slack_messages: list[SlackMessageEntry]


def source_snapshot(sources: MockSources, user_email: str, as_of: str) -> SourceSnapshot:
    with closing(sources.connect()) as conn, conn:
        conn.execute('BEGIN')
        return SourceSnapshot(user_email=user_email, as_of=as_of, contacts=source_db.contacts(conn), mail=source_db.mail(conn),
                              calendar_people=source_db.calendar_people(conn),
                              calendar_events=[CalendarEventEntry(id=row[0], person=row[1], title=row[2],
                                                                  start=row[3], end=row[4], all_day=bool(row[5]))
                                               for row in conn.execute('SELECT id,person_id,title,start,end,all_day FROM calendar_events ORDER BY id')],
                              slack_channels=source_db.slack_channels(conn),
                              slack_messages=[SlackMessageEntry(id=row[0], channel=row[1], author=row[2], ts=row[3], text=row[4])
                                              for row in conn.execute('SELECT id,channel_id,author_id,ts,text FROM slack_messages ORDER BY id')])
