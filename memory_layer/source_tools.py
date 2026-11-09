"""LangChain read tools for SQLite-backed synthetic data."""
import logging
import time
from collections.abc import Callable
from typing import Literal, TypeVar

from langchain_core.tools import BaseTool, ToolException, tool
from pydantic import BaseModel, Field

from .mail_direction import ensure_mail_direction, require_directional_tool_for_content_search
from .mock_sources import (CalendarEventEntry, CalendarPersonEntry, CalendarSlot, Contact, MailAddressResult,
                           MailThread, MockSources, SlackMessageEntry)
from .source_fts import MAX_RESULTS

TOOL_START = 'tool.start run_id=%s name=%s arguments=%s'
TOOL_END = 'tool.end run_id=%s name=%s status=ok count=%s duration_ms=%.1f result=%s'
TOOL_ERROR = 'tool.error run_id=%s name=%s error_type=%s duration_ms=%.1f'
TOOL_FAILED = 'Mock tool failed. Check the backend log.'
CONTACT_TOOL = 'search_contacts'
MAIL_TOOL = 'search_mail'
THREAD_TOOL = 'read_mail_thread'
SLACK_TOOL = 'search_slack'
CALENDAR_TOOL = 'read_calendar'
CALENDAR_PEOPLE_TOOL = 'list_calendar_people'
SLOTS_TOOL = 'find_available_slots'
MAIL_ADDRESS_TOOL = 'search_mail_by_address'
logger = logging.getLogger('uvicorn.error')
Result = TypeVar('Result', Contact, MailAddressResult, MailThread, SlackMessageEntry, CalendarEventEntry, CalendarPersonEntry, CalendarSlot)


class SearchArguments(BaseModel):
    query: str


class MailAddressArguments(BaseModel):
    direction: Literal['from', 'to', 'cc']
    address: str = Field(min_length=3, max_length=320)
    limit: int = Field(ge=1, le=MAX_RESULTS)


class ThreadArguments(BaseModel):
    thread_id: str


class CalendarArguments(BaseModel):
    person: str
    start: str
    end: str


class SlotArguments(BaseModel):
    people: list[str]
    start: str
    end: str
    duration_minutes: int


class EmptyArguments(BaseModel):
    pass


class ToolResults(BaseModel):
    items: list[Contact | MailAddressResult | MailThread | SlackMessageEntry | CalendarEventEntry | CalendarPersonEntry | CalendarSlot]
    truncated: bool = False
    direction: str | None = None
    matched_address: str | None = None


def source_tools(sources: MockSources, run_id: Callable[[], str], current_text: Callable[[], str] = lambda: '') -> list[BaseTool]:
    def run(name: str, arguments: BaseModel, operation: Callable[[], list[Result]],
            result_limit: int | None = None, direction: str | None = None, address: str | None = None) -> str:
        started = time.monotonic()
        logger.info(TOOL_START, run_id(), name, arguments.model_dump_json())
        try:
            result = operation()
            bounded = result_limit is not None or name in (MAIL_TOOL, SLACK_TOOL)
            limit = result_limit or MAX_RESULTS
            output = ToolResults(items=result[:limit] if bounded else result,
                                 truncated=bounded and len(result) > limit,
                                 direction=direction, matched_address=address).model_dump_json()
            logger.info(TOOL_END, run_id(), name, len(result), (time.monotonic() - started) * 1000, output)
            return output
        except Exception as exc:
            logger.error(TOOL_ERROR, run_id(), name, type(exc).__name__, (time.monotonic() - started) * 1000)
            raise ToolException(str(exc) if isinstance(exc, ValueError) else TOOL_FAILED) from exc

    @tool
    def search_contacts(query: str) -> str:
        """Find synthetic contacts by name, email, organization, or role. Use empty query to list all."""
        return run(CONTACT_TOOL, SearchArguments(query=query), lambda: sources.contacts(query))

    @tool
    def search_mail(query: str) -> str:
        """Search synthetic mail content with stemmed terms; use for subject/body content, not sender/recipient address direction. Returns at most 50 threads."""
        try:
            require_directional_tool_for_content_search(current_text())
            return run(MAIL_TOOL, SearchArguments(query=query), lambda: sources.mail(query))
        except ValueError as exc:
            raise ToolException(str(exc)) from exc

    @tool
    def search_mail_by_address(direction: Literal['from', 'to', 'cc'], address: str, limit: int = 10) -> str:
        """Find messages from, to, or cc an exact email address. Direction must match the user's request; use from/to/cc. Returns exact sender and recipient fields."""
        args = MailAddressArguments(direction=direction, address=address, limit=limit)
        try:
            ensure_mail_direction(current_text(), direction, address)
            return run(MAIL_ADDRESS_TOOL, args, lambda: sources.mail_by_address(direction, address, limit),
                       result_limit=limit, direction=direction, address=address)
        except ValueError as exc:
            raise ToolException(str(exc)) from exc

    @tool
    def read_mail_thread(thread_id: str) -> str:
        """Read the full synthetic mail thread using its exact thread_id."""
        return run(THREAD_TOOL, ThreadArguments(thread_id=thread_id), lambda: [sources.mail_thread(thread_id)])

    @tool
    def search_slack(query: str) -> str:
        """Search synthetic Slack message text, channel, or author with stemmed terms (all terms must match); broaden if empty. Returns at most 50 messages."""
        return run(SLACK_TOOL, SearchArguments(query=query), lambda: sources.slack(query))

    @tool
    def read_calendar(person: str, start: str, end: str) -> str:
        """Read a person's synthetic calendar. Person is calendar key; start and end are inclusive YYYY-MM-DD dates."""
        return run(CALENDAR_TOOL, CalendarArguments(person=person, start=start, end=end),
                   lambda: sources.calendar(person, start, end))

    @tool
    def list_calendar_people() -> str:
        """List available calendar owners and their exact calendar keys."""
        return run(CALENDAR_PEOPLE_TOOL, EmptyArguments(), sources.calendar_people)

    @tool
    def find_available_slots(people: list[str], start: str, end: str, duration_minutes: int) -> str:
        """Find free slots in one explicit time window (at most 24 hours); times must be ISO 8601 with timezone.

        Use the user's stated working hours and preferences. Ask if the window or preferences are unknown.
        """
        args = SlotArguments(people=people, start=start, end=end, duration_minutes=duration_minutes)
        return run(SLOTS_TOOL, args, lambda: sources.available_slots(people, start, end, duration_minutes))

    tools = [search_contacts, search_mail, search_mail_by_address, read_mail_thread, search_slack, read_calendar,
             list_calendar_people, find_available_slots]
    for item in tools:
        item.handle_tool_error = True
        item.handle_validation_error = True
    return tools
