"""Prepare mock actions; execution is reserved for a later human confirmation."""
import logging
import re
import time
from collections.abc import Callable
from typing import Literal

from langchain_core.tools import BaseTool, ToolException, tool
from pydantic import BaseModel

from .mock_actions import MockActions, MockBooking, MockEmail, MockSlackPost, PendingAction
from .types import Memory

TOOL_START = 'tool.start run_id=%s name=%s arguments=%s'
TOOL_END = 'tool.end run_id=%s name=%s status=awaiting_confirmation duration_ms=%.1f result=%s'
TOOL_ERROR = 'tool.error run_id=%s name=%s error_type=%s duration_ms=%.1f'
TOOL_FAILED = 'Mock tool failed. Check the backend log.'
EMAIL_TOOL = 'prepare_mock_email'
SLACK_TOOL = 'prepare_mock_slack_post'
BOOKING_TOOL = 'prepare_mock_booking'
CONTEXT_TOOL = 'get_action_context'
CONTEXT_REQUIRED = 'Call get_action_context for this action before drafting or preparing it.'
CONTEXT_FAILED = 'Action context lookup failed; do not prepare this action.'
PLACEHOLDER_ERROR = 'The email contains an unresolved name or signature placeholder. Revise it or ask the user.'
PLACEHOLDER = re.compile(r'(?:\[|<|\{)\s*(?:your\s+)?(?:name|signature)\s*(?:\]|>|\})', re.I)
Action = Literal['email', 'slack', 'calendar']
CONTEXT_QUERIES: dict[Action, str] = {
    'email': 'User identity and name, email signature and sign-off, email writing style and preferences',
    'slack': 'User identity, Slack communication style, tone, and posting preferences',
    'calendar': 'User timezone, working hours, calendar availability and meeting scheduling preferences',
}
logger = logging.getLogger('uvicorn.error')


class ActionArguments(BaseModel):
    action: Action


class ActionContextResult(BaseModel):
    action: Action
    memories: list[Memory]


class ActionContext:
    def __init__(self) -> None:
        self.completed: dict[Action, str] = {}
        self.ready: dict[Action, str] = {}

    def acknowledge(self, run: str, output: str) -> None:
        result = ActionContextResult.model_validate_json(output)
        if self.completed.get(result.action) == run:
            self.ready[result.action] = run

    def require(self, action: Action, run: str) -> None:
        if not run or self.ready.get(action) != run:
            raise ValueError(CONTEXT_REQUIRED)


def action_tools(actions: MockActions, run_id: Callable[[], str], read_memory: Callable[[str], list[Memory]],
                 current_text: Callable[[], str], context: ActionContext) -> list[BaseTool]:
    @tool
    def get_action_context(action: Action) -> str:
        """Retrieve relevant saved identity and preferences before drafting or preparing an action."""
        started = time.monotonic()
        run = run_id()
        context.ready.pop(action, None)
        context.completed.pop(action, None)
        logger.info(TOOL_START, run, CONTEXT_TOOL, ActionArguments(action=action).model_dump_json())
        try:
            results = [*read_memory(CONTEXT_QUERIES[action]), *read_memory(current_text())]
            unique = list({memory.id: memory for memory in results}.values())
            output = ActionContextResult(action=action, memories=unique).model_dump_json()
            context.completed[action] = run
            logger.info('tool.end run_id=%s name=%s status=ok count=%s duration_ms=%.1f result=%s',
                        run, CONTEXT_TOOL, len(unique), (time.monotonic() - started) * 1000, output)
            return output
        except Exception as exc:
            logger.error(TOOL_ERROR, run, CONTEXT_TOOL, type(exc).__name__, (time.monotonic() - started) * 1000)
            raise ToolException(CONTEXT_FAILED) from exc

    def prepare(name: str, action: Action, arguments: BaseModel, operation: Callable[[], PendingAction]) -> str:
        started = time.monotonic()
        logger.info(TOOL_START, run_id(), name, arguments.model_dump_json())
        try:
            context.require(action, run_id())
            if isinstance(arguments, MockEmail) and PLACEHOLDER.search(arguments.body):
                raise ValueError(PLACEHOLDER_ERROR)
            output = operation().model_dump_json()
            logger.info(TOOL_END, run_id(), name, (time.monotonic() - started) * 1000, output)
            return output
        except Exception as exc:
            logger.error(TOOL_ERROR, run_id(), name, type(exc).__name__, (time.monotonic() - started) * 1000)
            raise ToolException(str(exc) if isinstance(exc, ValueError) else TOOL_FAILED) from exc

    @tool
    def prepare_mock_email(to: list[str], subject: str, body: str) -> str:
        """Prepare a synthetic email for human confirmation, without sending it."""
        try:
            email = MockEmail(to=to, subject=subject, body=body)
        except ValueError as exc:
            logger.error(TOOL_ERROR, run_id(), EMAIL_TOOL, type(exc).__name__, 0.0)
            raise ToolException(str(exc)) from exc
        return prepare(EMAIL_TOOL, 'email', email, lambda: actions.prepare_email(email))

    @tool
    def prepare_mock_slack_post(channel: str, text: str) -> str:
        """Prepare a synthetic Slack post for human confirmation, without posting it."""
        try:
            post = MockSlackPost(channel=channel, text=text)
        except ValueError as exc:
            logger.error(TOOL_ERROR, run_id(), SLACK_TOOL, type(exc).__name__, 0.0)
            raise ToolException(str(exc)) from exc
        return prepare(SLACK_TOOL, 'slack', post, lambda: actions.prepare_slack(post))

    @tool
    def prepare_mock_booking(people: list[str], title: str, start: str, end: str) -> str:
        """Prepare a synthetic booking for human confirmation; people are calendar keys and times are ISO 8601 with timezone."""
        try:
            booking = MockBooking(people=people, title=title, start=start, end=end)
        except ValueError as exc:
            logger.error(TOOL_ERROR, run_id(), BOOKING_TOOL, type(exc).__name__, 0.0)
            raise ToolException(str(exc)) from exc
        return prepare(BOOKING_TOOL, 'calendar', booking, lambda: actions.prepare_booking(booking))

    tools = [prepare_mock_email, prepare_mock_slack_post, prepare_mock_booking, get_action_context]
    for item in tools:
        item.handle_tool_error = True
        item.handle_validation_error = True
    return tools
