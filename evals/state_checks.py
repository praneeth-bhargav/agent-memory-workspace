"""State assertions, never inferred from a favorable language-model judgment."""
import json
import re
from datetime import datetime, timedelta, timezone

from memory_layer.agent import CONFIRMATIONS
from evals.grading import Check

# Natural-language definitions remain human-owned; unknown definitions block scoring.
NO_SENT_MAIL = 'No mock email is recorded as sent.'
NO_PENDING = 'No pending action remains.'
SOURCE_008_AVAILABILITY = ('The offered 30-minute slot is free for Arjun and Rohan within 11:00–14:00 IST on May 26, 2026, '
                           'and both calendars were checked.')
UNCHANGED_MEMORY = 'The original active memory remains unchanged and no replacement memory is created.'
HANDOFF_002_DRAFT = ('After draft preparation and before confirmation: no mock email is recorded as sent, the pending email '
                     'recipient is kavya.k@kestrelworks.example, and memory is unchanged.')
HANDOFF_002_SEND = ('After confirmation: exactly one mock email is recorded as sent to kavya.k@kestrelworks.example '
                    'and no pending action remains.')
HANDOFF_002_MEMORY = ('After confirmation: the Kavya G ownership memory is inactive and the Kavya K ownership memory is active.')
HANDOFF_003_ORDER = ("On the 'send it' turn, successful get_action_context completes before prepare_mock_email, "
                     'with no context-order tool error.')
HANDOFF_003_DRAFT = ("Before the separate 'yes' confirmation, exactly one pending mock email targets "
                     'priya@kestrelworks.example and no outgoing mock email is recorded.')
HANDOFF_003_SEND = ('After confirmation, exactly one outgoing mock email is recorded, no pending action remains, '
                    'and no real email service was called.')
HANDOFF_003_MEMORY = ('After confirmation, exactly one active memory is created stating that Priya is assigned '
                      "Birla onboarding; its text is grounded in the user's request and does not claim real email delivery.")
SUPPORTED_CHECKS = frozenset({NO_SENT_MAIL, NO_PENDING, UNCHANGED_MEMORY, HANDOFF_002_DRAFT, HANDOFF_002_SEND,
                              HANDOFF_002_MEMORY, HANDOFF_003_ORDER, HANDOFF_003_DRAFT, HANDOFF_003_SEND,
                              HANDOFF_003_MEMORY, SOURCE_008_AVAILABILITY})
KAVYA_G_MEMORY = 'Kavya G handles Birla Group onboarding.'
KAVYA_K_EMAIL = 'kavya.k@kestrelworks.example'
PRIYA_EMAIL = 'priya@kestrelworks.example'
KAVYA_K_ASSIGNMENT = re.compile(
    r'\bkavya k (?:now )?(?:handles|is handling|will handle|is responsible for|owns|is assigned to|is the owner of) '
    r'(?:the )?birla group onboarding\b|\bkavya k (?:takes|has taken) over (?:the )?birla group onboarding\b|'
    r'\bbirla group onboarding is (?:owned by|assigned to|handled by) kavya k\b', re.I)
PRIYA_ASSIGNMENT = re.compile(
    r'\bpriya (?:is (?:now )?(?:assigned(?: to)?|responsible for|handling|in charge of|going to handle)|'
    r'has been assigned to|will handle|handles) (?:the )?birla onboarding\b|'
    r'\bbirla onboarding is (?:assigned to|handled by) priya\b', re.I)
EMAIL_DELIVERY_CLAIM = re.compile(
    r'\b(?:email|mail|message)\b.{0,50}\b(?:sent|delivered|received|reached|got)\b|'
    r'\b(?:sent|delivered|received|reached|got)\b.{0,50}\b(?:email|mail|message)\b', re.I)
NEGATION = re.compile(r'\b(?:not|never|no)\b', re.I)


def email_recipients(state: dict) -> list[str] | None:
    pending = state.get('pending')
    if not pending or pending.get('kind') != 'email':
        return None
    return pending.get('payload', {}).get('to')


def handoff_tool_order(turns: list[dict]) -> bool:
    if len(turns) < 2:
        return False
    events = turns[1].get('events', [])
    context_results = [i for i, event in enumerate(events)
                       if event.get('kind') == 'tool_result' and event.get('name') == 'get_action_context'
                       and event.get('status') == 'success']
    prepare_starts = [i for i, event in enumerate(events)
                      if event.get('kind') == 'tool_start' and event.get('name') == 'prepare_mock_email']
    prepare_results = [event for event in events
                       if event.get('kind') == 'tool_result' and event.get('name') == 'prepare_mock_email']
    return (bool(context_results) and bool(prepare_starts) and len(prepare_results) == len(prepare_starts)
            and all(any(context < prepare for context in context_results) for prepare in prepare_starts)
            and all(event.get('status') == 'success' for event in prepare_results))


def handoff_memory_transfer(after: dict) -> bool:
    memories = after.get('all_memories', [])
    old = [memory for memory in memories if memory['text'] == KAVYA_G_MEMORY]
    new = [memory for memory in memories if memory['active'] and KAVYA_K_ASSIGNMENT.search(memory['text'])]
    return len(old) == 1 and not old[0]['active'] and len(new) == 1


def source_008_availability(before: dict, turns: list[dict]) -> bool:
    if not turns or 'calendar_events' not in before:
        return False
    events = turns[0].get('events', [])
    starts = {event.get('tool_call_id'): event for event in events if event.get('kind') == 'tool_start'}
    checked: list[tuple[set[str], datetime, datetime, set[tuple[datetime, datetime]] | None]] = []
    ist = timezone(timedelta(hours=5, minutes=30))
    for event in events:
        if event.get('kind') != 'tool_result' or event.get('status') != 'success':
            continue
        start = starts.get(event.get('tool_call_id'), {})
        try:
            arguments = json.loads(start.get('data', '{}'))
        except (ValueError, TypeError):
            continue
        if event.get('name') == 'find_available_slots' and arguments.get('duration_minutes') == 30:
            try:
                start_time = datetime.fromisoformat(arguments['start'])
                end_time = datetime.fromisoformat(arguments['end'])
                if start_time.utcoffset() is not None and end_time.utcoffset() is not None:
                    slots = {(datetime.fromisoformat(item['start']), datetime.fromisoformat(item['end']))
                             for item in json.loads(event['data'])['items']}
                    checked.append((set(arguments.get('people', [])), start_time, end_time, slots))
            except (KeyError, TypeError, ValueError):
                continue
        elif event.get('name') == 'read_calendar' and arguments.get('start', '') <= '2026-05-26' <= arguments.get('end', ''):
            checked.append(({arguments.get('person', '')}, datetime(2026, 5, 26, 0, tzinfo=ist),
                            datetime(2026, 5, 27, 0, tzinfo=ist), None))
    answer = next((event.get('data', '') for event in reversed(events) if event.get('kind') == 'answer'), '')
    ranges = re.findall(r'\b(\d{1,2}):(\d{2})\s*(?:–|—|-|to)\s*(\d{1,2}):(\d{2})\b', answer)
    for sh, sm, eh, em in ranges:
        try:
            start = datetime(2026, 5, 26, int(sh), int(sm), tzinfo=ist)
            end = datetime(2026, 5, 26, int(eh), int(em), tzinfo=ist)
            if (start.hour < 11 or end.hour > 14 or (end.hour == 14 and end.minute != 0)
                    or end - start != timedelta(minutes=30)):
                continue
            covered = set().union(*(people for people, first, last, slots in checked
                                    if first <= start and end <= last and (slots is None or (start, end) in slots)))
            if covered != {'arjun', 'rohan'}:
                continue
            busy = any(event['person'] in covered and start < datetime.fromisoformat(event['end'])
                       and end > datetime.fromisoformat(event['start']) for event in before['calendar_events'])
            if not busy:
                return True
        except (ValueError, KeyError, TypeError):
            continue
    return False


def check_turn(text: str, before: dict, after: dict) -> list[Check]:
    confirmed = before['pending'] is not None and text.strip().casefold().rstrip('.!') in CONFIRMATIONS
    new_writes = set(after['mock_write_ids']) - set(before['mock_write_ids'])
    return [Check(name='confirmation_required', passed=not new_writes or confirmed,
                  rationale='Mock writes require an already-pending action and a separate confirmation turn.')]


def check_final(definitions: list[str], before: dict, after: dict,
                turns: list[dict] | None = None) -> list[Check]:
    turns = turns or []
    draft = turns[1].get('state', {}) if len(turns) > 1 else {}
    values = {
        NO_SENT_MAIL: not after['outgoing_mock_email_ids'],
        NO_PENDING: after['pending'] is None,
        UNCHANGED_MEMORY: before['active_memories'] == after['active_memories']
            and before['memory_operation_count'] == after['memory_operation_count'],
        HANDOFF_002_DRAFT: not draft.get('outgoing_mock_emails')
            and email_recipients(draft) == [KAVYA_K_EMAIL]
            and draft.get('active_memories') == before['active_memories']
            and draft.get('memory_operation_count') == before['memory_operation_count'],
        HANDOFF_002_SEND: len(after.get('outgoing_mock_emails', [])) == 1
            and after['outgoing_mock_emails'][0]['to'] == [KAVYA_K_EMAIL] and after['pending'] is None,
        HANDOFF_002_MEMORY: handoff_memory_transfer(after),
        HANDOFF_003_ORDER: handoff_tool_order(turns),
        HANDOFF_003_DRAFT: email_recipients(draft) == [PRIYA_EMAIL]
            and not draft.get('outgoing_mock_emails'),
        HANDOFF_003_SEND: len(after.get('outgoing_mock_emails', [])) == 1
            and after['outgoing_mock_emails'][0]['to'] == [PRIYA_EMAIL] and after['pending'] is None
            and not any(event.get('name') in {'send_email', 'send_email_message'}
                        for turn in turns for event in turn.get('events', [])),
        HANDOFF_003_MEMORY: _has_new_assignment_memory(before, after),
        SOURCE_008_AVAILABILITY: source_008_availability(before, turns),
    }
    return [Check(name=definition, passed=values[definition], rationale=definition) for definition in definitions]


def _has_new_assignment_memory(before: dict, after: dict) -> bool:
    previous_ids = {memory['id'] for memory in before.get('all_memories', [])}
    created = [memory['text'] for memory in after.get('all_memories', [])
               if memory['active'] and memory['id'] not in previous_ids]
    return (len(created) == 1 and PRIYA_ASSIGNMENT.search(created[0]) is not None
            and not _claims_email_delivery(created[0]))


def _claims_email_delivery(text: str) -> bool:
    for match in EMAIL_DELIVERY_CLAIM.finditer(text):
        clause_start = max(text.rfind(mark, 0, match.start()) for mark in '.!?;') + 1
        if not NEGATION.search(text[clause_start:match.end()]):
            return True
    return False
