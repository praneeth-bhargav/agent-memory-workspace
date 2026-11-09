"""Select and validate references to records read for the current answer."""
import json
import re
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel

SourceKind = Literal['mail', 'slack', 'calendar', 'contact', 'chat']


class SourceReferenceSelection(BaseModel):
    kind: SourceKind
    source_id: str


class SourceReference(BaseModel):
    kind: SourceKind
    source_id: str
    label: str
    chat_id: str | None = None


def _record_label(kind: SourceKind, record: Mapping[str, object], parent_label: str = '') -> str:
    if kind == 'mail':
        subject = str(record.get('subject') or parent_label or 'email')
        sender = str(record.get('sender') or record.get('from') or '')
        return f'Email · {subject}' + (f' · {sender}' if sender else '')
    if kind == 'slack':
        channel = str(record.get('channel') or '')
        return f'Slack message · {channel}' if channel else 'Slack message'
    if kind == 'calendar':
        return f"Calendar event · {record.get('title', 'event')}"
    return f"Contact · {record.get('name', 'contact')}"


def _add_record(records: dict[tuple[SourceKind, str], str], kind: SourceKind,
                record: object, parent_label: str = '') -> None:
    if not isinstance(record, dict):
        return
    source_id = record.get('id')
    if isinstance(source_id, str) and source_id:
        records[(kind, source_id)] = _record_label(kind, record, parent_label)


def _available_records(current_sources: list[str]) -> dict[tuple[SourceKind, str], str]:
    records: dict[tuple[SourceKind, str], str] = {}
    table_kinds = {'contacts': 'contact', 'mail_messages': 'mail',
                   'slack_messages': 'slack', 'calendar_events': 'calendar'}
    for raw in current_sources:
        try:
            entry = json.loads(raw)
            tool = entry['tool']
            result = json.loads(entry['result'])
        except (TypeError, ValueError, KeyError):
            continue
        if tool == 'memory_read':
            for cited in result.get('current_sources', []):
                kind, source_id, record = cited.get('kind'), cited.get('id'), cited.get('record')
                if kind in ('mail', 'slack', 'calendar') and isinstance(source_id, str):
                    _add_record(records, kind, record)
                    if (kind, source_id) not in records and source_id:
                        records[(kind, source_id)] = _record_label(kind, record or {})
            continue
        if tool == 'query_sources_sql':
            source_sql = entry.get('arguments', {}).get('sql', '')
            tables = set(re.findall(r'\b(?:FROM|JOIN)\s+([a-z_]+)', source_sql, re.I))
            kinds = {table_kinds[table] for table in tables if table in table_kinds}
            if len(kinds) != 1:
                continue
            kind = kinds.pop()
            columns = result.get('columns', [])
            id_index = next((i for i, column in enumerate(columns) if column in ('id', 'message_id')), None)
            if id_index is None:
                continue
            for row in result.get('rows', []):
                if isinstance(row, list) and id_index < len(row) and isinstance(row[id_index], str):
                    records[(kind, row[id_index])] = _record_label(kind, dict(zip(columns, row)))
            continue
        items = result.get('items', [])
        if tool == 'search_contacts':
            for item in items:
                _add_record(records, 'contact', item)
        elif tool == 'search_mail_by_address':
            for item in items:
                _add_record(records, 'mail', item)
        elif tool in ('search_mail', 'read_mail_thread'):
            for thread in items:
                if isinstance(thread, dict):
                    for message in thread.get('messages', []):
                        _add_record(records, 'mail', message, str(thread.get('subject') or ''))
        elif tool == 'search_slack':
            for item in items:
                _add_record(records, 'slack', item)
        elif tool == 'read_calendar':
            for item in items:
                _add_record(records, 'calendar', item)
    return records


def validate_source_references(selections: list[SourceReferenceSelection],
                               current_sources: list[str]) -> list[SourceReference]:
    available = _available_records(current_sources)
    chat_records: dict[str, str] = {}
    for raw in current_sources:
        try:
            entry = json.loads(raw)
            result = json.loads(entry['result'])
            if entry['tool'] == 'memory_read':
                for source in result.get('current_sources', []):
                    record = source.get('record')
                    if (source.get('kind') == 'chat' and isinstance(source.get('id'), str)
                            and isinstance(record, dict) and isinstance(record.get('chat_id'), str)):
                        chat_records[source['id']] = record['chat_id']
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
    references: list[SourceReference] = []
    seen: set[tuple[SourceKind, str]] = set()
    for selection in selections:
        key = (selection.kind, selection.source_id)
        chat_id = chat_records.get(selection.source_id) if selection.kind == 'chat' else None
        label = f'Message · #{selection.source_id}' if chat_id else available.get(key)
        if label and key not in seen:
            references.append(SourceReference(kind=selection.kind, source_id=selection.source_id, label=label, chat_id=chat_id))
            seen.add(key)
    return references
