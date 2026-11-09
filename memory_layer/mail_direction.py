"""Parse explicit mail-address direction phrases for safe tool routing."""
import re

EMAIL = re.compile(r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)


def requested_mail_direction(text: str) -> tuple[str, str] | None:
    for match in EMAIL.finditer(text):
        prefix = text[max(0, match.start() - 50):match.start()]
        if re.search(r"\bcc(?:'d|ed)?\s+(?:to\s+)?$", prefix, re.I):
            return 'cc', match.group().casefold()
        if re.search(r'\b(?:addressed\s+)?to\s+$', prefix, re.I):
            return 'to', match.group().casefold()
        if re.search(r'\bfrom\s+$', prefix, re.I):
            return 'from', match.group().casefold()
    return None


def ensure_mail_direction(text: str, direction: str, address: str) -> None:
    requested = requested_mail_direction(text)
    if requested and requested[1] == address.casefold() and requested[0] != direction:
        raise ValueError(f'The request asks for emails {requested[0]} {address}; use direction={requested[0]}.')


def require_directional_tool_for_content_search(text: str) -> None:
    requested = requested_mail_direction(text)
    if requested:
        direction, address = requested
        raise ValueError(f'The request asks for emails {direction} {address}; use search_mail_by_address with that direction.')
