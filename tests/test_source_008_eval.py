import json

from evals.state_checks import SOURCE_008_AVAILABILITY, check_final


def result(people=('arjun', 'rohan'), answer='11:00–11:30 IST', start='2026-05-26T11:00:00+05:30',
           end='2026-05-26T11:30:00+05:30', events=None):
    before = {'pending': None, 'outgoing_mock_email_ids': [], 'active_memories': [],
              'memory_operation_count': 0, 'mock_write_ids': [], 'calendar_events': events or []}
    start_call = {'kind': 'tool_start', 'name': 'find_available_slots', 'tool_call_id': '1',
                  'data': json.dumps({'people': list(people), 'start': '2026-05-26T11:00:00+05:30',
                                      'end': '2026-05-26T14:00:00+05:30', 'duration_minutes': 30})}
    end_call = {'kind': 'tool_result', 'name': 'find_available_slots', 'tool_call_id': '1', 'status': 'success',
                'data': json.dumps({'items': [{'start': start, 'end': end}]})}
    turns = [{'events': [start_call, end_call, {'kind': 'answer', 'data': answer}], 'state': before}]
    return check_final([SOURCE_008_AVAILABILITY], before, before, turns)[0].passed


def test_source_008_checks_actual_offered_interval_and_people():
    assert result()
    assert not result(people=('arjun', 'kavya'))
    assert not result(answer='12:00–12:30 IST')
    assert not result(start='2026-05-26T14:00:00+05:30', end='2026-05-26T14:30:00+05:30', answer='14:00–14:30')
    assert not result(events=[{'person': 'rohan', 'start': '2026-05-26T11:15:00+05:30',
                               'end': '2026-05-26T11:45:00+05:30'}])
    assert result(answer='13:30–14:00 IST', start='2026-05-26T13:30:00+05:30',
                  end='2026-05-26T14:00:00+05:30')
