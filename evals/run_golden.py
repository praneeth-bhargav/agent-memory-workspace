"""Run the user-labelled cases against the isolated local mock; never modify labels."""
import contextlib
import io
import json
import os
import tempfile
from pathlib import Path

# This baseline must never make paid or real-account calls, even when a key exists.
os.environ['OPENAI_API_KEY'] = ''
from fastapi.testclient import TestClient
from backend import main
from backend.chat import now
from backend.seed_mock import seed

ROOT = Path(__file__).resolve().parent.parent


def history_message(message_id):
    for path in sorted((ROOT / 'data/history').glob('*.jsonl')):
        with path.open() as source:
            for line in source:
                event = json.loads(line)
                if event['id'] == message_id:
                    return event['text']
    raise ValueError(f'Missing fixture source {message_id}')


def prepare_case(conn, thread_id, case_id):
    # Only these setups can be reproduced faithfully without history extraction.
    previous = {'case-004': 'm1011', 'case-005': 'm0051'}.get(case_id)
    if previous:
        stamp = now().isoformat()
        conn.execute('UPDATE threads SET last_user_at=? WHERE id=?', (stamp, thread_id))
        conn.execute('INSERT INTO chat_messages(id,thread_id,role,content,created_at,session) VALUES(?,?,?,?,?,0)',
                     (f'fixture-{case_id}', thread_id, 'assistant', history_message(previous), stamp))
    if case_id == 'case-004':
        draft = history_message('m1011').split('"', 2)[1].replace('\\n', '\n')
        action = {'type': 'send', 'to': 'maya.rao@acme-industries.example',
                  'cc': ['arjun@kestrelworks.example'], 'attachment': 'scope doc', 'content': draft}
        conn.execute('INSERT INTO pending_actions VALUES(?,?,?,?,?)',
                     ('fixture-pending', thread_id, json.dumps(action), 'pending', now().isoformat()))
    return case_id in {'case-004', 'case-005', 'case-012'}


def main_run():
    cases = json.loads((ROOT / 'evals/golden.json').read_text())
    results = []
    with tempfile.TemporaryDirectory(prefix='agent-golden-') as temp:
        main.DB = Path(temp) / 'evaluation.sqlite3'
        with TestClient(main.app) as client:
            with contextlib.redirect_stdout(io.StringIO()):
                seed()
            for case in cases:
                thread = client.post('/api/threads', json={'title': case['id']}).json()['id']
                with main.database() as conn:
                    complete_setup = prepare_case(conn, thread, case['id'])
                    before = conn.execute('SELECT count(*) FROM mock_actions').fetchone()[0]
                response = client.post(f'/api/threads/{thread}/messages', json={'text': case['user_message']})
                response.raise_for_status()
                message = client.get(f'/api/threads/{thread}/messages').json()[-1]
                with main.database() as conn:
                    actions = [json.loads(row['action']) for row in conn.execute('SELECT action FROM mock_actions ORDER BY rowid').fetchall()[before:]]
                results.append({'id': case['id'], 'category': case['category'],
                    'setup': 'reproduced' if complete_setup else 'incomplete: synthetic history is not extracted into memory',
                    'user_message': case['user_message'], 'expected_behavior': case['expected_behavior'],
                    'actual': message['content'], 'decision': message['details']['decision'],
                    'tools': [tool['name'] for tool in message['details']['tools']], 'mock_actions': actions,
                    'verdict': 'manual review required' if complete_setup else 'blocked: incomplete setup'})
    return {'mode': 'isolated mock data; OpenAI disabled',
            'note': 'No model judge or invented structured labels. Review actual output against the user-written expectation. Blocked cases cannot be fairly scored.',
            'results': results}


if __name__ == '__main__':
    print(json.dumps(main_run(), ensure_ascii=False, indent=2))
