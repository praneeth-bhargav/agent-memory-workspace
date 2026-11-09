import logging

from fastapi.testclient import TestClient

from memory_layer import lab
from memory_layer.mock_sources import MockSources
from memory_layer.source_tools import source_tools


def test_api_created_messages_default_to_scenario_and_are_searchable(tmp_path, monkeypatch, caplog):
    database = tmp_path / 'lab.db'
    monkeypatch.setattr(lab, 'DATABASE', database)
    lab.get_sources.cache_clear()
    with TestClient(lab.app, base_url='http://localhost') as client, caplog.at_level(logging.INFO, logger='uvicorn.error'):
        as_of = client.get('/api/sources').json()['as_of']
        assert client.post('/api/sources/slack-channels', json={'name': '#qa-support', 'members': ['rudra']}).status_code == 201
        response = client.post('/api/sources/slack-messages', json={
            'channel': '#qa-support', 'author': 'rudra', 'text': 'I handle support ticket tis week'})
        assert response.status_code == 201
        slack_id = response.json()['id']
        response = client.post('/api/sources/incoming-mail', json={
            'subject': 'Support tickets', 'sender': 'rudra@example.example', 'body': 'I handle ticket escalations'})
        assert response.status_code == 201
        mail_id = response.json()['id']
        sources = MockSources(database)
        assert slack_id in [item.id for item in sources.slack('support tickets')]
        assert any(any(message.id == mail_id for message in thread.messages) for thread in sources.mail('ticket escalation'))
        with sources.connect() as conn:
            assert conn.execute('SELECT ts FROM slack_messages WHERE id=?', (slack_id,)).fetchone()[0] == as_of
            assert conn.execute('SELECT ts FROM mail_messages WHERE id=?', (mail_id,)).fetchone()[0] == as_of
        tool = next(item for item in source_tools(sources, lambda: 'qa-run') if item.name == 'search_slack')
        assert slack_id in tool.invoke({'query': 'support tickets'})
        assert 'qa-run' in caplog.text and slack_id in caplog.text
        previous = {'channel': '#qa-support', 'author': 'rudra', 'text': 'I handle support ticket tis week', 'ts': as_of}
        updated = {**previous, 'text': 'I handle database migrations'}
        response = client.put(f'/api/sources/slack-messages/{slack_id}', json={'expected': previous, 'message': updated})
        assert response.status_code == 200
        assert slack_id not in [item.id for item in sources.slack('support tickets')]
        assert slack_id in [item.id for item in sources.slack('database migration')]
    lab.get_sources.cache_clear()
