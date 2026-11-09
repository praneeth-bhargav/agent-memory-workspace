import logging
import sqlite3

from fastapi.testclient import TestClient

from memory_layer import lab
from memory_layer.mock_actions import MockActions, MockSlackPost
from memory_layer.mock_sources import MockSources


def test_sources_api_seeds_and_reads_agent_changes(tmp_path, monkeypatch, caplog):
    database = tmp_path / 'lab.sqlite3'
    monkeypatch.setattr(lab, 'DATABASE', database)
    lab.get_sources.cache_clear()
    with TestClient(lab.app, base_url='http://localhost') as client, caplog.at_level('INFO', logger='uvicorn.error'):
        response = client.get('/api/sources')
        assert response.status_code == 200
        snapshot = response.json()
        assert snapshot['as_of'] == lab.Persona.model_validate_json(lab.PERSONA.read_text(encoding='utf-8')).as_of
        assert len(snapshot['contacts']) == 9
        assert len(snapshot['mail']) == 9
        assert len(snapshot['calendar_events']) == 31
        assert len(snapshot['slack_channels']) == 4
        actions = MockActions(MockSources(database), '2026-05-18T09:30:00+05:30', 'rohan@kestrelworks.example')
        actions.prepare_slack(MockSlackPost(channel='#ops', text='Isolated API refresh check'))
        actions.confirm()
        assert any(item['text'] == 'Isolated API refresh check' for item in client.get('/api/sources').json()['slack_messages'])
        for method in ('POST', 'PUT', 'PATCH', 'DELETE'):
            assert client.request(method, '/api/sources', json={}).status_code == 405
        assert 'sources.snapshot status=ok' not in caplog.text
        with caplog.at_level('DEBUG', logger='uvicorn.error'):
            assert client.get('/api/sources').status_code == 200
        assert any(record.levelno == logging.DEBUG and 'sources.snapshot status=ok' in record.message for record in caplog.records)
        with sqlite3.connect(database) as conn:
            conn.execute('DROP TABLE contacts')
        failed = client.get('/api/sources')
        assert failed.status_code == 503 and 'backend log' in failed.json()['detail']
        assert 'sources.snapshot status=error' in caplog.text and 'OperationalError' in caplog.text
    lab.get_sources.cache_clear()
