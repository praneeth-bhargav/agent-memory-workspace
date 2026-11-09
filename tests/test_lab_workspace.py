from fastapi.testclient import TestClient

from memory_layer import lab


def test_react_workspace_entry_and_missing_build(tmp_path, monkeypatch):
    page = tmp_path / 'memory-lab.html'
    monkeypatch.setattr(lab, 'WORKSPACE_PAGE', page)
    with TestClient(lab.app, base_url='http://localhost') as client:
        assert client.get('/workspace').status_code == 503
        page.write_text('<html>Workspace</html>', encoding='utf-8')
        response = client.get('/workspace')
        assert response.status_code == 200 and 'Workspace' in response.text
