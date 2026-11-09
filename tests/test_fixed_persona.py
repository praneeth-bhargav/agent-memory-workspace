import json
from pathlib import Path

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from memory_layer.persona import persona_context


def test_persona_is_injected_before_every_model_call(tmp_path):
    path = tmp_path / 'persona.json'
    path.write_text(Path('data/persona.json').read_text())
    middleware = persona_context(path, 'rohan@kestrelworks.example')
    request = ModelRequest(model=None, system_message=SystemMessage(content='Original instructions'),
                           messages=[HumanMessage(content='Find time for Arjun and me')])
    received = []
    response = ModelResponse(result=[AIMessage(content='ok')])
    for _ in range(2):
        middleware.wrap_model_call(request, lambda actual: received.append(actual) or response)
    assert all('Rohan Menon' in item.system_message.content for item in received)
    assert all('rohan@kestrelworks.example' in item.system_message.content for item in received)
    assert request.system_message.content == 'Original instructions'


def test_missing_or_mismatched_persona_blocks_model_call(tmp_path):
    path = tmp_path / 'persona.json'
    called = []
    request = ModelRequest(model=None, messages=[HumanMessage(content='hello')])
    middleware = persona_context(path, 'rohan@kestrelworks.example')
    with pytest.raises((OSError, ValueError)):
        middleware.wrap_model_call(request, lambda actual: called.append(actual))
    data = json.loads(Path('data/persona.json').read_text())
    data['email'] = 'different@synthetic.example'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        middleware.wrap_model_call(request, lambda actual: called.append(actual))
    assert not called
