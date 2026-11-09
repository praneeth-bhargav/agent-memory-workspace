import logging

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from memory_layer.model_trace import model_trace


def test_opt_in_model_trace_records_request_and_tool_call(caplog, monkeypatch):
    monkeypatch.setenv('AGENT_TRACE_MODEL_CONTEXT', '1')

    @tool
    def find_available_slots(people: list[str]) -> str:
        """Find slots."""
        return ''

    request = ModelRequest(model=None, system_message=SystemMessage(content='synthetic system'),
                           messages=[HumanMessage(content='Arjun and me'),
                                     ToolMessage(content='synthetic calendar', tool_call_id='prior')],
                           tools=[find_available_slots])
    response = ModelResponse(result=[AIMessage(content='', tool_calls=[
        {'name': 'find_available_slots', 'args': {'people': ['kavya', 'arjun']}, 'id': 'call-1'}])])
    middleware = model_trace(lambda: ('run-1', 'chat-1', 'message-1'))
    with caplog.at_level(logging.INFO, logger='uvicorn.error'):
        result = middleware.wrap_model_call(request, lambda _: response)
    assert result is response
    for value in ('run-1', 'chat-1', 'message-1', 'synthetic system', 'Arjun and me',
                  'synthetic calendar', 'find_available_slots', 'kavya', 'arjun', 'call-1'):
        assert value in caplog.text


def test_model_trace_disabled_by_default(caplog, monkeypatch):
    monkeypatch.delenv('AGENT_TRACE_MODEL_CONTEXT', raising=False)
    request = ModelRequest(model=None, messages=[HumanMessage(content='private content')])
    response = ModelResponse(result=[AIMessage(content='private response')])
    with caplog.at_level(logging.INFO, logger='uvicorn.error'):
        model_trace(lambda: ('run', 'chat', 'message')).wrap_model_call(request, lambda _: response)
    assert 'private content' not in caplog.text
    assert 'private response' not in caplog.text
