"""Opt-in diagnostics for the actual agent model request and response."""
import json
import logging
import os
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_core.utils.function_calling import convert_to_openai_tool

TRACE_ENV = 'AGENT_TRACE_MODEL_CONTEXT'
TRACE_ENABLED = '1'
logger = logging.getLogger('uvicorn.error')


def model_trace(identity: Callable[[], tuple[str, str, str]]) -> Any:
    @wrap_model_call
    def trace(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        if os.getenv(TRACE_ENV) != TRACE_ENABLED:
            return handler(request)
        run_id, chat_id, message_id = identity()
        payload = {
            'system_message': request.system_message.model_dump(mode='json') if request.system_message else None,
            'messages': [message.model_dump(mode='json') for message in request.messages],
            'tools': [convert_to_openai_tool(tool) for tool in request.tools],
            'tool_choice': request.tool_choice,
        }
        logger.info('agent.model_request run_id=%s chat_id=%s message_id=%s payload=%s',
                    run_id, chat_id, message_id, json.dumps(payload, ensure_ascii=False, default=str))
        response = handler(request)
        logger.info('agent.model_response run_id=%s chat_id=%s message_id=%s messages=%s',
                    run_id, chat_id, message_id,
                    json.dumps([message.model_dump(mode='json') for message in response.result], ensure_ascii=False, default=str))
        return response

    return trace
