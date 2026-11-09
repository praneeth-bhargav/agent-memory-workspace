import logging

from memory_layer.lab import SourcePollingFilter


def test_successful_source_poll_is_debug_but_other_requests_stay_visible(caplog):
    def access(method, path, status):
        return logging.LogRecord('uvicorn.access', logging.INFO, '', 0,
                                 '%s - "%s %s HTTP/%s" %d',
                                 ('127.0.0.1:123', method, path, '1.1', status), None)

    access_filter = SourcePollingFilter()
    with caplog.at_level(logging.DEBUG, logger='uvicorn.error'):
        assert not access_filter.filter(access('GET', '/api/sources?refresh=1', 200))
        assert access_filter.filter(access('GET', '/api/sources', 503))
        assert access_filter.filter(access('POST', '/api/sources', 405))
        assert access_filter.filter(access('GET', '/api/chats', 200))
    assert any(record.levelno == logging.DEBUG and 'GET /api/sources?refresh=1' in record.message
               for record in caplog.records)
