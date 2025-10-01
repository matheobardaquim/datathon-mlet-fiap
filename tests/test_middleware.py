import pytest
from starlette.requests import Request
from starlette.responses import Response
from starlette.middleware.base import RequestResponseEndpoint
from starlette.datastructures import URL
from src.helpers.middleware import PrometheusMiddleware
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_dispatch_metrics_path():
    request = Request({
        "type": "http",
        "method": "GET",
        "path": "/metrics",
        "headers": [],
        "scheme": "http",
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 80),
        "query_string": b"",
        "root_path": "",
        "http_version": "1.1",
        "extensions": {},
        "url": URL("http://testserver/metrics"),
    })
    call_next = AsyncMock(return_value=Response("metrics", status_code=200))
    middleware = PrometheusMiddleware(call_next)
    response = await middleware.dispatch(request, call_next)
    assert response.status_code == 200
    assert response.body == b"metrics"

@pytest.mark.asyncio
async def test_dispatch_normal_path(monkeypatch):
    request = Request({
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": [],
        "scheme": "http",
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 80),
        "query_string": b"",
        "root_path": "",
        "http_version": "1.1",
        "extensions": {},
        "url": URL("http://testserver/test"),
    })
    call_next = AsyncMock(return_value=Response("ok", status_code=201))
    middleware = PrometheusMiddleware(call_next)

    with patch("src.helpers.middleware.IN_PROGRESS_REQUESTS"), \
         patch("src.helpers.middleware.REQUESTS_TOTAL"), \
         patch("src.helpers.middleware.REQUEST_LATENCY"), \
         patch("src.helpers.middleware.REQUEST_ERRORS"):
        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 201
        assert response.body == b"ok"

@pytest.mark.asyncio
async def test_dispatch_exception(monkeypatch):
    request = Request({
        "type": "http",
        "method": "POST",
        "path": "/fail",
        "headers": [],
        "scheme": "http",
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 80),
        "query_string": b"",
        "root_path": "",
        "http_version": "1.1",
        "extensions": {},
        "url": URL("http://testserver/fail"),
    })
    async def failing_call_next(request):
        raise ValueError("fail")
    middleware = PrometheusMiddleware(failing_call_next)

    with patch("src.helpers.middleware.IN_PROGRESS_REQUESTS"), \
         patch("src.helpers.middleware.REQUESTS_TOTAL"), \
         patch("src.helpers.middleware.REQUEST_LATENCY"), \
         patch("src.helpers.middleware.REQUEST_ERRORS") as mock_errors:
        with pytest.raises(ValueError):
            await middleware.dispatch(request, failing_call_next)
        assert mock_errors.labels.called