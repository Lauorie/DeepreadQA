"""Request context middleware: request id, timing, access log, HTTP metrics.

Access logs deliberately never contain request/response bodies (questions and
answers stay out of logs); only method, route, status and duration are kept.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .errors import PROBLEM_CONTENT_TYPE, problem_body
from .metrics import Metrics

logger = logging.getLogger("deepreadqa_api.access")

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _route_template(request: Request, status: int) -> str:
    route = request.scope.get("route")
    if route is not None:
        return route.path
    # unmatched paths (404 etc.) share one label to bound cardinality
    return "<unmatched>" if status == 404 else request.url.path


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, metrics: Optional[Metrics] = None) -> None:
        super().__init__(app)
        self._metrics = metrics

    async def dispatch(self, request: Request,
                       call_next: RequestResponseEndpoint) -> Response:
        inbound = request.headers.get("X-Request-ID", "")
        request_id = (inbound if _REQUEST_ID_RE.fullmatch(inbound)
                      else f"req_{uuid.uuid4().hex[:12]}")
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001 - last-resort 500 with headers intact
            logger.error("unhandled error [%s]: %s", request_id, exc,
                         exc_info=True)
            response = JSONResponse(
                status_code=500,
                content=problem_body(
                    request, code="internal", status=500,
                    detail="an unexpected error occurred; contact the "
                           "operator with the request_id"),
                media_type=PROBLEM_CONTENT_TYPE)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        if self._metrics is not None:
            self._metrics.observe_http(request.method,
                                       _route_template(request,
                                                       response.status_code),
                                       response.status_code)
        logger.info("%s %s -> %d %.1fms [%s]", request.method,
                    request.url.path, response.status_code, elapsed_ms,
                    request_id)
        return response
