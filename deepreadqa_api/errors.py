"""RFC 9457 problem+json error model shared by every non-2xx response."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

PROBLEM_TYPE_BASE = "https://deepreadqa.dev/errors/"
PROBLEM_CONTENT_TYPE = "application/problem+json"

_STATUS_TITLES = {
    400: "Bad Request", 401: "Unauthorized", 404: "Not Found",
    405: "Method Not Allowed", 422: "Unprocessable Entity",
    429: "Too Many Requests", 500: "Internal Server Error",
    502: "Bad Gateway", 503: "Service Unavailable",
}


class ApiError(Exception):
    """Raise anywhere inside a route; rendered as problem+json by the handler."""

    def __init__(self, code: str, status: int, detail: str, *,
                 title: str | None = None,
                 headers: dict[str, str] | None = None,
                 extra: dict | None = None) -> None:
        super().__init__(detail)
        self.code = code
        self.status = status
        self.detail = detail
        self.title = title or _STATUS_TITLES.get(status, "Error")
        self.headers = headers or {}
        self.extra = extra or {}


def problem_body(request: Request, *, code: str, status: int, detail: str,
                 title: str | None = None, **extra) -> dict:
    body = {
        "type": PROBLEM_TYPE_BASE + code.replace("_", "-"),
        "title": title or _STATUS_TITLES.get(status, "Error"),
        "status": status,
        "detail": detail,
        "code": code,
        "request_id": getattr(request.state, "request_id", None),
    }
    body.update(extra)
    return body


def _problem_response(request: Request, *, code: str, status: int, detail: str,
                      title: str | None = None,
                      headers: dict[str, str] | None = None,
                      **extra) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=problem_body(request, code=code, status=status, detail=detail,
                             title=title, **extra),
        headers=headers,
        media_type=PROBLEM_CONTENT_TYPE,
    )


def install_handlers(app: FastAPI) -> None:
    """Route every error path through the problem+json renderer."""

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _problem_response(request, code=exc.code, status=exc.status,
                                 detail=exc.detail, title=exc.title,
                                 headers=exc.headers, **exc.extra)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request,
                          exc: RequestValidationError) -> JSONResponse:
        return _problem_response(
            request, code="invalid_request", status=422,
            detail="request body failed validation",
            errors=[{"loc": list(e["loc"]), "msg": e["msg"]}
                    for e in exc.errors()])

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request,
                    exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(
            exc.status_code, "http_error")
        return _problem_response(request, code=code, status=exc.status_code,
                                 detail=str(exc.detail),
                                 headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled error on %s %s: %s", request.method,
                     request.url.path, exc, exc_info=True)
        return _problem_response(
            request, code="internal", status=500,
            detail="an unexpected error occurred; contact the operator "
                   "with the request_id")
