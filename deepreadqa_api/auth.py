"""Bearer API-key authentication (constant-time comparison)."""
from __future__ import annotations

import hmac
from typing import Optional

from fastapi import Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .errors import ApiError

_bearer = HTTPBearer(auto_error=False, scheme_name="ApiKeyBearer",
                     description="运营方签发的 API key，置于 "
                                 "`Authorization: Bearer <key>`")

_WWW_AUTH = {"WWW-Authenticate": "Bearer"}


async def require_api_key(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> str:
    """FastAPI dependency: validate the bearer token, return the key."""
    cfg = request.app.state.api_cfg
    if cfg.auth_disabled:
        request.state.api_key = "anonymous"
        return "anonymous"
    if creds is None:
        raise ApiError("unauthorized", 401,
                       "missing bearer token: send Authorization: Bearer <key>",
                       headers=_WWW_AUTH)
    token = creds.credentials.encode()
    for key in cfg.api_keys:
        if hmac.compare_digest(token, key.encode()):
            request.state.api_key = key
            return key
    raise ApiError("unauthorized", 401, "invalid API key", headers=_WWW_AUTH)
