"""Token-based authentication dependency for state-changing API endpoints.

Fail-closed: a request without a valid shared-secret token is rejected before
any state mutation. The token is sent as the ``X-API-TOKEN`` header.
"""
from typing import Optional

from fastapi import Header, HTTPException, status

from backend.app.core.logging import log_security_event
from backend.app.core.config import settings

_TOKEN_HEADER = "X-API-TOKEN"


def require_api_token(x_api_token: Optional[str] = Header(default=None)) -> str:
    """Dependency that rejects any caller lacking the configured API token."""
    expected = (settings.API_TOKEN or "").strip()
    supplied = (x_api_token or "").strip()

    if not expected:
        # Config never converged to a usable token: refuse everything.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API token is not configured; refusing state-changing request.",
        )
    if not supplied:
        # Missing-header rejections are security-relevant too: log them like
        # invalid-token rejections (never logging the supplied value itself).
        log_security_event("API_AUTH_MISSING", {"header": _TOKEN_HEADER}, "WARNING")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-TOKEN header.",
        )
    # Constant-time compare to avoid token-guessing side channels.
    import hmac

    if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        log_security_event("API_AUTH_REJECTED", {"header": _TOKEN_HEADER}, "WARNING")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token.",
        )
    return supplied
