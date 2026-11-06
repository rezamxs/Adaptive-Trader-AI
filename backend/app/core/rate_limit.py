"""In-process sliding-window rate limiter for heavy CPU endpoints (buglist item 18).

POST /backtest/run and POST /ml/train can each pin a core for seconds (rule
loops over up to 100k bars; dataset builds + walk-forward evaluations). Until
now there was NO limit: any token holder could queue unbounded heavy work and
starve the worker and the API threads.

Scope: this process's own request traffic, keyed by API token (hashed — raw
secrets never sit in a long-lived dict). The deployment model is a single
uvicorn process on the owner's machine, so in-process state is sufficient;
HEAVY_RATE_LIMIT_PER_MINUTE=0 disables the limiter entirely.
"""
import hashlib
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

from fastapi import Header, HTTPException, Request

from backend.app.core.config import settings

_WINDOW_SECONDS = 60.0
# Memory bound: at most this many tracked keys; stale keys are pruned when the
# map grows past it (a per-token limiter must not itself become the DoS).
_MAX_TRACKED_KEYS = 1024

_calls: Dict[str, Deque[float]] = defaultdict(deque)
_lock = threading.Lock()


def _key_for(token: Optional[str], request: Request) -> str:
    """Stable limiter key: hashed token when supplied, else the client IP."""
    raw = (token or "").strip()
    if not raw:
        raw = request.client.host if request.client else "unknown"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _prune_locked(now: float) -> None:
    if len(_calls) <= _MAX_TRACKED_KEYS:
        return
    for key in [k for k, dq in _calls.items() if not dq or now - dq[-1] >= _WINDOW_SECONDS]:
        _calls.pop(key, None)


def check_heavy_rate_limit(token: Optional[str], request: Request) -> None:
    """Raises 429 when the caller exceeded the per-minute heavy-call cap."""
    limit = int(getattr(settings, "HEAVY_RATE_LIMIT_PER_MINUTE", 0) or 0)
    if limit <= 0:
        return
    key = _key_for(token, request)
    now = time.monotonic()
    with _lock:
        window = _calls[key]
        while window and now - window[0] >= _WINDOW_SECONDS:
            window.popleft()
        if len(window) >= limit:
            retry_after = max(1, int(_WINDOW_SECONDS - (now - window[0])) + 1)
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Heavy request rate limit exceeded ({limit}/minute); "
                    f"retry in ~{retry_after}s."
                ),
                headers={"Retry-After": str(retry_after)},
            )
        window.append(now)
        _prune_locked(now)


def reset_rate_limiter_for_tests() -> None:
    """Test isolation hook: clears every tracked window."""
    with _lock:
        _calls.clear()


async def heavy_rate_limiter(
    request: Request,
    x_api_token: Optional[str] = Header(default=None),
) -> None:
    """FastAPI dependency for the heavy endpoints. It must run AFTER
    require_api_token resolved (the token it keys on is the same header), and
    it is deliberately cheap: O(window) deque ops under one lock."""
    check_heavy_rate_limit(x_api_token, request)
