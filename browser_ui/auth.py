"""
Auth — bridge password + viewer session tokens for SpatialMind.

Config (env vars):
  BRIDGE_PASSWORD    — shared secret for robot-side bridge scripts (empty = disabled)
  DASHBOARD_PASSWORD — password shown on the browser login screen (empty = auth disabled)
  MAX_VIEWERS        — max concurrent dashboard sessions (default: 2)
  SESSION_TTL        — idle timeout in seconds before a session expires (default: 3600)
"""

import os
import secrets
import time

from fastapi import HTTPException, Query, Request

BRIDGE_PASSWORD    = os.environ.get("BRIDGE_PASSWORD", "")
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")
MAX_VIEWERS        = int(os.environ.get("MAX_VIEWERS", "2"))
SESSION_TTL        = float(os.environ.get("SESSION_TTL", "30"))

# token → last_active timestamp (sliding-window expiry)
_sessions: dict[str, float] = {}


def _cleanup() -> None:
    now = time.time()
    for tok in [t for t, ts in _sessions.items() if now - ts > SESSION_TTL]:
        del _sessions[tok]


def create_session() -> str:
    """Issue a new viewer session token; raises 429 if at the concurrent limit."""
    _cleanup()
    if len(_sessions) >= MAX_VIEWERS:
        raise HTTPException(
            status_code=429,
            detail=f"Server full — max {MAX_VIEWERS} concurrent viewers. Try again later.",
        )
    token = secrets.token_urlsafe(32)
    _sessions[token] = time.time()
    return token


def verify_dashboard_password(pw: str) -> None:
    """Raise 403 if pw does not match DASHBOARD_PASSWORD."""
    if not DASHBOARD_PASSWORD:
        return
    if not pw or not secrets.compare_digest(pw.encode(), DASHBOARD_PASSWORD.encode()):
        raise HTTPException(status_code=403, detail="Wrong password")


def verify_bridge(request: Request) -> None:
    """FastAPI dependency — validates X-Bridge-Password header on ingest endpoints."""
    if not BRIDGE_PASSWORD:
        return
    pw = request.headers.get("X-Bridge-Password", "")
    if not pw or not secrets.compare_digest(pw.encode(), BRIDGE_PASSWORD.encode()):
        raise HTTPException(status_code=403, detail="Invalid bridge password")


def verify_session(
    request: Request,
    token: str | None = Query(default=None),
) -> str:
    """FastAPI dependency — validates a viewer session token.

    Accepts the token via X-Session-Token header OR a ?token= query param.
    The query-param fallback is needed for <img src> / MJPEG streams where
    the browser cannot set custom headers.
    """
    if not DASHBOARD_PASSWORD:
        return "open"
    _cleanup()
    tok = request.headers.get("X-Session-Token") or token
    if not tok or tok not in _sessions:
        raise HTTPException(status_code=401, detail="Login required")
    _sessions[tok] = time.time()  # slide expiry on every activity
    return tok
