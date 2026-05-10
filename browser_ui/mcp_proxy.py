"""In-process request queue for cloud-server ↔ local DimOS MCP forwarding.

When browser_ui runs on EC2, it cannot connect to the laptop's
http://localhost:9990/mcp. The laptop-side mcp_proxy_bridge.py long-polls this
queue, executes each JSON-RPC request against local DimOS MCP, and posts the
response back.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from typing import Any

_pending: queue.Queue[dict[str, Any]] = queue.Queue()
_responses: dict[str, dict[str, Any]] = {}
_cv = threading.Condition()


def submit(payload: dict[str, Any], timeout: float = 65.0) -> dict[str, Any]:
    request_id = uuid.uuid4().hex
    with _cv:
        _pending.put({"id": request_id, "payload": payload, "ts": time.time()})
        _cv.notify_all()

        deadline = time.time() + timeout
        while request_id not in _responses:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for local MCP bridge")
            _cv.wait(timeout=min(remaining, 1.0))
        return _responses.pop(request_id)


def get_pending(timeout: float = 25.0) -> dict[str, Any] | None:
    try:
        return _pending.get(timeout=timeout)
    except queue.Empty:
        return None


def put_response(request_id: str, response: dict[str, Any]) -> None:
    with _cv:
        _responses[request_id] = response
        _cv.notify_all()


__all__ = ["submit", "get_pending", "put_response"]
