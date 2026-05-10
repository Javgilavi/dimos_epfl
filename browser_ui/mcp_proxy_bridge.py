"""Laptop-side MCP proxy bridge for cloud-hosted browser_ui.

Polls the cloud server for queued MCP JSON-RPC requests, forwards them to the
local DimOS MCP server, and posts the response back. This is what lets the AWS
dashboard Agent use the robot's local MCP tools.
"""

from __future__ import annotations

import argparse
import os
import time

import httpx


def _headers() -> dict[str, str]:
    pw = os.environ.get("BRIDGE_PASSWORD", "")
    return {"X-Bridge-Password": pw} if pw else {}


def run(cloud_url: str, mcp_url: str) -> None:
    cloud = cloud_url.rstrip("/")
    local = httpx.Client(timeout=70.0)
    remote = httpx.Client(timeout=75.0)
    print(f"[mcp_proxy] cloud={cloud} local_mcp={mcp_url}")
    while True:
        try:
            pending = remote.get(
                f"{cloud}/bridge/mcp/pending",
                params={"timeout": 25},
                headers=_headers(),
            ).json()
            req_id = pending.get("id")
            if not req_id:
                continue
            payload = pending.get("payload") or {}
            try:
                resp = local.post(
                    mcp_url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                    },
                )
                text = resp.text
                try:
                    data = resp.json()
                except Exception:
                    data = None
                    for line in text.splitlines():
                        if line.startswith("data: "):
                            try:
                                import json
                                data = json.loads(line[6:])
                            except Exception:
                                continue
                    if data is None:
                        data = {"error": {"message": f"non-JSON MCP response: {text[:200]}"}}
            except Exception as exc:
                data = {"error": {"message": str(exc)}}

            remote.post(
                f"{cloud}/bridge/mcp/response",
                json={"id": req_id, "response": data},
                headers=_headers(),
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"[mcp_proxy] retrying after error: {exc}")
            time.sleep(2.0)


def main() -> int:
    p = argparse.ArgumentParser(description="Forward cloud MCP requests to local DimOS MCP")
    p.add_argument("--cloud-url", default=os.environ.get("CLOUD_URL", "http://localhost:8080"))
    p.add_argument("--mcp-url", default=os.environ.get("DIMOS_MCP_URL", "http://localhost:9990/mcp"))
    args = p.parse_args()
    run(args.cloud_url, args.mcp_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
