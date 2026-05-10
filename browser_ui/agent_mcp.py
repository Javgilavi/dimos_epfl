"""
Bedrock + DimOS MCP Agent — connects Claude Sonnet 4.6 (Bedrock, us-west-2)
to the dimos robot's MCP server (localhost:9990 by default), discovers the
22+ robot tools (navigate_with_text, begin_exploration, look_out_for, …),
and runs a streaming converse_stream loop with tool_use handoff.

Drop-in replacement for agent.run_agent_stream — same generator interface so
main.py /query/stream can swap to it without changing protocols.

Usage:
    from agent_mcp import run_mcp_agent_stream
    for token in run_mcp_agent_stream("find a door"):
        print(token, end="", flush=True)
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────

MCP_URL = os.environ.get("DIMOS_MCP_URL", "http://localhost:9990/mcp")
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "auto").lower()
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6"
)
MAX_TOKENS = int(os.environ.get("BEDROCK_MAX_TOKENS", "1024"))
MCP_TIMEOUT = float(os.environ.get("DIMOS_MCP_TIMEOUT", "60"))

# Tools that physically move the robot or modify persistent state.
# Used in Ask mode to filter out anything that would let the LLM "do" things —
# Ask is read-only by design.
ASK_BLOCKED_TOOLS = {
    "navigate_with_text",
    "navigate_to_coordinates",
    "navigate_to_object",
    "begin_exploration",
    "end_exploration",
    "start_patrol",
    "stop_patrol",
    "start_security_patrol",
    "stop_security_patrol",
    "follow_person",
    "stop_following",
    "look_out_for",          # starts the perception loop, not strictly read-only
    "stop_looking_out",
    "relative_move",
    "execute_sport_command",
    "tag_location",          # writes to spatial memory
    "stop_navigation",
    "stop_robot",
    "patrol",
    "agent_send",            # could route arbitrary commands to other modules
}

# Agent mode — actionable. The MCP tool descriptions cover most behavior,
# but we add a short prompt to cover one recurring issue at this hackathon:
# "watch / look-out" tasks where the user wants the robot to stop or alert
# when it SEES something (e.g. a person). The default planner just walks and
# misses live detections, so we explicitly route those to look_out_for /
# follow_person and tell the LLM to re-check perception at every step.
AGENT_SYSTEM_PROMPT = [{
    "text": (
        "You drive a Unitree Go2 (real robot). You have the full DimOS MCP "
        "toolset: navigation (explore/start_exploration, navigate_to, "
        "stop_navigation), perception (look_out_for, stop_looking_out), "
        "follow_person/follow_object, semantic memory, etc. A separate "
        "YOLO11 worker publishes detections to /yolo11/detections.\n\n"
        "CRITICAL — search/find tasks must STOP on detection.\n"
        "When the user says any of: 'find a person', 'search for X', 'look "
        "for X', 'stop if you see X', 'tell me when X appears', etc., the "
        "robot must stop the moment X is detected. The frontier explorer "
        "runs blind — it never polls perception itself, so YOU must wire "
        "the perception → action link via the `then` parameter of "
        "`look_out_for`. Do this with ONE tool call up front, before any "
        "movement:\n"
        "    look_out_for(\n"
        "        description_of_things=[\"<the target>\"],\n"
        "        then={\"tool\": \"stop_navigation\", \"args\": {}}\n"
        "    )\n"
        "The `then` clause fires the moment a match is detected, with no "
        "agent round-trip — that is what makes the robot actually stop. "
        "Without `then=stop_navigation` the robot will keep exploring "
        "while the perception callback shouts at the agent, which is the "
        "bug from yesterday. Always use `then=` for find/search/watch.\n"
        "ONLY THEN start movement (start_exploration / navigate_to / etc.) "
        "in a separate tool call, in the same turn.\n\n"
        "FOLLOW tasks: ('follow X', 'come with me') → use look_out_for "
        "with then={\"tool\": \"follow_person\", \"args\": {...}}, or "
        "follow_person directly if a recent detection is already known.\n\n"
        "Other rules:\n"
        "- Before reporting that a movement task is done, call "
        "`get_semantic_map` once so your reply reflects what the robot "
        "actually saw, not stale memory.\n"
        "- If a tool returns no results, say so plainly. Don't fabricate.\n"
        "- Replies render in a small chat bubble — keep them concise."
    )
}]

ASK_SYSTEM_PROMPT = [{
    "text": (
        "You are the perceptual / awareness layer of a Unitree Go2 robot at "
        "EPFL RoboHack 2026. You have FULL READ access to the dimos MCP server: "
        "the camera, semantic map, spatial memory, robot pose, navigation state, "
        "module list, server status, and so on. Use whatever read tools you need "
        "to answer the user's question accurately.\n\n"
        "However, you CANNOT make the robot move or trigger any action — those "
        "tools are intentionally disabled in this mode. If the user asks for an "
        "action (go, navigate, explore, follow, stop), briefly explain what you "
        "would do and tell them to switch to the **Agent** tab (🤖) to execute. "
        "Don't refuse rudely — just hand off cleanly.\n\n"
        "When data is missing, say so plainly and suggest a next step (e.g. "
        "'no objects mapped yet — switch to Agent and ask it to explore'). "
        "When partially confident, give your best estimate with stated "
        "confidence ('the chair seems to be near (1.2, 0.3)m, ~70% sure'). "
        "Don't fabricate; don't over-refuse.\n\n"
        "Be concise — replies render in a small chat bubble."
    )
}]

# Default / legacy alias preserved for any caller that imported SYSTEM_PROMPT.
SYSTEM_PROMPT = AGENT_SYSTEM_PROMPT


def _load_dotenv_from_dimos() -> None:
    """If AWS_* env vars aren't set, pull them from dimos's .env so the
    laptop running dimos doesn't need to re-export creds for this server."""
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        return
    dimos_env = Path.home() / "robohack-epfl/dimos/.env"
    if not dimos_env.exists():
        return
    for line in dimos_env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k.startswith("AWS_") and k not in os.environ:
            os.environ[k] = v


_load_dotenv_from_dimos()


# ── MCP HTTP Client ──────────────────────────────────────────

class MCPClient:
    """Tiny HTTP-only MCP client. Speaks the same JSON-RPC envelope dimos's
    server expects (we already verified live: initialize, tools/list,
    tools/call all return 200 in <10 ms locally)."""

    def __init__(self, url: str = MCP_URL, timeout: float = MCP_TIMEOUT) -> None:
        self.url = url
        self.timeout = timeout
        self._next_id = 0
        self._client = httpx.Client(timeout=timeout)
        self._use_bridge = MCP_TRANSPORT == "bridge"

    def _bridge_rpc(self, body: dict) -> dict:
        import mcp_proxy

        data = mcp_proxy.submit(body, timeout=self.timeout + 10.0)
        if "error" in data:
            raise RuntimeError(f"MCP bridge error: {data['error']}")
        return data.get("result", {})

    def _rpc(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        body = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
        }
        if params is not None:
            body["params"] = params
        if self._use_bridge:
            return self._bridge_rpc(body)
        try:
            resp = self._client.post(
                self.url,
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
            )
            resp.raise_for_status()
        except Exception:
            if MCP_TRANSPORT == "auto":
                self._use_bridge = True
                return self._bridge_rpc(body)
            raise
        # Server may stream SSE for long-running tools; for our short calls it
        # returns plain JSON. Be lenient.
        text = resp.text
        try:
            data = resp.json()
        except Exception:
            # SSE fallback: extract last `data: { ... }` line
            data = None
            for line in text.splitlines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                    except Exception:
                        continue
            if data is None:
                raise RuntimeError(f"MCP returned non-JSON: {text[:200]}")
        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")
        return data.get("result", {})

    def initialize(self) -> dict:
        return self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "robohack2026-chat", "version": "1.0"},
        })

    def list_tools(self) -> list[dict]:
        return self._rpc("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> str:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        # MCP tool result: {"content": [{"type": "text", "text": "..."}], ...}
        content = result.get("content", [])
        parts: list[str] = []
        for block in content:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "image":
                parts.append("[image returned, not displayed]")
            else:
                parts.append(json.dumps(block))
        if not parts:
            # Some skills return structured result fields (e.g. observe)
            parts.append(json.dumps(result)[:400])
        return "\n".join(parts)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


# ── MCP → Bedrock tool-spec conversion ──────────────────────

def _mcp_to_bedrock_tools(mcp_tools: list[dict]) -> list[dict]:
    """Bedrock converse expects:
        {"toolSpec": {"name", "description", "inputSchema": {"json": {...}}}}
    MCP gives:
        {"name", "description", "inputSchema": {...}}
    """
    out: list[dict] = []
    for t in mcp_tools:
        schema = t.get("inputSchema") or {"type": "object", "properties": {}}
        # Bedrock requires properties to exist even if empty
        if "properties" not in schema:
            schema = {**schema, "properties": {}}
        out.append({
            "toolSpec": {
                "name": t["name"],
                "description": t.get("description", "")[:1023],
                "inputSchema": {"json": schema},
            }
        })
    return out


# ── Bedrock streaming agent loop ─────────────────────────────

def _bedrock_stream_with_mcp(
    user_message: str,
    mcp: MCPClient,
    bedrock_tools: list[dict],
    system_prompt: list[dict] | None = None,
) -> Generator[str, None, None]:
    """Multi-turn loop: ask the model → stream tokens → if it requested a
    tool_use, call MCP, append result, loop until the model stops."""
    import boto3

    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    messages: list[dict] = [{"role": "user", "content": [{"text": user_message}]}]
    sys_prompt = system_prompt if system_prompt is not None else AGENT_SYSTEM_PROMPT

    max_iterations = 8  # safety guard against runaway tool loops
    for _ in range(max_iterations):
        request = {
            "modelId": BEDROCK_MODEL_ID,
            "messages": messages,
            "inferenceConfig": {"maxTokens": MAX_TOKENS},
        }
        if sys_prompt:
            request["system"] = sys_prompt
        if bedrock_tools:
            request["toolConfig"] = {"tools": bedrock_tools}
        resp = bedrock.converse_stream(**request)

        assistant_content: list[dict] = []
        current_tool: dict[str, Any] = {}
        stop_reason: str | None = None

        for event in resp["stream"]:
            if "contentBlockStart" in event:
                tu = event["contentBlockStart"].get("start", {}).get("toolUse", {})
                if tu:
                    current_tool = {
                        "toolUseId": tu.get("toolUseId"),
                        "name": tu.get("name"),
                        "input_str": "",
                    }

            elif "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    yield delta["text"]
                    assistant_content.append({"text": delta["text"]})
                elif "toolUse" in delta:
                    current_tool["input_str"] += delta["toolUse"].get("input", "")

            elif "contentBlockStop" in event:
                if current_tool.get("name"):
                    raw = current_tool.pop("input_str", "") or "{}"
                    try:
                        tool_input = json.loads(raw)
                    except Exception:
                        tool_input = {}
                    assistant_content.append({
                        "toolUse": {
                            "toolUseId": current_tool["toolUseId"],
                            "name": current_tool["name"],
                            "input": tool_input,
                        }
                    })
                    current_tool = {}

            elif "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason")

        # Merge consecutive text blocks (Bedrock requires non-empty content)
        merged: list[dict] = []
        text_buf = ""
        for blk in assistant_content:
            if "text" in blk:
                text_buf += blk["text"]
            else:
                if text_buf:
                    merged.append({"text": text_buf})
                    text_buf = ""
                merged.append(blk)
        if text_buf:
            merged.append({"text": text_buf})
        if not merged:
            merged.append({"text": ""})
        messages.append({"role": "assistant", "content": merged})

        if stop_reason != "tool_use":
            return

        # Execute MCP tool calls and feed results back
        tool_results: list[dict] = []
        for blk in merged:
            if "toolUse" in blk:
                tu = blk["toolUse"]
                yield f"\n[→ {tu['name']}({json.dumps(tu.get('input', {}))})]\n"
                try:
                    if tu["name"] in CLOUD_TOOL_HANDLERS:
                        # Cloud-side read tool — no MCP roundtrip
                        output = CLOUD_TOOL_HANDLERS[tu["name"]](
                            tu.get("input", {})
                        )
                    else:
                        output = mcp.call_tool(tu["name"], tu.get("input", {}))
                except Exception as e:
                    output = f"Error calling {tu['name']}: {e}"
                    logger.exception("Tool call failed")
                # Truncate giant outputs so we don't blow the context
                if len(output) > 4000:
                    output = output[:4000] + "\n[...truncated]"
                tool_results.append({
                    "toolResult": {
                        "toolUseId": tu["toolUseId"],
                        "content": [{"text": output}],
                    }
                })
        messages.append({"role": "user", "content": tool_results})

    yield "\n[max tool-loop iterations reached]\n"


# ── Public Interface ─────────────────────────────────────────

# ── Cloud-side read tools (FastAPI loopback) ─────────────────
#
# These augment the dimos MCP toolset with tools that read the *cloud server's*
# own state — the cumulative semantic map (world_store), live lidar voxel map,
# pointcloud stats, robot pose, etc. This way Ask mode has full visibility into
# what the cloud knows about the world WITHOUT being able to drive the robot.

CLOUD_BASE = os.environ.get("CLOUD_SELF_URL", "http://localhost:8080")


def _cloud_get(path: str, timeout: float = 3.0) -> dict | list | None:
    try:
        r = httpx.get(f"{CLOUD_BASE}{path}", timeout=timeout)
        if r.status_code != 200:
            return {"_error": f"{path} returned {r.status_code}"}
        return r.json()
    except Exception as e:
        return {"_error": f"{path}: {e}"}


def _cloud_tool_get_semantic_map(args: dict) -> str:
    """All cumulative objects detected by the robot, with positions + confidence."""
    data = _cloud_get("/map") or {}
    if isinstance(data, dict) and data.get("_error"):
        return f"(could not reach cloud /map: {data['_error']})"
    objs = data.get("objects", []) if isinstance(data, dict) else []
    if not objs:
        return ("Semantic map is empty — no objects detected yet. "
                "If you want detections, switch to Agent and ask it to explore.")
    lines = [f"Semantic map ({len(objs)} object(s)):"]
    for o in sorted(objs, key=lambda x: -(x.get('confidence') or 0))[:30]:
        lines.append(
            f"  - {o.get('label')}: "
            f"({(o.get('pose') or {}).get('x', 0):.2f}, "
            f"{(o.get('pose') or {}).get('y', 0):.2f}) m  "
            f"conf {(o.get('confidence') or 0)*100:.0f}%  "
            f"seen {o.get('seen_count', 1)}×"
        )
    return "\n".join(lines)


def _cloud_tool_get_robot_pose(args: dict) -> str:
    """Live robot position + orientation from the most recent nav_bridge update."""
    data = _cloud_get("/pose/live") or {}
    if isinstance(data, dict) and data.get("_error"):
        return f"(could not reach /pose/live: {data['_error']})"
    pose = data.get("go2_a") if isinstance(data, dict) else None
    if not pose and isinstance(data, dict) and data:
        pose = next((v for v in data.values() if isinstance(v, dict)), None)
    if not pose:
        return ("Robot pose unknown — nav_bridge may not be running or the robot "
                "hasn't published odometry yet.")
    x = pose.get("x") if isinstance(pose, dict) else None
    y = pose.get("y") if isinstance(pose, dict) else None
    th = pose.get("yaw") if isinstance(pose, dict) else None
    return f"Robot at (x={x:.2f}, y={y:.2f}) m, heading ≈ {th:.2f} rad" if x is not None else str(pose)


def _cloud_tool_get_costmap_summary(args: dict) -> str:
    """Brief stats on the accumulated live lidar voxel map."""
    data = _cloud_get("/pointcloud/live") or {}
    if isinstance(data, dict) and data.get("_error"):
        return f"(could not reach /pointcloud/live: {data['_error']})"
    if not data:
        return "No accumulated lidar map yet (pc_bridge may not be running)."
    parts = []
    for robot, info in data.items():
        if not isinstance(info, dict):
            continue
        age = time.time() - (info.get("timestamp") or 0)
        parts.append(
            f"{robot}: {info.get('n', 0)} accumulated lidar voxels, "
            f"voxel size {info.get('voxel_cm', '?')} cm, age {age:.1f}s"
        )
    return "\n".join(parts) if parts else "Accumulated lidar map is empty."


def _cloud_tool_get_pointcloud_stats(args: dict) -> str:
    """How many lidar points the cloud currently has + their z-range."""
    data = _cloud_get("/pointcloud/live") or {}
    if isinstance(data, dict) and data.get("_error"):
        return f"(could not reach /pointcloud/live: {data['_error']})"
    if not data:
        return "No pointcloud yet — pc_bridge may not be running."
    parts = []
    for robot, info in data.items():
        if not isinstance(info, dict):
            continue
        n = info.get("n", 0)
        zmin = (info.get("z_min") or 0) / 100.0
        zmax = (info.get("z_max") or 0) / 100.0
        age = time.time() - (info.get("timestamp") or 0)
        voxel = info.get("voxel_cm")
        voxel_txt = f", voxel {voxel} cm" if voxel else ""
        parts.append(
            f"{robot}: {n} accumulated lidar points{voxel_txt}, "
            f"z in [{zmin:.2f}, {zmax:.2f}] m, age {age:.1f}s"
        )
    return "\n".join(parts) if parts else "Pointcloud state empty."


# Tool spec definitions (Bedrock format) for the cloud-read tools.
# These are appended to the dimos MCP tool list before being handed to the model.
CLOUD_READ_TOOLS_SPEC = [
    {
        "toolSpec": {
            "name": "get_semantic_map",
            "description": (
                "Cloud-side: list every object the robot has detected, with 3D "
                "position, confidence, and how many times it was seen. Read-only."
            ),
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }
    },
    {
        "toolSpec": {
            "name": "get_robot_pose",
            "description": (
                "Cloud-side: current robot position (x, y) in metres and heading "
                "(rad), from the most recent odometry update. Read-only."
            ),
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }
    },
    {
        "toolSpec": {
            "name": "get_costmap_summary",
            "description": (
                "Cloud-side: summary of the accumulated live lidar voxel map. "
                "Use this to verify the pointcloud bridge is alive. Read-only."
            ),
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }
    },
    {
        "toolSpec": {
            "name": "get_pointcloud_stats",
            "description": (
                "Cloud-side: number of accumulated lidar points and z-range "
                "(height) currently in the voxel map, plus age. Read-only."
            ),
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }
    },
]

CLOUD_TOOL_HANDLERS = {
    "get_semantic_map":      _cloud_tool_get_semantic_map,
    "get_robot_pose":        _cloud_tool_get_robot_pose,
    "get_costmap_summary":   _cloud_tool_get_costmap_summary,
    "get_pointcloud_stats":  _cloud_tool_get_pointcloud_stats,
}


def _import_time():
    """Allow `time` module access without polluting the top-level imports."""
    import time as _t
    return _t


# Patch in `time` for _cloud_tool_get_pointcloud_stats (avoid global pollution above)
import time  # noqa: E402


def _filter_tools_for_mode(
    mcp_tools: list[dict], mode: str
) -> tuple[list[dict], list[dict]]:
    """Pick the tool subset and system prompt for a given mode.

    Returns (mcp_tools_subset, system_prompt). The Ask filter is permissive:
    we keep every tool the dimos MCP exposes UNLESS its name matches the
    explicit blocklist (anything that moves the robot or writes state).
    That way new read-only tools added to dimos in the future still show up
    in Ask without having to be added to a whitelist.
    """
    if mode == "ask":
        kept = [t for t in mcp_tools if t["name"] not in ASK_BLOCKED_TOOLS]
        # If anything got through that's clearly an action (description
        # contains the word "move"/"navigate"/"explore" etc.), drop it too.
        action_kw = ("navigate", "move ", "explor", "patrol", "follow",
                     "drive", "walk", "stop ", "halt", "rotate ")
        kept = [t for t in kept
                if not any(k in t.get("description", "").lower()[:120]
                           for k in action_kw)]
        return kept, ASK_SYSTEM_PROMPT
    # Agent mode = unfiltered DimOS MCP tools and no system prompt/rules.
    return mcp_tools, AGENT_SYSTEM_PROMPT


def run_mcp_agent_stream(
    user_message: str,
    mode: str = "agent",
) -> Generator[str, None, None]:
    """Public entrypoint — same shape as agent.run_agent_stream so it drops
    into the existing /query/stream endpoint. Yields text tokens.

    Args:
        user_message: the user's chat text.
        mode: 'agent' (default) → full tool access, can move the robot.
              'ask' → read-only tool subset, never sends commands.

    Failure modes (each yields a human-readable error then returns):
      - MCP server unreachable  → "MCP not reachable, is dimos running?"
      - AWS not configured      → "Bedrock not configured (set AWS_*)"
      - Bedrock invocation fail → traceback hint
    """
    # 1. Connect to dimos MCP and discover tools
    mcp = MCPClient()
    try:
        try:
            mcp.initialize()
        except Exception as e:
            yield (
                f"⚠ Cannot reach dimos MCP at {MCP_URL}.\n"
                f"Make sure `dimos run unitree-go2-agentic` "
                f"(or `--simulation`) is running.\nError: {e}\n"
            )
            return

        try:
            mcp_tools = mcp.list_tools()
        except Exception as e:
            yield f"⚠ Failed to list dimos tools: {e}\n"
            return

        if not mcp_tools:
            yield (
                "⚠ MCP server returned 0 tools. Wait until dimos finishes "
                "booting (look for `Discovered tools from MCP server. n_tools=22`)\n"
            )
            return

        # Ask drops action tools; Agent keeps the complete DimOS MCP tool list.
        mcp_tools, system_prompt = _filter_tools_for_mode(mcp_tools, mode)
        bedrock_tools = _mcp_to_bedrock_tools(mcp_tools)
        # Cloud-side read tools are always exposed — they're pure reads of the
        # cloud server's own state (world_store, lidar map, pointcloud, pose).
        # In Agent mode we keep the toolset as pure DimOS MCP to avoid confusing
        # tool selection; Ask mode gets cloud reads because it is read-only.
        if mode == "ask":
            bedrock_tools = bedrock_tools + CLOUD_READ_TOOLS_SPEC

        # 2. Run the Bedrock conversation
        try:
            yield from _bedrock_stream_with_mcp(
                user_message, mcp, bedrock_tools, system_prompt=system_prompt
            )
        except ImportError:
            yield "⚠ boto3 not installed in this venv. `pip install boto3`.\n"
        except Exception as e:
            logger.exception("Bedrock stream error")
            err = str(e)
            if "ExpiredTokenException" in err or "credentials" in err.lower():
                yield (
                    "⚠ AWS credentials expired or missing.\n"
                    "Refresh AWS_* env vars (or update dimos/.env) and retry.\n"
                    f"Underlying error: {e}\n"
                )
            else:
                yield f"⚠ Bedrock error: {e}\n"
    finally:
        mcp.close()


__all__ = ["run_mcp_agent_stream", "MCPClient"]
