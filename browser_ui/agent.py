"""
Bedrock Agent Loop — Agentic streaming with tool_use.

Uses converse_stream API with Claude for natural language queries
against the robot's world state.

When AWS credentials are not configured, falls back to a simple
local response mode that reads the map and answers directly.
"""

import os
import json
import logging
from collections.abc import Generator

from models import DetectedObject
from world_store import WorldStateStore

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────

AWS_REGION = os.environ.get("AWS_REGION", "eu-west-1")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "eu.anthropic.claude-sonnet-4-6")
MAX_TOKENS = int(os.environ.get("BEDROCK_MAX_TOKENS", "500"))
USE_BEDROCK = os.environ.get("USE_BEDROCK", "false").lower() == "true"

# ── Tools Definition ──────────────────────────────────────────

TOOLS = [
    {
        "toolSpec": {
            "name": "get_semantic_map",
            "description": "Get all objects the robot detected with 3D positions and confidence.",
            "inputSchema": {"json": {"type": "object", "properties": {}}}
        }
    },
    {
        "toolSpec": {
            "name": "navigate_to_object",
            "description": "Send robot to navigate toward a detected object by label.",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Object label to navigate to"}
                },
                "required": ["label"]
            }}
        }
    },
    {
        "toolSpec": {
            "name": "stop_robot",
            "description": "Stop the robot immediately.",
            "inputSchema": {"json": {"type": "object", "properties": {}}}
        }
    }
]

SYSTEM_PROMPT = [{
    "text": (
        "You are the brain of a robot dog (Unitree Go2) running DimOS. "
        "Use tools to query the semantic map and send navigation commands. "
        "Be concise. Confirm navigation actions. "
        "If the map is empty, say so clearly."
    )
}]


# ── Tool Execution ────────────────────────────────────────────

def execute_tool(name: str, inputs: dict, world_store: WorldStateStore) -> str:
    """Execute a tool and return the result as text."""
    if name == "get_semantic_map":
        objects = world_store.merge()
        if not objects:
            return "No objects detected yet. The robot hasn't reported any detections."
        return "\n".join(
            f"- {o.label} at ({o.pose.x:.1f}, {o.pose.y:.1f})m "
            f"confidence {o.confidence:.0%} seen {o.seen_count}x"
            for o in objects
        )

    if name == "navigate_to_object":
        label = inputs.get("label", "unknown")
        # Find the object in world state
        objects = world_store.merge()
        target = next((o for o in objects if o.label.lower() == label.lower()), None)
        if target:
            return (
                f"Navigation command sent: go to {target.label} "
                f"at ({target.pose.x:.1f}, {target.pose.y:.1f})m"
            )
        return f"Object '{label}' not found in current map."

    if name == "stop_robot":
        return "Stop command sent. Robot halting."

    return f"Unknown tool: {name}"


# ── Fallback Mode (no Bedrock) ────────────────────────────────

def _local_response(user_message: str, world_store: WorldStateStore) -> Generator[str, None, None]:
    """Simple local response when Bedrock is not configured."""
    objects = world_store.merge()

    if not objects:
        yield "No objects detected yet. The robot hasn't reported any detections to the cloud."
        return

    # Simple keyword matching for demo
    msg_lower = user_message.lower()

    # Check if asking about a specific object
    for obj in objects:
        if obj.label.lower() in msg_lower:
            yield (
                f"The {obj.label} is at position ({obj.pose.x:.1f}, {obj.pose.y:.1f})m "
                f"with {obj.confidence:.0%} confidence (seen {obj.seen_count}x)."
            )
            return

    # General map summary
    yield "Here's what the robot sees:\n"
    for obj in objects:
        yield (
            f"- {obj.label} at ({obj.pose.x:.1f}, {obj.pose.y:.1f})m "
            f"({obj.confidence:.0%} confidence)\n"
        )


# ── Bedrock Agentic Loop ─────────────────────────────────────

def _bedrock_stream(user_message: str, world_store: WorldStateStore) -> Generator[str, None, None]:
    """Full agentic loop using Bedrock converse_stream with tool_use."""
    import boto3

    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    messages = [{"role": "user", "content": [{"text": user_message}]}]

    while True:
        resp = bedrock.converse_stream(
            modelId=MODEL_ID,
            system=SYSTEM_PROMPT,
            messages=messages,
            toolConfig={"tools": TOOLS},
            inferenceConfig={"maxTokens": MAX_TOKENS},
        )

        # Accumulate response while streaming text
        assistant_content = []
        current_tool = {}
        stop_reason = None

        for event in resp["stream"]:
            if "contentBlockStart" in event:
                block = event["contentBlockStart"].get("start", {})
                tool_use = block.get("toolUse", {})
                if tool_use:
                    current_tool = {
                        "toolUseId": tool_use.get("toolUseId"),
                        "name": tool_use.get("name"),
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
                    tool_input = json.loads(current_tool.pop("input_str") or "{}")
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

        messages.append({"role": "assistant", "content": assistant_content})

        if stop_reason != "tool_use":
            break

        # Execute tools, feed results back, loop
        tool_results = []
        for block in assistant_content:
            if "toolUse" in block:
                tool = block["toolUse"]
                yield f"\n[{tool['name']}...]\n"
                output = execute_tool(tool["name"], tool.get("input", {}), world_store)
                tool_results.append({
                    "toolResult": {
                        "toolUseId": tool["toolUseId"],
                        "content": [{"text": output}],
                    }
                })
        messages.append({"role": "user", "content": tool_results})


# ── Public Interface ──────────────────────────────────────────

def run_agent_stream(user_message: str, world_store: WorldStateStore) -> Generator[str, None, None]:
    """
    Main entry point. Yields text tokens.

    Uses Bedrock when USE_BEDROCK=true and credentials are available.
    Falls back to local response mode otherwise.
    """
    if USE_BEDROCK:
        try:
            yield from _bedrock_stream(user_message, world_store)
        except Exception as e:
            logger.error(f"Bedrock error, falling back to local: {e}")
            yield f"[Bedrock unavailable: {e}]\n\n"
            yield from _local_response(user_message, world_store)
    else:
        yield from _local_response(user_message, world_store)
