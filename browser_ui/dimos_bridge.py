"""
DimOS Bridge — reads real perception data from a running DimOS simulation
and pushes it to the cloud FastAPI server.

This script connects to the DimOS MCP server (default: localhost:9990/mcp),
discovers available perception tools, and periodically reads + pushes
whatever the robot actually sees.

Usage:
    # Start DimOS simulation first, then:
    python dimos_bridge.py

    # Custom URLs:
    python dimos_bridge.py --mcp-url http://localhost:9990/mcp \
                           --cloud-url http://localhost:8080 \
                           --robot-id go2_a

    # Just list what tools DimOS has (no pushing):
    python dimos_bridge.py --list-tools
"""

import argparse
import json
import logging
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
from typing import Any

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── DimOS MCP Client (minimal, no go2_dimos dependency) ──────

class McpClient:
    """Minimal JSON-RPC client for DimOS MCP server."""

    def __init__(self, url: str = "http://localhost:9990/mcp", timeout: float = 10.0):
        self.url = url
        self.timeout = timeout
        self._next_id = 0
        self._initialized = False

    def _request(self, method: str, params: dict | None = None) -> Any:
        self._next_id += 1
        payload = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params:
            payload["params"] = params

        body = json.dumps(payload).encode()
        req = Request(self.url, data=body, headers={"Content-Type": "application/json"})

        try:
            with urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except URLError as e:
            raise ConnectionError(f"Cannot reach DimOS MCP at {self.url}: {e}") from e

        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")
        return data.get("result")

    def initialize(self):
        if not self._initialized:
            self._request("initialize")
            self._initialized = True

    def list_tools(self) -> list[dict]:
        self.initialize()
        result = self._request("tools/list")
        return result.get("tools", []) if isinstance(result, dict) else []

    def call_tool(self, name: str, arguments: dict | None = None) -> Any:
        self.initialize()
        return self._request("tools/call", {"name": name, "arguments": arguments or {}})


# ── Perception Data Extraction ────────────────────────────────

# Known DimOS tools that return perception/spatial data
PERCEPTION_TOOLS = [
    "get_spatial_memory",
    "get_detections",
    "get_objects",
    "get_scene",
    "list_detected_objects",
    "get_detected_objects",
    "query_spatial_memory",
]

# Tools that might return camera frames
CAMERA_TOOLS = [
    "get_camera_frame",
    "get_image",
    "capture_frame",
]


def discover_perception_tools(mcp: McpClient) -> list[str]:
    """Find which perception tools are available on this DimOS instance."""
    all_tools = mcp.list_tools()
    tool_names = [t.get("name", "") for t in all_tools]

    # Match known perception tools
    found = [name for name in tool_names if name in PERCEPTION_TOOLS]

    # Also grab anything with "detect", "object", "spatial", "memory", "scene" in the name
    for name in tool_names:
        lower = name.lower()
        if any(kw in lower for kw in ("detect", "object", "spatial", "memory", "scene", "perceive")):
            if name not in found:
                found.append(name)

    return found


def discover_camera_tools(mcp: McpClient) -> list[str]:
    """Find which camera/image tools are available."""
    all_tools = mcp.list_tools()
    tool_names = [t.get("name", "") for t in all_tools]

    found = [name for name in tool_names if name in CAMERA_TOOLS]
    for name in tool_names:
        lower = name.lower()
        if any(kw in lower for kw in ("camera", "image", "frame", "capture")):
            if name not in found:
                found.append(name)

    return found


def extract_objects_from_result(result: Any) -> list[dict]:
    """
    Try to extract a list of detected objects from a tool result.
    DimOS tools return various formats — this handles common ones.
    """
    if result is None:
        return []

    # If result is already a list of objects
    if isinstance(result, list):
        return [normalize_object(obj) for obj in result if isinstance(obj, dict)]

    # If result is a dict with a list inside
    if isinstance(result, dict):
        # Try common keys
        for key in ("objects", "detections", "items", "results", "content"):
            if key in result and isinstance(result[key], list):
                return [normalize_object(obj) for obj in result[key] if isinstance(obj, dict)]

        # If it has 'content' as a list with text items (MCP format)
        if "content" in result and isinstance(result["content"], list):
            for item in result["content"]:
                if isinstance(item, dict) and item.get("type") == "text":
                    # Try to parse the text as JSON
                    try:
                        parsed = json.loads(item["text"])
                        if isinstance(parsed, list):
                            return [normalize_object(obj) for obj in parsed if isinstance(obj, dict)]
                        if isinstance(parsed, dict):
                            return extract_objects_from_result(parsed)
                    except (json.JSONDecodeError, TypeError):
                        # Try to parse as line-separated objects
                        return parse_text_detections(item["text"])

        # Single object
        if "label" in result or "name" in result:
            return [normalize_object(result)]

    # If result is a string, try to parse it
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            return extract_objects_from_result(parsed)
        except json.JSONDecodeError:
            return parse_text_detections(result)

    return []


def parse_text_detections(text: str) -> list[dict]:
    """Parse text-format detections (e.g., '- chair at (1.2, 0.3) confidence 85%')."""
    objects = []
    for line in text.strip().split("\n"):
        line = line.strip().lstrip("- ")
        if not line:
            continue
        # Try to extract label and position from common formats
        parts = line.split(" at ")
        if len(parts) >= 2:
            label = parts[0].strip()
            # Try to extract coordinates
            rest = parts[1]
            pose = {"x": 0.0, "y": 0.0, "z": 0.0}
            try:
                # Format: (x, y) or (x, y, z)
                coords_str = rest.split(")")[0].lstrip("(")
                coords = [float(c.strip()) for c in coords_str.split(",")]
                if len(coords) >= 2:
                    pose["x"] = coords[0]
                    pose["y"] = coords[1]
                if len(coords) >= 3:
                    pose["z"] = coords[2]
            except (ValueError, IndexError):
                pass

            confidence = 0.5
            if "confidence" in rest.lower():
                try:
                    conf_part = rest.lower().split("confidence")[1].strip()
                    conf_val = float(conf_part.rstrip("%").strip()) 
                    if conf_val > 1:
                        conf_val /= 100.0
                    confidence = conf_val
                except (ValueError, IndexError):
                    pass

            objects.append({
                "label": label,
                "confidence": confidence,
                "pose": pose,
                "seen_count": 1,
                "source": "dimos",
            })
    return objects


def normalize_object(obj: dict) -> dict:
    """Normalize a detection dict to the cloud API format."""
    label = obj.get("label") or obj.get("name") or obj.get("class") or "unknown"
    confidence = obj.get("confidence") or obj.get("score") or 0.5
    if isinstance(confidence, str):
        confidence = float(confidence.rstrip("%")) / 100.0

    # Handle various pose formats
    pose = obj.get("pose") or obj.get("position") or obj.get("location") or {}
    if isinstance(pose, (list, tuple)) and len(pose) >= 2:
        pose = {"x": float(pose[0]), "y": float(pose[1]), "z": float(pose[2]) if len(pose) > 2 else 0.0}
    elif isinstance(pose, dict):
        pose = {
            "x": float(pose.get("x", 0.0)),
            "y": float(pose.get("y", 0.0)),
            "z": float(pose.get("z", 0.0)),
        }
    else:
        pose = {"x": 0.0, "y": 0.0, "z": 0.0}

    return {
        "label": str(label),
        "confidence": float(confidence),
        "pose": pose,
        "seen_count": int(obj.get("seen_count", obj.get("count", 1))),
        "source": str(obj.get("source", "dimos")),
    }


# ── Cloud Push ────────────────────────────────────────────────

def push_to_cloud(cloud_url: str, robot_id: str, objects: list[dict]) -> bool:
    """Push detections to the cloud FastAPI server."""
    try:
        resp = httpx.post(
            f"{cloud_url}/ingest",
            json={"robot_id": robot_id, "objects": objects},
            timeout=3.0,
        )
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        logger.debug(f"Cloud push failed: {e}")
        return False


# ── Main Loop ─────────────────────────────────────────────────

def run_bridge(
    mcp_url: str = "http://localhost:9990/mcp",
    cloud_url: str = "http://localhost:8080",
    robot_id: str = "go2_a",
    interval: float = 3.0,
):
    """Main bridge loop: read from DimOS, push to cloud."""
    mcp = McpClient(url=mcp_url)

    # Connect and discover tools
    logger.info(f"Connecting to DimOS MCP at {mcp_url}...")
    try:
        tools = mcp.list_tools()
        logger.info(f"Connected. {len(tools)} tools available.")
    except ConnectionError as e:
        logger.error(f"Cannot connect to DimOS: {e}")
        logger.error("Make sure DimOS simulation is running first.")
        return 1

    # Find perception tools
    perception_tools = discover_perception_tools(mcp)
    if perception_tools:
        logger.info(f"Perception tools found: {perception_tools}")
    else:
        logger.warning("No known perception tools found. Will try all tools for object data.")
        # Fallback: try tools that might return useful data
        perception_tools = [t.get("name") for t in tools if t.get("name")]

    # Check cloud connectivity
    try:
        resp = httpx.get(f"{cloud_url}/health", timeout=2)
        if resp.status_code == 200:
            logger.info(f"Cloud server reachable at {cloud_url}")
        else:
            logger.warning(f"Cloud returned {resp.status_code} — will retry on push")
    except Exception:
        logger.warning(f"Cloud not reachable at {cloud_url} — will retry on push")

    # Main loop
    logger.info(f"Starting bridge loop (interval={interval}s, robot_id={robot_id})")
    logger.info("Press Ctrl+C to stop\n")

    consecutive_empty = 0

    while True:
        all_objects = []

        for tool_name in perception_tools:
            try:
                result = mcp.call_tool(tool_name)
                objects = extract_objects_from_result(result)
                if objects:
                    all_objects.extend(objects)
                    logger.debug(f"  {tool_name} → {len(objects)} objects")
            except Exception as e:
                logger.debug(f"  {tool_name} failed: {e}")

        if all_objects:
            consecutive_empty = 0
            # Deduplicate by label (keep highest confidence)
            seen = {}
            for obj in all_objects:
                key = obj["label"]
                if key not in seen or obj["confidence"] > seen[key]["confidence"]:
                    seen[key] = obj
            deduped = list(seen.values())

            ok = push_to_cloud(cloud_url, robot_id, deduped)
            status = "✓ pushed" if ok else "✗ cloud unreachable"
            logger.info(f"[{time.strftime('%H:%M:%S')}] {len(deduped)} objects {status}")
            for obj in deduped:
                logger.info(f"    {obj['label']} ({obj['confidence']:.0%}) at ({obj['pose']['x']:.1f}, {obj['pose']['y']:.1f})")
        else:
            consecutive_empty += 1
            if consecutive_empty <= 3 or consecutive_empty % 10 == 0:
                logger.info(f"[{time.strftime('%H:%M:%S')}] No objects detected (waiting...)")

        time.sleep(interval)


# ── Entry Point ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bridge: reads DimOS perception → pushes to cloud API"
    )
    parser.add_argument(
        "--mcp-url", default="http://localhost:9990/mcp",
        help="DimOS MCP server URL (default: http://localhost:9990/mcp)"
    )
    parser.add_argument(
        "--cloud-url", default="http://localhost:8080",
        help="Cloud FastAPI server URL (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--robot-id", default="go2_a",
        help="Robot identifier (default: go2_a)"
    )
    parser.add_argument(
        "--interval", type=float, default=3.0,
        help="Polling interval in seconds (default: 3.0)"
    )
    parser.add_argument(
        "--list-tools", action="store_true",
        help="Just list available DimOS tools and exit"
    )
    args = parser.parse_args()

    if args.list_tools:
        mcp = McpClient(url=args.mcp_url)
        try:
            tools = mcp.list_tools()
            print(f"\nDimOS MCP tools ({len(tools)} total):\n")
            for t in tools:
                name = t.get("name", "<unnamed>")
                desc = t.get("description", "")[:60]
                print(f"  {name:30s} {desc}")
            print()

            perception = discover_perception_tools(mcp)
            if perception:
                print(f"Perception tools: {perception}")
            camera = discover_camera_tools(mcp)
            if camera:
                print(f"Camera tools: {camera}")
        except ConnectionError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return 0

    try:
        return run_bridge(
            mcp_url=args.mcp_url,
            cloud_url=args.cloud_url,
            robot_id=args.robot_id,
            interval=args.interval,
        )
    except KeyboardInterrupt:
        print("\nBridge stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
