"""
WebSocket Bridge — connects to DimOS WebSocket visualization module (port 7779)
and relays data to/from the FastAPI cloud middleware.

DimOS uses Socket.IO (not plain WebSocket), so this uses python-socketio.
Install: pip install "python-socketio[asyncio_client]"

The DimOS WebsocketVisModule handles:
  - Odometry (robot position) → emits 'robot_pose' event
  - Goal requests (click-to-navigate) ← listens for 'click' event
  - Costmap / path visualization → emits 'path', 'costmap' events
  - Explore commands ← listens for 'start_explore', 'stop_explore'

Usage:
    # Standalone (reads from DimOS WS, pushes to cloud):
    python ws_bridge.py

    # Custom ports:
    python ws_bridge.py --ws-url http://localhost:7779 --cloud-url http://localhost:8080
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from typing import Any

import httpx

try:
    import socketio
    HAS_SOCKETIO = True
except ImportError:
    HAS_SOCKETIO = False

try:
    from dimos.core.transport import pSHMTransport
    from dimos.msgs.sensor_msgs.Image import Image as DimosImage, ImageFormat
    import cv2
    HAS_DIMOS_SHM = True
except ImportError:
    HAS_DIMOS_SHM = False

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Robot State Tracker ───────────────────────────────────────

class RobotState:
    """Tracks the latest robot state from Socket.IO messages."""

    def __init__(self):
        self.odom: dict | None = None   # DimOS format: {"type":"vector","c":[x,y,z]}
        self.path: dict | None = None   # DimOS format: {"type":"path","points":[[x,y],...]}
        self.navigation_state: str = "idle"
        self.goal: dict | None = None
        self.costmap: dict | None = None
        self.last_update: float = 0.0

    def to_cloud_objects(self) -> list[dict]:
        """Convert current state to cloud-compatible object list."""
        objects = []

        if self.odom:
            # DimOS robot_pose format: {"type": "vector", "c": [x, y, z]}
            c = self.odom.get("c", [0, 0, 0])
            objects.append({
                "label": "robot_position",
                "confidence": 1.0,
                "pose": {
                    "x": float(c[0]) if len(c) > 0 else 0.0,
                    "y": float(c[1]) if len(c) > 1 else 0.0,
                    "z": float(c[2]) if len(c) > 2 else 0.0,
                },
                "seen_count": 1,
                "source": "odom",
            })

        if self.goal and self.navigation_state != "idle":
            c = self.goal.get("c", self.goal.get("pos", [0, 0, 0]))
            objects.append({
                "label": f"nav_goal ({self.navigation_state})",
                "confidence": 0.9,
                "pose": {
                    "x": float(c[0]) if len(c) > 0 else 0.0,
                    "y": float(c[1]) if len(c) > 1 else 0.0,
                    "z": float(c[2]) if len(c) > 2 else 0.0,
                },
                "seen_count": 1,
                "source": "planner",
            })

        return objects


# ── Socket.IO Listener ────────────────────────────────────────

async def listen_dimos_ws(
    ws_url: str,
    state: RobotState,
    cloud_url: str,
    robot_id: str,
    push_interval: float = 2.0,
):
    """Connect to DimOS Socket.IO server and listen for state updates."""
    if not HAS_SOCKETIO:
        logger.error("socketio not installed. Run: pip install 'python-socketio[asyncio_client]'")
        return

    # DimOS Socket.IO uses http:// URL, not ws://
    http_url = ws_url.replace("ws://", "http://").replace("wss://", "https://")

    while True:
        sio = socketio.AsyncClient(reconnection=False, logger=False, engineio_logger=False)

        @sio.event
        async def connect():
            logger.info("Connected to DimOS Socket.IO")

        @sio.event
        async def disconnect():
            logger.info("Disconnected from DimOS Socket.IO")

        @sio.event
        async def robot_pose(data):
            state.odom = data  # {"type": "vector", "c": [x, y, z]}
            state.last_update = time.time()
            push_map_to_cloud(cloud_url, robot_id, robot_pose=data)

        @sio.event
        async def path(data):
            state.path = data  # {"type": "path", "points": [[x, y], ...]}
            state.last_update = time.time()
            push_map_to_cloud(cloud_url, robot_id, path=data)

        @sio.event
        async def costmap(data):
            state.costmap = data
            state.last_update = time.time()
            push_map_to_cloud(cloud_url, robot_id, costmap=data)

        @sio.event
        async def color_image(data):
            """Camera frame from DimOS — push to cloud as JPEG."""
            try:
                push_frame_to_cloud(cloud_url, robot_id, data)
            except Exception as e:
                logger.debug(f"Frame push error: {e}")

        @sio.event
        async def image(data):
            """Alternative image event name."""
            try:
                push_frame_to_cloud(cloud_url, robot_id, data)
            except Exception as e:
                logger.debug(f"Frame push error: {e}")

        @sio.event
        async def full_state(data):
            if isinstance(data, dict):
                if "robot_pose" in data:
                    state.odom = data["robot_pose"]
                if "path" in data:
                    state.path = data["path"]
                if "costmap" in data:
                    state.costmap = data["costmap"]
                if "navigation_state" in data:
                    state.navigation_state = str(data["navigation_state"])
                push_map_to_cloud(
                    cloud_url, robot_id,
                    costmap=data.get("costmap"),
                    path=data.get("path"),
                    robot_pose=data.get("robot_pose"),
                )
            state.last_update = time.time()

        @sio.event
        async def navigation_state(data):
            if isinstance(data, str):
                state.navigation_state = data
            elif isinstance(data, dict):
                state.navigation_state = str(data.get("data", "idle"))
            state.last_update = time.time()

        @sio.event
        async def goal_reached(data):
            state.navigation_state = "idle"
            state.goal = None
            state.last_update = time.time()

        try:
            logger.info(f"Connecting to DimOS at {http_url}...")
            await sio.connect(http_url)

            while sio.connected:
                objects = state.to_cloud_objects()
                if objects:
                    push_to_cloud(cloud_url, robot_id, objects)
                await asyncio.sleep(push_interval)

        except Exception as e:
            logger.warning(f"Socket.IO error: {e}. Reconnecting in 3s...")
        finally:
            try:
                await sio.disconnect()
            except Exception:
                pass

        await asyncio.sleep(3)


# ── Send Navigation Goal via Socket.IO ────────────────────────

async def send_goal_via_ws(ws_url: str, x: float, y: float, z: float = 0.0):
    """Send a navigation goal to DimOS via Socket.IO 'click' event."""
    if not HAS_SOCKETIO:
        logger.error("socketio not installed. Run: pip install 'python-socketio[asyncio_client]'")
        return False

    http_url = ws_url.replace("ws://", "http://").replace("wss://", "https://")
    sio = socketio.AsyncClient(logger=False, engineio_logger=False)

    try:
        await sio.connect(http_url)
        # DimOS 'click' event expects [x, y] (2D map coordinates)
        await sio.emit("click", [x, y])
        logger.info(f"Sent goal: ({x:.2f}, {y:.2f})")
        await asyncio.sleep(0.2)
        await sio.disconnect()
        return True
    except Exception as e:
        logger.error(f"Failed to send goal: {e}")
        try:
            await sio.disconnect()
        except Exception:
            pass
        return False


def send_goal_sync(ws_url: str, x: float, y: float, z: float = 0.0) -> bool:
    """Synchronous wrapper for sending a goal."""
    return asyncio.run(send_goal_via_ws(ws_url, x, y, z))


# ── Send Explore Command ──────────────────────────────────────

async def send_explore_command(ws_url: str, start: bool = True):
    """Send start/stop explore command to DimOS."""
    if not HAS_SOCKETIO:
        return False

    http_url = ws_url.replace("ws://", "http://").replace("wss://", "https://")
    sio = socketio.AsyncClient(logger=False, engineio_logger=False)

    event = "start_explore" if start else "stop_explore"
    try:
        await sio.connect(http_url)
        await sio.emit(event)
        logger.info(f"Sent {event}")
        await asyncio.sleep(0.2)
        await sio.disconnect()
        return True
    except Exception as e:
        logger.error(f"Failed to send {event}: {e}")
        try:
            await sio.disconnect()
        except Exception:
            pass
        return False


# ── Cloud Push ────────────────────────────────────────────────

def push_to_cloud(cloud_url: str, robot_id: str, objects: list[dict]) -> bool:
    """Push state to the cloud FastAPI server."""
    try:
        resp = httpx.post(
            f"{cloud_url}/ingest",
            json={"robot_id": robot_id, "objects": objects},
            timeout=2.0,
        )
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def push_map_to_cloud(
    cloud_url: str,
    robot_id: str,
    costmap: dict | None = None,
    path: dict | None = None,
    robot_pose: dict | None = None,
) -> bool:
    """Push live map state (costmap/path/pose) to the cloud server."""
    payload: dict = {"robot_id": robot_id, "timestamp": time.time()}
    if costmap is not None:
        payload["costmap"] = costmap
    if path is not None:
        payload["path"] = path
    if robot_pose is not None:
        payload["robot_pose"] = robot_pose
    try:
        resp = httpx.post(f"{cloud_url}/ingest/map", json=payload, timeout=2.0)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


_last_frame_push = 0.0
_FRAME_INTERVAL = 0.15  # ~6-7 fps max to cloud


def push_frame_to_cloud(cloud_url: str, robot_id: str, data: Any) -> bool:
    """
    Push a camera frame to the cloud.
    Handles various DimOS image formats:
      - Raw bytes (already JPEG)
      - Dict with 'data' key (base64 or raw)
      - Dict with 'image' key
    Throttled to ~7fps to avoid flooding.
    """
    global _last_frame_push
    now = time.time()
    if now - _last_frame_push < _FRAME_INTERVAL:
        return False
    _last_frame_push = now

    frame_bytes = None

    if isinstance(data, bytes):
        frame_bytes = data
    elif isinstance(data, dict):
        # Try common keys
        raw = data.get("data") or data.get("image") or data.get("frame")
        if isinstance(raw, bytes):
            frame_bytes = raw
        elif isinstance(raw, str):
            # Likely base64 encoded
            import base64
            try:
                frame_bytes = base64.b64decode(raw)
            except Exception:
                pass
        # If it has width/height/encoding, it's a raw sensor_msgs.Image
        # Convert to JPEG
        if frame_bytes is None and "width" in data and "height" in data:
            frame_bytes = _raw_image_to_jpeg(data)
    elif isinstance(data, str):
        import base64
        try:
            frame_bytes = base64.b64decode(data)
        except Exception:
            pass

    if frame_bytes is None:
        return False

    try:
        resp = httpx.post(
            f"{cloud_url}/frames",
            content=frame_bytes,
            headers={"Content-Type": "image/jpeg", "X-Robot-Id": robot_id},
            timeout=1.0,
        )
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def _raw_image_to_jpeg(img_data: dict) -> bytes | None:
    """Convert raw image data (sensor_msgs.Image format) to JPEG bytes."""
    try:
        import numpy as np
        from io import BytesIO

        width = img_data["width"]
        height = img_data["height"]
        encoding = img_data.get("encoding", "rgb8")
        raw = img_data.get("data")

        if raw is None:
            return None

        if isinstance(raw, str):
            import base64
            raw = base64.b64decode(raw)

        if isinstance(raw, list):
            raw = bytes(raw)

        arr = np.frombuffer(raw, dtype=np.uint8)

        if encoding in ("rgb8", "RGB8"):
            arr = arr.reshape((height, width, 3))
        elif encoding in ("bgr8", "BGR8"):
            arr = arr.reshape((height, width, 3))
            arr = arr[:, :, ::-1]  # BGR to RGB
        elif encoding in ("rgba8", "RGBA8"):
            arr = arr.reshape((height, width, 4))[:, :, :3]
        elif encoding in ("mono8", "MONO8"):
            arr = arr.reshape((height, width))
        else:
            # Try as 3-channel
            if len(arr) == height * width * 3:
                arr = arr.reshape((height, width, 3))
            else:
                return None

        # Encode to JPEG
        from PIL import Image
        img = Image.fromarray(arr)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return buf.getvalue()

    except ImportError:
        logger.debug("numpy/PIL not available for raw image conversion")
        return None
    except Exception as e:
        logger.debug(f"Image conversion error: {e}")
        return None


# ── Camera via DimOS pSHM ─────────────────────────────────────

_cam_last_push = 0.0
_CAM_MIN_INTERVAL = 0.2  # 5 fps max


async def stream_camera_from_shm(cloud_url: str, robot_id: str, topic: str = "color_image"):
    """
    Subscribe to DimOS color_image pSHM channel and forward JPEG frames to cloud.
    Only active when the dimos package is importable (i.e., running in dimos venv).
    """
    if not HAS_DIMOS_SHM:
        logger.debug("DimOS SHM not available — camera stream disabled")
        return

    global _cam_last_push
    transport = pSHMTransport(topic)
    latest: list = [None]  # mutable holder for latest frame

    def _on_frame(img):
        latest[0] = img

    try:
        transport.subscribe(_on_frame)  # subscribe() calls start() internally
        logger.info(f"Camera SHM subscriber started (topic={topic})")
    except Exception as e:
        logger.warning(f"Camera SHM subscribe failed: {e}")
        return

    try:
        while True:
            img = latest[0]
            if img is not None:
                now = time.time()
                if now - _cam_last_push >= _CAM_MIN_INTERVAL:
                    _cam_last_push = now
                    latest[0] = None
                    try:
                        arr = img.data  # numpy array, BGR format
                        if img.format == ImageFormat.RGB:
                            arr = arr[:, :, ::-1]  # RGB→BGR for cv2
                        ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 72])
                        if ok:
                            frame_bytes = buf.tobytes()
                            try:
                                httpx.post(
                                    f"{cloud_url}/frames",
                                    content=frame_bytes,
                                    headers={
                                        "Content-Type": "image/jpeg",
                                        "X-Robot-Id": robot_id,
                                    },
                                    timeout=1.0,
                                )
                            except (httpx.ConnectError, httpx.TimeoutException):
                                pass
                    except Exception as e:
                        logger.debug(f"Camera frame encode error: {e}")
            await asyncio.sleep(0.05)
    finally:
        try:
            transport.stop()
        except Exception:
            pass


# ── Main ──────────────────────────────────────────────────────

async def poll_goals(cloud_url: str, ws_url: str):
    """Poll EC2 for queued navigation goals and forward them to DimOS."""
    while True:
        try:
            resp = httpx.get(f"{cloud_url}/goals/pending", timeout=2.0)
            if resp.status_code == 200:
                for goal in resp.json().get("goals", []):
                    logger.info(f"Forwarding goal from cloud: ({goal['x']:.2f}, {goal['y']:.2f})")
                    await send_goal_via_ws(ws_url, goal["x"], goal["y"])
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        except Exception as e:
            logger.debug(f"Goal poll error: {e}")
        await asyncio.sleep(1.0)


async def main_async(args):
    state = RobotState()

    tasks = [
        listen_dimos_ws(
            ws_url=args.ws_url,
            state=state,
            cloud_url=args.cloud_url,
            robot_id=args.robot_id,
            push_interval=args.interval,
        ),
        poll_goals(args.cloud_url, args.ws_url),
        stream_camera_from_shm(args.cloud_url, args.robot_id),
    ]
    await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser(
        description="Socket.IO bridge: DimOS visualization → cloud API"
    )
    parser.add_argument(
        "--ws-url", default="http://localhost:7779",
        help="DimOS Socket.IO URL (default: http://localhost:7779)"
    )
    parser.add_argument(
        "--cloud-url",
        default=os.environ.get("CLOUD_URL", "http://localhost:8080"),
        help="Cloud FastAPI server URL (default: $CLOUD_URL or http://localhost:8080)"
    )
    parser.add_argument(
        "--robot-id", default="go2_a",
        help="Robot identifier (default: go2_a)"
    )
    parser.add_argument(
        "--interval", type=float, default=2.0,
        help="Push interval in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--send-goal", nargs=2, type=float, metavar=("X", "Y"),
        help="Send a navigation goal and exit (e.g., --send-goal 2.0 1.5)"
    )
    parser.add_argument(
        "--explore", action="store_true",
        help="Send start_explore and exit"
    )
    parser.add_argument(
        "--stop-explore", action="store_true",
        help="Send stop_explore and exit"
    )
    args = parser.parse_args()

    if args.send_goal:
        x, y = args.send_goal
        ok = send_goal_sync(args.ws_url, x, y)
        print(f"Goal sent: ({x}, {y})" if ok else "Failed to send goal")
        return 0 if ok else 1

    if args.explore:
        ok = asyncio.run(send_explore_command(args.ws_url, start=True))
        print("Explore started" if ok else "Failed")
        return 0 if ok else 1

    if args.stop_explore:
        ok = asyncio.run(send_explore_command(args.ws_url, start=False))
        print("Explore stopped" if ok else "Failed")
        return 0 if ok else 1

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nBridge stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
