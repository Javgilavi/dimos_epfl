"""
Direct-LCM nav bridge — replaces the old ws_bridge.py / Socket.IO path.

Two responsibilities:

  1. Subscribe to dimos's `/odom#geometry_msgs.PoseStamped` LCM channel and
     POST the latest pose to `/ingest/pose` at ~15 Hz.

  2. Poll the cloud's `/goals/pending` queue. Whenever a click-to-navigate
     goal is queued by the dashboard, publish a corresponding
     `geometry_msgs.PoseStamped` to dimos's `/goal_request` LCM channel.

Must run inside the dimos venv. Auto-launched by main.py's lifespan.
Standalone:

    python nav_bridge.py --cloud-url http://localhost:8080
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import threading
import time

import requests

logging.basicConfig(level=logging.INFO, format="[nav_bridge] %(message)s")
log = logging.getLogger(__name__)

POSE_CHANNEL = os.environ.get("NAV_POSE_CHANNEL", "/odom#geometry_msgs.PoseStamped")
GOAL_CHANNEL = os.environ.get("NAV_GOAL_CHANNEL", "/goal_request#geometry_msgs.PoseStamped")

DEFAULT_CLOUD = os.environ.get("CLOUD_URL", "http://localhost:8080")
DEFAULT_ROBOT_ID = os.environ.get("ROBOT_ID", "go2_a")
DEFAULT_POSE_HZ = float(os.environ.get("NAV_POSE_HZ", "15"))
DEFAULT_GOAL_POLL_HZ = float(os.environ.get("NAV_GOAL_POLL_HZ", "5"))
_BRIDGE_PW = os.environ.get("BRIDGE_PASSWORD", "")


# ── Pose subscriber → cloud push ─────────────────────────────

def _bridge_headers(extra: dict | None = None) -> dict:
    h = dict(extra or {})
    if _BRIDGE_PW:
        h["X-Bridge-Password"] = _BRIDGE_PW
    return h


def _push_pose(cloud_url: str, robot_id: str, pose_dict: dict) -> bool:
    try:
        r = requests.post(
            f"{cloud_url}/ingest/pose",
            json=pose_dict,
            headers=_bridge_headers({"X-Robot-Id": robot_id}),
            timeout=1.5,
        )
        return r.ok
    except Exception:
        return False


def run_pose_subscriber(cloud_url: str, robot_id: str, hz: float, lc):
    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

    interval = 1.0 / max(1.0, hz)
    last_push = 0.0
    latest = [None]

    # Delta-yaw estimation — derive heading from position delta rather than
    # trusting msg.yaw directly (LCM odom convention differs from Three.js).
    # EMA smoothing on the angle prevents jarring snaps; wraparound is handled
    # via atan2(sin(diff), cos(diff)) so 359°→1° doesn't jump through 180°.
    _MOVE_THRESH = 0.05   # metres — ignore deltas smaller than this (vibration filter)
    _SMOOTH      = 0.20   # EMA alpha — higher = snappier, lower = smoother
    _prev_xy: list[tuple[float, float] | None] = [None]
    _heading: list[float] = [0.0]

    def _on_msg(channel, data):
        try:
            latest[0] = PoseStamped.lcm_decode(data)
        except Exception as e:
            log.debug(f"pose decode error: {e}")

    lc.subscribe(POSE_CHANNEL, _on_msg)
    log.info(f"subscribed to {POSE_CHANNEL} (push @ {hz:.1f} Hz)")

    while True:
        lc.handle_timeout(50)
        msg = latest[0]
        now = time.time()
        if msg is None or now - last_push < interval:
            continue
        last_push = now
        try:
            x, y = float(msg.x), float(msg.y)

            # Update heading from position delta when movement is large enough.
            if _prev_xy[0] is not None:
                dx = x - _prev_xy[0][0]
                dy = y - _prev_xy[0][1]
                if math.sqrt(dx * dx + dy * dy) > _MOVE_THRESH:
                    raw = math.atan2(dy, dx)
                    # Shortest-path angle difference (handles ±π wraparound)
                    diff = math.atan2(math.sin(raw - _heading[0]),
                                      math.cos(raw - _heading[0]))
                    _heading[0] += _SMOOTH * diff
            _prev_xy[0] = (x, y)

            pose = {
                "x": x,
                "y": y,
                "z": float(msg.z),
                "yaw": _heading[0],   # delta-derived heading for dashboard
                "pitch": float(msg.pitch),
                "roll":  float(msg.roll),
                # raw quaternion kept for any consumer that wants the full orientation
                "qx": float(msg.orientation.x),
                "qy": float(msg.orientation.y),
                "qz": float(msg.orientation.z),
                "qw": float(msg.orientation.w),
                "ts": now,
            }
            _push_pose(cloud_url, robot_id, pose)
        except Exception as e:
            log.debug(f"pose serialization error: {e}")


# ── Goal poller → dimos LCM publisher ─────────────────────────

def _build_pose_stamped(x: float, y: float, z: float = 0.0):
    """Construct a PoseStamped pointing forward (identity quaternion)."""
    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
    from dimos.msgs.geometry_msgs.Vector3 import Vector3
    from dimos.msgs.geometry_msgs.Quaternion import Quaternion
    p = PoseStamped()
    p.position = Vector3(x, y, z)
    p.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    p.frame_id = "map"
    p.ts = time.time()
    return p


def run_goal_poller(cloud_url: str, robot_id: str, hz: float, lc):
    interval = 1.0 / max(0.5, hz)
    log.info(f"polling {cloud_url}/goals/pending @ {hz:.1f} Hz; "
             f"publishing → {GOAL_CHANNEL}")
    while True:
        time.sleep(interval)
        try:
            r = requests.get(f"{cloud_url}/goals/pending",
                             headers=_bridge_headers(), timeout=1.5)
            if not r.ok:
                continue
            data = r.json()
        except Exception:
            continue

        goals = data if isinstance(data, list) else data.get("goals", [])
        for g in goals:
            try:
                if g.get("type") == "stop":
                    # No-op for now; dimos exposes stop_navigation via MCP.
                    log.info("(stop goal received — leaving to MCP/UI to handle)")
                    continue
                if g.get("type") == "explore":
                    # Skip — exploration is also better triggered via MCP.
                    log.info("(explore goal received — leaving to MCP)")
                    continue
                x = float(g.get("x", 0.0))
                y = float(g.get("y", 0.0))
                z = float(g.get("z", 0.0))
                msg = _build_pose_stamped(x, y, z)
                lc.publish(GOAL_CHANNEL, msg.lcm_encode())
                log.info(f"published goal_request → ({x:.2f}, {y:.2f})m")
            except Exception as e:
                log.warning(f"failed to publish goal {g}: {e}")


# ── Entrypoint ───────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Direct-LCM pose + goal bridge")
    p.add_argument("--cloud-url", default=DEFAULT_CLOUD)
    p.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    p.add_argument("--pose-hz", type=float, default=DEFAULT_POSE_HZ)
    p.add_argument("--goal-hz", type=float, default=DEFAULT_GOAL_POLL_HZ)
    args = p.parse_args()

    try:
        import lcm as lcmlib
    except ImportError:
        raise RuntimeError("lcm not importable — run inside the dimos venv")

    lc = lcmlib.LCM()
    log.info(f"cloud={args.cloud_url} robot={args.robot_id}")

    # Run goal poller in a background thread; pose subscription on main thread
    # because lc.handle_timeout() needs to monopolise the LCM file descriptor.
    t = threading.Thread(
        target=run_goal_poller,
        args=(args.cloud_url, args.robot_id, args.goal_hz, lc),
        daemon=True,
        name="goal-poller",
    )
    t.start()

    try:
        run_pose_subscriber(args.cloud_url, args.robot_id, args.pose_hz, lc)
    except KeyboardInterrupt:
        log.info("stopped")


if __name__ == "__main__":
    main()
