"""
Pointcloud bridge — subscribes to dimos's lidar LCM channel
(`/lidar#sensor_msgs.PointCloud2`), downsamples, packs as int16 cm triplets
and POSTs to the FastAPI server's `/ingest/pointcloud` endpoint.

With it running, main.py accumulates the live lidar points into the dashboard's
3D voxel map.

Must run inside the dimos venv (needs `lcm` and `dimos.msgs`).
Auto-launched by main.py's lifespan; can also be run standalone:

    python pc_bridge.py --cloud-url http://localhost:8080 --fps 5
"""

from __future__ import annotations

import argparse
import logging
import os
import struct
import time
import zlib

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="[pc_bridge] %(message)s")
log = logging.getLogger(__name__)

# dimos publishes two pointcloud streams:
#   /lidar       — raw per-frame scan (~8 K pts, lively, what the user sees as "the lidar")
#   /global_map  — accumulated voxel grid built by VoxelGridMapper (denser, persistent)
# Default to /lidar — matches the user's request "show me the lidar pointcloud".
LCM_CHANNEL = os.environ.get("PC_BRIDGE_CHANNEL", "/lidar#sensor_msgs.PointCloud2")

DEFAULT_CLOUD = os.environ.get("CLOUD_URL", "http://localhost:8080")
DEFAULT_ROBOT_ID = os.environ.get("ROBOT_ID", "go2_a")
DEFAULT_FPS = int(os.environ.get("PC_BRIDGE_FPS", "4"))
MAX_POINTS = int(os.environ.get("PC_BRIDGE_MAX_POINTS", "20000"))
_BRIDGE_PW = os.environ.get("BRIDGE_PASSWORD", "")


def _push(cloud_url: str, robot_id: str, payload: bytes) -> bool:
    headers = {
        "Content-Type": "application/octet-stream",
        "X-Robot-Id": robot_id,
    }
    if _BRIDGE_PW:
        headers["X-Bridge-Password"] = _BRIDGE_PW
    try:
        r = requests.post(
            f"{cloud_url}/ingest/pointcloud",
            data=payload,
            headers=headers,
            timeout=2.0,
        )
        return r.ok
    except Exception as e:
        log.debug(f"push failed: {e}")
        return False


def _encode_pointcloud(xyz: np.ndarray) -> bytes:
    """Pack N×3 float xyz (metres) into int16 cm triplets, prefixed with N (int32 LE),
    then zlib-compressed. Matches /ingest/pointcloud's expected layout.
    """
    if xyz.size == 0:
        return b""
    pts_cm = np.clip(xyz * 100.0, -32767, 32767).astype("<i2")  # little-endian int16
    n = pts_cm.shape[0]
    raw = struct.pack("<i", n) + pts_cm.tobytes()
    return zlib.compress(raw, level=4)


def _downsample(xyz: np.ndarray, max_points: int) -> np.ndarray:
    if xyz.shape[0] <= max_points:
        return xyz
    idx = np.random.choice(xyz.shape[0], max_points, replace=False)
    return xyz[idx]


def run_lcm(cloud_url: str, robot_id: str, fps: int, max_points: int):
    try:
        import lcm as lcmlib
        from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
    except ImportError as e:
        raise RuntimeError(
            f"dimos / lcm not importable ({e}) — run inside the dimos venv"
        )

    log.info(f"subscribing to LCM channel '{LCM_CHANNEL}'")
    lc = lcmlib.LCM()
    latest = [None]

    def _on_msg(channel, data):
        try:
            latest[0] = PointCloud2.lcm_decode(data)
        except Exception as e:
            log.debug(f"decode error: {e}")

    lc.subscribe(LCM_CHANNEL, _on_msg)
    log.info("subscribed; waiting for pointcloud frames...")

    interval = 1.0 / max(1, fps)
    last_push = 0.0
    empty = 0

    pushes = 0
    while True:
        lc.handle_timeout(80)
        msg = latest[0]
        now = time.time()
        if msg is None or now - last_push < interval:
            empty += 1
            if empty == 200:
                log.warning(
                    "no lidar frames yet — is dimos running with lidar enabled? "
                    "(check `dimos --simulation run unitree-go2-agentic` is up)"
                )
            continue
        latest[0] = None
        last_push = now
        empty = 0

        try:
            xyz = msg.points_f32()  # (N, 3) float32 metres
            if xyz.ndim != 2 or xyz.shape[1] < 3:
                log.warning(f"unexpected pointcloud shape: {xyz.shape}")
                continue
            xyz = xyz[:, :3].astype("f4", copy=False)
            mask = np.isfinite(xyz).all(axis=1)
            xyz = xyz[mask]
            if xyz.size == 0:
                continue
            xyz = _downsample(xyz, max_points)
            payload = _encode_pointcloud(xyz)
            if not payload:
                continue
            ok = _push(cloud_url, robot_id, payload)
            pushes += 1
            if pushes <= 3 or pushes % 30 == 0:
                log.info(
                    f"push #{pushes}: {xyz.shape[0]} pts "
                    f"({len(payload)//1024} KB) {'ok' if ok else 'FAIL'}"
                )
        except Exception as e:
            log.warning(f"frame error: {e}")


def main():
    p = argparse.ArgumentParser(description="Lidar pointcloud → /ingest/pointcloud")
    p.add_argument("--cloud-url", default=DEFAULT_CLOUD)
    p.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    p.add_argument("--fps", type=int, default=DEFAULT_FPS,
                   help="upper bound on push rate (Hz). Default 4")
    p.add_argument("--max-points", type=int, default=MAX_POINTS,
                   help="downsample target. Default 20000 (≈40 KB compressed)")
    args = p.parse_args()
    log.info(
        f"cloud={args.cloud_url} robot={args.robot_id} "
        f"fps≤{args.fps} max_points={args.max_points}"
    )
    try:
        run_lcm(args.cloud_url, args.robot_id, args.fps, args.max_points)
    except KeyboardInterrupt:
        log.info("stopped")


if __name__ == "__main__":
    main()
