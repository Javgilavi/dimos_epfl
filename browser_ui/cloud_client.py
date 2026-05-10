"""
Cloud Client — pushes robot data to the FastAPI cloud middleware.

Runs alongside DimOS on the robot's machine (or in simulation).
Periodically pushes:
  - Detected objects (from DimOS perception / YOLO)
  - Camera frames (JPEG snapshots)

Usage:
    # As a standalone script (for testing):
    python cloud_client.py --url http://localhost:8080 --robot-id go2_a

    # Integrated into go2_dimos:
    from cloud_client import CloudClient
    client = CloudClient(cloud_url="http://your-ec2-ip:8080")
    client.push_detections(candidates)
    client.push_frame(jpeg_bytes)
"""
from __future__ import annotations

import time
import json
import logging
import threading
from dataclasses import dataclass, asdict
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class DetectionPayload:
    """Single detected object in cloud format."""
    label: str
    confidence: float
    pose: dict[str, float]
    seen_count: int = 1
    last_seen: float = 0.0
    source: str = "yolo"

    def __post_init__(self):
        if self.last_seen == 0.0:
            self.last_seen = time.time()


class CloudClient:
    """
    Pushes robot data to the cloud middleware.
    Non-blocking — never stalls the robot if cloud is unreachable.
    """

    def __init__(
        self,
        cloud_url: str = "http://localhost:8080",
        robot_id: str = "go2_a",
        timeout: float = 2.0,
    ):
        self.cloud_url = cloud_url.rstrip("/")
        self.robot_id = robot_id
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self._connected = False
        self._last_push_time = 0.0
        self._buffer: list[DetectionPayload] = []

    @property
    def connected(self) -> bool:
        return self._connected

    # ── Push Detections ───────────────────────────────────────

    def push_detections(self, objects: list[dict[str, Any]]) -> bool:
        """
        Push a list of detected objects to the cloud.

        Args:
            objects: List of dicts with keys: label, confidence, pose, seen_count, source
                     (matches TargetCandidate format from go2_dimos)

        Returns:
            True if push succeeded, False otherwise.
        """
        payload = {
            "robot_id": self.robot_id,
            "objects": objects,
        }

        try:
            resp = self._client.post(
                f"{self.cloud_url}/ingest",
                json=payload,
            )
            self._connected = True
            self._last_push_time = time.time()

            if resp.status_code == 200:
                logger.debug(f"Pushed {len(objects)} objects to cloud")
                return True
            else:
                logger.warning(f"Cloud returned {resp.status_code}: {resp.text}")
                return False

        except (httpx.ConnectError, httpx.TimeoutException) as e:
            self._connected = False
            logger.debug(f"Cloud unreachable: {e}")
            # Buffer for later if needed
            return False

    def push_candidates(self, candidates: list) -> bool:
        """
        Push TargetCandidate objects (from go2_dimos.contracts).

        Converts TargetCandidate dataclass format to cloud API format.
        """
        objects = []
        for c in candidates:
            obj = {
                "label": c.label if hasattr(c, "label") else str(c),
                "confidence": getattr(c, "confidence", None) or 0.5,
                "pose": getattr(c, "pose", None) or {"x": 0.0, "y": 0.0, "z": 0.0},
                "seen_count": 1,
                "source": getattr(c, "source", "dimos"),
            }
            objects.append(obj)

        return self.push_detections(objects)

    # ── Push Camera Frame ─────────────────────────────────────

    def push_frame(self, jpeg_bytes: bytes) -> bool:
        """
        Push a camera frame (JPEG) to the cloud.

        Args:
            jpeg_bytes: Raw JPEG image data.

        Returns:
            True if push succeeded.
        """
        try:
            resp = self._client.post(
                f"{self.cloud_url}/frames",
                content=jpeg_bytes,
                headers={
                    "Content-Type": "image/jpeg",
                    "X-Robot-Id": self.robot_id,
                },
            )
            self._connected = True
            return resp.status_code == 200

        except (httpx.ConnectError, httpx.TimeoutException):
            self._connected = False
            return False

    # ── Health Check ──────────────────────────────────────────

    def check_connection(self) -> bool:
        """Check if the cloud server is reachable."""
        try:
            resp = self._client.get(f"{self.cloud_url}/health")
            self._connected = resp.status_code == 200
        except Exception:
            self._connected = False
        return self._connected

    # ── Cleanup ───────────────────────────────────────────────

    def close(self):
        self._client.close()


# ── Standalone test mode ──────────────────────────────────────

def _fake_detections() -> list[dict]:
    """Generate fake detections for testing without DimOS."""
    import random
    objects = ["backpack", "chair", "bottle", "laptop", "person", "dog", "table"]
    detections = []
    for _ in range(random.randint(1, 4)):
        detections.append({
            "label": random.choice(objects),
            "confidence": round(random.uniform(0.5, 0.99), 2),
            "pose": {
                "x": round(random.uniform(-3.0, 3.0), 1),
                "y": round(random.uniform(-3.0, 3.0), 1),
                "z": 0.0,
            },
            "seen_count": random.randint(1, 5),
            "source": "yolo",
        })
    return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cloud client — push fake data for testing")
    parser.add_argument("--url", default="http://localhost:8080", help="Cloud middleware URL")
    parser.add_argument("--robot-id", default="go2_a", help="Robot identifier")
    parser.add_argument("--interval", type=float, default=3.0, help="Push interval in seconds")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    client = CloudClient(cloud_url=args.url, robot_id=args.robot_id)

    print(f"Pushing fake detections to {args.url} every {args.interval}s")
    print(f"Robot ID: {args.robot_id}")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            detections = _fake_detections()
            ok = client.push_detections(detections)
            status = "✓" if ok else "✗ (cloud unreachable)"
            print(f"[{time.strftime('%H:%M:%S')}] Pushed {len(detections)} objects {status}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")
        client.close()
