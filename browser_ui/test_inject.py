"""
Mock data injector — tests all new semantic/spatial UI features locally.

Pushes realistic YOLO11-style detections to the running browser_ui server
(http://localhost:8080 by default) so you can verify:

  📋 Objects tab     — classic table with confidence colouring
  🧠 Semantic tab    — cards with tags, flags, spatial relations, image crops
  📍 Spatial tab     — canvas graph with edges and legend

Run the server first:
    cd browser_ui
    uvicorn main:app --host 0.0.0.0 --port 8080 --reload

Then in a second terminal:
    python test_inject.py

Flags:
    --url  http://localhost:8080   target server
    --once                         push once then exit (default: loop every 3s)
    --robot-id go2_a               robot identifier
"""

import argparse
import base64
import io
import json
import math
import sys
import time
import urllib.request
import urllib.error


# ── Tiny 8×8 JPEG crop (1×1 white pixel encoded, just enough to test the
#    image-crop display path without a Pillow / numpy dependency). ─────────
_MOCK_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
    "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN"
    "DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
    "MjIyMjL/wAARCAAIAAgDASIAAhEBAxEB/8QAFgABAQEAAAAAAAAAAAAAAAAABgUE/8QAIB"
    "AAAQUBAQEBAAAAAAAAAAAAAQIDBAUREiEx/8QAFQEBAQAAAAAAAAAAAAAAAAAAAgP/xAAW"
    "EQEBAQAAAAAAAAAAAAAAAAABEjH/2gAMAwEAAhEDEQA/AJ65K2Q8ZLuvPZPaUm5JyYqST"
    "1bFKQKRmTmWkkSbg4gAAAAAAAAAAAA//9k="
)


def _make_objects():
    """Return a list of mock DetectedObject dicts — mix of labels/positions/conf."""
    now = time.time()
    return [
        {
            "label": "person",
            "confidence": 0.92,
            "pose": {"x": 1.2, "y": 0.3, "z": 0.9},
            "seen_count": 8,
            "last_seen": now,
            "source": "yolo_sim",
        },
        {
            "label": "backpack",
            "confidence": 0.78,
            "pose": {"x": 1.4, "y": 0.5, "z": 0.1},
            "seen_count": 3,
            "last_seen": now - 15,
            "source": "yolo_sim",
        },
        {
            "label": "chair",
            "confidence": 0.88,
            "pose": {"x": 2.5, "y": -0.8, "z": 0.4},
            "seen_count": 12,
            "last_seen": now - 5,
            "source": "yolo_sim",
        },
        {
            "label": "laptop",
            "confidence": 0.55,  # below 70% → triggers image crop + ⚠LOW badge
            "pose": {"x": 2.6, "y": -0.7, "z": 0.75},
            "seen_count": 2,
            "last_seen": now - 30,
            "source": "yolo_sim",
            "image_crop_b64": _MOCK_JPEG_B64,
            "bbox": [120.0, 80.0, 210.0, 155.0],
        },
        {
            "label": "door",
            "confidence": 0.95,
            "pose": {"x": 4.0, "y": 1.2, "z": 1.0},
            "seen_count": 20,
            "last_seen": now - 2,
            "source": "yolo_sim",
        },
        {
            "label": "potted plant",
            "confidence": 0.62,
            "pose": {"x": 3.5, "y": -1.5, "z": 0.3},
            "seen_count": 5,
            "last_seen": now - 60,
            "source": "yolo_sim",
        },
        {
            "label": "bottle",
            "confidence": 0.45,  # very low → should show red card + image crop
            "pose": {"x": 2.7, "y": -0.9, "z": 0.72},
            "seen_count": 1,
            "last_seen": now - 90,
            "source": "yolo_sim",
            "image_crop_b64": _MOCK_JPEG_B64,
        },
    ]


def _make_pose():
    """Fake robot odometry — circles slowly so the spatial map shifts."""
    t = time.time() * 0.05
    return {
        "x":   round(math.sin(t) * 0.4, 3),
        "y":   round(math.cos(t) * 0.4 - 0.5, 3),
        "z":   0.0,
        "yaw": round(t % (2 * math.pi), 4),
        "ts":  time.time(),
    }


def _post(url: str, payload: dict, label: str) -> bool:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=4) as resp:
            body = resp.read().decode()
            print(f"  ✓ {label}: {resp.status}  {body[:80]}")
            return True
    except urllib.error.URLError as e:
        print(f"  ✗ {label}: {e.reason}")
        return False


def inject(base_url: str, robot_id: str) -> None:
    objects = _make_objects()
    pose    = _make_pose()

    print(f"\n[{time.strftime('%H:%M:%S')}] Injecting {len(objects)} objects + pose → {base_url}")

    _post(
        f"{base_url}/ingest",
        {"robot_id": robot_id, "objects": objects},
        "objects /ingest",
    )
    _post(
        f"{base_url}/ingest/pose",
        pose,
        "pose   /ingest/pose",
    )


def main():
    ap = argparse.ArgumentParser(description="Mock YOLO11 data injector for browser_ui")
    ap.add_argument("--url",      default="http://localhost:8080")
    ap.add_argument("--robot-id", default="go2_a")
    ap.add_argument("--once",     action="store_true",
                    help="Push once then exit")
    args = ap.parse_args()

    # Health-check first so we fail fast with a clear message.
    try:
        with urllib.request.urlopen(f"{args.url}/health", timeout=3) as r:
            print(f"Server reachable: {r.read().decode()}")
    except Exception as e:
        print(f"\n✗ Cannot reach {args.url}/health — is the server running?\n"
              f"  Start it with:  uvicorn main:app --port 8080 --reload\n"
              f"  Error: {e}")
        sys.exit(1)

    print(f"\nOpen your browser at {args.url}")
    print("Switch between 📋 Objects / 🧠 Semantic / 📍 Spatial tabs to see the data.\n")

    inject(args.url, args.robot_id)
    if args.once:
        return

    print("Injecting every 3 s — Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(3)
            inject(args.url, args.robot_id)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
