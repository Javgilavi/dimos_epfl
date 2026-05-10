"""
Bridge runner — spawns all DimOS→cloud bridges on the robot side.

Usage:
    python run_bridges.py --cloud-url http://<ec2-ip>:8080

Spawns (all restart automatically on crash):
  ws_bridge.py      — Socket.IO bridge (DimOS viz events: pose, costmap, path)
  camera_bridge.py  — Camera frames (LCM / HTTP / RTSP, auto-detected)
  pc_bridge.py      — LiDAR point cloud (LCM)
  nav_bridge.py     — Pose push + goal forwarding (LCM)
  dimos_bridge.py   — Semantic objects via DimOS MCP (skip with --no-dimos-bridge)

Cloud side:
    python main.py --server-only           # EC2, no bridges
Robot side:
    python run_bridges.py --cloud-url http://ec2-ip:8080
"""

import argparse
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _spawn(name: str, cmd: list[str]) -> subprocess.Popen:
    p = subprocess.Popen(cmd, cwd=HERE)
    print(f"[run_bridges] spawned {name} pid={p.pid}")
    return p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spawn all DimOS→cloud bridges toward a remote server"
    )
    parser.add_argument(
        "--cloud-url",
        default=os.environ.get("CLOUD_URL", "http://localhost:8080"),
        help="Cloud FastAPI URL (default: $CLOUD_URL or http://localhost:8080)",
    )
    parser.add_argument(
        "--robot-id", default=os.environ.get("ROBOT_ID", "go2_a"),
        help="Robot identifier (default: go2_a)",
    )
    parser.add_argument(
        "--ws-url", default="http://localhost:7779",
        help="DimOS Socket.IO URL (default: http://localhost:7779)",
    )
    parser.add_argument(
        "--camera-source", default="auto",
        choices=["auto", "http", "dimos", "rtsp", "opencv"],
        help="Camera source for camera_bridge (default: auto)",
    )
    parser.add_argument("--camera-fps",      type=int,   default=8)
    parser.add_argument(
        "--camera-http-url",
        default=os.environ.get("CAMERA_HTTP_URL", "http://192.168.123.18:8888/frame"),
    )
    parser.add_argument("--pc-fps",          type=int,   default=4)
    parser.add_argument("--pose-hz",         type=float, default=15.0)
    parser.add_argument("--goal-hz",         type=float, default=5.0)
    parser.add_argument(
        "--no-dimos-bridge", action="store_true",
        help="Skip dimos_bridge.py (DimOS MCP semantic objects)",
    )
    args = parser.parse_args()

    py    = sys.executable
    cloud = args.cloud_url
    rid   = args.robot_id

    bridge_specs: list[tuple[str, list[str]]] = [
        ("ws_bridge", [
            py, "-u", os.path.join(HERE, "ws_bridge.py"),
            "--cloud-url", cloud,
            "--robot-id", rid,
            "--ws-url", args.ws_url,
        ]),
        ("camera_bridge", [
            py, "-u", os.path.join(HERE, "camera_bridge.py"),
            "--cloud-url", cloud,
            "--robot-id", rid,
            "--source", args.camera_source,
            "--fps", str(args.camera_fps),
            "--http-url", args.camera_http_url,
        ]),
        ("pc_bridge", [
            py, "-u", os.path.join(HERE, "pc_bridge.py"),
            "--cloud-url", cloud,
            "--robot-id", rid,
            "--fps", str(args.pc_fps),
        ]),
        ("nav_bridge", [
            py, "-u", os.path.join(HERE, "nav_bridge.py"),
            "--cloud-url", cloud,
            "--robot-id", rid,
            "--pose-hz", str(args.pose_hz),
            "--goal-hz", str(args.goal_hz),
        ]),
    ]
    if not args.no_dimos_bridge:
        bridge_specs.append(("dimos_bridge", [
            py, "-u", os.path.join(HERE, "dimos_bridge.py"),
            "--cloud-url", cloud,
            "--robot-id", rid,
        ]))

    procs: list[tuple[str, subprocess.Popen]] = [
        (name, _spawn(name, cmd)) for name, cmd in bridge_specs
    ]

    def _shutdown(sig, frame):
        print(f"\n[run_bridges] stopping {len(procs)} bridges...")
        for _, p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        for name, p in procs:
            try:
                p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[run_bridges] all bridges running → {cloud}")
    print("[run_bridges] Ctrl+C to stop all\n")

    while True:
        time.sleep(5)
        for i, (name, p) in enumerate(procs):
            if p.poll() is not None:
                print(f"[run_bridges] {name} exited (code={p.returncode}), restarting...")
                _, cmd = bridge_specs[i]
                procs[i] = (name, _spawn(name, cmd))


if __name__ == "__main__":
    main()
