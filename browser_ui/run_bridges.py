"""
Bridge runner — spawns all DimOS→cloud bridges on the robot side.

Usage:
    python run_bridges.py --cloud-url http://<ec2-ip>:8080

Spawns (all restart automatically on crash):
  mcp_proxy_bridge.py — Cloud Agent MCP requests → local DimOS MCP
  workstation_yolo.py — Jetson camera → YOLO overlay, DimOS LCM, semantic map
  pc_bridge.py      — LiDAR point cloud (LCM)
  nav_bridge.py     — Pose push + goal forwarding (LCM)
  dimos_bridge.py   — Semantic objects via DimOS MCP (skip with --no-dimos-bridge)

Cloud side:
    python main.py --server-only           # EC2, no bridges
Robot side:
    python run_bridges.py --cloud-url http://ec2-ip:8080

If cloud auth is enabled:
    python run_bridges.py --cloud-url http://ec2-ip:8080 \
        --bridge-password "$BRIDGE_PASSWORD"
"""

import argparse
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


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
        "--mcp-url",
        default=os.environ.get("DIMOS_MCP_URL", "http://localhost:9990/mcp"),
        help="Local DimOS MCP URL forwarded to the cloud Agent",
    )
    parser.add_argument(
        "--bridge-password",
        default=os.environ.get("BRIDGE_PASSWORD", ""),
        help="Shared BRIDGE_PASSWORD used by the cloud server, if enabled",
    )
    parser.add_argument("--no-mcp-proxy", action="store_true")
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument(
        "--with-camera-bridge",
        action="store_true",
        help="Also run camera_bridge.py. Usually unnecessary because YOLO publishes camera frames + LCM.",
    )
    parser.add_argument("--yolo-model", default=os.environ.get("YOLO_MODEL", "yolo11s-seg.pt"))
    parser.add_argument("--yolo-imgsz", type=int, default=int(os.environ.get("YOLO_IMGSZ", "480")))
    parser.add_argument("--yolo-conf", type=float, default=float(os.environ.get("YOLO_CONF", "0.30")))
    parser.add_argument("--semantic-threshold", type=float, default=float(os.environ.get("YOLO_SEMANTIC_THRESHOLD", "0.70")))
    parser.add_argument("--semantic-hz", type=float, default=float(os.environ.get("YOLO_SEMANTIC_HZ", "1.0")))
    parser.add_argument("--ui-frame-hz", type=float, default=float(os.environ.get("YOLO_UI_FRAME_HZ", "12.0")))
    parser.add_argument(
        "--no-dimos-bridge", action="store_true",
        help="Skip dimos_bridge.py (DimOS MCP semantic objects)",
    )
    args = parser.parse_args()

    py    = sys.executable
    cloud = args.cloud_url.rstrip("/")
    rid   = args.robot_id
    if args.bridge_password:
        os.environ["BRIDGE_PASSWORD"] = args.bridge_password
    os.environ["CLOUD_URL"] = cloud
    os.environ["ROBOT_ID"] = rid
    os.environ["DIMOS_MCP_URL"] = args.mcp_url

    bridge_specs: list[tuple[str, list[str]]] = [
        ("mcp_proxy_bridge", [
            py, "-u", os.path.join(HERE, "mcp_proxy_bridge.py"),
            "--cloud-url", cloud,
            "--mcp-url", args.mcp_url,
        ]),
        ("workstation_yolo", [
            py, "-u", os.path.join(ROOT, "scripts", "workstation_yolo.py"),
            "--stream-url", args.camera_http_url,
            "--model", args.yolo_model,
            "--imgsz", str(args.yolo_imgsz),
            "--conf", str(args.yolo_conf),
            "--feed-dimos",
            "--cloud-url", cloud,
            "--robot-id", rid,
            "--semantic-threshold", str(args.semantic_threshold),
            "--semantic-hz", str(args.semantic_hz),
            "--ui-frame-hz", str(args.ui_frame_hz),
            "--headless",
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
    if args.no_mcp_proxy:
        bridge_specs = [b for b in bridge_specs if b[0] != "mcp_proxy_bridge"]
    if args.no_yolo:
        bridge_specs = [b for b in bridge_specs if b[0] != "workstation_yolo"]
    if args.no_yolo or args.with_camera_bridge:
        bridge_specs.insert(1, ("camera_bridge", [
            py, "-u", os.path.join(HERE, "camera_bridge.py"),
            "--cloud-url", cloud,
            "--robot-id", rid,
            "--source", args.camera_source,
            "--fps", str(args.camera_fps),
            "--http-url", args.camera_http_url,
        ]))
    if not args.no_dimos_bridge:
        bridge_specs.append(("dimos_bridge", [
            py, "-u", os.path.join(HERE, "dimos_bridge.py"),
            "--cloud-url", cloud,
            "--robot-id", rid,
            "--mcp-url", args.mcp_url,
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
