# Unitree Go2 — Getting Started

The Unitree Go2 is DimOS's primary reference platform. Full autonomous navigation, mapping, and agentic control — no ROS required.

## Requirements

- Unitree Go2 Pro or Air (stock firmware 1.1.7+, no jailbreak needed)
- Ubuntu 22.04/24.04 with CUDA GPU (recommended), or macOS (experimental)
- Python 3.12

## Install

First, install system dependencies for your platform:
- [Ubuntu](/docs/installation/ubuntu.md)
- [macOS](/docs/installation/osx.md)
- [Nix](/docs/installation/nix.md)

Then install DimOS:

```bash
uv venv --python "3.12"
source .venv/bin/activate
uv pip install 'dimos[base,unitree]'
```

## Try It — No Hardware Needed

```bash
# Replay a recorded Go2 navigation session
# First run downloads ~2.4 GB of LiDAR/video data from LFS
dimos --replay run unitree-go2
```

Opens the command center at [localhost:7779](http://localhost:7779) with Rerun 3D visualization — watch the Go2 map and navigate an office in real time.

## Run on Your Go2

### First-time setup, connecting to wifi, finding robot IP

Use `dimos go2tool` to provision wifi and find the robot's IP. Skip if the robot is already on your network and you know its IP.

1. Power on the Go2 — it advertises over BLE immediately.

2. Provision wifi (one-time per network):

optionally use discover to make sure robot is detected

```bash
dimos go2tool discover
```

configure wifi

```bash
dimos go2tool connect-wifi --ssid <wifi> --password <password>
```

Scans BLE and connects to the only robot it finds, or prompts you to pick if there are several.

3. Find the robot's IP:

```bash
dimos go2tool discover
```

Prints `SOURCE NAME IP MAC SERIAL` for every robot it sees over BLE and LAN. Export the IP:

```bash
export ROBOT_IP=<discovered_ip>
```

### Pre-flight checks

1. Robot is reachable and low latency <10ms, 0% packet loss
```bash
ping $ROBOT_IP
```

2. Built-in obstacle avoidance is on. (DimOS handles path planning, but the onboard obstacle avoidance provides an extra safety layer around tight spots)

### Ready to run DimOS

```bash
export ROBOT_IP=<YOUR_GO2_IP>
dimos run unitree-go2
```

That's it. DimOS connects via WebRTC (no jailbreak required), starts the full navigation stack, and opens the command center in your browser.

On the real Go2, DimOS uses the external Jetson/USB HTTP camera by default
instead of the native WebRTC video stream:

```bash
export GO2_EXTERNAL_CAMERA_URL=http://192.168.123.18:8888/frame
dimos run unitree-go2
```

This replaces only the camera stream. LiDAR, odometry, and motion commands still
come from the Go2 WebRTC connection. Simulation and replay keep their normal
camera sources. To use the native Go2 camera again:

```bash
export GO2_USE_EXTERNAL_CAMERA=false
```

Optional camera calibration overrides:

```bash
export GO2_EXTERNAL_CAMERA_WIDTH=640
export GO2_EXTERNAL_CAMERA_HEIGHT=480
export GO2_EXTERNAL_CAMERA_FX=576
export GO2_EXTERNAL_CAMERA_FY=576
export GO2_EXTERNAL_CAMERA_CX=320
export GO2_EXTERNAL_CAMERA_CY=240
```

### YOLO11 segmentation from the external camera

For the real robot, run the Jetson USB camera server on the Jetson/camera host:

```bash
python scripts/jetson_camera_server.py --host 0.0.0.0 --port 8888 --device 0
```

Then run YOLO11 segmentation on the workstation. This publishes the same camera
image plus segmentation outputs to DimOS LCM:

```bash
python scripts/workstation_yolo.py \
  --stream-url http://192.168.123.18:8888/frame \
  --model yolo11s-seg.pt \
  --feed-dimos \
  --cloud-url http://localhost:8080 \
  --headless
```

Published topics:

| Topic | Message |
|-------|---------|
| `/color_image#sensor_msgs.Image` | Raw external camera image |
| `/yolo11/detections#vision_msgs.Detection2DArray` | YOLO boxes/classes/confidence, with segmentation-backed detections |
| `/yolo11/annotations#foxglove_msgs.ImageAnnotations` | Boxes + mask outlines/fill for DimOS/Foxglove visualization |
| `/yolo11/segmented_image#sensor_msgs.Image` | Camera image with YOLO masks overlaid |

Use the `yolo11*-seg.pt` models for masks. Non-segmentation models still publish
regular 2D detections, but without mask outlines.

If `--cloud-url` points to robohack2026, the script also pushes the YOLO mask
overlay into the UI camera panel and posts detections with confidence >= 0.70 to
the UI semantic map. The object location is projected from the latest `/odom`
pose plus the camera bearing. Tune `--semantic-distance` and `--camera-fx` if
the markers appear too near/far or laterally shifted.

In `unitree-go2-agentic`, these detections are available to the LLM through MCP
tools:

| Tool | Use |
|------|-----|
| `get_latest_yolo_detections` | List current YOLO detections with labels, confidence, and pixel bboxes |
| `get_best_yolo_detection` | Get the highest-confidence bbox for a label such as `person` |

This makes prompts like "what does YOLO see?" or "use YOLO to find the person
and follow them" route through the external camera detections.

### What's Running

| Module | What It Does |
|--------|-------------|
| **GO2Connection** | WebRTC connection to the robot — streams LiDAR, video, odometry |
| **VoxelGridMapper** | Builds a 3D voxel map using column-carving (CUDA accelerated) |
| **CostMapper** | Converts 3D map → 2D costmap via terrain slope analysis |
| **ReplanningAStarPlanner** | Continuous A* path planning with dynamic replanning |
| **WavefrontFrontierExplorer** | Autonomous exploration of unmapped areas |
| **RerunBridge** | 3D visualization in browser |
| **WebsocketVis** | Command center at localhost:7779 |

### Send Goals

From the command center ([localhost:7779](http://localhost:7779)):
- Click on the map to set navigation goals
- Toggle autonomous exploration
- Monitor robot pose, costmap, and planned path

## MuJoCo Simulation

```bash
uv pip install 'dimos[base,unitree,sim]'
dimos --simulation run unitree-go2
```

Full navigation stack in MuJoCo — same code, simulated robot.

## Agentic Control

Natural language control with an LLM agent that understands physical space:

```bash
export OPENAI_API_KEY=<YOUR_KEY>
export ROBOT_IP=<YOUR_GO2_IP>
dimos run unitree-go2-agentic
```

Then use the human CLI to talk to the agent:

```bash
humancli
> explore the space
```

The agent subscribes to camera, LiDAR, and spatial memory streams — it sees what the robot sees.

## Available Blueprints

| Blueprint | Description |
|-----------|-------------|
| `unitree-go2-basic` | Connection + visualization (no navigation) |
| `unitree-go2` | Full navigation stack |
| `unitree-go2-agentic` | Navigation + LLM agent + MCP tool access |
| `unitree-go2-agentic-ollama` | Agent with local Ollama models |
| `unitree-go2-spatial` | Navigation + spatial memory |
| `unitree-go2-detection` | Navigation + object detection |
| `unitree-go2-ros` | ROS 2 bridge mode |

## Deep Dive

- [Navigation Stack](/docs/capabilities/navigation/native/index.md) — column-carving voxel mapping, costmap generation, A* planning
- [Visualization](/docs/usage/visualization.md) — Rerun, Foxglove, performance tuning
- [Data Streams](/docs/usage/data_streams) — RxPY streams, backpressure, quality filtering
- [Transports](/docs/usage/transports/index.md) — LCM, SHM, DDS
- [Blueprints](/docs/usage/blueprints.md) — composing modules
