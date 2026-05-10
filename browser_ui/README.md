# SpatialMind — Cloud Middleware

FastAPI server that bridges a Unitree Go2 robot (running DimOS) to a live web dashboard with semantic querying via AWS Bedrock.

## Architecture

```
[Laptop / DimOS]                [EC2 :8080]              [Phone / Browser]
  nav_bridge.py ── POST ──────→  /ingest/pose ←── GET ──  /pose/live
  pc_bridge.py  ── POST ──────→  /ingest/pointcloud ←── GET ── /pointcloud/live
  dimos_bridge  ── POST ──────→  /ingest       ←── GET ──  /
  cloud_client  ── POST ──────→  /frames       ← POST ──→  /query/stream
                                                   POST ──→ /navigate
```

## Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI server — all endpoints + dashboard HTML |
| `nav_bridge.py` | Direct LCM bridge: `/odom` to `/ingest/pose`, queued goals to `/goal_request` |
| `pc_bridge.py` | Direct LCM bridge: `/lidar` pointcloud to `/ingest/pointcloud` |
| `camera_bridge.py` | Camera frame pusher — auto-detects source (Jetson HTTP / DimOS / RTSP / USB) |
| `dimos_bridge.py` | Reads DimOS MCP perception tools, pushes detected objects to cloud |
| `cloud_client.py` | `CloudClient` class — reusable push client for robot-side integration |
| `agent.py` | AWS Bedrock agent loop — natural language queries over the semantic map |
| `world_store.py` | Semantic map persistence — memory backend or S3 backend |
| `models.py` | Pydantic models shared across the API |

## Quickstart (local)

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
# Dashboard at http://localhost:8080
```

## Deploy on EC2

```bash
# On EC2 (Ubuntu 24.04, t3.small, port 8080 open in Security Group)
git clone https://github.com/your/repo.git && cd robohack_fastapi
pip install -r requirements.txt

export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=eu-north-1
export S3_BUCKET=your-bucket-name
export USE_MEMORY_STORE=false   # omit to use in-memory store (no AWS needed)

# Run in background
screen -S fastapi
uvicorn main:app --host 0.0.0.0 --port 8080
# Ctrl+A D to detach
```

## Camera stream

`camera_bridge.py` pushes frames to `/frames` so the dashboard shows live video.
For the real Go2 setup with the added Jetson/USB camera, use the HTTP JPEG
source. Auto mode tries this first:

```bash
python camera_bridge.py --source http --http-url http://192.168.123.18:8888/frame
```

By default, non-DimOS camera sources are also republished to DimOS LCM
(`/color_image#sensor_msgs.Image`) so DimOS visualizers can show the same camera.
Disable that with `--no-publish-lcm`.

**DimOS transport is platform-dependent:**
- **Linux** → `LCMTransport` (UDP multicast) — confirmed working
- **Mac** → `pSHMTransport` (shared memory) — `--source dimos` falls back to pSHM

Must run inside the dimos venv (`source dimos/.venv/bin/activate`) for `--source dimos`.

```bash
# Auto-detect (Jetson HTTP → LCM/pSHM → RTSP → USB webcam):
python camera_bridge.py

# Jetson/new USB camera HTTP stream:
python camera_bridge.py --source http --http-url http://192.168.123.18:8888/frame

# DimOS simulation (Linux: LCM multicast, Mac: pSHM):
python camera_bridge.py --source dimos

# Real Go2 over network:
python camera_bridge.py --source rtsp --rtsp-url rtsp://192.168.123.161:8554/video

# USB webcam — no DimOS needed, good for testing:
python camera_bridge.py --source opencv --device 0

# Push to EC2:
python camera_bridge.py --cloud-url http://<ec2-ip>:8080
```

Stream visible at `/frames/go2_a/stream` (MJPEG) and `/frames/go2_a` (latest JPEG).

For YOLO11 segmentation overlays and semantic-map ingestion, run DimOS'
workstation YOLO bridge against this UI:

```bash
cd ~/robohack-epfl/dimos
source .venv/bin/activate
python scripts/workstation_yolo.py \
  --stream-url http://192.168.123.18:8888/frame \
  --model yolo11s-seg.pt \
  --feed-dimos \
  --cloud-url http://localhost:8080 \
  --headless
```

Detections with confidence >= `0.70` are added to `/map` and appear as semantic
objects in the 3D map and object table. The script also pushes the YOLO mask
overlay frame to `/frames`, so the UI camera panel shows the segmented view.

## Voice input / speech-to-text

The microphone button records a short browser audio clip, uploads it to
`POST /speech/transcribe`, and uses Amazon Transcribe to convert it to text
before sending it to the active Ask/Agent tab. If AWS Transcribe is unavailable,
the UI falls back to the browser's built-in speech recognition where supported.

Amazon Transcribe uses the same AWS credentials already used for Bedrock/S3.
For short clips the server uploads the audio to S3, starts a transcription job,
polls it, returns the transcript, then deletes the temporary audio object.

```bash
export S3_BUCKET=your-bucket-name
export AWS_REGION=us-west-2
export AWS_TRANSCRIBE_LANGUAGE=en-US
export AWS_TRANSCRIBE_TIMEOUT=45
```

This lets voice commands such as "follow me", "find the chair", or "go to the
door" enter Agent mode exactly like typed commands, so the DimOS MCP tools stay
the execution path.

## Run the bridges (laptop, with DimOS running)

```bash
# Point nav/pointcloud bridges at EC2
python nav_bridge.py --cloud-url http://<ec2-public-ip>:8080
python pc_bridge.py --cloud-url http://<ec2-public-ip>:8080

# Or locally
python nav_bridge.py --cloud-url http://localhost:8080
python pc_bridge.py --cloud-url http://localhost:8080
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `USE_MEMORY_STORE` | `true` | `false` to persist map to S3 |
| `S3_BUCKET` | `robohack-map` | S3 bucket name |
| `AWS_REGION` | `eu-west-1` | AWS region (use `eu-north-1` for Stockholm) |
| `PC_ACCUM_VOXEL_CM` | `8` | Voxel size for accumulated lidar map |
| `PC_ACCUM_MAX_POINTS` | `120000` | Maximum accumulated lidar voxels served to the UI |
| `PC_OBS_RADIUS_CM` | `20` | Nearby XY radius treated as actively observed around each fresh lidar hit |
| `PC_RENDER_SCORE` | `1.2` | Minimum voxel confidence before a point is rendered in the UI |
| `PC_MISS_DECAY` | `0.72` | Confidence multiplier when an observed-area voxel is missed in a fresh scan |
| `PC_TIME_DECAY_PER_SEC` | `0.015` | Slow global confidence decay so old false positives eventually disappear |
| `PC_DELETE_SCORE` | `0.35` | Delete voxels below this confidence |
| `CAMERA_BRIDGE_SOURCE` | `auto` | Camera source for auto-spawn: `auto`, `http`, `dimos`, `rtsp`, or `opencv` |
| `CAMERA_HTTP_URL` | `http://192.168.123.18:8888/frame` | Jetson/new USB camera JPEG frame URL |
| `CAMERA_PUBLISH_LCM` | `true` | Republish non-DimOS camera frames to DimOS `/color_image` LCM |
| `AWS_TRANSCRIBE_LANGUAGE` | `en-US` | Language code for AWS Transcribe voice input |
| `AWS_TRANSCRIBE_TIMEOUT` | `45` | Seconds to wait for a short Transcribe batch job |
| `SPEECH_S3_PREFIX` | `speech` | S3 prefix for temporary uploaded voice clips |
| `AWS_ACCESS_KEY_ID` | — | IAM credentials for Bedrock + S3 + Transcribe |
| `AWS_SECRET_ACCESS_KEY` | — | IAM credentials for Bedrock + S3 + Transcribe |

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Live dashboard (map + chat + objects) |
| `GET` | `/health` | Health check |
| `POST` | `/ingest` | Robot pushes detected objects |
| `POST` | `/ingest/pose` | Nav bridge pushes live odometry pose |
| `POST` | `/ingest/pointcloud` | Pointcloud bridge pushes compressed lidar points |
| `GET` | `/map` | Merged semantic map (all robots) |
| `GET` | `/pose/live` | Latest robot pose by robot id |
| `GET` | `/pointcloud/live` | Accumulated lidar voxel pointcloud by robot id |
| `POST` | `/query/stream` | SSE — natural language query via Bedrock |
| `POST` | `/navigate` | Send navigation goal to robot |
| `POST` | `/frames` | Robot pushes JPEG camera frame |
| `GET` | `/frames/{robot_id}` | Latest camera frame |
| `GET` | `/frames/{robot_id}/stream` | MJPEG stream |
| `POST` | `/speech/transcribe` | AWS Transcribe short audio clip to text |

## Bedrock model

Uses `eu.anthropic.claude-sonnet-4-6` (cross-region inference profile, eu-north-1).
Requires IAM permissions: `bedrock:InvokeModelWithResponseStream`.
