"""
AWS Cloud Middleware — FastAPI Communication Layer

Endpoints:
  POST /ingest        — Robot pushes detections (Req 1)
  POST /query/stream  — User asks natural language question, SSE response (Req 2)
  GET  /map           — Merged world state from all robots
  GET  /              — Dashboard UI
  GET  /health        — Health check (no auth required)
"""

import base64
import os
import struct
import time
import json
import logging
import zlib
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, Response

from models import (
    IngestRequest, IngestResponse,
    QueryRequest, MapResponse,
    ErrorResponse, WorldState,
)
from world_store import WorldStateStore

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────

S3_BUCKET = os.environ.get("S3_BUCKET", "robohack-map")
AWS_REGION = os.environ.get("AWS_REGION", "eu-west-1")
USE_MEMORY_STORE = os.environ.get("USE_MEMORY_STORE", "true").lower() == "true"


# ── App Lifespan ──────────────────────────────────────────────

# Spawn direct-LCM bridges when dimos is reachable locally. Set
# AUTO_BRIDGES=false to disable, e.g. when running the cloud UI on EC2.
AUTO_BRIDGES = os.environ.get("AUTO_BRIDGES", "true").lower() == "true"
CAMERA_BRIDGE_FPS = os.environ.get("CAMERA_BRIDGE_FPS", "10")
CAMERA_BRIDGE_SOURCE = os.environ.get("CAMERA_BRIDGE_SOURCE", "auto")
CAMERA_HTTP_URL = os.environ.get("CAMERA_HTTP_URL", "http://192.168.123.18:8888/frame")
CAMERA_PUBLISH_LCM = os.environ.get("CAMERA_PUBLISH_LCM", "true")
SELF_URL = os.environ.get("SELF_URL", "http://localhost:8080")
PC_ACCUM_VOXEL_CM = max(1, int(os.environ.get("PC_ACCUM_VOXEL_CM", "8")))
PC_ACCUM_MAX_POINTS = max(1000, int(os.environ.get("PC_ACCUM_MAX_POINTS", "120000")))
PC_OBS_RADIUS_CM = max(0, int(os.environ.get("PC_OBS_RADIUS_CM", "20")))
PC_HIT_SCORE = float(os.environ.get("PC_HIT_SCORE", "1.0"))
PC_MAX_SCORE = float(os.environ.get("PC_MAX_SCORE", "8.0"))
PC_MISS_DECAY = float(os.environ.get("PC_MISS_DECAY", "0.72"))
PC_TIME_DECAY_PER_SEC = float(os.environ.get("PC_TIME_DECAY_PER_SEC", "0.015"))
PC_RENDER_SCORE = float(os.environ.get("PC_RENDER_SCORE", "1.2"))
PC_DELETE_SCORE = float(os.environ.get("PC_DELETE_SCORE", "0.35"))
AWS_TRANSCRIBE_LANGUAGE = os.environ.get("AWS_TRANSCRIBE_LANGUAGE", "en-US")
AWS_TRANSCRIBE_TIMEOUT = float(os.environ.get("AWS_TRANSCRIBE_TIMEOUT", "45"))
SPEECH_S3_PREFIX = os.environ.get("SPEECH_S3_PREFIX", "speech")


def _load_aws_env_from_dimos() -> None:
    """Load AWS credentials from the sibling DimOS .env when not already set."""
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return
    candidates = [
        os.path.expanduser("~/robohack-epfl/dimos/.env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dimos", ".env"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key.startswith("AWS_") or key in {"S3_BUCKET"}:
                        os.environ.setdefault(key, val)
            return
        except Exception as exc:
            logger.debug("Could not load AWS env from %s: %s", path, exc)


def _audio_format_from_content_type(content_type: str) -> tuple[str, str]:
    ct = (content_type or "").lower()
    if "webm" in ct:
        return "webm", "audio/webm"
    if "mp4" in ct or "m4a" in ct:
        return "mp4", "audio/mp4"
    if "mpeg" in ct or "mp3" in ct:
        return "mp3", "audio/mpeg"
    if "ogg" in ct:
        return "ogg", "audio/ogg"
    if "wav" in ct or "wave" in ct:
        return "wav", "audio/wav"
    return "webm", "audio/webm"


def _spawn_bridges() -> list:
    """Launch local bridges as subprocesses.

    Each bridge will reconnect on its own if dimos starts late. Failures here
    are logged but never fatal; the dashboard still serves without bridges.
    """
    import subprocess
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    procs: list = []
    py = sys.executable

    # All bridges talk to dimos via LCM directly — no Socket.IO / command-center.
    bridge_specs = [
        (
            "camera_bridge",
            [py, "-u", os.path.join(here, "camera_bridge.py"),
             "--cloud-url", SELF_URL,
             "--fps", str(CAMERA_BRIDGE_FPS),
             "--source", CAMERA_BRIDGE_SOURCE,
             "--http-url", CAMERA_HTTP_URL,
             "--publish-lcm" if CAMERA_PUBLISH_LCM.lower() == "true" else "--no-publish-lcm"],
        ),
        (
            "pc_bridge",
            [py, "-u", os.path.join(here, "pc_bridge.py"),
             "--cloud-url", SELF_URL,
             "--fps", os.environ.get("PC_BRIDGE_FPS", "5")],
        ),
        (
            "nav_bridge",
            [py, "-u", os.path.join(here, "nav_bridge.py"),
             "--cloud-url", SELF_URL,
             "--pose-hz", os.environ.get("NAV_POSE_HZ", "15"),
             "--goal-hz", os.environ.get("NAV_GOAL_POLL_HZ", "5")],
        ),
    ]
    for name, cmd in bridge_specs:
        try:
            p = subprocess.Popen(
                cmd,
                cwd=here,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            procs.append((name, p))
            logger.info(f"[lifespan] spawned {name} pid={p.pid}")
        except Exception as e:
            logger.warning(f"[lifespan] failed to spawn {name}: {e}")
    return procs


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup."""
    app.state.world_store = WorldStateStore(
        bucket=S3_BUCKET,
        use_memory=USE_MEMORY_STORE,
        region=AWS_REGION,
    )
    logger.info(
        f"World store initialized: {'memory' if USE_MEMORY_STORE else 'S3'} "
        f"(bucket={S3_BUCKET})"
    )
    app.state.bridge_procs = _spawn_bridges() if AUTO_BRIDGES else []
    try:
        yield
    finally:
        # Best-effort cleanup of bridge subprocesses on shutdown.
        for name, p in app.state.bridge_procs:
            try:
                p.terminate()
                p.wait(timeout=2)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
            logger.info(f"[lifespan] stopped {name}")


app = FastAPI(
    title="AWS Cloud Middleware — DimOS",
    description="Communication and inference hub for Unitree Go2 robot",
    lifespan=lifespan,
)


# ── Health Check ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "store": "memory" if USE_MEMORY_STORE else "s3"}


# ── Speech-to-text (AWS Transcribe) ───────────────────────────

@app.post("/speech/transcribe")
async def transcribe_speech(request: Request):
    """Transcribe a short browser-recorded audio clip via Amazon Transcribe.

    The browser posts the audio blob directly. We upload it to S3 because
    Transcribe batch jobs require an S3 media URI, then poll the short job and
    return the text for the active Ask/Agent chat tab.
    """
    audio = await request.body()
    if not audio:
        raise HTTPException(400, "empty audio body")

    _load_aws_env_from_dimos()
    bucket = os.environ.get("S3_BUCKET", S3_BUCKET)
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or AWS_REGION
    lang = os.environ.get("AWS_TRANSCRIBE_LANGUAGE", AWS_TRANSCRIBE_LANGUAGE)
    timeout = float(os.environ.get("AWS_TRANSCRIBE_TIMEOUT", str(AWS_TRANSCRIBE_TIMEOUT)))
    fmt, content_type = _audio_format_from_content_type(
        request.headers.get("content-type", "")
    )

    try:
        import boto3
        import httpx
    except Exception as exc:
        raise HTTPException(500, f"missing AWS transcription dependency: {exc}")

    job = f"robohack-speech-{uuid.uuid4().hex}"
    key = f"{SPEECH_S3_PREFIX}/{job}.{fmt}"
    s3_uri = f"s3://{bucket}/{key}"

    s3 = boto3.client("s3", region_name=region)
    transcribe = boto3.client("transcribe", region_name=region)
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=audio, ContentType=content_type)
        transcribe.start_transcription_job(
            TranscriptionJobName=job,
            Media={"MediaFileUri": s3_uri},
            MediaFormat=fmt,
            LanguageCode=lang,
        )

        deadline = time.time() + timeout
        last_status = "QUEUED"
        while time.time() < deadline:
            info = transcribe.get_transcription_job(TranscriptionJobName=job)[
                "TranscriptionJob"
            ]
            last_status = info["TranscriptionJobStatus"]
            if last_status == "COMPLETED":
                transcript_uri = info["Transcript"]["TranscriptFileUri"]
                data = httpx.get(transcript_uri, timeout=10).json()
                text = (data.get("results", {}).get("transcripts") or [{}])[0].get(
                    "transcript", ""
                )
                return {"text": text, "job": job, "language": lang}
            if last_status == "FAILED":
                raise HTTPException(
                    502,
                    info.get("FailureReason", "Amazon Transcribe job failed"),
                )
            time.sleep(1.0)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"speech transcription failed: {exc}")
    finally:
        try:
            s3.delete_object(Bucket=bucket, Key=key)
        except Exception:
            pass

    raise HTTPException(504, f"speech transcription timed out ({last_status})")


# ── Robot Endpoints (Req 1: Ingestion) ────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    store: WorldStateStore = app.state.world_store
    previous = store.load(request.robot_id)
    objects = _merge_detected_objects(previous.objects if previous else [], request.objects)
    state = WorldState(
        robot_id=request.robot_id,
        timestamp=time.time(),
        objects=objects,
    )
    try:
        store.save(state)
    except Exception as e:
        logger.error(f"Failed to save world state for {request.robot_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Storage error: {e}")

    return IngestResponse(
        status="saved",
        robot_id=request.robot_id,
        count=len(objects),
        timestamp=state.timestamp,
    )


def _merge_detected_objects(existing, incoming):
    """Merge semantic detections instead of replacing the whole map each push."""
    merged = [obj.model_copy(deep=True) for obj in existing]
    for obj in incoming:
        match = None
        for prev in merged:
            if prev.label.lower() != obj.label.lower():
                continue
            dx = prev.pose.x - obj.pose.x
            dy = prev.pose.y - obj.pose.y
            dz = prev.pose.z - obj.pose.z
            if (dx * dx + dy * dy + dz * dz) ** 0.5 <= 1.0:
                match = prev
                break
        if match is None:
            merged.append(obj)
            continue
        old_n = max(1, match.seen_count)
        new_n = old_n + max(1, obj.seen_count)
        match.pose.x = (match.pose.x * old_n + obj.pose.x) / new_n
        match.pose.y = (match.pose.y * old_n + obj.pose.y) / new_n
        match.pose.z = (match.pose.z * old_n + obj.pose.z) / new_n
        match.confidence = max(match.confidence, obj.confidence)
        match.seen_count = new_n
        match.last_seen = max(match.last_seen, obj.last_seen)
        match.source = obj.source or match.source
    return sorted(merged, key=lambda o: o.last_seen, reverse=True)[:200]


# ── User Endpoints (Req 2: Query) ────────────────────────────

def _stream_mcp_response(text: str, mode: str) -> StreamingResponse:
    """Shared SSE wrapper around the MCP-aware Bedrock agent."""
    from agent_mcp import run_mcp_agent_stream

    def generate():
        try:
            for token in run_mcp_agent_stream(text, mode=mode):
                yield f"data: {json.dumps(token)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception("MCP agent stream error")
            yield f"data: {json.dumps(f'Error: {e}')}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """🧠 Ask mode — read-only Q&A over the dimos MCP.

    The agent has the full set of dimos read-only tools (observe, server_status,
    spatial-memory queries, …) but NO movement / action tools. If the user
    asks the robot to act, the agent explains and points them to the Agent tab.
    """
    store: WorldStateStore = app.state.world_store
    use_mcp = os.environ.get("USE_MCP_AGENT", "true").lower() == "true"

    if use_mcp:
        try:
            return _stream_mcp_response(request.text, mode="ask")
        except ImportError as e:
            logger.warning(f"MCP agent unavailable, using fallback: {e}")

    try:
        from agent import run_agent_stream
    except ImportError:
        raise HTTPException(status_code=503, detail="Agent module not available.")

    def generate():
        try:
            for token in run_agent_stream(request.text, store):
                yield f"data: {json.dumps(token)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Agent stream error: {e}")
            yield f"data: {json.dumps(f'Error: {e}')}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/command/stream")
async def command_stream(request: QueryRequest):
    """🤖 Agent mode — full tool access. Can move the robot, explore, etc."""
    use_mcp = os.environ.get("USE_MCP_AGENT", "true").lower() == "true"
    if use_mcp:
        try:
            return _stream_mcp_response(request.text, mode="agent")
        except ImportError as e:
            logger.warning(f"MCP agent unavailable: {e}")
    raise HTTPException(
        status_code=503,
        detail="Agent backend unavailable. Set USE_MCP_AGENT=true and ensure dimos MCP is running.",
    )


# ── Camera Frames ─────────────────────────────────────────────

_latest_frames: dict[str, bytes] = {}


@app.post("/frames")
async def receive_frame(request: Request):
    robot_id = request.headers.get("X-Robot-Id", "go2_a")
    body = await request.body()
    if not body:
        raise HTTPException(400, "Empty frame")
    _latest_frames[robot_id] = body
    return {"status": "ok", "robot_id": robot_id, "size": len(body)}


@app.get("/frames/{robot_id}")
async def get_frame(robot_id: str):
    frame = _latest_frames.get(robot_id)
    if frame is None:
        raise HTTPException(404, "No frame available")
    return Response(content=frame, media_type="image/jpeg")


@app.get("/frames/{robot_id}/stream")
async def stream_frames(robot_id: str):
    import asyncio
    async def generate():
        while True:
            frame = _latest_frames.get(robot_id)
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            await asyncio.sleep(0.1)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


# ── LiDAR Point Cloud ─────────────────────────────────────────
# pc_bridge.py POSTs zlib-compressed int16 triplets (x_cm, y_cm, z_cm).
# We accumulate those raw lidar points into a coarse voxel map so the UI shows
# the explored area, not only the most recent scan.

_live_pc: dict[str, dict] = {}  # robot_id -> {b64, n, z_min, z_max, timestamp}
_pc_voxels: dict[str, dict[tuple[int, int, int], dict]] = {}


def _rebuild_accumulated_pointcloud(robot_id: str) -> None:
    voxels = _pc_voxels.get(robot_id, {})
    pts = [
        (v["x"], v["y"], v["z"])
        for v in voxels.values()
        if v.get("score", 0.0) >= PC_RENDER_SCORE
    ]
    if not pts:
        _live_pc[robot_id] = {
            "b64": "",
            "n": 0,
            "z_min": 0,
            "z_max": 0,
            "timestamp": time.time(),
            "voxel_cm": PC_ACCUM_VOXEL_CM,
            "obs_radius_cm": PC_OBS_RADIUS_CM,
            "render_score": PC_RENDER_SCORE,
            "accumulated": True,
        }
        return

    raw = bytearray(len(pts) * 6)
    z_min = 32767
    z_max = -32768
    for i, (x, y, z) in enumerate(pts):
        struct.pack_into("<hhh", raw, i * 6, x, y, z)
        z_min = min(z_min, z)
        z_max = max(z_max, z)

    _live_pc[robot_id] = {
        "b64": base64.b64encode(raw).decode(),
        "n": len(pts),
        "z_min": z_min,
        "z_max": z_max,
        "timestamp": time.time(),
        "voxel_cm": PC_ACCUM_VOXEL_CM,
        "obs_radius_cm": PC_OBS_RADIUS_CM,
        "render_score": PC_RENDER_SCORE,
        "accumulated": True,
    }


def _update_accumulated_voxels(
    robot_id: str,
    pts_bytes: bytes,
    n: int,
) -> None:
    """Merge a fresh lidar scan into a confidence-weighted accumulated map.

    Repeated hits increase a voxel's score and update its position by running
    average. Missing voxels in the currently observed neighborhood decay instead
    of being deleted immediately. This keeps stable structure visible while
    allowing one-off false positives to fade after the robot looks there again.
    """
    voxels = _pc_voxels.setdefault(robot_id, {})
    now = time.time()
    step = PC_ACCUM_VOXEL_CM
    obs_cells = max(0, round(PC_OBS_RADIUS_CM / step))
    current: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    observed_xy: set[tuple[int, int]] = set()

    for i in range(n):
        x, y, z = struct.unpack_from("<hhh", pts_bytes, i * 6)
        kx = round(x / step)
        ky = round(y / step)
        kz = round(z / step)
        key = (kx, ky, kz)
        current[key] = (x, y, z)
        if obs_cells:
            for dx in range(-obs_cells, obs_cells + 1):
                for dy in range(-obs_cells, obs_cells + 1):
                    observed_xy.add((kx + dx, ky + dy))
        else:
            observed_xy.add((kx, ky))

    # Gentle global aging prevents old one-off artifacts from living forever.
    for key, val in list(voxels.items()):
        age = max(0.0, now - val.get("updated_at", now))
        if age > 0:
            val["score"] = max(0.0, val.get("score", 0.0) - age * PC_TIME_DECAY_PER_SEC)
            val["updated_at"] = now

    for key, (x, y, z) in current.items():
        val = voxels.get(key)
        if val is None:
            voxels[key] = {
                "x": x,
                "y": y,
                "z": z,
                "score": PC_HIT_SCORE,
                "hits": 1,
                "updated_at": now,
            }
            continue

        hits = val.get("hits", 1) + 1
        alpha = min(0.35, 1.0 / hits)
        val["x"] = round(val["x"] * (1.0 - alpha) + x * alpha)
        val["y"] = round(val["y"] * (1.0 - alpha) + y * alpha)
        val["z"] = round(val["z"] * (1.0 - alpha) + z * alpha)
        val["hits"] = hits
        val["score"] = min(PC_MAX_SCORE, val.get("score", 0.0) + PC_HIT_SCORE)
        val["updated_at"] = now

    # Stronger local decay: if the robot is actively observing this XY region
    # and a stored voxel is not seen again, lower its confidence but do not
    # erase it immediately. Consistent surfaces survive; false positives fade.
    if observed_xy:
        for key, val in list(voxels.items()):
            if key not in current and (key[0], key[1]) in observed_xy:
                val["score"] = val.get("score", 0.0) * PC_MISS_DECAY

    for key, val in list(voxels.items()):
        if val.get("score", 0.0) < PC_DELETE_SCORE:
            del voxels[key]

    # Bound memory and UI payload size. Dict order is insertion order; this
    # drops the weakest/oldest voxels first.
    overflow = len(voxels) - PC_ACCUM_MAX_POINTS
    if overflow > 0:
        drop = sorted(
            voxels,
            key=lambda k: (voxels[k].get("score", 0.0), voxels[k].get("updated_at", 0.0)),
        )[:overflow]
        for key in drop:
            del voxels[key]


@app.post("/ingest/pointcloud")
async def ingest_pointcloud(request: Request):
    robot_id = request.headers.get("X-Robot-Id", "go2_a")
    body = await request.body()
    if not body:
        raise HTTPException(400, "Empty payload")
    try:
        raw = zlib.decompress(body)
        n = struct.unpack_from("<i", raw)[0]
        pts_bytes = raw[4:]  # N × 6 bytes of int16 triplets
        if len(pts_bytes) < n * 6:
            raise ValueError(f"payload too short for {n} points")

        _update_accumulated_voxels(robot_id, pts_bytes, n)
        _rebuild_accumulated_pointcloud(robot_id)
    except Exception as e:
        raise HTTPException(400, f"Decode error: {e}")
    return {"status": "ok", "robot_id": robot_id, "n": _live_pc[robot_id]["n"]}


@app.get("/pointcloud/live")
async def get_live_pointcloud():
    return _live_pc


# ── Live robot pose (replaces costmap-bound pose path) ───────

# Latest pose per robot. nav_bridge.py POSTs at ~15 Hz from /odom LCM.
_live_pose: dict[str, dict] = {}


@app.post("/ingest/pose")
async def ingest_pose(request: Request):
    robot_id = request.headers.get("X-Robot-Id", "go2_a")
    try:
        data = await request.json()
    except Exception as e:
        raise HTTPException(400, f"bad JSON: {e}")
    if not isinstance(data, dict):
        raise HTTPException(400, "expected pose dict")
    data["robot_id"] = robot_id
    _live_pose[robot_id] = data
    return {"status": "ok"}


@app.get("/pose/live")
async def get_live_pose():
    return _live_pose


# ── Navigation — goal queue ───────────────────────────────────

_goal_queue: list[dict] = []


@app.post("/navigate")
async def navigate_to_point(x: float, y: float, z: float = 0.0):
    """Queue a navigation goal — nav_bridge forwards it to DimOS over LCM."""
    _goal_queue.append({"x": x, "y": y, "z": z, "ts": time.time()})
    return {"status": "queued", "target": {"x": x, "y": y, "z": z}}


@app.get("/goals/pending")
async def get_pending_goals():
    goals = list(_goal_queue)
    _goal_queue.clear()
    return {"goals": goals}


# ── Map Endpoint ──────────────────────────────────────────────

@app.get("/map", response_model=MapResponse)
async def get_map():
    store: WorldStateStore = app.state.world_store
    states = store.load_all()
    merged = store.merge()
    return MapResponse(objects=merged, robot_count=len(states), timestamp=time.time())


# ── Dashboard UI ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


# ── Dashboard HTML ────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>SpatialMind — Go2 3D Live</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script type="importmap">
  {
    "imports": {
      "three": "https://cdn.jsdelivr.net/npm/three@0.161/build/three.module.js",
      "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.161/examples/jsm/"
    }
  }
  </script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #080c10; color: #dde;
           height: 100vh; overflow: hidden; }

    header { display: flex; align-items: center; gap: 10px; padding: 9px 16px;
             background: #0d1117; border-bottom: 1px solid #1e2d3d; height: 44px; }
    header h1 { font-size: 16px; color: #4fc3f7; font-weight: 600; }
    .badge { font-size: 10px; padding: 2px 8px; border-radius: 10px;
             background: #1e3a1e; color: #81c784; border: 1px solid #2e5a2e; }
    #robot-pos { font-size: 11px; color: #4fc3f7; font-family: monospace;
                 opacity: 0.8; margin-left: 6px; }
    #conn-dot { width: 8px; height: 8px; border-radius: 50%; background: #444;
                transition: background 0.3s; margin-left: auto; }
    #conn-dot.live { background: #81c784; box-shadow: 0 0 6px #81c784; }

    /* Layout: map + draggable horizontal handle + sidebar.
       Sidebar width = --sidebar-w (default 32vw, min 280px, max 60vw).
       The user drags #h-resize to change it. */
    .layout { display: grid; height: calc(100vh - 44px);
              grid-template-columns: 1fr 5px clamp(280px, var(--sidebar-w, 32vw), 60vw); }
    @media (max-width: 768px) {
      .layout { grid-template-columns: 1fr; grid-template-rows: 45vh 5px 1fr; }
    }
    #h-resize { background: transparent; cursor: col-resize;
                border-left: 1px solid #1e2d3d; border-right: 1px solid #1e2d3d;
                transition: background 0.15s; }
    #h-resize:hover, #h-resize.active { background: rgba(79,195,247,0.3); }

    #map-panel { position: relative; background: #080c10; overflow: hidden; }
    #map-panel canvas { display: block; }
    #nav-status { position: absolute; top: 12px; left: 50%; transform: translateX(-50%);
                  font-size: 12px; color: #ffb74d; background: rgba(0,0,0,0.85);
                  padding: 5px 14px; border-radius: 14px; pointer-events: none;
                  opacity: 0; transition: opacity 0.3s; white-space: nowrap; z-index: 10; }
    #nav-status.show { opacity: 1; }
    #view-hint { position: absolute; bottom: 8px; left: 10px; font-size: 10px;
                 color: rgba(255,255,255,0.22); pointer-events: none; z-index: 10; }

    #sidebar { display: flex; flex-direction: column; border-left: 1px solid #1e2d3d;
               overflow: hidden; }
    .section { padding: 9px 12px; border-bottom: 1px solid #1e2d3d; flex-shrink: 0; }
    .section-title { font-size: 10px; font-weight: 700; color: #4fc3f7;
                     letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }

    /* Sidebar split into 2 vertical regions: chat (top) + camera/objects (bottom).
       The split is governed by a CSS variable --chat-frac (0..1) and a draggable
       horizontal handle. Default 60% chat / 40% bottom panel. */
    #sidebar { --chat-frac: 0.60; }
    /* Chat-tab section: takes (chat-frac × 100)% of the sidebar height */
    #chat-section { flex: 0 0 calc(var(--chat-frac) * 100%);
                    display: flex; flex-direction: column;
                    min-height: 0; }
    /* Vertical drag handle to resize chat vs bottom panel */
    #v-resize { flex: 0 0 5px; cursor: row-resize; background: transparent;
                border-top: 1px solid #1e2d3d; border-bottom: 1px solid #1e2d3d;
                transition: background 0.15s; }
    #v-resize:hover, #v-resize.active { background: rgba(79,195,247,0.3); }
    /* Bottom panel: camera + objects share remaining space */
    #bottom-panel { flex: 1 1 auto; display: flex; flex-direction: column;
                    min-height: 0; overflow: hidden; }
    .mode-toggle { display: flex; gap: 4px; padding: 6px 10px 0; flex-shrink: 0; }
    .mode-btn { flex: 1; padding: 5px 0; font-size: 10px; font-weight: 700;
                letter-spacing: 0.06em; text-transform: uppercase; border: none;
                border-radius: 5px; cursor: pointer; background: transparent;
                color: #556; transition: background 0.15s, color 0.15s; }
    .mode-btn.active { background: #1e2d3d; color: #4fc3f7; }
    .mode-btn:hover:not(.active) { color: #99a; }
    .chat-pane { flex: 1; overflow-y: auto; padding: 8px 12px; min-height: 0;
                 display: flex; flex-direction: column; gap: 6px; }
    .chat-pane.hidden { display: none; }
    .msg { font-size: 12px; line-height: 1.5; padding: 7px 10px; border-radius: 10px;
           max-width: 88%; word-wrap: break-word; }
    .msg b { display: block; font-size: 10px; letter-spacing: 0.05em;
             text-transform: uppercase; margin-bottom: 2px; opacity: 0.8; }
    /* User: cyan bubble, right-aligned */
    .msg-user { align-self: flex-end; background: #0d3b66; color: #e8f4fd;
                border: 1px solid #1565c0; }
    .msg-user b { color: #4fc3f7; }
    /* Ask-mode robot reply: green bubble */
    .msg-robot { align-self: flex-start; background: #1a2e1f; color: #d8f3dc;
                 border: 1px solid #2d5a3d; }
    .msg-robot b { color: #81c784; }
    /* Agent-mode reply: amber/orange to signal "this might move the robot" */
    .msg-cmd { align-self: flex-start; background: #2e1f0a; color: #ffe0b2;
               border: 1px solid #5a3d1a; }
    .msg-cmd b { color: #ffb74d; }
    .msg-cmd .tool { color: #ffd54f; }
    /* Preserve agent newlines */
    .msg span { white-space: pre-wrap; }
    /* Tool-use lines like [→ navigate_with_text({...})] rendered dim/orange */
    .msg-robot .tool {
        display: block; font-size: 10px; color: #ffb74d; font-style: italic;
        padding: 2px 0; opacity: 0.85;
    }
    #chat-input-row { display: flex; gap: 6px; padding: 8px 10px;
                      border-top: 1px solid #1e2d3d; flex-shrink: 0; }
    #q { flex: 1; padding: 7px 10px; background: #0d1117; border: 1px solid #2a3a4a;
         border-radius: 6px; color: #eee; font-size: 13px; outline: none; }
    #q:focus { border-color: #4fc3f7; }
    #send-btn { padding: 7px 12px; background: #1565c0; border: none; border-radius: 6px;
                color: #fff; cursor: pointer; font-weight: 600; font-size: 12px; }
    #send-btn:hover { background: #1976d2; }
    #mic-btn { padding: 7px 10px; background: #1a2a3a; border: 1px solid #2a3a4a;
               border-radius: 6px; color: #4fc3f7; cursor: pointer; font-size: 14px;
               transition: background 0.2s, color 0.2s; flex-shrink: 0; }
    #mic-btn:hover { background: #1e3a4a; }
    #mic-btn.listening { background: #7f1d1d; border-color: #ef4444; color: #fca5a5;
                         animation: pulse 1s infinite; }
    @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.6; } }

    /* Camera fills its parent block (now driven by the resizable bottom panel) */
    #cam-section { flex: 1 1 50%; min-height: 80px; padding: 9px 12px;
                   display: flex; flex-direction: column; overflow: hidden;
                   border-bottom: 1px solid #1e2d3d; }
    #cam-img { width: 100%; flex: 1; border-radius: 4px; background: #0a0e14;
               min-height: 80px; object-fit: cover; }
    #obj-section { flex: 1 1 50%; min-height: 80px; overflow-y: auto;
                   padding: 9px 12px; }
    table { width: 100%; border-collapse: collapse; font-size: 11px; }
    td, th { padding: 4px 6px; text-align: left; border-bottom: 1px solid #1a2530; }
    th { color: #4fc3f7; font-weight: 600; background: #0d1117; position: sticky; top: 0; }
    .hi { color: #81c784; } .lo { color: #e57373; }
    .empty-row td { color: #555; text-align: center; padding: 10px; }
  </style>
</head>
<body>

<header>
  <h1>SpatialMind</h1>
  <span class="badge">Go2 · 3D Live</span>
  <span id="robot-pos"></span>
  <div id="conn-dot"></div>
</header>

<div class="layout">
  <div id="map-panel">
    <div id="nav-status"></div>
    <div id="view-hint">drag to orbit · scroll to zoom · click floor to navigate</div>
  </div>
  <div id="h-resize" title="drag to resize sidebar"></div>
  <div id="sidebar">
    <div id="chat-section">
      <div class="mode-toggle">
        <button class="mode-btn active" id="mode-ask"
                onclick="window._setMode('ask')">🧠 Ask</button>
        <button class="mode-btn" id="mode-cmd"
                onclick="window._setMode('cmd')">🤖 Agent</button>
      </div>
      <div id="chat-ask" class="chat-pane"></div>
      <div id="chat-cmd" class="chat-pane hidden"></div>
      <div id="chat-input-row">
        <input id="q" placeholder="Where is the chair?"
               onkeydown="if(event.key==='Enter')window._chat()">
        <button id="mic-btn" onclick="window._mic()" title="Voice input">🎤</button>
        <button id="send-btn" onclick="window._chat()">Ask</button>
      </div>
    </div>
    <!-- Vertical drag handle — chat ↕ camera/objects -->
    <div id="v-resize" title="drag to resize chat vs camera/objects"></div>
    <div id="bottom-panel">
      <div id="cam-section">
        <div class="section-title">Camera</div>
        <img id="cam-img" src="/frames/go2_a/stream"
             onerror="this.style.opacity='0.12'" alt="">
      </div>
      <div id="obj-section">
        <div class="section-title">Objects in semantic memory</div>
        <table>
          <thead><tr><th>Label</th><th>Pos (m)</th><th>Conf</th></tr></thead>
          <tbody id="obj-body">
            <tr class="empty-row"><td colspan="3">Waiting for robot...</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
/* Chat — two tabs (Ask 🧠 / Agent 🤖) with fully isolated histories */
let _mode = 'ask';

window._setMode = function(mode) {
  _mode = mode;
  document.getElementById('mode-ask').classList.toggle('active', mode === 'ask');
  document.getElementById('mode-cmd').classList.toggle('active', mode === 'cmd');
  document.getElementById('chat-ask').classList.toggle('hidden', mode !== 'ask');
  document.getElementById('chat-cmd').classList.toggle('hidden', mode !== 'cmd');
  const q = document.getElementById('q');
  q.placeholder = mode === 'ask'
    ? 'Where is the chair? · How many doors do you see?'
    : 'Go to the door · explore the room · stop';
  document.getElementById('send-btn').textContent =
    mode === 'ask' ? 'Ask' : 'Send';
  q.focus();
};

window._chat = async function() {
  const inp = document.getElementById('q');
  const q = inp.value.trim();
  if (!q) return;
  const isCmd = _mode === 'cmd';
  const paneId = isCmd ? 'chat-cmd' : 'chat-ask';
  const pane = document.getElementById(paneId);
  const endpoint = isCmd ? '/command/stream' : '/query/stream';

  _msg(paneId, 'You', q, 'msg-user');
  inp.value = '';
  const el = _msg(paneId, isCmd ? 'Agent' : 'Go2', '',
                  isCmd ? 'msg-cmd' : 'msg-robot');
  const span = el.querySelector('span');

  let buffer = '';
  function flushText() {
    if (buffer) { span.appendChild(document.createTextNode(buffer)); buffer = ''; }
  }
  function appendChunk(s) {
    // Pull out [→ tool(...)] lines and render them as orange italic blocks.
    const re = /\\[→ [^\\]]*\\]/g;
    let last = 0, m;
    while ((m = re.exec(s)) !== null) {
      buffer += s.slice(last, m.index);
      flushText();
      const tool = document.createElement('span');
      tool.className = 'tool';
      tool.textContent = m[0];
      span.appendChild(tool);
      last = m.index + m[0].length;
    }
    buffer += s.slice(last);
    flushText();
  }
  try {
    const resp = await fetch(endpoint, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: q})
    });
    const reader = resp.body.getReader(), dec = new TextDecoder();
    let leftover = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      const chunk = leftover + dec.decode(value, {stream: true});
      const lines = chunk.split('\\n');
      leftover = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const tok = line.slice(6);
        if (tok === '[DONE]') return;
        try {
          appendChunk(JSON.parse(tok));
          pane.scrollTop = 99999;
        } catch(e) {}
      }
    }
  } catch(e) {
    span.appendChild(document.createTextNode(' [error: ' + e.message + ']'));
  }
};

function _msg(paneId, who, text, cls) {
  const pane = document.getElementById(paneId);
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  const b = document.createElement('b');
  b.textContent = who;
  const s = document.createElement('span');
  s.textContent = text;
  d.appendChild(b); d.appendChild(s);
  pane.appendChild(d); pane.scrollTop = 99999;
  return d;
}

window._mic = (function() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  function browserFallback() {
    if (!SR) {
      alert('Voice input unavailable: no MediaRecorder and no browser speech recognition.');
      return;
    }
    const rec = new SR();
    rec.lang = 'en-US';
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    rec.onresult = function(e) {
      const text = e.results[0][0].transcript;
      document.getElementById('q').value = text;
      window._chat();
    };
    rec.onend = function() {
      document.getElementById('mic-btn').classList.remove('listening');
    };
    rec.onerror = rec.onend;
    document.getElementById('mic-btn').classList.add('listening');
    rec.start();
  }

  if (!navigator.mediaDevices || !window.MediaRecorder) {
    return browserFallback;
  }

  let recorder = null, chunks = [], stream = null, stopTimer = null;

  async function finishRecording() {
    const btn = document.getElementById('mic-btn');
    btn.classList.remove('listening');
    btn.disabled = true;
    try {
      const blob = new Blob(chunks, {type: (recorder && recorder.mimeType) || 'audio/webm'});
      const resp = await fetch('/speech/transcribe', {
        method: 'POST',
        headers: {'Content-Type': blob.type || 'audio/webm'},
        body: blob
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || 'transcription failed');
      const text = (data.text || '').trim();
      if (text) {
        document.getElementById('q').value = text;
        window._chat();
      }
    } catch (e) {
      console.warn('AWS speech transcription failed; falling back to browser STT', e);
      browserFallback();
    } finally {
      if (stream) stream.getTracks().forEach(function(t) { t.stop(); });
      recorder = null; stream = null; chunks = [];
      btn.disabled = false;
    }
  }

  return async function() {
    if (recorder && recorder.state === 'recording') {
      recorder.stop();
      return;
    }
    try {
      chunks = [];
      stream = await navigator.mediaDevices.getUserMedia({audio: true});
      const options = MediaRecorder.isTypeSupported('audio/webm')
        ? {mimeType: 'audio/webm'} : {};
      recorder = new MediaRecorder(stream, options);
      recorder.ondataavailable = function(e) {
        if (e.data && e.data.size) chunks.push(e.data);
      };
      recorder.onstop = finishRecording;
      document.getElementById('mic-btn').classList.add('listening');
      recorder.start();
      clearTimeout(stopTimer);
      stopTimer = setTimeout(function() {
        if (recorder && recorder.state === 'recording') recorder.stop();
      }, 9000);
    } catch (e) {
      browserFallback();
    }
  };
})();

/* ── Resizable splits ────────────────────────────────────────────
   #h-resize  (vertical bar between map and sidebar)  →  --sidebar-w
   #v-resize  (horizontal bar inside sidebar between chat and bottom panel)
              →  #sidebar.style.--chat-frac
*/
(function setupResizers() {
  const layout = document.querySelector('.layout');
  const sidebar = document.getElementById('sidebar');
  const hHandle = document.getElementById('h-resize');
  const vHandle = document.getElementById('v-resize');

  let dragH = false, dragV = false;

  hHandle.addEventListener('pointerdown', function(e) {
    dragH = true; hHandle.classList.add('active');
    document.body.style.userSelect = 'none';
    hHandle.setPointerCapture(e.pointerId);
  });
  hHandle.addEventListener('pointermove', function(e) {
    if (!dragH) return;
    // Distance from right edge of viewport in px → set as sidebar width.
    const w = Math.max(280, Math.min(window.innerWidth - e.clientX,
                                     window.innerWidth * 0.60));
    layout.style.setProperty('--sidebar-w', w + 'px');
  });
  hHandle.addEventListener('pointerup', function(e) {
    dragH = false; hHandle.classList.remove('active');
    document.body.style.userSelect = '';
    try { hHandle.releasePointerCapture(e.pointerId); } catch(_) {}
  });

  vHandle.addEventListener('pointerdown', function(e) {
    dragV = true; vHandle.classList.add('active');
    document.body.style.userSelect = 'none';
    vHandle.setPointerCapture(e.pointerId);
  });
  vHandle.addEventListener('pointermove', function(e) {
    if (!dragV) return;
    const rect = sidebar.getBoundingClientRect();
    // Position of pointer within sidebar (0..1) — leaves 80px headroom each side.
    const frac = Math.max(0.18,
                          Math.min(0.85, (e.clientY - rect.top) / rect.height));
    sidebar.style.setProperty('--chat-frac', frac.toFixed(3));
  });
  vHandle.addEventListener('pointerup', function(e) {
    dragV = false; vHandle.classList.remove('active');
    document.body.style.userSelect = '';
    try { vHandle.releasePointerCapture(e.pointerId); } catch(_) {}
  });
})();
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Renderer ──────────────────────────────────────────────────
const panel = document.getElementById('map-panel');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(panel.clientWidth, panel.clientHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
panel.appendChild(renderer.domElement);

// ── Scene ─────────────────────────────────────────────────────
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x080c10);
scene.fog = new THREE.FogExp2(0x080c10, 0.03);

// ── Camera ────────────────────────────────────────────────────
const camera = new THREE.PerspectiveCamera(55, panel.clientWidth / panel.clientHeight, 0.01, 400);
camera.position.set(0, 10, 14);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.5;
controls.maxPolarAngle = Math.PI * 0.47;
controls.minDistance = 0.3;
controls.maxDistance = 150;

// Auto-rotate the camera as a hint, but stop **permanently** after the
// first user interaction (drag, scroll, click) — no resume timer.
let _userInteracted = false;
function _stopAutoRotate() {
  if (_userInteracted) return;
  _userInteracted = true;
  controls.autoRotate = false;
}
renderer.domElement.addEventListener('pointerdown', _stopAutoRotate);
renderer.domElement.addEventListener('wheel', _stopAutoRotate, { passive: true });

// ── Lights ────────────────────────────────────────────────────
scene.add(new THREE.HemisphereLight(0x334d66, 0x0a1020, 4.5));
const sun = new THREE.DirectionalLight(0xbbd0ee, 6.5);
sun.position.set(20, 40, 25);
scene.add(sun);
const fill = new THREE.DirectionalLight(0x4466aa, 2.0);
fill.position.set(-15, 10, -15);
scene.add(fill);
const rim = new THREE.DirectionalLight(0x4fc3f7, 1.5);
rim.position.set(0, -8, 20);
scene.add(rim);

// ── Floor & grid ──────────────────────────────────────────────
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(400, 400),
  new THREE.MeshStandardMaterial({ color: 0x0a1018, roughness: 1 })
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.002;
scene.add(floor);
scene.add(new THREE.GridHelper(400, 400, 0x1c3550, 0x152840));

// ── Sprite texture for point clouds (lidar uses this) ─────────
function makeSpriteTex() {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0,   'rgba(255,255,255,1.0)');
  g.addColorStop(0.45,'rgba(255,255,255,0.9)');
  g.addColorStop(1,   'rgba(255,255,255,0.0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

let cameraFitted = false;

// ── Robot — procedural Go2-style quadruped ─────────────────────
// Local axes: +X = forward (heading), +Y = up, +Z = right.
const robotGrp = new THREE.Group();

const matBody  = new THREE.MeshStandardMaterial({ color: 0x222831, roughness: 0.5, metalness: 0.4 });
const matLight = new THREE.MeshStandardMaterial({ color: 0x4fc3f7, emissive: 0x4fc3f7, emissiveIntensity: 0.6 });
const matLeg   = new THREE.MeshStandardMaterial({ color: 0x111418, roughness: 0.7 });

// Body
const body = new THREE.Mesh(new THREE.BoxGeometry(0.50, 0.16, 0.22), matBody);
body.position.y = 0.0;
robotGrp.add(body);
// Head/snout
const head = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.12, 0.16), matBody);
head.position.set(0.30, 0.04, 0.0);
robotGrp.add(head);
// Eyes (forward indicator + heading proxy)
const eyeL = new THREE.Mesh(new THREE.SphereGeometry(0.025, 10, 8), matLight);
eyeL.position.set(0.385, 0.06, 0.05); robotGrp.add(eyeL);
const eyeR = new THREE.Mesh(new THREE.SphereGeometry(0.025, 10, 8), matLight);
eyeR.position.set(0.385, 0.06, -0.05); robotGrp.add(eyeR);
// Forward LED bar
const led = new THREE.Mesh(new THREE.BoxGeometry(0.005, 0.015, 0.10), matLight);
led.position.set(0.392, 0.02, 0.0); robotGrp.add(led);
// Legs — 4 cylinders (front-left, front-right, back-left, back-right)
function _addLeg(x, z) {
  const g = new THREE.Group();
  const upper = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.16, 8), matLeg);
  upper.position.y = -0.16; g.add(upper);
  const foot = new THREE.Mesh(new THREE.SphereGeometry(0.035, 10, 8), matLeg);
  foot.position.y = -0.27; g.add(foot);
  g.position.set(x, -0.08, z);
  robotGrp.add(g);
  return g;
}
const legFL = _addLeg( 0.20,  0.10);
const legFR = _addLeg( 0.20, -0.10);
const legBL = _addLeg(-0.20,  0.10);
const legBR = _addLeg(-0.20, -0.10);
// Soft glow ring underneath
const ringMesh = new THREE.Mesh(
  new THREE.RingGeometry(0.32, 0.45, 36),
  new THREE.MeshBasicMaterial({ color: 0x4fc3f7, transparent: true, opacity: 0.22, side: THREE.DoubleSide })
);
ringMesh.rotation.x = -Math.PI / 2;
ringMesh.position.y = -0.30;
robotGrp.add(ringMesh);

robotGrp.visible = false;
scene.add(robotGrp);

let _robotX = 0, _robotY = 0, _robotZ = 0;
let _lastRobotX = 0, _lastRobotY = 0, _robotSpeed = 0;
function updateRobot(rp) {
  // rp = {x, y, z, yaw, pitch, roll, qx, qy, qz, qw, ts} from /pose/live
  if (!rp || rp.x === undefined) return;
  _lastRobotX = _robotX;
  _lastRobotY = _robotY;
  _robotX = rp.x;
  _robotY = rp.y;
  _robotZ = rp.z || 0;
  _robotSpeed = Math.hypot(_robotX - _lastRobotX, _robotY - _lastRobotY);
  // Three.js: world axes are X=east, Y=up, Z=south.  We use X=robot-X, Z=-robot-Y.
  // Yaw rotation about world Y mirrors the robot's heading in the (x,y) plane.
  const yaw = (rp.yaw !== undefined) ? rp.yaw : 0;
  robotGrp.rotation.set(0, -yaw, 0);
  robotGrp.visible = true;
  document.getElementById('robot-pos').textContent =
    '(' + _robotX.toFixed(2) + ', ' + _robotY.toFixed(2) +
    ') · ψ ' + (yaw * 180 / Math.PI).toFixed(0) + '°';
  // First-time camera framing once we have a real pose
  if (!cameraFitted) {
    cameraFitted = true;
    controls.target.set(_robotX, 0.3, -_robotY);
    camera.position.set(_robotX + 4, 4, -_robotY + 4);
    controls.update();
  }
  updateGoalLine();
}

// ── Semantic objects ──────────────────────────────────────────
const objPool = [];
const oGeo = new THREE.SphereGeometry(0.12, 8, 6);
const C_HI = new THREE.Color(0x66bb6a);
const C_LO = new THREE.Color(0xef9a9a);

for (let i = 0; i < 80; i++) {
  const m = new THREE.Mesh(
    oGeo,
    new THREE.MeshStandardMaterial({ roughness: 0.4 })
  );
  m.visible = false;
  scene.add(m);
  objPool.push(m);
}

function updateObjects(objs) {
  const disp = objs.filter(function(o) {
    return o.pose && o.label !== 'robot_position' && o.label.indexOf('nav_goal') !== 0;
  });
  for (let i = 0; i < objPool.length; i++) {
    if (i < disp.length) {
      const o = disp[i];
      objPool[i].position.set(o.pose.x, 0.15, -o.pose.y);
      const col = (o.confidence || 0) > 0.7 ? C_HI : C_LO;
      objPool[i].material.color.copy(col);
      objPool[i].material.emissive.copy(col).multiplyScalar(0.15);
      objPool[i].visible = true;
    } else {
      objPool[i].visible = false;
    }
  }
  const tbody = document.getElementById('obj-body');
  if (!disp.length) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="3">No objects</td></tr>';
    return;
  }
  tbody.innerHTML = disp.map(function(o) {
    const c = ((o.confidence || 0) * 100).toFixed(0);
    const cls = c > 70 ? 'hi' : 'lo';
    return '<tr><td>' + o.label + '</td>'
         + '<td>(' + (o.pose.x || 0).toFixed(1) + ', ' + (o.pose.y || 0).toFixed(1) + ')</td>'
         + '<td class="' + cls + '">' + c + '%</td></tr>';
  }).join('');
}

// ── Goal/path overlay ────────────────────────────────────────
// DimOS is not currently publishing a planner path topic on LCM here. Until it
// does, show the commanded route segment in amber from current pose to goal.
const goalLineGeo = new THREE.BufferGeometry();
const goalLine = new THREE.Line(
  goalLineGeo,
  new THREE.LineBasicMaterial({ color: 0xffb74d, transparent: true, opacity: 0.95 })
);
goalLine.visible = false;
scene.add(goalLine);

const goalMarker = new THREE.Mesh(
  new THREE.RingGeometry(0.18, 0.28, 32),
  new THREE.MeshBasicMaterial({ color: 0xffb74d, transparent: true, opacity: 0.9, side: THREE.DoubleSide })
);
goalMarker.rotation.x = -Math.PI / 2;
goalMarker.visible = false;
scene.add(goalMarker);

let _activeGoal = null;
function updateGoalLine() {
  if (!_activeGoal) return;
  const dist = Math.hypot(_activeGoal.x - _robotX, _activeGoal.y - _robotY);
  if (dist < 0.25) {
    _activeGoal = null;
    goalLine.visible = false;
    goalMarker.visible = false;
    return;
  }
  goalLineGeo.setFromPoints([
    new THREE.Vector3(_robotX, 0.08, -_robotY),
    new THREE.Vector3(_activeGoal.x, 0.08, -_activeGoal.y),
  ]);
  goalMarker.position.set(_activeGoal.x, 0.04, -_activeGoal.y);
  goalLine.visible = true;
  goalMarker.visible = true;
}

// ── Click → navigate ──────────────────────────────────────────
const ray = new THREE.Raycaster();
const m2 = new THREE.Vector2();
const navPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const navPt = new THREE.Vector3();

// Distinguish a true click (for nav goal) from a drag (for camera rotate).
// Only fire when pointerup happens within ~5 px of pointerdown AND under 350 ms.
let _downX = 0, _downY = 0, _downT = 0, _downBtn = -1;
const CLICK_PX = 6;
const CLICK_MS = 350;

renderer.domElement.addEventListener('pointerdown', function(e) {
  _downX = e.clientX; _downY = e.clientY; _downT = Date.now(); _downBtn = e.button;
});
renderer.domElement.addEventListener('pointerup', async function(e) {
  if (_downBtn !== 0 || e.button !== 0) return;            // primary button only
  const dx = e.clientX - _downX, dy = e.clientY - _downY;
  const dist = Math.hypot(dx, dy);
  const dt = Date.now() - _downT;
  if (dist > CLICK_PX || dt > CLICK_MS) return;            // it was a drag, not a click
  const r = renderer.domElement.getBoundingClientRect();
  m2.x =  ((e.clientX - r.left) / r.width)  * 2 - 1;
  m2.y = -((e.clientY - r.top)  / r.height) * 2 + 1;
  ray.setFromCamera(m2, camera);
  if (!ray.ray.intersectPlane(navPlane, navPt)) return;
  const wx = navPt.x, wy = -navPt.z;
  _showNav('Sending (' + wx.toFixed(2) + ', ' + wy.toFixed(2) + ')...');
  try {
    const res = await fetch('/navigate?x=' + wx.toFixed(3) + '&y=' + wy.toFixed(3), {method: 'POST'});
    const d = await res.json();
    _showNav(res.ok
      ? 'Goal sent (' + wx.toFixed(2) + ', ' + wy.toFixed(2) + ')'
      : 'Failed: ' + (d.detail || 'error'), !res.ok);
    if (res.ok) {
      _activeGoal = {x: wx, y: wy};
      updateGoalLine();
    }
  } catch(err) { _showNav('Error: ' + err.message, true); }
});

function _showNav(msg, err) {
  const el = document.getElementById('nav-status');
  el.textContent = msg;
  el.style.color = err ? '#ef9a9a' : '#ffb74d';
  el.classList.add('show');
  clearTimeout(el._t);
  el._t = setTimeout(function() { el.classList.remove('show'); }, 3000);
}

// ── LiDAR point cloud renderer ───────────────────────────────
const lidarGeo = new THREE.BufferGeometry();
const lidarPts = new THREE.Points(lidarGeo, new THREE.PointsMaterial({
  size: 0.12,
  sizeAttenuation: true,
  vertexColors: true,
  transparent: true,
  opacity: 0.92,
  map: makeSpriteTex(),
  alphaTest: 0.04,
  depthWrite: false,
}));
scene.add(lidarPts);

// Turbo-like 5-stop palette for height: blue → cyan → green → amber → red
const _LC = [
  new THREE.Color(0x4040ff),
  new THREE.Color(0x00d4ff),
  new THREE.Color(0x39ff14),
  new THREE.Color(0xffd000),
  new THREE.Color(0xff3300),
];
function lidarColor(t) {
  const s = Math.min(1, Math.max(0, t)) * (_LC.length - 1);
  const i = Math.min(_LC.length - 2, Math.floor(s));
  return new THREE.Color().lerpColors(_LC[i], _LC[i + 1], s - i);
}

let _lastPcTs = 0;
function updateLidar(data) {
  // data: {b64, n, z_min, z_max, timestamp} from /pointcloud/live
  if (!data || data.timestamp === _lastPcTs) return;
  _lastPcTs = data.timestamp;

  const n    = data.n;
  const zMin = data.z_min;  // int16 cm
  const zMax = data.z_max;
  const zRng = Math.max(1, zMax - zMin);

  // Decode base64 → signed Int16Array (little-endian, matches Python struct '<hhh')
  const bin = atob(data.b64);
  const ab  = new ArrayBuffer(bin.length);
  const u8v = new Uint8Array(ab);
  for (let i = 0; i < bin.length; i++) u8v[i] = bin.charCodeAt(i);
  const i16 = new Int16Array(ab);

  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const xM =  i16[i * 3]     / 100;
    const yM =  i16[i * 3 + 1] / 100;
    const zCm = i16[i * 3 + 2];
    const zM  = zCm / 100;
    pos[i * 3]     = xM;
    pos[i * 3 + 1] = zM;   // z → Three.js Y (height)
    pos[i * 3 + 2] = -yM;  // y → Three.js -Z
    const c = lidarColor((zCm - zMin) / zRng);
    col[i * 3]     = c.r;
    col[i * 3 + 1] = c.g;
    col[i * 3 + 2] = c.b;
  }

  lidarGeo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  lidarGeo.setAttribute('color',    new THREE.Float32BufferAttribute(col, 3));

  // Frame the camera once on the first pointcloud arrival, if we don't have pose yet.
  if (!cameraFitted) {
    cameraFitted = true;
    controls.target.set(_robotX, 0.3, -_robotY);
    camera.position.set(_robotX + 4, 4, -_robotY + 4);
    controls.update();
  }
}

// ── Polling — direct from MCP/dimos LCM (via cloud bridges) ──
//
//   /pose/live        — nav_bridge pushes from /odom @ ~15 Hz
//   /pointcloud/live  — pc_bridge  pushes from /lidar @ ~2 Hz
//   /map              — semantic objects from dimos_bridge / /ingest
//
let _lastPoseTs = 0;

async function pollPose() {
  try {
    const d = await (await fetch('/pose/live')).json();
    const entry = d['go2_a'] || Object.values(d)[0];
    if (entry && entry.ts !== _lastPoseTs) {
      _lastPoseTs = entry.ts;
      updateRobot(entry);
      document.getElementById('conn-dot').classList.add('live');
    }
  } catch(e) {}
}

async function pollSem() {
  try {
    const d = await (await fetch('/map')).json();
    updateObjects(d.objects || []);
  } catch(e) {}
}

async function pollPointcloud() {
  try {
    const d = await (await fetch('/pointcloud/live')).json();
    const entry = d['go2_a'] || Object.values(d)[0];
    if (entry) updateLidar(entry);
  } catch(e) {}
}

// Pose at ~10 Hz so the marker tracks live; pointcloud at 4 Hz; objects at 0.5 Hz.
setInterval(pollPose,       100);
setInterval(pollPointcloud, 250);
setInterval(pollSem,       2000);
pollPose(); pollPointcloud(); pollSem();

// ── Resize: ResizeObserver covers BOTH window resize AND sidebar drags ──
function _fitRendererToPanel() {
  const w = panel.clientWidth, h = panel.clientHeight;
  if (!w || !h) return;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}
window.addEventListener('resize', _fitRendererToPanel);
new ResizeObserver(_fitRendererToPanel).observe(panel);

// ── Render loop ───────────────────────────────────────────────
(function animate() {
  requestAnimationFrame(animate);
  controls.update();

  if (robotGrp.visible) {
    const t = Date.now() * 0.001;
    // Body height tuned to dog stance: ~0.30m above ground.
    robotGrp.position.set(_robotX, 0.30, -_robotY);
    ringMesh.material.opacity = 0.18 + Math.sin(t * 2.8) * 0.12;
    ringMesh.scale.setScalar(1.0 + Math.sin(t * 1.6) * 0.10);
    const moving = Math.min(1, _robotSpeed * 18);
    const gait = Math.sin(t * 10) * 0.08 * moving;
    legFL.rotation.z =  gait; legBR.rotation.z =  gait;
    legFR.rotation.z = -gait; legBL.rotation.z = -gait;
    goalMarker.rotation.z += 0.025;
  }

  renderer.render(scene, camera);
})();
</script>

</body>
</html>"""


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="SpatialMind cloud server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    parser.add_argument(
        "--server-only", action="store_true",
        help="Disable auto-spawning of local LCM bridges (use on cloud/EC2)"
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="Enable uvicorn auto-reload (development only)"
    )
    args = parser.parse_args()

    if args.server_only:
        os.environ["AUTO_BRIDGES"] = "false"

    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)
