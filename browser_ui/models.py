"""
Pydantic data models for the AWS Cloud Middleware.

Defines the S3 schema contract and API request/response models
for Robot State Ingestion (Req 1) and World State Query (Req 2).
"""

import time
from typing import List, Optional
from pydantic import BaseModel, Field


# ── Core Data Models (S3 Schema) ─────────────────────────────


class Pose(BaseModel):
    """3D position in meters relative to robot's map origin."""
    x: float
    y: float
    z: float = 0.0


class DetectedObject(BaseModel):
    """A single object detected by the robot's perception pipeline.

    Base fields come from the Jetson Nano YOLO11 pipeline (on-device inference).
    Enriched fields (category, semantic_tags, spatial_relations, state flags) are
    computed server-side by taxonomy.py after every /ingest call so that the LLM
    chat boxes receive semantically grounded context without adding Jetson latency.
    """
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    pose: Pose
    seen_count: int = 1
    last_seen: float = Field(default_factory=time.time)
    source: str = "yolo"

    # ── Taxonomy enrichment (set by taxonomy.enrich_object) ──────────────────
    category: str = "object"
    semantic_tags: List[str] = Field(default_factory=list)
    is_blocking_path: bool = False
    is_dynamic: bool = False
    is_fragile: bool = False
    spatial_relations: List[str] = Field(default_factory=list)

    # ── Optional visual context (sent from Jetson when available) ─────────────
    # JPEG crop of the detection, base-64 encoded.  Used by the Bedrock vision
    # path to resolve ambiguous labels (confidence < 0.7).
    image_crop_b64: Optional[str] = None
    # Bounding box [x1, y1, x2, y2] in pixels, relative to the source frame.
    bbox: Optional[List[float]] = None


class WorldState(BaseModel):
    """
    Complete world state for a single robot.
    Persisted as JSON in S3 at {robot_id}/world.json.
    """
    robot_id: str
    timestamp: float = Field(default_factory=time.time)
    objects: List[DetectedObject]


# ── API Request Models ────────────────────────────────────────


class IngestRequest(BaseModel):
    """POST /ingest — robot pushes detections to the cloud."""
    robot_id: str
    objects: List[DetectedObject]


class QueryRequest(BaseModel):
    """POST /query/stream — user asks a natural language question."""
    text: str
    robot_id: Optional[str] = None  # optional: filter to specific robot


# ── API Response Models ───────────────────────────────────────


class IngestResponse(BaseModel):
    """Response from POST /ingest."""
    status: str = "saved"
    robot_id: str
    count: int
    timestamp: float


class MapResponse(BaseModel):
    """Response from GET /map — merged world state."""
    objects: List[DetectedObject]
    robot_count: int
    timestamp: float


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str
    message: str
    details: Optional[dict] = None
