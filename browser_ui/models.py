"""
Pydantic data models for the AWS Cloud Middleware.

Defines the S3 schema contract and API request/response models
for Robot State Ingestion (Req 1) and World State Query (Req 2).
"""

import time
from pydantic import BaseModel, Field


# ── Core Data Models (S3 Schema) ─────────────────────────────


class Pose(BaseModel):
    """3D position in meters relative to robot's map origin."""
    x: float
    y: float
    z: float = 0.0


class DetectedObject(BaseModel):
    """A single object detected by the robot's perception pipeline."""
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    pose: Pose
    seen_count: int = 1
    last_seen: float = Field(default_factory=time.time)
    source: str = "yolo"


class WorldState(BaseModel):
    """
    Complete world state for a single robot.
    Persisted as JSON in S3 at {robot_id}/world.json.
    """
    robot_id: str
    timestamp: float = Field(default_factory=time.time)
    objects: list[DetectedObject]


# ── API Request Models ────────────────────────────────────────


class IngestRequest(BaseModel):
    """POST /ingest — robot pushes detections to the cloud."""
    robot_id: str
    objects: list[DetectedObject]


class QueryRequest(BaseModel):
    """POST /query/stream — user asks a natural language question."""
    text: str
    robot_id: str | None = None  # optional: filter to specific robot


# ── API Response Models ───────────────────────────────────────


class IngestResponse(BaseModel):
    """Response from POST /ingest."""
    status: str = "saved"
    robot_id: str
    count: int
    timestamp: float


class MapResponse(BaseModel):
    """Response from GET /map — merged world state."""
    objects: list[DetectedObject]
    robot_count: int
    timestamp: float


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str
    message: str
    details: dict | None = None
