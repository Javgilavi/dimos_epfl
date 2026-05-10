"""
World State Store — abstracts persistence of robot world state.

Supports two backends:
- S3 (production): reads/writes JSON to S3 bucket
- In-memory (dev/test): dict-based, no AWS credentials needed

Usage:
    # Production (needs AWS creds)
    store = WorldStateStore(bucket="robohack-map-yourname")

    # Development (no AWS needed)
    store = WorldStateStore(bucket="test", use_memory=True)
"""
from __future__ import annotations

import json
import math
import os
import time
import logging
from typing import Protocol

from models import WorldState, DetectedObject

logger = logging.getLogger(__name__)


class StorageBackend(Protocol):
    """Protocol for swappable storage backends."""

    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes | None: ...
    def list_keys(self, prefix: str = "", suffix: str = "") -> list[str]: ...


class MemoryBackend:
    """In-memory storage backend for development and testing."""

    def __init__(self):
        self._store: dict[str, bytes] = {}

    def put(self, key: str, data: bytes) -> None:
        self._store[key] = data

    def get(self, key: str) -> bytes | None:
        return self._store.get(key)

    def list_keys(self, prefix: str = "", suffix: str = "") -> list[str]:
        return [
            k for k in self._store
            if k.startswith(prefix) and k.endswith(suffix)
        ]


class S3Backend:
    """S3 storage backend for production use."""

    def __init__(self, bucket: str, region: str = "eu-west-1", s3_client=None):
        self.bucket = bucket
        if s3_client is None:
            import boto3
            self._s3 = boto3.client("s3", region_name=region)
        else:
            self._s3 = s3_client

    def put(self, key: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self.bucket, Key=key, Body=data)

    def get(self, key: str) -> bytes | None:
        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()
        except self._s3.exceptions.NoSuchKey:
            return None
        except Exception as e:
            logger.warning(f"S3 get failed for {key}: {e}")
            return None

    def list_keys(self, prefix: str = "", suffix: str = "") -> list[str]:
        try:
            resp = self._s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            keys = [obj["Key"] for obj in resp.get("Contents", [])]
            if suffix:
                keys = [k for k in keys if k.endswith(suffix)]
            return keys
        except Exception as e:
            logger.warning(f"S3 list failed: {e}")
            return []


class WorldStateStore:
    """
    Manages reading and writing of WorldState objects.

    Each robot's state is stored at: {robot_id}/world.json
    """

    def __init__(self, bucket: str, use_memory: bool = False,
                 region: str = "eu-west-1", s3_client=None):
        if use_memory:
            self._backend: StorageBackend = MemoryBackend()
        else:
            self._backend = S3Backend(bucket, region, s3_client)

    def save(self, state: WorldState) -> None:
        """Persist a robot's world state."""
        key = f"{state.robot_id}/world.json"
        data = json.dumps(state.model_dump()).encode()
        self._backend.put(key, data)

    def load(self, robot_id: str) -> WorldState | None:
        """Load a single robot's world state."""
        key = f"{robot_id}/world.json"
        data = self._backend.get(key)
        if data is None:
            return None
        try:
            return WorldState.model_validate(json.loads(data))
        except Exception as e:
            logger.warning(f"Failed to parse world state for {robot_id}: {e}")
            return None

    def load_all(self) -> list[WorldState]:
        """Load world state from all robots."""
        keys = self._backend.list_keys(suffix="world.json")
        states = []
        for key in keys:
            data = self._backend.get(key)
            if data is not None:
                try:
                    states.append(WorldState.model_validate(json.loads(data)))
                except Exception as e:
                    logger.warning(f"Skipping corrupted state at {key}: {e}")
        return states

    def merge(self) -> list[DetectedObject]:
        """Merge all robots' world states and apply DynaMem-style temporal decay.

        Confidence decays exponentially with time since last_seen.  Objects
        whose effective confidence falls below the minimum threshold are dropped
        so the LLM never reasons over stale detections as if they were current.

        Tunables (env vars):
          OBJECT_DECAY_RATE   — fraction of confidence lost per second (default 0.0002).
                                At the default, a detection loses ~50% confidence
                                after ~57 minutes and is dropped after ~2 hours.
          OBJECT_MIN_CONFIDENCE — drop objects below this effective confidence
                                  (default 0.10).
        """
        decay_rate   = float(os.environ.get("OBJECT_DECAY_RATE",      "0.0002"))
        min_conf     = float(os.environ.get("OBJECT_MIN_CONFIDENCE",  "0.10"))
        now = time.time()

        states = self.load_all()
        result: list[DetectedObject] = []
        for state in states:
            for obj in state.objects:
                age_s = max(0.0, now - obj.last_seen)
                effective_conf = obj.confidence * math.exp(-decay_rate * age_s)
                if effective_conf < min_conf:
                    continue
                obj_copy = obj.model_copy(deep=True)
                obj_copy.confidence = round(effective_conf, 4)
                result.append(obj_copy)
        return result
