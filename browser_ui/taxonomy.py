"""YOLO11 label taxonomy for the browser_ui semantic memory.

Ported from go2_dimos/memory/updater.py and adapted for the cloud middleware.

Enrichment runs server-side on every detection that arrives via /ingest from
the Jetson Nano YOLO11 pipeline.  The Jetson handles raw image inference
(lowest latency, on-device); this module enriches the resulting label strings
with category, semantic tags, state flags, and inter-object spatial relations
before the data is stored and served to the LLM chat boxes.
"""

from __future__ import annotations

import math


# ── Label taxonomy ──────────────────────────────────────────────────────────

_LABEL_TO_CATEGORY: dict[str, str] = {
    # People
    "person": "person",
    # Furniture
    "chair": "furniture", "sofa": "furniture", "couch": "furniture",
    "bed": "furniture", "desk": "furniture", "table": "furniture",
    "bench": "furniture", "bookshelf": "furniture", "shelf": "furniture",
    "cabinet": "furniture", "wardrobe": "furniture",
    # Personal items
    "backpack": "personal_item", "handbag": "personal_item",
    "suitcase": "personal_item", "umbrella": "personal_item",
    "hat": "personal_item", "tie": "personal_item",
    # Containers / tableware
    "bottle": "container", "cup": "container", "bowl": "container",
    "wine glass": "container", "can": "container", "vase": "container",
    # Electronics
    "laptop": "electronics", "monitor": "electronics",
    "keyboard": "electronics", "mouse": "electronics",
    "cell phone": "electronics", "phone": "electronics",
    "tablet": "electronics", "tv": "electronics",
    "remote": "electronics", "clock": "electronics",
    # Vehicles
    "car": "vehicle", "bicycle": "vehicle", "motorcycle": "vehicle",
    "truck": "vehicle", "bus": "vehicle",
    # Infrastructure
    "door": "infrastructure", "window": "infrastructure",
    "wall": "infrastructure", "floor": "infrastructure",
    "ceiling": "infrastructure", "stairs": "infrastructure",
    "elevator": "infrastructure",
    # Sports
    "sports ball": "sports", "skateboard": "sports", "surfboard": "sports",
    # Nature
    "plant": "nature", "potted plant": "nature",
    # Obstacles
    "fire hydrant": "obstacle", "stop sign": "obstacle",
    "traffic light": "obstacle",
}

_CATEGORY_TO_TAGS: dict[str, list[str]] = {
    "person":         ["dynamic", "person", "agent", "obstacle"],
    "furniture":      ["static", "furniture", "obstacle"],
    "personal_item":  ["movable", "personal_item", "soft_object"],
    "container":      ["movable", "container"],
    "electronics":    ["movable", "electronics", "fragile"],
    "vehicle":        ["dynamic", "vehicle", "large_obstacle"],
    "infrastructure": ["static", "infrastructure", "fixed"],
    "sports":         ["movable", "sports_equipment"],
    "nature":         ["static", "nature", "organic"],
    "obstacle":       ["static", "obstacle", "fixed"],
    "object":         ["movable", "object"],
}

_DYNAMIC_CATEGORIES  = {"person", "vehicle"}
_FRAGILE_CATEGORIES  = {"electronics"}
_BLOCKING_CATEGORIES = {"furniture", "infrastructure", "vehicle", "obstacle"}

# ── Spatial relation tunables ────────────────────────────────────────────────

RELATION_DISTANCE: float = 1.5   # metres — pairs within this get a relation label
_ABOVE_Z_THRESHOLD: float = 0.3  # metres — z-delta to call one object "above" another


# ── Public helpers ───────────────────────────────────────────────────────────

def category_for(label: str) -> str:
    """Return the category string for a YOLO11 label (default: 'object')."""
    return _LABEL_TO_CATEGORY.get(label.lower(), "object")


def tags_for(category: str) -> list[str]:
    """Return semantic tags list for a given category."""
    return list(_CATEGORY_TO_TAGS.get(category, ["movable", "object"]))


def is_dynamic(category: str) -> bool:
    return category in _DYNAMIC_CATEGORIES


def is_fragile(category: str) -> bool:
    return category in _FRAGILE_CATEGORIES


def is_blocking(category: str) -> bool:
    return category in _BLOCKING_CATEGORIES


def enrich_object(obj: "DetectedObject") -> "DetectedObject":  # type: ignore[name-defined]
    """Apply taxonomy metadata in-place and return the same object.

    Sets category, semantic_tags, is_dynamic, is_fragile, is_blocking_path.
    Does NOT compute spatial_relations — call compute_spatial_relations() after
    all objects in the batch have been enriched.
    """
    cat = category_for(obj.label)
    obj.category = cat
    obj.semantic_tags = tags_for(cat)
    obj.is_dynamic = is_dynamic(cat)
    obj.is_fragile = is_fragile(cat)
    obj.is_blocking_path = is_blocking(cat)
    return obj


def compute_spatial_relations(objects: list["DetectedObject"]) -> None:  # type: ignore[name-defined]
    """Compute human-readable spatial relation strings between all object pairs.

    Mutates each object's `spatial_relations` list in-place.  Only pairs whose
    Euclidean distance is within RELATION_DISTANCE metres are annotated.

    Call this once after all objects for a batch have been enriched.
    """
    for obj in objects:
        obj.spatial_relations = []

    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            dx = a.pose.x - b.pose.x
            dy = a.pose.y - b.pose.y
            dz = a.pose.z - b.pose.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist > RELATION_DISTANCE:
                continue

            if abs(dz) >= _ABOVE_Z_THRESHOLD:
                if dz > 0:
                    a.spatial_relations.append(f"above {b.label}")
                    b.spatial_relations.append(f"below {a.label}")
                else:
                    a.spatial_relations.append(f"below {b.label}")
                    b.spatial_relations.append(f"above {a.label}")
            else:
                a.spatial_relations.append(f"next to {b.label}")
                b.spatial_relations.append(f"next to {a.label}")
