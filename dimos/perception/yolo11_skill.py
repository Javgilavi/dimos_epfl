# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import math
import time
from threading import RLock
from typing import Any

from reactivex.disposable import Disposable

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.vision_msgs.Detection2DArray import Detection2DArray
from dimos.types.timestamped import to_timestamp


COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


class Yolo11DetectionConfig(ModuleConfig):
    max_age_seconds: float = 2.0
    semantic_distance: float = 1.5
    camera_fx: float = 576.0


class Yolo11DetectionSkill(Module):
    """Expose workstation YOLO11 detections to the MCP agent."""

    config: Yolo11DetectionConfig
    detections: In[Detection2DArray]
    odom: In[PoseStamped]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest: list[dict[str, Any]] = []
        self._latest_ts: float = 0.0
        self._pose: dict[str, float] | None = None
        self._lock = RLock()

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.detections.subscribe(self._on_detections)))
        self.register_disposable(Disposable(self.odom.subscribe(self._on_pose)))

    def _on_pose(self, pose: PoseStamped) -> None:
        with self._lock:
            self._pose = {"x": float(pose.x), "y": float(pose.y), "yaw": float(pose.yaw)}

    def _estimate_world_pose(self, cx: float, image_width: float) -> dict[str, float] | None:
        pose = self._pose
        if pose is None:
            return None
        bearing = math.atan2(cx - image_width / 2.0, self.config.camera_fx)
        heading = pose["yaw"] + bearing
        return {
            "x": round(pose["x"] + self.config.semantic_distance * math.cos(heading), 3),
            "y": round(pose["y"] + self.config.semantic_distance * math.sin(heading), 3),
            "z": 0.0,
        }

    def _on_detections(self, msg: Detection2DArray) -> None:
        items = []
        image_width = 2.0 * self.config.camera_fx
        if msg.detections:
            max_cx = max(float(det.bbox.center.position.x) for det in msg.detections)
            image_width = max(image_width, max_cx * 2.0)
        for det in msg.detections[:50]:
            cx = float(det.bbox.center.position.x)
            cy = float(det.bbox.center.position.y)
            w = float(det.bbox.size_x)
            h = float(det.bbox.size_y)
            class_id = -1
            confidence = 0.0
            if det.results:
                hypothesis = det.results[0].hypothesis
                try:
                    class_id = int(hypothesis.class_id)
                except Exception:
                    class_id = -1
                confidence = float(hypothesis.score)
            name = COCO_NAMES.get(class_id, f"class_{class_id}")
            items.append(
                {
                    "label": name,
                    "class_id": class_id,
                    "confidence": round(confidence, 3),
                    "bbox": [
                        round(cx - w / 2.0, 1),
                        round(cy - h / 2.0, 1),
                        round(cx + w / 2.0, 1),
                        round(cy + h / 2.0, 1),
                    ],
                    "center": [round(cx, 1), round(cy, 1)],
                    "estimated_map_pose": self._estimate_world_pose(cx, image_width),
                }
            )
        ts = to_timestamp(msg.header.stamp) if msg.header else time.time()
        with self._lock:
            self._latest = items
            self._latest_ts = ts

    @skill
    def get_latest_yolo_detections(self) -> str:
        """Return the latest YOLO11 segmentation detections from the external camera.

        Use this when the user asks what YOLO currently sees, whether a person or
        object is visible, or before starting a visual task such as following a
        person. The returned bbox is [x1, y1, x2, y2] in camera pixels.
        """
        with self._lock:
            detections = list(self._latest)
            ts = self._latest_ts
        if not detections:
            return (
                "No YOLO11 detections have arrived. Make sure "
                "`python scripts/workstation_yolo.py --feed-dimos` is running."
            )
        age = time.time() - ts
        stale = age > self.config.max_age_seconds
        payload = {
            "age_seconds": round(age, 2),
            "stale": stale,
            "detections": detections,
        }
        return json.dumps(payload)

    @skill
    def get_best_yolo_detection(self, label: str = "person") -> str:
        """Return the highest-confidence YOLO detection matching a label.

        The bbox can be passed to `follow_person(initial_bbox=...)` when label is
        "person", or used as evidence that an object is visible.
        """
        wanted = label.strip().lower()
        with self._lock:
            detections = list(self._latest)
            ts = self._latest_ts
        matches = [d for d in detections if d["label"].lower() == wanted]
        if not matches:
            return f"No current YOLO11 detection with label {label!r}."
        best = max(matches, key=lambda d: d["confidence"])
        best = dict(best)
        best["age_seconds"] = round(time.time() - ts, 2)
        return json.dumps(best)


__all__ = ["Yolo11DetectionSkill"]
