"""Workstation-side YOLO11 segmentation on the Jetson USB camera stream.

Pulls JPEG frames from the Jetson HTTP stream, runs YOLO segmentation, displays
annotated results, and can publish the raw image, segmentation detections, and
Foxglove annotations to DimOS LCM so DimOS modules and visualizers see the same
camera/perception stream.

Usage:
    python scripts/workstation_yolo.py
    python scripts/workstation_yolo.py --stream-url http://192.168.123.18:8888/frame
    python scripts/workstation_yolo.py --model yolo11s-seg.pt --feed-dimos --headless

Requires: ultralytics, opencv-python
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np

COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}


def fetch_frame(url: str) -> np.ndarray | None:
    """Fetch a single JPEG frame from the Jetson HTTP stream."""
    try:
        with urlopen(url, timeout=5) as resp:
            data = resp.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except (URLError, OSError, ValueError) as exc:
        print(f"Failed to fetch frame: {exc}")
        return None


def draw_detections(frame: np.ndarray, results, show_masks: bool = True) -> np.ndarray:
    """Draw bounding boxes and optional segmentation masks on the frame."""
    annotated = frame.copy()

    boxes = results[0].boxes
    if boxes is None:
        return annotated

    if show_masks and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        for i, mask in enumerate(masks):
            color = np.random.RandomState(int(boxes.cls[i])).randint(0, 255, 3).tolist()
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            overlay = annotated.copy()
            overlay[mask_resized > 0.5] = color
            annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = COCO_NAMES.get(cls_id, f"cls{cls_id}")

        color = np.random.RandomState(cls_id).randint(0, 255, 3).tolist()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return annotated


def collect_detections(results) -> list[dict]:
    """Extract compact YOLO detections from one Ultralytics result batch."""
    detections = []
    boxes = results[0].boxes
    if boxes is None:
        return detections
    for box in boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        detections.append(
            {
                "label": COCO_NAMES.get(cls_id, f"cls{cls_id}"),
                "class_id": cls_id,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            }
        )
    return detections


class DimosYoloPublisher:
    """Publish YOLO image + segmentation outputs to DimOS LCM."""

    def __init__(
        self,
        image_topic: str,
        detections_topic: str,
        annotations_topic: str,
        segmented_image_topic: str,
        publish_raw_image: bool = True,
    ) -> None:
        import lcm as lcmlib
        from dimos.msgs.sensor_msgs.Image import Image
        from dimos.perception.detection.type.detection2d.imageDetections2D import (
            ImageDetections2D,
        )

        self.lc = lcmlib.LCM()
        self.image_cls = Image
        self.image_detections_cls = ImageDetections2D
        self.image_topic = image_topic
        self.detections_topic = detections_topic
        self.annotations_topic = annotations_topic
        self.segmented_image_topic = segmented_image_topic
        self.publish_raw_image = publish_raw_image

    def publish(self, frame: np.ndarray, results, annotated: np.ndarray | None = None) -> None:
        ts = time.time()
        image = self.image_cls.from_opencv(
            frame,
            frame_id="camera_optical",
            ts=ts,
        )
        if self.publish_raw_image:
            self.lc.publish(self.image_topic, image.lcm_encode())

        detections = self.image_detections_cls.from_ultralytics_result(image, results)
        self.lc.publish(
            self.detections_topic,
            detections.to_ros_detection2d_array().lcm_encode(),
        )
        self.lc.publish(
            self.annotations_topic,
            detections.to_foxglove_annotations().lcm_encode(),
        )

        if annotated is not None:
            segmented = self.image_cls.from_opencv(
                annotated,
                frame_id="camera_optical",
                ts=ts,
            )
            self.lc.publish(self.segmented_image_topic, segmented.lcm_encode())


class CloudSemanticPublisher:
    """Push YOLO overlays + high-confidence semantic objects to robohack2026."""

    def __init__(
        self,
        cloud_url: str,
        robot_id: str,
        threshold: float,
        semantic_distance: float,
        camera_fx: float,
        pose_topic: str,
        semantic_hz: float,
        frame_hz: float,
    ) -> None:
        import lcm as lcmlib
        from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

        self.cloud_url = cloud_url.rstrip("/")
        self.robot_id = robot_id
        self.threshold = threshold
        self.semantic_distance = semantic_distance
        self.camera_fx = camera_fx
        self.semantic_period = 1.0 / max(0.1, semantic_hz)
        self.frame_period = 1.0 / max(0.1, frame_hz)
        self.pose_cls = PoseStamped
        self.pose: dict | None = None
        self.last_semantic_push = 0.0
        self.last_frame_push = 0.0
        self.lc = lcmlib.LCM()
        self.lc.subscribe(pose_topic, self._on_pose)
        self.thread = threading.Thread(target=self._pose_loop, daemon=True)
        self.thread.start()

    def _pose_loop(self) -> None:
        while True:
            self.lc.handle_timeout(200)

    def _on_pose(self, _channel: str, data: bytes) -> None:
        try:
            msg = self.pose_cls.lcm_decode(data)
            self.pose = {"x": float(msg.x), "y": float(msg.y), "yaw": float(msg.yaw)}
        except Exception:
            return

    def _object_pose(self, det: dict, width: int) -> dict[str, float]:
        pose = self.pose
        if pose is None:
            return {"x": 0.0, "y": 0.0, "z": 0.0}
        cx = det["center"][0]
        bearing = math.atan2(cx - width / 2.0, self.camera_fx)
        heading = pose["yaw"] + bearing
        return {
            "x": pose["x"] + self.semantic_distance * math.cos(heading),
            "y": pose["y"] + self.semantic_distance * math.sin(heading),
            "z": 0.0,
        }

    def push_frame(self, frame: np.ndarray) -> None:
        now = time.time()
        if now - self.last_frame_push < self.frame_period:
            return
        self.last_frame_push = now
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        req = Request(
            f"{self.cloud_url}/frames",
            data=buf.tobytes(),
            headers={"Content-Type": "image/jpeg", "X-Robot-Id": self.robot_id},
            method="POST",
        )
        try:
            with urlopen(req, timeout=1.5):
                pass
        except Exception:
            pass

    def push_semantics(self, detections: list[dict], frame_shape: tuple[int, ...]) -> None:
        now = time.time()
        if now - self.last_semantic_push < self.semantic_period:
            return
        self.last_semantic_push = now
        width = int(frame_shape[1])
        objects = []
        for det in detections:
            if det["confidence"] < self.threshold:
                continue
            objects.append(
                {
                    "label": det["label"],
                    "confidence": det["confidence"],
                    "pose": self._object_pose(det, width),
                    "seen_count": 1,
                    "last_seen": now,
                    "source": "yolo11_segmentation",
                }
            )
        if not objects:
            return
        payload = json.dumps({"robot_id": self.robot_id, "objects": objects}).encode()
        req = Request(
            f"{self.cloud_url}/ingest",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=2.0):
                pass
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Workstation YOLO on Jetson camera stream")
    parser.add_argument(
        "--stream-url",
        default="http://192.168.123.18:8888/frame",
        help="URL to fetch JPEG frames from",
    )
    parser.add_argument(
        "--model",
        default="yolo11s-seg.pt",
        help="YOLO model to use, e.g. yolo11s.pt or yolo11s-seg.pt",
    )
    parser.add_argument("--imgsz", type=int, default=480, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--headless", action="store_true", help="No display window")
    parser.add_argument(
        "--publish-lcm",
        action="store_true",
        help="Publish raw frames to DimOS /color_image LCM for visualizers",
    )
    parser.add_argument(
        "--feed-dimos",
        action="store_true",
        help=(
            "Publish raw frames, Detection2DArray, ImageAnnotations, and "
            "segmented image to DimOS LCM"
        ),
    )
    parser.add_argument(
        "--lcm-topic",
        default="/color_image#sensor_msgs.Image",
        help="LCM topic for --publish-lcm",
    )
    parser.add_argument(
        "--detections-topic",
        default="/yolo11/detections#vision_msgs.Detection2DArray",
        help="LCM topic for YOLO Detection2DArray",
    )
    parser.add_argument(
        "--annotations-topic",
        default="/yolo11/annotations#foxglove_msgs.ImageAnnotations",
        help="LCM topic for YOLO Foxglove image annotations",
    )
    parser.add_argument(
        "--segmented-image-topic",
        default="/yolo11/segmented_image#sensor_msgs.Image",
        help="LCM topic for the YOLO mask-overlay image",
    )
    parser.add_argument(
        "--cloud-url",
        default=os.environ.get("ROBOHACK_CLOUD_URL", "http://localhost:8080"),
        help="robohack2026 FastAPI URL for UI frames and semantic map ingestion",
    )
    parser.add_argument("--robot-id", default=os.environ.get("ROBOT_ID", "go2_a"))
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.70,
        help="Only detections at or above this confidence are added to semantic memory",
    )
    parser.add_argument(
        "--semantic-distance",
        type=float,
        default=1.5,
        help="Estimated object distance in metres when projecting 2D YOLO boxes into the map",
    )
    parser.add_argument(
        "--camera-fx",
        type=float,
        default=float(os.environ.get("GO2_EXTERNAL_CAMERA_FX", "576")),
        help="Camera focal length in pixels for bearing estimate",
    )
    parser.add_argument(
        "--pose-topic",
        default="/odom#geometry_msgs.PoseStamped",
        help="LCM pose topic used to project YOLO detections into world coordinates",
    )
    parser.add_argument("--semantic-hz", type=float, default=1.0)
    parser.add_argument("--ui-frame-hz", type=float, default=12.0)
    parser.add_argument(
        "--no-cloud",
        action="store_true",
        help="Disable robohack2026 UI frame + semantic map pushes",
    )
    args = parser.parse_args()

    from ultralytics import YOLO

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    publisher = (
        DimosYoloPublisher(
            image_topic=args.lcm_topic,
            detections_topic=args.detections_topic,
            annotations_topic=args.annotations_topic,
            segmented_image_topic=args.segmented_image_topic,
            publish_raw_image=args.publish_lcm or args.feed_dimos,
        )
        if args.publish_lcm or args.feed_dimos
        else None
    )
    cloud = (
        CloudSemanticPublisher(
            cloud_url=args.cloud_url,
            robot_id=args.robot_id,
            threshold=args.semantic_threshold,
            semantic_distance=args.semantic_distance,
            camera_fx=args.camera_fx,
            pose_topic=args.pose_topic,
            semantic_hz=args.semantic_hz,
            frame_hz=args.ui_frame_hz,
        )
        if not args.no_cloud and args.cloud_url
        else None
    )

    print(f"Fetching frames from: {args.stream_url}")
    if cloud:
        print(
            f"Pushing YOLO overlay + semantic detections >= {args.semantic_threshold:.2f} "
            f"to {args.cloud_url}"
        )
    print("Press 'q' to quit")

    fps_history: list[float] = []
    while True:
        frame = fetch_frame(args.stream_url)
        if frame is None:
            time.sleep(0.5)
            continue

        start = time.time()
        results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        elapsed = time.time() - start
        fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        yolo_detections = collect_detections(results)
        detections = [
            f"{det['label']}({det['confidence']:.2f})"
            for det in yolo_detections
        ]

        annotated = draw_detections(frame, results)
        if publisher:
            publisher.publish(frame, results, annotated=annotated)
        if cloud:
            cloud.push_frame(annotated)
            cloud.push_semantics(yolo_detections, frame.shape)

        if args.headless:
            print(f"[{avg_fps:.1f} FPS] {', '.join(detections)}")
            time.sleep(0.03)
            continue

        cv2.putText(
            annotated,
            f"FPS: {avg_fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Workstation YOLO", annotated)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
