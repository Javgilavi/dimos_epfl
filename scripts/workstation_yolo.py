"""Workstation-side YOLO inference on the Jetson USB camera stream.

Pulls JPEG frames from the Jetson HTTP stream, runs YOLO segmentation, displays
annotated results, and can optionally publish the raw camera image to DimOS LCM
so the DimOS visualizer sees the same camera.

Usage:
    python scripts/workstation_yolo.py
    python scripts/workstation_yolo.py --stream-url http://192.168.123.18:8888/frame
    python scripts/workstation_yolo.py --model yolo11s-seg.pt --publish-lcm

Requires: ultralytics, opencv-python
"""

from __future__ import annotations

import argparse
import time
from urllib.error import URLError
from urllib.request import urlopen

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

    if show_masks and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        for i, mask in enumerate(masks):
            color = np.random.RandomState(int(results[0].boxes.cls[i])).randint(
                0, 255, 3
            ).tolist()
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            overlay = annotated.copy()
            overlay[mask_resized > 0.5] = color
            annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)

    for box in results[0].boxes:
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


class DimosImagePublisher:
    """Publish frames to DimOS's standard color image LCM channel."""

    def __init__(self, topic: str) -> None:
        import lcm as lcmlib
        from dimos.msgs.sensor_msgs.Image import Image

        self.lc = lcmlib.LCM()
        self.image_cls = Image
        self.topic = topic

    def publish(self, frame: np.ndarray) -> None:
        image = self.image_cls.from_opencv(
            frame,
            frame_id="camera_optical",
            ts=time.time(),
        )
        self.lc.publish(self.topic, image.lcm_encode())


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
        "--lcm-topic",
        default="/color_image#sensor_msgs.Image",
        help="LCM topic for --publish-lcm",
    )
    args = parser.parse_args()

    from ultralytics import YOLO

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    publisher = DimosImagePublisher(args.lcm_topic) if args.publish_lcm else None

    print(f"Fetching frames from: {args.stream_url}")
    print("Press 'q' to quit")

    fps_history: list[float] = []
    while True:
        frame = fetch_frame(args.stream_url)
        if frame is None:
            time.sleep(0.5)
            continue

        if publisher:
            publisher.publish(frame)

        start = time.time()
        results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        elapsed = time.time() - start
        fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = COCO_NAMES.get(cls_id, f"cls{cls_id}")
            detections.append(f"{name}({conf:.2f})")

        if args.headless:
            print(f"[{avg_fps:.1f} FPS] {', '.join(detections)}")
            time.sleep(0.03)
            continue

        annotated = draw_detections(frame, results)
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
