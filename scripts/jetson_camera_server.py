"""Serve a USB camera as single-frame JPEG over HTTP for real Go2 runs.

Run this on the Jetson that has the added USB camera attached:

    python scripts/jetson_camera_server.py --host 0.0.0.0 --port 8888 --device 0

DimOS then consumes:

    http://<jetson-ip>:8888/frame

The endpoint returns one fresh JPEG per request, matching
GO2_EXTERNAL_CAMERA_URL's default format.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

import cv2


class SharedCamera:
    def __init__(self, device: int, width: int, height: int, fps: int) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.running = True

    def start(self) -> None:
        thread = threading.Thread(target=self._loop, daemon=True, name="usb-camera-capture")
        thread.start()

    def _loop(self) -> None:
        cap = cv2.VideoCapture(self.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {self.device}")

        interval = 1.0 / max(self.fps, 1)
        while self.running:
            started = time.time()
            ok, frame = cap.read()
            if ok:
                encoded, buf = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 80],
                )
                if encoded:
                    with self.lock:
                        self.jpeg = buf.tobytes()
            elapsed = time.time() - started
            if elapsed < interval:
                time.sleep(interval - elapsed)

        cap.release()

    def latest(self) -> bytes | None:
        with self.lock:
            return self.jpeg


def make_handler(camera: SharedCamera) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path not in ("/", "/frame", "/image.jpg", "/snapshot"):
                self.send_error(404)
                return

            if self.path == "/":
                body = (
                    b"<html><body><h1>Jetson Camera</h1>"
                    b"<p>Use <a href='/frame'>/frame</a> for JPEG frames.</p>"
                    b"<img src='/frame'></body></html>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            jpeg = camera.latest()
            if not jpeg:
                self.send_error(503, "No frame yet")
                return

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpeg)

        def log_message(self, fmt: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve USB camera JPEG frames")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    camera = SharedCamera(args.device, args.width, args.height, args.fps)
    camera.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(camera))
    print(f"Serving camera on http://{args.host}:{args.port}/frame", flush=True)
    try:
        server.serve_forever()
    finally:
        camera.running = False
        server.server_close()


if __name__ == "__main__":
    main()
