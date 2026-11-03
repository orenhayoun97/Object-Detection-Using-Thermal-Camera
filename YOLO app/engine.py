"""Model loading and the background detection thread."""
from __future__ import annotations

import bisect
import csv
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .draw import INFER_KEYS, Detections, Options, draw


@dataclass
class Source:
    kind: str            # "video" | "camera"
    path: str = ""
    camera_index: int = 0

    @property
    def live(self) -> bool:
        return self.kind == "camera"

    def describe(self) -> str:
        return f"Camera {self.camera_index}" if self.live else os.path.basename(self.path)


def pick_device(choice: str = "auto") -> str:
    if choice != "auto":
        return choice
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_model_path(path: str) -> str:
    """A bare model name like "yolo26n.pt" is looked up in the installed app's bundled models."""
    if os.path.exists(path):
        return path
    base = getattr(sys, "_MEIPASS", None)  # set when running as the packaged app
    if base:
        bundled = os.path.join(base, "models", os.path.basename(path))
        if os.path.exists(bundled):
            return bundled
    return path


def load_model(path: str, device: str):
    """Load a YOLO model and run one warm-up pass so errors surface immediately."""
    from ultralytics import YOLO

    model = YOLO(resolve_model_path(path))
    if model.task == "classify":
        raise ValueError("Classification models are not supported: choose a detection or segmentation model.")
    model.predict(np.zeros((320, 320, 3), np.uint8), device=device, verbose=False)
    return model


class ModelLoader(QThread):
    done = Signal(object, str)  # model, error message

    def __init__(self, path: str, device: str):
        super().__init__()
        self.path, self.device = path, device

    def run(self):
        try:
            self.done.emit(load_model(self.path, self.device), "")
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.done.emit(None, str(e))


MAX_WIDTH = 1920    # larger frames (4K) are scaled down once on input
BUFFER_SEC = 30.0   # how far back a live camera can be rewound
BUFFER_WIDTH = 1280


def to_qimage(bgr: np.ndarray) -> QImage:
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    h, w = bgra.shape[:2]
    return QImage(bgra.data, w, h, 4 * w, QImage.Format.Format_RGB32).copy()


def limit_size(frame: np.ndarray, width: int = MAX_WIDTH) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def fmt_time(sec: float) -> str:
    sec = max(0.0, sec)
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


class CameraGrabber:
    """Reads the camera on its own thread and keeps only the newest frame."""

    def __init__(self, cap):
        self.cap = cap
        self.cond = threading.Condition()
        self.frame = None
        self.seq = 0
        self.alive = True
        self._stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        failures = 0
        while not self._stop:
            ok, frame = self.cap.read()
            with self.cond:
                if ok:
                    self.frame, failures = frame, 0
                    self.seq += 1
                else:
                    failures += 1
                    self.alive = failures <= 50
                self.cond.notify_all()
            if not ok:
                if not self.alive:
                    return
                time.sleep(0.02)

    def read(self, last_seq: int, timeout: float = 0.5):
        """Waits for a frame newer than last_seq. Returns (frame or None, seq)."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq or not self.alive or self._stop, timeout)
            if self.seq == last_seq:
                return None, last_seq
            return self.frame, self.seq

    def stop(self):
        self._stop = True
        self.thread.join(1.0)


class DetectionWorker(QThread):
    frame_ready = Signal()               # a new frame is waiting in take_frame()
    position = Signal(int, int, float)   # frame index, total frames, source fps (video files)
    ended = Signal()
    error = Signal(str)
    warning = Signal(str)
    recording = Signal(bool, str)        # is recording, file path
    logged = Signal(list)                # new rows of the detection log

    LOG_HEADER = ["time", "video_time", "class", "track_id", "confidence"]

    def __init__(self, model, model_path: str, device: str, source: Source, opts: Options, log_path: str):
        super().__init__()
        self.model = model
        self.names = model.names
        self.model_path = model_path
        self.device = device
        self.source = source
        self.log_path = log_path
        self._opts = opts
        self._lock = threading.Lock()
        self._infer_v = self._draw_v = 0
        self._track_model = None  # separate instance: ultralytics tracker callbacks stick to a model
        self._track_failed = False
        self._stop = False
        self._paused = False
        self._seek: int | None = None
        self._rec_cmd: tuple[str, str] | None = None
        self._writer = None
        self._rec_path = ""
        self._out_lock = threading.Lock()
        self._out = None
        self._buf: deque = deque()       # (monotonic time, jpeg bytes) of the live camera
        self._buf_lock = threading.Lock()
        self._seen_ids: set[int] = set()
        self._present: set[int] = set()
        self._log_file = None
        self._log_csv = None

    # ---- controls (called from the UI thread) ----
    def set_options(self, **kw):
        with self._lock:
            for key, value in kw.items():
                setattr(self._opts, key, value)
                if key in INFER_KEYS:
                    self._infer_v += 1
                else:
                    self._draw_v += 1

    def _snapshot(self):
        with self._lock:
            o = self._opts
            return replace(o, classes=set(o.classes) if o.classes is not None else None), self._infer_v, self._draw_v

    def stop(self):
        self._stop = True
        self.wait(5000)

    def set_paused(self, paused: bool):
        self._paused = paused

    @property
    def paused(self) -> bool:
        return self._paused

    def seek(self, frame_index: int):
        self._seek = max(0, frame_index)

    def start_recording(self, path: str):
        self._rec_cmd = ("start", path)

    def stop_recording(self):
        self._rec_cmd = ("stop", "")

    def take_frame(self) -> QImage | None:
        with self._out_lock:
            out, self._out = self._out, None
        return out

    # ---- live camera rewind buffer ----
    def buffer_span(self):
        with self._buf_lock:
            return (self._buf[0][0], self._buf[-1][0]) if self._buf else None

    def buffered_frame(self, t: float) -> QImage | None:
        with self._buf_lock:
            if not self._buf:
                return None
            i = min(bisect.bisect_left([b[0] for b in self._buf], t), len(self._buf) - 1)
            data = self._buf[i][1]
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return to_qimage(img) if img is not None else None

    # ---- internals ----
    def _deliver(self, img: np.ndarray):
        with self._out_lock:
            had, self._out = self._out is not None, to_qimage(img)
        if not had:  # the UI only ever gets the newest frame, so it never falls behind
            self.frame_ready.emit()

    def _tracker_model(self):
        if self._track_model is None:
            from ultralytics import YOLO
            self._track_model = YOLO(resolve_model_path(self.model_path))
        return self._track_model

    def _reset_tracker(self):
        pred = getattr(self._track_model, "predictor", None)
        for t in getattr(pred, "trackers", []) or []:
            t.reset()
        self._seen_ids.clear()
        self._present.clear()

    def _infer(self, frame, o: Options, track: bool) -> Detections:
        classes = sorted(o.classes) if o.classes is not None else None
        if classes is not None and not classes:
            return Detections.empty()
        kw = dict(conf=o.conf, classes=classes, device=self.device, verbose=False)
        if track and not self._track_failed:
            try:
                return Detections.from_result(
                    self._tracker_model().track(frame, persist=True, tracker="bytetrack.yaml", **kw)[0])
            except Exception as e:  # noqa: BLE001 - keep detecting without tracking
                self._track_failed = True
                self.warning.emit(f"Tracking is unavailable, detecting only: {e}")
        return Detections.from_result(self.model.predict(frame, **kw)[0])

    def _log(self, det: Detections, media_t: float):
        """One row per object appearance: a new track ID, or (without tracking) a class that showed up."""
        rows = []
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        vt = fmt_time(media_t)
        if det.ids is not None:
            for c, tid, cf in zip(det.cls, det.ids, det.conf):
                if int(tid) not in self._seen_ids:
                    self._seen_ids.add(int(tid))
                    rows.append([stamp, vt, self.names.get(int(c), str(c)), int(tid), f"{cf:.2f}"])
        else:
            best: dict[int, float] = {}
            for c, cf in zip(det.cls, det.conf):
                best[int(c)] = max(best.get(int(c), 0.0), float(cf))
            for c, cf in best.items():
                if c not in self._present:
                    rows.append([stamp, vt, self.names.get(c, str(c)), "", f"{cf:.2f}"])
            self._present = set(best)
        if not rows:
            return
        try:
            if self._log_csv is None:
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                self._log_file = open(self.log_path, "w", newline="", encoding="utf-8")
                self._log_csv = csv.writer(self._log_file)
                self._log_csv.writerow(self.LOG_HEADER)
            self._log_csv.writerows(rows)
            self._log_file.flush()
        except OSError as e:
            self.warning.emit(f"Could not write the log file: {e}")
        self.logged.emit(rows)

    def _handle_recording(self, frame, fps: float):
        if self._rec_cmd is None:
            return
        cmd, path = self._rec_cmd
        self._rec_cmd = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self.recording.emit(False, self._rec_path)
        if cmd == "start":
            os.makedirs(os.path.dirname(path), exist_ok=True)
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), max(5.0, min(60.0, fps)), (w, h))
            if writer.isOpened():
                self._writer, self._rec_path = writer, path
                self.recording.emit(True, path)
            else:
                self.error.emit("Could not start recording.")

    def _output(self, img, new_frame: bool, rec_fps: float, repeat: int = 0):
        self._handle_recording(img, rec_fps)
        if new_frame and self._writer is not None:
            for _ in range(1 + repeat):  # skipped frames are repeated to keep the recording's duration right
                self._writer.write(img)
        if new_frame and self.source.live:
            ok, data = cv2.imencode(".jpg", limit_size(img, BUFFER_WIDTH), [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                now = time.monotonic()
                with self._buf_lock:
                    self._buf.append((now, data))
                    while now - self._buf[0][0] > BUFFER_SEC:
                        self._buf.popleft()
        self._deliver(img)

    def run(self):
        src = self.source
        cap = cv2.VideoCapture(src.camera_index if src.live else src.path)
        if not cap.isOpened():
            self.error.emit("Could not open the camera." if src.live else "Could not open the video file.")
            return
        grabber = CameraGrabber(cap) if src.live else None
        try:
            self._loop(cap, grabber)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Error: {e}")
        finally:
            if grabber is not None:
                grabber.stop()
            cap.release()
            if self._writer is not None:
                self._writer.release()
                self._writer = None
                self.recording.emit(False, self._rec_path)
            if self._log_file is not None:
                self._log_file.close()

    def _loop(self, cap, grabber):
        live = grabber is not None
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if src_fps > 240:
            src_fps = 30.0
        total = 0 if live else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        raw, det = None, None
        last_iv = last_dv = -1
        last_track = None
        fps_ema, last_t = 0.0, None
        next_due = time.perf_counter()
        cam_seq, skipped = 0, 0
        t_start = time.monotonic()

        while not self._stop:
            o, iv, dv = self._snapshot()
            if o.tracking != last_track:
                if last_track is not None:
                    self._reset_tracker()
                last_track = o.tracking

            if not self._paused or self._seek is not None:
                if self._seek is not None and not live:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, min(self._seek, max(0, total - 1)))
                    self._reset_tracker()
                    next_due = time.perf_counter()
                    skipped = 0
                self._seek = None
                if live:
                    frame, cam_seq = grabber.read(cam_seq)
                    if frame is None:
                        if not grabber.alive:
                            self.error.emit("The camera stopped sending frames.")
                            return
                        continue
                else:
                    ok, frame = cap.read()
                    if not ok:
                        self._paused = True
                        self.ended.emit()
                        continue
                raw = limit_size(frame)
                pos = 0 if live else int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                media_t = time.monotonic() - t_start if live else pos / src_fps
                det = self._infer(raw, o, o.tracking)
                last_iv, last_dv = iv, dv
                now = time.perf_counter()
                if last_t is not None:
                    inst = 1.0 / max(1e-6, now - last_t)
                    fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
                last_t = now
                self._log(det, media_t)
                self._output(draw(raw, det, self.names, o, fps_ema), True,
                             (fps_ema or 20.0) if live else src_fps, skipped)
                skipped = 0
                if not live:
                    self.position.emit(pos, total, src_fps)
                    next_due += 1.0 / src_fps
                    delay = next_due - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        # Detection is slower than the video: skip frames (grab() doesn't decode)
                        # to keep playing at true speed instead of in slow motion.
                        skip = min(int(-delay * src_fps), int(src_fps))
                        for _ in range(skip):
                            if not cap.grab():
                                break
                            skipped += 1
                        next_due += skip / src_fps
                        if next_due < time.perf_counter() - 1.0:
                            next_due = time.perf_counter()
            else:
                last_t = None  # don't count pauses in FPS
                if raw is None:
                    self.msleep(20)
                    continue
                if iv != last_iv:  # model options changed on a paused frame: detect again
                    det = self._infer(raw, o, track=False)
                elif dv == last_dv and self._rec_cmd is None:
                    self.msleep(15)
                    continue
                last_iv, last_dv = iv, dv
                self._output(draw(raw, det, self.names, o, fps_ema), False, src_fps)
