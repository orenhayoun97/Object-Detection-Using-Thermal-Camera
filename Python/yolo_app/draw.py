"""Detections and drawing them on a frame."""
from __future__ import annotations

import colorsys
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Options:
    """Everything the user can change while the video plays."""
    conf: float = 0.35
    classes: set[int] | None = None  # None = all classes
    tracking: bool = False
    show_boxes: bool = True
    show_labels: bool = True
    show_conf: bool = True
    show_fps: bool = True
    show_count: bool = True


# Options that require re-running the model when changed (the rest only affect drawing).
INFER_KEYS = {"conf", "classes", "tracking"}
VIEW_OPTIONS = [("show_boxes", "Boxes"), ("show_labels", "Class name"), ("show_conf", "Confidence"),
                ("show_fps", "FPS"), ("show_count", "Number of objects")]


@dataclass
class Detections:
    xyxy: np.ndarray            # (N, 4) float
    conf: np.ndarray            # (N,)
    cls: np.ndarray             # (N,) int
    ids: np.ndarray | None      # (N,) int, only when tracking

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int), None)

    @classmethod
    def from_result(cls, r) -> "Detections":
        boxes = r.boxes
        if boxes is None and getattr(r, "obb", None) is not None:
            boxes = r.obb  # oriented boxes: use their axis-aligned bounds
        if boxes is None or len(boxes) == 0:
            return cls.empty()
        ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else None
        return cls(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy().astype(int), ids)

    def __len__(self):
        return len(self.cls)


def class_color_bgr(idx: int) -> tuple[int, int, int]:
    """Stable, well-separated color per class index (golden-ratio hue walk)."""
    h = (idx * 0.618033988749895 + 0.08) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.72, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _panel(img, x1, y1, x2, y2, alpha=0.55):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
        img[y1:y2, x1:x2] = (img[y1:y2, x1:x2] * (1 - alpha)).astype(np.uint8)


def _text_color(bgr):
    b, g, r = bgr
    return (0, 0, 0) if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else (255, 255, 255)


def draw(frame: np.ndarray, det: Detections, names: dict, o: Options, fps: float) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    scale = max(h, w) / 1280
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.42, 0.62 * scale)
    font_th = max(1, round(1.6 * scale))
    box_th = max(2, round(2 * scale))

    for i in range(len(det)):
        c = int(det.cls[i])
        color = class_color_bgr(c)
        x1, y1, x2, y2 = (int(v) for v in det.xyxy[i])
        if o.show_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, box_th, cv2.LINE_AA)
        parts = []
        if o.show_labels:
            parts.append(names.get(c, str(c)))
        if o.show_conf:
            parts.append(f"{det.conf[i]:.2f}")
        if parts and det.ids is not None:
            parts.append(f"#{det.ids[i]}")
        if parts:
            text = " ".join(parts)
            (tw, th), bl = cv2.getTextSize(text, font, font_scale, font_th)
            ty = y1 - 4 if y1 - th - bl - 6 > 0 else y1 + th + bl + 4
            cv2.rectangle(img, (x1, ty - th - bl - 2), (x1 + tw + 8, ty + 2), color, -1)
            cv2.putText(img, text, (x1 + 4, ty - bl + 1), font, font_scale, _text_color(color), font_th, cv2.LINE_AA)

    hud = []
    if o.show_fps:
        hud.append(f"FPS {fps:.1f}")
    if o.show_count:
        hud.append(f"{len(det)} objects")
    if hud:
        text = "   ".join(hud)
        pad = int(10 * max(1.0, scale))
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_th)
        _panel(img, pad, pad, 3 * pad + tw, 2 * pad + th + pad // 2)
        cv2.putText(img, text, (2 * pad, pad + th + pad // 2), font, font_scale, (120, 255, 140), font_th, cv2.LINE_AA)
    return img
