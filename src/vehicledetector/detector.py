from __future__ import annotations
from typing import List, Optional
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None  # type: ignore


class YoloDetector:
    def __init__(self, model_path: str, classes: Optional[List[int]] = None, conf: float = 0.25, iou: float = 0.5,
                 device: str | int | None = None, imgsz: int | None = None):
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO is not available. Please install requirements.")
        self.model = YOLO(model_path)
        self.classes = classes
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz

    def infer(self, frame: np.ndarray):
        kw = {"classes": self.classes, "conf": self.conf, "iou": self.iou, "verbose": False}
        if self.device not in (None, "auto"):
            kw["device"] = self.device
        if self.imgsz:
            kw["imgsz"] = self.imgsz
        res = self.model.predict(source=frame, **kw)
        if not res:
            return []
        r = res[0]
        detections = []
        if r.boxes is None:
            return detections
        for b in r.boxes:
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            conf = float(b.conf[0].cpu().numpy()) if b.conf is not None else 0.0
            cls = int(b.cls[0].cpu().numpy()) if b.cls is not None else -1
            detections.append((xyxy, conf, cls))
        return detections
