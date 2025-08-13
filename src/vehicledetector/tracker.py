from __future__ import annotations
from typing import List, Tuple
import numpy as np

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception:
    DeepSort = None  # type: ignore


class MultiObjectTracker:
    def __init__(self, max_age=30, n_init=3, max_iou_distance=0.7, max_cosine_distance=0.2, nn_budget=100):
        if DeepSort is None:
            raise RuntimeError("deep-sort-realtime is not available. Please install requirements.")
        self.tracker = DeepSort(max_age=max_age,
                                n_init=n_init,
                                max_iou_distance=max_iou_distance,
                                max_cosine_distance=max_cosine_distance,
                                nn_budget=nn_budget,
                                embedder='mobilenet',
                                half=True)

    def update(self, detections: List[Tuple[np.ndarray, float, int]], frame: np.ndarray):
        # Convert to format expected by deep-sort-realtime: [(xyxy, conf, class), ...]
        in_dets = []
        for (xyxy, conf, cls) in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            # DeepSort default expects TLWH (x, y, w, h)
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            in_dets.append(([x1, y1, w, h], conf, str(cls)))
        tracks = self.tracker.update_tracks(in_dets, frame=frame)
        results = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            ltrb = t.to_ltrb()  # left, top, right, bottom
            bbox = [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])]
            results.append((tid, bbox))
        return results
