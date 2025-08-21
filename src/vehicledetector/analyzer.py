from __future__ import annotations
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from .detector import YoloDetector
from .tracker import MultiObjectTracker
from .color_utils import check_white
from .reid_filter import score_against_gallery
# Segment export removed per user request; keep module focused on stills only


class VideoAnalyzer:
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        # Load configuration
        if config is not None:
            self.cfg = config
        else:
            if config_path is None:
                # Default to config.yaml alongside this module if present
                default_cfg = Path(__file__).with_name("config.yaml")
                config_path = str(default_cfg) if default_cfg.exists() else None
            self.cfg = {}
            if config_path and Path(config_path).exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    self.cfg = yaml.safe_load(f) or {}

        # Detector
        self.detector = YoloDetector(
            self.cfg.get("model", "yolov8n.pt"),
            self.cfg.get("classes"),
            self.cfg.get("confidence", 0.25),
            self.cfg.get("iou", 0.5),
            self.cfg.get("device", "auto"),
            self.cfg.get("imgsz", 640),
        )

        # Tracker
        # Accept both 'tracker' and legacy 'tracking' key from YAML
        tracker_cfg = self.cfg.get("tracker", {}) or self.cfg.get("tracking", {})
        self.tracker = MultiObjectTracker(
            max_age=int(tracker_cfg.get("max_age", 30)),
            n_init=int(tracker_cfg.get("n_init", 3)),
            max_iou_distance=float(tracker_cfg.get("max_iou_distance", 0.7)),
            max_cosine_distance=float(tracker_cfg.get("max_cosine_distance", 0.2)),
            nn_budget=int(tracker_cfg.get("nn_budget", 100)),
        )

        # Sampling and runtime
        self.stride_frames = int(self.cfg.get("stride_frames", 1))
        self.max_fps = self.cfg.get("max_fps")
        self.fps_sanity_max = float(self.cfg.get("fps_sanity_max", 240.0))
        self.batch_size = max(1, int(self.cfg.get("batch_size", 1)))

        # Behavior
        self.color_cfg = self.cfg.get("color_filter", {})
        self.seg_cfg = self.cfg.get("segment_export", {})
        self.sedan_only = bool(self.cfg.get("sedan_only", False))

    def maybe_cap_fps(self, fps: float) -> int:
        """Compute an effective stride to respect max_fps and cap absurd FPS metadata."""
        stride = max(1, int(self.stride_frames or 1))
        if self.max_fps:
            try:
                maxfps = float(self.max_fps)
                if fps > maxfps and maxfps > 0:
                    stride = max(stride, int(max(1, round(fps / maxfps))))
            except Exception:
                pass
        # Cap absurd FPS values (e.g., some AVIs report 1000fps)
        try:
            if fps > self.fps_sanity_max and self.fps_sanity_max > 0:
                stride = max(stride, int(max(1, round(fps / self.fps_sanity_max))))
        except Exception:
            pass
        return max(1, stride)

    def sedan_shape_heuristic(self, bbox: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        ar = w / float(h)
        area = w * h
        min_area = int(self.color_cfg.get("min_box_area", 1000))
        return (area >= min_area) and (1.2 <= ar <= 5.0)

    def analyze_video(self, video_path: str, output_dir: str) -> Dict:
        start_time = time.time()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        vid_stride = self.maybe_cap_fps(fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        track_last_seen: Dict[int, int] = {}
        track_first_seen: Dict[int, int] = {}
        best_still: Dict[int, Tuple[float, np.ndarray, Tuple[int, int, int, int], int]] = {}
        frame_idx = -1
        processed_idx = -1
        results: List[Dict[str, Any]] = []
        debug_rejects_saved = 0
        # Diagnostics
        frames_read = 0
        frames_processed = 0
        raw_detections_total = 0
        color_pass_total = 0
        color_reject_total = 0
        tracks_confirmed_total = 0

        batch_frames: List[np.ndarray] = []
        batch_indices: List[int] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_read += 1
            frame_idx += 1
            if frame_idx % vid_stride != 0:
                continue
            processed_idx += 1
            frames_processed += 1
            batch_frames.append(frame)
            batch_indices.append(frame_idx)
            if len(batch_frames) < self.batch_size:
                continue

            # Run batch prediction
            if len(batch_frames) > 1:
                dets_list = self.detector.model.predict(
                    source=batch_frames,
                    classes=self.detector.classes,
                    conf=self.detector.conf,
                    iou=self.detector.iou,
                    verbose=False,
                    device=self.detector.device,
                    imgsz=self.detector.imgsz,
                )
            else:
                single_res = self.detector.model.predict(
                    source=batch_frames[0],
                    classes=self.detector.classes,
                    conf=self.detector.conf,
                    iou=self.detector.iou,
                    verbose=False,
                    device=self.detector.device,
                    imgsz=self.detector.imgsz,
                )
                dets_list = [single_res[0]] if single_res else []

            for det_idx, det in enumerate(dets_list):
                cur_frame = batch_frames[det_idx]
                cur_idx = batch_indices[det_idx]
                detections = []
                if getattr(det, "boxes", None) is not None:
                    for b in det.boxes:
                        xyxy = b.xyxy[0].cpu().numpy().astype(int)
                        conf = float(b.conf[0].cpu().numpy()) if b.conf is not None else 0.0
                        cls = int(b.cls[0].cpu().numpy()) if b.cls is not None else -1
                        detections.append((xyxy, conf, cls))
                raw_detections_total += len(detections)
                tracks = self.tracker.update(detections, cur_frame)
                for tid, bbox in tracks:
                    x1, y1, x2, y2 = bbox
                    if self.sedan_only and not self.sedan_shape_heuristic((x1, y1, x2, y2)):
                        continue
                    if self.color_cfg.get("enabled", True):
                        ok, ratio, crop, mask = check_white(cur_frame, (x1, y1, x2, y2), self.color_cfg)
                        if not ok:
                            color_reject_total += 1
                            dbg_dir = self.color_cfg.get("debug_dir")
                            if dbg_dir:
                                try:
                                    Path(dbg_dir).mkdir(parents=True, exist_ok=True)
                                    max_dbg = int(self.color_cfg.get("debug_max", 30))
                                    if debug_rejects_saved < max_dbg and crop is not None:
                                        fn_base = f"rej_f{cur_idx}_tid{tid}_r{ratio:.2f}"
                                        cv2.imwrite(str(Path(dbg_dir) / f"{fn_base}_crop.jpg"), crop)
                                        if mask is not None:
                                            cv2.imwrite(str(Path(dbg_dir) / f"{fn_base}_mask.png"), mask)
                                        debug_rejects_saved += 1
                                except Exception:
                                    pass
                            continue
                        else:
                            color_pass_total += 1
                    if tid not in track_first_seen:
                        track_first_seen[tid] = cur_idx
                    track_last_seen[tid] = cur_idx
                    tracks_confirmed_total += 1
                    results.append({
                        "frame": cur_idx,
                        "tid": tid,
                        "bbox": bbox,
                        "time_sec": cur_idx / fps,
                    })
                    stills_cfg = self.cfg.get("stills", {"enabled": False})
                    if stills_cfg.get("enabled", False):
                        H, W = cur_frame.shape[:2]
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        dx = abs(cx - W / 2) / (W / 2)
                        dy = abs(cy - H / 2) / (H / 2)
                        center_penalty = dx + dy
                        margin = int(stills_cfg.get("edge_margin", 8))
                        complete = (x1 >= margin and y1 >= margin and x2 <= W - margin and y2 <= H - margin)
                        completeness_bonus = 0.2 if complete else 0.0
                        area = (x2 - x1) * (y2 - y1)
                        norm_area = area / float(W * H)
                        score = completeness_bonus + (0.5 * norm_area) + (0.5 * (1.0 - min(1.0, center_penalty)))
                        prev = best_still.get(tid)
                        if prev is None or score > prev[0]:
                            best_still[tid] = (score, cur_frame.copy(), (x1, y1, x2, y2), cur_idx)
            batch_frames = []
            batch_indices = []

        # Process any remaining frames in the last batch
        if batch_frames:
            if len(batch_frames) > 1:
                dets_list = self.detector.model.predict(
                    source=batch_frames,
                    classes=self.detector.classes,
                    conf=self.detector.conf,
                    iou=self.detector.iou,
                    verbose=False,
                    device=self.detector.device,
                    imgsz=self.detector.imgsz,
                )
            else:
                single_res = self.detector.model.predict(
                    source=batch_frames[0],
                    classes=self.detector.classes,
                    conf=self.detector.conf,
                    iou=self.detector.iou,
                    verbose=False,
                    device=self.detector.device,
                    imgsz=self.detector.imgsz,
                )
                dets_list = [single_res[0]] if single_res else []
            for det_idx, det in enumerate(dets_list):
                cur_frame = batch_frames[det_idx]
                cur_idx = batch_indices[det_idx]
                detections = []
                if getattr(det, "boxes", None) is not None:
                    for b in det.boxes:
                        xyxy = b.xyxy[0].cpu().numpy().astype(int)
                        conf = float(b.conf[0].cpu().numpy()) if b.conf is not None else 0.0
                        cls = int(b.cls[0].cpu().numpy()) if b.cls is not None else -1
                        detections.append((xyxy, conf, cls))
                raw_detections_total += len(detections)
                tracks = self.tracker.update(detections, cur_frame)
                for tid, bbox in tracks:
                    x1, y1, x2, y2 = bbox
                    if self.sedan_only and not self.sedan_shape_heuristic((x1, y1, x2, y2)):
                        continue
                    if self.color_cfg.get("enabled", True):
                        ok, ratio, crop, mask = check_white(cur_frame, (x1, y1, x2, y2), self.color_cfg)
                        if not ok:
                            color_reject_total += 1
                            dbg_dir = self.color_cfg.get("debug_dir")
                            if dbg_dir:
                                try:
                                    Path(dbg_dir).mkdir(parents=True, exist_ok=True)
                                    max_dbg = int(self.color_cfg.get("debug_max", 30))
                                    if debug_rejects_saved < max_dbg and crop is not None:
                                        fn_base = f"rej_f{cur_idx}_tid{tid}_r{ratio:.2f}"
                                        cv2.imwrite(str(Path(dbg_dir) / f"{fn_base}_crop.jpg"), crop)
                                        if mask is not None:
                                            cv2.imwrite(str(Path(dbg_dir) / f"{fn_base}_mask.png"), mask)
                                        debug_rejects_saved += 1
                                except Exception:
                                    pass
                            continue
                        else:
                            color_pass_total += 1
                    if tid not in track_first_seen:
                        track_first_seen[tid] = cur_idx
                    track_last_seen[tid] = cur_idx
                    tracks_confirmed_total += 1
                    results.append({
                        "frame": cur_idx,
                        "tid": tid,
                        "bbox": bbox,
                        "time_sec": cur_idx / fps,
                    })
                    stills_cfg = self.cfg.get("stills", {"enabled": False})
                    if stills_cfg.get("enabled", False):
                        H, W = cur_frame.shape[:2]
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        dx = abs(cx - W / 2) / (W / 2)
                        dy = abs(cy - H / 2) / (H / 2)
                        center_penalty = dx + dy
                        margin = int(stills_cfg.get("edge_margin", 8))
                        complete = (x1 >= margin and y1 >= margin and x2 <= W - margin and y2 <= H - margin)
                        completeness_bonus = 0.2 if complete else 0.0
                        area = (x2 - x1) * (y2 - y1)
                        norm_area = area / float(W * H)
                        score = completeness_bonus + (0.5 * norm_area) + (0.5 * (1.0 - min(1.0, center_penalty)))
                        prev = best_still.get(tid)
                        if prev is None or score > prev[0]:
                            best_still[tid] = (score, cur_frame.copy(), (x1, y1, x2, y2), cur_idx)
        cap.release()

        # Export best still frames per track
        still_records: List[Dict[str, Any]] = []
        stills_cfg = self.cfg.get("stills", {"enabled": False})
        if stills_cfg.get("enabled", False) and best_still:
            out_root = Path(output_dir) / Path(video_path).stem
            still_dir = out_root / str(stills_cfg.get("out_dirname", "stills"))
            crop_dir = still_dir / "crops"
            still_dir.mkdir(parents=True, exist_ok=True)
            if stills_cfg.get("save_crop", False):
                crop_dir.mkdir(parents=True, exist_ok=True)

            # Global stills directory (flat corpus)
            global_dir = None
            global_crop_dir = None
            global_dirname = stills_cfg.get("global_dir")
            if global_dirname:
                global_dir = Path(output_dir) / global_dirname
                global_dir.mkdir(parents=True, exist_ok=True)
                if stills_cfg.get("save_crop", False):
                    global_crop_dir = global_dir / "crops"
                    global_crop_dir.mkdir(parents=True, exist_ok=True)

            video_stem = Path(video_path).stem

            # Optional chronological sort of stills by frame index
            items = list(best_still.items())
            if bool(stills_cfg.get("sort_by_frame", False)):
                items.sort(key=lambda kv: kv[1][3])  # sort by fidx

            for tid, (score, img, bbox, fidx) in items:
                x1, y1, x2, y2 = bbox
                img_to_save = img.copy()
                if stills_cfg.get("annotate", True):
                    cv2.rectangle(img_to_save, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_to_save, f"ID {tid}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                # Filename
                try:
                    tid_int = int(tid)
                    tid_str = f"{tid_int:04d}"
                except Exception:
                    tid_str = str(tid)
                # Optionally prefix with frame index for filename-based chronological sorting
                if bool(stills_cfg.get("filename_prefix_frame", False)):
                    still_fn = f"f{fidx:07d}_still_{tid_str}.jpg"
                else:
                    still_fn = f"still_{tid_str}_f{fidx}.jpg"
                still_path = still_dir / still_fn
                cv2.imwrite(str(still_path), img_to_save)
                # Also save to global stills dir if enabled
                if global_dir is not None:
                    global_still_fn = f"{video_stem}_{still_fn}"
                    cv2.imwrite(str(global_dir / global_still_fn), img_to_save)
                crop_path = None
                if stills_cfg.get("save_crop", False):
                    crop = img[y1:y2, x1:x2]
                    if bool(stills_cfg.get("filename_prefix_frame", False)):
                        crop_fn = f"f{fidx:07d}_still_{tid_str}_crop.jpg"
                    else:
                        crop_fn = f"still_{tid_str}_f{fidx}_crop.jpg"
                    crop_path = crop_dir / crop_fn
                    if crop.size > 0:
                        cv2.imwrite(str(crop_path), crop)
                        if global_crop_dir is not None:
                            global_crop_fn = f"{video_stem}_{crop_fn}"
                            cv2.imwrite(str(global_crop_dir / global_crop_fn), crop)
                still_records.append({
                    "track_id": tid,
                    "frame": fidx,
                    "bbox": [x1, y1, x2, y2],
                    "score": float(score),
                    "image": str(still_path),
                    "crop": str(crop_path) if crop_path else None,
                })

            # Optionally write an index CSV sorted by frame for easy review
            if bool(stills_cfg.get("write_index_csv", False)) and still_records:
                try:
                    import csv
                    index_path = still_dir / "stills_index.csv"
                    rows = sorted(still_records, key=lambda r: r.get("frame", 0))
                    with open(index_path, "w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow(["frame", "track_id", "score", "image", "crop", "bbox"])
                        for r in rows:
                            w.writerow([
                                r.get("frame"), r.get("track_id"), f"{r.get('score', 0.0):.4f}", r.get("image"), r.get("crop"), r.get("bbox"),
                            ])
                except Exception:
                    pass

        # Optional re-ID post-filter: copy high-confidence stills to a separate folder
        reid_cfg = self.cfg.get("reid_filter", {}) or {}
        if bool(reid_cfg.get("enabled", False)) and still_records:
            try:
                gal_dir = reid_cfg.get("gallery_dir")
                if gal_dir and str(gal_dir).strip():
                    # choose which images to query: use crop if exists else full still
                    query_paths = [r.get("crop") or r.get("image") for r in still_records]
                    query_paths = [q for q in query_paths if q]
                    # map query path back to still record for richer CSV
                    rec_by_path = {}
                    for r in still_records:
                        qp = r.get("crop") or r.get("image")
                        if qp:
                            rec_by_path[os.path.abspath(qp)] = r
                    scores = score_against_gallery(
                        gallery_dir=str(gal_dir),
                        query_paths=query_paths,
                        arch=str(reid_cfg.get("arch", "osnet_x1_0")),
                        weights=(reid_cfg.get("weights") or None),
                        size=int(reid_cfg.get("size", 256)),
                        batch=int(reid_cfg.get("batch", 64)),
                        device=(None if reid_cfg.get("device", "auto") in (None, "auto") else str(reid_cfg.get("device"))),
                    )
                    thr = float(reid_cfg.get("threshold", 0.88))
                    out_root = Path(output_dir) / Path(video_path).stem
                    hc_dir = out_root / str(reid_cfg.get("out_dirname", "reid_high_conf"))
                    hc_dir.mkdir(parents=True, exist_ok=True)
                    # Optional global aggregate directory
                    global_dir_name = reid_cfg.get("global_dir")
                    global_hc_dir = None
                    if global_dir_name:
                        global_hc_dir = Path(output_dir) / str(global_dir_name)
                        global_hc_dir.mkdir(parents=True, exist_ok=True)

                    # Build an index CSV
                    import csv, shutil
                    index_csv = hc_dir / "reid_high_conf.csv"
                    with open(index_csv, "w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow(["frame", "track_id", "still_score", "orig_image", "orig_crop", "similarity", "best_gallery", "copied_path"])                        
                        for qp, sc, gp in scores:
                            if sc >= thr:
                                src = Path(qp)
                                dst = hc_dir / src.name
                                try:
                                    shutil.copy2(src, dst)
                                    if global_hc_dir is not None:
                                        shutil.copy2(src, global_hc_dir / (Path(video_path).stem + "_" + src.name))
                                except Exception:
                                    pass
                                rec = rec_by_path.get(os.path.abspath(str(src)))
                                w.writerow([
                                    (rec.get("frame") if rec else ""),
                                    (rec.get("track_id") if rec else ""),
                                    (f"{rec.get('score', 0.0):.4f}" if rec else ""),
                                    (rec.get("image") if rec else ""),
                                    (rec.get("crop") if rec else ""),
                                    f"{sc:.4f}",
                                    gp or "",
                                    str(dst),
                                ])
            except Exception:
                # Post-filter is optional; ignore errors to avoid blocking main results
                pass

        runtime = time.time() - start_time
        # Try to get YOLO device string
        try:
            yolo_device = str(getattr(self.detector.model, 'device', getattr(self.detector.model, 'args', {}).get('device', 'unknown')))
        except Exception:
            yolo_device = 'unknown'
        return {
            "video": video_path,
            "fps": fps,
            "width": width,
            "height": height,
            "frames": total_frames,
            "runtime_sec": runtime,
            "yolo_device": yolo_device,
            "vid_stride": vid_stride,
            "diagnostics": {
                "frames_read": frames_read,
                "frames_processed": frames_processed,
                "raw_detections_total": raw_detections_total,
                "color_pass_total": color_pass_total,
                "color_reject_total": color_reject_total,
                "tracks_confirmed_total": tracks_confirmed_total,
            },
            "detections": results,
            "segments": [],
            "stills": still_records,
        }
