from __future__ import annotations
import cv2
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import numpy as np
from .detector import YoloDetector
from .tracker import MultiObjectTracker
from .color_utils import is_pearl_white, check_white
from .ffmpeg_export import export_segment, have_ffmpeg
from collections import defaultdict


def load_config(path: Optional[str]) -> dict:
    base = Path(__file__).with_name("config.yaml")
    cfg = {}
    if base.exists():
        cfg = yaml.safe_load(base.read_text())
    if path and Path(path).exists():
        user = yaml.safe_load(Path(path).read_text())
        cfg.update(user)
    return cfg


class VideoAnalyzer:
    def __init__(self, config_path: Optional[str] = None):
        self.cfg = load_config(config_path)
        self.detector = YoloDetector(
            model_path=self.cfg.get("model", "yolov8n.pt"),
            classes=self.cfg.get("classes", [2, 3, 5, 7]),
            conf=float(self.cfg.get("confidence", 0.25)),
            iou=float(self.cfg.get("iou", 0.5)),
            device=self.cfg.get("device", "auto"),
            imgsz=int(self.cfg.get("imgsz", 640)),
        )
        tcfg = self.cfg.get("tracking", {})
        self.tracker = MultiObjectTracker(
            max_age=int(tcfg.get("max_age", 30)),
            n_init=int(tcfg.get("n_init", 3)),
            max_iou_distance=float(tcfg.get("max_iou_distance", 0.7)),
            max_cosine_distance=float(tcfg.get("max_cosine_distance", 0.2)),
            nn_budget=int(tcfg.get("nn_budget", 100)),
        )
        self.color_cfg = self.cfg.get("color_filter", {})
        self.sedan_only = bool(self.cfg.get("sedan_only", False))
        self.seg_cfg = self.cfg.get("segment_export", {})
        self.max_fps = float(self.cfg.get("max_fps", 0))
        self.stride_frames = max(1, int(self.cfg.get("stride_frames", 2)))
        self.batch_size = int(self.cfg.get("batch_size", 1))  # Default 1, user can override

    def frame_should_process(self, frame_idx: int) -> bool:
        return frame_idx % self.stride_frames == 0

    def maybe_cap_fps(self, vid_fps: float) -> int:
        if self.max_fps and vid_fps > 0:
            stride = int(max(1, round(vid_fps / self.max_fps)))
            return stride
        return self.stride_frames

    def sedan_shape_heuristic(self, bbox: Tuple[int, int, int, int]) -> bool:
        # Rough heuristic: sedans tend to have width>height and moderate aspect between 1.2 and 3.5
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        aspect = w / h
        return 1.2 <= aspect <= 3.5


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

        segments: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        track_last_seen: Dict[int, int] = {}
        track_first_seen: Dict[int, int] = {}
        best_still: Dict[int, Tuple[float, np.ndarray, Tuple[int, int, int, int], int]] = {}
        frame_idx = -1
        processed_idx = -1
        results = []
        debug_rejects_saved = 0

        batch_frames = []
        batch_indices = []

        def process_batch(frames, indices):
            if not frames:
                return []
            dets_list = self.detector.model.predict(source=frames, classes=self.detector.classes, conf=self.detector.conf, iou=self.detector.iou, verbose=False, device=self.detector.device, imgsz=self.detector.imgsz) if len(frames) > 1 else [self.detector.model.predict(source=frames[0], classes=self.detector.classes, conf=self.detector.conf, iou=self.detector.iou, verbose=False, device=self.detector.device, imgsz=self.detector.imgsz)[0]]
            return dets_list

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % vid_stride != 0:
                continue
            processed_idx += 1
            batch_frames.append(frame)
            batch_indices.append(frame_idx)
            if len(batch_frames) < self.batch_size:
                continue

            dets_list = process_batch(batch_frames, batch_indices)
            for det_idx, det in enumerate(dets_list):
                cur_frame = batch_frames[det_idx]
                cur_idx = batch_indices[det_idx]
                detections = []
                if det.boxes is not None:
                    for b in det.boxes:
                        xyxy = b.xyxy[0].cpu().numpy().astype(int)
                        conf = float(b.conf[0].cpu().numpy()) if b.conf is not None else 0.0
                        cls = int(b.cls[0].cpu().numpy()) if b.cls is not None else -1
                        detections.append((xyxy, conf, cls))
                tracks = self.tracker.update(detections, cur_frame)
                for tid, bbox in tracks:
                    x1, y1, x2, y2 = bbox
                    if self.sedan_only and not self.sedan_shape_heuristic((x1, y1, x2, y2)):
                        continue
                    if self.color_cfg.get("enabled", True):
                        ok, ratio, crop, mask = check_white(cur_frame, (x1, y1, x2, y2), self.color_cfg)
                        if not ok:
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
                    if tid not in track_first_seen:
                        track_first_seen[tid] = cur_idx
                    track_last_seen[tid] = cur_idx
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
            dets_list = process_batch(batch_frames, batch_indices)
            for det_idx, det in enumerate(dets_list):
                cur_frame = batch_frames[det_idx]
                cur_idx = batch_indices[det_idx]
                detections = []
                if det.boxes is not None:
                    for b in det.boxes:
                        xyxy = b.xyxy[0].cpu().numpy().astype(int)
                        conf = float(b.conf[0].cpu().numpy()) if b.conf is not None else 0.0
                        cls = int(b.cls[0].cpu().numpy()) if b.cls is not None else -1
                        detections.append((xyxy, conf, cls))
                tracks = self.tracker.update(detections, cur_frame)
                for tid, bbox in tracks:
                    x1, y1, x2, y2 = bbox
                    if self.sedan_only and not self.sedan_shape_heuristic((x1, y1, x2, y2)):
                        continue
                    if self.color_cfg.get("enabled", True):
                        ok, ratio, crop, mask = check_white(cur_frame, (x1, y1, x2, y2), self.color_cfg)
                        if not ok:
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
                    if tid not in track_first_seen:
                        track_first_seen[tid] = cur_idx
                    track_last_seen[tid] = cur_idx
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

        # Build segments for each track
        pre_s = float(self.seg_cfg.get("pre_seconds", 5))
        post_s = float(self.seg_cfg.get("post_seconds", 5))
        codec = self.seg_cfg.get("codec", "libx264")
        crf = int(self.seg_cfg.get("crf", 23))
        preset = self.seg_cfg.get("preset", "veryfast")
        reencode = bool(self.seg_cfg.get("reencode", False))

        video_name = Path(video_path).stem
        out_root = Path(output_dir) / video_name
        seg_dir = out_root / "segments"
        out_root.mkdir(parents=True, exist_ok=True)
        seg_dir.mkdir(parents=True, exist_ok=True)

        segment_records = []
        for tid, start_f in track_first_seen.items():
            end_f = track_last_seen.get(tid, start_f)
            start_t = max(0.0, (start_f / fps) - pre_s)
            end_t = (end_f / fps) + post_s
            # Format tid for filename: zero-padded if int, else as string
            try:
                tid_int = int(tid)
                tid_str = f"{tid_int:04d}"
            except Exception:
                tid_str = str(tid)
            out_path = seg_dir / f"track_{tid_str}.mp4"
            ok = False
            if have_ffmpeg():
                ok = export_segment(video_path, start_t, end_t, str(out_path), codec, crf, preset, reencode)
            segment_records.append({
                "track_id": tid,
                "start_frame": start_f,
                "end_frame": end_f,
                "start_time": start_t,
                "end_time": end_t,
                "output": str(out_path),
                "exported": bool(ok),
            })

        # Export best still frames per track
        still_records = []
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
            for tid, (score, img, bbox, fidx) in best_still.items():
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
            "detections": results,
            "segments": segment_records,
            "stills": still_records,
        }
