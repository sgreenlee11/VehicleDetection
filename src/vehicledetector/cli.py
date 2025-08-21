from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
import json
import pandas as pd
from rich.progress import Progress
from .analyzer import VideoAnalyzer


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".webm"}


def collect_videos(path: str, recurse: bool = True) -> List[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)] if p.suffix.lower() in VIDEO_EXTS else []
    vids = []
    if recurse:
        for f in p.rglob("*"):
            if f.suffix.lower() in VIDEO_EXTS:
                vids.append(str(f))
    else:
        for f in p.iterdir():
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
                vids.append(str(f))
    return vids


def main():
    ap = argparse.ArgumentParser(description="Detect and export stills of white vehicles (optionally sedans).")
    ap.add_argument("--input", required=True, help="Video file or directory")
    ap.add_argument("--output", default="outputs", help="Output directory")
    ap.add_argument("--model", default=None, help="YOLO model path, overrides config")
    ap.add_argument("--config", default=None, help="YAML config override path")
    ap.add_argument("--stride-frames", type=int, default=None, help="Sample every Nth frame")
    ap.add_argument("--max-fps", type=float, default=None, help="Cap processing FPS")
    ap.add_argument("--confidence", type=float, default=None, help="Detection confidence threshold")
    ap.add_argument("--device", default=None, help="YOLO device: auto/cpu/cuda/0/1...")
    ap.add_argument("--imgsz", type=int, default=None, help="YOLO image size")
    ap.add_argument("--iou", type=float, default=None, help="NMS IoU threshold")
    ap.add_argument("--classes", default=None, help="Comma-separated class IDs to detect (e.g., 2,5,7)")
    ap.add_argument("--batch-size", type=int, default=None, help="Batch size for inference")
    ap.add_argument("--sedan-only", type=lambda x: str(x).lower() in {"1","true","yes"}, default=None)
    # Color filter knobs
    ap.add_argument("--min-box-area", type=int, default=None, help="Minimum bbox area for color check (px)")
    ap.add_argument("--white-ratio-thr", type=float, default=None, help="White mask ratio threshold [0-1]")
    ap.add_argument("--min-box-area-pct", type=float, default=None, help="Minimum bbox area as fraction of frame area (e.g., 0.001)")
    ap.add_argument("--color-enabled", type=lambda x: str(x).lower() in {"1","true","yes"}, default=None, help="Enable/disable color filter")
    ap.add_argument("--color-mode", choices=["hsv","lab"], default=None, help="Color filter space")
    ap.add_argument("--color-debug-dir", default=None, help="Directory to dump rejected crops/masks for tuning")
    ap.add_argument("--color-debug-max", type=int, default=None, help="Max rejected samples to save")
    ap.add_argument("--report", default="summary.csv", help="CSV filename under output for overall report")
    # Re-ID post-filter options
    ap.add_argument("--reid-enabled", type=lambda x: str(x).lower() in {"1","true","yes"}, default=None, help="Enable re-ID post-filter")
    ap.add_argument("--reid-gallery", default=None, help="Path to gallery images (png/jpg mix supported)")
    ap.add_argument("--reid-threshold", type=float, default=None, help="Cosine similarity threshold (e.g., 0.88)")
    ap.add_argument("--reid-arch", default=None, help="Embedding backbone: osnet_x1_0 (default) or resnet50")
    ap.add_argument("--reid-weights", default=None, help="Optional weights path matching arch")
    ap.add_argument("--reid-size", type=int, default=None, help="Embedding input size (default 256)")
    ap.add_argument("--reid-batch", type=int, default=None, help="Batch size for embedding (default 64)")
    ap.add_argument("--reid-device", default=None, help="Device for embedding: auto/cpu/cuda/0/1...")
    args = ap.parse_args()

    analyzer = VideoAnalyzer(config_path=args.config)
    # CLI overrides
    if args.model:
        analyzer.cfg["model"] = args.model
        analyzer.detector = analyzer.detector.__class__(
            args.model,
            analyzer.cfg.get("classes"),
            analyzer.cfg.get("confidence", 0.25),
            analyzer.cfg.get("iou", 0.5),
            analyzer.cfg.get("device", "auto"),
            analyzer.cfg.get("imgsz", 640),
        )
    if args.stride_frames:
        analyzer.stride_frames = max(1, int(args.stride_frames))
    if args.max_fps is not None:
        analyzer.max_fps = float(args.max_fps)
    if args.confidence is not None:
        analyzer.cfg["confidence"] = float(args.confidence)
        analyzer.detector = analyzer.detector.__class__(
            analyzer.cfg.get("model"),
            analyzer.cfg.get("classes"),
            analyzer.cfg.get("confidence"),
            analyzer.cfg.get("iou"),
            analyzer.cfg.get("device", "auto"),
            analyzer.cfg.get("imgsz", 640),
        )
    if args.iou is not None:
        analyzer.cfg["iou"] = float(args.iou)
        analyzer.detector = analyzer.detector.__class__(
            analyzer.cfg.get("model"), analyzer.cfg.get("classes"), analyzer.cfg.get("confidence", 0.25), analyzer.cfg.get("iou", 0.5), analyzer.cfg.get("device", "auto"), analyzer.cfg.get("imgsz", 640)
        )
    if args.device is not None:
        analyzer.cfg["device"] = args.device
        analyzer.detector = analyzer.detector.__class__(
            analyzer.cfg.get("model"), analyzer.cfg.get("classes"), analyzer.cfg.get("confidence"), analyzer.cfg.get("iou"), args.device, analyzer.cfg.get("imgsz", 640)
        )
    if args.imgsz is not None:
        analyzer.cfg["imgsz"] = int(args.imgsz)
        analyzer.detector = analyzer.detector.__class__(
            analyzer.cfg.get("model"), analyzer.cfg.get("classes"), analyzer.cfg.get("confidence"), analyzer.cfg.get("iou"), analyzer.cfg.get("device", "auto"), int(args.imgsz)
        )
    if args.classes is not None:
        try:
            classes = [int(x) for x in str(args.classes).split(',') if x.strip() != '']
        except Exception:
            classes = None
        analyzer.cfg["classes"] = classes
        analyzer.detector = analyzer.detector.__class__(
            analyzer.cfg.get("model"), analyzer.cfg.get("classes"), analyzer.cfg.get("confidence", 0.25), analyzer.cfg.get("iou", 0.5), analyzer.cfg.get("device", "auto"), analyzer.cfg.get("imgsz", 640)
        )
    if args.batch_size is not None:
        analyzer.batch_size = max(1, int(args.batch_size))
    if args.sedan_only is not None:
        analyzer.sedan_only = bool(args.sedan_only)
    # Color filter overrides
    if args.min_box_area is not None:
        analyzer.cfg.setdefault("color_filter", {})["min_box_area"] = int(args.min_box_area)
    if args.white_ratio_thr is not None:
        analyzer.cfg.setdefault("color_filter", {})["white_ratio_threshold"] = float(args.white_ratio_thr)
    if args.min_box_area_pct is not None:
        analyzer.cfg.setdefault("color_filter", {})["min_box_area_pct"] = float(args.min_box_area_pct)
    if args.color_enabled is not None:
        analyzer.cfg.setdefault("color_filter", {})["enabled"] = bool(args.color_enabled)
    if args.color_mode is not None:
        analyzer.cfg.setdefault("color_filter", {})["mode"] = str(args.color_mode)
    if args.color_debug_dir is not None:
        analyzer.cfg.setdefault("color_filter", {})["debug_dir"] = str(args.color_debug_dir)
    if args.color_debug_max is not None:
        analyzer.cfg.setdefault("color_filter", {})["debug_max"] = int(args.color_debug_max)

    # Re-ID post-filter overrides
    if any([
        args.reid_enabled is not None,
        args.reid_gallery is not None,
        args.reid_threshold is not None,
        args.reid_arch is not None,
        args.reid_weights is not None,
        args.reid_size is not None,
        args.reid_batch is not None,
        args.reid_device is not None,
    ]):
        rf = analyzer.cfg.setdefault("reid_filter", {})
        if args.reid_enabled is not None:
            rf["enabled"] = bool(args.reid_enabled)
        if args.reid_gallery is not None:
            rf["gallery_dir"] = str(args.reid_gallery)
        if args.reid_threshold is not None:
            rf["threshold"] = float(args.reid_threshold)
        if args.reid_arch is not None:
            rf["arch"] = str(args.reid_arch)
        if args.reid_weights is not None:
            rf["weights"] = str(args.reid_weights)
        if args.reid_size is not None:
            rf["size"] = int(args.reid_size)
        if args.reid_batch is not None:
            rf["batch"] = int(args.reid_batch)
        if args.reid_device is not None:
            rf["device"] = str(args.reid_device)

    videos = collect_videos(args.input, recurse=True)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    summary_rows = []
    with Progress() as progress:
        task = progress.add_task("Processing videos", total=len(videos))
        for v in videos:
            res = analyzer.analyze_video(v, args.output)
            # Write per-video JSON
            out_json = Path(args.output) / Path(v).stem / "result.json"
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(res, indent=2))
            # Summary is currently focused on per-video runtime and stills count
            summary_rows.append({
                "video": v,
                "runtime_sec": res.get("runtime_sec"),
                "frames": res.get("frames"),
                "stills_count": len(res.get("stills", [])),
            })
            progress.advance(task, 1)

    # Write CSV summary
    if summary_rows:
        import pandas as pd
        df = pd.DataFrame(summary_rows)
        df.to_csv(Path(args.output) / args.report, index=False)


if __name__ == "__main__":
    main()
