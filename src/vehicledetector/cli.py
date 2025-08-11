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
    ap = argparse.ArgumentParser(description="Detect and export segments containing white vehicles (optionally sedans).")
    ap.add_argument("--input", required=True, help="Video file or directory")
    ap.add_argument("--output", default="outputs", help="Output directory")
    ap.add_argument("--model", default=None, help="YOLO model path, overrides config")
    ap.add_argument("--config", default=None, help="YAML config override path")
    ap.add_argument("--stride-frames", type=int, default=None, help="Sample every Nth frame")
    ap.add_argument("--max-fps", type=float, default=None, help="Cap processing FPS")
    ap.add_argument("--confidence", type=float, default=None, help="Detection confidence threshold")
    ap.add_argument("--device", default=None, help="YOLO device: auto/cpu/cuda/0/1...")
    ap.add_argument("--imgsz", type=int, default=None, help="YOLO image size")
    ap.add_argument("--sedan-only", type=lambda x: str(x).lower() in {"1","true","yes"}, default=None)
    ap.add_argument("--report", default="summary.csv", help="CSV filename under output for overall report")
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
    if args.sedan_only is not None:
        analyzer.sedan_only = bool(args.sedan_only)

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
            # Add summary rows for exported segments
            for seg in res.get("segments", []):
                summary_rows.append({
                    "video": v,
                    **seg,
                })
            progress.advance(task, 1)

    # Write CSV summary
    if summary_rows:
        import pandas as pd
        df = pd.DataFrame(summary_rows)
        df.to_csv(Path(args.output) / args.report, index=False)


if __name__ == "__main__":
    main()
