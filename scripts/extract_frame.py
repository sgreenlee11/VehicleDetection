import argparse
from pathlib import Path
import cv2
import numpy as np


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def parse_hms(hms: str) -> float:
    parts = hms.split(":")
    parts = [p.strip() for p in parts]
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        m, s = parts
        return float(m) * 60.0 + float(s)
    h, m, s = parts[-3:]
    return float(h) * 3600.0 + float(m) * 60.0 + float(s)


def is_gray_frame(frame: np.ndarray, thr: float = 2.0) -> bool:
    if frame is None or frame.size == 0:
        return True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.var()) < thr


def main():
    ap = argparse.ArgumentParser(description="Extract a single frame as PNG (OpenCV-based, robust to gray pre-roll frames)")
    ap.add_argument("--input", required=True, help="Video file path (.mp4, .mkv, etc.)")
    ap.add_argument("--out", required=True, help="Output PNG path")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--at", help="Timestamp (HH:MM:SS(.ms))")
    g.add_argument("--frame", type=int, help="Absolute frame index (0-based)")
    ap.add_argument("--probe", action="store_true", help="Print basic video info")
    ap.add_argument("--fallback-scan", type=int, default=10, help="If gray frame, scan forward up to N frames to find non-gray")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if args.probe:
        print(f"Video: {args.input}\n  fps={fps:.3f} frames={total} size={w}x{h}")

    # Seek
    target_idx = None
    if args.at:
        sec = parse_hms(args.at)
        # Accurate seek: set by MSEC and read a frame
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
        target_idx = int(round((cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)))
    else:
        target_idx = max(0, int(args.frame))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)

    ret, frame = cap.read()
    if not ret:
        # Fallback: try a nearby region
        if target_idx is not None:
            start = max(0, target_idx - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for _ in range(5):
                ret, frame = cap.read()
                if ret:
                    break
        if not ret:
            cap.release()
            raise SystemExit("Failed to decode a frame at the requested position")

    # If gray, scan forward a few frames (common with pre-roll/keyframe issues)
    if is_gray_frame(frame) and args.fallback_scan > 0:
        for _ in range(args.fallback_scan):
            ret, nextf = cap.read()
            if not ret:
                break
            if not is_gray_frame(nextf):
                frame = nextf
                break

    cap.release()
    out_path = Path(args.out)
    ensure_dir(out_path)
    ok = cv2.imwrite(str(out_path), frame)
    if not ok:
        raise SystemExit("Failed to write PNG output")
    print(f"Saved {out_path} ({frame.shape[1]}x{frame.shape[0]})")


if __name__ == "__main__":
    main()
