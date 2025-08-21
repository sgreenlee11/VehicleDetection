import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from glob import glob
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


def is_gray_frame(frame: np.ndarray, thr: float = 5.0) -> bool:
    if frame is None or frame.size == 0:
        return True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.var()) < thr


def main():
    ap = argparse.ArgumentParser(description="Extract a single frame as PNG with robust fallback (OpenCV + ffmpeg)")
    ap.add_argument("--input", required=True, help="Video file path (.mp4, .mkv, etc.)")
    ap.add_argument("--out", required=True, help="Output PNG path")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--at", help="Timestamp (HH:MM:SS(.ms))")
    g.add_argument("--frame", type=int, help="Absolute frame index (0-based)")
    ap.add_argument("--probe", action="store_true", help="Print basic video info")
    ap.add_argument("--fallback-scan", type=int, default=10, help="If gray with OpenCV, scan forward up to N frames to find non-gray")
    ap.add_argument("--backend", choices=["auto","opencv","ffmpeg"], default="auto", help="Which backend to prefer")
    ap.add_argument("--window", type=float, default=1.0, help="Burst window seconds for ffmpeg fallback around --at (0.5..3.0)")
    args = ap.parse_args()

    def have_ffmpeg() -> bool:
        return shutil.which("ffmpeg") is not None

    def ffmpeg_single(input_path: str, out_path: str, ts: str | None, frame_idx: int | None) -> bool:
        # Accurate seek: put -ss after -i
        if ts is None and frame_idx is not None:
            # Convert frame to timestamp using fps
            sec = (frame_idx / (fps or 30.0)) if fps else 0
            ts = f"{sec:.3f}"
        if ts is None:
            return False
        cmd = [
            "ffmpeg","-y","-hide_banner","-loglevel","error",
            "-i", input_path,
            "-ss", ts,
            "-frames:v","1",
            "-vsync","vfr",
            out_path,
        ]
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def ffmpeg_burst_pick(input_path: str, out_path: str, ts: str, window: float = 1.0) -> bool:
        # Decode a small window around ts and pick the sharpest non-gray frame
        win = max(0.2, min(3.0, float(window)))
        # Start a bit before ts
        try:
            sec = parse_hms(ts)
        except Exception:
            # parse simple seconds
            sec = float(ts)
        start = max(0.0, sec - win * 0.3)
        with tempfile.TemporaryDirectory(prefix="burst_") as td:
            pat = str(Path(td) / "f_%05d.png")
            cmd = [
                "ffmpeg","-y","-hide_banner","-loglevel","error",
                "-ss", f"{start:.3f}", "-i", input_path,
                "-t", f"{win:.3f}",
                "-vsync","vfr",
                pat,
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                return False
            imgs = sorted(glob(str(Path(td) / "f_*.png")))
            if not imgs:
                return False
            best_path = None
            best_score = -1.0
            for p in imgs:
                im = cv2.imread(p)
                if im is None or im.size == 0:
                    continue
                sc = float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                # Low-variance frames are likely gray/blank
                if sc > best_score and sc > 3.0:
                    best_score = sc
                    best_path = p
            if best_path is None:
                # choose max anyway
                for p in imgs:
                    im = cv2.imread(p)
                    if im is None:
                        continue
                    sc = float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                    if sc > best_score:
                        best_score = sc
                        best_path = p
            if best_path is None:
                return False
            ensure_dir(Path(out_path))
            return shutil.copy2(best_path, out_path) is not None

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if args.probe:
        print(f"Video: {args.input}\n  fps={fps:.3f} frames={total} size={w}x{h}")

    def try_opencv() -> np.ndarray | None:
        # Seek
        target_idx = None
        if args.at:
            sec = parse_hms(args.at)
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
            target_idx = int(round((cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)))
        else:
            target_idx = max(0, int(args.frame))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame0 = cap.read()
        if not ret:
            if target_idx is not None:
                start = max(0, target_idx - 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                for _ in range(5):
                    ret, frame0 = cap.read()
                    if ret:
                        break
        if not ret:
            return None
        # If gray, scan forward a few frames
        frame = frame0
        if is_gray_frame(frame) and args.fallback_scan > 0:
            for _ in range(args.fallback_scan):
                ret, nextf = cap.read()
                if not ret:
                    break
                if not is_gray_frame(nextf):
                    frame = nextf
                    break
        return frame

    out_path = Path(args.out)
    ensure_dir(out_path)

    used_backend = None
    saved = False
    frame = None

    if args.backend in ("auto","opencv"):
        frame = try_opencv()
        if frame is not None and not is_gray_frame(frame):
            ok = cv2.imwrite(str(out_path), frame)
            if not ok:
                raise SystemExit("Failed to write PNG output (OpenCV)")
            used_backend = "opencv"
            saved = True
        elif args.backend == "opencv":
            raise SystemExit("OpenCV decode returned gray/blank frames. Try --backend ffmpeg.")

    if (args.backend == "ffmpeg" or (not saved and have_ffmpeg())):
        ts = args.at
        if ts is None and args.frame is not None:
            ts = f"{(args.frame/(fps or 30.0)):.3f}"
        # First try single-frame accurate seek
        if ts and ffmpeg_single(args.input, str(out_path), ts, None) and out_path.exists():
            test = cv2.imread(str(out_path))
            if test is not None and not is_gray_frame(test):
                used_backend = "ffmpeg-single"
                saved = True
        if not saved and ts:
            # Burst around timestamp and pick sharpest
            if ffmpeg_burst_pick(args.input, str(out_path), ts, window=args.window) and out_path.exists():
                test = cv2.imread(str(out_path))
                if test is not None:
                    used_backend = "ffmpeg-burst"
                    saved = True

    cap.release()
    if not saved:
        raise SystemExit("Failed to extract a non-gray frame with available backends.")
    print(f"Saved {out_path} via {used_backend}")


if __name__ == "__main__":
    main()
