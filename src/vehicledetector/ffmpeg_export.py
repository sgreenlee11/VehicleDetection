from __future__ import annotations
import subprocess
import shutil
from pathlib import Path
from typing import Optional


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def build_time_range_cmd(input_path: str, start_time: float, duration: float, out_path: str,
                         codec: str = "libx264", crf: int = 23, preset: str = "veryfast", reencode: bool = False) -> list:
    args = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_time:.3f}", "-i", input_path]
    if reencode:
        args += ["-t", f"{duration:.3f}", "-c:v", codec, "-crf", str(crf), "-preset", preset, "-c:a", "aac", out_path]
    else:
        # Try stream copy for speed; may fail for some containers/codecs
        args += ["-t", f"{duration:.3f}", "-c", "copy", out_path]
    return args


def export_segment(input_path: str, start_time: float, end_time: float, out_path: str,
                   codec: str = "libx264", crf: int = 23, preset: str = "veryfast", reencode: bool = False) -> bool:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, end_time - start_time)
    cmd = build_time_range_cmd(input_path, start_time, duration, out_path, codec, crf, preset, reencode)
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        if not reencode:
            # Retry with reencode fallback
            cmd = build_time_range_cmd(input_path, start_time, duration, out_path, codec, crf, preset, True)
            try:
                subprocess.run(cmd, check=True)
                return True
            except subprocess.CalledProcessError:
                return False
        return False
