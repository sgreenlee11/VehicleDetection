# VehicleDetection Tool

A Python pipeline to detect and extract video segments containing white sedans (or white vehicles) across large sets of videos using YOLOv8, DeepSORT, and ffmpeg.

## Features
- Aggressive frame sampling with configurable stride or FPS cap
- YOLOv8 object detection (cars, trucks, buses, motorcycles)
- DeepSORT multi-object tracking with robust scaling
- Pearl-white color filter in Lab color space with adaptive thresholds
- Optional sedan-only filtering (configurable; off by default)
- Segment export: 5s before first detection to 5s after last detection per track using ffmpeg
- CLI with batching over folders and CSV report output

## Install

1. Create a virtual environment (optional but recommended):
```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```pwsh
pip install -r requirements.txt
```

3. Ensure ffmpeg is installed and on PATH:
- Windows: install from https://www.gyan.dev/ffmpeg/builds/ and add `ffmpeg.exe` to PATH.

## Usage

```pwsh
python -m src.vehicledetector.cli `
  --input "c:\path\to\videos" `
  --output "outputs" `
  --model yolov8n.pt `
  --stride-frames 2 `
  --max-fps 15 `
  --confidence 0.25 `
  --sedan-only false
```

- Use `--input` to point to a file or directory (recurses by default).
- Segments are written under `outputs/<video-stem>/segments/` with a CSV report.

## Notes
- Sedan-only requires a classifier or better model labels; by default we filter by COCO vehicle classes and color only.
- Color thresholds are tuned for pearl/ivory whites in typical daylight. Adjust in `config.yaml`.
