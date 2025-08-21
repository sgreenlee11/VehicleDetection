# VehicleDetection Tool

A Python pipeline to detect and export stills for white sedans (or white vehicles) across large sets of videos using YOLOv8 and DeepSORT, with optional re-ID high-confidence matching.

## Features

- Aggressive frame sampling with configurable stride or FPS cap
- YOLOv8 object detection (cars, trucks, buses, motorcycles)
- DeepSORT multi-object tracking with robust scaling
- Pearl-white color filter in Lab color space with adaptive thresholds
- Optional sedan-only filtering (configurable; off by default)
- Chronological, per-track best still export with optional crops and index CSV
- Optional re-ID post-filter to copy high-confidence matches to a dedicated folder
- CLI with batching over folders and CSV report output

## Install

1. Create a virtual environment (optional but recommended):

```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

1. Install dependencies:

```pwsh
pip install -r requirements.txt
```

1. (Optional) If you plan to use TorchReID backends, install additional dependencies as noted in `requirements.txt`.

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
- Stills are written under `outputs/<video-stem>/stills/` (and `crops/` if enabled). If `reid_filter.enabled` is true, high-confidence matches will also be copied to `outputs/<video-stem>/reid_high_conf/` and optionally a global folder.

## Notes

- Sedan-only is a heuristic on bbox shape; by default we filter by COCO vehicle classes and color only.
- Color thresholds are tuned for pearl/ivory whites in typical daylight. Adjust in `config.yaml`.
