param(
  [string]$Input = "samples",
  [string]$Output = "outputs",
  [string]$Model = "yolov8n.pt"
)

python -m src.vehicledetector.cli --input "$Input" --output "$Output" --model "$Model" --stride-frames 2 --max-fps 15 --confidence 0.25 --sedan-only false
