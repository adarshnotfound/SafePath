# Module 1: Hazard Detection & Perception

**Owner**: Owner A  
**Key Responsibilities**:
- Sample 1 frame every 1–2 seconds from road video footage.
- Baseline lighting estimation (luminance bucketing into `well lit`, `dim`, `dark`).
- Object detection comparing Baseline (MobileNet-SSD / Lightweight CNN) vs Proposed (YOLOv8-nano).
- Classes: `pothole`, `speed_breaker`, `animal` (mapped from COCO classes).
- Metric evaluation (`mAP@0.5`, precision, recall, model size in MB, latency per frame in ms).
- Edge case & failure mode documentation (shadows, night, glare, blur).
- Output compliance with Section 8 file contract (`detections.csv`).

---

## Architecture Overview

```text
Input (Video / Image Frames)
     │
     ├──> Frame Extractor (1-2s interval sampling via OpenCV)
     │         │
     │         └──> Lighting Estimator (Mean Luminance -> 'well lit', 'dim', 'dark')
     │
     ├──> Baseline Model (MobileNet-SSD / CNN feature extractor heuristic)
     │
     └──> Proposed Model (YOLOv8-nano anchor-free CNN via PyTorch/Ultralytics)
               │
               ├──> Section 8 Exporter (detections.csv)
               └──> Benchmarking & Metrics Logger (detection_comparison.json)
```

---

## Running Module 1

### 1. Run Comparative Benchmark
```bash
python -m detection.evaluate
```
Output results are logged to `evaluation/results/detection_comparison.json`.

### 2. Run Failure Case Analysis
```bash
python -m detection.error_analysis
```
Saves report to `evaluation/results/detection_failure_cases.json`.

### 3. Run Unit & Contract Tests
```bash
pytest -v detection/test_detection.py
```

---

## Viva Talking Points
- **Why YOLOv8-nano?**: Anchor-free head reduces hyperparameter tuning for irregular aspect ratios (e.g. transverse speed breakers vs circular potholes), with ~3.2M parameters (~6.2 MB) ideal for dashcam edge inference.
- **Why evaluate lighting?**: Poor street lighting directly degrades perception recall; feeding lighting status to Module 3 (fuzzy logic) allows the system to lower safety confidence in dark areas.
- **Handling False Positives**: Shadows often mimic potholes in luminance; downstream Module 2 (Living Hazard Map) uses temporal clustering across drives to reject transient false detections.

