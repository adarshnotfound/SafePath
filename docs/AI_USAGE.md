# AI and Pretrained Model Usage Disclosure

This document tracks all external AI tools, coding assistants, pretrained models, and generated code used in **SafePath**, adhering to Section 0 and Section 12 of the project specification.

---

## 1. AI Coding Assistants & Agent Usage

| Date | Tool / Agent | Module(s) Affected | Purpose / Prompt Description | Human Reviewer |
| :--- | :--- | :--- | :--- | :--- |
| 2026-11-01 | Antigravity | All (Skeleton) | Created repository skeleton, Section 8 contracts, CI workflow | Team Owner A |
| 2026-11-11 | Antigravity | `/detection` | Implemented frame sampling, lighting estimator, baseline vs YOLOv8-nano, benchmark evaluation | Team Owner A |
| 2026-11-11 | Antigravity | `/detection`, `/docs` | Implemented RDD2022 dataset preparation pipeline, VOC-to-YOLO conversion, and Data Card | Team Owner A |
| 2026-11-11 | Antigravity | `/detection` | Implemented video drive processor, trajectory waypoint interpolation, and Section 4.2 GPS noise modeling | Team Owner A |
| 2026-11-11 | Antigravity | `/detection` | Implemented LightingCNN classifier, baseline vs CNN glare benchmark, and Section 3 stretch goal | Team Owner A |
| 2026-11-11 | Antigravity | `/detection`, `/docs` | Unified multi-source dataset builder (RDD2022, Roboflow, Kaggle), corrupt filtering, SHA-256 deduplication, 3-way partition (seed 42), and comprehensive Data Card | Team Owner A |
| 2026-11-11 | Antigravity | `/detection`, `/docs` | Fair detector training & benchmark (hardware profiler, 640x640 resolution, shared augmentations), 3x3 confusion matrix, 20 visual failure images, and hyperparameter justification doc | Team Owner A |
| 2026-11-11 | Antigravity | `/detection` | Custom-threshold lighting classifier (`classify_lighting`) with calibration, and batch YOLO folder detector exporting contract-compliant `detections.csv` | Team Owner A |
| 2026-11-11 | Antigravity | `/accidents`, `/evaluation`, `/docs` | Implemented UK STATS19 tabular accident severity prediction pipeline, leakage-free 3-way split, baseline Logistic Regression vs proposed MLP, class imbalance handling, metrics JSON, and domain transfer analysis | Team Owner B / A |

---

## 2. Pretrained Models & Weights

| Model Name | Source / Provider | Base Weights | Task in SafePath | Fine-tuned? (Yes/No) |
| :--- | :--- | :--- | :--- | :--- |
| *e.g., YOLOv8-nano* | *Ultralytics* | *yolov8n.pt (COCO)* | *Hazard detection (potholes, speed breakers, animals)* | *Fine-tuned on RDD2022* |
| *e.g., MobileNetV2-SSD*| *Torchvision / TF* | *ImageNet / VOC* | *Baseline detection comparison* | *No / Feature extractor* |
| `LightingCNN` | SafePath In-House | Scratch (3-stage CNN) | Lighting classification (well lit, dim, dark) | Trained on calibrated road patches |

---

## 3. External Datasets & APIs

| Dataset / API | Source URL / Provider | Native Geography | SafePath Usage | Provenance Label (`real` / `simulated` / `substitute`) |
| :--- | :--- | :--- | :--- | :--- |
| *e.g., OpenStreetMap* | *OSMnx / Overpass* | *India (Chosen city)* | *Road network graph & proxies* | `real` |
| *e.g., RDD2022* | *CRDDC 2022* | *India* | *Pothole detection training* | `real` |
| *e.g., UK STATS19* | *data.gov.uk* | *United Kingdom* | *Accident risk model fallback* | `substitute` |
| *e.g., Simulated Drives*| *In-house generator*| *Synthetic* | *Merge/decay timeline validation* | `simulated` |

---

## 4. Code Generation Notes & Viva Explanation Readiness
- **Rule Verification**: Does the human team understand the math and algorithmic flow of every generated file?
- **Viva Preparation Checklist**:
  - [ ] Neural network architecture and loss functions explained.
  - [ ] Fuzzy membership functions, Mamdani inference, and defuzzification explained.
  - [ ] DBSCAN clustering metric and distance threshold rationale explained.
  - [ ] A* / Dijkstra / OSMnx routing aggregation method explained.

