# AI and Pretrained Model Usage Disclosure

This document tracks all external AI tools, coding assistants, pretrained models, and generated code used in **SafePath**, adhering to Section 0 and Section 12 of the project specification.

---

## 1. AI Coding Assistants & Agent Usage

| Date | Tool / Agent | Module(s) Affected | Purpose / Prompt Description | Human Reviewer |
| :--- | :--- | :--- | :--- | :--- |
| 2026-11-01 | Antigravity | All (Skeleton) | Created repository skeleton, Section 8 contracts, CI workflow | Team Owner A |
| 2026-11-11 | Antigravity | `/detection` | Implemented frame sampling, lighting estimator, baseline vs YOLOv8-nano, benchmark evaluation | Team Owner A |
| 2026-11-11 | Antigravity | `/detection`, `/docs` | Implemented RDD2022 dataset preparation pipeline, VOC-to-YOLO conversion, and Data Card | Team Owner A |
| 2026-11-11 | Antigravity | `/fuzzy`, `/survey`, `/evaluation` | Implemented Mamdani fuzzy system, weighted baseline, GA tuning, survey pipeline, ablation | Team Owner C |

---

## 2. Pretrained Models & Weights

| Model Name | Source / Provider | Base Weights | Task in SafePath | Fine-tuned? (Yes/No) |
| :--- | :--- | :--- | :--- | :--- |
| *e.g., YOLOv8-nano* | *Ultralytics* | *yolov8n.pt (COCO)* | *Hazard detection (potholes, speed breakers, animals)* | *Fine-tuned on RDD2022* |
| *e.g., MobileNetV2-SSD*| *Torchvision / TF* | *ImageNet / VOC* | *Baseline detection comparison* | *No / Feature extractor* |

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

