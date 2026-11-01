# SafePath (RaahSuraksha)

> **A Soft Computing Route Safety Analyzer**  
> Evaluates and scores road segments based on road hazards, accident blackspots, isolation, and lighting proxies using neural networks, fuzzy logic, neuro-fuzzy systems (ANFIS), and genetic algorithms.

---

## 1. Project Overview

Standard navigation systems prioritize the fastest path, often steering commuters through hazardous, poorly lit, or high-accident stretches. SafePath calculates and compares 2–3 alternative routes between an origin and destination, providing an aggregate safety score alongside travel duration and human-readable explanations (e.g., *"Route B is 4 minutes slower but avoids two accident hotspots and a poorly lit stretch"*).

### Soft Computing Pillars
* **Perception (Neural Networks)**: Pothole, speed breaker, and animal detection from road video frames (YOLOv8-nano vs. MobileNet-SSD / CNN baseline).
* **Spatial & Temporal Memory (Living Hazard Map)**: DBSCAN clustering to merge duplicate observations, with confidence rise and type-specific decay.
* **Accident Risk (Clustering & Risk MLP)**: Fuzzy C-Means vs. DBSCAN for hotspot clustering; MLP vs. Logistic Regression for accident risk classification.
* **Safety Reasoning (Fuzzy & Neuro-Fuzzy Engine)**: Mamdani fuzzy rules tuned via ANFIS / Genetic Algorithms, benchmarked against weighted sum baselines and validated via human survey ratings.
* **Safety Routing**: Length-weighted safety score aggregation across OpenStreetMap (OSMnx) graph edges with alternative route ranking.

---

## 2. Repository Structure & Team Ownership

The repository is organized into isolated modules communicating strictly via the data contracts in `data/sample/` (Section 8 of the specification):

```text
SafePath/
├── .github/
│   ├── workflows/ci.yml          # GitHub Actions (ruff + pytest)
│   ├── CODEOWNERS                # Folder-based team codeownership
│   └── PULL_REQUEST_TEMPLATE.md  # PR checklist for contracts & data honesty
├── app/                          # Owner D: Streamlit frontend & FastAPI backend
├── routing/                      # Owner D: OSMnx road graph & route aggregation
├── detection/                    # Owner A: Hazard detection models & training
├── hazard_map/                   # Owner B: Merge & decay logic, simulated drive generator
├── accidents/                    # Owner B: Hotspot clustering (FCM) & accident-risk MLP
├── fuzzy/                        # Owner C: Mamdani rules, ANFIS/GA tuning, explainability
├── survey/                       # Owner C: Human safety rating surveys & validation
├── data/
│   ├── sample/                   # Section 8 file contract sample files (tracked)
│   ├── raw/                      # Raw datasets (gitignored)
│   └── processed/                # Processed datasets (gitignored)
├── docs/
│   └── AI_USAGE.md               # AI disclosure, pretrained weights & data provenance
├── evaluation/                   # Shared: Baselines, ablations, benchmark results
├── tests/                        # Automated sanity checks and schema validation
├── requirements.txt              # Project dependency manifest
└── README.md
```

### Module Ownership Matrix (Section 11)
* **Owner A**: `/detection/` (Detector models, comparison, failure analysis)
* **Owner B**: `/hazard_map/`, `/accidents/` (Merge/decay, simulated drives, accident clustering, risk MLP)
* **Owner C**: `/fuzzy/`, `/survey/` (Mamdani rules, ANFIS/GA parameter tuning, survey data)
* **Owner D**: `/routing/`, `/app/` (OSMnx graph, alternative route ranking, Streamlit/FastAPI UI)
* **Shared / Integration**: `/data/`, `/evaluation/`, `/docs/`

---

## 3. Getting Started & Setup

### Prerequisites
* Python 3.10 or higher
* `pip` and `virtualenv`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-org>/SafePath.git
   cd SafePath
   ```

2. Create and activate a Python virtual environment:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 4. Running Module Hello-World Scripts

Each module can be verified independently:
```bash
python -m detection.hello
python -m hazard_map.hello
python -m accidents.hello
python -m fuzzy.hello
python -m survey.hello
python -m routing.hello
python -m app.hello
python -m evaluation.hello
```

---

## 5. Testing & Code Quality

Run lint checks with `ruff`:
```bash
ruff check .
```

Run test suite with `pytest`:
```bash
pytest -v
```

All pull requests trigger the GitHub Actions workflow (`.github/workflows/ci.yml`), enforcing linting and data contract verification.

---

## 6. Data Contracts (Section 8)

All inter-module communication is decoupled via plain CSV/JSON files in `data/sample/`:
* `detections.csv`: Individual bounding box detections from video frames.
* `hazards.csv`: Aggregated living hazard records with confidence, status, and freshness.
* `accident_hotspots.csv`: Clustered accident blackspots with severity weights.
* `segment_features.csv`: Fused OSM segment metrics (hazard density, lighting, isolation).
* `survey_ratings.csv`: Subjective human safety scores (1–5 scale).
* `segment_scores.csv`: Fuzzy/ANFIS inference outputs, risk levels, and explanations.
* `routes.json`: Ranked routes with trade-off analysis between duration and safety.

---

## 7. Responsible AI & Data Honesty (Sections 0, 4.8, 12)

1. **No Fieldwork / Data Honesty**: All simulated or substitute datasets must be explicitly marked (`is_simulated = True` or labeled `substitute`). Never present simulated data as real.
2. **No Crime-Based Profiling**: SafePath uses observable physical proxies only (lighting, street activity, isolation). Crime datasets (e.g. NCRB) are not used to avoid stigmatising neighborhoods.
3. **Disclosure**: Any external pretrained model (YOLO, MobileNet) or AI-assisted code must be logged in [`docs/AI_USAGE.md`](docs/AI_USAGE.md).

