# Accident Data & Severity Modeling: Limitations & Domain Shift Analysis

**SafePath Module 2 — Accident Severity / Risk Prediction**  
**Document Provenance**: Substitute Dataset (UK STATS19 / Department for Transport Road Safety Data)  
**Status**: Experimental Benchmark (Baseline Logistic Regression vs. Proposed MLP)

---

## 1. Overview & Dataset Provenance

In the SafePath system, accident severity and segment risk prediction are modeled using tabular road, environmental, and situational features:
- `road_type`: Road classification (`motorway`, `primary`, `secondary`, `tertiary`, `residential`, `unclassified`)
- `speed_limit`: Posted speed limit (mph / km/h)
- `time_of_day`: Circadian phase (`day`, `dusk`, `night`)
- `lighting`: Illumination status (`well_lit`, `dim`, `dark`)
- `weather`: Atmospheric condition (`fine`, `rain`, `fog`, `snow`)
- `junction`: Intersection proximity (`yes`, `no`)
- **Target (`severity`)**: Categorical risk severity: `slight` (0), `serious` (1), `fatal` (2)

### Provenance Notice: Substitute Dataset
Due to the absence of publicly available, open-source microdata containing individual incident coordinates linked with granular road geometry, lighting conditions, and speed telemetry in India, SafePath uses **UK STATS19** (published by the UK Department for Transport) as a **substitute dataset** for training and benchmarking, explicitly labeled as such in accordance with Sections 0, 4.4, 4.5, 5, 8, and 12 of the SafePath specification.

---

## 2. Experimental Results & The "Accuracy Paradox"

### 2.1 Benchmark Summary

Held-out Test Evaluation ($N = 450$ stratified test samples, seed 42):

| Model Architecture | Class Weighting | Overall Accuracy | Macro F1 | Serious Recall | Fatal Recall | Fatal F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | Unweighted (`None`) | **73.11%** | 0.2881 | 0.0000 | **0.0000** | 0.0000 |
| **Logistic Regression (Baseline)** | Balanced ($w_j = \frac{N}{K \cdot N_j}$) | 49.11% | **0.3736** | 0.2941 | **0.4444** | **0.1143** |
| **Multi-Layer Perceptron (Proposed)** | Balanced ($w_j = \frac{N}{K \cdot N_j}$) | 46.89% | 0.3473 | 0.2941 | **0.4444** | 0.1026 |

### 2.2 Confusion Matrices

#### Baseline: Unweighted Logistic Regression
```
Predicted ->     Slight    Serious    Fatal
True Slight        329        0         0
True Serious       102        0         0
True Fatal          18        0         0
```
> **Observation**: The unweighted baseline suffers from complete majority-class collapse. It predicts `slight` for 100% of samples, achieving 73.11% raw accuracy while completely failing to detect a single fatal or serious accident ($0\%$ recall). In safety-critical navigation, this is unacceptable.

#### Proposed: Balanced MLP Confusion Matrix
```
Predicted ->     Slight    Serious    Fatal
True Slight        173       56        101
True Serious        43       30         29
True Fatal           6        4          8
```
> **Observation**: Applying balanced class loss weights shifts the decision boundary, successfully identifying **44.44% of fatal incidents** and **29.41% of serious incidents**, ensuring high-risk road configurations are flagged to routing algorithms.

---

## 3. Critical Limitations & Domain Shift (UK vs. India)

While UK STATS19 provides standardized, rigorously curated data, **models trained on UK accident records cannot be transferred directly to Indian road networks without localization**. The primary domain discrepancies include:

### 3.1 Traffic Heterogeneity & Vehicle Modality
- **UK STATS19**: Traffic consists predominantly of passenger cars, heavy goods vehicles (HGVs), and buses operating within demarcated lanes.
- **Indian Roads**: Traffic is highly heterogeneous and unsegregated. Two-wheelers (motorcycles, scooters) and three-wheelers (auto-rickshaws) constitute over 70% of vehicle volume and account for the largest proportion of fatalities (MoRTH Road Accidents in India report). Vulnerable road users (pedestrians, street vendors, stray cattle) share vehicular right-of-way.

### 3.2 Road Infrastructure & Access Control
- **UK Infrastructure**: Strict physical grade separation, roundabouts, pedestrian safety barriers, well-maintained lane markings, and standardized speed limit enforcement.
- **Indian Infrastructure**: Pervasive ribbon development along state and national highways, frequent unauthorized median cuts, lack of pedestrian grade crossings, and sudden transitions between divided and undivided carriageways.

### 3.3 Road Surface Hazards & Dynamic Impediments
- **UK**: Potholes and speed bumps are strictly managed and engineered to standard profiles; cattle on roads are virtually non-existent.
- **Indian Realities**: Potholes, unscientific speed bumps (often unmarked and unlit), waterlogging, and stray animals are primary causal or contributing factors in accidents. The UK STATS19 feature set lacks these hazard attributes, which SafePath’s Module 1 (`/detection`) explicitly captures.

### 3.4 Ambient Lighting & Secondary Road Illumination
- **UK**: Standardized street lighting on primary and urban routes, with retroreflective road studs (cat's eyes) on unlit rural routes.
- **Indian Context**: Large sections of peri-urban and rural corridors lack operational illumination. Headlight glare from high-beam misuse on undivided roads substantially degrades driver visibility.

---

## 4. Architectural Integration in SafePath

To bridge this domain gap, SafePath does not rely solely on the tabular STATS19 model:
1. **Multimodal Fusion (Module 3 - Hazard Mapping & Section 4.4)**:
   - Tabular model probabilities provide a prior segment baseline risk.
   - Vision-detected real-time hazards (potholes, speed breakers, stray animals from `/detection`) provide direct localized dynamic risk adjustments.
2. **Transfer Learning & Calibration**:
   - The tabular model is calibrated with local city survey ratings (`survey_ratings.csv`) via Module 5 (Fuzzy Logic Safety Index).
3. **Audit Trail**:
   - Every inference response in `evaluation/results/accident_risk_experiment.json` retains `"provenance": "substitute"`.

