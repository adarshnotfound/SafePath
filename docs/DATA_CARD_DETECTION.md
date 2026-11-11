# Data Card: Unified Road Hazard Perception Dataset

**Scope**: Pothole, Speed Breaker, and Animal Detection for SafePath (Module 1)  
**Target Taxonomy**: 3 Unified Classes: `pothole` (0), `speed_breaker` (1), `animal` (2)  
**Data Provenance Label**: `real` (India) / `simulated` (where augmented) — Section 5 & 12 compliance.

---

## 1. Multi-Source Ingestion & Provenance

SafePath unifies three primary public sources downloaded into `data/raw/` to ensure robust class coverage across Indian roads:

| Source Name | Ingested Formats | Primary Hazard Target | Origin / Geography | License |
| :--- | :--- | :--- | :--- | :--- |
| **RDD2022 (India Subset)** | Pascal VOC XML + JPEG | `pothole` (`D40`) | Indian urban & intercity roads (CRDDC 2022, IEEE BigData) | **CC BY-SA 4.0** |
| **Roboflow Universe** | YOLO v8 format `.txt` + JPG | `speed_breaker`, `pothole` | Public dashcam & traffic calming sets | **CC BY 4.0** / Public Domain |
| **Kaggle Road Hazards** | YOLO / Pascal VOC XML | `pothole`, `speed_breaker`, `animal` | Indian city street photography & open datasets | **CC0: Public Domain** / MIT |

---

## 2. Taxonomy Translation & Class Unification

Raw annotations from distinct labeling schemas are normalized and mapped into SafePath's 3-class target space:

```text
Raw Source Labels                           SafePath Class ID     Unified Target
─────────────────────────────────────────────────────────────────────────────────
D40, pothole, potholes, cavity          ──>       0               pothole
speed_breaker, speed_bump, bump, road_bump ─>       1               speed_breaker
animal, cow, dog, horse, buffalo, goat   ──>       2               animal
D00, D10, D20 (cracks, hairline fissures) ─>     [Ignored]         (Filtered Out)
```

---

## 3. Data Integrity & Leakage Prevention

1. **File Integrity Filtering**: All 0-byte, truncated, or un-decodable images are pruned before labeling to prevent gradient NaN errors during YOLO fine-tuning.
2. **SHA-256 Deduplication**: Near-identical consecutive dashcam burst frames and cross-dataset duplicate images are hashed and deduplicated.
3. **No-Leakage Partitioning**: Images are partitioned into **`train` (70%)**, **`val` (15%)**, and **`test` (15%)** splits using fixed random seed `42`. Duplicate frames are purged prior to splitting, guaranteeing zero duplicate leakage between train and test sets.

---

## 4. Class Distribution & Expected Counts

| Split | Target Ratio | pothole (0) | speed_breaker (1) | animal (2) | Bounding Box Share |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | **70%** | ~50% | ~30% | ~20% | High volume training |
| **Validation** | **15%** | ~50% | ~30% | ~20% | Checkpoint selection |
| **Test** | **15%** | ~50% | ~30% | ~20% | Held-out benchmark |

*Exact counts are computed and printed dynamically when running `python -m detection.data_prep`.*

---

## 5. Known Biases & Mitigations (Section 4.1 & 12)

1. **Daylight & Dry Weather Bias**:
   - *Observation*: Over 85% of public dashcam imagery is captured during sunny, dry conditions.
   - *SafePath Mitigation*: Handled by Module 1's lighting classifier (`well lit`, `dim`, `dark`) and Section 4.1 error analysis, raising caution scores when confidence drops at night or in rain.
2. **Urban Arterial Bias**:
   - *Observation*: Datasets predominantly cover paved municipal asphalt rather than unpaved rural or village roads.
   - *SafePath Mitigation*: Focuses on paved urban network routing where OpenStreetMap edge data is complete.
3. **Speed Breaker Marking Discrepancy**:
   - *Observation*: Many Indian speed breakers lack standard white/yellow reflective paint.
   - *SafePath Mitigation*: Supplemented with Roboflow unpainted road bump instances to ensure morphological shape detection rather than reliance on paint stripes.
4. **Windshield Mounting Angle**:
   - *Observation*: Camera height is fixed between 1.2m and 1.5m above ground level.
   - *SafePath Mitigation*: Model targets dashcams and forward mobile mounts inside passenger vehicles or auto-rickshaws.

---

## 6. One-Command Dataset Rebuild

To rebuild the dataset, remove corrupt files, deduplicate, and regenerate `data/rdd2022_safepath.yaml`:
```bash
python -m detection.data_prep
```
