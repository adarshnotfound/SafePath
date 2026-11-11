# Data Card: RDD2022 (Road Damage Dataset 2022 - India Subset)

**Dataset Name**: RDD2022 - Road Damage Dataset 2022 (India Subset)  
**Task in SafePath**: Training and benchmarking object detection perception models (Module 1: Baseline vs. YOLOv8-nano).  
**Data Provenance Label**: `real` (India) — per Section 5 & 12 specification.

---

## 1. Dataset Summary & Origin

The Road Damage Dataset 2022 (RDD2022) was released as part of the Global Road Damage Detection Challenge (CRDDC 2022) hosted at IEEE BigData 2022. It comprises multi-country smartphone/dashcam road imagery, including a dedicated **India subset** collected across Indian urban and arterial roads.

- **Primary Source**: IEEE BigData CRDDC 2022 (University of Tokyo / Consortium)
- **Geographic Coverage**: Indian urban roads, arterial routes, and highways.
- **Collection Mode**: Smartphone cameras mounted inside passenger vehicle dashboards driving in daylight.
- **Native Format**: Pascal VOC XML annotations paired with standard RGB JPEG imagery (typical resolution $1280 \times 720$).

---

## 2. Taxonomy & Label Mapping

In RDD2022, surface damages are categorized according to international road damage standards (`D00`, `D10`, `D20`, `D40`). For **SafePath**, these are translated into Section 4.1's core hazard classes:

| Raw Dataset Class | Description | SafePath Target Class | SafePath Class ID | Handling in SafePath |
| :--- | :--- | :--- | :--- | :--- |
| **`D40`** | Pothole / rutting cavity | **`pothole`** | `0` | Primary training target for road depressions. |
| **`speed_breaker`** / `bump` | Speed breaker / calming hump | **`speed_breaker`** | `1` | Sourced from RDD bump tags & Roboflow road bump additions. |
| **`animal`** / `cow` / `dog` | Stray cattle, dogs, horses | **`animal`** | `2` | Sourced from IDD & COCO mammal classes. |
| **`D00`, `D10`, `D20`** | Linear / alligator surface cracks | *Filtered out* | N/A | Excluded to focus navigation hazard scoring on vehicle-destabilizing obstacles. |

---

## 3. Dataset Splits & Preprocessing

- **Format Conversion**: Transformed via [`detection/data_prep.py`](../detection/data_prep.py) from VOC pixel coordinates `[xmin, ymin, xmax, ymax]` to YOLO normalized relative coordinates `[class_id, x_center, y_center, width, height]`.
- **Partitioning**: Deterministic 80/20 train/validation split using random seed `42` (Rule 4).
- **Configuration**: Exported as [`data/rdd2022_safepath.yaml`](../data/rdd2022_safepath.yaml) for direct ingestion by Ultralytics YOLOv8.

---

## 4. Optical Characteristics & Failure Modes (Section 4.1)

1. **Tree & Canopy Shadows**: Heavy foliage cast dark irregular shadow patches on Indian roads, frequently causing false-positive pothole alarms in the baseline detector.
2. **Monsoon / Wet Asphalt**: Specular reflection from wet bitumen obscures depth cavity borders.
3. **Class Imbalance**: Raw potholes (`D40`) are less frequent than surface hairline cracks; speed breakers require supplemental datasets (Roboflow Universe) to maintain balanced precision.

---

## 5. Licensing & Ethical Considerations

- **License**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
- **Privacy & PII**: Faces and license plates in public road imagery are either blurred or out of focal range.
- **No Crime Data / Responsible AI**: Contains purely physical, observable road surface hazards. Does not contain any neighbourhood socio-economic or crime indicators (Section 4.8 & 12).
