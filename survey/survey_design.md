# SafePath Road Safety Validation Survey: Complete Blueprint

**Target:** Human Road Safety Perceptions for Fuzzy Logic Engine Calibration  
**Module Owner:** Member C (Fuzzy Engine & Validation Lead)  
**Sample Target:** 25–30 Road Segments evaluated by $\ge 10$ Participants  

---

## 1. Google Form Setup Guide (Step-by-Step)

### Section 1: Study Information & Ethical Consent
- **Form Title:** SafePath: Road Segment Safety Assessment Survey
- **Form Description:**
  > Welcome to the SafePath road safety research study. This study examines how drivers, riders, and pedestrians evaluate road travel safety under varied real-world Indian road conditions (surface damage, illumination, emergency accessibility, and time of day). Your anonymous feedback establishes the ground-truth benchmark for calibrating soft computing (fuzzy logic) safety models.
- **Question 1 (Consent Gate):**
  - *Type:* Multiple choice (Required)
  - *Prompt:* **Participant Information & Informed Consent**
    - Your participation is entirely voluntary; you may withdraw or close the form at any point without penalty.
    - **Zero Personal Data:** We do NOT collect names, emails, IP addresses, student IDs, or location data.
    - Responses are assigned an anonymous identifier (`rater_01`, `rater_02`, etc.) and aggregated statistically for academic course reporting.
  - *Options:*
    - `[X] I am 18+, have read the information above, and consent to participate anonymously.`
    - `[ ] I do not consent (Exit survey).` *(Set logic: Submit form if selected)*

---

## 2. Objective Rating Guidelines (De-biasing Instructions)

*(To be placed as the Section 2 header in Google Forms to anchor participant scales)*

> ### Rating Instructions (Please Read Carefully)
> - Rate each road stretch on a scale of **1 to 5 Stars / Points**:
>   - **1 = Very Risky:** Hazardous corridor; severe surface potholes/speed breakers, pitch dark, isolated, or history of fatal accidents. Travel should be avoided or navigated at minimal speeds.
>   - **2 = Risky:** Noticeable safety vulnerabilities (e.g., unlit street, significant potholes, or absence of nearby emergency help).
>   - **3 = Caution / Moderate:** Typical urban/suburban road with manageable risks (e.g., occasional speed bumps, moderate traffic, average lighting).
>   - **4 = Relatively Safe:** Good visibility, functioning streetlights, few or well-marked speed breakers, nearby shops/clinics.
>   - **5 = Very Safe:** Well-paved, illuminated primary arterial road with clear lane markings, median barrier, and zero recorded hazards.
>
> **Important De-biasing Rules:**
> 1. Rate safety strictly on the **objective physical features** described (lighting, potholes, proximity to police/hospitals, accidents).
> 2. **Do not rate aesthetic appeal or scenery.** A quiet tree-lined lane that is pitch dark and isolated is high-risk at night despite looking scenic.
> 3. **Avoid assuming night is always unsafe:** A wide, well-lit urban highway with CCTV, 24-hr petrol pumps, and hospitals nearby is significantly safer than an unlit rural road.

---

## 3. Road Segment Catalog (28 Representative Scenarios)

The following 28 segments span the full parameter envelope:

| # | Segment ID | Road Type | Speed Limit | Lighting | Hazard Density | Accident History | Isolation | Time Context | Scenario Description |
| :- | :--- | :--- | :-: | :--- | :--- | :--- | :--- | :-: | :--- |
| 1 | `seg_101` | Secondary Arterial | 40 km/h | Yes (Well-lit) | Moderate (0.83 / 100m) | Minor Blackspot (0.65) | Low (250m to clinic) | Night | Urban lit corridor with noticeable potholes |
| 2 | `seg_102` | Tertiary Street | 30 km/h | No (Pitch dark) | High (1.08 / 100m) | Moderate (0.40) | Medium (800m) | Night | Dark unlit residential connector with road damage |
| 3 | `seg_103` | Secondary Road | 40 km/h | Yes (Well-lit) | Moderate (0.64 / 100m) | Severe Blackspot (0.80)| Low (150m) | Night | Well-lit road intersecting a major accident junction |
| 4 | `seg_104` | Residential Lane | 25 km/h | No (Dark) | Moderate (0.67 / 100m) | Low (0.10) | High (1.5 km to police) | Dusk | Narrow isolated residential stretch during twilight |
| 5 | `seg_105` | Primary Arterial | 50 km/h | Yes (Well-lit) | Low (0.24 / 100m) | Low (0.30) | Low (Urban center) | Day | Wide multi-lane divided road in broad daylight |
| 6 | `seg_106` | Secondary Road | 40 km/h | No (Pitch dark) | High (0.73 / 100m) | High (0.85) | High (Isolated outskirts)| Night | Severe unlit accident hotspot with potholes |
| 7 | `seg_107` | Tertiary Street | 30 km/h | Yes (Well-lit) | Zero (0.00 / 100m) | Low (0.15) | Low (Commercial area) | Day | Smooth paved quiet street with active shops |
| 8 | `seg_108` | Primary Arterial | 50 km/h | Yes (Well-lit) | Low (0.29 / 100m) | Low (0.25) | Low (Near hospital) | Dusk | Evening commute along well-lit primary avenue |
| 9 | `seg_109` | Trunk Highway | 60 km/h | No (Unlit) | High (1.40 / 100m) | High (0.75) | High (Rural edge) | Night | Fast unlit bypass road with multiple unmarked bumps |
| 10| `seg_110` | Residential Road | 20 km/h | Yes (Well-lit) | Low (0.10 / 100m) | Zero (0.05) | Medium (Gated area) | Day | Slow daylight residential neighborhood |
| 11| `seg_111` | Secondary Road | 40 km/h | Unknown (Dim) | High (1.25 / 100m) | Moderate (0.45) | Medium (500m) | Dusk | Dimly lit evening road with active pothole cluster |
| 12| `seg_112` | Primary Arterial | 50 km/h | Yes (Well-lit) | Zero (0.00 / 100m) | Low (0.10) | Low (Police nearby) | Day | Pristine central boulevard during midday |
| 13| `seg_113` | Tertiary Street | 30 km/h | No (Dark) | Moderate (0.50 / 100m) | Low (0.20) | High (Industrial park) | Night | Isolated dark backroad behind warehouses |
| 14| `seg_114` | Secondary Road | 40 km/h | Yes (Well-lit) | Moderate (0.70 / 100m) | Low (0.20) | Low (Market street) | Night | Busy night street with active shops and potholes |
| 15| `seg_115` | Trunk Highway | 60 km/h | Yes (Highway lights)| Low (0.15 / 100m) | High (0.70) | Medium (Flyover) | Night | Elevated flyover section with accident history |
| 16| `seg_116` | Residential Street | 25 km/h | Yes (Well-lit) | High (1.10 / 100m) | Low (0.10) | Low (Suburban colony) | Day | Residential lane undergoing water pipe repair work |
| 17| `seg_117` | Secondary Road | 45 km/h | No (Unlit) | Moderate (0.60 / 100m) | High (0.80) | Medium (Canal bank) | Night | Unlit canal road with sharp bend and crash history |
| 18| `seg_118` | Tertiary Street | 30 km/h | Yes (Well-lit) | Low (0.10 / 100m) | Low (0.15) | Low (Near school) | Day | School zone street with speed bumps and crossings |
| 19| `seg_119` | Primary Arterial | 50 km/h | Unknown (Dim) | Moderate (0.55 / 100m) | Moderate (0.40) | Low (Urban corridor) | Dusk | Twilight arterial with malfunctioning streetlights |
| 20| `seg_120` | Secondary Road | 40 km/h | No (Dark) | High (1.50 / 100m) | Moderate (0.50) | High (Farmlands) | Night | Rural-urban fringe road with roaming cattle hazards |
| 21| `seg_121` | Residential Lane | 20 km/h | No (Dark) | Low (0.20 / 100m) | Zero (0.05) | Low (Dense colony) | Night | Narrow quiet dark colony lane with dead ends |
| 22| `seg_122` | Primary Arterial | 50 km/h | Yes (Well-lit) | Low (0.30 / 100m) | Moderate (0.35) | Low (Commercial hub) | Night | Lit commercial arterial with light nighttime traffic |
| 23| `seg_123` | Secondary Road | 40 km/h | Yes (Well-lit) | High (1.20 / 100m) | Low (0.20) | Low (Near bus station) | Day | Heavy bus corridor with surface rutting and potholes |
| 24| `seg_124` | Tertiary Street | 30 km/h | No (Dark) | Zero (0.00 / 100m) | High (0.75) | High (Blind corner) | Night | Dark unlit blind corner with high rollover history |
| 25| `seg_125` | Primary Arterial | 50 km/h | Yes (Well-lit) | Low (0.10 / 100m) | Low (0.10) | Low (Hospital avenue) | Dusk | Clear hospital avenue during twilight hours |
| 26| `seg_126` | Secondary Road | 40 km/h | Unknown (Dim) | Moderate (0.80 / 100m) | High (0.60) | Medium (Suburban bridge)| Night | Narrow bridge approach with dim lighting and potholes |
| 27| `seg_127` | Residential Street | 25 km/h | Yes (Well-lit) | Low (0.15 / 100m) | Low (0.10) | Low (Apartments area) | Night | Well-lit residential lane with active security |
| 28| `seg_128` | Trunk Highway | 60 km/h | Yes (Well-lit) | Zero (0.00 / 100m) | Low (0.20) | Low (Toll plaza area) | Day | Paved highway straight with wide shoulders |

---

## 4. Google Form Question Template

For each segment $i \in [1, 28]$, add a **Linear Scale** question:
- **Title:** `Segment seg_101: Secondary Arterial Road`
- **Description / Context Box:**
  ```
  - Surface Speed Limit: 40 km/h
  - Illumination: Yes (Streetlights active)
  - Surface Hazards: Moderate (0.83 potholes/bumps per 100m)
  - Accident Exposure: Minor hotspot within 150m
  - Distance to Emergency Support: 250m to nearest clinic / 24-hr petrol pump
  - Time of Day Context: Night
  ```
- **Scale:** `1 to 5`
  - `1 = Very Risky`
  - `5 = Very Safe`

---

## 5. Cleaning & Agreement Pipeline

Once responses are collected from Google Forms (or using simulated development data):
1. Export Google Form responses as CSV (`responses.csv`).
2. Run the SafePath survey cleaning tool:
   ```powershell
   python survey/clean_survey.py
   ```
3. The script validates:
   - Anonymous participant ID assignment (`rater_01`, `rater_02`, ...)
   - Score boundaries ($1 \le \text{rating} \le 5$)
   - Formats to Section 8 contract: `segment_id, rater_id, rating, time_of_day_context`
   - Exports statistical inter-rater agreement to `evaluation/results/survey_agreement.json` (calculating mean pairwise deviation, exact match rate, and Intraclass Correlation Coefficient $ICC(1, 1)$).
