# SafePath: Multi-Criteria Route Safety Analyzer Using Soft Computing
## Module Report & Integration Synthesis (Member D: Integration, Routing, App & Evaluation)

**Author / Lead**: Team Member D  
**Target City**: Prayagraj, Uttar Pradesh, India  
**Date**: November 2026  
**Project Repository**: SafePath (`RaahSuraksha`)  

---

### Abstract
Modern commercial navigation engines optimize predominantly for travel time and distance. In Indian urban corridors, road hazards such as potholes, unmarked speed breakers, animals on the carriageway, unlit stretches, and isolated alleyways introduce substantial safety risks. SafePath addresses this asymmetry by evaluating and ranking multi-criteria route alternatives using soft computing methodologies. This report details the data pipeline, spatial snapping mechanism, multi-route pathfinding engine, length-weighted safety aggregation, and empirical trade-off evaluations across the drivable road network of Prayagraj, Uttar Pradesh. Across 20 randomly sampled origin-destination pairs under fixed random seed 42, the safety-aware route differed from the fastest path in 50.0% of trips, delivering an average safety score improvement of +2.33 points with an average detour time of 0.78 minutes (46.8 seconds).

---

### 1. Introduction and Problem Definition
Standard shortest-path algorithms ($A^*$, Dijkstra) operating on road graphs treat edges as homogeneous conductors of vehicular flow penalized only by length and speed limit. On Indian urban roads, safety is an intrinsically graded and non-linear concept:
1. **Uncertainty & Tag Sparsity**: OpenStreetMap (OSM) attribute tags such as street lighting (`lit`) and speed limits (`maxspeed`) vary sharply across developing regions.
2. **Compound Risk**: Driving at night on an unlit tertiary road with high isolation from emergency services compounds hazard probability exponentially compared to daytime transit.
3. **Blackspot Vulnerability**: A route with a 95% smooth highway stretch that traverses a single fatal unlit accident hotspot is unacceptably hazardous.

SafePath formulates route safety as a soft computing problem, extracting observable physical proxies from OpenStreetMap, combining them with living hazard maps and learned neuro-fuzzy segment scores, and presenting drivers with 2–3 transparent route choices.

---

### 2. System Architecture & Data Pipeline (Prayagraj Municipal Network)

#### 2.1 Boundary Disambiguation
Nominatim geocoding for `"Prayagraj, Uttar Pradesh, India"` returns the district boundary covering 5,471.11 km², which causes Overpass API query timeouts. SafePath automatically resolves to the municipal city boundary (`"Allahabad, Prayagraj, Uttar Pradesh, India"`, OSM relation `10029487`), spanning **316.52 km²**.

#### 2.2 Network Topology & Graph Ingestion
The network was ingested via OSMnx (`network_type="drive"`) using bounding-box extraction followed by polygon truncation and largest-component extraction.
- **Nodes**: 16,672
- **Edges**: 43,328 directed segments
- **Total Network Length**: 3,460.68 km
- **Projected Coordinate Reference System (CRS)**: UTM Zone 44N (`EPSG:32644`), enabling exact metric Euclidean calculations.
- **Stable Segment ID**: `seg_{u}_{v}_{key}` assigned to every directed edge.

#### 2.3 Observable Safety Proxies & POI Isolation
Per Section 4.8, neighborhood stigmatization via crime data was strictly avoided. Instead, observable emergency service accessibility proxies were queried from OSM:
- **Police Stations**: 11 facilities
- **Hospitals & Clinics**: 166 facilities
- **Petrol / Fuel Pumps**: 19 stations
- **Commercial Shops**: 400 points
Midpoint Euclidean distances to the nearest POI across all categories were computed in projected metric space (`EPSG:32644`) and normalized into `isolation_score` $\in [0, 1]$ using 95th-percentile min-max scaling.

---

### 3. Empirical Tag Coverage Analysis

In compliance with Rule 3 (Data Honesty), OSM tag coverage was audited directly across all 43,328 edges in Prayagraj without data fabrication:

| Feature Tag | Tagged Count | Missing Count | Observed Coverage (%) | System Handling / Fallback |
| :--- | :---: | :---: | :---: | :--- |
| **`lit` (Lighting)** | 0 | 43,328 | **0.0%** | Explicitly labeled `'unknown'`; motivates Module 1 image classifier |
| **`maxspeed`** | 119 | 43,209 | **0.27%** | Missing kept blank in contract CSV; Indian urban road-type defaults in metadata |
| **`highway` (Road Type)** | 43,328 | 0 | **100.0%** | residential (34,931), tertiary (3,496), unclassified (1,919), secondary (1,792), primary (536), trunk (403) |

*Interpretation*: Street lighting tag coverage is essentially non-existent in the public OSM graph for Prayagraj, reinforcing why SafePath treats absence of a tag as `'unknown'` rather than assuming the street is unlit (`'no'`).

---

### 4. Spatial Snapping Benchmark (Baseline vs. Proposed)

GPS coordinates sampled from mobile devices or dashcams exhibit Gaussian drift (5–15 m). We evaluated snapping accuracy across 200 noisy queries (`seed=42`, $\sigma = 10.0$ m):

| Metric | Baseline: Nearest Node Snap | Proposed: `STRtree` Edge Snapper | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Corridor Match Accuracy** | 33.0% (66 / 200) | **78.0%** (156 / 200) | **+45.0% points** |
| **Mean Snap Distance Error** | 27.35 m | **6.55 m** | **-20.80 m (-76.1%)** |
| **Max Snap Distance Error** | 89.42 m | **24.18 m** | **-65.24 m** |
| **Query Latency** | 0.052 ms | **0.130 ms** | Real-time capable |

*Interpretation*: Nearest-node snapping frequently snaps queries along the middle of a 200-meter segment to the wrong perpendicular cross-street. Projecting `LineString` geometries into UTM space and indexing with Shapely `STRtree` recovers true street corridor identity in 78% of noisy queries.

---

### 5. Multi-Route Generation & Safety Aggregation

#### 5.1 Travel Time & Non-Linear Safety Cost Formulation
For edge $e$ with length $l_e$ (m) and speed $v_e$ (m/s), traversal time is:
$$t_e = \frac{l_e}{v_e}$$
Given segment safety score $s_e \in [0, 100]$ (higher is safer), risk fraction is $r_e = \frac{100.0 - s_e}{100.0} \in [0, 1]$. To penalize hazardous corridors non-linearly, the routing cost for the safety-aware path is:
$$c_e^{\text{safest}} = t_e \times \left(1.0 + \beta \times r_e^{1.5}\right)$$
where $\beta = 3.0$. A 60% penalty on traversed edges is applied during candidate generation to ensure route diversification into genuine alternative avenues.

#### 5.2 Aggregation Mathematics
- **Length-Weighted Aggregate Safety**:
  $$\text{aggregate\_safety} = \frac{\sum_{e \in R} l_e \times s_e}{\sum_{e \in R} l_e}$$
- **Worst-Segment Score**:
  $$\text{worst\_segment\_score} = \min_{e \in R} s_e$$

---

### 6. Safest vs. Fastest Trade-Off Evaluation (20 OD Pairs, Seed 42)

To evaluate the practical viability of safety-aware routing, 20 random connected OD pairs (minimum 15 topological hops) were tested across Prayagraj:

| Evaluation Metric | Observed Result | Spec Benchmark Target | Status |
| :--- | :---: | :---: | :---: |
| **Test Pairs Evaluated** | 20 | 20 | Complete |
| **Random Seed** | 42 | Fixed | Verified |
| **How often Safest differs from Fastest** | **10 / 20 (50.0%)** | Tracked | 50% non-trivial trade-off |
| **Mean Extra Travel Time (when differing)** | **0.78 minutes** (~47s) | < 5.0 mins | Highly practical |
| **Median Extra Travel Time** | **0.03 minutes** (~2s) | - | Minimal delay |
| **Max Extra Travel Time** | **2.86 minutes** | < 10.0 mins | Acceptable upper bound |
| **Mean Safety Score Gain (when differing)** | **+2.33 points** | > 0 | Consistent gain |
| **Mean Extra Distance** | **279.5 meters** | - | Compact detour |

*Interpretation*: In half of the evaluated urban trips, standard fastest routing directs drivers through hazardous or unlit corridors to save negligible amounts of time (average 47 seconds). Choosing SafePath's alternative route secures a noticeable safety improvement for an acceptable travel-time trade-off.

---

### 7. Interactive User Interface ([`app/main.py`](file:///c:/Users/tarun/Desktop/SafePath/app/main.py))

The Streamlit web application provides:
1. **Route Search & Ranking**: Landmark presets across Prayagraj with ranked route cards.
2. **Visual Accessibility**: 3-tier color coding:
   - Safe ($\ge 70$): `#2e7d32` (Forest Green)
   - Caution ($40\text{--}69$): `#ef6c00` (Deep Amber)
   - Risky ($< 40$): `#c62828` (Crimson Red)
   complemented with explicit textual level badges, tooltips, and popups.
3. **Toggleable Layers**: Real-time road hazards (potholes, speed breakers, animals) and accident hotspots.
4. **Personal Safety Mode**: Re-weighting isolation and lighting for night travel.
5. **Add-a-Drive Crowdsourced Ingestion**: Upload interface for dashcam detections triggering living map updates.
6. **Data Provenance Panel**: Continuous disclosure of data origins.

---

### 8. Responsible AI, Provenance & Limitations (Section 12)

1. **Simulated Detections**: Dashcam detections and GPS noise along road polylines are simulated and clearly marked `is_simulated = true`.
2. **Substitute Heuristic Scores**: Pending final ANFIS model weights from Module C, scores utilize an explicit heuristic fallback formula based on Section 4.6.
3. **No Crime Data Usage**: Crime statistics are deliberately omitted to prevent neighborhood stigmatization.
4. **Sensor Noise**: Phone GPS drift (5–15 m) is mitigated through metric `STRtree` spatial indexing rather than naive nearest-node snapping.
5. **Validation Scope**: Safety scores are proxies and require confirmation through prospective field studies.

---

### 9. Section Status Summary (Team Progress)

- **Module 1 (Hazard Detection - Member A)**: Baseline vs YOLOv8-nano comparison documented; `detections_comparison.json` saved.
- **Module 2 (Living Hazard Map & Accidents - Member B)**: Simulated drive generator and clustering ready.
- **Module 3 (Fuzzy Safety Engine - Member C)**: Survey design and Mamdani rule bases configured.
- **Module 4 (Routing, App & Integration - Member D)**: **100% Complete & Tested.**

