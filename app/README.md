# Module 4 (D4): SafePath Streamlit Web Application (`app/`)

**Owner**: Member D (Integration, routing, app and report lead)

## Purpose
Provides the interactive web interface for **SafePath (RaahSuraksha)**. Implements Section 4.10 and Task D4 of the project specification, enabling users to find and compare alternative routes, inspect color-coded road risk segments on an interactive Folium map, toggle hazard and accident hotspot layers, switch to Personal Safety Mode, upload new dashcam drives, and view empirical evaluation benchmarks.

---

## Architecture & Files

| File | Purpose |
| :--- | :--- |
| `main.py` | Primary Streamlit entrypoint. Contains 3 top-level tabs (Route Search, Add-a-Drive, Evaluation) and sidebar controls. |
| `config.py` | Single configuration path managing switching between **Mock Data Mode** (`data/sample/`) and **Real Pipeline Mode** (`data/processed/`), landmark coordinates, and explicit Data Provenance labels (Section 12). |
| `map_utils.py` | Geospatial visualization module rendering interactive Folium maps with accessible 3-tier color coding, LineString segment geometry caching, hazard markers, and accident hotspots. |
| `hello.py` | Readiness probe confirming dataset availability and UI status. |
| `test_app.py` | Automated unit tests covering configuration modes, landmark coordinates, accessible color thresholds, and Folium map rendering. |

---

## Features & Specification Adherence

### 1. Route Search & Ranked Alternatives (Page 1)
- **Landmark & Coordinate Inputs**: Origin (Point A) and Destination (Point B) selectable via preset Prayagraj landmarks (*Prayagraj Junction*, *Civil Lines*, *Sangam*, *High Court*, *University of Allahabad*) or custom coordinates.
- **Ranked Alternatives**: Displays 2–3 alternatives (Safest, Fastest, Balanced) with Route ID badges, travel time in minutes, total distance in km, length-weighted aggregate safety score, worst segment score, and plain-language comparative explanations.
- **Accessible Color Coding**:
  - **Safe** ($\ge 70$): `#2e7d32` (Forest Green)
  - **Caution** ($40\text{--}69$): `#ef6c00` (Deep Amber)
  - **Risky** ($< 40$): `#c62828` (Crimson Red)
  - Augmented with text badges, tooltips, and popups so risk is never communicated by color alone.

### 2. Toggleable Map Layers
- **Road Hazards Layer**: Visualizes active hazards from `hazards.csv` (potholes, speed breakers, animals) with detection confidence, observation counts, and `last_seen` timestamps.
- **Accident Hotspots Layer**: Renders clustered accident blackspots with severity radius.

### 3. Personal Safety Mode
- Sidebar switch that increases the weight of lighting, street isolation, and proximity to police stations and open shops, especially for night-time travel.

### 4. "Add-a-Drive" Ingestion Page (Page 2)
- Implements Section 4.9. Allows uploading a new `detections.csv` file from a dashcam drive, previews detections matching Section 8 contract columns, and triggers hazard clustering and confidence update/decay.

### 5. Evaluation & Trade-Off Dashboard (Page 3)
- Directly displays the empirical metrics from `evaluation/results/tradeoff_fastest_vs_safest.json` (20 OD-pairs trade-off) and `evaluation/results/routing_osm_features.json` (tag coverage and snapping accuracy).

### 6. Data Provenance Note
- Visible on the sidebar across all views, clearly classifying data as `real` (OSM network, POIs), `simulated` (GPS noise, dashcam drive timeline), or `substitute` (baseline safety scores) per Rule 3 and Section 12.

---

## How to Run

```bash
# Start the Streamlit application
streamlit run app/main.py

# Run application unit tests
pytest -v app/test_app.py
```

