# Module 4 (D2): OSM Road Network & Segment Features (`routing/`)

**Owner**: Member D (Integration, routing, app and report lead)

## Purpose
Downloads the drivable OpenStreetMap road network and observable safety-proxy POIs for **Prayagraj, Uttar Pradesh, India**, assigns deterministic segment identifiers to every directed edge, builds `data/processed/segment_features.csv` matching Section 8, reports honest OSM tag coverage (`lit`, `maxspeed`, `highway`), and provides metric-CRS `(lat, lon)` edge snapping.

---

## Files

| File | Responsibility |
| :--- | :--- |
| `osm_network.py` | Downloads and caches the OSMnx drivable graph (`network_type="drive"`) and safety POIs (`police`, `hospital`/`clinic`, `fuel`, `shop`) in `data/raw/`. Assigns stable `segment_id = seg_{u}_{v}_{key}`. |
| `build_segment_features.py` | Normalizes `highway`, `lit` (`yes`/`no`/`unknown`), and `maxspeed`; computes projected metric `isolation_score`; exports Section 8 `segment_features.csv` and sidecar `segment_metadata.csv` to `data/processed/`. |
| `snap.py` | `SegmentSnapper` and `snap_to_segment(lat, lon, target)` using projected UTM coordinates (`EPSG:32644`) and a Shapely `STRtree` spatial index, plus `NearestNodeSnapBaseline` for benchmarking. |
| `evaluate_features.py` | Runs the end-to-end Prayagraj pipeline, prints OSM tag coverage, benchmarks baseline nearest-node snapping vs proposed `STRtree` edge snapping (`seed=42`), and saves metrics to `evaluation/results/routing_osm_features.json`. |
| `router.py` | Core multi-route navigation engine. Computes 2-3 alternative routes (Safest, Fastest, Balanced) via Dijkstra and corridor diversification; ingests segment safety scores; calculates length-weighted aggregate safety and worst-segment score; synthesizes plain-language comparative explanations; exports Section 8 `routes.json`. |
| `evaluate_tradeoff.py` | Evaluates fastest vs safest route trade-off across 20 random connected OD pairs (`seed=42`). Reports differ rate, extra travel time (mins), safety score gain, and exports `evaluation/results/tradeoff_fastest_vs_safest.json` and `data/processed/routes.json`. |
| `test_routing.py` | Offline unit tests on a synthetic Prayagraj-coordinate network verifying contracts, metric distances, `isolation_score`, and snapping. |
| `test_routes.py` | Unit tests verifying length-weighted safety math against hand calculations, `routes.json` schema compliance, multi-route differentiation, and fallback score ingestion. |

---

## Design Decisions & Data Honesty

1. **Stable `segment_id` (`seg_{u}_{v}_{key}`)**: Uses OSMnx MultiDiGraph endpoint node IDs `(u, v)` and parallel-edge `key`. Unlike `osmid` (which can cover multiple segments across intersections), `(u, v, key)` uniquely identifies each directed edge and maps directly back to the NetworkX graph for routing.
2. **Missing `speed_limit` Handling**: In India, OSM `maxspeed` coverage is often low. By default (`impute_missing_speed=False`), `segment_features.csv` keeps missing OSM `maxspeed` entries blank (`""`) so raw OSM data is never confused with imputed defaults. Sidecar `data/processed/segment_metadata.csv` records `speed_limit_source` (`"osm"` vs `"missing"`) alongside `speed_limit_fallback_kmh` (conservative Indian urban road-type defaults) for downstream travel-time estimation (`D3`) and fuzzy scoring (`C`), or `--impute-speed-limit` can be passed explicitly.
3. **Metric CRS for Isolation & Snapping**: Prayagraj sits in UTM Zone 44N (`EPSG:32644`). All POI distances (`isolation_score`) and point-to-edge snapping (`distance_m`) are computed in projected metres rather than WGS84 degrees.
4. **Isolation Score Normalization**: For each segment midpoint, Euclidean distance (in metres) is computed to the nearest POI in each of the 4 categories (`police`, `hospital`, `petrol_pump`, `shop`). The per-segment mean across available categories is min-max normalized using the 95th percentile as the upper clip so extreme city-edge segments do not compress urban scores toward zero.
5. **Prayagraj City Boundary & Observed OSM Tag Coverage**: In OSM Nominatim, `"Prayagraj, Uttar Pradesh, India"` matches the 5,471.11 km² rural district, whereas the 316.52 km² municipal city boundary (relation `10029487`) is indexed under `"Allahabad, Prayagraj, Uttar Pradesh, India"`. Across the 43,328 municipal road edges (`16,672` nodes, `3,460.682` km), `lit` tag coverage inside the municipal boundary is `0 / 43,328` (`0.0%`, all labeled `'unknown'`) and `maxspeed` coverage is `119 / 43,328` (`0.27%`), reinforcing the need for Module 1's image-based lighting estimate and explicit uncertainty handling in Module 3.

---

## How to Run

```bash
# 1. Build Prayagraj segment_features.csv, print tag coverage, and run snapping benchmark
python -m routing.evaluate_features

# 2. Run 20-pair routing trade-off experiment (fastest vs safest) and export routes.json
python -m routing.evaluate_tradeoff

# 3. Run all offline unit tests for routing
pytest -v routing/test_routing.py routing/test_routes.py
```

---

## Task D3: Multi-Route Generation & Trade-Off Evaluation

### 1. Mathematical Formulas
- **Traversal Time ($t_e$)**:
  $$t_e = \frac{\text{length}_e}{\text{speed}_e}$$
  where $\text{speed}_e$ uses OSM `maxspeed` or road-type fallback speeds.
- **Safety Cost ($c_e^{\text{safest}}$)**:
  $$c_e^{\text{safest}} = t_e \times \left(1.0 + \beta \times \left(\frac{100.0 - s_e}{100.0}\right)^{1.5}\right)$$
  where $s_e \in [0, 100]$ is the segment safety score and $\beta = 3.0$.
- **Length-Weighted Aggregate Safety**:
  $$\text{aggregate\_safety} = \frac{\sum_{e \in R} \text{length}_e \times s_e}{\sum_{e \in R} \text{length}_e}$$
- **Worst Segment Score**:
  $$\text{worst\_segment\_score} = \min_{e \in R} s_e$$

### 2. Trade-Off Findings (20 Random OD Pairs, Seed 42)
- **Baseline**: Fastest route (shortest travel time).
- **Proposed**: Safety-aware route (safety-penalized cost).
- **How often safest differs from fastest**: **10 / 20 pairs (50.0%)**
- **Mean extra travel time when differing**: **0.78 minutes** (~47 seconds)
- **Max extra travel time**: **2.86 minutes**
- **Mean safety score gain**: **+2.33 points**
- **Exported Artifacts**:
  - `evaluation/results/tradeoff_fastest_vs_safest.json` (Full reproducible metrics)
  - `data/processed/routes.json` (Representative 3-route alternative set matching Section 8 contract)

---

## Viva Talking Points
- **Why project to UTM before computing distances?** One degree of latitude is ~111 km, while one degree of longitude at 25.4° N is ~100 km; Euclidean distance in degrees distorts geometry and produces wrong metre thresholds.
- **Why compare against nearest-node snapping?** Snapping GPS points only to intersection nodes fails near mid-points of long segments; `STRtree` over `LineString` edge geometries recovers the road corridor accurately under 5–15 m simulated GPS noise.
- **Why explicit `'unknown'` for `lit`?** Absence of an OSM `lit` tag does not mean a street is unlit (`lit=no`); keeping `'unknown'` separate lets the fuzzy safety engine model uncertainty explicitly.
- **Why use length-weighted safety aggregation instead of simple mean?** A route with nine 10-meter safe segments and one 1-kilometer dangerous unlit segment would look safe under a simple average; length-weighting ensures the long dangerous segment dominates the overall journey risk.
- **Why report `worst_segment_score` alongside aggregate safety?** A route can have an acceptable average safety score (e.g. 75/100) while traversing a single fatal unlit accident blackspot; reporting the worst segment gives drivers the complete risk profile.

