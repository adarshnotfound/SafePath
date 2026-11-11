# Member D Viva Defense Guide & Examination Preparation
## Module: Integration, OSM Road Network, Routing & Streamlit Web App (`routing/`, `app/`)

**Role**: Team Member D (Integration, Routing, Web App & Evaluation Lead)  
**Target City**: Prayagraj, Uttar Pradesh, India  
**Project**: SafePath (`RaahSuraksha`)

---

## 1. The 60-Second Viva Pitch

> *"Standard navigation engines like Google Maps optimize exclusively for travel time and distance. In Indian cities, this frequently directs drivers onto narrow, poorly lit residential alleyways with unmapped speed breakers, stray animals, potholes, and severe isolation from emergency services. SafePath reformulates navigation as a multi-criteria soft computing problem. For Prayagraj, I extracted all 43,328 drivable road segments and 596 safety POIs from OpenStreetMap, resolved them into a metric UTM coordinate space, and engineered an edge-level safety cost function. SafePath generates 2 to 3 diverse alternatives—Fastest, Safest, and Balanced—aggregates safety using length-weighted averaging so long dangerous stretches dominate, and synthesizes plain-language explanations. Across 20 random journeys under seed 42, SafePath found a safer alternative in 50% of trips, providing a +2.33 point safety improvement for an average detour of only 47 seconds."*

---

## 2. Algorithm Walkthrough in 4 Simple Steps

```
[Start/End Lat,Lon] 
       │
       ▼
1. Metric Snapping (STRtree in EPSG:32644) ──► Recovers nearest graph edges (6.55m error vs 27.35m node snap)
       │
       ▼
2. Travel Time & Safety Cost Computation  ──► c_e = travel_time * (1 + 3.0 * ((100 - score)/100)^1.5)
       │
       ▼
3. Multi-Route Search (Dijkstra + Penalty) ──► Computes Fastest, Safest, and 60% Penalty Diversified Balanced Path
       │
       ▼
4. Aggregation & Natural Explanation      ──► Length-weighted mean + worst segment minimum + factor comparison
       │
       ▼
[Interactive Folium UI & routes.json]
```

### Step 1: Metric CRS Snapping
A query `(lat, lon)` is projected from `EPSG:4326` (degrees) into UTM Zone 44N `EPSG:32644` (metres). An `STRtree` spatial index over edge `LineString` geometries finds the nearest road centerline and snaps to the nearest vertex in $0.13$ milliseconds.

### Step 2: Edge Cost Formulation
Each edge has length $l_e$ (m) and speed $v_e$ (m/s), giving traversal time $t_e = \frac{l_e}{v_e}$. The safety-penalized cost is:
$$c_e^{\text{safest}} = t_e \times \left(1.0 + \beta \times \left(\frac{100.0 - s_e}{100.0}\right)^{1.5}\right)$$
where $s_e \in [0, 100]$ is the segment safety score and $\beta = 3.0$.

### Step 3: Multi-Route Diversification
- **Route 1 (Fastest)**: Standard Dijkstra minimizing $\sum t_e$.
- **Route 2 (Safest)**: Dijkstra minimizing $\sum c_e^{\text{safest}}$.
- **Route 3 (Balanced)**: Penalizes edges used in Route 1 by 1.6× ($60\%$ cost increase), forcing the router to explore an alternative arterial corridor.

### Step 4: Length-Weighted Aggregation & Worst Segment
- $\text{aggregate\_safety} = \frac{\sum l_e \times s_e}{\sum l_e}$
- $\text{worst\_segment\_score} = \min_{e} s_e$
- Explanations are synthesized by comparing travel times and inspecting the `top_factors` of the lowest-scoring segments.

---

## 3. Parameter Choices & Design Justifications

| Parameter / Decision | Value | Why this was chosen |
| :--- | :---: | :--- |
| **Projected CRS** | `EPSG:32644` (UTM 44N) | At 25.4°N (Prayagraj), $1^\circ$ lon $\approx 100$ km while $1^\circ$ lat $\approx 111$ km. Degree distances distort buffers by $>10\%$. UTM ensures isotropic distances in metres. |
| **Risk Exponent** | $1.5$ (Non-linear) | A linear penalty treats a drop from 90 to 80 the same as 30 to 20. The $1.5$ exponent heavily penalizes segments entering the critical danger zone ($<40$). |
| **Risk Multiplier ($\beta$)** | $3.0$ | Balances safety with detour realism. $\beta < 1.5$ produces identical routes to fastest; $\beta > 6.0$ causes absurd detours across ring roads for a minor alley pothole. |
| **Corridor Penalty** | $1.6$ (60% penalty) | Standard $k$-shortest paths (Yen's) produces trivial 1-block bypasses. Edge-penalization forces the router to discover distinct major corridors. |
| **Isolation Percentile Clip** | 95th Percentile | Prevents extreme rural outliers on the municipal boundary from compressing all central urban isolation scores to 0. |

---

## 4. Empirical Results & What They Mean

### A. OSM Tag Coverage in Prayagraj
- **Lit Tag (`lit`)**: $0 / 43,328$ ($0.0\%$). All explicitly set to `'unknown'`.
- **Speed Limit (`maxspeed`)**: $119 / 43,328$ ($0.27\%$). Missing kept blank in contract; conservative urban defaults used in metadata.
- *Viva Meaning*: Proves tag sparsity in developing countries; justifies why Module 1 built an image-based lighting estimator and why Module 3 handles uncertainty explicitly.

### B. Snapping Benchmark (200 GPS queries, 10 m noise)
- **Baseline (Nearest Intersection Node)**: $33.0\%$ corridor accuracy, $27.35$ m mean error.
- **Proposed (`STRtree` Edge Snapper)**: $78.0\%$ corridor accuracy, $6.55$ m mean error ($-76.1\%$ error reduction).
- *Viva Meaning*: Nearest-node snapping frequently selects the perpendicular cross-street when GPS drift occurs along long segments. Edge projection recovers the true street.

### C. Safest vs. Fastest Trade-Off (20 OD Pairs, Seed 42)
- **How often Safest differs from Fastest**: **10 / 20 pairs (50.0%)**.
- **Mean Extra Travel Time when differing**: **0.78 minutes (46.8 seconds)**.
- **Max Extra Travel Time**: **2.86 minutes**.
- **Mean Safety Score Gain**: **+2.33 points**.
- *Viva Meaning*: Navigation apps sacrifice significant safety to save mere seconds. SafePath delivers a safer journey for under a 1-minute detour.

---

## 5. 10 Hard Faculty Viva Questions with Model Answers

### Q1: Why did you use length-weighted averaging rather than a simple arithmetic mean?
> **Answer**: "An arithmetic mean weights every graph segment equally regardless of length. If a 10 km route has nine 50-meter safe segments ($s=90$) and one 9.55 km dangerous unlit segment ($s=20$), a simple mean gives $\frac{9 \times 90 + 20}{10} = 83.0$ (Safe), which is dangerously misleading. Length-weighting weights by exposure distance, giving an aggregate score of $\approx 23.3$ (Risky), correctly reflecting the driver's true risk exposure."

### Q2: Why report `worst_segment_score` alongside aggregate safety?
> **Answer**: "Because safety risks in road navigation are non-compensatory. An otherwise great arterial route with a high average score of 82/100 is unacceptable if it forces the vehicle through a single unlit, high-isolation accident blackspot ($s=18$). Reporting the worst segment gives drivers the complete risk profile and guides the explanation synthesizer."

### Q3: Why not use $A^*$ with a Euclidean distance heuristic for the safety-aware route?
> **Answer**: "For $A^*$ to guarantee optimality, the heuristic function $h(n)$ must be *admissible* (it must never overestimate the true remaining cost to the goal). For pure travel time or distance, Euclidean distance divided by max speed is admissible. However, safety costs depend on dynamic segment scores that can drop effective costs lower than distance-derived bounds relative to arbitrary penalty weights. Dijkstra operates without heuristic admissibility constraints, guarantees exact global optimality on non-negative edge costs, and runs in under 3 milliseconds on our 16,672-node graph."

### Q4: OpenStreetMap has 0% lighting data in Prayagraj. Did you fabricate it?
> **Answer**: "No, absolutely not. Per Rule 3 of our specification, we never fabricate data. In `segment_features.csv`, all 43,328 edges have `lit` explicitly set to `'unknown'`. This empirical finding directly motivated Module 1's computer vision classifier to estimate lighting from dashcam video frames, and prompted Module 3 to create dedicated fuzzy rules for uncertain lighting."

### Q5: Why did you exclude crime data from your isolation score?
> **Answer**: "Section 4.8 of our specification strictly forbids crime-based neighborhood labeling. In India, FIR crime statistics are only available at coarse district police-station levels, and using them to score street segments stigmatizes marginalized neighborhoods. Instead, we use observable, objective physical proxies: metric distance to emergency facilities (police stations, hospitals) and commercial hubs (shops, petrol pumps)."

### Q6: How does Personal Safety Mode work?
> **Answer**: "Personal Safety Mode is designed for night travel. It increases the penalty weights on street isolation and unlit/unknown segments by $1.8\times$. This forces Dijkstra to route vehicles along well-lit primary commercial avenues and near open facilities, avoiding deserted residential shortcuts even if they are slightly shorter."

### Q7: What is the time complexity of your routing engine?
> **Answer**: "Graph construction with edge weights is $O(E)$, precomputed once. For each route query, node snapping with `STRtree` (2D R-tree) takes $O(\log E)$. Dijkstra's algorithm with a binary min-heap runs in $O((V + E) \log V)$. On Prayagraj ($V = 16,672$, $E = 43,328$), an entire 3-alternative query finishes in approximately $2.6$ seconds, of which Dijkstra takes under $15$ ms."

### Q8: What could go wrong in practice? What are the failure modes?
> **Answer**: "Three primary failure modes:
> 1. *Topological Disconnections*: Graph components separated by unmapped barriers. We solved this by extracting the largest strongly connected component during OSMnx ingestion.
> 2. *Extreme Detours*: If safety penalties are too high, the router might detour 20 km around a ring road to avoid a 50-meter pothole. We calibrated $\beta = 3.0$ with a non-linear $1.5$ exponent to bound max detour under 3 minutes.
> 3. *GPS Drift*: Naive snapping to nearest intersection nodes fails on long links. Our metric projected `STRtree` snapper reduced mean snap error from 27.35 m to 6.55 m."

### Q9: How does the living hazard map update affect your routing module?
> **Answer**: "Module 2 maintains `hazards.csv` with confidence scores and decay tracking. When new drives are uploaded via the 'Add-a-Drive' page in our Streamlit app, newly confirmed hazards increase the hazard density on intersecting segments, which automatically lowers their safety score and prompts the router to bypass that road on subsequent queries."

### Q10: How can you prove the whole pipeline works right now?
> **Answer**: "By running `python run_demo.py` in the terminal. It verifies all 10 contract files, initializes the Prayagraj network, computes a live route between Civil Lines and Sangam, validates Section 8 JSON compliance, and summarizes our 20-pair trade-off benchmark in under 15 seconds."

---

## 6. Live Demo Quick-Reference Commands

```powershell
# 1. Run the End-to-End Integration Demo
.\.venv\Scripts\python.exe run_demo.py

# 2. Launch the Interactive Streamlit Web UI
.\.venv\Scripts\python.exe -m streamlit run app/main.py

# 3. Run All Test Suites
.\.venv\Scripts\pytest.exe -v tests app/test_app.py routing/test_routing.py routing/test_routes.py

# 4. Run Code Cleanliness Checks
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\flake8.exe app routing tests detection run_demo.py --count --select=E9,F63,F7,F82 --show-source --statistics
```
