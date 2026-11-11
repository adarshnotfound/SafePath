# SafePath Fuzzy Safety Engine: Architecture & Design Note

**Author:** Member C (Fuzzy Engine & Validation Lead)  
**Status:** Approved & Implemented  
**Date:** November 2026  

---

## 1. Executive Overview

Road safety on Indian urban and semi-urban road networks cannot be accurately captured by binary labels (*safe* vs *unsafe*). Road safety is intrinsically **graded, uncertain, and context-dependent**:
- A road with potholes is moderately risky in clear daylight at slow speeds, but becomes hazardous at night under poor lighting.
- High accident blackspots are risky at all times, but distant emergency amenities (isolation) exacerbate the vulnerability.
- Old hazard reports may have decayed or been repaired, whereas fresh reports have high predictive value.

The **SafePath Fuzzy Safety Engine** solves this using soft computing:
1. **Linguistic variables** transform crisp sensor and geospatial metrics into fuzzy membership values.
2. **Mamdani fuzzy inference** applies human-interpretable linguistic IF-THEN rules.
3. **Centroid defuzzification** outputs an intuitive, calibrated **Safety Score (0–100)** and a classification level (*safe*, *caution*, *risky*).
4. **Neuro-fuzzy / Genetic Algorithm tuning** optimizes membership function parameters and rule weights against human safety validation surveys.

---

## 2. Linguistic Variables & Membership Functions

### 2.1 Input Variables (Derived from `segment_features.csv`)

| Variable | Physical Meaning | Range | Fuzzy Sets & Shapes |
| :--- | :--- | :--- | :--- |
| **`hazard_density`** | Confidence-weighted hazards per 100m | $[0.0, 3.0+]$ | `low` (trapmf), `medium` (trimf), `high` (trapmf) |
| **`mean_hazard_confidence`** | Average detector confidence | $[0.0, 1.0]$ | `low` (trapmf), `medium` (trimf), `high` (trapmf) |
| **`mean_freshness`** | Temporal freshness factor ($1.0 = \text{fresh}$) | $[0.0, 1.0]$ | `stale` (trapmf), `aging` (trimf), `fresh` (trapmf) |
| **`accident_score`** | Blackspot density / severity proxy | $[0.0, 1.0]$ | `low` (trapmf), `medium` (trimf), `high` (trapmf) |
| **`lighting`** | Streetlight presence (`lit` tag / camera proxy) | $[0.0, 1.0]$ | `dark` (trapmf), `dim` (trimf), `well_lit` (trapmf) |
| **`time_of_day`** | Diurnal cycle (`night`=0, `dusk`=0.5, `day`=1.0) | $[0.0, 1.0]$ | `night` (trapmf), `dusk` (trimf), `day` (trapmf) |
| **`isolation_score`** | Distance to emergency amenities (police, clinics) | $[0.0, 1.0]$ | `low` (trapmf), `medium` (trimf), `high` (trapmf) |
| **`speed_limit`** | Road speed limit (km/h) | $[0, 100+]$ | `slow` (trapmf), `moderate` (trimf), `fast` (trapmf) |

### 2.2 Output Variable: `safety_score`

- **Domain:** $[0.0, 100.0]$
- **Fuzzy Sets:**
  - `risky`: Trapezoidal $[0, 0, 25, 42]$
  - `caution`: Triangular $[35, 52.5, 70]$
  - `safe`: Trapezoidal $[62, 80, 100, 100]$
- **Classification Rules:**
  - $\text{Safety Score} \ge 70.0 \implies \textbf{safe}$
  - $40.0 \le \text{Safety Score} < 70.0 \implies \textbf{caution}$
  - $\text{Safety Score} < 40.0 \implies \textbf{risky}$

---

## 3. Fuzzy Inference System (Mamdani Min-Max Centroid)

1. **Fuzzification:** Crisp numeric feature inputs $x_i$ are evaluated across triangular ($\text{trimf}$) and trapezoidal ($\text{trapmf}$) membership functions $\mu_A(x_i)$.
2. **Rule Evaluation (T-norm min):** For each rule $k$ with antecedents $A_1, A_2, \dots$:
   $$\alpha_k = w_k \cdot \min_{j} \mu_{A_j}(x_j)$$
   where $w_k \in [0.0, 1.0]$ is the rule weight.
3. **Aggregation (S-norm max):** The aggregated output fuzzy set $\mu_{out}(y)$ across all fired rules:
   $$\mu_{out}(y) = \max_{k} \min(\alpha_k, \mu_{C_k}(y))$$
4. **Defuzzification (Centroid):**
   $$y^* = \frac{\int y \cdot \mu_{out}(y)\,dy}{\int \mu_{out}(y)\,dy}$$

---

## 4. Explainability & Contributing Factors

For every scored road segment:
- Each active rule whose firing strength $\alpha_k > 0.05$ is recorded in `fired_rules`.
- Contributing risk/safety factors are ranked by their normalized activation and expressed in plain English (e.g., *"unlit street with high isolation at night"*, *"high density of newly reported road hazards"*).
- The output strictly satisfies the Section 8 contract for `segment_scores.csv`.

---

## 5. Personal Safety Mode

When enabled, Personal Safety Mode addresses solitary travelers (especially vulnerable road users at night):
- Re-weights feature sensitivity towards **lighting**, **isolation score**, and **time of day**.
- Increases caution and risky activation for unlit or isolated stretches even in the absence of physical potholes.
- Fully ablatable and compared against standard navigation metrics.

