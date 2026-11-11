# SafePath - Fuzzy Logic & Soft Computing Engine (`/fuzzy`)

**Module Owner:** Member C (Fuzzy Engine & Validation Lead)  
**Tech Stack:** Python 3.10+, NumPy, SciPy, scikit-learn, DEAP  

---

## 1. Overview & Problem Solved

Standard GPS navigation systems optimize strictly for speed or distance. However, on Indian urban and suburban road networks, travelers face unlit stretches, acute pothole clusters, animals on roads, and hazardous blackspots.

Road safety is inherently **vague and multi-factorial**; a street is not simply "100% safe" or "0% safe". It depends on complex interactions:
- Potholes become far more dangerous in unlit night conditions than in clear daylight.
- Isolated segments amplify risk because emergency support is distant.

This module implements a **Soft Computing Fuzzy Safety Engine**:
1. **Weighted-Sum Baseline (Baseline A):** Linear combination of normalized segment indicators.
2. **Hand-Written Mamdani Fuzzy System (Baseline B):** 20 expert rules using min-max composition and centroid defuzzification.
3. **Learned Soft Computing Engine (Proposed Method):** Genetic Algorithm (DEAP) tuned rule weights and membership functions calibrated against validation survey ratings.

---

## 2. Directory Structure

```
fuzzy/
├── __init__.py
├── config.py             # Linguistic terms, membership parameters, 20 Mamdani rules & weights
├── membership.py         # Triangular (trimf) and Trapezoidal (trapmf) membership logic
├── baseline_weighted.py  # Linear weighted-sum baseline engine (Baseline A)
├── mamdani_engine.py     # Mamdani inference engine, explainability, defuzzification (Baseline B)
├── ga_tuner.py           # Genetic Algorithm tuning of rule weights & MF bounds (Proposed Method)
├── evaluate.py           # 5-fold cross-validation experiment runner (saves to /evaluation/results/)
├── ablation.py           # Factor removal ablation study & personal safety mode sensitivity
├── score_segments.py     # CLI tool to score any segment_features.csv -> segment_scores.csv
├── test_fuzzy.py         # Unit tests including 5 hand-checked validation test cases
└── README.md             # Module documentation and viva defense guide
```

---

## 3. How the Algorithms Work

### Fuzzification
Continuous sensor proxies and OSM tags are mapped into fuzzy sets $\mu \in [0.0, 1.0]$:
- `hazard_density`: low, medium, high (trapezoidal / triangular)
- `accident_score`: low, medium, high
- `lighting`: dark ($0.10$), dim ($0.50$), well_lit ($0.95$)
- `time_of_day`: night ($0.10$), dusk ($0.50$), day ($0.90$)
- `isolation_score`: low, medium, high
- `speed_limit`: slow, moderate, fast

### Mamdani Inference & Defuzzification
For each rule $k$:
$$\alpha_k = w_k \cdot \min_{j} \mu_{A_j}(x_j)$$
Aggregation across active rules uses maximum combination:
$$\mu_{\text{agg}}(y) = \max_k \min(\alpha_k, \mu_{C_k}(y))$$
Centroid defuzzification over $y \in [0, 100]$:
$$y^* = \frac{\sum y \cdot \mu_{\text{agg}}(y)}{\sum \mu_{\text{agg}}(y)}$$

### Output Contract (`segment_scores.csv`)
Columns:
`segment_id, safety_score, level, top_factors, fired_rules, model_name`
- **`level`**: `safe` ($\ge 70$), `caution` ($40 \le \text{score} < 70$), `risky` ($< 40$).

---

## 4. Reproducing Evaluation & Experiments

All experiments use fixed random seeds (**seed 42**).

1. **Run 5-Fold Cross-Validation Benchmark:**
   ```powershell
   python fuzzy/evaluate.py
   ```
   Outputs metrics to `evaluation/results/fuzzy_comparison.json`.

2. **Run Factor Ablation & Personal Safety Mode Test:**
   ```powershell
   python fuzzy/ablation.py
   ```
   Outputs metrics to `evaluation/results/fuzzy_ablation.json`.

3. **Score Segments from CLI:**
   ```powershell
   python fuzzy/score_segments.py --model mamdani --input data/sample/segment_features.csv --output data/sample/segment_scores.csv
   ```

4. **Run Unit Tests:**
   ```powershell
   python -m unittest fuzzy/test_fuzzy.py
   ```

---

## 5. Viva Defense Notes (Top Questions & Answers)

1. **Q: Why use Fuzzy Logic instead of a standard deep neural network?**
   *A:* Road safety requires verifiable explanations for navigational trust. Neural networks are black-boxes; fuzzy systems provide transparent IF-THEN rules where users and traffic engineers can inspect exactly which factors fired the alert.

2. **Q: How are membership function boundaries chosen and tuned?**
   *A:* Hand-crafted initial bounds were set using domain safety principles. We then used a Genetic Algorithm (GA) with tournament selection and blend crossover to tune the rule weights and high-risk boundaries against human perception ratings.

3. **Q: How do you prevent overfitting with small survey sample sizes?**
   *A:* We use $k$-fold cross-validation and severely constrain the chromosome to 20 rule weights and 4 macro scaling parameters, preventing the optimizer from memorizing individual ratings.

