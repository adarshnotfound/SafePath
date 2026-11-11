# SafePath - Validation Survey Module (`/survey`)

**Module Owner:** Member C (Fuzzy Engine & Validation Lead)  
**Tech Stack:** Python 3.10+, NumPy, SciPy  

---

## 1. Overview & Objectives

The Survey Module collects, cleans, and standardizes human perception safety ratings across diverse road segments. These ratings provide ground-truth targets to calibrate, validate, and compare SafePath soft computing models against baseline estimators.

---

## 2. Directory Structure

```
survey/
├── __init__.py
├── survey_design.md               # Survey blueprint, ethical consent, and objective rating criteria
├── generate_simulated_ratings.py  # Realistic simulated survey generator (fallback per Rule 3)
├── clean_survey.py                # Cleaning script enforcing Section 8 contracts & inter-rater agreement
├── test_survey.py                 # Unit tests for data cleaning and statistical metrics
└── README.md                      # Module guide
```

---

## 3. Workflow & Data Contract

### Input: Raw Survey / Google Form Export
Participants rate road segments on a 1–5 scale ($1 = \text{Very Risky}$, $5 = \text{Very Safe}$).

### Output Contract: `survey_ratings.csv`
`segment_id, rater_id, rating, time_of_day_context`
- `segment_id`: Unique identifier (e.g., `seg_101`)
- `rater_id`: Anonymous identifier (e.g., `rater_01`)
- `rating`: Integer between 1 and 5
- `time_of_day_context`: `day`, `dusk`, or `night`

---

## 4. Scripts & Usage

1. **Clean Raw Survey Data & Compute Inter-Rater Reliability:**
   ```powershell
   python survey/clean_survey.py
   ```
   Outputs cleaned dataset and saves reliability metrics (`icc_approx`, `mean_pairwise_abs_diff`, `percent_exact_agreement`) to `evaluation/results/survey_agreement.json`.

2. **Generate Simulated Survey Fallback (per Rule 3):**
   ```powershell
   python survey/generate_simulated_ratings.py
   ```

3. **Run Unit Tests:**
   ```powershell
   python -m unittest survey/test_survey.py
   ```

