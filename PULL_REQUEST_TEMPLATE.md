## Description
<!-- Provide a brief description of the changes introduced in this PR -->

### Module & Ownership
- [ ] Module 1: Detection (`/detection`) - Owner A
- [ ] Module 2: Hazard Map & Accidents (`/hazard_map`, `/accidents`) - Owner B
- [ ] Module 3: Fuzzy Engine & Survey (`/fuzzy`, `/survey`) - Owner C
- [ ] Module 4: Routing & App (`/routing`, `/app`) - Owner D
- [ ] Evaluation / Integration / Docs (`/evaluation`, `/docs`, `/data`)

---

## File Contracts & Data Honesty (Section 8 & 12 Checklist)
- [ ] Conforms strictly to Section 8 column headers and schemas.
- [ ] No unlabelled synthetic data: all simulated or substitute (non-Indian) datasets are explicitly labelled (`is_simulated` / data provenance noted).
- [ ] No crime-based scoring or stigmatising proxies used.
- [ ] If this PR adds an experiment, it includes a baseline, fixed random seed, and writes metrics to `/evaluation/results/`.

---

## Testing & Quality
- [ ] Linting passed (`ruff check .` or `flake8`)
- [ ] Tests passed (`pytest`)
- [ ] Code is documented, readable, and ready for viva explanation.

