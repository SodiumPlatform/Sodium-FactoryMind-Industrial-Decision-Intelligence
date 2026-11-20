# FactoryMind - Architecture

Team Sodium | International AI Builders Congress (hackathon MVP)

## Overview

FactoryMind is a single-process Streamlit application with a modular Python
backend. No external services, no LLM APIs, no secrets. After dataset and
model preparation the app runs fully offline.

```
                       +------------------------------------------+
 CSV upload / form --> | Machine Data Upload (pages/data_upload)  |
                       |   tolerant header matching               |
                       |   row-level validation                   |
                       +--------------------+---------------------+
                            validated df    |
                                            v
   data/ai4i2020.csv --> src/data_loader --> src/model.py
   (UCI AI4I, CC BY 4.0)   (fallback:          |  stratified 60/20/20 split
                            flagged            |  preprocessing pipeline
                            synthetic)         |  RandomForest(400, balanced)
                                               |  validation-informed thresholds
                                               v
                                   models/factorymind_model.joblib
                                            |
              +-----------------------------+----------------------------+
              v                            v                            v
   pages/dashboard       pages/predictions            pages/model_performance
   (KPIs, trends,        (single + batch scoring,      (metrics, confusion matrix,
    alerts, priorities)   gauge, contributions)         ROC/PR, thresholds)
              |                            |
              |                            v
              |                 src/prediction.local_contributions
              |                 (one-at-a-time sensitivity)
              v                            v
   pages/explainability        src/recommendations (9 deterministic rules)
   (permutation importance)                 |
                                            v
                                 pages/maintenance (P1-P4 + human approval)
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `src/config.py` | Single source of truth: paths, canonical feature list, leakage policy, palette, rule-engine thresholds. |
| `src/data_loader.py` | Dataset loading (UCI copy -> synthetic fallback), CSV schema validation with tolerant header aliases, demo-fleet and template generation. |
| `src/model.py` | Pipeline construction (imputation + one-hot + RandomForest), stratified splits, threshold derivation, evaluation, permutation importance, bundle persistence, `load_model_any()` auto-retrain. |
| `src/prediction.py` | Runtime scoring (single/batch), risk tiers, OCAT local explanations, value context vs training distributions. |
| `src/recommendations.py` | Deterministic rule engine: 9 traceable rules, severity composition to P1-P4 priorities. |
| `src/visualization.py` | Plotly template (dark navy, blue/teal) and figure factory: donut, stacked bars, trends, scatter, gauge, contributions, confusion matrix, ROC, PR, importance. |
| `src/widgets.py` | Shared UI blocks: page header, badges, demo disclosures, machine input form. |
| `pages/*.py` | One `render()` per page; routing in `app.py` via `st.navigation`. |

## Data flow and caching

- `st.cache_resource` caches the model bundle per server process
  (`src/prediction.load_bundle`). Retraining happens only when the bundle
  file is missing or unreadable (e.g. sklearn version mismatch).
- Uploaded/validated datasets live in `st.session_state["validated_df"]` and
  become the active dataset across dashboard, batch scoring and advisor.
- Demo fleet reads `data/sample_data.csv` (deterministic generator, seed 7)
  and is always labelled as demonstration records.

## Threshold design (validation-informed)

1. Alert threshold: the cut-off in `[0.02, 0.98]` that maximises F1 on the
   **validation** split. Current value: **0.32**.
2. Medium tier: 80th percentile of validation predicted scores -> **0.024**.
3. High tier: 95th percentile of validation predicted scores -> **0.24**.

No constant in the pipeline is hand-picked from test data; the test split is
evaluated exactly once. Guards (`np.clip`) only protect against degenerate
score distributions and are documented in `choose_thresholds()`.

## Leakage prevention

- Canonical features: `Type` + 5 operational numerics (`config.FEATURES`).
- `config.LEAKAGE_COLUMNS` = TWF/HDF/PWF/OSF/RNF labels + `Machine failure`
  target + UDI/Product ID identifiers; `prepare_features()` selects only the
  canonical list and asserts the leakage set is disjoint.
- The uploaded-CSV schema accepts `Machine failure` as an **optional** column
  used solely for post-hoc agreement stats in batch scoring, never as input.
- Tests enforce the disjointness (`tests/test_factorymind.py::test_leakage_columns_never_in_features`).

## Deployment topology

- Entry point `app.py` uses relative paths via `Path(__file__)` anchors, so
  the app runs from any working directory (Streamlit Cloud, Replit, VPS).
- `requirements.txt` pins major-version ranges; no system packages required.
- Cold start: load committed bundle (~1 s). Warm retrain fallback (~1-2 min
  on a free cloud instance) if the bundle is deleted.
