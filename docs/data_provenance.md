# FactoryMind - Data Provenance

Team Sodium | International AI Builders Congress (hackathon MVP)

## Canonical dataset

| Field | Value |
|---|---|
| Name | AI4I 2020 Predictive Maintenance Dataset |
| Source | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset |
| Download | https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| DOI | 10.24432/C5HS5C |
| Rows | 10,000 |
| Columns | 14 |
| Failure rate | 3.39% (339 failures) |
| File in repo | `data/ai4i2020.csv` |
| sha256 (first 16 hex) | `dc6630cd9b1f0f85` |

## Academic citation

S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance
Applications," Third International Conference on Artificial Intelligence for
Industries (AI4I), 2020, pp. 69-74, doi: 10.24432/C5HS5C.

## Integrity handling

- The committed `data/ai4i2020.csv` is byte-identical to the file inside the
  official UCI zip archive (downloaded 2026-11-20). No rows or values were
  modified.
- The original file ships a header quirk: the tool-wear column is named
  `Tool wear [min]` instead of the documented `Tool wear [min]`. FactoryMind normalises this
  **in memory only** (`src/data_loader.load_ai4i`), keeping the file on disk
  untouched so the hash stays verifiable.
- `sha256` is recomputed and shown in the app's provenance panels.

## Column schema

| Column | Role | Used as model input? |
|---|---|---|
| UDI | record identifier | No (identifier) |
| Product ID | serial identifier (encodes quality variant) | No (identifier) |
| Type | product quality variant (L / M / H) | Yes (one-hot encoded) |
| Air temperature [K] | ambient temperature sensor | Yes |
| Process temperature [K] | process temperature sensor | Yes |
| Rotational speed [rpm] | spindle speed | Yes |
| Torque [Nm] | spindle torque | Yes |
| Tool wear [min] | accumulated tool usage | Yes |
| Machine failure | target label | Target only |
| TWF / HDF / PWF / OSF / RNF | failure-mode labels | **Never** (leakage) |

## Synthetic fallbacks (clearly labelled, never presented as real)

1. **Dataset fallback** (`src/data_loader.generate_synthetic_ai4i`): if
   `data/ai4i2020.csv` is missing, a synthetic dataset with the same schema
   and the documented AI4I failure mechanisms is generated (seed 42) so the
   app still boots offline. Every UI surface that shows this data prints
   "Synthetic fallback - NOT real telemetry", and the provenance dict flags
   `synthetic_fallback: True` (persisted in `models/metrics.json`).
2. **Demo fleet** (`data/sample_data.csv`, generator
   `src/data_loader.generate_demo_fleet`, seed 7): 12 demonstration machines
   (`M-001`...`M-012`) x 48 hourly readings with a few deliberate anomaly
   patterns (ageing tool, cooling drift, torque spikes). It exists so the
   dashboard has presentable content on first run and is labelled
   "Demonstration records" everywhere.
3. **Upload template** (`data/sample_template.csv`): three clearly-marked
   example rows for format discovery only.

## Regeneration

```bash
python scripts/download_dataset.py     # re-download the UCI original
python scripts/generate_demo_data.py   # regenerate demo fleet + template
python -m src.model                    # retrain and refresh metrics.json
```
