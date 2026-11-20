"""Runtime prediction services (single + batch) and local explanations.

Team Sodium | International AI Builders Congress (hackathon MVP).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src import model as src_model


@st.cache_resource(show_spinner="Loading FactoryMind model ...")
def load_bundle() -> dict:
    """Load the trained model bundle (auto-trains on first run / fresh deploy)."""
    return src_model.load_model_any()


def risk_tier(probability: float, thresholds: dict) -> str:
    """Map a predicted probability to a risk tier using validation-derived tiers."""
    if probability >= thresholds["high"]:
        return "High"
    if probability >= thresholds["medium"]:
        return "Medium"
    return "Low"


def _normalise_type(value) -> str:
    return str(value).strip().upper()


def predict_row(bundle: dict, row: dict) -> dict:
    """Score one machine record. Returns probability, tier and alert flag.

    Uses ONLY config.FEATURES as input - failure-mode labels and identifiers
    are never accepted, even if present in the row dict.
    """
    X = pd.DataFrame([{f: row.get(f) for f in config.FEATURES}])[config.FEATURES]
    X["Type"] = X["Type"].map(_normalise_type)
    proba = float(bundle["pipeline"].predict_proba(X)[:, 1][0])
    return {
        "probability": proba,
        "tier": risk_tier(proba, bundle["thresholds"]),
        "alert": bool(proba >= bundle["thresholds"]["alert"]),
    }


def predict_batch(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Score many records; adds failure_probability / risk_tier / alert columns.

    Rows with missing feature values are scored via the pipeline's imputers;
    input columns beyond the canonical features are carried through untouched.
    """
    X = df[config.FEATURES].copy()
    X["Type"] = X["Type"].map(_normalise_type)
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    out = df.copy()
    out["failure_probability"] = np.round(proba, 4)
    out["risk_tier"] = [risk_tier(p, bundle["thresholds"]) for p in proba]
    out["alert"] = proba >= bundle["thresholds"]["alert"]
    return out


# ------------------------------------------------ local explanation (OCAT) --
def local_contributions(bundle: dict, row: dict) -> tuple:
    """One-at-a-time (OCAT) sensitivity analysis for a single prediction.

    For each feature: 'what would the predicted risk become if this reading
    were set to the typical-healthy training value?' A negative delta means
    the observed reading pushes risk UP relative to a typical healthy machine.

    This is a correlational sensitivity analysis on a synthetic dataset - it
    is NOT a causal explanation.
    """
    base = predict_row(bundle, row)["probability"]
    stats = bundle["train_stats"]
    rows = []
    for f in config.NUMERIC_FEATURES:
        lo, hi = stats["ranges"][f]
        baseline = float(np.clip(stats["healthy"][f]["q50"], lo, hi))
        alt = dict(row)
        alt[f] = baseline
        p_alt = predict_row(bundle, alt)["probability"]
        val = row.get(f)
        rows.append({
            "feature": f,
            "value": None if val is None or (isinstance(val, float) and np.isnan(val)) else float(val),
            "healthy_typical": baseline,
            "delta_if_normalised": float(p_alt - base),
        })
    if _normalise_type(row.get("Type", "L")) != "L":
        alt = dict(row)
        alt["Type"] = "L"
        p_alt = predict_row(bundle, alt)["probability"]
        rows.append({
            "feature": "Type",
            "value": row.get("Type"),
            "healthy_typical": "L",
            "delta_if_normalised": float(p_alt - base),
        })
    rows.sort(key=lambda r: r["delta_if_normalised"])
    return base, rows


def value_context(bundle: dict, row: dict) -> pd.DataFrame:
    """Compare each reading against healthy/failed training distributions."""
    stats = bundle["train_stats"]
    records = []
    for f in config.NUMERIC_FEATURES:
        val = row.get(f)
        h = stats["healthy"][f]
        if val is None:
            band = "no value"
        elif val < h["q10"]:
            band = "below typical healthy range"
        elif val <= h["q90"]:
            band = "typical healthy range"
        elif val <= h["q95"]:
            band = "elevated (above 90% of healthy readings)"
        else:
            band = "unusually high (> 95% of healthy readings)"
        records.append({
            "Indicator": f,
            "This machine": None if val is None else round(float(val), 2),
            "Typical healthy (median)": round(h["q50"], 2),
            "Typical failed (median)": round(stats["failed"][f]["q50"], 2),
            "Healthy q90": round(h["q90"], 2),
            "Assessment": band,
        })
    return pd.DataFrame(records)


def model_summary_text(bundle: dict) -> str:
    """One-paragraph honest model summary for UI panels."""
    mt = bundle["metrics_test"]
    prov = bundle["data_provenance"]
    return (
        f"Random Forest (400 trees, class-weighted) trained on the {prov['source']} "
        f"({prov['rows']:,} rows, synthetic dataset). Held-out test performance: "
        f"ROC-AUC {mt['roc_auc']:.2f}, PR-AUC {mt['average_precision']:.2f}, "
        f"recall {mt['recall']:.2f} and precision {mt['precision']:.2f} at the "
        f"validation-derived alert threshold {mt['threshold']:.2f}. Outputs are "
        "ranking scores, not calibrated probabilities."
    )
