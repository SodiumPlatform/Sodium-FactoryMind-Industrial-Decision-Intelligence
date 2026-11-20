"""FactoryMind test suite (pytest).

Run from the project root:
    python -m pytest tests -q

Covers: dataset integrity, CSV validation (good/bad/invalid), leakage policy,
model bundle sanity, single + batch prediction, recommendation determinism and
app/page imports. The model bundle is loaded once per session; if it is
missing, load_model_any() trains it (same behaviour as a fresh deployment).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, data_loader
from src import model as src_model
from src import prediction, recommendations


@pytest.fixture(scope="session")
def bundle():
    return src_model.load_model_any()


@pytest.fixture(scope="session")
def demo_fleet():
    return data_loader.generate_demo_fleet()


# ------------------------------------------------------------------ data ----
def test_committed_dataset_loads():
    df, prov = data_loader.load_ai4i()
    assert len(df) >= 1000, "expected the full 10k-row dataset"
    for col in config.FEATURES + [config.TARGET]:
        assert col in df.columns, f"missing canonical column {col}"
    assert not df[config.FEATURES + [config.TARGET]].isna().any().any()
    if not prov["synthetic_fallback"]:
        assert prov["license"] == "CC BY 4.0"
        assert prov["doi"] == "10.24432/C5HS5C"


def test_leakage_columns_never_in_features():
    assert set(config.FEATURES).isdisjoint(set(config.LEAKAGE_COLUMNS))
    # failure-mode labels must not be reachable through prepare_features
    df, _ = data_loader.load_ai4i()
    X = src_model.prepare_features(df)
    assert list(X.columns) == config.FEATURES


def test_template_has_required_columns():
    df = pd.read_csv(pd.io.common.BytesIO(data_loader.sample_template_bytes()))
    res = data_loader.validate_dataframe(df)
    assert res.ok, res.errors


# ------------------------------------------------------------ validation ----
def _good_df():
    return pd.DataFrame({
        "Machine ID": ["M-A", "M-B"],
        "Type": ["L", "H"],
        "Air temperature [K]": [298.1, 300.2],
        "Process temperature [K]": [308.0, 310.5],
        "Rotational speed [rpm]": [1551, 1400],
        "Torque [Nm]": [42.8, 51.7],
        "Tool wear [min]": [12, 187],
    })


def test_validation_accepts_good_csv():
    res = data_loader.validate_dataframe(_good_df())
    assert res.ok and res.n_valid_rows == 2
    assert res.invalid_rows.empty


def test_validation_rejects_missing_column():
    bad = _good_df().drop(columns=["Torque [Nm]"])
    res = data_loader.validate_dataframe(bad)
    assert not res.ok
    assert any("Torque [Nm]" in e for e in res.errors)


def test_validation_detects_invalid_rows():
    bad = pd.concat([_good_df(), pd.DataFrame([{
        "Machine ID": "M-C", "Type": "X", "Air temperature [K]": 299.0,
        "Process temperature [K]": 309.0, "Rotational speed [rpm]": "not-a-number",
        "Torque [Nm]": 44.0, "Tool wear [min]": 30}])], ignore_index=True)
    res = data_loader.validate_dataframe(bad)
    assert res.ok  # valid rows survive
    assert len(res.invalid_rows) == 1
    assert "Type" in res.invalid_rows.iloc[0]["reason"]
    assert res.n_valid_rows == 2


def test_validation_tolerant_headers():
    renamed = _good_df().rename(columns={
        "Air temperature [K]": "air temp (K)", "Torque [Nm]": "TORQUE"})
    res = data_loader.validate_dataframe(renamed)
    assert res.ok and res.n_valid_rows == 2


def test_validation_flags_out_of_range_but_keeps_rows():
    df = _good_df()
    df.loc[0, "Tool wear [min]"] = 900.0  # beyond plausible range
    res = data_loader.validate_dataframe(df)
    assert res.ok and res.n_valid_rows == 2
    assert any("outside the plausible range" in w for w in res.warnings)


# ---------------------------------------------------------------- model ----
def test_bundle_sanity(bundle):
    assert bundle["model_name"] == "RandomForestClassifier"
    thr = bundle["thresholds"]
    assert 0 < thr["medium"] < thr["high"] <= 1
    assert 0 < thr["alert"] < 1
    assert thr["derivation"]
    mt = bundle["metrics_test"]
    assert 0.5 < mt["roc_auc"] < 1.0  # must beat chance; real value reported in docs
    assert mt["average_precision"] > mt["support"]["positives"] / mt["support"]["n"]
    assert set(bundle["permutation_importance"].keys()) == set(config.FEATURES)


def test_test_split_untouched_sizes(bundle):
    split = bundle["split_info"]
    assert split["test_n"] > 0 and split["val_n"] > 0 and split["train_n"] > 0
    assert split["train_pos"] / split["train_n"] == pytest.approx(
        split["test_pos"] / split["test_n"], abs=0.02)  # stratification


# ------------------------------------------------------------- prediction ----
def test_predict_single_range_and_tier(bundle):
    row = {"Type": "M", "Air temperature [K]": 300.0,
           "Process temperature [K]": 310.0, "Rotational speed [rpm]": 1500.0,
           "Torque [Nm]": 45.0, "Tool wear [min]": 100.0}
    pred = prediction.predict_row(bundle, row)
    assert 0.0 <= pred["probability"] <= 1.0
    assert pred["tier"] in ("Low", "Medium", "High")
    assert isinstance(pred["alert"], bool)


def test_predict_batch_adds_columns(demo_fleet, bundle):
    sample = demo_fleet.head(50)
    scored = prediction.predict_batch(bundle, sample)
    assert len(scored) == len(sample)
    for col in ("failure_probability", "risk_tier", "alert"):
        assert col in scored.columns
    assert scored["failure_probability"].between(0, 1).all()


def test_local_contributions_shape(bundle):
    row = {"Type": "H", "Air temperature [K]": 300.0,
           "Process temperature [K]": 313.0, "Rotational speed [rpm]": 1300.0,
           "Torque [Nm]": 60.0, "Tool wear [min]": 220.0}
    _, contrib = prediction.local_contributions(bundle, row)
    assert len(contrib) == len(config.FEATURES)
    deltas = [c["delta_if_normalised"] for c in contrib]
    assert all(-1.0 <= d <= 1.0 for d in deltas)


# -------------------------------------------------------- recommendations ----
def test_recommendation_deterministic_and_traceable(bundle):
    row = {"Machine ID": "M-TEST", "Type": "H",
           "Air temperature [K]": 299.0, "Process temperature [K]": 307.0,
           "Rotational speed [rpm]": 1320.0, "Torque [Nm]": 62.0,
           "Tool wear [min]": 245.0}
    pred = prediction.predict_row(bundle, row)
    r1 = recommendations.generate_recommendation(row, pred, bundle)
    r2 = recommendations.generate_recommendation(row, pred, bundle)
    assert r1["priority"] == r2["priority"]
    assert r1["summary"] == r2["summary"]
    assert [t.code for t in r1["triggers"]] == [t.code for t in r2["triggers"]]
    assert r1["priority"] in ("P1", "P2", "P3", "P4")
    # wear 245 -> TL-1 must fire with evidence containing the actual value
    assert any(t.code == "TL-1" for t in r1["triggers"])
    assert "245" in next(t for t in r1["triggers"] if t.code == "TL-1").evidence
    assert config.APPROVAL_TEXT == r1["approval"]


def test_low_risk_machine_gets_routine_priority(bundle):
    row = {"Machine ID": "M-OK", "Type": "L",
           "Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
           "Rotational speed [rpm]": 1600.0, "Torque [Nm]": 40.0,
           "Tool wear [min]": 20.0}
    pred = prediction.predict_row(bundle, row)
    rec = recommendations.generate_recommendation(row, pred, bundle)
    assert rec["priority"] in ("P3", "P4")
    assert config.APPROVAL_TEXT == rec["approval"]


# -------------------------------------------------------------- app import ----
def test_app_and_pages_import():
    import importlib
    import app  # noqa: F401
    for mod in ("dashboard", "data_upload", "predictions", "explainability",
                "maintenance", "model_performance", "about"):
        importlib.import_module(f"pages.{mod}")
