"""Feature preparation and model training (strict no-leakage policy).

Only the six operational features in config.FEATURES may be used as input:
Type, Air temperature, Process temperature, Rotational speed, Torque, Tool wear.
Failure-mode labels (TWF/HDF/PWF/OSF/RNF) and identifiers (UDI, Product ID)
are structurally excluded; the target is 'Machine failure'.

Training protocol (reproducible, seed 42):
  - stratified 60/20/20 train/validation/test split
  - preprocessing pipeline: median/mode imputation + one-hot encoding of Type
  - RandomForestClassifier with class_weight='balanced_subsample'
  - decision threshold chosen on the VALIDATION split (max F1)
  - risk-tier thresholds derived from the validation score distribution
    (Medium = 80th percentile, High = 95th percentile of predicted scores)
  - test split is touched exactly once, for final reporting

Team Sodium | International AI Builders Congress (hackathon MVP).
"""
from __future__ import annotations

import datetime as _dt
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config, data_loader


def build_pipeline() -> Pipeline:
    """Preprocessing + classifier pipeline (all fitting happens inside it)."""
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    num = Pipeline([("impute", SimpleImputer(strategy="median"))])
    pre = ColumnTransformer(
        [("cat", cat, config.CATEGORICAL_FEATURES),
         ("num", num, config.NUMERIC_FEATURES)],
        remainder="drop",
    )
    clf = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", pre), ("clf", clf)])


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return X restricted to the canonical feature columns (leakage-safe)."""
    missing = [c for c in config.FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns missing: {missing}")
    leaked = [c for c in config.LEAKAGE_COLUMNS if c in config.FEATURES]
    assert not leaked, f"Leakage columns present in feature set: {leaked}"
    return df[config.FEATURES].copy()


def stratified_splits(df: pd.DataFrame):
    """Stratified 60/20/20 train/validation/test split (seed 42)."""
    work = df.dropna(subset=[config.TARGET]).copy()
    X = prepare_features(work)
    y = work[config.TARGET].astype(int)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=config.RANDOM_STATE)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=config.RANDOM_STATE)
    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)


def evaluate_at_threshold(y_true, proba: np.ndarray, threshold: float) -> dict:
    """Threshold-free and threshold-conditioned metrics for one split."""
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "support": {"n": int(len(y_true)), "positives": int((pd.Series(y_true) == 1).sum())},
        "classification_report": classification_report(
            y_true, pred, zero_division=0, output_dict=True),
    }


def choose_thresholds(y_val, val_proba: np.ndarray) -> dict:
    """Derive operating thresholds from the VALIDATION split only.

    alert  : maximises F1 on validation (decision operating point)
    medium : 80th percentile of validation predicted scores
    high   : 95th percentile of validation predicted scores

    Nothing here is hand-picked: both derivations are computed from held-out
    validation data and reported back to the user.
    """
    sweep = np.linspace(0.02, 0.98, 97)
    f1s = [f1_score(y_val, (val_proba >= t).astype(int), zero_division=0) for t in sweep]
    t_alert = float(sweep[int(np.argmax(f1s))])
    t_med = float(np.quantile(val_proba, 0.80))
    t_high = float(np.quantile(val_proba, 0.95))
    # guards against degenerate score distributions
    t_med = float(np.clip(t_med, 0.02, 0.99))
    t_high = float(np.clip(t_high, t_med + 0.02, 0.995))
    return {
        "alert": t_alert,
        "medium": t_med,
        "high": t_high,
        "derivation": (
            "Alert threshold = F1-maximising cut-off on the held-out validation "
            "split; Medium and High risk tiers = 80th / 95th percentiles of the "
            "validation predicted-score distribution. No hand-picked constants."
        ),
    }


def _train_stats(X_tr: pd.DataFrame, y_tr: pd.Series) -> dict:
    """Descriptive stats of the TRAIN split (healthy vs failed), for context."""
    df = X_tr.copy()
    df["_y"] = y_tr.values
    stats = {"healthy": {}, "failed": {}, "ranges": {}}
    for f in config.NUMERIC_FEATURES:
        h = df.loc[df["_y"] == 0, f]
        s = df.loc[df["_y"] == 1, f]
        stats["healthy"][f] = {
            "q10": float(h.quantile(0.10)), "q50": float(h.quantile(0.50)),
            "q90": float(h.quantile(0.90)), "q95": float(h.quantile(0.95)),
            "mean": float(h.mean()),
        }
        stats["failed"][f] = {"q50": float(s.quantile(0.50)), "mean": float(s.mean())}
        stats["ranges"][f] = [float(df[f].min()), float(df[f].max())]
    return stats


def train_and_save(data_path=None) -> dict:
    """Full training run: fit, derive thresholds, evaluate, save the bundle.

    Returns the model bundle dict (also persisted to models/ as joblib + JSON).
    """
    if data_path is None:
        df, prov = data_loader.load_ai4i()
    else:
        df = pd.read_csv(data_path)
        df = df.rename(columns={c: c for c in df.columns})
        prov = {"source": str(data_path), "synthetic_fallback": False}

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = stratified_splits(df)
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)

    val_proba = pipe.predict_proba(X_val)[:, 1]
    thresholds = choose_thresholds(y_val, val_proba)
    metrics_val = evaluate_at_threshold(y_val, val_proba, thresholds["alert"])
    te_proba = pipe.predict_proba(X_te)[:, 1]
    metrics_test = evaluate_at_threshold(y_te, te_proba, thresholds["alert"])

    perm = permutation_importance(
        pipe, X_te, y_te, scoring="neg_log_loss",
        n_repeats=8, random_state=config.RANDOM_STATE, n_jobs=-1)
    importances = {f: float(m) for f, m in zip(config.FEATURES, perm.importances_mean)}
    importances_std = {f: float(s) for f, s in zip(config.FEATURES, perm.importances_std)}

    bundle = {
        "pipeline": pipe,
        "features": list(config.FEATURES),
        "thresholds": thresholds,
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
        "permutation_importance": importances,
        "permutation_importance_std": importances_std,
        "train_stats": _train_stats(X_tr, y_tr),
        "X_test": X_te,
        "y_test": y_te.reset_index(drop=True),
        "split_info": {
            "train_n": int(len(y_tr)), "train_pos": int(y_tr.sum()),
            "val_n": int(len(y_val)), "val_pos": int(y_val.sum()),
            "test_n": int(len(y_te)), "test_pos": int(y_te.sum()),
        },
        "data_provenance": prov,
        "model_name": "RandomForestClassifier",
        "params": {k: v for k, v in pipe.named_steps["clf"].get_params().items()
                   if k in ("n_estimators", "min_samples_leaf", "max_features",
                            "class_weight", "random_state")},
        "sklearn_version": sklearn.__version__,
        "trained_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "app_version": config.APP_VERSION,
    }
    joblib.dump(bundle, config.MODEL_PATH)

    json_safe = {
        "trained_at": bundle["trained_at"],
        "model_name": bundle["model_name"],
        "params": bundle["params"],
        "sklearn_version": bundle["sklearn_version"],
        "thresholds": {k: v for k, v in thresholds.items()},
        "metrics_val": {k: v for k, v in metrics_val.items()
                        if k not in ("classification_report",)},
        "metrics_test": {k: v for k, v in metrics_test.items()
                         if k not in ("classification_report",)},
        "permutation_importance": importances,
        "split_info": bundle["split_info"],
        "data_provenance": prov,
    }
    config.METRICS_PATH.write_text(json.dumps(json_safe, indent=2, default=str))
    return bundle


def load_model_any() -> dict:
    """Load the persisted bundle; retrain automatically if it is missing or
    unreadable (e.g. sklearn version mismatch on a fresh deployment)."""
    if config.MODEL_PATH.exists():
        try:
            return joblib.load(config.MODEL_PATH)
        except Exception:  # noqa: BLE001 - fall through to retrain
            pass
    return train_and_save()


if __name__ == "__main__":
    b = train_and_save()
    mt = b["metrics_test"]
    print("Model trained and saved to", config.MODEL_PATH)
    print(f"  test ROC-AUC        : {mt['roc_auc']:.3f}")
    print(f"  test PR-AUC         : {mt['average_precision']:.3f}")
    print(f"  test F1 @ {mt['threshold']:.2f}    : {mt['f1']:.3f}")
    print(f"  test precision      : {mt['precision']:.3f}")
    print(f"  test recall         : {mt['recall']:.3f}")
    print(f"  thresholds          : {b['thresholds']}")
    print(f"  importance          : {b['permutation_importance']}")
