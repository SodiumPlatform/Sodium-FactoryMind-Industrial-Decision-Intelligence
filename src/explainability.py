"""Explainability facade: global and per-prediction explanations.

Global importance (permutation, computed at training time) lives in the model
bundle; local one-at-a-time explanations live in src/prediction.py. This
module gives pages a single import point, matching the documented
architecture.

Team Sodium | International AI Builders Congress (hackathon MVP).
"""
from __future__ import annotations

import pandas as pd

from src.prediction import local_contributions, value_context  # noqa: F401 (re-export)


def global_importance(bundle: dict) -> pd.DataFrame:
    """Permutation importance (mean decrease in log-loss, test split)."""
    imp = bundle["permutation_importance"]
    std = bundle["permutation_importance_std"]
    df = pd.DataFrame({
        "Feature": list(imp.keys()),
        "Mean importance": [round(v, 4) for v in imp.values()],
        "Std": [round(std[k], 4) for k in imp.keys()],
    }).sort_values("Mean importance", ascending=False)
    return df.reset_index(drop=True)


def is_global_ready(bundle: dict) -> bool:
    """True when the bundle carries permutation-importance results."""
    return bool(bundle.get("permutation_importance"))
