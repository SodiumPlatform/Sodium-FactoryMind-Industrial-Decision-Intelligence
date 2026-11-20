"""Central configuration for FactoryMind.

Single source of truth for paths, feature definitions, theme colors and the
domain thresholds used by the maintenance rule engine.

Data-leakage policy: only the six operational features listed in FEATURES may
ever be used as model input. Failure-mode labels (TWF/HDF/PWF/OSF/RNF) and
record identifiers (UDI, Product ID) are structurally excluded.

Team Sodium | International AI Builders Congress (hackathon MVP).
"""
from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------- paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

RAW_DATA_PATH = DATA_DIR / "ai4i2020.csv"
SAMPLE_DATA_PATH = DATA_DIR / "sample_data.csv"
TEMPLATE_PATH = DATA_DIR / "sample_template.csv"
MODEL_PATH = MODEL_DIR / "factorymind_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

APP_VERSION = "1.0.0-mvp"
RANDOM_STATE = 42

APP_NAME = "FactoryMind"
APP_TAGLINE = "Industrial Decision Intelligence"

# ---------------------------------------------------------------- schema ---
TARGET = "Machine failure"
CATEGORICAL_FEATURES = ["Type"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Never used as model input: target labels, failure-mode labels, identifiers.
LEAKAGE_COLUMNS = [
    TARGET,
    "TWF", "HDF", "PWF", "OSF", "RNF",
    "UDI", "Product ID",
]

TYPE_LEVELS = ["L", "M", "H"]

# ------------------------------------------------- rule-engine thresholds --
# Calibrated to the failure mechanisms documented for the UCI AI4I 2020
# dataset (S. Matzka, "Explainable Artificial Intelligence for Predictive
# Maintenance Applications", AI4I 2020). They are heuristics for this SYNTHETIC
# dataset and must be re-validated against real equipment before production.
WEAR_WATCH_MIN = 160.0      # schedule monitoring
WEAR_HIGH_MIN = 200.0       # plan tool service
WEAR_CRITICAL_MIN = 240.0   # near the end-of-life window used by the TWF mechanism
HDF_TEMP_DELTA_K = 8.6      # below this delta (and low rpm) -> heat-dissipation pattern
HDF_RPM_LIMIT = 1380.0
POWER_MIN_W = 3500.0        # PWF mechanism band (Torque x angular speed)
POWER_MAX_W = 9000.0
OSF_WEAR_TORQUE = {"L": 11000.0, "M": 12000.0, "H": 13000.0}  # min x Nm

# ---------------------------------------------------------------- palette --
BG = "#0B1220"          # dark navy
PANEL = "#101B31"
BORDER = "#1E2A44"
PRIMARY = "#3B82F6"     # professional blue
ACCENT = "#22D3EE"      # teal highlight
TEXT = "#E9EFFA"
MUTED = "#93A4BC"
SUCCESS = "#34D399"
WARNING = "#FBBF24"
DANGER = "#F87171"

RISK_COLORS = {"Low": SUCCESS, "Medium": WARNING, "High": DANGER}
RISK_LEVELS = ["Low", "Medium", "High"]
PRIORITY_COLORS = {"P1": DANGER, "P2": WARNING, "P3": ACCENT, "P4": MUTED}

APPROVAL_TEXT = (
    "Advisory only. FactoryMind never authorises shutdowns or physical "
    "intervention automatically. Every suggested action requires review and "
    "sign-off by a qualified maintenance engineer."
)

DATASET_CITATION = (
    "S. Matzka, 'Explainable Artificial Intelligence for Predictive Maintenance "
    "Applications', Third International Conference on Artificial Intelligence "
    "for Industries (AI4I), 2020, pp. 69-74. DOI: 10.24432/C5HS5C (UCI ML "
    "Repository), license CC BY 4.0."
)
