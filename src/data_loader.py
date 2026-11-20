"""Data loading, CSV schema validation and fallback dataset generation.

Canonical dataset: UCI AI4I 2020 Predictive Maintenance Dataset (CC BY 4.0),
DOI 10.24432/C5HS5C. The committed copy in data/ai4i2020.csv is kept
byte-identical to the original download (see docs/data_provenance.md); column
names are normalised in-memory only.

If the committed copy is missing, a clearly-flagged synthetic dataset with the
same schema and failure mechanisms is generated so the app still runs offline.
Uploaded files are validated against the canonical schema before any use.

Team Sodium | International AI Builders Congress (hackathon MVP).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from src import config

REQUIRED_COLUMNS = list(config.FEATURES)                 # Type + 5 numerics
OPTIONAL_COLUMNS = ["Machine ID", "Timestamp", config.TARGET]

# Tolerant header matching: normalised name -> canonical name.
_ALIASES = {
    "airtemperaturek": "Air temperature [K]",
    "airtemperature": "Air temperature [K]",
    "airtempk": "Air temperature [K]",
    "processtemperaturek": "Process temperature [K]",
    "processtemperature": "Process temperature [K]",
    "processtempk": "Process temperature [K]",
    "rotationalspeedrpm": "Rotational speed [rpm]",
    "rotationalspeed": "Rotational speed [rpm]",
    "speedrpm": "Rotational speed [rpm]",
    "torquenm": "Torque [Nm]",
    "torque": "Torque [Nm]",
    "toolwearmin": "Tool wear [min]",
    "toolwear": "Tool wear [min]",
    # the original UCI file ships the header quirk 'Tool wear in]'
    "toolwearin": "Tool wear [min]",
    "type": "Type",
    "machinetype": "Type",
    "quality": "Type",
    "machinefailure": config.TARGET,
    "failure": config.TARGET,
    "machineid": "Machine ID",
    "machine": "Machine ID",
    "timestamp": "Timestamp",
    "time": "Timestamp",
}


def _norm(name: str) -> str:
    """Normalise a column name for tolerant matching (lowercase, alphanumeric only)."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def read_csv_flexible(source: Union[str, Path, object]) -> pd.DataFrame:
    """Read a CSV from a path or file-like object with delimiter/encoding sniffing."""
    if isinstance(source, (str, Path)):
        source = open(source, "rb")
        own_handle = True
    else:
        own_handle = False
    try:
        try:
            return pd.read_csv(source, sep=None, engine="python")
        except UnicodeDecodeError:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, sep=None, engine="python", encoding="latin-1")
    finally:
        if own_handle:
            source.close()


# ------------------------------------------------------------ validation ---
@dataclass
class ValidationResult:
    """Outcome of validating an uploaded CSV against the canonical schema."""

    ok: bool = False
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    cleaned: Optional[pd.DataFrame] = None
    invalid_rows: Optional[pd.DataFrame] = None
    column_report: list = field(default_factory=list)
    n_input_rows: int = 0
    n_valid_rows: int = 0


def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    """Validate and clean an uploaded dataframe.

    Rows are only rejected when they cannot be scored (missing/non-numeric
    readings or an unknown machine Type). Out-of-range values generate
    warnings but are kept, so no data is dropped silently.
    """
    res = ValidationResult(n_input_rows=len(df))
    if df.empty:
        res.errors.append("The uploaded file contains no data rows.")
        return res

    # 1) tolerant column mapping: canonical names first, then alias table
    lookup = {_norm(c): c for c in df.columns}
    rename = {}
    for canon in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        n = _norm(canon)
        if n in lookup and lookup[n] != canon:
            rename[lookup[n]] = canon
        elif n not in lookup:
            for a_norm, a_canon in _ALIASES.items():
                if a_canon == canon and a_norm in lookup:
                    rename[lookup[a_norm]] = canon
                    break
    out = df.rename(columns=rename)
    for old, new in rename.items():
        res.column_report.append(
            {"column_in_file": old, "mapped_to": new, "status": "renamed"})
    present = set(out.columns)
    for canon in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        if canon in rename.values():
            continue
        status = "ok" if canon in present else "MISSING (required)" \
            if canon in REQUIRED_COLUMNS else "optional, not present"
        res.column_report.append(
            {"column_in_file": canon if canon in present else "-",
             "mapped_to": canon, "status": status})

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        res.errors.append(
            "Missing required column(s): "
            + ", ".join(f"'{c}'" for c in missing)
            + f". Columns found in the file: {', '.join(map(str, df.columns))}. "
            "Tip: download the sample template below and keep its header row.")
        return res

    work = out.copy()
    for col in config.NUMERIC_FEATURES:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    bad_mask = pd.Series(False, index=work.index)
    reasons = pd.Series("", index=work.index, dtype=object)

    # 2) rows that cannot be scored: missing/non-numeric readings
    num_na = work[config.NUMERIC_FEATURES].isna().any(axis=1)
    for idx in work.index[num_na]:
        cols = [c for c in config.NUMERIC_FEATURES if pd.isna(work.at[idx, c])]
        reasons[idx] = f"missing/non-numeric value(s): {', '.join(cols)}"
    bad_mask |= num_na

    # 3) machine Type must be one of L / M / H
    types = out["Type"].astype(str).str.strip().str.upper()
    type_bad = ~types.isin(config.TYPE_LEVELS)
    for idx in work.index[type_bad]:
        extra = f"invalid Type value: '{out.at[idx, 'Type']}'"
        reasons[idx] = f"{reasons[idx]}; {extra}" if reasons[idx] else extra
    bad_mask |= type_bad
    work["Type"] = types.where(~type_bad, work["Type"])

    # 4) optional label column (used for evaluation only, never as input)
    if config.TARGET in work.columns:
        lab = pd.to_numeric(work[config.TARGET], errors="coerce")
        n_bad_label = int(lab.notna().sum() - lab.isin([0, 1]).sum())
        work[config.TARGET] = lab.where(lab.isin([0, 1]))
        if n_bad_label:
            res.warnings.append(
                f"{n_bad_label} row(s) had a non-0/1 '{config.TARGET}' value; "
                "label was cleared (rows kept for prediction).")

    # 5) physical range warnings (rows kept and flagged, never dropped)
    for col, lo, hi in [
        ("Air temperature [K]", 250, 340),
        ("Process temperature [K]", 250, 360),
        ("Rotational speed [rpm]", 0, 4000),
        ("Torque [Nm]", -100, 200),
        ("Tool wear [min]", 0, 500),
    ]:
        vals = pd.to_numeric(work[col], errors="coerce").dropna()
        n_out = int((~vals.between(lo, hi)).sum())
        if n_out:
            res.warnings.append(
                f"{n_out} row(s) have {col} outside the plausible range "
                f"[{lo}, {hi}] - kept, but flagged as unusual.")

    res.invalid_rows = work.loc[bad_mask].copy()
    if not res.invalid_rows.empty:
        res.invalid_rows.insert(0, "source_row", [i + 2 for i in res.invalid_rows.index])
        res.invalid_rows.insert(1, "reason", reasons[bad_mask])
    res.cleaned = work.loc[~bad_mask].reset_index(drop=True)
    res.n_valid_rows = len(res.cleaned)
    if res.n_valid_rows == 0:
        res.errors.append("No valid rows remain after validation - nothing to analyse.")
        return res
    res.ok = True
    return res


# ---------------------------------------------------------------- loaders --
def load_ai4i() -> tuple:
    """Load the committed UCI AI4I dataset, or a flagged synthetic fallback.

    Returns (dataframe, provenance dict).
    """
    if config.RAW_DATA_PATH.exists():
        df = pd.read_csv(config.RAW_DATA_PATH)
        # normalise the UCI header quirk 'Tool wear in]' -> canonical name
        df = df.rename(columns={c: _ALIASES.get(_norm(c), c) for c in df.columns})
        sha = hashlib.sha256(config.RAW_DATA_PATH.read_bytes()).hexdigest()[:16]
        prov = {
            "source": "UCI AI4I 2020 Predictive Maintenance Dataset (committed copy)",
            "url": "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset",
            "license": "CC BY 4.0",
            "doi": "10.24432/C5HS5C",
            "rows": int(len(df)),
            "sha256_16": sha,
            "synthetic_fallback": False,
        }
        return df, prov
    df = generate_synthetic_ai4i()
    prov = {
        "source": "Synthetic fallback (committed UCI file missing) - NOT real telemetry",
        "url": "-", "license": "generated locally", "doi": "-",
        "rows": int(len(df)), "sha256_16": "-",
        "synthetic_fallback": True,
    }
    return df, prov


def load_sample_fleet() -> pd.DataFrame:
    """Load the demonstration machine fleet used by the dashboard."""
    if config.SAMPLE_DATA_PATH.exists():
        df = pd.read_csv(config.SAMPLE_DATA_PATH)
    else:
        df = generate_demo_fleet()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def sample_template_bytes() -> bytes:
    """CSV template for uploads, with three clearly-marked example rows."""
    df = pd.DataFrame({
        "Machine ID": ["M-101", "M-102", "M-103"],
        "Type": ["L", "M", "H"],
        "Air temperature [K]": [298.1, 299.3, 300.0],
        "Process temperature [K]": [308.2, 309.5, 310.6],
        "Rotational speed [rpm]": [1551, 1408, 1602],
        "Torque [Nm]": [42.8, 51.7, 38.4],
        "Tool wear [min]": [12, 187, 45],
    })
    return df.to_csv(index=False).encode("utf-8")


# ------------------------------------------------- synthetic generators ----
def generate_synthetic_ai4i(n: int = 10000, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Generate a synthetic dataset mimicking the documented AI4I 2020
    failure mechanisms. Used ONLY when the committed UCI file is missing.

    The result is clearly flagged as synthetic via the provenance dict and is
    never presented as real factory telemetry.
    """
    rng = np.random.default_rng(seed)
    types = rng.choice(config.TYPE_LEVELS, n, p=[0.6, 0.3, 0.1])
    air = rng.normal(300.0, 2.0, n)
    delta = rng.normal(9.0, 1.4, n)
    proc = air + delta
    rpm = np.clip(rng.normal(1538.0, 150.0, n), 1168.0, 2886.0)
    torque = np.clip(45.0 + (1500.0 - rpm) * 0.055 + rng.normal(0, 4.0, n), 3.8, 76.6)
    wear = rng.uniform(0, 253.0, n)

    twf = rng.random(n) < 0.0046
    hdf = (delta < config.HDF_TEMP_DELTA_K) & (rpm < config.HDF_RPM_LIMIT) & (rng.random(n) < 0.08)
    power = torque * rpm * 2 * np.pi / 60
    pwf = ((power < config.POWER_MIN_W) | (power > config.POWER_MAX_W)) & (rng.random(n) < 0.12)
    osf_lim = np.array([config.OSF_WEAR_TORQUE[t] for t in types])
    osf = (wear * torque > osf_lim) & (rng.random(n) < 0.10)
    rnf = rng.random(n) < 0.0019
    failure = (twf | hdf | pwf | osf | rnf).astype(int)

    return pd.DataFrame({
        "UDI": np.arange(1, n + 1),
        "Product ID": [f"{t}{10000 + i}" for i, t in enumerate(types)],
        "Type": types,
        "Air temperature [K]": air.round(1),
        "Process temperature [K]": proc.round(1),
        "Rotational speed [rpm]": rpm.round(0).astype(int),
        "Torque [Nm]": torque.round(1),
        "Tool wear [min]": wear.round(0).astype(int),
        config.TARGET: failure,
        "TWF": twf.astype(int), "HDF": hdf.astype(int), "PWF": pwf.astype(int),
        "OSF": osf.astype(int), "RNF": rnf.astype(int),
    })


def generate_demo_fleet(hours: int = 48, seed: int = 7) -> pd.DataFrame:
    """Generate the demonstration fleet (M-001..M-012, 48 hourly readings).

    Clearly labelled as demonstration records everywhere in the UI. A few
    machines carry deliberately abnormal patterns (ageing tool, cooling
    problem, torque spikes) so the dashboard shows realistic behaviour.
    """
    rng = np.random.default_rng(seed)
    machines = [
        ("M-001", "L", {}), ("M-002", "L", {}), ("M-003", "M", {}),
        ("M-004", "M", {}),
        # heavily worn tool pushing high cutting load -> genuine high-risk pattern
        ("M-005", "M", {"wear0": 210.0, "wear_rate": 0.80,
                        "rpm_mean": 1310.0, "torque_shift": 14.0}),
        ("M-006", "L", {}),
        ("M-007", "H", {"wear0": 200.0, "wear_rate": 0.45}),          # ageing tool
        ("M-008", "M", {}), ("M-009", "H", {}),
        ("M-010", "M", {"delta_q": 6.2, "rpm_mean": 1260.0}),         # cooling issue
        ("M-011", "L", {}),
        ("M-012", "H", {"wear0": 130.0, "wear_rate": 0.45, "torque_spike": True}),
    ]
    delta_by_type = {"L": 9.6, "M": 10.0, "H": 10.4}
    rpm_by_type = {"L": 1760.0, "M": 1790.0, "H": 1820.0}
    end = pd.Timestamp.now("UTC").floor("h")
    rows = []
    for mid, mtype, opt in machines:
        wear = opt.get("wear0", float(rng.uniform(8, 70)))
        wear_rate = opt.get("wear_rate", float(rng.uniform(0.08, 0.22)))
        torque_shift = opt.get("torque_shift", 0.0)
        for t in range(hours):
            ts = end - pd.Timedelta(hours=hours - 1 - t)
            air = 298.6 + 1.2 * np.sin(2 * np.pi * ts.hour / 24) + rng.normal(0, 0.15)
            delta = opt.get("delta_q", delta_by_type[mtype]) + rng.normal(0, 0.45)
            proc = air + delta
            rpm = opt.get("rpm_mean", rpm_by_type[mtype]) + rng.normal(0, 60)
            torque = (6200.0 * 60 / (2 * np.pi * max(rpm, 900.0))
                      + torque_shift + rng.normal(0, 2.2))
            if opt.get("torque_spike") and rng.random() < 0.18:
                torque += rng.uniform(12, 28)
            wear = min(wear + wear_rate + rng.normal(0, 0.05), 253.0)
            rows.append({
                "Machine ID": mid,
                "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "Type": mtype,
                "Air temperature [K]": round(float(air), 2),
                "Process temperature [K]": round(float(proc), 2),
                "Rotational speed [rpm]": int(round(rpm)),
                "Torque [Nm]": round(float(torque), 2),
                "Tool wear [min]": round(float(wear), 1),
            })
    return pd.DataFrame(rows)
