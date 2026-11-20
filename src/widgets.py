"""Reusable Streamlit UI building blocks shared across pages.

Team Sodium | International AI Builders Congress (hackathon MVP).
"""
from __future__ import annotations

import streamlit as st

from src import config


def page_header(title: str, subtitle: str = "") -> None:
    """Consistent page header with a subtle accent rule."""
    st.markdown(
        f"<h2 style='margin-bottom:0.15rem; letter-spacing:-0.2px;'>{title}</h2>",
        unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)
    st.markdown(
        "<div style='height:2px; width:64px; margin:4px 0 14px 0; "
        f"background:linear-gradient(90deg,{config.PRIMARY},{config.ACCENT});"
        "border-radius:2px;'></div>",
        unsafe_allow_html=True)


def risk_badge(tier: str) -> str:
    """Inline HTML badge for a risk tier."""
    cls = {"Low": "low", "Medium": "medium", "High": "high"}.get(tier, "low")
    return f"<span class='fm-badge {cls}'>{tier} risk</span>"


def priority_badge(priority: str) -> str:
    cls = {"P1": "high", "P2": "medium", "P3": "info", "P4": "muted"}.get(priority, "muted")
    return f"<span class='fm-badge {cls}'>{priority}</span>"


def demo_note(text: str = None) -> None:
    """Standard demo-data disclosure line."""
    st.caption(text or
               "Demonstration records generated for this MVP - not real "
               "factory telemetry. Statistics on this page are sample data.")


def machine_input_form(bundle: dict, defaults: dict = None, key: str = "mi") -> dict:
    """Operational-value entry form (Type + 5 sensor readings).

    Widget ranges come from the training data ranges so users cannot submit
    physically impossible values; defaults are the healthy medians.
    """
    stats = bundle["train_stats"]
    d = defaults or {}
    ranges = stats["ranges"]
    healthy = stats["healthy"]

    def _bounds(f, pad_frac=0.06):
        lo, hi = ranges[f]
        pad = max((hi - lo) * pad_frac, 1e-6)
        return float(lo - pad), float(hi + pad)

    with st.container(border=True):
        c0, c1, c2, c3 = st.columns([1.1, 1.2, 1.2, 1.4])
        with c0:
            mid = st.text_input("Machine ID (optional)",
                                value=d.get("Machine ID") or "", key=f"{key}_id",
                                placeholder="e.g. M-042")
            mtype = st.selectbox("Machine type", config.TYPE_LEVELS,
                                 index=config.TYPE_LEVELS.index(d.get("Type", "L")),
                                 key=f"{key}_type",
                                 help="L / M / H product quality variant (UCI schema).")
        air_lo, air_hi = _bounds("Air temperature [K]")
        proc_lo, proc_hi = _bounds("Process temperature [K]")
        rpm_lo, rpm_hi = _bounds("Rotational speed [rpm]")
        tq_lo, tq_hi = _bounds("Torque [Nm]")
        with c1:
            air = st.number_input("Air temperature (K)", value=float(
                d.get("Air temperature [K]", healthy["Air temperature [K]"]["q50"])),
                min_value=air_lo, max_value=air_hi, step=0.1, format="%.1f", key=f"{key}_air")
            proc = st.number_input("Process temperature (K)", value=float(
                d.get("Process temperature [K]", healthy["Process temperature [K]"]["q50"])),
                min_value=proc_lo, max_value=proc_hi, step=0.1, format="%.1f", key=f"{key}_proc")
        with c2:
            rpm = st.number_input("Rotational speed (rpm)", value=float(
                d.get("Rotational speed [rpm]", healthy["Rotational speed [rpm]"]["q50"])),
                min_value=rpm_lo, max_value=rpm_hi, step=5.0, format="%.0f", key=f"{key}_rpm")
            wear = st.number_input("Tool wear (min)", value=float(
                d.get("Tool wear [min]", healthy["Tool wear [min]"]["q50"])),
                min_value=0.0, max_value=300.0, step=1.0, format="%.0f", key=f"{key}_wear")
        with c3:
            torque = st.number_input("Torque (Nm)", value=float(
                d.get("Torque [Nm]", healthy["Torque [Nm]"]["q50"])),
                min_value=tq_lo, max_value=tq_hi, step=0.5, format="%.1f", key=f"{key}_torque")
            st.caption(f"Input ranges follow the training data: "
                       f"{ranges['Torque [Nm]'][0]:.0f}-{ranges['Torque [Nm]'][1]:.0f} Nm etc.")

    return {
        "Machine ID": mid.strip() or None,
        "Type": mtype,
        "Air temperature [K]": float(air),
        "Process temperature [K]": float(proc),
        "Rotational speed [rpm]": float(rpm),
        "Torque [Nm]": float(torque),
        "Tool wear [min]": float(wear),
    }


def healthy_defaults(bundle: dict) -> dict:
    """Typical-healthy median inputs (used as form defaults)."""
    stats = bundle["train_stats"]
    row = {f: stats["healthy"][f]["q50"] for f in config.NUMERIC_FEATURES}
    row["Type"] = "L"
    return row


def metrics_mini_table(metrics: dict) -> str:
    """Markdown table of the key metrics at the alert threshold."""
    return (
        f"| Metric (test split, threshold {metrics['threshold']:.2f}) | Value |\n"
        "|---|---|\n"
        f"| ROC-AUC | {metrics['roc_auc']:.3f} |\n"
        f"| PR-AUC (average precision) | {metrics['average_precision']:.3f} |\n"
        f"| Precision | {metrics['precision']:.3f} |\n"
        f"| Recall | {metrics['recall']:.3f} |\n"
        f"| F1-score | {metrics['f1']:.3f} |\n"
        f"| Accuracy | {metrics['accuracy']:.3f} |"
    )
