"""Page 5: Maintenance Advisor.

Deterministic, traceable maintenance recommendations built from the predicted
risk and the actual sensor readings. Every suggestion carries its evidence and
requires human approval - FactoryMind never authorises shutdowns or physical
intervention.

Team Sodium | International AI Builders Congress (hackathon MVP).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config, prediction, recommendations
from src.prediction import load_bundle
from src import widgets as ui


def _rules_reference(bundle: dict) -> None:
    alert = bundle["thresholds"]["alert"]
    with st.expander("Rule reference (all thresholds, fully traceable)"):
        st.markdown(
            f"| Rule | Trigger | Threshold(s) | Severity |\n"
            f"|---|---|---|---|\n"
            f"| ML-1 | Model tier = High | p >= High tier (validation q95) | 3 |\n"
            f"| ML-2 | Model tier = Medium | p >= Medium tier (validation q80) | 1 |\n"
            f"| ML-3 | Score >= alert point | p >= {alert:.2f} (F1-optimal on validation) | 2-3 |\n"
            f"| TL-1/2/3 | Tool wear | >= {config.WEAR_CRITICAL_MIN:.0f} / {config.WEAR_HIGH_MIN:.0f} / "
            f"{config.WEAR_WATCH_MIN:.0f} min | 3/2/1 |\n"
            f"| HD-1 | Heat-dissipation pattern | temp delta < {config.HDF_TEMP_DELTA_K} K and "
            f"speed < {config.HDF_RPM_LIMIT:.0f} rpm | 2 |\n"
            f"| PW-1 | Spindle power out of band | < {config.POWER_MIN_W:.0f} W or "
            f"> {config.POWER_MAX_W:.0f} W | 2 |\n"
            f"| OS-1 | Overstrain | wear x torque > type limit "
            f"(L {config.OSF_WEAR_TORQUE['L']:,.0f} / M {config.OSF_WEAR_TORQUE['M']:,.0f} / "
            f"H {config.OSF_WEAR_TORQUE['H']:,.0f} min*Nm) | 2 |\n"
            f"| TQ-1 | Torque above healthy range | > healthy q95 (from training data) | 1 |")
        st.caption("Thresholds calibrated to the failure mechanisms documented for "
                   "the UCI AI4I 2020 dataset. They are heuristics for a synthetic "
                   "dataset and must be re-validated against real equipment.")


def render() -> None:
    bundle = load_bundle()
    ui.page_header(
        "Maintenance Advisor",
        "Deterministic recommendations from the predicted risk and the actual "
        "readings - every trigger cites the value that fired it.",
    )
    st.warning(config.APPROVAL_TEXT, icon=None)

    # ---------------- choose the machine ----------------
    last_input = st.session_state.get("last_input")
    active_row = None
    source = "none"

    if last_input is not None:
        pred = st.session_state.get("last_pred")
        st.markdown(
            f"Using the last prediction: **{last_input.get('Machine ID') or 'unspecified machine'}** "
            f"- {ui.risk_badge(pred['tier']) if pred else ''} "
            f"(p = {pred['probability']:.2f}).",
            unsafe_allow_html=True)
        active_row, source = last_input, "last_prediction"
        with st.expander("Enter / edit machine values manually"):
            manual = ui.machine_input_form(bundle, last_input, key="adv_manual")
            if st.button("Use these values instead", type="primary"):
                pred_m = prediction.predict_row(bundle, manual)
                st.session_state["last_input"] = manual
                st.session_state["last_pred"] = pred_m
                st.rerun()
    else:
        st.info("No prediction in this session yet. Enter the machine's current "
                "readings below - the advisor will score and advise immediately.")
        manual = ui.machine_input_form(bundle, ui.healthy_defaults(bundle),
                                       key="adv_direct")
        if st.button("Score and advise", type="primary"):
            pred_m = prediction.predict_row(bundle, manual)
            st.session_state["last_input"] = manual
            st.session_state["last_pred"] = pred_m
            st.rerun()
        if st.session_state.get("last_pred") is None:
            _rules_reference(bundle)
            return
        active_row = st.session_state["last_input"]
        source = "manual_entry"

    # ---------------- recommendation ----------------
    pred = prediction.predict_row(bundle, active_row)
    rec = recommendations.generate_recommendation(active_row, pred, bundle)

    st.divider()
    badge_cls = {"P1": "high", "P2": "medium", "P3": "info", "P4": "muted"}[rec["priority"]]
    st.markdown(
        f"<div style='margin:2px 0 10px 0'>"
        f"<span class='fm-badge {badge_cls}' style='font-size:14px; padding:6px 14px;'>"
        f"{rec['priority']} - {rec['priority_label']}</span></div>",
        unsafe_allow_html=True)
    st.markdown(f"**Machine:** {rec['machine']} | **Model risk:** {pred['tier']} "
                f"(p = {pred['probability']:.2f})")
    st.markdown(rec["summary"])

    st.markdown("**Why (trigger evidence):**")
    if rec["triggers"]:
        trig_df = pd.DataFrame([{
            "Rule": t.code, "Finding": t.title,
            "Evidence (from actual input values)": t.evidence,
            "Severity": t.severity,
        } for t in rec["triggers"]])
        st.dataframe(trig_df, use_container_width=True, hide_index=True)
    else:
        st.markdown("- No rule triggered; readings are within typical ranges.")

    st.markdown("**Suggested inspection actions:**")
    for a in rec["actions"]:
        st.markdown(f"- {a}")

    if rec["priority"] == "P1":
        st.error("P1 priority: escalate to a qualified maintenance engineer "
                 "now. Do NOT stop or intervene on the machine without a "
                 "human decision.")
    st.success("Human approval required: an engineer must review and sign off "
               "on every action above before anything is executed.")

    # ---------------- traceability ----------------
    st.markdown("**Traceability - exact values used for this recommendation:**")
    trace = {k: v for k, v in active_row.items()
             if k in (["Machine ID", "Type"] + config.NUMERIC_FEATURES)}
    trace_df = pd.DataFrame(
        [{"Input": k, "Value": round(v, 2) if isinstance(v, (int, float)) else v}
         for k, v in trace.items()])
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.dataframe(trace_df, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Thresholds applied (validation-derived):**")
        thr = bundle["thresholds"]
        st.markdown(
            f"- Alert operating point: **{thr['alert']:.2f}**\n"
            f"- Medium tier: >= **{thr['medium']:.2f}** (validation q80)\n"
            f"- High tier: >= **{thr['high']:.2f}** (validation q95)\n"
            f"- Healthy torque q95 (training): "
            f"**{bundle['train_stats']['healthy']['Torque [Nm]']['q95']:.1f} Nm**")

    _rules_reference(bundle)
