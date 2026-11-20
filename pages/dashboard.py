"""Page 1: Executive Dashboard.

Live KPIs, risk distribution, operational trends, alerts and maintenance
priority overview - all computed by the trained model from the active dataset
(demo fleet by default, validated upload otherwise).

Team Sodium | International AI Builders Congress (hackathon MVP).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config, data_loader, prediction, recommendations
from src.prediction import load_bundle
from src import visualization as viz
from src import widgets as ui


def _resolve_source() -> tuple:
    """Pick the active dataset: validated upload > demo fleet."""
    uploaded = st.session_state.get("validated_df")
    if uploaded is not None and len(uploaded) > 0:
        return uploaded.copy(), "Uploaded dataset (validated on the Machine Data Upload page)", False
    return data_loader.load_sample_fleet(), "Demonstration fleet (data/sample_data.csv)", True


def _latest_snapshot(scored: pd.DataFrame) -> pd.DataFrame:
    """Latest reading per machine when timestamps exist, else all rows."""
    if "Timestamp" in scored.columns:
        return (scored.sort_values("Timestamp")
                .drop_duplicates(subset=["Machine ID"], keep="last"))
    return scored


def _overall_status(p1: int, p2: int, high: int) -> str:
    if p1 > 0 or high > 0:
        return "Alarm"
    if p2 > 0:
        return "Watch"
    return "Nominal"


def render() -> None:
    bundle = load_bundle()
    ui.page_header(
        "Executive Dashboard",
        "Machine health and failure-risk overview computed live by the trained "
        "Random Forest model - thresholds are validation-derived.",
    )

    df, source_name, is_demo = _resolve_source()
    if is_demo:
        ui.demo_note()
    else:
        st.caption(f"Active dataset: {source_name}")

    # ensure a Machine ID column exists for aggregation
    if "Machine ID" not in df.columns:
        id_col = next((c for c in ["Product ID", "UDI"] if c in df.columns), None)
        df["Machine ID"] = df[id_col].astype(str) if id_col else df.index.astype(str)

    scored = prediction.predict_batch(bundle, df)
    snapshot = _latest_snapshot(scored)
    thr = bundle["thresholds"]

    tier_counts = snapshot["risk_tier"].value_counts().to_dict()
    total = len(snapshot)
    low = int(tier_counts.get("Low", 0))
    medium = int(tier_counts.get("Medium", 0))
    high = int(tier_counts.get("High", 0))
    alerts = int(snapshot["alert"].sum())

    # maintenance priority overview (deterministic rules, cheap to compute)
    recs = [recommendations.generate_recommendation(r, prediction.predict_row(bundle, r), bundle)
            for r in snapshot.head(40).to_dict("records")]
    prio = recommendations.recommendation_counts(recs)
    status = _overall_status(prio["P1"], prio["P2"], high)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total machines", total)
    c2.metric("Healthy (low risk)", low, delta=f"{low / total:.0%}" if total else None)
    c3.metric("Medium risk", medium, delta=f"{medium / total:.0%}" if total else None,
              delta_color="off")
    c4.metric("High risk", high, delta=f"{high / total:.0%}" if total else None,
              delta_color="off")
    c5.metric("Failure alerts", alerts,
              help=f"Machines with predicted probability >= alert threshold {thr['alert']:.2f} "
                   "(F1-optimal on validation data).")
    c6.metric("Overall status", status)

    st.divider()
    tab_risk, tab_trend, tab_ops, tab_alerts, tab_prio = st.tabs(
        ["Risk overview", "Operational trends", "Speed vs torque", "Recent alerts",
         "Maintenance priority"])

    with tab_risk:
        col1, col2 = st.columns([1, 1.6])
        with col1:
            st.markdown("**Machine health distribution**")
            st.plotly_chart(viz.fig_health_donut(tier_counts), width="stretch")
        with col2:
            st.markdown("**Risk by machine type**")
            st.plotly_chart(viz.fig_risk_by_type(snapshot), width="stretch")

    with tab_trend:
        if "Timestamp" in df.columns:
            overlay = st.multiselect(
                "Overlay individual machines (up to 3)",
                options=sorted(df["Machine ID"].unique()),
                default=[], max_selections=3)
            df = df.copy()
            df["temp_delta"] = (df["Process temperature [K]"]
                                - df["Air temperature [K]"])
            st.plotly_chart(
                viz.fig_trend(df, ["Torque [Nm]", "temp_delta"], overlay),
                width="stretch")
            st.plotly_chart(
                viz.fig_trend(df, ["Tool wear [min]"], overlay),
                width="stretch")
        else:
            st.info("The active dataset has no Timestamp column, so time trends "
                    "are unavailable. The demo fleet and template uploads include one.")

    with tab_ops:
        st.plotly_chart(viz.fig_scatter_ops(snapshot), width="stretch")
        st.caption("Each point is a machine's latest reading, coloured by the "
                   "model's risk tier. The inverse speed-torque relationship "
                   "matches constant spindle power operation.")

    with tab_alerts:
        flagged = snapshot[snapshot["risk_tier"] != "Low"].sort_values(
            "failure_probability", ascending=False)
        if flagged.empty:
            st.success("No machines above the medium-risk tier in the current snapshot.")
        else:
            cols = ["Machine ID", "Type", "failure_probability", "risk_tier",
                    "Torque [Nm]", "Tool wear [min]"]
            if "Timestamp" in flagged.columns:
                cols.insert(4, "Timestamp")
            show = viz.style_risk_table(flagged[cols]).rename(columns={
                "failure_probability": "Failure probability",
                "risk_tier": "Risk tier"})
            st.dataframe(
                show.head(10),
                width="stretch", hide_index=True,
                column_config={
                    "Failure probability": st.column_config.ProgressColumn(
                        format="%.3f", min_value=0.0, max_value=1.0),
                })
            st.caption(f"Top {min(10, len(flagged))} of {len(flagged)} flagged "
                       "machine(s), ranked by predicted failure probability.")

    with tab_prio:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("P1 Critical", prio["P1"])
        p2.metric("P2 High", prio["P2"])
        p3.metric("P3 Medium", prio["P3"])
        p4.metric("P4 Routine", prio["P4"])
        st.markdown("**Top maintenance priorities** (deterministic rule engine, "
                    "human approval required for any action):")
        top = [r for r in recs if r["priority"] in ("P1", "P2", "P3")]
        top.sort(key=lambda r: r["severity"], reverse=True)
        if top:
            for r in top[:5]:
                st.markdown(
                    f"- **{r['machine']}** - {r['priority']} - {r['priority_label']} "
                    f"| {r['summary']}")
        else:
            st.markdown("- Nothing above routine priority right now.")
        st.caption("Full detail per machine on the Maintenance Advisor page.")
