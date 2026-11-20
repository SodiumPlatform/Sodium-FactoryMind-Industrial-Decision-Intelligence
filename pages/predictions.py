"""Page 3: AI Failure Prediction.

Real predictions from the trained Random Forest pipeline: single-machine form
plus batch scoring. Risk tiers use validation-derived thresholds; the model
info panel discloses evaluation numbers, provenance and limitations.

Team Sodium | International AI Builders Congress (hackathon MVP).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config, data_loader, prediction, recommendations
from src.prediction import load_bundle
from src import visualization as viz
from src import widgets as ui


def _render_single(bundle: dict) -> None:
    st.markdown("**Single machine** - enter operational values and run the "
                "trained model. Defaults are typical healthy medians from "
                "training data.")

    inputs = ui.machine_input_form(bundle, ui.healthy_defaults(bundle), key="pred")

    if st.button("Run prediction", type="primary"):
        with st.spinner("Scoring with the trained Random Forest ..."):
            pred = prediction.predict_row(bundle, inputs)
            base, contrib = prediction.local_contributions(bundle, inputs)
            st.session_state["last_input"] = inputs
            st.session_state["last_pred"] = pred
            st.session_state["last_contrib"] = contrib

    pred = st.session_state.get("last_pred")
    if pred is None:
        st.info("Submit the form to receive the model's prediction, risk tier "
                "and contributing factors.")
        return

    inputs = st.session_state.get("last_input", inputs)
    st.divider()
    col_g, col_r = st.columns([1.2, 1.8])
    with col_g:
        st.plotly_chart(viz.fig_gauge(pred["probability"], bundle["thresholds"]),
                        width="stretch")
    with col_r:
        st.markdown(
            f"<div style='margin-top:18px'>{ui.risk_badge(pred['tier'])}</div>",
            unsafe_allow_html=True)
        st.markdown(
            f"### Predicted failure probability: **{pred['probability']:.2f}**")
        if pred["alert"]:
            st.error(f"This score is at or above the alert threshold "
                     f"({bundle['thresholds']['alert']:.2f}, F1-optimal on "
                     "validation data) - route to the Maintenance Advisor and "
                     "a human engineer.")
        else:
            st.success(f"Below the alert threshold "
                       f"({bundle['thresholds']['alert']:.2f}). Tiers: "
                       f"Medium >= {bundle['thresholds']['medium']:.2f}, "
                       f"High >= {bundle['thresholds']['high']:.2f} "
                       "(validation-derived).")
        st.caption("The score is a Random Forest vote share - a ranking signal, "
                   "not a calibrated probability.")
        if st.button("Open this machine in the Maintenance Advisor",
                     width="stretch"):
            st.session_state["advisor_source"] = "last_prediction"

    st.markdown("**Contributing operational factors** (one-at-a-time sensitivity "
                "analysis - correlational, not causal):")
    st.plotly_chart(viz.fig_contributions(st.session_state.get("last_contrib", [])),
                    width="stretch")
    st.dataframe(prediction.value_context(bundle, inputs),
                 width="stretch", hide_index=True)

    with st.expander("Model information, provenance and limitations"):
        st.markdown(ui.metrics_mini_table(bundle["metrics_test"]))
        st.caption(prediction.model_summary_text(bundle))
        st.caption("Trained on the UCI AI4I 2020 dataset, which is SYNTHETIC. "
                   "Scores are not proof of real-world factory performance. "
                   "Full evaluation on the Model Performance page.")


def _batch_sources(bundle: dict) -> list:
    sources = []
    if st.session_state.get("validated_df") is not None:
        sources.append(("Validated upload", st.session_state["validated_df"].copy()))
    sources.append(("Demonstration fleet (sample data)",
                    data_loader.load_sample_fleet()))
    raw, _ = data_loader.load_ai4i()
    sources.append(("Committed AI4I dataset (first 500 rows)", raw.head(500).copy()))
    return sources


def _render_batch(bundle: dict) -> None:
    st.markdown("**Batch scoring** - run the model over an entire dataset and "
                "export the scored result.")
    sources = _batch_sources(bundle)
    labels = [s[0] for s in sources]
    choice = st.selectbox("Data source", labels)
    df = sources[labels.index(choice)][1]

    if st.button("Run batch scoring", type="primary"):
        with st.spinner(f"Scoring {len(df):,} rows ..."):
            scored = prediction.predict_batch(bundle, df)
            st.session_state["batch_scored"] = scored
            st.session_state["batch_source"] = choice

    scored = st.session_state.get("batch_scored")
    if scored is None:
        st.info("Choose a source and run the batch scorer. The demonstration "
                "fleet option is clearly labelled sample data.")
        return

    if st.session_state.get("batch_source") == labels[0] and \
            st.session_state.get("validated_df") is not None:
        pass  # upload already stored

    counts = scored["risk_tier"].value_counts().to_dict()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(viz.fig_health_donut(counts), width="stretch")
    with c2:
        st.metric("Rows scored", len(scored))
        st.metric("Alerts (>= alert threshold)",
                  int(scored["alert"].sum()),
                  help=f"Threshold {bundle['thresholds']['alert']:.2f} (F1-optimal on validation).")

    show = viz.style_risk_table(scored).rename(columns={
        "failure_probability": "Failure probability", "risk_tier": "Risk tier"})
    st.dataframe(
        show.head(200),
        width="stretch", hide_index=True,
        column_config={
            "Failure probability": st.column_config.ProgressColumn(
                format="%.3f", min_value=0.0, max_value=1.0),
        })
    if len(scored) > 200:
        st.caption("Showing the first 200 rows - use the download for the full set.")

    fname = "factorymind_scored_batch.csv"
    st.download_button("Download scored CSV",
                       data=scored.to_csv(index=False).encode("utf-8"),
                       file_name=fname, mime="text/csv")

    # optional evaluation when ground-truth labels are present
    if config.TARGET in scored.columns:
        y = pd.to_numeric(scored[config.TARGET], errors="coerce")
        valid = y.isin([0, 1])
        if valid.sum() > 0:
            st.markdown("**Agreement with provided labels (evaluation only - "
                        "labels were NOT used as model input):**")
            yv = y[valid].astype(int)
            pv = (scored.loc[valid, "alert"]).astype(int)
            acc = float((yv == pv).mean())
            tp = int(((yv == 1) & (pv == 1)).sum())
            fp = int(((yv == 0) & (pv == 1)).sum())
            fn = int(((yv == 1) & (pv == 0)).sum())
            st.markdown(
                f"- Alert-threshold accuracy on labelled rows: **{acc:.2f}** "
                f"({valid.sum():,} labelled rows)\n"
                f"- Alerts: {tp} true positives, {fp} false positives, "
                f"{fn} missed failures")
            st.caption("Full evaluation on the held-out test split is on the "
                       "Model Performance page.")


def render() -> None:
    bundle = load_bundle()
    ui.page_header(
        "AI Failure Prediction",
        "Score machine health with the trained Random Forest. Risk tiers come "
        "from validation-derived thresholds, not hand-picked constants.",
    )
    tab_single, tab_batch = st.tabs(["Single machine", "Batch scoring"])
    with tab_single:
        _render_single(bundle)
    with tab_batch:
        _render_batch(bundle)
