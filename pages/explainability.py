"""Page 4: Explainable AI.

Global model behaviour (permutation importance on the held-out test split) and
per-prediction local explanations (one-at-a-time sensitivity analysis).
Correlation is explicitly distinguished from causation throughout.

Team Sodium | International AI Builders Congress (hackathon MVP).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import prediction
from src.prediction import load_bundle
from src import visualization as viz
from src import widgets as ui


def render() -> None:
    bundle = load_bundle()
    ui.page_header(
        "Explainable AI",
        "Which indicators drive the model globally, and which readings drove "
        "one specific prediction? Both are sensitivity analyses on synthetic "
        "data - associations, not causal effects.",
    )

    # ---------------- global importance ----------------
    st.subheader("Global feature importance (permutation, held-out test split)")
    imp = bundle["permutation_importance"]
    std = bundle["permutation_importance_std"]
    st.plotly_chart(viz.fig_importance(imp, std), use_container_width=True)
    imp_df = (pd.DataFrame({"Feature": list(imp.keys()),
                            "Mean importance": [round(v, 4) for v in imp.values()],
                            "Std": [round(std[k], 4) for k in imp.keys()]})
              .sort_values("Mean importance", ascending=False))
    st.dataframe(imp_df, use_container_width=True, hide_index=True)

    with st.expander("How to read permutation importance"):
        st.markdown(
            "Each feature's values are randomly shuffled on the held-out test "
            "split (8 repeats); the resulting degradation in log-loss measures "
            "how much the model relies on that feature. **This describes the "
            "model, not the physics**: importance reflects associations learned "
            "from the synthetic AI4I dataset and may differ on real equipment.")

    # ---------------- local explanation ----------------
    st.divider()
    st.subheader("Individual prediction explanation")

    last = st.session_state.get("last_pred")
    if last is None:
        st.info("No prediction yet in this session - run one on the AI Failure "
                "Prediction page, or enter values here.")
        inputs = ui.machine_input_form(bundle, ui.healthy_defaults(bundle), key="exp")
        if st.button("Explain this machine", type="primary"):
            pred = prediction.predict_row(bundle, inputs)
            _, contrib = prediction.local_contributions(bundle, inputs)
            st.session_state["last_input"] = inputs
            st.session_state["last_pred"] = pred
            st.session_state["last_contrib"] = contrib
            last = pred
        else:
            return
    else:
        inputs = st.session_state.get("last_input")

    pred = st.session_state["last_pred"]
    contrib = st.session_state.get("last_contrib", [])
    st.markdown(
        f"Explaining the last prediction: {ui.risk_badge(pred['tier'])} "
        f"**p = {pred['probability']:.2f}** "
        f"(alert threshold {bundle['thresholds']['alert']:.2f}).",
        unsafe_allow_html=True)

    c1, c2 = st.columns([1.4, 1.6])
    with c1:
        st.plotly_chart(viz.fig_contributions(contrib), use_container_width=True)
    with c2:
        st.markdown("**Reading context vs training population**")
        st.dataframe(prediction.value_context(bundle, inputs),
                     use_container_width=True, hide_index=True)

    if contrib:
        top = contrib[0]
        direction = "raise" if top["delta_if_normalised"] < 0 else "lower"
        st.markdown(
            f"**Plain-language reading:** if the machine's "
            f"**{top['feature']}** were set to the typical healthy value "
            f"({top['healthy_typical']}), the predicted risk would "
            f"**{direction}** by about "
            f"**{abs(top['delta_if_normalised']) * 100:.1f} percentage points**. "
            "This is a one-at-a-time sensitivity check on the current model - "
            "it does not prove the reading caused the risk level.")

    with st.expander("Methodology and limitations"):
        st.markdown(
            "- **Global importance:** permutation importance (mean decrease in "
            "log-loss, 8 repeats, test split) - shows what the model relies on "
            "overall.\n"
            "- **Individual explanation:** one-at-a-time (OCAT) sensitivity - "
            "each feature is swapped to its typical healthy value while the "
            "others stay fixed; the change in predicted risk is reported.\n"
            "- OCAT cannot detect interaction effects; correlated features "
            "(speed and torque) share credit.\n"
            "- Both methods describe the MODEL's behaviour, not physical "
            "causation, and the training data is SYNTHETIC.")
