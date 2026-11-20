"""Page 7: About FactoryMind.

Product story, architecture, technology stack, dataset attribution, honest
limitations and roadmap. Team Sodium, International AI Builders Congress.

Team Sodium | International AI Builders Congress (hackathon MVP).
"""
from __future__ import annotations

import streamlit as st

from src import config
from src.prediction import load_bundle
from src import widgets as ui


def render() -> None:
    bundle = load_bundle()
    ui.page_header(
        "About FactoryMind",
        "A decision-support prototype for industrial maintenance, built for the "
        "International AI Builders Congress.",
    )

    st.markdown("### Product overview")
    st.markdown(
        "FactoryMind helps manufacturing teams understand machine health, "
        "anticipate failure risk and make better maintenance decisions. It "
        "ingests machine operational data (CSV), validates it, scores it with "
        "a trained Random Forest classifier, explains the drivers behind each "
        "prediction and turns all of it into traceable maintenance suggestions. "
        "It is a **decision-support prototype** - a human engineer approves "
        "every action; FactoryMind is not an autonomous industrial control "
        "system.")

    col_p, col_s = st.columns(2)
    with col_p:
        st.markdown("### Problem")
        st.markdown(
            "Unplanned machine failure stops production lines and consumes "
            "maintenance budgets. Small and mid-size factories rarely have the "
            "data teams or tooling to build predictive maintenance; their "
            "sensor data sits unused while technicians react only after "
            "something breaks.")
    with col_s:
        st.markdown("### Proposed solution")
        st.markdown(
            "A lightweight, offline-capable web application: upload machine "
            "readings, get an ML-assessed risk score, see which indicators "
            "drive it, and receive a priority-ranked, evidence-based inspection "
            "list. Every number is traceable to a documented threshold and "
            "every action requires human sign-off.")

    st.markdown("### AI architecture")
    st.code(
        "CSV upload  -->  schema validation (tolerant headers, row-level checks)\n"
        "            -->  preprocessing (median/mode imputation, one-hot Type)\n"
        "            -->  Random Forest (400 trees, class-weighted, seed 42)\n"
        "            -->  failure probability\n"
        "            -->  validation-informed thresholds (alert / medium / high)\n"
        "            -->  explanation layer\n"
        "                 |  global: permutation importance (test split)\n"
        "                 |  local:  one-at-a-time sensitivity vs healthy baseline\n"
        "            -->  deterministic rule engine (9 traceable rules)\n"
        "            -->  priority-ranked advisory + human approval gate",
        language="text")

    st.markdown("### Technology stack")
    st.markdown(
        "| Layer | Technology |\n|---|---|\n"
        "| Interface | Streamlit (single-page process, sidebar routing) |\n"
        "| Data processing | pandas, NumPy |\n"
        "| Machine learning | scikit-learn (RandomForest, metrics, permutation importance) |\n"
        "| Visualisation | Plotly (dark industrial theme) |\n"
        "| Persistence | joblib model bundle + metrics JSON |\n"
        "| Language | Python 3.11+ |")

    st.markdown("### Dataset attribution")
    st.markdown(
        f"- Dataset: [UCI AI4I 2020 Predictive Maintenance]("
        "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)\n"
        f"- License: **CC BY 4.0** | DOI: 10.24432/C5HS5C\n"
        f"- Citation: {config.DATASET_CITATION}\n"
        "- The dataset is synthetic; a clearly-labelled synthetic fallback "
        "generator is included in case the committed file is missing, and the "
        "dashboard fleet (M-001...M-012) is generated demonstration data - "
        "never presented as real telemetry.")

    st.markdown("### Current prototype limitations")
    st.markdown(
        "- Trained and evaluated on a synthetic dataset; metrics are not proof "
        "of real-world performance.\n"
        "- Model outputs are ranking scores, not calibrated probabilities.\n"
        "- No streaming/live sensor integration; CSV and form inputs only.\n"
        "- Explanations are sensitivity analyses on correlated features, not "
        "causal attribution.\n"
        "- No user accounts, audit logs or CMMS integration in this MVP.")

    st.markdown("### Future roadmap")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("**Near-term**\n- Calibration + threshold tuning per line\n"
                    "- Live sensor connectors (OPC UA / MQTT)\n- User roles and audit trail")
    with r2:
        st.markdown("**Mid-term**\n- SHAP-based local explanations\n"
                    "- Remaining useful life estimation\n- CMMS (work-order) integration")
    with r3:
        st.markdown("**Long-term**\n- Fleet-level analytics and benchmarking\n"
                    "- Validation framework for real equipment\n- On-prem deployment option")

    st.divider()
    st.markdown("### Team Sodium")
    st.markdown(
        "**Team Sodium** - International AI Builders Congress.\n\n"
        "Built as a hackathon MVP: end-to-end ML pipeline, explainability, "
        "deterministic advisory layer and a judge-friendly dashboard in one "
        "Streamlit application. Model bundle trained on this machine, saved to "
        f"`models/`, evaluated at {bundle['trained_at']} "
        f"(test ROC-AUC {bundle['metrics_test']['roc_auc']:.2f} - full numbers "
        "on the Model Performance page).")
