"""Page 6: Model Performance.

Real evaluation numbers only - every figure here is computed from the held-out
test split of the committed dataset and persisted in the model bundle.

Team Sodium | International AI Builders Congress (hackathon MVP).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import config, model as src_model
from src.prediction import load_bundle
from src import visualization as viz
from src import widgets as ui


def render() -> None:
    bundle = load_bundle()
    ui.page_header(
        "Model Performance",
        "Evaluation of the trained Random Forest on a held-out test split it "
        "never saw during training or threshold tuning.",
    )

    prov = bundle["data_provenance"]
    split = bundle["split_info"]

    # ---------------- dataset & provenance ----------------
    st.info(
        f"**Training data:** {prov['source']} - {prov['rows']:,} rows, 3.4% "
        f"failure rate. License: {prov.get('license', '-')}. "
        "The AI4I 2020 dataset is SYNTHETIC: results demonstrate the pipeline, "
        "not real-world factory performance.")

    st.markdown("**Splits (stratified, seed 42)**")
    st.dataframe(pd.DataFrame([
        {"Split": "Train (fit)", "Rows": split["train_n"],
         "Failures": split["train_pos"],
         "Failure rate": f"{split['train_pos'] / split['train_n']:.2%}"},
        {"Split": "Validation (thresholds)", "Rows": split["val_n"],
         "Failures": split["val_pos"],
         "Failure rate": f"{split['val_pos'] / split['val_n']:.2%}"},
        {"Split": "Test (reporting only)", "Rows": split["test_n"],
         "Failures": split["test_pos"],
         "Failure rate": f"{split['test_pos'] / split['test_n']:.2%}"},
    ]), use_container_width=True, hide_index=True)

    mt = bundle["metrics_test"]
    mv = bundle["metrics_val"]

    # ---------------- headline metrics ----------------
    st.divider()
    st.subheader(f"Test-set metrics at alert threshold {mt['threshold']:.2f}")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("ROC-AUC", f"{mt['roc_auc']:.3f}",
              help="Threshold-free ranking quality.")
    m2.metric("PR-AUC", f"{mt['average_precision']:.3f}",
              help="Average precision - informative under class imbalance.")
    m3.metric("Precision", f"{mt['precision']:.3f}")
    m4.metric("Recall", f"{mt['recall']:.3f}")
    m5.metric("F1-score", f"{mt['f1']:.3f}")
    m6.metric("Balanced accuracy", f"{mt['balanced_accuracy']:.3f}")
    st.caption(f"Chance-level PR-AUC for this split is "
               f"{mt['support']['positives'] / mt['support']['n']:.3f} "
               "(the positive class rate). Validation-split F1 at the same "
               f"threshold: {mv['f1']:.3f}.")

    # ---------------- curves & matrix ----------------
    X_te = bundle["X_test"]
    y_te = bundle["y_test"]
    proba = bundle["pipeline"].predict_proba(X_te)[:, 1]

    tab_cm, tab_roc, tab_pr, tab_report = st.tabs(
        ["Confusion matrix", "ROC curve", "Precision-Recall", "Classification report"])
    with tab_cm:
        c1, c2 = st.columns([1, 1.6])
        with c1:
            st.plotly_chart(viz.fig_confusion(mt["confusion_matrix"]),
                            use_container_width=True)
        with c2:
            cm = mt["confusion_matrix"]
            st.markdown(
                f"- True negatives: **{cm[0][0]}** - healthy machines correctly cleared\n"
                f"- False positives: **{cm[0][1]}** - healthy machines flagged "
                "(cost: unnecessary inspection)\n"
                f"- False negatives: **{cm[1][0]}** - missed failures "
                "(cost: unplanned downtime)\n"
                f"- True positives: **{cm[1][1]}** - failures correctly caught")
            st.caption("Threshold choice trades precision against recall; the "
                       "operating point is documented below.")
    with tab_roc:
        st.plotly_chart(viz.fig_roc(y_te, proba), use_container_width=True)
    with tab_pr:
        st.plotly_chart(viz.fig_pr_curve(y_te, proba), use_container_width=True)
    with tab_report:
        rep = pd.DataFrame(mt["classification_report"]).T
        st.dataframe(rep.round(3), use_container_width=True)

    # ---------------- thresholds ----------------
    st.divider()
    st.subheader("Operating thresholds (validation-derived, not hand-picked)")
    thr = bundle["thresholds"]
    st.dataframe(pd.DataFrame([
        {"Threshold": "Alert (decision)", "Value": thr["alert"],
         "Derivation": "Maximises F1 on the validation split"},
        {"Threshold": "Medium risk tier", "Value": thr["medium"],
         "Derivation": "80th percentile of validation predicted scores"},
        {"Threshold": "High risk tier", "Value": thr["high"],
         "Derivation": "95th percentile of validation predicted scores"},
    ]), use_container_width=True, hide_index=True)
    st.caption(thr["derivation"])

    # ---------------- importance ----------------
    st.subheader("Feature importance (global)")
    st.plotly_chart(viz.fig_importance(bundle["permutation_importance"],
                                       bundle["permutation_importance_std"]),
                    use_container_width=True)
    st.caption("Full interpretation on the Explainable AI page.")

    # ---------------- model card ----------------
    c1, c2 = st.columns([1.4, 1])
    with c1:
        with st.expander("Model card summary"):
            st.markdown(
                f"- Algorithm: **{bundle['model_name']}** with params "
                f"`{bundle['params']}`\n"
                f"- Preprocessing: median imputation (numeric), most-frequent "
                f"imputation + one-hot encoding (Type)\n"
                f"- Class imbalance: `class_weight=balanced_subsample`\n"
                f"- Features: {', '.join(config.FEATURES)}\n"
                f"- Excluded (leakage): {', '.join(config.LEAKAGE_COLUMNS)}\n"
                f"- scikit-learn: {bundle['sklearn_version']} | trained: "
                f"{bundle['trained_at']}")
    with c2:
        if st.button("Retrain model from committed dataset"):
            with st.spinner("Retraining (fit, thresholds, permutation importance) ..."):
                st.cache_resource.clear()
                src_model.train_and_save()
            st.success("Model retrained and saved. Reloading pages.")
            st.rerun()

    # ---------------- limitations ----------------
    st.divider()
    st.subheader("Limitations and dataset provenance")
    st.markdown(
        "- The UCI AI4I 2020 dataset is **synthetic**; metrics demonstrate the "
        "pipeline, not real factory performance.\n"
        "- Outputs are Random Forest vote shares - **not calibrated "
        "probabilities** (no calibration step was implemented or evaluated).\n"
        "- One-hot speed/torque are strongly anti-correlated; permutation "
        "importance splits their credit.\n"
        "- The test split is small in positives; precision/recall carry "
        "sampling uncertainty.\n"
        "- Thresholds were tuned on a validation split of the same synthetic "
        "distribution; re-validate before production use on real equipment.\n"
        "- Data provenance, license and citation: see docs/data_provenance.md.")
