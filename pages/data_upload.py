"""Page 2: Machine Data Upload.

Upload, validate, preview and export CSV files against the canonical FactoryMind
schema. Validation results are stored in session state and become the active
dataset for the dashboard, batch prediction and advisor pages.

Team Sodium | International AI Builders Congress (hackathon MVP).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import data_loader
from src import widgets as ui

_REQUIRED_HINT = ("Type, Air temperature [K], Process temperature [K], "
                  "Rotational speed [rpm], Torque [Nm], Tool wear [min]")


def render() -> None:
    ui.page_header(
        "Machine Data Upload",
        "Bring your own machine readings (CSV). Files are validated against the "
        "canonical schema; only rows that cannot be scored are removed, and "
        "nothing is dropped silently.",
    )

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        st.download_button(
            "Download sample CSV template",
            data=data_loader.sample_template_bytes(),
            file_name="factorymind_template.csv",
            mime="text/csv",
            width="stretch",
        )
    with action_col2:
        if st.button("Load demonstration fleet as active dataset",
                     width="stretch"):
            demo = data_loader.load_sample_fleet()
            st.session_state["validated_df"] = demo
            st.session_state["validated_name"] = "Demonstration fleet (sample_data.csv)"
            st.success("Demo fleet loaded and validated - it is now the active "
                       "dataset for the dashboard and batch prediction.")

    st.divider()
    uploaded = st.file_uploader(
        "Upload a machine readings CSV",
        type=["csv"],
        help=f"Required columns: {_REQUIRED_HINT}. Optional: Machine ID, "
             "Timestamp, Machine failure (used for evaluation only, never as "
             "model input).",
    )

    if uploaded is None:
        st.info("Awaiting a CSV file. The demo fleet is used everywhere until "
                "you upload and validate a dataset.")
        _current_dataset_panel()
        return

    # --- read ------------------------------------------------------------
    try:
        raw = data_loader.read_csv_flexible(uploaded)
    except Exception as exc:  # noqa: BLE001 - surfaced as a friendly message
        st.error(f"Could not parse the file as CSV. Details: {exc}")
        return

    if raw.empty:
        st.error("The file contains a header but no data rows.")
        return

    # --- validate ----------------------------------------------------------
    result = data_loader.validate_dataframe(raw)

    st.subheader("Schema check")
    st.dataframe(pd.DataFrame(result.column_report),
                 width="stretch", hide_index=True)

    if result.errors:
        st.error("**Validation failed**\n" + "\n".join(f"- {e}" for e in result.errors))
        _current_dataset_panel()
        return

    if result.warnings:
        st.warning("**Validation passed with warnings**\n"
                   + "\n".join(f"- {w}" for w in result.warnings))

    # --- preview -----------------------------------------------------------
    st.subheader("Data preview (first 10 valid rows)")
    preview = result.cleaned.head(10)
    st.dataframe(preview, width="stretch", hide_index=True)

    miss = result.cleaned[list(set(raw.columns) & set(result.cleaned.columns))] \
        .isna().sum()
    miss = miss[miss > 0]
    if not miss.empty:
        st.markdown("**Remaining missing values per column**")
        st.dataframe(miss.rename("missing_count").to_frame(),
                     width="stretch")

    if result.invalid_rows is not None and len(result.invalid_rows) > 0:
        st.markdown(f"**Invalid or unscoreable rows: {len(result.invalid_rows)}**")
        st.caption("These rows were excluded (missing/non-numeric readings or "
                   "unknown machine Type). Source row numbers include the header.")
        st.dataframe(result.invalid_rows.head(20),
                     width="stretch", hide_index=False)

    # --- persist -------------------------------------------------------------
    st.session_state["validated_df"] = result.cleaned
    st.session_state["validated_name"] = f"Uploaded file: {uploaded.name}"
    st.success(
        f"Validation passed - {result.n_valid_rows:,} of {result.n_input_rows:,} "
        "rows accepted. This dataset is now active across the app.")

    col_a, col_b, _ = st.columns([1, 1, 1.4])
    with col_a:
        st.download_button(
            "Download validated dataset",
            data=result.cleaned.to_csv(index=False).encode("utf-8"),
            file_name=f"validated_{uploaded.name}",
            mime="text/csv",
            width="stretch",
        )
    with col_b:
        if st.button("Clear uploaded dataset", width="stretch"):
            st.session_state.pop("validated_df", None)
            st.session_state.pop("validated_name", None)
            st.rerun()

    _current_dataset_panel()


def _current_dataset_panel() -> None:
    name = st.session_state.get("validated_name")
    df = st.session_state.get("validated_df")
    if df is not None:
        st.markdown(f"**Active dataset:** {name} - {len(df):,} rows, "
                    f"{df['Type'].nunique()} machine type(s).")
    else:
        st.markdown("**Active dataset:** demonstration fleet "
                    "(clearly labelled sample data).")
