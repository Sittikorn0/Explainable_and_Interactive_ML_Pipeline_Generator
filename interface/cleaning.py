import streamlit as st
import pandas as pd
import os
from data_prepare.features.data_distribute import data_distribution
from data_prepare.features.loading_data import save_cleaned_data
from explainable.features.trace_log import commit_cleaning
from interface.cleaning_components.ui_cleaning_components import (
    render_profile_tab,
    render_drop_columns,
    render_duplicates,
    render_missing_values,
    render_outliers,
    render_summary,
    _fmt_pct,
)

def render_cleaning():
    # If app layout requires st.navigation (new streamlit) or custom routing
    # we might not need navigate() but using the existing one:
    from app import navigate, page_header

    page_header(
        "Data Cleaning",
        "กระบวนการลบหรือแก้ไขข้อมูลที่ผิดพลาด ไม่สมบูรณ์ หรือไม่มีความสอดคล้องกันจากชุดข้อมูล",
    )

    if st.session_state.get("main_df") is None:
        navigate("upload")
        return

    df = st.session_state["main_df"]
    file_name = st.session_state.get("last_uploaded_file", "Unknown File")

    st.info(f"**Current Dataset:** {file_name}")
    is_confirmed = st.session_state.get("cleaning_confirmed", False)
    with st.expander("Cleaned Data" if is_confirmed else "Raw Data"):
        st.dataframe(df, width="stretch")

    # ── working_df init ───────────────────────────────────────
    current_shape = df.shape
    if (
        "working_df" not in st.session_state
        or st.session_state.get("working_df_source_shape") != current_shape
    ):
        st.session_state["working_df"] = df.copy()
        st.session_state["working_df_source_shape"] = current_shape
        st.session_state["original_dup_count"] = int(df.duplicated().sum())
        st.session_state.pop("original_outlier_count", None)
        st.session_state.pop("_treated_outlier_cols", None)

    # ไม่ต้องสร้าง original_df ซ้ำซ้อนเพราะมี df (main_df) อยู่แล้ว ซึ่งประหยัด RAM ไป 1 ชุดเต็มๆ

    working_df = st.session_state["working_df"]

    # ── Dataset Overview ──────────────────────────────────────
    st.subheader("Dataset Overview")

    if "original_outlier_bounds" not in st.session_state:
        from data_prepare.features.statistics import get_outlier_bounds
        bounds_dict = {}
        for col in df.select_dtypes(include=["number"]).columns:
            series = df[col].dropna()
            if len(series) > 0:
                bounds_dict[col] = get_outlier_bounds(series)
        st.session_state["original_outlier_bounds"] = bounds_dict

    fixed_bounds = st.session_state["original_outlier_bounds"]

    _dist_key = ("_dist_cache", working_df.shape, int(pd.util.hash_pandas_object(working_df).sum()))
    if st.session_state.get("_dist_key") != _dist_key:
        with st.spinner("Calculating Data..."):
            total_outl, outls_details = data_distribution(working_df, fixed_bounds=fixed_bounds)
        st.session_state["_dist_key"] = _dist_key
        st.session_state["_dist_result"] = (total_outl, outls_details)
    else:
        total_outl, outls_details = st.session_state["_dist_result"]

    if "original_outlier_count" not in st.session_state:
        st.session_state["original_outlier_count"] = total_outl

    total_cells = working_df.size
    null_counts = working_df.isnull().sum()
    total_missing = int(null_counts.sum())
    missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0
    duplicate_count = int(working_df.duplicated().sum())
    dup_pct = (duplicate_count / working_df.shape[0] * 100) if working_df.shape[0] > 0 else 0
    outlier_pct = (total_outl / total_cells * 100) if total_cells > 0 else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", f"{working_df.shape[0]:,}")
    m2.metric("Columns", working_df.shape[1])
    m3.metric("Missing Values", _fmt_pct(total_missing, missing_pct))
    m4.metric("Duplicate Rows", _fmt_pct(duplicate_count, dup_pct))
    m5.metric("Outliers", _fmt_pct(total_outl, outlier_pct))

    tab1, tab2 = st.tabs(["Profile", "Cleaning"], width="stretch")

    # ── Tab 1: Profile ────────────────────────────────────────
    with tab1:
        render_profile_tab(working_df, outls_details)

    # ── Tab 2: Cleaning ───────────────────────────────────────
    with tab2:
        working_df = st.session_state["working_df"]
        target_col = st.session_state.get("target_col")

        # ── Drop Columns ──
        render_drop_columns(working_df, target_col)
        st.divider()

        # ── Duplicates ──
        render_duplicates(working_df, duplicate_count)
        st.divider()

        # ── Missing Values ──
        missing_cols = {col: int(count) for col, count in null_counts.items() if count > 0}
        render_missing_values(working_df, missing_cols, null_counts)
        st.divider()

        # ── Outliers ──
        outlier_cols = {
            item["Column"]: item for item in outls_details 
            if item["Outliers"] > 0 and pd.api.types.is_numeric_dtype(working_df.get(item["Column"], pd.Series()))
        }
        render_outliers(working_df, outlier_cols, outls_details)
        st.divider()

        # ── Summary ──
        dup_before = st.session_state.get("original_dup_count", int(df.duplicated().sum()))
        outl_before = st.session_state["original_outlier_count"]
        render_summary(working_df, df, dup_before, outl_before, total_missing, duplicate_count, total_outl)

        # ── Confirm / Reset ───────────────────────────────────
        _, cf1, cf2, _ = st.columns([2, 1.5, 1.5, 2])
        with cf1:
            if st.button("Confirm & Save", type="primary", width="stretch"):
                original_filename = st.session_state.get("last_uploaded_file", "dataset.csv")
                st.session_state["cleaning_summary_snapshot"] = {
                    "before": {
                        "rows": df.shape[0], "cols": df.shape[1],
                        "missing": int(df.isnull().sum().sum()), "dups": dup_before, "outliers": outl_before,
                    },
                    "after": {
                        "rows": working_df.shape[0], "cols": working_df.shape[1],
                        "missing": total_missing, "dups": duplicate_count, "outliers": total_outl,
                    },
                }
                save_cleaned_data(st.session_state["working_df"], original_filename)
                st.session_state["main_df"] = st.session_state["working_df"].copy()
                st.session_state["cleaning_confirmed"] = True
                commit_cleaning(df, st.session_state["working_df"])
                from explainable.features.pipeline_state import commit_step
                commit_step("cleaning", st.session_state["cleaning_summary_snapshot"])
                st.success("บันทึกข้อมูลที่ Cleaned แล้ว")
                st.rerun()
        with cf2:
            if st.button("Reset", type="secondary", width="stretch"):
                st.session_state["working_df"] = df.copy()
                st.session_state["cleaning_confirmed"] = False
                st.session_state.pop("_treated_outlier_cols", None)
                st.session_state.pop("cleaning_summary_snapshot", None)
                st.session_state.pop("original_outlier_bounds", None)
                st.info("Reset กลับ original data แล้ว")
                st.rerun()

        # ── Download (หลัง Confirm เท่านั้น) ──────────────────
        if st.session_state.get("cleaning_confirmed"):
            csv_path = st.session_state.get("cleaned_csv_path")
            if csv_path and os.path.exists(csv_path):
                with open(csv_path, "rb") as f:
                    st.download_button(
                        label=f"Download {st.session_state['last_uploaded_file']}",
                        data=f,
                        file_name=st.session_state["last_uploaded_file"],
                        mime="text/csv",
                        width="stretch",
                    )

    # ── Navigation ────────────────────────────────────────────
    st.divider()
    confirmed = st.session_state.get("cleaning_confirmed", False)
    if not confirmed:
        st.info("กรุณากด **Confirm & Save** ก่อนไปขั้นตอนถัดไป")
    col1, _, col2 = st.columns([0.8, 8, 0.8])
    with col1:
        if st.button("Back", type="secondary", width="stretch"):
            st.session_state.pop("working_df", None)
            st.session_state.pop("working_df_source_shape", None)
            st.session_state.pop("cleaning_confirmed", None)
            st.session_state.pop("_treated_outlier_cols", None)
            navigate("upload")
    with col2:
        if st.button(
            "Next Step",
            type="primary",
            width="stretch",
            disabled=not confirmed,
        ):
            navigate("eda")
