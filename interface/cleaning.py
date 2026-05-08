import streamlit as st
import pandas as pd
import os
from data_prepare.logic.data_distribute import data_distribution
from data_prepare.loading_data import save_cleaned_data
from explainable.state_manager.trace_log import commit_cleaning
from interface.cleaning_components.ui_cleaning_components import (
    render_profile_tab,
    render_drop_columns,
    render_duplicates,
    render_missing_values,
    render_outliers,
    render_summary,
    format_percentage,
)

def render_cleaning():
    from app import navigate, page_header

    page_header(
        "Data Cleaning",
        "กระบวนการลบหรือแก้ไขข้อมูลที่ผิดพลาด ไม่สมบูรณ์ หรือไม่มีความสอดคล้องกันจากชุดข้อมูล",
    )

    if st.session_state.get("main_df") is None:
        navigate("upload")
        return

    dataframe = st.session_state["main_df"]
    file_name = st.session_state.get("last_uploaded_file", "Unknown File")

    st.info(f"**Current Dataset:** {file_name}")
    is_confirmed = st.session_state.get("cleaning_confirmed", False)
    with st.expander("Cleaned Data" if is_confirmed else "Raw Data"):
        st.dataframe(dataframe, width="stretch")

    # working_df initialization
    current_shape = dataframe.shape
    if (
        "working_df" not in st.session_state
        or st.session_state.get("working_df_source_shape") != current_shape
    ):
        st.session_state["working_df"] = dataframe.copy()
        st.session_state["working_df_source_shape"] = current_shape
        st.session_state["original_dup_count"] = int(dataframe.duplicated().sum())
        st.session_state.pop("original_outlier_count", None)
        st.session_state.pop("_treated_outlier_cols", None)

    working_dataframe = st.session_state["working_df"]

    # Dataset Overview
    st.subheader("Dataset Overview")

    if "original_outlier_bounds" not in st.session_state:
        from data_prepare.logic.statistics import get_outlier_bounds
        bounds_dictionary = {}
        for column in dataframe.select_dtypes(include=["number"]).columns:
            series = dataframe[column].dropna()
            if len(series) > 0:
                bounds_dictionary[column] = get_outlier_bounds(series)
        st.session_state["original_outlier_bounds"] = bounds_dictionary

    fixed_outlier_bounds = st.session_state["original_outlier_bounds"]

    distribution_cache_key = ("_dist_cache", working_dataframe.shape, int(pd.util.hash_pandas_object(working_dataframe).sum()))
    if st.session_state.get("_dist_key") != distribution_cache_key:
        with st.spinner("Calculating Data..."):
            total_outlier, outlier_details = data_distribution(working_dataframe, fixed_bounds=fixed_outlier_bounds)
        st.session_state["_dist_key"] = distribution_cache_key
        st.session_state["_dist_result"] = (total_outlier, outlier_details)
    else:
        total_outlier, outlier_details = st.session_state["_dist_result"]

    if "original_outlier_count" not in st.session_state:
        st.session_state["original_outlier_count"] = total_outlier

    total_cells_count = working_dataframe.size
    null_counts_series = working_dataframe.isnull().sum()
    total_missing_values = int(null_counts_series.sum())
    missing_percentage = (total_missing_values / total_cells_count * 100) if total_cells_count > 0 else 0
    duplicate_rows_count = int(working_dataframe.duplicated().sum())
    duplicate_percentage = (duplicate_rows_count / working_dataframe.shape[0] * 100) if working_dataframe.shape[0] > 0 else 0
    outlier_percentage = (total_outlier / total_cells_count * 100) if total_cells_count > 0 else 0

    from interface.ui_helpers import render_metrics_row
    metrics_data = [
        ("Rows", f"{working_dataframe.shape[0]:,}"),
        ("Columns", str(working_dataframe.shape[1])),
        ("Missing Values", format_percentage(total_missing_values, missing_percentage)),
        ("Duplicate Rows", format_percentage(duplicate_rows_count, duplicate_percentage)),
        ("Outliers", format_percentage(total_outlier, outlier_percentage)),
    ]
    render_metrics_row(metrics_data)

    tab_profile, tab_cleaning = st.tabs(["Profile", "Cleaning"])

    # Tab: Profile
    with tab_profile:
        render_profile_tab(working_dataframe, outlier_details)

    # Tab: Cleaning
    with tab_cleaning:
        working_dataframe = st.session_state["working_df"]
        target_column = st.session_state.get("target_col")

        render_drop_columns(working_dataframe, target_column)
        st.divider()

        render_duplicates(working_dataframe, duplicate_rows_count)
        st.divider()

        missing_columns_dict = {col: int(count) for col, count in null_counts_series.items() if count > 0}
        render_missing_values(working_dataframe, missing_columns_dict, null_counts_series)
        st.divider()

        outlier_columns_dict = {
            item["Column"]: item for item in outlier_details 
            if item["Outliers"] > 0 and pd.api.types.is_numeric_dtype(working_dataframe.get(item["Column"], pd.Series()))
        }
        render_outliers(working_dataframe, outlier_columns_dict, outlier_details)
        st.divider()

        duplicate_before = st.session_state.get("original_dup_count", int(dataframe.duplicated().sum()))
        outlier_before = st.session_state["original_outlier_count"]
        render_summary(working_dataframe, dataframe, duplicate_before, outlier_before, total_missing_values, duplicate_rows_count, total_outlier)

        # Confirm / Reset
        _, confirm_col1, confirm_col2, _ = st.columns([2, 1.5, 1.5, 2])
        with confirm_col1:
            if st.button("Confirm & Save", type="primary", width="stretch"):
                original_filename = st.session_state.get("last_uploaded_file", "dataset.csv")
                st.session_state["cleaning_summary_snapshot"] = {
                    "before": {
                        "rows": dataframe.shape[0], "cols": dataframe.shape[1],
                        "missing": int(dataframe.isnull().sum().sum()), "dups": duplicate_before, "outliers": outlier_before,
                    },
                    "after": {
                        "rows": working_dataframe.shape[0], "cols": working_dataframe.shape[1],
                        "missing": total_missing_values, "dups": duplicate_rows_count, "outliers": total_outlier,
                    },
                }
                save_cleaned_data(st.session_state["working_df"], original_filename)
                st.session_state["main_df"] = st.session_state["working_df"].copy()
                st.session_state["cleaning_confirmed"] = True
                commit_cleaning(dataframe, st.session_state["working_df"])
                
                from explainable.state_manager.pipeline_state import commit_step
                commit_step("cleaning", st.session_state["cleaning_summary_snapshot"])
                st.success("บันทึกข้อมูลที่ Cleaned แล้ว")
                st.rerun()
        with confirm_col2:
            if st.button("Reset", type="secondary", width="stretch"):
                st.session_state["working_df"] = dataframe.copy()
                st.session_state["cleaning_confirmed"] = False
                st.session_state.pop("_treated_outlier_cols", None)
                st.session_state.pop("cleaning_summary_snapshot", None)
                st.session_state.pop("original_outlier_bounds", None)
                st.info("Reset กลับ original data แล้ว")
                st.rerun()

        # Download
        if st.session_state.get("cleaning_confirmed"):
            csv_path = st.session_state.get("cleaned_csv_path")
            if csv_path and os.path.exists(csv_path):
                with open(csv_path, "rb") as file_obj:
                    st.download_button(
                        label=f"Download {st.session_state['last_uploaded_file']}",
                        data=file_obj,
                        file_name=st.session_state["last_uploaded_file"],
                        mime="text/csv",
                        width="stretch",
                    )

    # Navigation
    st.divider()
    is_confirmed = st.session_state.get("cleaning_confirmed", False)
    if not is_confirmed:
        st.info("กรุณากด **Confirm & Save** ก่อนไปขั้นตอนถัดไป")
        
    back_col, _, next_col = st.columns([1.2, 7.6, 1.2])
    with back_col:
        if st.button("Back", type="secondary", width="stretch"):
            st.session_state.pop("working_df", None)
            st.session_state.pop("working_df_source_shape", None)
            st.session_state.pop("cleaning_confirmed", None)
            st.session_state.pop("_treated_outlier_cols", None)
            navigate("upload")
    with next_col:
        if st.button(
            "Next Step",
            type="primary",
            width="stretch",
            disabled=not is_confirmed,
        ):
            navigate("eda")
