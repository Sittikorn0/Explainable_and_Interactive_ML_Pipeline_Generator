import streamlit as st
import pandas as pd
from data_prepare.logic.data_distribute import data_distribution
from data_prepare.logic.target_col import describe_target

from interface.eda_components.ui_profile import render_profile_tab
from interface.eda_components.ui_distributions import render_distributions_tab
from interface.eda_components.ui_relationships import render_relationships_tab

def format_percentage(count: int, percentage: float) -> str:
    if count == 0:
        return "0 (0.0%)"
    percentage_string = f"{percentage:.1f}%" if percentage >= 0.1 else "< 0.1%"
    return f"{count:,} ({percentage_string})"

def render_eda():
    from app import page_header

    page_header("Exploratory Data Analysis", "สำรวจและทำความเข้าใจข้อมูลก่อนสร้างโมเดล")

    if st.session_state.get("main_df") is None:
        from app import navigate
        navigate("upload")
        return

    has_working_dataframe = "working_df" in st.session_state
    dataframe = st.session_state["working_df"] if has_working_dataframe else st.session_state["main_df"]
    file_name = st.session_state.get("last_uploaded_file", "Unknown File")
    is_cleaned = has_working_dataframe and st.session_state.get("cleaning_confirmed")

    st.info(f"**Current Dataset:** {file_name}")
    with st.expander("Cleaned Data" if is_cleaned else "Raw Data"):
        st.dataframe(dataframe, width="stretch")

    # Dataset Overview
    st.subheader("Dataset Overview")
    
    from data_prepare.loading_data import load_outlier_bounds
    if "original_outlier_bounds" not in st.session_state:
        saved_bounds = load_outlier_bounds()
        if saved_bounds:
            st.session_state["original_outlier_bounds"] = saved_bounds
            
    fixed_outlier_bounds = st.session_state.get("original_outlier_bounds", None)

    bounds_signature = id(fixed_outlier_bounds) if fixed_outlier_bounds else 0
    distribution_cache_key = ("_eda_dist_cache", dataframe.shape, int(pd.util.hash_pandas_object(dataframe).sum()), bounds_signature)
    if st.session_state.get("_eda_dist_key") != distribution_cache_key:
        with st.spinner("Calculating Data..."):
            total_outlier, outlier_details = data_distribution(dataframe, fixed_bounds=fixed_outlier_bounds)
        st.session_state["_eda_dist_key"] = distribution_cache_key
        st.session_state["_eda_dist_result"] = (total_outlier, outlier_details)
    else:
        total_outlier, outlier_details = st.session_state["_eda_dist_result"]

    total_cells = dataframe.size
    total_missing_values = dataframe.isnull().sum().sum()
    missing_percentage = (total_missing_values / total_cells * 100) if total_cells > 0 else 0
    duplicate_rows_count = int(dataframe.duplicated().sum())
    duplicate_percentage = (duplicate_rows_count / dataframe.shape[0] * 100) if dataframe.shape[0] > 0 else 0
    outlier_percentage = (total_outlier / total_cells * 100) if total_cells > 0 else 0

    from interface.ui_helpers import render_metrics_row
    metrics_data = [
        ("Rows", f"{dataframe.shape[0]:,}"),
        ("Columns", str(dataframe.shape[1])),
        ("Missing Values", format_percentage(total_missing_values, missing_percentage)),
        ("Duplicate Rows", format_percentage(duplicate_rows_count, duplicate_percentage)),
        ("Outliers", format_percentage(total_outlier, outlier_percentage)),
    ]
    render_metrics_row(metrics_data)

    target_column = st.session_state.get("target_col", dataframe.columns[-1])
    if target_column not in dataframe.columns:
        target_column = dataframe.columns[-1]
        st.warning(f"Target column ที่เลือกไว้ไม่พบใน dataset — ใช้ **{target_column}** แทน")
    st.info(f"**Target Column:** {target_column}  \n{describe_target(dataframe, target_column)}")

    if not pd.api.types.is_numeric_dtype(dataframe[target_column]) and dataframe[target_column].nunique() <= 20:
        target_class_counts = dataframe[target_column].value_counts()
        minimum_count = int(target_class_counts.min())
        maximum_count = int(target_class_counts.max())
        if maximum_count > 3 * minimum_count:
            st.warning(
                f"**Class Imbalance ตรวจพบใน Target '{target_column}':** "
                f"ค่าที่พบมากสุดมากกว่าค่าที่พบน้อยสุดถึง **{maximum_count / minimum_count:.1f} เท่า** "
                "— อาจต้องจัดการก่อนสร้างโมเดล เช่น Oversampling, SMOTE, หรือปรับ class_weight"
            )

    tab_profile, tab_distributions, tab_relationships = st.tabs(
        ["Profile", "Distributions", "Relationships"], width="stretch"
    )

    with tab_profile:
        render_profile_tab(dataframe, target_column, outlier_details)

    with tab_distributions:
        render_distributions_tab(dataframe, target_column)

    with tab_relationships:
        render_relationships_tab(dataframe, target_column)

    # Page Navigation
    back_button_col, _, next_button_col = st.columns([1.2, 7.6, 1.2])
    with back_button_col:
        if st.button("Back", type="secondary", width="stretch", key="back"):
            from app import navigate
            navigate("cleaning")
    with next_button_col:
        if st.button("Next Step", type="primary", width="stretch"):
            from app import navigate
            navigate("transformation")