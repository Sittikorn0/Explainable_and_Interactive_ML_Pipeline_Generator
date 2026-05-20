# Libraries
import streamlit as st
import pandas as pd
import os

# Logic Import
from main import navigate
from backend.core.cleaning.data_distribution import *
from backend.core.cleaning.statistic import *
from backend.core.cleaning.main_logic import *
from backend.core.session.state import load_outlier_bounds, save_outlier_bounds, save_cleaned_data
from backend.core.session.pipeline_state import commit_step
from backend.core.cleaning.metric import *

# Component Import
from main_compo import render_metrics_row
from user_interface.pages.Cleaning.cleaning_components.cleaning_compo import *

# Button
def handle_confirm(df: pd.DataFrame, working_df: pd.DataFrame,
                    dup_before: int, outlier_before: int,
                    missing_now: int, dup_now: int, outlier_now: int) -> None:
    """Confirm & Save  บันทึก working_df เป็น main_df และ log pipeline"""

    snapshot = {
        "before": {
            "rows": df.shape[0], "cols": df.shape[1],
            "missing": int(df.isnull().sum().sum()),
            "dups": dup_before, "outliers": outlier_before,
        },
        "after": {
            "rows": working_df.shape[0], "cols": working_df.shape[1],
            "missing": missing_now, "dups": dup_now, "outliers": outlier_now,
        },
    }
    original_filename = st.session_state.get("last_uploaded_file", "dataset.csv")
    save_cleaned_data(working_df, original_filename)
    st.session_state["main_df"]                    = working_df.copy()
    st.session_state["cleaning_confirmed"]         = True
    st.session_state["cleaning_summary_snapshot"]  = snapshot
    commit_cleaning(df, working_df)
    commit_step("cleaning", snapshot)
    st.success("บันทึกข้อมูลเรียบร้อย")
    st.rerun()

def handle_reset(df: pd.DataFrame) -> None:
    """Reset working_df กลับเป็น main_df ต้นฉบับ"""
    st.session_state["working_df"]        = df.copy()
    st.session_state["cleaning_confirmed"] = False
    for key in ["treated_outlier_cols", "cleaning_summary_snapshot", "original_outlier_bounds"]:
        st.session_state.pop(key, None)
    st.info("Reset ข้อมูลเรียบร้อย")
    st.rerun()

# Render Page
def render_cleaning():
    from main_compo import page_header
    
    page_header(
        "Data Cleaning",
        "กระบวนการลบหรือแก้ไขข้อมูลที่ผิดพลาด ไม่สมบูรณ์ หรือไม่มีความสอดคล้องกันจากชุดข้อมูล",
    )
    
    if st.session_state.get("main_df") is None:
        navigate("upload")
        return
    
    df = st.session_state["main_df"]
    file_name = st.session_state.get("last_uploaded_file", "Unknown File")
    
    # Data Preview
    st.info(f"**Current Dataset:** {file_name}")
    is_confirmed = st.session_state.get("cleaning_confirmed", False)
    with st.expander("Cleaned Data" if is_confirmed else "Raw Data"):
        preview = df.head(100) if len(df) > 100 else df
        if len(df) > 100:
            st.caption(f"แสดงตัวอย่างข้อมูล 100 แถวแรก (จากทั้งหมด {df.shape[0]:,} แถว)")
        st.dataframe(preview, width="stretch")
        
    # Stats
    init_working_df(df)
    working_df = st.session_state["working_df"]
    bounds     = load_or_compute_outlier_bounds(df)
    total_outlier, outlier_details = get_distribution(working_df, bounds)

    if "original_outlier_count" not in st.session_state:
        st.session_state["original_outlier_count"] = total_outlier

    total_cells = working_df.size
    null_counts = working_df.isnull().sum()
    total_missing = int(null_counts.sum())
    dup_count = int(working_df.duplicated().sum())
    missing_pct = total_missing / total_cells * 100 if total_cells > 0 else 0
    dup_pct = dup_count / working_df.shape[0] * 100 if working_df.shape[0] > 0 else 0
    outlier_pct = total_outlier / total_cells * 100 if total_cells > 0 else 0
    
    # Metrics
    st.subheader("Dataset Overview")
    render_metrics_row([
        ("Rows",            f"{working_df.shape[0]:,}"),
        ("Columns",         str(working_df.shape[1])),
        ("Missing Values",  format_percentage(total_missing, missing_pct)),
        ("Duplicate Rows",  format_percentage(dup_count, dup_pct)),
        ("Outliers",        format_percentage(total_outlier, outlier_pct)),
    ])
    
    # Tabs
    tab_profile, tab_cleaning = st.tabs(["Profile", "Cleaning"])

    with tab_profile:
        render_cleaning_profile_tab(working_df, outlier_details)

    with tab_cleaning:
        working_df    = st.session_state["working_df"]
        target_col    = st.session_state.get("target_col")
        dup_before    = st.session_state.get("original_dup_count", int(df.duplicated().sum()))
        outlier_before= st.session_state["original_outlier_count"]

        render_drop_columns(working_df, target_col);    st.divider()
        render_duplicates(working_df, dup_count);       st.divider()

        missing_cols = {col: int(n) for col, n in null_counts.items() if n > 0}
        render_missing_values(working_df, missing_cols, null_counts); st.divider()

        outlier_cols = {
            item["Column"]: item for item in outlier_details
            if item["Outliers"] > 0
            and pd.api.types.is_numeric_dtype(working_df.get(item["Column"], pd.Series()))
        }
        render_outliers(working_df, outlier_cols, outlier_details); st.divider()

        render_summary(working_df, df, dup_before, outlier_before, total_missing, dup_count, total_outlier)

        # 5. Action buttons
        is_confirmed = st.session_state.get("cleaning_confirmed", False)
        if is_confirmed:
            _, b1, b2, b3, _ = st.columns([1, 2, 1.8, 3.2, 1])
        else:
            _, b1, b2, _ = st.columns([2, 2, 2, 2])

        with b1:
            if st.button("Confirm & Save", type="primary", width="stretch"):
                handle_confirm(df, working_df, dup_before, outlier_before,
                                total_missing, dup_count, total_outlier)
        with b2:
            if st.button("Reset", type="secondary", width="stretch"):
                handle_reset(df)

        if is_confirmed:
            with b3:
                csv_path = st.session_state.get("cleaned_csv_path")
                if csv_path and os.path.exists(csv_path):
                    with open(csv_path, "rb") as f:
                        st.download_button(
                            "Download Cleaned Dataset",
                            data=f,
                            file_name=st.session_state["last_uploaded_file"],
                            mime="text/csv",
                            width="stretch",
                        )
                        
    st.divider()
    if not st.session_state.get("cleaning_confirmed", False):
        st.info("กรุณากด **Confirm & Save** ก่อนไปขั้นตอนถัดไป")

    back_col, _, next_col = st.columns([1.2, 7.6, 1.2])
    with back_col:
        if st.button("Back", type="secondary", width="stretch"):
            for key in ["working_df", "working_df_source_shape", "cleaning_confirmed", "treated_outlier_cols"]:
                st.session_state.pop(key, None)
            navigate("upload")
    with next_col:
        if st.button("Next Step", type="primary", width="stretch",
                     disabled=not st.session_state.get("cleaning_confirmed", False)):
            navigate("eda")