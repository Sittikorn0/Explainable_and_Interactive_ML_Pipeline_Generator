# Libraries
import streamlit as st
import pandas as pd

# Logic Import
from backend.core.cleaning.data_distribution import *
from backend.core.cleaning.statistic import *
from backend.core.cleaning.main_logic import *
from backend.core.session.state import load_outlier_bounds, save_outlier_bounds

def render_cleaning_profile_tab(working_dataframe: pd.DataFrame, outlier_details: list):
    st.subheader("Data Profile")

    outlier_dictionary = {item["Column"]: item for item in outlier_details}
    profile_data_list = []
    for column_name in working_dataframe.columns:
        data_series = working_dataframe[column_name]
        profile_data_list.append({
            "Column": column_name,
            "Missing": int(data_series.isnull().sum()),
            "Outliers": outlier_dictionary.get(column_name, {}).get("Outliers", 0),
            "Unique": int(data_series.nunique()),
        })

    st.dataframe(
        pd.DataFrame(profile_data_list),
        width="stretch",
        hide_index=True,
        column_config={
            "Column": st.column_config.TextColumn("Column"),
            "Missing": st.column_config.NumberColumn("Missing", format="%d"),
            "Unique": st.column_config.ProgressColumn(
                "Unique", min_value=0, max_value=working_dataframe.shape[0], format="%d"
            ),
            "Outliers": st.column_config.NumberColumn("Outliers", format="%d"),
        },
    )
    
def init_working_df(df: pd.DataFrame) -> None:
    """สร้าง working_df จาก main_df ถ้ายังไม่มี หรือ main_df เปลี่ยนรูปร่าง"""
    if (
        "working_df" not in st.session_state
        or st.session_state.get("working_df_source_shape") != df.shape
    ):
        st.session_state["working_df"] = df.copy()
        st.session_state["working_df_source_shape"] = df.shape
        st.session_state["original_dup_count"] = int(df.duplicated().sum())
        st.session_state.pop("original_outlier_count", None)
        st.session_state.pop("_treated_outlier_cols", None)

def load_or_compute_outlier_bounds(df: pd.DataFrame) -> dict:
    """โหลด outlier bounds จาก disk หรือคำนวณจาก df แล้ว save"""
    if "original_outlier_bounds" not in st.session_state:
        saved = load_outlier_bounds()
        if saved:
            st.session_state["original_outlier_bounds"] = saved
        else:
            bounds = {
                col: get_outlier_bounds(df[col].dropna())
                for col in df.select_dtypes(include="number").columns
                if len(df[col].dropna()) > 0
            }
            st.session_state["original_outlier_bounds"] = bounds
            save_outlier_bounds(bounds)
    return st.session_state["original_outlier_bounds"]

def get_distribution(working_df: pd.DataFrame, bounds: dict) -> tuple:
    """คำนวณ distribution พร้อม cache คืน (total_outlier, outlier_details)"""
    cache_key = ("dist_cache", working_df.shape, int(pd.util.hash_pandas_object(working_df).sum()))
    if st.session_state.get("dist_key") != cache_key:
        with st.spinner("Calculating Data..."):
            result = data_distribution(working_df, fixed_bounds=bounds)
        st.session_state["dist_key"] = cache_key
        st.session_state["dist_result"] = result
    return st.session_state["dist_result"]

def render_summary(working_dataframe: pd.DataFrame, original_dataframe: pd.DataFrame, duplicate_before: int, outlier_before: int, total_missing: int, duplicate_count: int, total_outlier: int):
    st.subheader("Summary")

    if st.session_state.get("cleaning_confirmed") and "cleaning_summary_snapshot" in st.session_state:
        snapshot = st.session_state["cleaning_summary_snapshot"]
        before_state = snapshot["before"]
        after_state = snapshot["after"]
        changed_values = [
            after_state["rows"] - before_state["rows"],
            after_state["cols"] - before_state["cols"],
            after_state["missing"] - before_state["missing"],
            after_state["dups"] - before_state["dups"],
            after_state["outliers"] - before_state["outliers"],
        ]
        summary_dataframe = pd.DataFrame({
            "Metric": ["Rows", "Columns", "Missing Values", "Duplicates", "Outliers"],
            "Before": [before_state['rows'], before_state['cols'], before_state['missing'], before_state['dups'], before_state['outliers']],
            "After":  [after_state['rows'], after_state['cols'], after_state['missing'], after_state['dups'], after_state['outliers']],
            "Changed": changed_values,
        })
    else:
        changed_values = [
            working_dataframe.shape[0] - original_dataframe.shape[0],
            working_dataframe.shape[1] - original_dataframe.shape[1],
            total_missing - int(original_dataframe.isnull().sum().sum()),
            duplicate_count - duplicate_before,
            total_outlier - outlier_before,
        ]
        summary_dataframe = pd.DataFrame({
            "Metric": ["Rows", "Columns", "Missing Values", "Duplicates", "Outliers"],
            "Before": [original_dataframe.shape[0], original_dataframe.shape[1], int(original_dataframe.isnull().sum().sum()), duplicate_before, outlier_before],
            "After": [working_dataframe.shape[0], working_dataframe.shape[1], int(working_dataframe.isnull().sum().sum()), duplicate_count, total_outlier],
            "Changed": changed_values,
        })

    styled_summary = (
        summary_dataframe.style
        .apply(color_changed, subset=["Changed"])
        .format({
            "Before": "{:,}",
            "After": "{:,}",
            "Changed": lambda value: "—" if value == 0 else f"+{value:,}" if value > 0 else f"{value:,}"
        })
    )
    st.dataframe(styled_summary, width="stretch", hide_index=True)