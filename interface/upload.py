import pandas as pd
import streamlit as st
from data_prepare.loading_data import (
    process_data, save_to_local, save_target_col, delete_ml_cache,
)
from data_prepare.logic.target_col import get_column_reasons
from interface.upload_components.ui_target_selection import render_target_selection
from interface.upload_components.ui_json_config import render_json_config

FILE_TYPE_INFO = {
    "csv":  ("CSV",   "#28a745"),
    "xlsx": ("Excel", "#0d6efd"),
    "xls":  ("Excel", "#0d6efd"),
    "json": ("JSON",  "#fd7e14"),
}

def file_badge(extension: str) -> str:
    label, color = FILE_TYPE_INFO.get(extension.lower(), (extension.upper(), "#6c757d"))
    return (
        f"<span style='background:{color};color:#fff;padding:3px 12px;"
        f"border-radius:12px;font-size:0.78rem;font-weight:600'>{label}</span>"
    )

def render_upload():
    from app import page_header

    page_header(
        "Upload Dataset",
        "Support CSV, Excel, JSON ( Maximum size 200 MB )",
    )

    uploaded_file = st.file_uploader(
        "Upload",
        type=["csv", "xlsx", "xls", "json"],
        label_visibility="collapsed",
    )

    if not uploaded_file:
        return

    extension = uploaded_file.name.rsplit(".", 1)[-1].lower()
    is_json = extension == "json"

    # Load & cache logic
    if (
        "last_uploaded_file" not in st.session_state
        or st.session_state["last_uploaded_file"] != uploaded_file.name
    ):
        try:
            dataframe, file_warnings, json_metadata = process_data(uploaded_file)
        except ValueError as error:
            st.error(f"ไม่สามารถโหลดไฟล์ได้: {error}")
            return
            
        if dataframe is not None:
            st.session_state["main_df"]              = dataframe
            st.session_state["last_uploaded_file"]   = uploaded_file.name
            st.session_state["file_warnings"]        = file_warnings
            st.session_state["json_raw_df"]          = json_metadata.get("raw_df")
            st.session_state["json_col_decisions"]   = json_metadata.get("col_decisions", [])
            st.session_state.pop("target_col", None)
            
            # Clear old state keys
            keys_to_clear = [
                "working_df", "working_df_source_shape", "cleaning_confirmed",
                "original_df", "original_dup_count", "original_outlier_count",
                "original_outlier_bounds",
                "_dist_key", "_dist_result", "_treated_outlier_cols",
                "transformed_df", "trans_confirmed", "trans_summary",
                "_trans_cache_key", "_trans_analysis",
                "_trans_target_saved", "ml_target_col_preset",
                "_main_df_backup",
                "ml_result", "ml_metrics", "_fi_data", "ml_task_type",
                "_ml_scaling_used", "_ml_leakage_warnings",
            ]
            for key in keys_to_clear:
                st.session_state.pop(key, None)
                
            for key in [k for k in st.session_state.keys() if k.startswith("_xai_")]:
                st.session_state.pop(key, None)
            for key in [k for k in st.session_state.keys() if k.startswith("json_choice_")]:
                st.session_state.pop(key, None)
                
            delete_ml_cache()
            save_to_local(dataframe, uploaded_file.name)
        else:
            st.error("Failed to load data. Please check the file format and content.")
            return

    dataframe = st.session_state.get("main_df")
    if dataframe is None:
        return

    file_warnings = st.session_state.get("file_warnings") or st.session_state.get("json_warnings", [])
    excel_sheets  = parse_excel_sheets(file_warnings)
    column_decisions = st.session_state.get("json_col_decisions", [])
    raw_dataframe = st.session_state.get("json_raw_df")

    # Success + file type badge
    msg_column, badge_column = st.columns([7, 1])
    with msg_column:
        st.success(f"โหลดไฟล์ '{uploaded_file.name}' สำเร็จ!")
    with badge_column:
        st.markdown(file_badge(extension), unsafe_allow_html=True)

    metrics_col1, metrics_col2 = st.columns(2)
    metrics_col1.metric("Rows", f"{dataframe.shape[0]:,}")
    metrics_col2.metric("Columns", f"{dataframe.shape[1]}")

    # Excel multi-sheet alert
    if excel_sheets:
        sheet_list = "  |  ".join(f"`{sheet}`" for sheet in excel_sheets)
        st.info(
            f"**ไฟล์นี้มี {len(excel_sheets)} sheets** — ระบบโหลด sheet แรก (`{excel_sheets[0]}`) อัตโนมัติ\n\n"
            f"Sheets ทั้งหมด: {sheet_list}"
        )

    # JSON alert
    if is_json:
        if column_decisions:
            st.warning(
                "**ตรวจพบ Nested Fields ใน JSON** — ระบบเลือก action ให้อัตโนมัติแล้ว "
                "กรุณาตรวจสอบและปรับได้ใน tab **JSON Config** ด้านล่าง"
            )
        else:
            st.success("โครงสร้าง JSON ตรงไปตรงมา ไม่พบ Nested Fields", icon="✅")

    # Main Tabs
    tab_names = ["Data Preview", "Target Column"] + (["JSON Config"] if is_json else [])
    tabs = st.tabs(tab_names)
    
    with tabs[0]: # Data Preview
        st.dataframe(dataframe.head(10), width="stretch")

    with tabs[1]: # Target Column
        render_target_selection(dataframe)

    if is_json:
        with tabs[2]: # JSON Config
            render_json_config(column_decisions, raw_dataframe)

    # Navigation
    st.markdown("<div style='margin-top:3rem'></div>", unsafe_allow_html=True)
    _, nav_col = st.columns([7.4, 1.4])
    with nav_col:
        if st.button("Next Step", type="primary", width="stretch"):
            try:
                from app import navigate
                from explainable.state_manager.trace_log import clear, log_upload
                from ml_process.logic.data_analyzer import detect_task
                
                save_target_col(st.session_state["target_col"])
                target_col = st.session_state["target_col"]
                task_type   = detect_task(dataframe, target_col)
                target_reasons = get_column_reasons(dataframe, target_col)
                
                clear()
                log_upload(dataframe, uploaded_file.name, target_col, task_type, target_reasons=target_reasons)
                
                from explainable.state_manager.pipeline_state import commit_step
                commit_step("upload", {
                    "filename": uploaded_file.name,
                    "rows": dataframe.shape[0],
                    "cols": dataframe.shape[1],
                    "target": target_col,
                    "task": task_type
                })
                
                navigate("cleaning")
            except Exception as error:
                st.error(f"ไม่สามารถไปขั้นตอนต่อไปได้ — {error}")

def parse_excel_sheets(warnings: list[str]) -> list[str]:
    for warning in warnings:
        if warning.startswith("__excel_sheets__:"):
            return warning.removeprefix("__excel_sheets__:").split(",")
    return []
