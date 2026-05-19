# Libraries
import streamlit as st
import pandas as pd

# Logic
from backend.function.data_loader.file_reader import process_data
from backend.core.session.state import *
from main import navigate
from backend.core.session.pipeline_state import *
from backend.core.insight.trace_log import *
from backend.function.analyzer.task_detection import *

from user_interface.pages.Upload.upload_components.upload_compo import *

# Extension
FILE_TYPE_INFO = {
    "csv": ("CSV", "#28a745"),
    "xlsx": ("Excel", "#0d6efd"),
    "xls": ("Excel", "#0d6efd"),
    "json": ("JSON", "#fd7e14"),
}

# Keys ที่ต้องล้างทุกครั้งที่อัปโหลดไฟล์ใหม่
_KEYS_TO_CLEAR = [
    "working_df", "working_df_source_shape", "cleaning_confirmed",
    "original_df", "original_dup_count", "original_outlier_count",
    "original_outlier_bounds",
    "dist_key", "dist_result", "treated_outlier_cols",
    "transformed_df", "trans_confirmed", "trans_summary",
    "trans_cache_key", "trans_analysis",
    "trans_target_saved", "ml_target_col_preset",
    "main_df_backup",
    "ml_result", "ml_metrics", "fi_data", "ml_task_type",
    "ml_scaling_used", "ml_leakage_warnings",
]

def parse_excel_sheets(warnings: list[str]) -> list[str]:
    """ดึงรายชื่อ sheet จาก warning string ที่ฝังไว้โดย file_reader"""
    for w in warnings:
        if w.startswith("__excel_sheets__:"):
            return w.removeprefix("__excel_sheets__:").split(",")
    return []

def file_badge(extension: str) -> str:
    label, color = FILE_TYPE_INFO.get(extension.lower(), (extension.upper(), "#6c757d"))
    return (
        f"<span style='background:{color};color:#fff;padding:3px 12px;"
        f"border-radius:12px;font-size:0.78rem;font-weight:600'>{label}</span>"
    )
    
def save_new_file(df: pd.DataFrame, filename: str, warnings: list, json_metadata: dict) -> None:
    """บันทึก state ทั้งหมดสำหรับไฟล์ที่เพิ่งโหลด"""
    st.session_state["main_df"] = df
    st.session_state["last_uploaded_file"] = filename
    st.session_state["file_warnings"] = warnings
    st.session_state["json_raw_df"] = json_metadata.get("raw_df")
    st.session_state["json_col_decisions"] = json_metadata.get("col_decisions", [])
    st.session_state.pop("target_col", None)

    for key in _KEYS_TO_CLEAR:
        st.session_state.pop(key, None)
    for key in [k for k in st.session_state if k.startswith("_xai_")]:
        st.session_state.pop(key, None)
    for key in [k for k in st.session_state if k.startswith("json_choice_")]:
        st.session_state.pop(key, None)

    delete_ml_cache()
    save_to_local(df, filename)
    
def handle_next_step(df: pd.DataFrame, uploaded_file) -> None:
    """บันทึก target, log pipeline trace แล้วไปหน้า cleaning"""
    try:
        target_col     = st.session_state["target_col"]
        task_type      = detect_task(df, target_col)
        target_reasons = get_column_reasons(df, target_col)

        save_target_col(target_col)
        clear()
        log_upload(df, uploaded_file.name, target_col, task_type, target_reasons=target_reasons)
        commit_step("upload", {
            "filename": uploaded_file.name,
            "rows":     df.shape[0],
            "cols":     df.shape[1],
            "target":   target_col,
            "task":     task_type,
        })
        navigate("cleaning")
    except Exception as e:
        st.error(f"ไม่สามารถไปขั้นตอนต่อไปได้ — {e}")

# Render Page
def render_upload():
    from main_compo import page_header
    
    page_header(
        "Upload Dataset",
        "Support CSV, Excel, JSON ( Maximum size 200 MB )",
    )
    
    upload_file = st.file_uploader(
        "Upload",
        type=["csv", "xlsx", "xls", "json"],
        label_visibility="collapsed",
    )
    
    if not upload_file:
        return
    
    # Check File Type
    file_type = upload_file.name.rsplit(".", 1)[-1].lower()
    
    # JSON
    is_json = file_type == "json"
    
    # Load File
    if st.session_state.get("last_uploaded_file") != upload_file.name:
        try:
            df, warnings, json_metadata = process_data(upload_file)
        except ValueError as e:
            st.error(f"ไม่สามารถโหลดไฟล์ได้: {e}")
            return

        if df is None:
            st.error("Failed to load data. Please check the file format and content.")
            return

        save_new_file(df, upload_file.name, warnings, json_metadata)

    df = st.session_state.get("main_df")
    warnings = st.session_state.get("file_warnings") or []
    excel_sheets = parse_excel_sheets(warnings)
    col_decisions = st.session_state.get("json_col_decisions", [])
    raw_df = st.session_state.get("json_raw_df")

    if df is None:
        return
    
    # Show Data
    msg_col, badge_col = st.columns([7, 1])
    with msg_col:
        st.success(f"โหลดไฟล์ '{upload_file.name}' สำเร็จ!")
    with badge_col:
        st.markdown(file_badge(file_type), unsafe_allow_html=True)

    row_col, col_col = st.columns(2)
    row_col.metric("Rows", f"{df.shape[0]:,}")
    col_col.metric("Columns", f"{df.shape[1]}")

    if excel_sheets:
        sheet_list = "  |  ".join(f"`{s}`" for s in excel_sheets)
        st.info(
            f"**ไฟล์นี้มี {len(excel_sheets)} sheets** ระบบโหลด sheet แรก (`{excel_sheets[0]}`) อัตโนมัติ\n\n"
            f"Sheets ทั้งหมด: {sheet_list}"
        )

    if is_json:
        if col_decisions:
            st.warning(
                "**ตรวจพบ Nested Fields ใน JSON** ระบบเลือก action ให้อัตโนมัติแล้ว "
                "กรุณาตรวจสอบและปรับได้ในแท็บ **JSON Config** ด้านล่าง"
            )
        else:
            st.success("โครงสร้าง JSON ตรงไปตรงมา ไม่พบ Nested Fields", icon="✅")
            
    # Tabs
    tab_names = ["Data Preview", "Target Column"] + (["JSON Config"] if is_json else [])
    tabs = st.tabs(tab_names)
    
    with tabs[0]:
        st.dataframe(df.head(10), width="stretch")
    with tabs[1]:
        render_target_selection(df)
    if is_json:
        with tabs[2]:
            render_json_config(col_decisions, raw_df)
    
    # Navigate
    st.markdown("<div style='margin-top:3rem'></div>", unsafe_allow_html=True)
    _, nav_col = st.columns([7.4, 1.4])
    with nav_col:
        if st.button("Next Step", type="primary", width="stretch"):
            handle_next_step(df, upload_file)
    