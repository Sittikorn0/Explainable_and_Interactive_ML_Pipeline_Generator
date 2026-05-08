import pandas as pd
import streamlit as st
from data_prepare.loading_data import (
    process_data, save_to_local, save_target_col, delete_ml_cache, apply_json_overrides,
)
from data_prepare.logic.target_col import suggest_target, describe_target, get_column_reasons

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

    # Load & cache
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

    # Sub-tabs
    tab_names = ["Data Preview", "Target Column"] + (["JSON Config"] if is_json else [])
    tabs = st.tabs(tab_names)
    tab_preview = tabs[0]
    tab_target  = tabs[1]
    tab_json    = tabs[2] if is_json else None

    # Data Preview
    with tab_preview:
        st.dataframe(dataframe.head(10), width="stretch")

    # Target Column
    with tab_target:
        suggested_column, suggested_reasons = suggest_target(dataframe)

        if "target_col" not in st.session_state:
            st.session_state["target_col"] = suggested_column
        if st.session_state.pop("_revert_target", False):
            st.session_state["target_col"] = suggested_column

        target_col1, target_col2 = st.columns([2, 4])
        with target_col1:
            selected_target = st.selectbox(
                "เลือก Target Column",
                options=list(dataframe.columns),
                key="target_col",
                label_visibility="collapsed",
                on_change=lambda: save_target_col(st.session_state["target_col"]),
            )
        with target_col2:
            if selected_target == suggested_column:
                reason_bullets = "\n".join(f"- {reason}" for reason in suggested_reasons)
                st.info(f"**ระบบแนะนำ column นี้เพราะ:**\n\n{reason_bullets}")
            else:
                selected_score_reasons = get_column_reasons(dataframe, selected_target)
                reason_bullets = "\n".join(f"- {reason}" for reason in selected_score_reasons)
                st.warning(
                    f"**วิเคราะห์ column ที่คุณเลือก ({selected_target}):**\n\n{reason_bullets}"
                )
                if st.button(f"กลับไปใช้ที่ระบบแนะนำ ({suggested_column})", key="revert_target"):
                    st.session_state["_revert_target"] = True
                    st.rerun()
            st.markdown(describe_target(dataframe, selected_target))

    # JSON Config
    if tab_json is not None:
        with tab_json:
            render_json_config(column_decisions, raw_dataframe)

    # Navigation
    _, nav_col = st.columns([7.6, 1.2])
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


# Helpers

def parse_excel_sheets(warnings: list[str]) -> list[str]:
    for warning in warnings:
        if warning.startswith("__excel_sheets__:"):
            return warning.removeprefix("__excel_sheets__:").split(",")
    return []


TYPE_BADGE_INFO: dict[str, tuple[str, str]] = {
    "array":                 ("Array",          "#0d6efd"),
    "dict":                  ("Object",         "#6c757d"),
    "nested_array_of_dicts": ("⚠ Nested Array", "#fd7e14"),
}

ACTION_BASE_LABELS: dict[str, str] = {
    "join":        "Join เป็น string",
    "first":       "ค่าแรก (First only)",
    "count":       "นับจำนวน (Count)",
    "to_string":   "แปลงเป็น string",
    "count_keys":  "นับ keys",
    "flatten_more":"Flatten เพิ่มเติม",
    "drop":        "Drop column",
}

def action_label(action: str) -> str:
    if action.startswith("extract_field:"):
        return f"ดึง field: {action.removeprefix('extract_field:')}"
    return ACTION_BASE_LABELS.get(action, action)

def badge_html(label: str, color: str) -> str:
    return (
        f"<span style='background:{color};color:#fff;padding:2px 8px;"
        f"border-radius:10px;font-size:0.72rem;font-weight:600'>{label}</span>"
    )


def render_json_config(col_decisions: list[dict], raw_dataframe: pd.DataFrame | None) -> None:
    if not col_decisions:
        st.success("ไม่พบ Nested Columns — โครงสร้าง JSON ตรงไปตรงมา ไม่ต้องปรับแต่งเพิ่มเติม")
        return

    # แสดงผลลัพธ์ Apply จาก render ก่อนหน้า (ถ้ามี)
    apply_result: list[dict] | None = st.session_state.pop("json_apply_result", None)
    if apply_result is not None:
        result_lines = [
            f"- `{item['col']}` → **{action_label(item['action'])}**"
            for item in apply_result
        ]
        st.success("**Apply สำเร็จ** — actions ที่ถูก apply:\n\n" + "\n".join(result_lines))

    st.markdown("#### จัดการ Nested Fields")
    st.caption("★ = action ที่ระบบแนะนำ — ปรับได้แล้วกด Apply")

    type_order = {"nested_array_of_dicts": 0, "array": 1, "dict": 2}
    sorted_decisions = sorted(col_decisions, key=lambda decision: type_order.get(decision["type"], 9))

    with st.container(border=True):
        header1, header2, header3, header4 = st.columns([2, 1.5, 2.5, 3.5])
        header1.markdown("**Column**")
        header2.markdown("**ประเภท**")
        header3.markdown("**Action**")
        header4.markdown("**ตัวอย่างค่า**")
        st.divider()

        for index, decision in enumerate(sorted_decisions):
            col_name = decision["col"]
            col_type = decision["type"]
            available_actions = decision["available_actions"]
            default_action = decision["default_action"]
            recommended_action = decision.get("recommended_action", default_action)
            previews = decision.get("previews", {})

            badge_label_text, badge_color_code = TYPE_BADGE_INFO.get(col_type, (col_type, "#6c757d"))

            col1, col2, col3, col4 = st.columns([2, 1.5, 2.5, 3.5])
            with col1:
                st.markdown(f"`{col_name}`")
            with col2:
                st.markdown(badge_html(badge_label_text, badge_color_code), unsafe_allow_html=True)
            with col3:
                st.selectbox(
                    "action",
                    options=available_actions,
                    format_func=lambda action, rec=recommended_action: (
                        f"★ {action_label(action)}" if action == rec else action_label(action)
                    ),
                    index=available_actions.index(default_action) if default_action in available_actions else 0,
                    key=f"json_choice_{col_name}",
                    label_visibility="collapsed",
                )
            with col4:
                current_choice = st.session_state.get(f"json_choice_{col_name}", default_action)
                st.caption(previews.get(current_choice, "—"))

            if index < len(sorted_decisions) - 1:
                st.divider()

    _, apply_column, reset_column = st.columns([5, 1, 1])
    with apply_column:
        apply_clicked = st.button("Apply", type="primary", use_container_width=True)
    with reset_column:
        reset_clicked = st.button("Reset", use_container_width=True)

    if apply_clicked:
        if raw_dataframe is None:
            st.error("ไม่พบข้อมูลต้นฉบับ — กรุณาอัปโหลดไฟล์ใหม่อีกครั้ง")
        else:
            try:
                user_choices = {
                    decision["col"]: st.session_state.get(f"json_choice_{decision['col']}", decision["default_action"])
                    for decision in col_decisions
                }
                new_dataframe = apply_json_overrides(raw_dataframe, col_decisions, user_choices)
                new_dataframe = new_dataframe.replace("", pd.NA)
                st.session_state["main_df"] = new_dataframe
                st.session_state.pop("target_col", None)

                excel_warnings_part = [
                    warning for warning in st.session_state.get("file_warnings", [])
                    if warning.startswith("__excel_sheets__:")
                ]
                json_warnings_part = [
                    decision["col"] for decision in col_decisions
                    if user_choices.get(decision["col"], decision["default_action"]) != "drop"
                ]
                st.session_state["file_warnings"] = excel_warnings_part + json_warnings_part
                st.session_state["json_apply_result"] = [
                    {"col": decision["col"], "action": user_choices.get(decision["col"], decision["default_action"])}
                    for decision in col_decisions
                ]
                st.rerun()
            except Exception as error:
                st.error(f"ไม่สามารถ apply ได้ — {error}")

    if reset_clicked:
        if raw_dataframe is None:
            st.error("ไม่พบข้อมูลต้นฉบับ — กรุณาอัปโหลดไฟล์ใหม่อีกครั้ง")
        else:
            try:
                for key in [k for k in st.session_state if k.startswith("json_choice_")]:
                    st.session_state.pop(key)
                    
                reset_dataframe = apply_json_overrides(raw_dataframe, col_decisions, {})
                reset_dataframe = reset_dataframe.replace("", pd.NA)
                st.session_state["main_df"] = reset_dataframe
                st.session_state.pop("target_col", None)
                st.session_state.pop("json_apply_result", None)

                excel_warnings_part = [
                    warning for warning in st.session_state.get("file_warnings", [])
                    if warning.startswith("__excel_sheets__:")
                ]
                json_warnings_part = [decision["col"] for decision in col_decisions if decision["default_action"] != "drop"]
                
                st.session_state["file_warnings"] = excel_warnings_part + json_warnings_part
                st.toast("Reset สำเร็จ — กลับสู่ค่าที่ระบบตั้งไว้แล้ว", icon="↩")
                st.rerun()
            except Exception as error:
                st.error(f"ไม่สามารถ reset ได้ — {error}")
