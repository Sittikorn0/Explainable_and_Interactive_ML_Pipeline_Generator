import streamlit as st
import pandas as pd
from data_prepare.loading_data import apply_json_overrides

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
    """ส่วนการตั้งค่า JSON สำหรับ Nested Fields"""
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
        apply_clicked = st.button("Apply", type="primary", width="stretch")
    with reset_column:
        reset_clicked = st.button("Reset", width="stretch")

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
