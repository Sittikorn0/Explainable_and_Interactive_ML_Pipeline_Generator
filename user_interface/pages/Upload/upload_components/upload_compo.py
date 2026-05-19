# Libraries
import pandas as pd
import streamlit as st

# Logic Import
from backend.function.analyzer.task_detection import detect_task
from backend.core.session.pipeline_state import save_target_col
from backend.function.data_loader.json_parser import apply_json_overrides
from backend.core.upload.target import suggest_target, describe_target
from backend.core.upload.column import get_column_reasons

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
    "flatten_more":"Flatten เพิ่มเติม",
    "drop":        "Drop column",
}

# Functions
def render_target_selection(dataframe: pd.DataFrame):
    """แสดงส่วนการเลือก Target Column และบทวิเคราะห์ที่เกี่ยวข้อง"""
    suggested_column, suggested_reasons = suggest_target(dataframe)

    if "target_col" not in st.session_state:
        st.session_state["target_col"] = suggested_column
    if st.session_state.pop("_revert_target", False):
        st.session_state["target_col"] = suggested_column

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    
    target_col1, target_col2 = st.columns([1.5, 3])
    
    with target_col1:
        st.markdown("""
            <div style="border-left: 3px solid #7AA2F7; padding-left: 15px; margin-bottom: 12px">
                <div style="font-weight:700; font-size:1.25rem; color:#FFFFFF">Select Target</div>
                <div style="font-size:0.85rem; color:#94A3B8">เลือกคอลัมน์ที่ต้องการพยากรณ์</div>
            </div>
        """, unsafe_allow_html=True)
        
        selected_target = st.selectbox(
            "เลือก Target Column",
            options=list(dataframe.columns),
            key="target_col",
            label_visibility="collapsed",
            on_change=lambda: save_target_col(st.session_state["target_col"]),
        )
        
        # --- Metadata Badges ---
        dtype = str(dataframe[selected_target].dtype)
        unique_count = dataframe[selected_target].nunique()
        task_type = detect_task(dataframe, selected_target)
        
        badge_style = "display:inline-block;padding:2px 10px;border-radius:4px;font-size:0.75rem;font-weight:600;margin-right:6px;margin-top:10px;text-transform:uppercase;letter-spacing:0.02em;"
        st.markdown(f"""
            <div style="margin-top:15px">
                <span style="{badge_style}background:rgba(122, 162, 247, 0.1);color:#7AA2F7;border:1px solid rgba(122, 162, 247, 0.2)">{dtype}</span>
                <span style="{badge_style}background:rgba(187, 154, 247, 0.1);color:#BB9AF7;border:1px solid rgba(187, 154, 247, 0.2)">Unique: {unique_count}</span>
                <span style="{badge_style}background:rgba(158, 206, 106, 0.1);color:#9ECE6A;border:1px solid rgba(158, 206, 106, 0.2)">{task_type}</span>
            </div>
        """, unsafe_allow_html=True)

    with target_col2:
        is_suggested = (selected_target == suggested_column)
        title = "Recommended Analysis" if is_suggested else "Manual Selection Analysis"
        accent_color = "#7AA2F7" if is_suggested else "#E0AF68"
        bg_color = "rgba(122, 162, 247, 0.03)" if is_suggested else "rgba(224, 175, 104, 0.03)"
        border_color = "rgba(122, 162, 247, 0.15)" if is_suggested else "rgba(224, 175, 104, 0.15)"
        
        reasons = suggested_reasons if is_suggested else get_column_reasons(dataframe, selected_target)
        reason_html = "".join(f"<li style='margin-bottom:8px;color:#A9B1D6;font-size:0.95rem'><span style='color:{accent_color};margin-right:8px'>•</span>{r}</li>" for r in reasons)
        
        st.markdown(f"""
            <div style="background:{bg_color}; border:1px solid {border_color}; border-radius:8px; padding:1.5rem; height:100%">
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:15px">
                    <div style="width:10px; height:10px; border-radius:50%; background:{accent_color}; box-shadow: 0 0 8px {accent_color}66"></div>
                    <span style="font-weight:700; font-size:1rem; color:#FFFFFF; text-transform:uppercase; letter-spacing:0.05em">{title}</span>
                </div>
                <ul style="margin:0; padding:0; list-style:none">
                    {reason_html}
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        if not is_suggested:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            if st.button(f"Reset to recommended ({suggested_column})", key="revert_target", width="stretch"):
                st.session_state["_revert_target"] = True
                st.rerun()

def generate_previews(data_sample: pd.Series, available_actions: list[str]) -> dict[str, str]:
    """สร้าง preview text สำหรับแต่ละ action — ใช้แสดงใน UI เท่านั้น"""
    MAX_LEN = 70
    head_values = data_sample.head(2).tolist()

    def fmt(values: list) -> str:
        s = "  |  ".join(str(v) for v in values)
        return s[:MAX_LEN] + "…" if len(s) > MAX_LEN else s

    previews: dict[str, str] = {}
    for action in available_actions:
        if action == "drop":
            previews[action] = "(column จะถูกลบ)"
        elif action == "join":
            previews[action] = fmt([", ".join(str(v) for v in item) if isinstance(item, list) else str(item) for item in head_values])
        elif action == "first":
            previews[action] = fmt([item[0] if isinstance(item, list) and item else None for item in head_values])
        elif action == "count":
            previews[action] = fmt([len(item) if isinstance(item, (list, dict)) else item for item in head_values])
        elif action == "to_string":
            previews[action] = fmt([str(item)[:50] for item in head_values])
        elif action == "flatten_more":
            keys: set[str] = set()
            for item in head_values:
                if isinstance(item, dict):
                    keys.update(item.keys())
            key_str = ", ".join(f"`{k}`" for k in sorted(keys)[:5])
            extra = f" …+{len(keys) - 5}" if len(keys) > 5 else ""
            previews[action] = f"เพิ่ม {len(keys)} columns ใหม่: {key_str}{extra}"
        elif action.startswith("extract_field:"):
            field = action.removeprefix("extract_field:")
            previews[action] = fmt([
                ", ".join(str(d.get(field, "")) for d in item if isinstance(d, dict))
                if isinstance(item, list) else str(item)
                for item in head_values
            ])
    return previews


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
        st.success("ไม่พบ Nested Columns มีโครงสร้าง JSON ตรงไปตรงมา ไม่ต้องปรับแต่งเพิ่มเติม")
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
    st.caption("★ = action ที่ระบบแนะนำ หลังจากปรับแล้วสามารถกด Apply")

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
            data_sample = raw_dataframe[col_name].dropna() if raw_dataframe is not None and col_name in raw_dataframe.columns else pd.Series([], dtype=object)
            previews = generate_previews(data_sample, available_actions)

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
            st.error("ไม่พบข้อมูลต้นฉบับ กรุณาอัปโหลดไฟล์ใหม่อีกครั้ง")
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
                st.error(f"ไม่สามารถ apply ได้ , {error}")

    if reset_clicked:
        if raw_dataframe is None:
            st.error("ไม่พบข้อมูลต้นฉบับ กรุณาอัปโหลดไฟล์ใหม่อีกครั้ง")
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
                st.toast("Reset สำเร็จ กลับสู่ค่าที่ระบบตั้งไว้แล้ว")
                st.rerun()
            except Exception as error:
                st.error(f"ไม่สามารถ reset ได้ , {error}")