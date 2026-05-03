import pandas as pd
import streamlit as st
from data_prepare.features.loading_data import (
    process_data, save_to_local, save_target_col, delete_ml_cache, apply_json_overrides,
)
from data_prepare.features.target_col import suggest_target, describe_target, get_column_reasons

_FILE_TYPE_INFO = {
    "csv":  ("CSV",   "#28a745"),
    "xlsx": ("Excel", "#0d6efd"),
    "xls":  ("Excel", "#0d6efd"),
    "json": ("JSON",  "#fd7e14"),
}

def _file_badge(ext: str) -> str:
    label, color = _FILE_TYPE_INFO.get(ext.lower(), (ext.upper(), "#6c757d"))
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

    ext     = uploaded_file.name.rsplit(".", 1)[-1].lower()
    is_json = ext == "json"

    # ── Load & cache ──────────────────────────────────────────────────────────
    if (
        "last_uploaded_file" not in st.session_state
        or st.session_state["last_uploaded_file"] != uploaded_file.name
    ):
        try:
            df, file_warnings, json_meta = process_data(uploaded_file)
        except ValueError as e:
            st.error(f"ไม่สามารถโหลดไฟล์ได้: {e}")
            return
        if df is not None:
            st.session_state["main_df"]              = df
            st.session_state["last_uploaded_file"]   = uploaded_file.name
            st.session_state["file_warnings"]        = file_warnings
            st.session_state["_json_raw_df"]         = json_meta.get("raw_df")
            st.session_state["_json_col_decisions"]  = json_meta.get("col_decisions", [])
            st.session_state.pop("target_col", None)
            for _k in [
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
            ]:
                st.session_state.pop(_k, None)
            for _k in [k for k in st.session_state.keys() if k.startswith("_xai_")]:
                st.session_state.pop(_k, None)
            for _k in [k for k in st.session_state.keys() if k.startswith("_json_choice_")]:
                st.session_state.pop(_k, None)
            delete_ml_cache()
            save_to_local(df, uploaded_file.name)
        else:
            st.error("Failed to load data. Please check the file format and content.")
            return

    df = st.session_state.get("main_df")
    if df is None:
        return

    # รองรับ key เก่า "json_warnings" สำหรับ session ที่ cache ไว้ก่อน update
    file_warnings = st.session_state.get("file_warnings") or st.session_state.get("json_warnings", [])
    excel_sheets  = _parse_excel_sheets(file_warnings)
    col_decisions = st.session_state.get("_json_col_decisions", [])
    raw_df        = st.session_state.get("_json_raw_df")

    # ── Success + file type badge ─────────────────────────────────────────────
    c_msg, c_badge = st.columns([7, 1])
    with c_msg:
        st.success(f"โหลดไฟล์ '{uploaded_file.name}' สำเร็จ!")
    with c_badge:
        st.markdown(_file_badge(ext), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]}")

    # ── Excel multi-sheet alert ───────────────────────────────────────────────
    if excel_sheets:
        sheet_list = "  |  ".join(f"`{s}`" for s in excel_sheets)
        st.info(
            f"**ไฟล์นี้มี {len(excel_sheets)} sheets** — ระบบโหลด sheet แรก (`{excel_sheets[0]}`) อัตโนมัติ\n\n"
            f"Sheets ทั้งหมด: {sheet_list}"
        )

    # ── JSON alert ────────────────────────────────────────────────────────────
    if is_json:
        if col_decisions:
            st.warning(
                "**ตรวจพบ Nested Fields ใน JSON** — ระบบเลือก action ให้อัตโนมัติแล้ว "
                "กรุณาตรวจสอบและปรับได้ใน tab **JSON Config** ด้านล่าง"
            )
        else:
            st.success("โครงสร้าง JSON ตรงไปตรงมา ไม่พบ Nested Fields", icon="✅")

    # ── Sub-tabs ──────────────────────────────────────────────────────────────
    tab_names = ["Data Preview", "Target Column"] + (["JSON Config"] if is_json else [])
    tab_list  = st.tabs(tab_names)
    tab_preview = tab_list[0]
    tab_target  = tab_list[1]
    tab_json    = tab_list[2] if is_json else None

    # ── Data Preview ──────────────────────────────────────────────────────────
    with tab_preview:
        st.dataframe(df.head(10), width="stretch")

    # ── Target Column ─────────────────────────────────────────────────────────
    with tab_target:
        suggested_col, suggested_reasons = suggest_target(df)

        if "target_col" not in st.session_state:
            st.session_state["target_col"] = suggested_col
        if st.session_state.pop("_revert_target", False):
            st.session_state["target_col"] = suggested_col

        c1, c2 = st.columns([2, 4])
        with c1:
            selected = st.selectbox(
                "เลือก Target Column",
                options=list(df.columns),
                key="target_col",
                label_visibility="collapsed",
                on_change=lambda: save_target_col(st.session_state["target_col"]),
            )
        with c2:
            if selected == suggested_col:
                reason_bullets = "\n".join(f"- {r}" for r in suggested_reasons)
                st.info(f"**ระบบแนะนำ column นี้เพราะ:**\n\n{reason_bullets}")
            else:
                selected_score_reasons = get_column_reasons(df, selected)
                reason_bullets = "\n".join(f"- {r}" for r in selected_score_reasons)
                st.warning(
                    f"**วิเคราะห์ column ที่คุณเลือก ({selected}):**\n\n{reason_bullets}"
                )
                if st.button(f"กลับไปใช้ที่ระบบแนะนำ ({suggested_col})", key="revert_target"):
                    st.session_state["_revert_target"] = True
                    st.rerun()
            st.markdown(describe_target(df, selected))

    # ── JSON Config ───────────────────────────────────────────────────────────
    if tab_json is not None:
        with tab_json:
            _render_json_config(col_decisions, raw_df)

    # ── Navigation ────────────────────────────────────────────────────────────
    _, col1 = st.columns([8, 0.8])
    with col1:
        if st.button("Next Step", type="primary", width="stretch"):
            try:
                from app import navigate
                from explainable.features.trace_log import clear, log_upload
                from ml_process.features.preprocessing import detect_task
                save_target_col(st.session_state["target_col"])
                _target = st.session_state["target_col"]
                _task   = detect_task(df, _target)
                _reasons = get_column_reasons(df, _target)
                clear()
                log_upload(df, uploaded_file.name, _target, _task, target_reasons=_reasons)
                navigate("cleaning")
            except Exception as e:
                st.error(f"ไม่สามารถไปขั้นตอนต่อไปได้ — {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_excel_sheets(warnings: list[str]) -> list[str]:
    for w in warnings:
        if w.startswith("__excel_sheets__:"):
            return w.removeprefix("__excel_sheets__:").split(",")
    return []


_TYPE_BADGE: dict[str, tuple[str, str]] = {
    "array":                 ("Array",          "#0d6efd"),
    "dict":                  ("Object",         "#6c757d"),
    "nested_array_of_dicts": ("⚠ Nested Array", "#fd7e14"),
}

_ACTION_BASE_LABELS: dict[str, str] = {
    "join":        "Join เป็น string",
    "first":       "ค่าแรก (First only)",
    "count":       "นับจำนวน (Count)",
    "to_string":   "แปลงเป็น string",
    "count_keys":  "นับ keys",
    "flatten_more":"Flatten เพิ่มเติม",
    "drop":        "Drop column",
}

def _action_label(action: str) -> str:
    if action.startswith("extract_field:"):
        return f"ดึง field: {action.removeprefix('extract_field:')}"
    return _ACTION_BASE_LABELS.get(action, action)

def _badge_html(label: str, color: str) -> str:
    return (
        f"<span style='background:{color};color:#fff;padding:2px 8px;"
        f"border-radius:10px;font-size:0.72rem;font-weight:600'>{label}</span>"
    )


def _render_json_config(col_decisions: list[dict], raw_df: pd.DataFrame | None) -> None:
    if not col_decisions:
        st.success("ไม่พบ Nested Columns — โครงสร้าง JSON ตรงไปตรงมา ไม่ต้องปรับแต่งเพิ่มเติม")
        return

    # แสดงผลลัพธ์ Apply จาก render ก่อนหน้า (ถ้ามี)
    apply_result: list[dict] | None = st.session_state.pop("_json_apply_result", None)
    if apply_result is not None:
        lines = [
            f"- `{item['col']}` → **{_action_label(item['action'])}**"
            for item in apply_result
        ]
        st.success("**Apply สำเร็จ** — actions ที่ถูก apply:\n\n" + "\n".join(lines))

    st.markdown("#### จัดการ Nested Fields")
    st.caption("★ = action ที่ระบบแนะนำ — ปรับได้แล้วกด Apply")

    _order = {"nested_array_of_dicts": 0, "array": 1, "dict": 2}
    sorted_decisions = sorted(col_decisions, key=lambda d: _order.get(d["type"], 9))

    with st.container(border=True):
        h1, h2, h3, h4 = st.columns([2, 1.5, 2.5, 3.5])
        h1.markdown("**Column**")
        h2.markdown("**ประเภท**")
        h3.markdown("**Action**")
        h4.markdown("**ตัวอย่างค่า**")
        st.divider()

        for i, d in enumerate(sorted_decisions):
            col_name = d["col"]
            col_type = d["type"]
            avail    = d["available_actions"]
            default  = d["default_action"]
            rec      = d.get("recommended_action", default)
            previews = d.get("previews", {})

            badge_label, badge_color = _TYPE_BADGE.get(col_type, (col_type, "#6c757d"))

            c1, c2, c3, c4 = st.columns([2, 1.5, 2.5, 3.5])
            with c1:
                st.markdown(f"`{col_name}`")
            with c2:
                st.markdown(_badge_html(badge_label, badge_color), unsafe_allow_html=True)
            with c3:
                st.selectbox(
                    "action",
                    options=avail,
                    format_func=lambda a, _rec=rec: (
                        f"★ {_action_label(a)}" if a == _rec else _action_label(a)
                    ),
                    index=avail.index(default) if default in avail else 0,
                    key=f"_json_choice_{col_name}",
                    label_visibility="collapsed",
                )
            with c4:
                current = st.session_state.get(f"_json_choice_{col_name}", default)
                st.caption(previews.get(current, "—"))

            if i < len(sorted_decisions) - 1:
                st.divider()

    _, c_apply, c_reset = st.columns([5, 1, 1])
    with c_apply:
        apply_clicked = st.button("Apply", type="primary", use_container_width=True)
    with c_reset:
        reset_clicked = st.button("Reset", use_container_width=True)

    if apply_clicked:
        if raw_df is None:
            st.error("ไม่พบข้อมูลต้นฉบับ — กรุณาอัปโหลดไฟล์ใหม่อีกครั้ง")
        else:
            try:
                user_choices = {
                    d["col"]: st.session_state.get(f"_json_choice_{d['col']}", d["default_action"])
                    for d in col_decisions
                }
                new_df = apply_json_overrides(raw_df, col_decisions, user_choices)
                new_df = new_df.replace("", pd.NA)
                st.session_state["main_df"] = new_df
                st.session_state.pop("target_col", None)

                excel_part = [
                    w for w in st.session_state.get("file_warnings", [])
                    if w.startswith("__excel_sheets__:")
                ]
                json_part = [
                    d["col"] for d in col_decisions
                    if user_choices.get(d["col"], d["default_action"]) != "drop"
                ]
                st.session_state["file_warnings"] = excel_part + json_part
                st.session_state["_json_apply_result"] = [
                    {"col": d["col"], "action": user_choices.get(d["col"], d["default_action"])}
                    for d in col_decisions
                ]
                st.rerun()
            except Exception as e:
                st.error(f"ไม่สามารถ apply ได้ — {e}")

    if reset_clicked:
        if raw_df is None:
            st.error("ไม่พบข้อมูลต้นฉบับ — กรุณาอัปโหลดไฟล์ใหม่อีกครั้ง")
        else:
            try:
                for _k in [k for k in st.session_state if k.startswith("_json_choice_")]:
                    st.session_state.pop(_k)
                reset_df = apply_json_overrides(raw_df, col_decisions, {})
                reset_df = reset_df.replace("", pd.NA)
                st.session_state["main_df"] = reset_df
                st.session_state.pop("target_col", None)
                st.session_state.pop("_json_apply_result", None)

                excel_part = [
                    w for w in st.session_state.get("file_warnings", [])
                    if w.startswith("__excel_sheets__:")
                ]
                json_part = [d["col"] for d in col_decisions if d["default_action"] != "drop"]
                st.session_state["file_warnings"] = excel_part + json_part
                st.toast("Reset สำเร็จ — กลับสู่ค่าที่ระบบตั้งไว้แล้ว", icon="↩")
                st.rerun()
            except Exception as e:
                st.error(f"ไม่สามารถ reset ได้ — {e}")
