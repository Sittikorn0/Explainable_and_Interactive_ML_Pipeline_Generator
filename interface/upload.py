import streamlit as st
from data_prepare.features.loading_data import process_data, save_to_local, save_target_col
from data_prepare.features.target_col import suggest_target, describe_target, get_column_reasons

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

    if uploaded_file:
        # โหลดและ cache เฉพาะเมื่อเป็นไฟล์ใหม่
        if (
            "last_uploaded_file" not in st.session_state
            or st.session_state["last_uploaded_file"] != uploaded_file.name
        ):
            try:
                df, json_warnings = process_data(uploaded_file)
            except ValueError as e:
                st.error(f"ไม่สามารถโหลดไฟล์ได้: {e}")
                return
            if df is not None:
                st.session_state["main_df"] = df
                st.session_state["last_uploaded_file"] = uploaded_file.name
                st.session_state["json_warnings"] = json_warnings
                st.session_state.pop("target_col", None)  # reset เมื่อโหลดไฟล์ใหม่
                # reset state ทั้งหมดเมื่อโหลดไฟล์ใหม่
                for _k in [
                    # cleaning
                    "working_df", "working_df_source_shape", "cleaning_confirmed",
                    "original_df", "original_dup_count", "original_outlier_count",
                    "_dist_key", "_dist_result", "_treated_outlier_cols",
                    # transformation
                    "transformed_df", "trans_confirmed", "trans_summary",
                    "_trans_cache_key", "_trans_analysis",
                    "_trans_target_saved", "ml_target_col_preset",
                    "_main_df_backup",
                    # ml
                    "ml_result", "ml_metrics", "_fi_data", "ml_task_type",
                    "_ml_scaling_used", "_ml_leakage_warnings",
                ]:
                    st.session_state.pop(_k, None)
                save_to_local(df, uploaded_file.name)
            else:
                st.error("Failed to load data. Please check the file format and content.")
                return

        df = st.session_state.get("main_df")

        if df is not None:
            st.success(f"Load data from '{uploaded_file.name}' successfully!")

            json_warnings = st.session_state.get("json_warnings", [])
            if json_warnings:
                array_cols = [c for c in json_warnings if not c.endswith("(object → string)")]
                dict_cols = [c.replace(" (object → string)", "") for c in json_warnings if c.endswith("(object → string)")]
                sections = []
                if array_cols:
                    items = "\n".join(f"- `{c}`" for c in array_cols)
                    sections.append(f"**Array columns** (ค่าหลายรายการถูก join เป็น string ด้วย ', '):\n\n{items}")
                if dict_cols:
                    items = "\n".join(f"- `{c}`" for c in dict_cols)
                    sections.append(f"**Object columns** (ถูกแปลงเป็น string เพราะไม่สามารถ flatten ได้):\n\n{items}")
                body = "\n\n".join(sections)
                footer = "*คอลัมน์เหล่านี้ถูกแปลงเป็น text — ควรตรวจสอบใน Cleaning หากไม่ต้องการ*"
                st.warning(f"**พบ Nested Columns ใน JSON — ถูกจัดการอัตโนมัติดังนี้:**\n\n{body}\n\n{footer}")

            col1, col2 = st.columns(2)
            col1.metric("Rows", f"{df.shape[0]:,}")
            col2.metric("Columns", f"{df.shape[1]}")

            st.subheader("Data Preview")
            st.dataframe(df.head(10))

            # ── Target Column Selection ───────────────────────────────────────
            st.subheader("Target Column")

            suggested_col, suggested_reasons = suggest_target(df)

            # set ค่าเริ่มต้นก่อน widget render (ต้องทำก่อน selectbox เสมอ)
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

            # ── Navigation ───────────────────────────────────────────────────
            _, col1 = st.columns([8, 0.8])
            with col1:
                if st.button("Next Step", type="primary", width="stretch"):
                    from app import navigate
                    save_target_col(st.session_state["target_col"])
                    navigate("cleaning")