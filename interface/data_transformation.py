import streamlit as st
import pandas as pd

from ml_process.features.data_analyzer    import analyze_all
from ml_process.features.data_transformer import apply_all
from ml_process.features.scaling           import _render_scaling, SCALING_LABELS
from ml_process.features.encoding          import _render_encoding
from ml_process.features.feature_select    import _render_feature_selection, _render_leakage_check

def _render_summary(df: pd.DataFrame, transformed_df: pd.DataFrame,
                    summary: dict, target_col: str):
    st.markdown("---")
    st.subheader("สรุปผล Data Transformation")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original Columns",     summary["original_cols"])
    c2.metric("Dropped",              f'-{summary["dropped_cols"]}')
    c3.metric("After Encoding",       summary["final_cols"])
    c4.metric("Scaling",              SCALING_LABELS.get(summary["scaling_method"], "—"))

    with st.expander("ดู Transformed Data (5 rows แรก)"):
        st.caption(f"หมายเหตุ: ตัวเลขใน Preview นี้ยังไม่ถูก Scale ({SCALING_LABELS.get(summary['scaling_method'], summary['scaling_method'])}) โดยระบบจะนำไปคำนวณจริงในขั้นตอน ML Process เพื่อความแม่นยำสูงสุด")
        st.dataframe(transformed_df.head(5), width="stretch")

def render_transformation():
    from app import page_header

    page_header(
        "Data Transformation",
        "แปลงข้อมูลให้พร้อมสำหรับ ML — ระบบวิเคราะห์และแนะนำวิธีที่เหมาะสมพร้อมเหตุผล",
    )

    # Guard
    if st.session_state.get("main_df") is None:
        from app import navigate
        navigate("upload")
        return

    is_cleaned = st.session_state.get("cleaning_confirmed", False)
    df        = st.session_state["working_df"] if is_cleaned and "working_df" in st.session_state else st.session_state["main_df"]
    file_name = st.session_state.get("last_uploaded_file", "Unknown File")

    st.info(
        f"**Current Dataset:** {file_name}  "
        f"|  {df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    # ── Target Column (อ่านค่าจาก Upload — ไม่ให้แก้ที่นี่) ──────
    cols = df.columns.tolist()
    saved_target = st.session_state.get("target_col", cols[-1])
    target_col   = saved_target if saved_target in cols else cols[-1]
    st.markdown("**Target Column** (จะไม่ถูก transform)")
    st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-left:5px solid #58a6ff;
border-radius:6px;padding:10px 14px;font-family:monospace;font-size:1.2rem;
color:#58a6ff;font-weight:600">
{target_col}
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── วิเคราะห์ข้อมูล (cache ตาม df shape + target) ────────
    _df_fingerprint = hash((df.shape, tuple(df.columns), str(df.iloc[0].values.tolist()) if len(df) else ""))
    cache_key = f"trans_analysis_{_df_fingerprint}_{target_col}"
    if st.session_state.get("_trans_cache_key") != cache_key:
        with st.spinner("วิเคราะห์ข้อมูล..."):
            analysis = analyze_all(df, target_col)
        st.session_state["_trans_analysis"]  = analysis
        st.session_state["_trans_cache_key"] = cache_key
    else:
        analysis = st.session_state["_trans_analysis"]

    enc_analysis = analysis["encoding"]
    sc_analysis  = analysis["scaling"]
    fs_analysis  = analysis["feature_selection"]

    # ── Render sections ───────────────────────────────────────
    enc_decisions  = _render_encoding(df, target_col, enc_analysis)
    st.markdown("---")
    scaling_method = _render_scaling(df, target_col, sc_analysis)
    st.markdown("---")
    leakage_drops  = _render_leakage_check(df, target_col)
    st.markdown("---")
    # เรียงลำดับเพื่อให้การเปรียบเทียบใน Choice Tracker เสถียร (ป้องกันลำดับสลับไปมาใน set)
    drop_cols      = sorted(list(set(_render_feature_selection(df, target_col, fs_analysis) + leakage_drops)))

    st.markdown("---")
    
    if st.button("Apply Transformation",  type="primary", width="stretch"):
        with st.spinner("กำลังประมวลผล Transformation..."):
            try:
                # [Fail-safe] ดึงค่าตรงจาก Session State ของ Widget
                final_sc_method = st.session_state.get("scaling_method", scaling_method)
                
                transformed_df, summary = apply_all(
                    df, enc_decisions, final_sc_method, drop_cols, target_col
                )
                
                # [Force Update] มั่นใจว่า summary เก็บค่าที่เราเลือกจริงๆ
                summary["scaling_method"] = final_sc_method
                
                st.session_state["transformed_df"]      = transformed_df
                st.session_state["_trans_target_saved"] = target_col
                st.session_state["trans_summary"]       = summary
                st.session_state["trans_confirmed"]     = True
                
                # ลบผล ML เก่าออกเพื่อให้ต้องเริ่มใหม่
                for _k in ["ml_result", "ml_metrics", "_fi_data", "ml_task_type"]:
                    st.session_state.pop(_k, None)
                
                from explainable.features.trace_log import log_transformation
                log_transformation(summary, enc_decisions, final_sc_method, drop_cols)
                
                from explainable.features.pipeline_state import commit_step
                commit_step("transformation", summary)
                
                # แจ้งเตือนแบบชัดเจน
                method_label = SCALING_LABELS.get(final_sc_method, final_sc_method)
                st.toast(f"Apply สำเร็จ! ใช้ {method_label}", icon="✅")
                
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")
                import traceback
                st.code(traceback.format_exc())

    # แสดง summary ถ้า apply แล้ว
    if st.session_state.get("trans_confirmed"):
        transformed_df = st.session_state["transformed_df"]
        summary        = st.session_state["trans_summary"]
        _render_summary(df, transformed_df, summary, target_col)
        
        method_name = SCALING_LABELS.get(summary["scaling_method"], summary["scaling_method"])
        st.success(f"✅ Transform สำเร็จ! ระบบจะใช้ **{method_name}** ในขั้นตอนถัดไป — กด Next Step เพื่อไป ML Process")

    # ── Navigation ────────────────────────────────────────────
    st.markdown("---")
    col1, _space, col2 = st.columns([0.8, 8, 0.8])

    with col1:
        if st.button("Back", type="secondary", width="stretch"):
            from app import navigate
            # restore main_df กลับไปก่อน transformation (ถ้ามี backup)
            if "_main_df_backup" in st.session_state:
                st.session_state["main_df"] = st.session_state.pop("_main_df_backup")
            st.session_state.pop("trans_confirmed", None)
            st.session_state.pop("transformed_df", None)
            navigate("eda")

    with col2:
        confirmed = st.session_state.get("trans_confirmed", False)
        if st.button(
            "Next Step", type="primary", width="stretch",
            disabled=not confirmed,
        ):
            from app import navigate
            # backup main_df ก่อน overwrite เพื่อให้ Back สามารถ restore ได้
            st.session_state["_main_df_backup"] = st.session_state.get("main_df")
            st.session_state["main_df"]      = st.session_state["transformed_df"]
            st.session_state["ml_target_col_preset"] = st.session_state.get("_trans_target_saved")
            navigate("ml_process")
        if not confirmed:
            st.caption(
                "กด Apply Transformation ก่อนไปขั้นตอนถัดไป",
                width="content", text_alignment="center",
            )