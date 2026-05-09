import streamlit as st
import pandas as pd

from ml_process.logic.data_analyzer    import analyze_all
from ml_process.logic.data_transformer import apply_all
from ml_process.ui_components.scaling           import render_scaling, SCALING_LABELS
from ml_process.ui_components.encoding          import render_encoding
from ml_process.ui_components.feature_select    import render_feature_selection, render_leakage_check

def render_summary_view(dataframe: pd.DataFrame, transformed_dataframe: pd.DataFrame,
                    summary_dict: dict, target_column: str):
    st.markdown("---")
    st.markdown('<div class="section-header">SUMMARY REPORT</div>', unsafe_allow_html=True)

    method_label = SCALING_LABELS.get(summary_dict["scaling_method"], summary_dict["scaling_method"]).upper()
    
    st.markdown(f"""
<div class="premium-card premium-card-blue" style="padding: 24px !important;">
    <div style="display: flex; justify-content: space-between; text-align: center; gap: 20px;">
        <div style="flex: 1;">
            <div class="technical-label" style="justify-content: center; margin-bottom: 6px; font-size: 0.8rem;">ORIGINAL</div>
            <div class="technical-value" style="font-size: 1.4rem;">{summary_dict["original_cols"]}</div>
        </div>
        <div style="flex: 1; border-left: 1px solid rgba(148, 163, 184, 0.1);">
            <div class="technical-label" style="justify-content: center; margin-bottom: 6px; font-size: 0.8rem; color: #F59E0B;">PURGED</div>
            <div class="technical-value" style="font-size: 1.4rem; color: #F59E0B;">{summary_dict.get('dropped_cols', 0)}</div>
        </div>
        <div style="flex: 1; border-left: 1px solid rgba(148, 163, 184, 0.1);">
            <div class="technical-label" style="justify-content: center; margin-bottom: 6px; font-size: 0.8rem; color: #BB9AF7;">FINAL</div>
            <div class="technical-value" style="font-size: 1.4rem; color: #BB9AF7;">{summary_dict["final_cols"]}</div>
        </div>
        <div style="flex: 2; border-left: 1px solid rgba(148, 163, 184, 0.1);">
            <div class="technical-label" style="justify-content: center; margin-bottom: 6px; font-size: 0.8rem;">SCALING STRATEGY</div>
            <div class="technical-value" style="font-size: 1.1rem; letter-spacing: 0.05em; color: #7AA2F7;">{method_label}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    with st.expander("TRANSFORMED DATA PREVIEW"):
        st.markdown(f"""
<div style="font-family: monospace; font-size: 0.85rem; color: #94A3B8; margin-bottom: 12px;">
    [INFO] Scaling ({method_label}) will be applied during ML Process for maximum precision.
</div>
""", unsafe_allow_html=True)
        st.dataframe(transformed_dataframe.head(5), width="stretch")

    # Success message minimal style
    st.markdown(f"""
<div style="background: rgba(16, 185, 129, 0.05); border-left: 3px solid #10B981; padding: 16px 20px; border-radius: 4px; margin-top: 20px;">
    <div style="color: #10B981; font-family: monospace; font-size: 0.9rem; font-weight: 700; letter-spacing: 0.1em; margin-bottom: 4px;">[STATUS: READY]</div>
    <div style="color: #6EE7B7; font-size: 0.95rem;">
        Transform completed. System will use <b>{summary_dict["scaling_method"]}</b> in the next step.
    </div>
</div>
""", unsafe_allow_html=True)


def render_transformation():
    from app import page_header

    page_header(
        "Data Transformation",
        "แปลงข้อมูลให้พร้อมสำหรับ ML — ระบบวิเคราะห์และแนะนำวิธีที่เหมาะสมพร้อมเหตุผล",
    )

    if st.session_state.get("main_df") is None:
        from app import navigate
        navigate("upload")
        return

    is_cleaned = st.session_state.get("cleaning_confirmed", False)
    dataframe        = st.session_state["working_df"] if is_cleaned and "working_df" in st.session_state else st.session_state["main_df"]
    file_name = st.session_state.get("last_uploaded_file", "Unknown File")

    st.info(
        f"**Current Dataset:** {file_name}  "
        f"|  {dataframe.shape[0]:,} rows × {dataframe.shape[1]} columns"
    )

    # [Safety Check] If the summary in state doesn't match the current file or state, clear it
    if st.session_state.get("trans_confirmed"):
        summary = st.session_state.get("trans_summary", {})
        if summary.get("original_cols") != dataframe.shape[1]:
             st.session_state.pop("trans_confirmed", None)
             st.session_state.pop("trans_summary", None)
             st.session_state.pop("transformed_df", None)
             st.rerun()

    # Target Column
    columns_list = dataframe.columns.tolist()
    saved_target = st.session_state.get("target_col", columns_list[-1])
    target_column   = saved_target if saved_target in columns_list else columns_list[-1]
    
    st.markdown(f"""
<div class="premium-card premium-card-blue">
    <div class="technical-label" style="color: #7AA2F7; letter-spacing: 0.15em;">TARGET COLUMN</div>
    <div class="technical-value" style="color: #7AA2F7; font-size: 1.4rem;">
        {target_column}
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # วิเคราะห์ข้อมูล (cache ตาม df shape + target)
    dataframe_fingerprint = hash((dataframe.shape, tuple(dataframe.columns), str(dataframe.iloc[0].values.tolist()) if len(dataframe) else ""))
    cache_key = f"trans_analysis_v4_{dataframe_fingerprint}_{target_column}"
    if st.session_state.get("_trans_cache_key") != cache_key:
        with st.spinner("วิเคราะห์ข้อมูล..."):
            analysis_result = analyze_all(dataframe, target_column)
        st.session_state["_trans_analysis"]  = analysis_result
        st.session_state["_trans_cache_key"] = cache_key
    else:
        analysis_result = st.session_state["_trans_analysis"]

    encoding_analysis = analysis_result["encoding"]
    scaling_analysis  = analysis_result["scaling"]
    leakage_analysis  = analysis_result.get("leakage", [])
    feature_selection_analysis = analysis_result["feature_selection"]

    # Render sections
    encoding_decisions  = render_encoding(dataframe, target_column, encoding_analysis)
    st.markdown("---")
    chosen_scaling_method = render_scaling(dataframe, target_column, scaling_analysis)
    st.markdown("---")
    leakage_drops_list  = render_leakage_check(leakage_analysis)
    st.markdown("---")
    
    # เรียงลำดับเพื่อให้การเปรียบเทียบใน Choice Tracker เสถียร (ป้องกันลำดับสลับไปมาใน set)
    dropped_columns      = sorted(list(set(render_feature_selection(dataframe, target_column, feature_selection_analysis) + leakage_drops_list)))

    st.markdown("---")
    
    if st.button("Apply Transformation",  type="primary", width="stretch"):
        with st.spinner("กำลังประมวลผล Transformation..."):
            try:
                # [Fail-safe] ดึงค่าตรงจาก Session State ของ Widget
                final_scaling_method = st.session_state.get("scaling_method", chosen_scaling_method)
                
                transformed_dataframe, transformation_summary = apply_all(
                    dataframe, encoding_decisions, final_scaling_method, dropped_columns, target_column
                )
                
                # [Force Update] มั่นใจว่า summary เก็บค่าที่เราเลือกจริงๆ
                transformation_summary["scaling_method"] = final_scaling_method
                
                st.session_state["transformed_df"]      = transformed_dataframe
                st.session_state["_trans_target_saved"] = target_column
                st.session_state["trans_summary"]       = transformation_summary
                st.session_state["trans_confirmed"]     = True
                
                # [Persistence] Save data and metadata immediately
                from data_prepare.loading_data import save_to_local, save_trans_metadata
                save_to_local(transformed_dataframe, file_name)
                save_trans_metadata(transformation_summary, target_column)
                
                # ลบผล ML เก่าออกเพื่อให้ต้องเริ่มใหม่
                for key_to_remove in ["ml_result", "ml_metrics", "_fi_data", "ml_task_type"]:
                    st.session_state.pop(key_to_remove, None)
                
                from explainable.state_manager.trace_log import log_transformation
                log_transformation(transformation_summary, encoding_decisions, final_scaling_method, dropped_columns)
                
                from explainable.state_manager.pipeline_state import commit_step
                commit_step("transformation", transformation_summary)
                
                method_label = SCALING_LABELS.get(final_scaling_method, final_scaling_method)
                st.toast(f"Apply สำเร็จ! ใช้ {method_label}", icon="✅")
                
            except Exception as error:
                st.error(f"เกิดข้อผิดพลาด: {error}")
                import traceback
                st.code(traceback.format_exc())

    # แสดง summary ถ้า apply แล้ว
    if st.session_state.get("trans_confirmed"):
        transformed_dataframe = st.session_state["transformed_df"]
        transformation_summary        = st.session_state["trans_summary"]
        render_summary_view(dataframe, transformed_dataframe, transformation_summary, target_column)

    # Navigation
    st.markdown("---")
    back_button_col, _space, next_button_col = st.columns([1.2, 7.6, 1.2])

    with back_button_col:
        if st.button("Back", type="secondary", width="stretch"):
            from app import navigate
            if "_main_df_backup" in st.session_state:
                st.session_state["main_df"] = st.session_state.pop("_main_df_backup")
            st.session_state.pop("trans_confirmed", None)
            st.session_state.pop("transformed_df", None)
            navigate("eda")

    with next_button_col:
        is_confirmed = st.session_state.get("trans_confirmed", False)
        if st.button(
            "Next Step", type="primary", width="stretch",
            disabled=not is_confirmed,
        ):
            from app import navigate
            st.session_state["_main_df_backup"] = st.session_state.get("main_df")
            st.session_state["main_df"]      = st.session_state["transformed_df"]
            st.session_state["ml_target_col_preset"] = st.session_state.get("_trans_target_saved")
            navigate("ml_process")
        if not is_confirmed:
            st.caption(
                "กด Apply Transformation ก่อนไปขั้นตอนถัดไป",
                width="content", text_alignment="center",
            )