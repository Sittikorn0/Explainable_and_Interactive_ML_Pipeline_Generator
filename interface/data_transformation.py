import streamlit as st
import pandas as pd

from ml_process.logic.data_analyzer    import analyze_all
from ml_process.logic.data_transformer import apply_all
from ml_process.ui_components.scaling           import render_scaling, SCALING_LABELS
from ml_process.ui_components.encoding          import render_encoding
from ml_process.ui_components.feature_select    import render_feature_selection, render_leakage_check

def render_summary_view(dataframe: pd.DataFrame, transformed_dataframe: pd.DataFrame,
                    summary_dict: dict, target_column: str):
    st.markdown('<div class="section-header">รายงานสรุปการแปลงข้อมูล (SUMMARY REPORT)</div>', unsafe_allow_html=True)

    method_label = SCALING_LABELS.get(summary_dict["scaling_method"], summary_dict["scaling_method"]).upper()
    
    st.markdown(f"""
<div style="display: flex; justify-content: space-around; background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 20px; margin-bottom: 24px;">
    <div style="text-align: center;">
        <div style="color: #94A3B8; font-size: 0.85rem; font-family: monospace; margin-bottom: 4px;">คอลัมน์ตั้งต้น</div>
        <div style="color: #7AA2F7; font-size: 1.8rem; font-weight: bold;">{summary_dict["original_cols"]}</div>
    </div>
    <div style="text-align: center;">
        <div style="color: #94A3B8; font-size: 0.85rem; font-family: monospace; margin-bottom: 4px;">คัดออก (Drop)</div>
        <div style="color: #f85149; font-size: 1.8rem; font-weight: bold;">{summary_dict.get('dropped_cols', 0)}</div>
    </div>
    <div style="text-align: center;">
        <div style="color: #94A3B8; font-size: 0.85rem; font-family: monospace; margin-bottom: 4px;">คอลัมน์ที่เหลือ</div>
        <div style="color: #BB9AF7; font-size: 1.8rem; font-weight: bold;">{summary_dict["final_cols"]}</div>
    </div>
    <div style="text-align: center;">
        <div style="color: #94A3B8; font-size: 0.85rem; font-family: monospace; margin-bottom: 4px;">เทคนิคปรับสเกล</div>
        <div style="color: #F59E0B; font-size: 1.25rem; font-weight: bold; line-height: 1.5; margin-top: 6px;">{method_label}</div>
    </div>
</div>
""", unsafe_allow_html=True)

    with st.expander("ตัวอย่างข้อมูลหลังการแปลง (DATA PREVIEW)"):
        st.info(f"**ข้อมูล:** การปรับสเกล ({method_label}) จะถูกคำนวณและใช้งานจริงในขั้นตอน ML เพื่อความแม่นยำสูงสุดและป้องกันข้อมูลรั่วไหล")
        st.dataframe(transformed_dataframe.head(5), width="stretch")


    # Success message minimal style
    st.success(f"**[ READY ]** การแปลงข้อมูลเสร็จสมบูรณ์ ระบบพร้อมนำไปสอนโมเดลในขั้นตอนถัดไปโดยใช้เทคนิค **{method_label}**")


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
    
    st.metric("TARGET COLUMN", target_column)

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
    chosen_scaling_method = render_scaling(dataframe, target_column, scaling_analysis)
    leakage_drops_list  = render_leakage_check(leakage_analysis)
    
    # เรียงลำดับเพื่อให้การเปรียบเทียบใน Choice Tracker เสถียร (ป้องกันลำดับสลับไปมาใน set)
    dropped_columns      = sorted(list(set(render_feature_selection(dataframe, target_column, feature_selection_analysis) + leakage_drops_list)))

    st.markdown("---")
    
    if st.button("Apply Transformation",  type="primary", width="stretch"):
        with st.spinner("กำลังประมวลผล Transformation..."):
            try:
                # [Fail-safe] ดึงค่าตรงจาก Session State ของ Widget
                final_scaling_method = st.session_state.get("scaling_method", chosen_scaling_method)
                
                transformed_dataframe, transformation_summary = apply_all(
                    dataframe, encoding_decisions, final_scaling_method, dropped_columns, target_column,
                    scaling_analysis=scaling_analysis,
                    encoding_analysis=encoding_analysis,
                )
                
                # [Force Update] มั่นใจว่า summary เก็บค่าที่เราเลือกจริงๆ
                transformation_summary["scaling_method"] = final_scaling_method
                
                st.session_state["transformed_df"]      = transformed_dataframe
                st.session_state["_trans_target_saved"] = target_column
                st.session_state["trans_summary"]       = transformation_summary
                st.session_state["trans_confirmed"]     = True
                
                # [Persistence] Save data and metadata immediately
                from data_prepare.loading_data import save_transformed_df, save_trans_metadata
                save_transformed_df(transformed_dataframe)
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
    back_button_col, _space, next_button_col = st.columns([1.2, 7.6, 1.2])

    with back_button_col:
        if st.button("Back", type="secondary", width="stretch"):
            from app import navigate
            navigate("eda")

    with next_button_col:
        is_confirmed = st.session_state.get("trans_confirmed", False)
        if st.button(
            "Next Step", type="primary", width="stretch",
            disabled=not is_confirmed,
        ):
            from app import navigate
            st.session_state["ml_target_col_preset"] = st.session_state.get("_trans_target_saved")
            navigate("ml_process")
        if not is_confirmed:
            st.caption(
                "กด Apply Transformation ก่อนไปขั้นตอนถัดไป",
                width="content", text_alignment="center",
            )