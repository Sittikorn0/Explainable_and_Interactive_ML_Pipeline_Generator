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
    st.subheader("สรุปผล Data Transformation")

    from interface.ui_helpers import render_metrics_row
    metrics_data = [
        ("Original Columns", str(summary_dict["original_cols"])),
        ("Dropped", f"-{summary_dict['dropped_cols']}"),
        ("After Encoding", str(summary_dict["final_cols"])),
        ("Scaling Method", SCALING_LABELS.get(summary_dict["scaling_method"], "—")),
    ]
    render_metrics_row(metrics_data)

    with st.expander("ดู Transformed Data (5 rows แรก)"):
        st.caption(f"หมายเหตุ: ตัวเลขใน Preview นี้ยังไม่ถูก Scale ({SCALING_LABELS.get(summary_dict['scaling_method'], summary_dict['scaling_method'])}) โดยระบบจะนำไปคำนวณจริงในขั้นตอน ML Process เพื่อความแม่นยำสูงสุด")
        st.dataframe(transformed_dataframe.head(5), width="stretch")

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

    # Target Column
    columns_list = dataframe.columns.tolist()
    saved_target = st.session_state.get("target_col", columns_list[-1])
    target_column   = saved_target if saved_target in columns_list else columns_list[-1]
    st.markdown("**Target Column** (จะไม่ถูก transform)")
    st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-left:5px solid #58a6ff;
border-radius:6px;padding:10px 14px;font-family:monospace;font-size:1.2rem;
color:#58a6ff;font-weight:600">
{target_column}
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # วิเคราะห์ข้อมูล (cache ตาม df shape + target)
    dataframe_fingerprint = hash((dataframe.shape, tuple(dataframe.columns), str(dataframe.iloc[0].values.tolist()) if len(dataframe) else ""))
    cache_key = f"trans_analysis_{dataframe_fingerprint}_{target_column}"
    if st.session_state.get("_trans_cache_key") != cache_key:
        with st.spinner("วิเคราะห์ข้อมูล..."):
            analysis_result = analyze_all(dataframe, target_column)
        st.session_state["_trans_analysis"]  = analysis_result
        st.session_state["_trans_cache_key"] = cache_key
    else:
        analysis_result = st.session_state["_trans_analysis"]

    encoding_analysis = analysis_result["encoding"]
    scaling_analysis  = analysis_result["scaling"]
    feature_selection_analysis  = analysis_result["feature_selection"]

    # Render sections
    encoding_decisions  = render_encoding(dataframe, target_column, encoding_analysis)
    st.markdown("---")
    chosen_scaling_method = render_scaling(dataframe, target_column, scaling_analysis)
    st.markdown("---")
    leakage_drops_list  = render_leakage_check(dataframe, target_column)
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
        
        method_name_label = SCALING_LABELS.get(transformation_summary["scaling_method"], transformation_summary["scaling_method"])
        st.success(f"✅ Transform สำเร็จ! ระบบจะใช้ **{method_name_label}** ในขั้นตอนถัดไป — กด Next Step เพื่อไป ML Process")

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