# Libraries
import streamlit as st
import datetime as dt_module
import io as io_module
import zipfile as zf_module

# Logic Import
from backend.core.insight.explain.explainer import *
from backend.core.session.pipeline_state import *
from backend.core.insight.model_guide.guide import *
from backend.function.export.html_export import *
from backend.function.export.df_export import *


# UI Import
from main import navigate
from user_interface.pages.Insight.insight_components.insight_compo import *

# Functions

# Render Page
def render_insight():
    from main_compo import page_header
    
    page_header(
        "Insight & Explainable",
        "ข้อมูลเชิงลึกและการอธิบายกระบวนการ",
    )
    
    competition_result = st.session_state.get("ml_result")
    if competition_result is None:
        st.warning("ไม่พบผล ML  กรุณา Run Model ก่อน")
        if st.button("กลับ ML Process"):
            navigate("model_process")
        return

    dataframe = st.session_state.get("transformed_df", st.session_state.get("main_df"))
    transformation_summary = st.session_state.get("trans_summary", {})
    target_column = (st.session_state.get("_trans_target_saved") or st.session_state.get("target_col"))
    best_model_key = competition_result["best_key"]
    best_model_label = competition_result["best_label"]
    best_hyperparameters = competition_result["best_params"]
    task_type = competition_result["task_type"]
    evaluation_metrics = st.session_state.get("ml_metrics", {})

    # Summary banner
    banner_items_list = [
        ("Best Model", best_model_label, "#7AA2F7"),
        ("Task", task_type.upper(), "#9ECE6A"),
        ("Dataset", f"{dataframe.shape[0]:,} × {dataframe.shape[1]}", "#E0AF68"),
    ]
    banner_columns = st.columns(3)
    for col, (label, value, color) in zip(banner_columns, banner_items_list):
        with col:
            st.markdown(
                f'<div style="background:{BACKGROUND_COLOR};border:1px solid {BORDER_COLOR};'
                f'border-radius:{BORDER_RADIUS};padding:{PADDING_STYLE};text-align:center">'
                f'<div style="color:{TEXT_DIM_COLOR};font-size:0.85rem;font-weight:600;'
                f'letter-spacing:0.07em;text-transform:uppercase;margin-bottom:8px">{label}</div>'
                f'<div style="color:{color};font-size:1.25rem;font-weight:800">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Fit model (cached)
    dataset_file_id = st.session_state.get("last_uploaded_file", "")
    caching_key = f"_xai_cache_{best_model_key}_{hash(str(sorted(best_hyperparameters.items())))}_{dataset_file_id}"
    if st.session_state.get("_xai_cache_id") != caching_key:
        with st.spinner(f"กำลัง train {best_model_label} เพื่อวิเคราะห์..."):
            try:
                fitted_model, X_train, X_test, _, y_test, _ = get_fitted_model(
                    dataframe, target_column, best_model_key, best_hyperparameters, transformation_summary,
                    missing_rules=st.session_state.get("missing_rules"),
                    outlier_rules=st.session_state.get("outlier_rules"),
                )
                st.session_state["_xai_model"]    = fitted_model
                st.session_state["_xai_X_train"]  = X_train
                st.session_state["_xai_X_test"]   = X_test
                st.session_state["_xai_y_test"]   = y_test
                st.session_state["_xai_cache_id"] = caching_key
                for session_key in list(st.session_state.keys()):
                    if session_key.startswith("_xai_perm_"):
                        del st.session_state[session_key]
            except Exception as error:
                st.error(f"เกิดข้อผิดพลาด: {error}")
                import traceback
                st.code(traceback.format_exc())
                return

    fitted_model  = st.session_state["_xai_model"]
    X_test = st.session_state["_xai_X_test"]
    y_test = st.session_state["_xai_y_test"]

    tabs_labels = ["Feature Importance", "Leaderboard", "Data Visualization", "Model Guide", "Pipeline Trace"]
    if has_comparison():
        tabs_labels.append("Diff Views")

    tabs_objects = st.tabs(tabs_labels)

    with tabs_objects[0]:
        st.markdown("<br>", unsafe_allow_html=True)
        render_importance(fitted_model, X_test, y_test, task_type)

    with tabs_objects[1]:
        st.markdown("<br>", unsafe_allow_html=True)
        render_leaderboard_insight(competition_result)

    with tabs_objects[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        render_viz_insight(dataframe, target_column, task_type)

    with tabs_objects[3]:
        st.markdown("<br>", unsafe_allow_html=True)
        render_guide(best_model_label, task_type, evaluation_metrics)

    with tabs_objects[4]:
        st.markdown("<br>", unsafe_allow_html=True)
        render_trace()

    if has_comparison():
        with tabs_objects[5]:
            st.markdown("<br>", unsafe_allow_html=True)
            render_comparison()

    st.divider()

    # Export + Nav
    feature_importance_df_export = st.session_state.get("_fi_data", {}).get("fi_df")
    timestamp_string = dt_module.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = best_model_label.replace(" ", "_")

    html_report_content = build_html_report(competition_result, evaluation_metrics, feature_importance_df_export).encode("utf-8")

    memory_buffer = io_module.BytesIO()
    with zf_module.ZipFile(memory_buffer, "w", zf_module.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("leaderboard.csv", build_leaderboard_df(competition_result["competition"]).to_csv(index=False))
        zip_file.writestr("predictions.csv", build_predictions_df(competition_result["y_test"], competition_result["y_pred"], task_type).to_csv(index=False))
        zip_file.writestr("metrics.csv",     "\n".join(["Metric,Value"] + [f"{key},{value}" for key, value in evaluation_metrics.items()]))
        if feature_importance_df_export is not None:
            zip_file.writestr("feature_importance.csv", feature_importance_df_export.to_csv(index=False))

    back_col, _space, html_col, zip_col, finish_col = st.columns([1.2, 2.7, 1.2, 1.2, 1.2])
    with back_col:
        if st.button("Back", type="secondary", width="stretch"):
            navigate("model_process")
    with html_col:
        st.download_button(
            "HTML Report",
            html_report_content,
            file_name=f"ml_report_{safe_model_name}_{timestamp_string}.html",
            mime="text/html",
            width="stretch",
        )
    with zip_col:
        st.download_button(
            "All CSV (ZIP)",
            memory_buffer.getvalue(),
            file_name=f"ml_results_{safe_model_name}_{timestamp_string}.zip",
            mime="application/zip",
            width="stretch",
        )
    with finish_col:
        @st.dialog("จบ Pipeline?")
        def confirm_finish():
            st.markdown(
                f"ผลลัพธ์ทั้งหมดของ **{best_model_label}** จะถูกล้างออก\n\n"
                "คุณต้องการจบและกลับไปหน้า Upload เพื่อเริ่มใหม่หรือไม่?"
            )
            confirm_col1, confirm_col2 = st.columns(2)
            with confirm_col1:
                if st.button("ยืนยัน", type="primary", width="stretch"):
                    delete_local()
                    for session_key in list(st.session_state.keys()):
                        del st.session_state[session_key]
                    navigate("upload")
            with confirm_col2:
                if st.button("ยกเลิก", width="stretch"):
                    st.rerun()

        if st.button("Finish", type="primary", width="stretch"):
            confirm_finish()