# Libraries
import streamlit as st
import traceback

# Logic Import
from backend.function.analyzer.task_detection import detect_task
from backend.core.model_training.trainer.train_model import *
from backend.function.analyzer.leakage import *
from backend.core.insight.trace_log import *
from backend.core.session.pipeline_state import *
from backend.core.session.state import *
from backend.core.model_training.evaluation.eval import *
from backend.core.model_training.preprocess.pipeline import preprocess
from backend.function.export.df_export import *
from backend.function.export.html_export import *
from backend.core.model_training.feature_importance import *

# UI Import
from main import navigate
from user_interface.pages.Model_process.model_components.model_compo import *

# Functions


# Render Page
def render_model_process():
    from main_compo import page_header
    
    page_header(
        "Model Process",
        "Auto Model Competition: Train ทุก model, เลือกตัวที่ดีสุด, Evaluate บน Test set",
    )
    
    if st.session_state.get("main_df") is None:
        st.warning("ไม่พบข้อมูล  กรุณากลับไปทำ Data Preparation ก่อน")
        if st.button("กลับไป Upload"):
            navigate("upload")
        return
    
    dataframe = st.session_state.get("transformed_df", st.session_state.get("main_df"))
    file_name = st.session_state.get("last_uploaded_file", "Unknown File")
    st.info(f"**Current Dataset:** {file_name}  |  {dataframe.shape[0]:,} rows × {dataframe.shape[1]} columns")

    # 1. Target Column
    columns_list   = dataframe.columns.tolist()
    # ใช้ _trans_target_saved (set โดย transformation step) เป็น source of truth
    # fallback ไป target_col ซึ่ง set โดย upload step
    preset_target = (st.session_state.get("_trans_target_saved") or
              st.session_state.get("target_col"))
    target_column = preset_target if preset_target and preset_target in columns_list else columns_list[-1]

    if not preset_target or preset_target not in columns_list:
        target_column = st.selectbox(
            "Target Column", columns_list, index=len(columns_list) - 1,
            key="ml_target_fallback",
            help="เลือก column ที่ต้องการทำนาย"
        )

    raw_df_for_task = st.session_state.get("main_df", dataframe)
    task_preview = detect_task(raw_df_for_task, target_column)
    render_target_info(target_column, task_preview, dataframe[target_column].nunique())

    # 2. Data Splitting
    st.subheader("Data Splitting (80 / 20)")
    total_rows = len(dataframe)
    split_col1, split_col2, split_col3 = st.columns(3)
    split_col1.metric("Total Rows",      f"{total_rows:,}")
    split_col2.metric("Train Set (80%)", f"{int(total_rows * 0.8):,}")
    split_col3.metric("Test Set (20%)",  f"{total_rows - int(total_rows * 0.8):,}")

    # 3. Model Competition
    st.subheader("Model Competition + Auto Hyperparameter Tuning")
    render_competition_desc()

    available_models = get_available_models(task_preview)
    render_model_cards(available_models, MODEL_DESC)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("▶ Run All Models", type="primary"):
        st.session_state.pop("ml_result", None)
        st.session_state.pop("ml_metrics", None)
        st.session_state.pop("_fi_data", None)
        st.session_state.pop("_ml_scaling_used", None)
        st.session_state.pop("_ml_leakage_warnings", None)
        
        for key in list(st.session_state.keys()):
            if key.startswith("_xai_") or key.startswith("_eval_cache_"):
                st.session_state.pop(key, None)

        with st.spinner("กำลัง preprocess..."):
            try:
                transformation_summary = st.session_state.get("trans_summary", {})
                scaling_method = transformation_summary.get("scaling_method", "standard_scaler")
                encoding_decisions = transformation_summary.get("encoding_decisions") or None
                X_train, X_test, y_train, y_test, task_type = preprocess(
                    dataframe, target_column,
                    scaling_method=scaling_method,
                    missing_rules=st.session_state.get("missing_rules"),
                    outlier_rules=st.session_state.get("outlier_rules"),
                    encoding_decisions=encoding_decisions,
                )
                st.session_state["ml_task_type"]    = task_type
                st.session_state["_ml_scaling_used"] = scaling_method
            except Exception as error:
                st.error(f"Preprocess ล้มเหลว: {error}")
                render_nav(None)
                return

        # เก็บ leakage warnings ไว้ใน session_state เพื่อแสดงหลัง rerun
        st.session_state["_ml_leakage_warnings"] = analyze_leakage(dataframe, target_column)

        progress_bar = st.progress(0, text="กำลัง train...")
        def progress_callback(label, current_index, total_models):
            progress_bar.progress((current_index + 1) / total_models, text=f"Training {label}... ({current_index+1}/{total_models})")

        try:
            competition_result  = run_competition(X_train, X_test, y_train, y_test, task_type, on_progress=progress_callback)
            evaluation_metrics = get_metrics(competition_result["y_test"], competition_result["y_pred"], task_type)
            progress_bar.empty()
            st.session_state["ml_result"]  = competition_result
            st.session_state["ml_metrics"] = evaluation_metrics
            
            log_model_process(competition_result, evaluation_metrics)
            commit_step("model_process", evaluation_metrics)
            save_ml_cache(
                competition_result, evaluation_metrics,
                st.session_state.get("trans_summary", {}),
                target_column,
                scaling_used=st.session_state.get("_ml_scaling_used"),
                leakage_warnings=st.session_state.get("_ml_leakage_warnings")
            )
            st.rerun()
        except Exception as error:
            progress_bar.empty()
            st.error(f"เกิดข้อผิดพลาด: {error}")
            st.code(traceback.format_exc())
            render_nav(None)
            return

    # Results
    competition_result  = st.session_state.get("ml_result")
    evaluation_metrics = st.session_state.get("ml_metrics")
    if not (competition_result and evaluation_metrics):
        render_nav(competition_result)
        return

    task_type   = competition_result["task_type"]
    best_model_label  = competition_result["best_label"]
    best_model_key    = competition_result["best_key"]
    best_hyperparameters = competition_result["best_params"]

    # แสดง scaling info และ leakage warnings ที่เก็บไว้ก่อน rerun
    scaling_used = st.session_state.get("_ml_scaling_used")
    if scaling_used:
        st.caption(f"Scaling: **{scaling_used}** (fit บน Train set เท่านั้นเพื่อป้องกัน Data Leakage)")

    leakage_items = st.session_state.get("_ml_leakage_warnings", [])
    high_risk_items = [item for item in leakage_items if item["severity"] == "high"]
    
    if high_risk_items:
        leakage_color  = "#f85149"
        leakage_title  = "ตรวจพบความเสี่ยง Data Leakage ค่า Metric อาจสูงผิดปกติ"

        rows_html_content = ""
        for item in high_risk_items:
            severity_color = "#f85149"
            severity_label = "HIGH RISK"
            reasons_string = " · ".join(item["reasons"])
            rows_html_content += (
                f'<div style="display:flex;gap:12px;align-items:flex-start;'
                f'padding:8px 0;border-bottom:1px solid #30363d">'
                f'<span style="background:{severity_color}22;color:{severity_color};'
                f'font-size:0.75rem;font-weight:700;padding:2px 7px;border-radius:4px;'
                f'flex-shrink:0;margin-top:2px">{severity_label}</span>'
                f'<div><code style="color:#e6edf3">{item["col"]}</code>'
                f'<div style="color:#8b949e;font-size:0.85rem;margin-top:2px">{reasons_string}</div>'
                f'</div></div>'
            )

        st.markdown(
            f'<div style="background-color: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px;'
            f'padding: 20px 24px; margin: 16px 0">'
            f'<div style="color: #EF4444; font-weight: bold; font-family: monospace; font-size: 1.05rem; margin-bottom: 4px">'
            f'[!] DATA LEAKAGE WARNING</div>'
            f'<div style="color: #E2E8F0; font-size: 0.95rem; margin-bottom: 16px">{leakage_title}</div>'
            f'{rows_html_content}'
            f'<div style="color: #94A3B8; font-size: 0.85rem; margin-top: 16px; border-top: 1px solid rgba(239, 68, 68, 0.1); padding-top: 12px;">'
            f'แนะนำ: ย้อนกลับไปหน้า <b>Transformation</b> เพื่อตัดคอลัมน์เหล่านี้ออกก่อนประมวลผลใหม่</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.success(f"Best Model: **{best_model_label}**")

    tab_leaderboard, tab_evaluation = st.tabs([
        "Leaderboard", "Evaluation",
    ])

    with tab_leaderboard:
        show_leaderboard(competition_result["competition"])
        if best_hyperparameters:
            with st.expander(f"Best Hyperparameters ของ {best_model_label}"):
                for param_name, param_value in best_hyperparameters.items():
                    st.write(f"- **{param_name}**: `{param_value}`")
        render_best_model_card(competition_result, best_model_label)
        leaderboard_csv = build_leaderboard_df(competition_result["competition"]).to_csv(index=False).encode("utf-8-sig")
        st.markdown('<div style="margin-top:24px"></div>', unsafe_allow_html=True)
        st.download_button("Leaderboard CSV", leaderboard_csv, file_name=f"leaderboard_{best_model_label}.csv", mime="text/csv")

    with tab_evaluation:
        # ── Model Selector ──
        competition_data = competition_result["competition"]
        valid_models = {k: v for k, v in competition_data.items() if v["cv_score"] is not None}
        model_options = list(valid_models.keys())
        model_labels  = {k: v["label"] for k, v in valid_models.items()}

        saved_eval_key = st.session_state.get("eval_model_key", best_model_key)
        if saved_eval_key not in model_options:
            saved_eval_key = best_model_key

        selected_key = st.selectbox(
            "เลือก Model สำหรับ Evaluation",
            options=model_options,
            format_func=lambda k: f"{model_labels[k]}{'(Best)' if k == best_model_key else ''}",
            index=model_options.index(saved_eval_key),
            key="eval_model_key",
        )

        # ── ถ้าเลือก best model ใช้ผลเดิม ถ้าเลือกตัวอื่น refit + predict ──
        if selected_key == best_model_key:
            eval_y_pred   = competition_result["y_pred"]
            eval_y_test   = competition_result["y_test"]
            eval_metrics  = evaluation_metrics
            eval_label    = best_model_label
        else:
            cache_key = f"_eval_cache_{selected_key}"
            if st.session_state.get(cache_key) is None:
                with st.spinner(f"กำลัง refit {model_labels[selected_key]}..."):
                    try:
                        transformation_summary = st.session_state.get("trans_summary", {})
                        scaling_decisions = transformation_summary.get("scaling_decisions") or None
                        scaling_method    = transformation_summary.get("scaling_method", "standard_scaler")
                        encoding_decisions = transformation_summary.get("encoding_decisions") or None
                        X_tr, X_te, y_tr, y_te, _ = preprocess(
                            dataframe, target_column,
                            scaling_method=scaling_method,
                            scaling_decisions=scaling_decisions,
                            missing_rules=st.session_state.get("missing_rules"),
                            outlier_rules=st.session_state.get("outlier_rules"),
                            encoding_decisions=encoding_decisions,
                        )
                        sel_model = get_model_map()[selected_key]()
                        sel_params = competition_data[selected_key].get("best_params", {})
                        if sel_params:
                            try:
                                sel_model.set_params(**sel_params)
                            except Exception:
                                pass
                        from sklearn.utils.class_weight import compute_sample_weight
                        fit_kw = {}
                        if task_type == "classification" and selected_key in ("gradient_boosting", "xgboost"):
                            fit_kw["sample_weight"] = compute_sample_weight("balanced", y_tr)
                        sel_model.fit(X_tr, y_tr, **fit_kw)
                        sel_pred = sel_model.predict(X_te)

                        from sklearn.preprocessing import LabelEncoder
                        if task_type == "classification":
                            le = LabelEncoder()
                            le.fit(y_tr)
                            sel_pred = le.inverse_transform(sel_pred) if hasattr(le, "classes_") else sel_pred

                        sel_metrics = get_metrics(y_te, sel_pred, task_type)
                        st.session_state[cache_key] = {"y_pred": sel_pred, "y_test": y_te, "metrics": sel_metrics}
                    except Exception as err:
                        st.error(f"Refit ล้มเหลว: {err}")
                        st.stop()

            cached = st.session_state[cache_key]
            eval_y_pred  = cached["y_pred"]
            eval_y_test  = cached["y_test"]
            eval_metrics = cached["metrics"]
            eval_label   = model_labels[selected_key]

        st.caption(f"ผลลัพธ์จาก **{eval_label}** บน Test set ที่ยังไม่เคยเห็น")
        # 1.0 Guard - Perfect Metrics Warning
        is_suspiciously_perfect = False
        if task_type == "classification":
            is_suspiciously_perfect = all(v >= 0.9999 for v in eval_metrics.values() if isinstance(v, (int, float)))
        else:
            r2 = eval_metrics.get("R² Score", 0)
            is_suspiciously_perfect = r2 >= 0.9999

        if is_suspiciously_perfect:
            feature_importance_rows = ""
            try:
                transformation_summary = st.session_state.get("trans_summary", {})
                feature_importance_df, _ = compute_fi(dataframe, target_column, best_model_key, best_hyperparameters, transformation_summary)
                if feature_importance_df is not None:
                    top_features = feature_importance_df[feature_importance_df["Importance"] > 0].head(5)
                    for _, row in top_features.iterrows():
                        importance_percent = row["Importance"] * 100
                        feature_importance_rows += (
                            f'<div style="display:flex;align-items:center;gap:10px;padding:4px 0">'
                            f'<code style="color:#e6edf3;min-width:160px;font-size:0.85rem">{row["Feature"]}</code>'
                            f'<div style="flex:1;background:#21262d;border-radius:4px;height:8px">'
                            f'<div style="width:{min(importance_percent,100)}%;background:#f85149;height:8px;border-radius:4px"></div></div>'
                            f'<span style="color:#f85149;font-size:0.85rem;min-width:45px">{importance_percent:.1f}%</span>'
                            f'</div>'
                        )
            except Exception:
                pass

            st.markdown(
                f'<div style="background: rgba(248, 81, 73, 0.05); border-left: 4px solid #f85149; border-radius: 4px;'
                f'padding: 20px 24px; margin: 12px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1)">'
                f'<div style="display: flex; align-items: baseline; gap: 10px; margin-bottom: 10px;">'
                f'<span style="color: #f85149; font-family: monospace; font-weight: 900; font-size: 1.2rem;">[!]</span>'
                f'<strong style="color: #f85149; font-size: 1.05rem; letter-spacing: 0.02em; text-transform: uppercase;">Anomaly Detected: Perfect Score (1.000)</strong>'
                f'</div>'
                f'<div style="color: #8b949e; font-size: 0.92rem; line-height: 1.7; margin-bottom: 16px">'
                f'โมเดลได้คะแนนสูงสุดซึ่งมักเกิดจาก <b>Data Leakage</b> (ความผิดปกติของข้อมูลที่มีเฉลยปนอยู่) '
                f'หรือความผิดพลาดในการเลือกฟีเจอร์ที่ไม่เป็นกลางต่อผลลัพธ์'
                f'</div>'
                + ('<div style="color:#565f89; font-size:0.8rem; text-transform:uppercase; font-weight:600; margin-bottom:8px; letter-spacing:0.05em">Suspicious Features:</div>' if feature_importance_rows else '')
                + f'{feature_importance_rows}'
                f'<div style="color: #565f89; font-size: 0.85rem; margin-top: 20px; border-top: 1px solid rgba(148, 163, 184, 0.1); padding-top: 16px">'
                f'<b>Resolution:</b> กลับไปหน้า <b>Transformation</b> และพิจารณาตัด (Drop) ฟีเจอร์ที่มีอิทธิพลสูงผิดปกติออก</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        show_metrics(eval_metrics)
        render_metrics_explain(eval_metrics, task_type)
        st.divider()
        if task_type == "classification":
            show_confusion_matrix(eval_y_test, eval_y_pred)
            render_cm_explain(eval_y_test, eval_y_pred)
        else:
            ev_col1, ev_col2 = st.columns(2)
            with ev_col1:
                st.markdown("**Actual vs Predicted**")
                show_pred_vs_actual(eval_y_test.values, eval_y_pred)
                render_scatter_explain()
            with ev_col2:
                st.markdown("**Residual Plot (ความคลาดเคลื่อน)**")
                show_residual_plot(eval_y_test.values, eval_y_pred)
                st.markdown("""
                <div style="background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px;
                padding: 16px 20px; margin: 12px 0; font-size: 0.9rem; color: #94A3B8; line-height: 1.8">
                  <b style="color: #E2E8F0;">วิธีอ่าน Residual Plot</b><br>
                  • ดูความกระจายของ <b style="color: #F59E0B;">Error</b> เทียบกับค่าที่ทำนาย<br>
                  • จุดควรกระจายรอบเส้น 0 แบบไม่มีรูปแบบ (Random)<br>
                  • ถ้าจุดมีรูปแบบชัดเจน (เช่น เป็นรูปกรวย) แปลว่า model ยังมีจุดอ่อนบางพื้นที่
                </div>""", unsafe_allow_html=True)

            st.divider()
            st.markdown("**Error Distribution (การกระจายตัวของความผิดพลาด)**")
            show_error_dist(eval_y_test.values, eval_y_pred)
            st.markdown("""
            <div style="background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px;
            padding: 16px 20px; margin: 12px 0; font-size: 0.9rem; color: #94A3B8; line-height: 1.8">
              • กราฟนี้แสดงว่า model มักทำนายผิดพลาดในช่วงไหนมากที่สุด<br>
              • <b style="color: #BB9AF7;">รูประฆังคว่ำ (Normal)</b> ที่จุด 0 หมายถึง model มีความแม่นยำสูงสม่ำเสมอ
            </div>""", unsafe_allow_html=True)
        predictions_csv = build_predictions_df(eval_y_test, eval_y_pred, task_type).to_csv(index=False).encode("utf-8-sig")
        st.markdown('<div style="margin-top:24px"></div>', unsafe_allow_html=True)
        st.download_button("Predictions CSV", predictions_csv, file_name=f"predictions_{eval_label}.csv", mime="text/csv")


    render_nav(competition_result)