"""ml_process/code.py — Orchestration (thin entry point)"""
import traceback
import streamlit as st

from ml_process.preprocess  import preprocess, detect_task
from ml_process.runner      import run_competition, get_available_models
from ml_process.evaluation  import get_metrics, show_metrics, show_leaderboard, show_confusion_matrix, show_pred_vs_actual
from ml_process.config      import MODEL_DESC
from ml_process.logic       import detect_leakage, compute_fi
from ml_process.views       import (
    render_target_info, render_competition_desc, render_model_cards,
    render_best_model_card, render_metrics_explain, render_cm_explain,
    render_scatter_explain, render_fi, render_viz, render_nav,
)


def render_ml_process():
    from app import page_header
    page_header("ML Process",
                "Auto Model Competition — Train ทุก model, เลือกตัวที่ดีสุด, Evaluate บน Test set")

    if st.session_state.get("main_df") is None:
        st.warning("ไม่พบข้อมูล — กรุณากลับไปทำ Data Preparation ก่อน")
        if st.button("← กลับไป Upload"):
            st.query_params["step"] = "upload"
            st.rerun()
        return

    df        = st.session_state["main_df"]
    file_name = st.session_state.get("last_uploaded_file", "Unknown File")
    st.info(f"**Current Dataset:** {file_name}  |  {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── 1. Target Column ──────────────────────────────────────────────────────
    cols   = df.columns.tolist()
    preset = (st.session_state.get("_trans_target_saved") or
              st.session_state.get("ml_target_col_preset") or
              st.session_state.get("target_col"))
    target_col = preset if preset and preset in cols else cols[-1]

    if not preset or preset not in cols:
        target_col = st.selectbox(
            "Target Column", cols, index=len(cols) - 1,
            key="ml_target_fallback",
            help="เลือก column ที่ต้องการทำนาย"
        )

    task_preview = detect_task(df, target_col)
    render_target_info(target_col, task_preview, df[target_col].nunique())

    # ── 2. Data Splitting ─────────────────────────────────────────────────────
    st.subheader("2. Data Splitting (80 / 20)")
    n = len(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows",      f"{n:,}")
    c2.metric("Train Set (80%)", f"{int(n * 0.8):,}")
    c3.metric("Test Set (20%)",  f"{n - int(n * 0.8):,}")

    # ── 3. Model Competition ──────────────────────────────────────────────────
    st.subheader("3. Model Competition + Auto Hyperparameter Tuning")
    render_competition_desc()

    avail = get_available_models(task_preview)
    render_model_cards(avail, MODEL_DESC)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("▶ Run All Models", type="primary"):
        st.session_state.pop("ml_result",  None)
        st.session_state.pop("ml_metrics", None)
        st.session_state.pop("_fi_data",   None)

        with st.spinner("กำลัง preprocess..."):
            try:
                trans_summary       = st.session_state.get("trans_summary", {})
                scaling_method      = trans_summary.get("scaling_method", "standard_scaler")
                label_encoding_cols = trans_summary.get("label_encoding_cols", [])
                X_train, X_test, y_train, y_test, task_type = preprocess(
                    df, target_col,
                    scaling_method=scaling_method,
                    label_encoding_cols=label_encoding_cols,
                )
                st.session_state["ml_task_type"] = task_type
                st.caption(
                    f"Scaling: **{scaling_method}** (fit บน Train set เท่านั้น — ป้องกัน Data Leakage)"
                )
            except Exception as e:
                st.error(f"Preprocess ล้มเหลว: {e}")
                return

        warnings = detect_leakage(df, target_col)
        if warnings:
            st.warning(
                "⚠️ **ตรวจพบ Data Leakage ที่อาจทำให้ค่า Evaluation สูงผิดปกติ:**\n\n" +
                "\n".join(f"- {w}" for w in warnings)
            )

        bar = st.progress(0, text="กำลัง train...")
        def _prog(label, i, total):
            bar.progress((i + 1) / total, text=f"Training {label}... ({i+1}/{total})")

        try:
            result  = run_competition(X_train, X_test, y_train, y_test, task_type, on_progress=_prog)
            metrics = get_metrics(result["y_test"], result["y_pred"], task_type)
            bar.empty()
            st.session_state["ml_result"]  = result
            st.session_state["ml_metrics"] = metrics
            st.rerun()
        except Exception as e:
            bar.empty()
            st.error(f"เกิดข้อผิดพลาด: {e}")
            st.code(traceback.format_exc())
            return

    # ── Results ───────────────────────────────────────────────────────────────
    result  = st.session_state.get("ml_result")
    metrics = st.session_state.get("ml_metrics")
    if not (result and metrics):
        render_nav(result)
        return

    task_type   = result["task_type"]
    best_label  = result["best_label"]
    best_key    = result["best_key"]
    best_params = result["best_params"]

    st.success(f"Best Model: **{best_label}**")

    tab_leader, tab_eval, tab_fi, tab_viz = st.tabs([
        "Leaderboard", "Evaluation", "Feature Importance",
        "Data Visualization",
    ])

    with tab_leader:
        show_leaderboard(result["competition"])
        if best_params:
            with st.expander(f"Best Hyperparameters ของ {best_label}"):
                for k, v in best_params.items():
                    st.write(f"- **{k}**: `{v}`")
        render_best_model_card(result, best_label)

    with tab_eval:
        st.caption(f"ผลลัพธ์จาก **{best_label}** บน Test set ที่ยังไม่เคยเห็น")
        if task_type == "classification" and all(v >= 1.0 for v in metrics.values()):
            st.error(
                "⚠️ **ค่า Metrics ทั้งหมด = 1.0 — สงสัยว่ามี Data Leakage!**\n\n"
                "ตรวจสอบ dataset ว่ามี column ที่คำนวณมาจาก target โดยตรงหรือไม่ "
                "(เช่น target_encoded, target_label, หรือ column ที่มีชื่อคล้าย target) "
                "ให้ย้อนกลับไป Transformation แล้วตัด column นั้นออก"
            )
        show_metrics(metrics)
        render_metrics_explain(metrics, task_type)
        st.divider()
        if task_type == "classification":
            show_confusion_matrix(result["y_test"], result["y_pred"])
            render_cm_explain(result["y_test"], result["y_pred"])
        else:
            show_pred_vs_actual(result["y_test"].values, result["y_pred"])
            render_scatter_explain()

    with tab_fi:
        trans_summary = st.session_state.get("trans_summary", {})
        fi_data = st.session_state.get("_fi_data")
        if fi_data is None or fi_data.get("model_key") != best_key:
            with st.spinner(f"คำนวณ Feature Importance ของ {best_label}..."):
                fi_df, fi_error = compute_fi(df, target_col, best_key, best_params, trans_summary)
                st.session_state["_fi_data"] = {
                    "model_key": best_key, "fi_df": fi_df, "error": fi_error
                }
            fi_data = st.session_state["_fi_data"]
        render_fi(fi_data.get("fi_df"), best_label, fi_data.get("error"))

    with tab_viz:
        render_viz(df)

    render_nav(result)
