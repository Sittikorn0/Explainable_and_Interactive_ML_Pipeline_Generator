import traceback
import streamlit as st

from ml_process.features.preprocessing  import preprocess, detect_task
from ml_process.features.runner      import run_competition, get_available_models
from ml_process.features.evaluation  import get_metrics, show_metrics, show_leaderboard, show_confusion_matrix, show_pred_vs_actual
from ml_process.features.export      import build_leaderboard_df, build_predictions_df
from ml_process.features.config      import MODEL_DESC
from ml_process.features.logic       import detect_leakage, analyze_leakage, compute_fi
from ml_process.features.views       import (
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
            from app import navigate
            navigate("upload")
        return

    df        = st.session_state["main_df"]
    file_name = st.session_state.get("last_uploaded_file", "Unknown File")
    st.info(f"**Current Dataset:** {file_name}  |  {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── 1. Target Column ──────────────────────────────────────────────────────
    cols   = df.columns.tolist()
    # ใช้ _trans_target_saved (set โดย transformation step) เป็น source of truth
    # fallback ไป target_col ซึ่ง set โดย upload step
    preset = (st.session_state.get("_trans_target_saved") or
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
        st.session_state.pop("ml_result",             None)
        st.session_state.pop("ml_metrics",            None)
        st.session_state.pop("_fi_data",              None)
        st.session_state.pop("_ml_scaling_used",      None)
        st.session_state.pop("_ml_leakage_warnings",  None)
        for _k in [k for k in st.session_state.keys() if k.startswith("_xai_")]:
            st.session_state.pop(_k, None)

        with st.spinner("กำลัง preprocess..."):
            try:
                trans_summary  = st.session_state.get("trans_summary", {})
                scaling_method = trans_summary.get("scaling_method", "standard_scaler")
                X_train, X_test, y_train, y_test, task_type = preprocess(
                    df, target_col,
                    scaling_method=scaling_method,
                )
                st.session_state["ml_task_type"]    = task_type
                st.session_state["_ml_scaling_used"] = scaling_method
            except Exception as e:
                st.error(f"Preprocess ล้มเหลว: {e}")
                render_nav(None)
                return

        # เก็บ leakage warnings ไว้ใน session_state เพื่อแสดงหลัง rerun
        st.session_state["_ml_leakage_warnings"] = analyze_leakage(df, target_col)

        bar = st.progress(0, text="กำลัง train...")
        def _prog(label, i, total):
            bar.progress((i + 1) / total, text=f"Training {label}... ({i+1}/{total})")

        try:
            result  = run_competition(X_train, X_test, y_train, y_test, task_type, on_progress=_prog)
            metrics = get_metrics(result["y_test"], result["y_pred"], task_type)
            bar.empty()
            st.session_state["ml_result"]  = result
            st.session_state["ml_metrics"] = metrics
            from explainable.features.trace_log import log_model_process
            log_model_process(result, metrics)
            from data_prepare.features.loading_data import save_ml_cache
            save_ml_cache(
                result, metrics,
                st.session_state.get("trans_summary", {}),
                target_col,
            )
            st.rerun()
        except Exception as e:
            bar.empty()
            st.error(f"เกิดข้อผิดพลาด: {e}")
            st.code(traceback.format_exc())
            render_nav(None)
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

    # แสดง scaling info และ leakage warnings ที่เก็บไว้ก่อน rerun
    scaling_used = st.session_state.get("_ml_scaling_used")
    if scaling_used:
        st.caption(f"Scaling: **{scaling_used}** (fit บน Train set เท่านั้น — ป้องกัน Data Leakage)")

    leakage_items = st.session_state.get("_ml_leakage_warnings", [])
    if leakage_items:
        high   = [x for x in leakage_items if x["severity"] == "high"]
        medium = [x for x in leakage_items if x["severity"] == "medium"]

        color  = "#f85149" if high else "#d29922"
        title  = "ตรวจพบ Data Leakage — ค่า Metric อาจสูงผิดปกติ" if high \
                 else "พบ column ที่น่าสงสัย — ควรตรวจสอบ"

        rows_html = ""
        for item in leakage_items:
            sev_color = "#f85149" if item["severity"] == "high" else \
                        "#d29922" if item["severity"] == "medium" else "#8b949e"
            sev_label = {"high": "HIGH", "medium": "MED", "low": "LOW"}[item["severity"]]
            reasons_str = " · ".join(item["reasons"])
            rows_html += (
                f'<div style="display:flex;gap:12px;align-items:flex-start;'
                f'padding:8px 0;border-bottom:1px solid #30363d">'
                f'<span style="background:{sev_color}22;color:{sev_color};'
                f'font-size:0.75rem;font-weight:700;padding:2px 7px;border-radius:4px;'
                f'flex-shrink:0;margin-top:2px">{sev_label}</span>'
                f'<div><code style="color:#e6edf3">{item["col"]}</code>'
                f'<div style="color:#8b949e;font-size:0.85rem;margin-top:2px">{reasons_str}</div>'
                f'</div></div>'
            )

        st.markdown(
            f'<div style="background:#1a0f0f;border:1px solid {color};border-radius:10px;'
            f'padding:16px 20px;margin:12px 0">'
            f'<div style="color:{color};font-weight:700;font-size:1rem;margin-bottom:4px">'
            f'Data Leakage Warning</div>'
            f'<div style="color:#c9d1d9;font-size:0.9rem;margin-bottom:12px">{title}</div>'
            f'{rows_html}'
            f'<div style="color:#8b949e;font-size:0.85rem;margin-top:12px">'
            f'ย้อนกลับไป <b>Transformation</b> แล้ว drop column ที่น่าสงสัยออกก่อน Run ใหม่</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

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
        lb_csv = build_leaderboard_df(result["competition"]).to_csv(index=False).encode("utf-8-sig")
        st.markdown('<div style="margin-top:24px"></div>', unsafe_allow_html=True)
        st.download_button("Leaderboard CSV", lb_csv, file_name=f"leaderboard_{best_label}.csv", mime="text/csv")

    with tab_eval:
        st.caption(f"ผลลัพธ์จาก **{best_label}** บน Test set ที่ยังไม่เคยเห็น")
        if task_type == "classification" and all(v >= 0.9999 for v in metrics.values()):
            # คำนวณ feature importance จาก best model เพื่อหาตัวการ
            fi_rows = ""
            try:
                trans_summary = st.session_state.get("trans_summary", {})
                fi_df, _ = compute_fi(df, target_col, best_key, best_params, trans_summary)
                if fi_df is not None:
                    top = fi_df[fi_df["Importance"] > 0].head(5)
                    for _, row in top.iterrows():
                        pct = row["Importance"] * 100
                        bar = int(pct / 5)
                        fi_rows += (
                            f'<div style="display:flex;align-items:center;gap:10px;padding:4px 0">'
                            f'<code style="color:#e6edf3;min-width:160px;font-size:0.85rem">{row["Feature"]}</code>'
                            f'<div style="flex:1;background:#21262d;border-radius:4px;height:8px">'
                            f'<div style="width:{min(bar*5,100)}%;background:#f85149;height:8px;border-radius:4px"></div></div>'
                            f'<span style="color:#f85149;font-size:0.85rem;min-width:45px">{pct:.1f}%</span>'
                            f'</div>'
                        )
            except Exception:
                pass

            fi_section = (
                f'<div style="margin-top:12px">'
                f'<div style="color:#8b949e;font-size:0.875rem;margin-bottom:8px">'
                f'Feature ที่ model ใช้ตัดสินใจ (feature importance):</div>'
                f'{fi_rows}</div>'
            ) if fi_rows else ""

            st.markdown(
                f'<div style="background:#1a0f0f;border:1px solid #f85149;border-radius:10px;'
                f'padding:16px 20px;margin:8px 0">'
                f'<div style="color:#f85149;font-weight:700;font-size:1rem;margin-bottom:8px">'
                f'ค่า Metrics ทั้งหมด = 1.0 — สงสัยว่ามี Data Leakage</div>'
                f'<div style="color:#c9d1d9;font-size:0.9rem;line-height:1.7">'
                f'Feature ที่มี importance สูงสุดคือตัวการที่น่าสงสัยที่สุด<br>'
                f'ย้อนกลับไป <b>Transformation → Data Leakage Check</b> แล้ว drop feature นั้นออก'
                f'</div>'
                f'{fi_section}'
                f'</div>',
                unsafe_allow_html=True,
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
        pred_csv = build_predictions_df(result["y_test"], result["y_pred"], task_type).to_csv(index=False).encode("utf-8-sig")
        st.markdown('<div style="margin-top:24px"></div>', unsafe_allow_html=True)
        st.download_button("Predictions CSV", pred_csv, file_name=f"predictions_{best_label}.csv", mime="text/csv")

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
        if fi_data.get("fi_df") is not None:
            fi_csv = fi_data["fi_df"].to_csv(index=False).encode("utf-8-sig")
            st.markdown('<div style="margin-top:24px"></div>', unsafe_allow_html=True)
            st.download_button("Feature Importance CSV", fi_csv, file_name=f"feature_importance_{best_label}.csv", mime="text/csv")

    with tab_viz:
        render_viz(df)

    render_nav(result)