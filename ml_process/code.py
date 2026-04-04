"""ml_process/code.py — Entry point UI"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from ml_process.preprocess  import preprocess, detect_task
from ml_process.runner      import run_competition, get_available_models, get_model_map
from ml_process.evaluation  import get_metrics, show_metrics, show_leaderboard, show_confusion_matrix, show_pred_vs_actual
from ml_process.config      import MODEL_DESC, MODEL_WHY


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

    df         = st.session_state["main_df"]
    file_name  = st.session_state.get("last_uploaded_file", "Unknown File")
    st.info(f"**Current Dataset:** {file_name}  |  {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── 1. Target Column ─────────────────────────────────────
    st.subheader("1. กำหนด Target Column")
    preset = st.session_state.get("ml_target_col_preset")
    cols   = df.columns.tolist()
    idx    = cols.index(preset) if preset and preset in cols else len(cols) - 1
    target_col = st.selectbox("Target", cols, index=idx, key="ml_target_col",
                               label_visibility="collapsed")
    task_preview = detect_task(df, target_col)
    st.caption(f"Task: **{task_preview.upper()}**  |  Unique values: **{df[target_col].nunique()}**")

    # ── 2. Data Splitting ─────────────────────────────────────
    st.subheader("2. Data Splitting (80 / 20)")
    n = len(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows",      f"{n:,}")
    c2.metric("Train Set (80%)", f"{int(n*0.8):,}")
    c3.metric("Test Set (20%)",  f"{n - int(n*0.8):,}")

    # ── 3. Model Competition ──────────────────────────────────
    st.subheader("3. Model Competition + Auto Hyperparameter Tuning")

    st.markdown("""
<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
padding:14px 18px;margin-bottom:16px;font-size:0.83rem;color:#8b949e;line-height:1.8">
  1. Train <b style="color:#e6edf3">ทุก model</b> บน Train set (80%) พร้อม Auto Hyperparameter Tuning<br>
  2. เปรียบเทียบ Cross-Validation Score · 3. เลือก <b style="color:#58a6ff">Best Model</b> อัตโนมัติ
  · 4. Evaluate บน Test set (20%)
</div>""", unsafe_allow_html=True)

    avail = get_available_models(task_preview)
    st.markdown(f'<div style="font-weight:600;font-size:0.85rem;color:#8b949e;margin-bottom:10px">'
                f'Models ที่จะแข่งขัน ({len(avail)} ตัว)</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    for i, (key, label) in enumerate(avail.items()):
        card = (f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;'
                f'padding:10px 14px;margin:4px 0;display:flex;align-items:flex-start;gap:10px">'
                f'<div style="width:8px;height:8px;border-radius:50%;background:#3fb950;'
                f'flex-shrink:0;margin-top:5px"></div>'
                f'<div><div style="font-family:monospace;font-weight:600;font-size:0.83rem;'
                f'color:#e6edf3">{label}</div>'
                f'<div style="font-size:0.74rem;color:#8b949e;margin-top:1px">'
                f'{MODEL_DESC.get(key,"")}</div></div></div>')
        with (col_a if i % 2 == 0 else col_b):
            st.markdown(card, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("▶  Run All Models", type="primary"):
        st.session_state.pop("ml_result",  None)
        st.session_state.pop("ml_metrics", None)
        st.session_state.pop("_fi_data",   None)

        with st.spinner("กำลัง preprocess..."):
            try:
                trans_summary  = st.session_state.get("trans_summary", {})
                already_scaled = trans_summary.get("scaling_method", "no_scaling") != "no_scaling"
                X_train, X_test, y_train, y_test, task_type = preprocess(
                    df, target_col, already_scaled=already_scaled)
                st.session_state["ml_task_type"] = task_type
                if already_scaled:
                    st.caption(f"ℹ️ ข้าม Scaling เพราะ Transformation ใช้ **{trans_summary['scaling_method']}** ไปแล้ว")
            except Exception as e:
                st.error(f"Preprocess ล้มเหลว: {e}")
                return

        bar = st.progress(0, text="กำลัง train...")
        def _prog(label, i, total):
            bar.progress((i+1)/total, text=f"Training {label}... ({i+1}/{total})")

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
            import traceback; st.code(traceback.format_exc())
            return

    # ── Results ───────────────────────────────────────────────
    result  = st.session_state.get("ml_result")
    metrics = st.session_state.get("ml_metrics")
    if not (result and metrics):
        _nav(result)
        return

    task_type   = result["task_type"]
    best_label  = result["best_label"]
    best_key    = result["best_key"]
    best_params = result["best_params"]

    st.success(f"✅  Best Model: **{best_label}**")

    tab_leader, tab_eval, tab_fi, tab_viz, tab_trace = st.tabs([
        "🏆 Leaderboard", "📊 Evaluation", "🔑 Feature Importance",
        "🔬 Data Visualization", "📋 Trace Log",
    ])

    # ── Leaderboard ───────────────────────────────────────────
    with tab_leader:
        show_leaderboard(result["competition"])
        if best_params:
            with st.expander(f"Best Hyperparameters ของ {best_label}"):
                for k, v in best_params.items():
                    st.write(f"- **{k}**: `{v}`")
        _explain_best(result, best_label, best_key, metrics)

    # ── Evaluation ────────────────────────────────────────────
    with tab_eval:
        st.caption(f"ผลลัพธ์จาก **{best_label}** บน Test set ที่ยังไม่เคยเห็น")
        show_metrics(metrics)
        _explain_metrics(metrics, task_type)
        st.divider()
        if task_type == "classification":
            show_confusion_matrix(result["y_test"], result["y_pred"])
            _explain_cm(result["y_test"], result["y_pred"])
        else:
            show_pred_vs_actual(result["y_test"].values, result["y_pred"])
            _explain_scatter()

    # ── Feature Importance ────────────────────────────────────
    with tab_fi:
        _show_fi(df, target_col, best_key, best_label, best_params,
                 st.session_state.get("trans_summary", {}))

    # ── Data Visualization ────────────────────────────────────
    with tab_viz:
        _show_viz(df, target_col)

    # ── Trace Log ─────────────────────────────────────────────
    with tab_trace:
        _show_trace(df, target_col, result, metrics, best_label)

    _nav(result)


# ═══════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════

def _explain_best(result, best_label, best_key, metrics):
    ranked = sorted([(k,v) for k,v in result["competition"].items() if v["cv_score"]],
                    key=lambda x: x[1]["cv_score"], reverse=True)
    best_cv, best_std = ranked[0][1]["cv_score"], ranked[0][1]["cv_std"]
    avg_std = sum(v["cv_std"] for _,v in ranked) / len(ranked)

    r2 = f"ห่างจาก {ranked[1][1]['label']} {round(best_cv-ranked[1][1]['cv_score'],4)} ({round((best_cv-ranked[1][1]['cv_score'])/(ranked[1][1]['cv_score']+1e-9)*100,1)}%)" if len(ranked)>=2 else ""
    r3 = f"±Std={best_std:.4f} {'≤' if best_std<=avg_std else '>'} avg({avg_std:.4f}) — {'stable' if best_std<=avg_std else 'score ยังสูงสุด'}"
    r4 = MODEL_WHY.get(best_label, f"{best_label} ให้ผลดีที่สุดกับชุดข้อมูลนี้")

    st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
padding:14px 18px;margin-top:12px">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
    <span style="background:#3fb950;color:#0d1117;font-weight:700;font-size:0.75rem;
    padding:2px 10px;border-radius:10px">BEST MODEL</span>
    <span style="font-family:monospace;font-weight:700;color:#e6edf3">{best_label}</span>
  </div>
  <div style="font-size:0.82rem;color:#c9d1d9;line-height:1.9">
    <b style="color:#58a6ff">1. CV Score สูงสุด</b> ({best_cv:.4f}) {r2}<br>
    <b style="color:#3fb950">2. Stability</b> — {r3}<br>
    <b style="color:#d29922">3. Model</b> — {r4}
  </div>
  <div style="margin-top:10px;padding-top:10px;border-top:1px solid #30363d;
  font-size:0.76rem;color:#8b949e">
    💡 Cross-Validation = train/validate 3 รอบ สะท้อนประสิทธิภาพกับข้อมูลใหม่ได้ดีกว่า training score
  </div>
</div>""", unsafe_allow_html=True)


def _explain_metrics(metrics, task_type):
    if task_type == "classification":
        acc, f1 = metrics.get("Accuracy",0), metrics.get("F1-Score",0)
        lines = (f"• <b style='color:#58a6ff'>Accuracy {acc}</b> — ทำนายถูก {acc*100:.1f}% ของ test set<br>"
                 f"• <b style='color:#58a6ff'>Precision</b> — จากที่ทำนายว่าเป็น class นั้น ถูกกี่ %<br>"
                 f"• <b style='color:#58a6ff'>Recall</b> — จาก class นั้นทั้งหมด จับได้กี่ %<br>"
                 f"• <b style='color:#58a6ff'>F1-Score {f1}</b> — ค่าเฉลี่ย Precision+Recall")
    else:
        r2, rmse = metrics.get("R² Score",0), metrics.get("RMSE",0)
        lines = (f"• <b style='color:#58a6ff'>R² {r2}</b> — อธิบาย variance ได้ {max(0,r2)*100:.1f}% (1=perfect)<br>"
                 f"• <b style='color:#58a6ff'>RMSE {rmse}</b> — error เฉลี่ยในหน่วยเดียวกับ target<br>"
                 f"• <b style='color:#58a6ff'>MSE</b> — เหมือน RMSE แต่ยกกำลัง 2")
    st.markdown(f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;'
                f'padding:12px 16px;margin:8px 0;font-size:0.81rem;color:#c9d1d9;line-height:1.8">'
                f'{lines}</div>', unsafe_allow_html=True)


def _explain_cm(y_test, y_pred):
    from sklearn.metrics import confusion_matrix as _cm
    labels = sorted(list(set(y_test)|set(y_pred)), key=str)
    cm_arr = _cm(y_test, y_pred, labels=labels)
    correct   = int(np.trace(cm_arr))
    incorrect = int(cm_arr.sum() - correct)
    total     = correct + incorrect
    st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
padding:12px 16px;margin:8px 0;font-size:0.81rem;color:#c9d1d9;line-height:1.8">
  <b style="color:#e6edf3">วิธีอ่าน Confusion Matrix</b><br>
  • <b style="color:#58a6ff">Diagonal</b> = ทำนายถูก ({correct}/{total} = {correct/total*100:.1f}%)<br>
  • <b style="color:#d29922">นอก Diagonal</b> = ทำนายผิด ({incorrect} samples)<br>
  • แถว = Actual (ค่าจริง), คอลัมน์ = Predicted (ที่ model ทำนาย)
</div>""", unsafe_allow_html=True)


def _explain_scatter():
    st.markdown("""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
padding:12px 16px;margin:8px 0;font-size:0.81rem;color:#c9d1d9;line-height:1.8">
  <b style="color:#e6edf3">วิธีอ่าน Actual vs Predicted</b><br>
  • <b style="color:#58a6ff">แกน X</b> = ค่าจริง, <b style="color:#58a6ff">แกน Y</b> = ค่าที่ทำนาย<br>
  • <b style="color:#3fb950">เส้นประแดง</b> = Perfect line — จุดที่ดีควรอยู่บนหรือใกล้เส้นนี้<br>
  • จุดที่กระจายสม่ำเสมอรอบเส้น = model ไม่มี systematic bias
</div>""", unsafe_allow_html=True)


def _show_fi(df, target_col, best_key, best_label, best_params, trans_summary):
    st.caption("features ไหนมีผลต่อการตัดสินใจของ model มากที่สุด")

    fi_data = st.session_state.get("_fi_data")
    if fi_data is None or fi_data.get("model_key") != best_key:
        with st.spinner(f"คำนวณ Feature Importance ของ {best_label}..."):
            try:
                already_scaled = trans_summary.get("scaling_method","no_scaling") != "no_scaling"
                X_tr, _, y_tr, _, _ = preprocess(df, target_col, already_scaled=already_scaled)
                m = get_model_map()[best_key]()
                if best_params:
                    try: m.set_params(**best_params)
                    except Exception: pass
                m.fit(X_tr, y_tr)
                importances = None
                if hasattr(m, "feature_importances_"):
                    importances = m.feature_importances_
                elif hasattr(m, "coef_"):
                    coef = m.coef_
                    importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

                if importances is not None:
                    fi_df = pd.DataFrame({"Feature": X_tr.columns, "Importance": importances})\
                              .sort_values("Importance", ascending=False).reset_index(drop=True)
                    st.session_state["_fi_data"] = {"model_key": best_key, "fi_df": fi_df}
                else:
                    st.session_state["_fi_data"] = {"model_key": best_key, "fi_df": None}
            except Exception as e:
                st.session_state["_fi_data"] = {"model_key": best_key, "fi_df": None, "error": str(e)}
        fi_data = st.session_state["_fi_data"]

    fi_df = fi_data.get("fi_df")
    if fi_df is not None:
        top_n  = min(20, len(fi_df))
        fig_fi = px.bar(fi_df.head(top_n), x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale="Blues",
                        text=fi_df.head(top_n)["Importance"].round(4))
        fig_fi.update_layout(template="plotly_dark", height=max(350, top_n*28),
                             yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
                             margin=dict(t=20, b=20))
        fig_fi.update_traces(textposition="outside")
        st.plotly_chart(fig_fi, use_container_width=True)

        total = fi_df["Importance"].sum()
        medals = ["🥇","🥈","🥉"]
        explain_html = '<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 16px;margin:8px 0">'
        explain_html += '<div style="font-size:0.81rem;color:#c9d1d9;line-height:1.8">• ยิ่งแท่งยาว = feature มีผลต่อ model มากกว่า<br>• feature ที่ importance ต่ำมากอาจพิจารณาตัดออกเพื่อลด complexity</div>'
        explain_html += '<div style="margin-top:8px">'
        for i, row in fi_df.head(3).iterrows():
            pct = row["Importance"]/(total+1e-9)*100
            explain_html += (f'<div style="display:flex;align-items:center;gap:8px;padding:4px 8px;'
                             f'background:#0d1117;border-radius:4px;margin:2px 0">'
                             f'<span>{medals[i]}</span>'
                             f'<span style="font-family:monospace;color:#e6edf3;flex:1">{row["Feature"]}</span>'
                             f'<span style="color:#58a6ff">{row["Importance"]:.4f}</span>'
                             f'<span style="color:#8b949e;font-size:0.76rem">({pct:.1f}%)</span></div>')
        explain_html += '</div></div>'
        st.markdown(explain_html, unsafe_allow_html=True)
    elif fi_data.get("error"):
        st.warning(f"คำนวณไม่ได้: {fi_data['error']}")
    else:
        st.info(f"{best_label} ไม่รองรับ Feature Importance โดยตรง")


def _show_viz(df, target_col):
    st.caption(f"ข้อมูลปัจจุบัน {df.shape[0]:,} rows × {df.shape[1]} columns (หลัง transformation)")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    all_cols = num_cols + cat_cols
    if not all_cols:
        st.info("ไม่มีคอลัมน์สำหรับ visualize")
        return
    v1, v2 = st.columns(2)
    with v1:
        sel_col = st.selectbox("เลือกคอลัมน์", all_cols, key="viz_col_ml")
    with v2:
        color_col = st.selectbox("Color by", [None]+cat_cols, key="viz_color_ml") if cat_cols else None

    if sel_col in num_cols:
        fig = px.histogram(df, x=sel_col, color=color_col, marginal="box", nbins=30,
                           color_discrete_sequence=px.colors.qualitative.Set2)
    else:
        counts = df[sel_col].value_counts().reset_index()
        counts.columns = [sel_col, "count"]
        fig = px.bar(counts.head(20), x=sel_col, y="count", color_discrete_sequence=["#58a6ff"])
    fig.update_layout(template="plotly_dark", height=360, showlegend=bool(color_col),
                      margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    if len(num_cols) >= 2:
        st.divider()
        fig2 = px.imshow(df[num_cols].corr(), text_auto=".2f",
                         color_continuous_scale="RdBu_r", range_color=[-1,1], aspect="auto")
        fig2.update_layout(template="plotly_dark", height=380, margin=dict(t=20,b=20))
        st.plotly_chart(fig2, use_container_width=True)


def _show_trace(df, target_col, result, metrics, best_label):
    steps = []
    fname = st.session_state.get("last_uploaded_file","—")
    steps.append(("📁 Upload",     f"โหลด `{fname}`",
                  f"{df.shape[0]:,} rows × {df.shape[1]} columns", "#3fb950"))
    if st.session_state.get("cleaning_confirmed"):
        steps.append(("🧹 Cleaning", "ผ่านขั้นตอน Cleaning",
                      "ลบ duplicate / จัดการ missing / outlier", "#3fb950"))
    ts = st.session_state.get("trans_summary")
    if ts:
        steps.append(("🔄 Transformation",
                      f"Encoding: {ts.get('encoded_cols',0)} cols | Scaling: {ts.get('scaling_method','—')} | Dropped: {ts.get('dropped_cols',0)}",
                      f"{ts.get('original_cols','?')} → {ts.get('final_cols','?')} columns", "#58a6ff"))
    steps.append(("🤖 ML Process",
                  f"Competition: {len(result['competition'])} models | Task: {result['task_type'].upper()}",
                  f"Best: {best_label} | Train/Test 80/20", "#d29922"))
    metric_str = " | ".join(f"{k}: {v}" for k, v in metrics.items())
    steps.append(("📊 Evaluation", f"Test set — {best_label}", metric_str, "#d29922"))

    for i, (step, action, detail, color) in enumerate(steps):
        connector = f"<div style='width:2px;height:16px;background:#30363d;margin:0 auto'></div>" if i < len(steps)-1 else ""
        st.markdown(f"""
<div style="display:flex;gap:12px;align-items:flex-start">
  <div style="display:flex;flex-direction:column;align-items:center;flex-shrink:0;width:16px">
    <div style="width:14px;height:14px;border-radius:50%;background:{color};margin-top:16px"></div>
    {connector}
  </div>
  <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
  padding:10px 14px;margin:6px 0;flex:1">
    <div style="font-weight:600;font-size:0.85rem;color:{color}">{step}</div>
    <div style="font-size:0.81rem;color:#e6edf3;margin:2px 0">{action}</div>
    <div style="font-family:monospace;font-size:0.74rem;color:#8b949e;
    background:#0d1117;padding:3px 8px;border-radius:4px;margin-top:4px">{detail}</div>
  </div>
</div>""", unsafe_allow_html=True)


def _nav(result):
    st.divider()
    col1, _, col2 = st.columns([0.8, 8, 0.8])
    with col1:
        if st.button("Back", type="secondary", width="stretch"):
            st.query_params["step"] = "transformation"
            st.rerun()
    with col2:
        if st.button("Next Step", type="primary", width="stretch", disabled=(result is None)):
            st.query_params["step"] = "explainable"
            st.rerun()
        if result is None:
            st.caption("กด Run All Models ก่อนไปขั้นตอนถัดไป",
                       width="content", text_alignment="center")