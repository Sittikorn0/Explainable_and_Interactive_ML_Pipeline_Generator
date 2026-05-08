"""ml_process/views.py — Streamlit / Plotly rendering (no business logic)"""
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd

from ml_process.logic.config import MODEL_WHY


# ── Target Info ───────────────────────────────────────────────────────────────

def render_target_info(target_col: str, task: str, n_unique: int):
    st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
padding:10px 16px;margin-bottom:16px;display:flex;gap:16px;align-items:center">
  <div>
    <div style="font-size:0.75rem;color:#8b949e;text-transform:uppercase;
    letter-spacing:1px;margin-bottom:2px">Target Column</div>
    <div style="font-family:monospace;font-weight:700;font-size:0.95rem;
    color:#58a6ff">{target_col}</div>
  </div>
  <div style="width:1px;height:32px;background:#30363d"></div>
  <div>
    <div style="font-size:0.75rem;color:#8b949e;text-transform:uppercase;
    letter-spacing:1px;margin-bottom:2px">Task</div>
    <div style="font-weight:700;font-size:0.88rem;color:#3fb950">
      {task.upper()}
    </div>
  </div>
  <div style="width:1px;height:32px;background:#30363d"></div>
  <div>
    <div style="font-size:0.75rem;color:#8b949e;text-transform:uppercase;
    letter-spacing:1px;margin-bottom:2px">Unique Values</div>
    <div style="font-weight:700;font-size:0.88rem;color:#e6edf3">
      {n_unique}
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ── Competition Description ───────────────────────────────────────────────────

def render_competition_desc():
    st.markdown("""
<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
padding:14px 18px;margin-bottom:16px;font-size:0.83rem;color:#8b949e;line-height:1.8">
  1. Train <b style="color:#e6edf3">ทุก model</b> บน Train set (80%) พร้อม Auto Hyperparameter Tuning<br>
  2. เปรียบเทียบ Cross-Validation Score<br>
  3. เลือก <b style="color:#58a6ff">Best Model</b> อัตโนมัติ<br>
  4. Evaluate บน Test set (20%)<br>
</div>""", unsafe_allow_html=True)


# ── Model Cards ───────────────────────────────────────────────────────────────

def render_model_cards(avail: dict, model_desc: dict):
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
                f'{model_desc.get(key, "")}</div></div></div>')
        with (col_a if i % 2 == 0 else col_b):
            st.markdown(card, unsafe_allow_html=True)


# ── Best Model Explanation ────────────────────────────────────────────────────

def render_best_model_card(result: dict, best_label: str):
    ranked = sorted([(k, v) for k, v in result["competition"].items() if v["cv_score"]],
                    key=lambda x: x[1]["cv_score"], reverse=True)
    if not ranked:
        st.warning("ไม่มีโมเดลที่ผ่าน cross-validation — ไม่สามารถแสดง Best Model ได้")
        return
    best_cv, best_std = ranked[0][1]["cv_score"], ranked[0][1]["cv_std"]
    avg_std = sum(v["cv_std"] for _, v in ranked) / len(ranked)

    r2 = (f"ห่างจาก {ranked[1][1]['label']} {round(best_cv - ranked[1][1]['cv_score'], 4)} "
          f"({round((best_cv - ranked[1][1]['cv_score']) / (ranked[1][1]['cv_score'] + 1e-9) * 100, 1)}%)"
          if len(ranked) >= 2 else "")
    r3 = (f"±Std={best_std:.4f} {'≤' if best_std <= avg_std else '>'} "
          f"avg({avg_std:.4f}) — {'stable' if best_std <= avg_std else 'score ยังสูงสุด'}")
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
    Cross-Validation = train/validate 3 รอบ สะท้อนประสิทธิภาพกับข้อมูลใหม่ได้ดีกว่า training score
  </div>
</div>""", unsafe_allow_html=True)


# ── Metrics Explanation ───────────────────────────────────────────────────────

def render_metrics_explain(metrics: dict, task_type: str):
    if task_type == "classification":
        acc = metrics.get("Accuracy", 0)
        f1  = metrics.get("F1(Mac)", 0)
        lines = (f"• <b style='color:#58a6ff'>Accuracy {acc}</b> — ทำนายถูก {acc*100:.1f}% ของ test set<br>"
                 f"• <b style='color:#58a6ff'>Precision (Macro)</b> — ค่าเฉลี่ย Precision ของทุก class เท่ากัน<br>"
                 f"• <b style='color:#58a6ff'>Recall (Macro)</b> — ค่าเฉลี่ย Recall ของทุก class เท่ากัน<br>"
                 f"• <b style='color:#58a6ff'>F1 (Macro) {f1}</b> — ค่าเฉลี่ย F1 ของทุก class เท่ากัน")
    else:
        r2, rmse = metrics.get("R² Score", 0), metrics.get("RMSE", 0)
        lines = (f"• <b style='color:#58a6ff'>R² {r2}</b> — อธิบาย variance ได้ {max(0, r2)*100:.1f}% (1=perfect)<br>"
                 f"• <b style='color:#58a6ff'>RMSE {rmse}</b> — error เฉลี่ยในหน่วยเดียวกับ target<br>"
                 f"• <b style='color:#58a6ff'>MSE</b> — เหมือน RMSE แต่ยกกำลัง 2")
    st.markdown(f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;'
                f'padding:12px 16px;margin:8px 0;font-size:0.81rem;color:#c9d1d9;line-height:1.8">'
                f'{lines}</div>', unsafe_allow_html=True)


# ── Confusion Matrix Explanation ──────────────────────────────────────────────

def render_cm_explain(y_test, y_pred):
    from sklearn.metrics import confusion_matrix as _cm
    y_test = np.array(y_test).flatten()
    y_pred = np.array(y_pred).flatten()
    labels  = sorted(list(set(y_test) | set(y_pred)), key=str)
    cm_arr  = _cm(y_test, y_pred, labels=labels)
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


# ── Scatter Explanation ───────────────────────────────────────────────────────

def render_scatter_explain():
    st.markdown("""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
padding:12px 16px;margin:8px 0;font-size:0.81rem;color:#c9d1d9;line-height:1.8">
  <b style="color:#e6edf3">วิธีอ่าน Actual vs Predicted</b><br>
  • <b style="color:#58a6ff">แกน X</b> = ค่าจริง, <b style="color:#58a6ff">แกน Y</b> = ค่าที่ทำนาย<br>
  • <b style="color:#3fb950">เส้นประแดง</b> = Perfect line — จุดที่ดีควรอยู่บนหรือใกล้เส้นนี้<br>
  • จุดที่กระจายสม่ำเสมอรอบเส้น = model ไม่มี systematic bias
</div>""", unsafe_allow_html=True)


# ── Feature Importance Chart ──────────────────────────────────────────────────

def render_fi(fi_df: pd.DataFrame | None, best_label: str, fi_error: str | None):
    st.caption("features ไหนมีผลต่อการตัดสินใจของ model มากที่สุด")

    if fi_df is not None:
        top_n  = min(20, len(fi_df))
        fig_fi = px.bar(fi_df.head(top_n), x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale="Blues",
                        text=fi_df.head(top_n)["Importance"].round(4))
        fig_fi.update_layout(template="plotly_dark", height=max(350, top_n * 28),
                             yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
                             margin=dict(t=20, b=20))
        fig_fi.update_traces(textposition="outside")
        st.plotly_chart(fig_fi, width="stretch")

        total  = fi_df["Importance"].sum()
        medals = ["#1", "#2", "#3"]
        html = ('<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 16px;margin:8px 0">'
                '<div style="font-size:0.81rem;color:#c9d1d9;line-height:1.8">'
                '• ยิ่งแท่งยาว = feature มีผลต่อ model มากกว่า<br>'
                '• feature ที่ importance ต่ำมากอาจพิจารณาตัดออกเพื่อลด complexity</div>'
                '<div style="margin-top:8px">')
        for i, row in fi_df.head(3).iterrows():
            pct = row["Importance"] / (total + 1e-9) * 100
            html += (f'<div style="display:flex;align-items:center;gap:8px;padding:4px 8px;'
                     f'background:#0d1117;border-radius:4px;margin:2px 0">'
                     f'<span>{medals[i]}</span>'
                     f'<span style="font-family:monospace;color:#e6edf3;flex:1">{row["Feature"]}</span>'
                     f'<span style="color:#58a6ff">{row["Importance"]:.4f}</span>'
                     f'<span style="color:#8b949e;font-size:0.76rem">({pct:.1f}%)</span></div>')
        html += '</div></div>'
        st.markdown(html, unsafe_allow_html=True)

    elif fi_error:
        st.warning(f"คำนวณไม่ได้: {fi_error}")
    else:
        st.info(f"{best_label} ไม่รองรับ Feature Importance โดยตรง")


# ── Data Visualization Tab ────────────────────────────────────────────────────

def render_viz(df: pd.DataFrame):
    st.caption(f"ข้อมูลปัจจุบัน {df.shape[0]:,} rows × {df.shape[1]} columns (หลัง transformation)")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    all_cols = num_cols + cat_cols
    if not all_cols:
        st.info("ไม่มีคอลัมน์สำหรับ visualize")
        return

    v1, v2 = st.columns(2)
    with v1:
        sel_col = st.selectbox("เลือกคอลัมน์", all_cols, key="viz_col_ml")
    with v2:
        color_col = st.selectbox("Color by", [None] + cat_cols, key="viz_color_ml") if cat_cols else None

    if sel_col in num_cols:
        fig = px.histogram(df, x=sel_col, color=color_col, marginal="box", nbins=30,
                           color_discrete_sequence=px.colors.qualitative.Set2)
    else:
        counts = df[sel_col].value_counts().reset_index()
        counts.columns = [sel_col, "count"]
        fig = px.bar(counts.head(20), x=sel_col, y="count", color_discrete_sequence=["#58a6ff"])
    fig.update_layout(template="plotly_dark", height=360, showlegend=bool(color_col),
                      margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")

    if len(num_cols) >= 2:
        st.divider()
        fig2 = px.imshow(df[num_cols].corr(), text_auto=".2f",
                         color_continuous_scale="RdBu_r", range_color=[-1, 1], aspect="auto")
        fig2.update_layout(template="plotly_dark", height=380, margin=dict(t=20, b=20))
        st.plotly_chart(fig2, width="stretch")


# ── Navigation ────────────────────────────────────────────────────────────────

def render_nav(result):
    st.divider()
    col1, _, col2 = st.columns([1.2, 7.6, 1.2])
    with col1:
        if st.button("Back", type="secondary", width="stretch"):
            from app import navigate
            for _k in ["ml_result", "ml_metrics", "_fi_data", "ml_task_type",
                       "_ml_scaling_used", "_ml_leakage_warnings"]:
                st.session_state.pop(_k, None)
            navigate("transformation")
    with col2:
        if st.button("Next Step", type="primary", width="stretch", disabled=(result is None)):
            from app import navigate
            navigate("explainable")
        if result is None:
            st.caption("กด Run All Models ก่อนไปขั้นตอนถัดไป",
                       width="content", text_alignment="center")