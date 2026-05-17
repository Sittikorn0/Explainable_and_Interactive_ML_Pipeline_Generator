"""ml_process/views.py — Streamlit / Plotly rendering (no business logic)"""
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd

from ml_process.logic.config import MODEL_WHY


# ── Target Info ───────────────────────────────────────────────────────────────

def render_target_info(target_col: str, task: str, n_unique: int):
    st.markdown(f"""
<div style="background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px;
padding: 16px 20px; margin-bottom: 24px; display: flex; gap: 24px; align-items: center;">
  <div>
    <div style="font-size: 0.85rem; color: #94A3B8; font-family: monospace; text-transform: uppercase; margin-bottom: 4px;">Target Column</div>
    <div style="font-family: monospace; font-weight: bold; font-size: 1.1rem; color: #7AA2F7;">{target_col}</div>
  </div>
  <div style="width: 1px; height: 32px; background: rgba(255,255,255,0.1);"></div>
  <div>
    <div style="font-size: 0.85rem; color: #94A3B8; font-family: monospace; text-transform: uppercase; margin-bottom: 4px;">Task</div>
    <div style="font-weight: bold; font-size: 1.1rem; color: #10B981;">
      {task.upper()}
    </div>
  </div>
  <div style="width: 1px; height: 32px; background: rgba(255,255,255,0.1);"></div>
  <div>
    <div style="font-size: 0.85rem; color: #94A3B8; font-family: monospace; text-transform: uppercase; margin-bottom: 4px;">Unique Values</div>
    <div style="font-weight: bold; font-size: 1.1rem; color: #E2E8F0;">
      {n_unique}
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ── Competition Description ───────────────────────────────────────────────────

def render_competition_desc():
    st.markdown("""
<div style="background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px;
padding: 16px 20px; margin-bottom: 24px; font-size: 0.95rem; color: #94A3B8; line-height: 1.8;">
  1. Train <b style="color: #E2E8F0;">ทุก model</b> บน Train set (80%) พร้อม Auto Hyperparameter Tuning<br>
  2. เปรียบเทียบ Cross-Validation Score<br>
  3. เลือก <b style="color: #7AA2F7;">Best Model</b> อัตโนมัติ<br>
  4. Evaluate บน Test set (20%)<br>
</div>""", unsafe_allow_html=True)


# ── Model Cards ───────────────────────────────────────────────────────────────

def render_model_cards(avail: dict, model_desc: dict):
    st.markdown(f'<div style="font-weight:600; font-size:0.95rem; color:#94A3B8; margin-bottom:12px">'
                f'Models ที่จะแข่งขัน ({len(avail)} ตัว)</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    for i, (key, label) in enumerate(avail.items()):
        card = (f'<div style="background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px;'
                f'padding: 16px; margin: 6px 0; display: flex; align-items: flex-start; gap: 16px;'
                f'min-height: 110px;">'
                f'<div style="width: 8px; height: 8px; border-radius: 50%; background: #10B981;'
                f'flex-shrink: 0; margin-top: 8px; box-shadow: 0 0 8px rgba(16, 185, 129, 0.4);"></div>'
                f'<div><div style="font-family: monospace; font-weight: bold; font-size: 1.05rem;'
                f'color: #E2E8F0;">{label}</div>'
                f'<div style="font-size: 0.9rem; color: #94A3B8; margin-top: 6px; line-height: 1.6;">'
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

    r2_desc = (f"นำหน้า {ranked[1][1]['label']} อยู่ {round(best_cv - ranked[1][1]['cv_score'], 4)} "
               f"({round((best_cv - ranked[1][1]['cv_score']) / (ranked[1][1]['cv_score'] + 1e-9) * 100, 1)}%)"
               if len(ranked) >= 2 else "เป็นโมเดลที่มีประสิทธิภาพสูงสุดในทุกการทดสอบ")
    r3_desc = ("มีความเสถียรสูงกว่าค่าเฉลี่ย" if best_std <= avg_std else "แม้มีความผันผวนแต่คะแนนยังคงเป็นอันดับหนึ่ง")
    r4_desc = MODEL_WHY.get(best_label, "สถาปัตยกรรมเหมาะสมกับรูปแบบของข้อมูลชุดนี้ที่สุด")

    st.markdown(f"""
<div style="background-color: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px;
padding: 20px; margin-top: 16px; margin-bottom: 24px;">
  <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
    <span style="background: #10B981; color: #0d1117; font-weight: bold; font-family: monospace; font-size: 0.9rem;
    padding: 4px 12px; border-radius: 4px;">BEST MODEL</span>
    <span style="font-family: monospace; font-weight: bold; color: #E2E8F0; font-size: 1.25rem;">{best_label}</span>
  </div>
  <div style="font-size: 1rem; color: #E2E8F0; line-height: 2.0;">
    <b style="color: #7AA2F7;">1. Cross-Val Score สูงสุด</b> {best_cv:.4f} <span style="color: #94A3B8;">({r2_desc})</span><br>
    <b style="color: #10B981;">2. Consistency สูงสุด</b> ±Std = {best_std:.4f} <span style="color: #94A3B8;">({r3_desc})</span><br>
    <b style="color: #F59E0B;">3. Model Insight</b> <span style="color: #94A3B8;">{r4_desc}</span>
  </div>
  <div style="margin-top: 16px; padding-top: 12px; border-top: 1px solid rgba(16, 185, 129, 0.1);
  font-size: 0.85rem; color: #64748b; font-family: monospace;">
    Cross-Validation คือการจำลองการทดสอบหลายรอบเพื่อให้มั่นใจว่าโมเดลจะทำงานได้ดีกับข้อมูลใหม่ในอนาคต
  </div>
</div>""", unsafe_allow_html=True)


# ── Metrics Explanation ───────────────────────────────────────────────────────

def render_metrics_explain(metrics: dict, task_type: str):
    if task_type == "classification":
        acc = metrics.get("Accuracy", 0)
        f1  = metrics.get("F1(Mac)", 0)
        items = [
            ("#3fb950", "Accuracy", f"ทำนายถูก {acc*100:.1f}% ของข้อมูลทั้งหมดใน Test set"),
            ("#58a6ff", "Precision", "ความแม่นยำของการทำนายในแต่ละ Class (ทายว่าเป็น A แล้วเป็น A จริงไหม)"),
            ("#d29922", "Recall", "ความสามารถในการหาตัวอย่างใน Class นั้นๆ (ในบรรดาที่เป็น A จริงๆ เราหาเจอเท่าไหร่)"),
            ("#bc8cff", "F1 Score", f"ค่าเฉลี่ยแบบ Harmonic ระหว่าง Precision และ Recall (ค่าปัจจุบัน: {f1:.4f})")
        ]
    else:
        r2, rmse = metrics.get("R² Score", 0), metrics.get("RMSE", 0)
        items = [
            ("#3fb950", "R² Score", f"ความสามารถในการอธิบายข้อมูล (0-1) ยิ่งสูงยิ่งดี โมเดลนี้อธิบายได้ {max(0, r2)*100:.1f}%"),
            ("#f85149", "RMSE", f"ค่าความผิดพลาดเฉลี่ย (หน่วยเดียวกับ Target) ยิ่งต่ำยิ่งดี ปัจจุบันผิดพลาดเฉลี่ย ±{rmse:.4f}"),
            ("#f85149", "MSE", "ค่าความผิดพลาดกำลังสอง (ให้น้ำหนักกับ Error ตัวที่ใหญ่เป็นพิเศษ)")
        ]

    rows_html = ""
    for color, label, desc in items:
        rows_html += f"""<div style="display:flex; align-items:flex-start; gap:16px; margin-bottom:14px">
<div style="background:{color}22; color:{color}; border:1px solid {color}44; font-size:0.8rem; font-weight:700; padding:4px 0; border-radius:6px; text-transform:uppercase; margin-top:2px; width:110px; text-align:center; flex-shrink:0">{label}</div>
<div style="font-size:1.0rem; color:#8b949e; line-height:1.6; padding-top:2px">{desc}</div>
</div>"""

    st.markdown(f"""<div style="background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 24px 28px; margin: 24px 0">
<div style="font-size: 0.85rem; color: #94A3B8; font-weight: bold; font-family: monospace; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 20px;">Metric Glossary — คำอธิบายตัวชี้วัด</div>
<div style="display:flex; flex-direction:column">
{rows_html}
</div>
</div>""", unsafe_allow_html=True)


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
    accuracy_pct = (correct / total) * 100 if total > 0 else 0
    incorrect_pct = (incorrect / total) * 100 if total > 0 else 0
    
    st.markdown(f"""
    <div style="background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; 
    padding: 16px 24px; margin-top: 8px; margin-bottom: 24px; display: flex; align-items: center; justify-content: center; gap: 32px;">
        <div style="color: #E2E8F0; font-size: 0.95rem; display: flex; align-items: center; gap: 8px;">
            <span style="color: #10B981; font-size: 1.1rem; line-height: 1;">●</span> 
            <span><span style="font-weight: bold;">ทำนายถูก:</span> {correct:,} รายการ <span style="color: #94A3B8; margin-left: 4px;">({accuracy_pct:.1f}%)</span></span>
        </div>
        <div style="width: 1px; height: 20px; background-color: rgba(255,255,255,0.1);"></div>
        <div style="color: #E2E8F0; font-size: 0.95rem; display: flex; align-items: center; gap: 8px;">
            <span style="color: #EF4444; font-size: 1.1rem; line-height: 1;">●</span> 
            <span><span style="font-weight: bold;">ทำนายผิด:</span> {incorrect:,} รายการ <span style="color: #94A3B8; margin-left: 4px;">({incorrect_pct:.1f}%)</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Scatter Explanation ───────────────────────────────────────────────────────

def render_scatter_explain():
    st.markdown("""
<div style="background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px;
padding: 16px 20px; margin: 12px 0; font-size: 0.9rem; color: #94A3B8; line-height: 1.8">
  <b style="color: #E2E8F0;">วิธีอ่าน Actual vs Predicted</b><br>
  • <b style="color: #7AA2F7;">แกน X</b> = ค่าจริง, <b style="color: #7AA2F7;">แกน Y</b> = ค่าที่ทำนาย<br>
  • <b style="color: #10B981;">เส้นประแดง</b> = Perfect line — จุดที่ดีควรอยู่บนหรือใกล้เส้นนี้<br>
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
        html = ('<div style="background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 16px 20px; margin: 12px 0">'
                '<div style="font-size: 0.9rem; color: #94A3B8; line-height: 1.8">'
                '• ยิ่งแท่งยาว = feature มีผลต่อ model มากกว่า<br>'
                '• feature ที่ importance ต่ำมากอาจพิจารณาตัดออกเพื่อลด complexity</div>'
                '<div style="margin-top: 12px">')
        for i, row in fi_df.head(3).iterrows():
            pct = row["Importance"] / (total + 1e-9) * 100
            html += (f'<div style="display:flex; align-items:center; gap:12px; padding: 8px 12px;'
                     f'background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.05); border-radius: 6px; margin: 6px 0">'
                     f'<span style="font-weight: bold; color: #BB9AF7;">{medals[i]}</span>'
                     f'<span style="font-family:monospace; color: #E2E8F0; flex: 1; font-size: 1rem;">{row["Feature"]}</span>'
                     f'<span style="color: #7AA2F7; font-weight: bold;">{row["Importance"]:.4f}</span>'
                     f'<span style="color: #64748b; font-size: 0.85rem">({pct:.1f}%)</span></div>')
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


def render_metric_cards(metrics_dict: dict):
    """แสดง metrics ในรูปแบบพรีเมียมการ์ด"""
    # กรองเฉพาะค่าที่เป็นตัวเลข
    display_metrics = {k: v for k, v in metrics_dict.items() if isinstance(v, (int, float))}
    cols = st.columns(len(display_metrics))
    
    style_map = {
        "Accuracy":       {"color": "#3fb950"},
        "Precision(Mac)": {"color": "#58a6ff"},
        "Recall(Mac)":    {"color": "#d29922"},
        "F1(Mac)":        {"color": "#bc8cff"},
        "MSE":            {"color": "#f85149"},
        "RMSE":           {"color": "#f85149"},
        "R² Score":       {"color": "#3fb950"},
    }

    for i, (name, val) in enumerate(display_metrics.items()):
        style = style_map.get(name, {"color": "#8b949e"})
        formatted_val = f"{val:.4f}" if isinstance(val, (float, np.float64, np.float32)) else str(val)
        with cols[i]:
            st.markdown(f"""
            <div style="border-bottom: 2px solid {style['color']}33; padding: 12px 0; margin-bottom: 15px">
                <div style="color:#8b949e; font-size:0.85rem; font-weight:600; text-transform:uppercase; 
                display:flex; align-items:center; gap:8px; letter-spacing:0.07em">
                    <span style="color:{style['color']}; font-size:14px">●</span> {name}
                </div>
                <div style="color:#e6edf3; font-size:2.2rem; font-weight:800; margin-top:4px; letter-spacing:-0.02em">
                    {formatted_val}
                </div>
            </div>
            """, unsafe_allow_html=True)


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